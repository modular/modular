# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Centralized graph capture runner for overlap serving.

Flow:
- Model worker creates the runner and executes pre-ready warmup.
- Warmup captures hot decode buckets, largest-first.
- Serving path replays by batch_token_count key; misses raise.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager

from max.driver import Accelerator, Buffer
from max.engine import Model

from .interfaces import ModelInputs, ModelOutputs

logger = logging.getLogger("max.pipelines")


GraphKey = int
GraphEntry = tuple[tuple[Buffer, ...], ModelOutputs]


def _graph_key(model_inputs: ModelInputs) -> GraphKey:
    # TODO(SERVOPT-1010): Extend GraphKey with decode_max_cache_length
    # (kv_cache_inputs.max_lengths[0, 1]) because decode kernel
    # launch parameters are derived from this value.
    return int(model_inputs.buffers[0].shape[0])


class ServeGraphCaptureRunner:
    """Central owner for serve-time graph capture state."""

    def __init__(
        self,
        *,
        model: Model,
        warmup_model_inputs: Callable[
            [int], AbstractContextManager[ModelInputs]
        ],
        execute_model: Callable[[ModelInputs], ModelOutputs],
        max_batch_size: int,
    ) -> None:
        self._model = model
        self._warmup_model_inputs = warmup_model_inputs
        self._execute_model = execute_model
        if max_batch_size < 1:
            raise ValueError(
                "Device graph capture requires a positive decode capture "
                "batch-size upper bound."
            )
        self._max_batch_size = max_batch_size

        self.graph_entries: dict[GraphKey, GraphEntry] = {}

    def warmup_pre_ready(self) -> None:
        """Captures decode buckets before the worker becomes ready."""
        logger.info(
            "Pre-capturing overlap device graphs for decode batch sizes [1..%d] "
            "with num_steps=1.",
            self._max_batch_size,
        )
        # Conservative/defensive warmup: capture largest-first so peak
        # allocations happen up front and oversized configs fail fast.
        for batch_size in range(self._max_batch_size, 0, -1):
            with self._warmup_model_inputs(batch_size) as model_inputs:
                # Warmup eager twice for stable kernel/runtime initialization.
                self._execute_model(model_inputs)
                self._execute_model(model_inputs)

                input_buffers = model_inputs.buffers
                graph_key = _graph_key(model_inputs)
                self.graph_entries[graph_key] = (
                    input_buffers,
                    ModelOutputs(
                        *self._model.capture(graph_key, *input_buffers)
                    ),
                )
                for device in self._model.input_devices:
                    Accelerator(id=device.id).synchronize()
        logger.info(
            "Overlap device graph pre-capture complete for decode batch sizes "
            "[1..%d] with num_steps=1.",
            self._max_batch_size,
        )

    def replay(
        self,
        *,
        model_inputs: ModelInputs,
        debug_verify_replay: bool = False,
        debug_verify_model_inputs: ModelInputs | None = None,
    ) -> ModelOutputs:
        """Replays a captured graph entry for a replay-safe decode batch.

        ``model_inputs`` are copied into captured replay buffers and used for
        replay. When ``debug_verify_replay`` is enabled, callers may provide
        ``debug_verify_model_inputs`` to verify eager traces against a
        different (but graph-key-equivalent) input shape.
        """
        input_buffers = model_inputs.buffers
        graph_key = _graph_key(model_inputs)
        try:
            captured_inputs, outputs = self.graph_entries[graph_key]
        except KeyError as exc:
            raise RuntimeError(
                "No captured device graph found for batch_token_count: "
                f"{graph_key}."
            ) from exc

        for src_value, dst_value in zip(
            input_buffers, captured_inputs, strict=True
        ):
            dst_value.inplace_copy_from(src_value)

        if debug_verify_replay:
            verify_inputs = debug_verify_model_inputs or model_inputs
            verify_graph_key = _graph_key(verify_inputs)
            if verify_graph_key != graph_key:
                raise ValueError(
                    "debug_verify_model_inputs must map to the same graph key "
                    f"as replay inputs: {verify_graph_key} != {graph_key}."
                )
            self._model.debug_verify_replay(graph_key, *verify_inputs.buffers)

        self._model.replay(graph_key, *captured_inputs)
        return outputs
