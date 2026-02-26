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
- Serving path replays by (batch_token_count, partition_regime) key.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager

from max.driver import Accelerator, Buffer
from max.engine import Model
from max.nn.kv_cache import KVCacheInputsSequence, RaggedKVCacheInputs

from .interfaces import ModelInputs, ModelOutputs

logger = logging.getLogger("max.pipelines")


GraphKey = tuple[int, int]
GraphEntry = tuple[tuple[Buffer, ...], ModelOutputs]

_DECODE_PARTITION_GRANULARITY = 512
_MAX_DECODE_PARTITION_REGIME = 32
_GRAPH_KEY_COMPONENT_MAX = 2**32 - 1


def _max_cache_length_to_decode_partition_regime(max_cache_length: int) -> int:
    # TODO(SERVOPT-1010): Refactor and generalize decode partition derivation.
    # - Reuse one shared helper for serve keying, warmup targeting, and kernel
    #   launch-shape decisions.
    # - Remove duplicated assumptions about granularity/caps across Python and
    #   kernel-side logic.
    # - Add coverage that enforces parity when partitioning rules change.
    # MHA decode partitioning scales in powers of two by max cache length with
    # a fixed 512-token granularity and an upper bound of 32 partitions.
    base_partitions = max(max_cache_length // _DECODE_PARTITION_GRANULARITY, 1)
    return min(
        1 << (base_partitions - 1).bit_length(),
        _MAX_DECODE_PARTITION_REGIME,
    )


def _canonical_decode_max_cache_length_for_partition_regime(
    partition_regime: int,
    *,
    max_cache_length_upper_bound: int,
) -> int:
    """Returns a regime-stable max-cache length for capture/replay.

    Use an upper-bound representative so replay remains correct for all decode
    requests in the same partition regime.
    """
    if partition_regime < 1:
        raise ValueError("partition regime must be positive.")

    if partition_regime >= _MAX_DECODE_PARTITION_REGIME:
        return max_cache_length_upper_bound

    regime_upper_bound = (
        (partition_regime + 1) * _DECODE_PARTITION_GRANULARITY
    ) - 1
    return min(regime_upper_bound, max_cache_length_upper_bound)


def _ragged_kv_inputs_from_model_inputs(
    model_inputs: ModelInputs,
) -> tuple[RaggedKVCacheInputs, ...] | None:
    kv_cache_inputs = model_inputs.kv_cache_inputs
    if kv_cache_inputs is None:
        return None

    if isinstance(kv_cache_inputs, KVCacheInputsSequence):
        kv_inputs_seq = kv_cache_inputs.kv_cache_inputs
    else:
        kv_inputs_seq = [kv_cache_inputs]

    if not kv_inputs_seq:
        raise ValueError("Expected at least one KV cache input for replay.")

    ragged_inputs: list[RaggedKVCacheInputs] = []
    for kv_inputs in kv_inputs_seq:
        if not isinstance(kv_inputs, RaggedKVCacheInputs):
            raise TypeError(
                "Expected RaggedKVCacheInputs for overlap graph capture, got "
                f"{type(kv_inputs).__name__}."
            )
        ragged_inputs.append(kv_inputs)

    return tuple(ragged_inputs)


def _set_decode_max_cache_length_for_model_inputs(
    model_inputs: ModelInputs,
    *,
    max_cache_length: int,
) -> bool:
    """Overwrites KV max cache-length metadata for warmup capture."""
    ragged_inputs = _ragged_kv_inputs_from_model_inputs(model_inputs)
    if ragged_inputs is None:
        return False

    for kv_inputs in ragged_inputs:
        max_lengths_np = kv_inputs.max_lengths.to_numpy()
        if max_lengths_np.ndim != 2 or max_lengths_np.shape[1] != 2:
            raise ValueError(
                "Expected max_lengths to have shape [batch_size, 2], got "
                f"{max_lengths_np.shape}."
            )
        max_lengths_np[:, 1] = max_cache_length
    return True


def _replay_graph_key_from_model_inputs(model_inputs: ModelInputs) -> GraphKey:
    batch_token_count = int(model_inputs.buffers[0].shape[0])
    ragged_inputs = _ragged_kv_inputs_from_model_inputs(model_inputs)
    if ragged_inputs is None:
        decode_max_cache_length: int | None = None
    else:
        max_lengths_np = ragged_inputs[0].max_lengths.to_numpy()
        if (
            max_lengths_np.ndim != 2
            or max_lengths_np.shape[0] < 1
            or max_lengths_np.shape[1] != 2
        ):
            raise ValueError(
                "Expected max_lengths to have shape [batch_size, 2], got "
                f"{max_lengths_np.shape}."
            )
        decode_max_cache_length = int(max_lengths_np[0, 1])

    partition_regime = (
        _max_cache_length_to_decode_partition_regime(decode_max_cache_length)
        if decode_max_cache_length is not None
        else 1
    )
    return (batch_token_count, partition_regime)


def _pack_model_graph_key(replay_graph_key: GraphKey) -> int:
    batch_token_count, partition_regime = replay_graph_key
    if batch_token_count < 0 or partition_regime < 0:
        raise ValueError(
            f"graph key values must be non-negative, got {replay_graph_key}."
        )
    if (
        batch_token_count > _GRAPH_KEY_COMPONENT_MAX
        or partition_regime > _GRAPH_KEY_COMPONENT_MAX
    ):
        raise ValueError(
            f"graph key values exceed uint32 packing range: {replay_graph_key}."
        )
    return (partition_regime << 32) | batch_token_count


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
        decode_max_cache_length_upper_bound: int,
    ) -> None:
        self._model = model
        self._warmup_model_inputs = warmup_model_inputs
        self._execute_model = execute_model
        if max_batch_size < 1:
            raise ValueError(
                "Device graph capture requires a positive decode capture "
                "batch-size upper bound."
            )
        if decode_max_cache_length_upper_bound < 1:
            raise ValueError(
                "Device graph capture requires a positive decode max-cache "
                "length upper bound."
            )
        self._max_batch_size = max_batch_size
        self._decode_max_cache_length_upper_bound = (
            decode_max_cache_length_upper_bound
        )

        self.graph_entries: dict[GraphKey, GraphEntry] = {}

    def warmup_pre_ready(self) -> None:
        """Captures decode buckets before the worker becomes ready."""
        logger.info(
            "Pre-capturing overlap device graphs for decode batch sizes [1..%d] "
            "with num_steps=1.",
            self._max_batch_size,
        )
        max_cache_lengths = [1, self._decode_max_cache_length_upper_bound]
        max_cache_lengths.extend(
            range(
                _DECODE_PARTITION_GRANULARITY,
                self._decode_max_cache_length_upper_bound + 1,
                _DECODE_PARTITION_GRANULARITY,
            )
        )
        warmup_partition_regimes = {
            _max_cache_length_to_decode_partition_regime(max_cache_length)
            for max_cache_length in max_cache_lengths
        }
        warmup_targets = tuple(
            (
                partition_regime,
                _canonical_decode_max_cache_length_for_partition_regime(
                    partition_regime,
                    max_cache_length_upper_bound=(
                        self._decode_max_cache_length_upper_bound
                    ),
                ),
            )
            for partition_regime in sorted(
                warmup_partition_regimes,
                reverse=True,
            )
        )
        logger.info(
            "Device graph warmup partition regimes: %s",
            [target[0] for target in warmup_targets],
        )
        # Conservative/defensive warmup: capture largest-first so peak
        # allocations happen up front and oversized configs fail fast.
        for batch_size in range(self._max_batch_size, 0, -1):
            with self._warmup_model_inputs(batch_size) as model_inputs:
                for _, max_cache_length in warmup_targets:
                    updated = _set_decode_max_cache_length_for_model_inputs(
                        model_inputs,
                        max_cache_length=max_cache_length,
                    )
                    if not updated:
                        logger.warning(
                            "Unable to override decode max cache length for "
                            "warmup batch_size=%d; capturing default key only.",
                            batch_size,
                        )

                    # Warmup eager twice for stable kernel/runtime initialization.
                    self._execute_model(model_inputs)
                    self._execute_model(model_inputs)

                    replay_graph_key = _replay_graph_key_from_model_inputs(
                        model_inputs
                    )
                    input_buffers = model_inputs.buffers
                    packed_model_graph_key = _pack_model_graph_key(
                        replay_graph_key
                    )
                    self.graph_entries[replay_graph_key] = (
                        input_buffers,
                        ModelOutputs(
                            *self._model.capture(
                                packed_model_graph_key,
                                *input_buffers,
                            )
                        ),
                    )
                    for device in self._model.input_devices:
                        Accelerator(id=device.id).synchronize()

                    if not updated:
                        break
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
        replay_graph_key = _replay_graph_key_from_model_inputs(model_inputs)
        packed_model_graph_key = _pack_model_graph_key(replay_graph_key)
        _, partition_regime = replay_graph_key
        _set_decode_max_cache_length_for_model_inputs(
            model_inputs,
            max_cache_length=_canonical_decode_max_cache_length_for_partition_regime(
                partition_regime,
                max_cache_length_upper_bound=self._decode_max_cache_length_upper_bound,
            ),
        )
        try:
            captured_inputs, outputs = self.graph_entries[replay_graph_key]
        except KeyError as exc:
            batch_token_count, _ = replay_graph_key
            available_partition_regimes = sorted(
                key[1]
                for key in self.graph_entries
                if key[0] == batch_token_count
            )
            available_batch_token_counts = sorted(
                {key[0] for key in self.graph_entries}
            )
            raise RuntimeError(
                "No captured device graph found for key: "
                f"{replay_graph_key}. Available partition regimes for "
                f"batch_token_count={batch_token_count}: "
                f"{available_partition_regimes}. Available batch_token_counts: "
                f"{available_batch_token_counts}."
            ) from exc

        for src_value, dst_value in zip(
            input_buffers, captured_inputs, strict=True
        ):
            dst_value.inplace_copy_from(src_value)

        if debug_verify_replay:
            verify_inputs = debug_verify_model_inputs or model_inputs
            verify_graph_key = _replay_graph_key_from_model_inputs(
                verify_inputs
            )
            if verify_graph_key != replay_graph_key:
                raise ValueError(
                    "debug_verify_model_inputs must map to the same graph key "
                    "as replay inputs: "
                    f"{verify_graph_key} != {replay_graph_key}."
                )
            self._model.debug_verify_replay(
                packed_model_graph_key,
                *verify_inputs.buffers,
            )

        self._model.replay(packed_model_graph_key, *captured_inputs)
        return outputs
