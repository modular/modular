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
"""Tests that graph capture, replay and execute bind the same model inputs.

The graph takes the pinned Engram tables as trailing CPU inputs, so every
path that binds the model's inputs must include them. Under data parallelism
the batch processor also stages each replica's bounds in the distributed
graph's input order.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import Model
from max.graph import DeviceRef
from max.nn.legacy.kv_cache import BatchCharacteristics, MHAAttnKey
from max.nn.legacy.transformer import ReturnLogits
from max.pipelines.architectures.deepseekV41.model import (
    DeepseekV41BatchProcessor,
    DeepseekV41Inputs,
    DeepseekV41Model,
)
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.lib.graph_capture import ServeGraphCaptureRunner
from max.pipelines.lib.interfaces.batch_processor import BatchProcessorRuntime

NUM_DEVICES = 4
ENGRAM_LAYERS = (1, 14)


def _buf(*shape: int) -> Buffer:
    return Buffer.from_numpy(np.zeros(shape, dtype=np.int64))


class _RecordingModel:
    """Stands in for the compiled engine model; records every binding."""

    def __init__(self) -> None:
        self.output = Buffer.zeros((1, 4), dtype=DType.float32)
        self.calls: dict[str, list[list[Buffer]]] = {
            "capture": [],
            "replay": [],
            "execute": [],
        }

    def capture(self, graph_key: int, *buffers: Buffer) -> list[Buffer]:
        self.calls["capture"].append(list(buffers))
        return [self.output]

    def replay(self, graph_key: int, *buffers: Buffer) -> None:
        self.calls["replay"].append(list(buffers))

    def execute(self, *buffers: Buffer) -> list[Buffer]:
        self.calls["execute"].append(list(buffers))
        return [self.output]

    def _await_device_graphs(self) -> None:
        """Capture instantiates nothing asynchronously here."""


class _StubProcessor:
    """Builds ``DeepseekV41Inputs`` with placeholder buffers for each field."""

    def __init__(self) -> None:
        self.signal_buffers = [_buf(1) for _ in range(NUM_DEVICES)]
        self.kv_leaves = [_buf(2) for _ in range(3 * NUM_DEVICES)]

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[Any]],
        kv_cache_inputs: Any = None,
        return_n_logits: int = 1,
    ) -> DeepseekV41Inputs:
        batch = sum(len(b) for b in replica_batches)
        return DeepseekV41Inputs(
            tokens=_buf(batch),
            input_row_offsets=_buf(batch + 1),
            return_n_logits=_buf(1),
            kv_cache_inputs=cast(Any, list(self.kv_leaves)),
            signal_buffers=list(self.signal_buffers),
            engram_lookback_ids=_buf(batch, 2),
        )

    def process_outputs(self, outputs: Sequence[Buffer]) -> Sequence[Buffer]:
        return outputs


def _pipeline_model() -> tuple[DeepseekV41Model, list[Buffer]]:
    """A ``DeepseekV41Model`` with only the state these paths read."""
    pipeline_model = object.__new__(DeepseekV41Model)
    engram = [
        _buf(3, 2)
        for _ in range(NUM_DEVICES)
        for _ in ENGRAM_LAYERS
        for _ in ("weight", "scale")
    ]
    pipeline_model._engram_host_buffers = engram
    pipeline_model._batch_processor = cast(Any, _StubProcessor())
    pipeline_model.model = cast(Model, _RecordingModel())
    return pipeline_model, engram


def _kv_params() -> MagicMock:
    kv_params = MagicMock()
    kv_params.devices = [DeviceRef.CPU()]
    kv_params.num_draft_tokens_per_step = 1
    kv_params.graph_capture_probe_cache_lengths.return_value = [1, 10]
    kv_params.resolve_attn_key.side_effect = lambda b, q, c: MHAAttnKey(
        batch_size=b, max_prompt_length=q, num_partitions=c
    )
    return kv_params


def test_capture_replay_and_execute_bind_the_same_inputs() -> None:
    """Capture, replay and execute bind one input list, Engram tables last."""
    pipeline_model, engram = _pipeline_model()
    engine = cast(_RecordingModel, pipeline_model.model)

    @contextmanager
    def warmup_inputs(
        batch_size: int, batch_characteristics: BatchCharacteristics
    ) -> Iterator[DeepseekV41Inputs]:
        # Builds warmup inputs through the pipeline model, as serving does.
        yield cast(
            DeepseekV41Inputs,
            pipeline_model.prepare_initial_token_inputs(
                replica_batches=cast(Any, [[object()] * batch_size]),
                kv_cache_inputs=None,
                return_n_logits=0,
            ),
        )

    runner = ServeGraphCaptureRunner(
        model=cast(Model, engine),
        kv_params=_kv_params(),
        warmup_model_inputs=warmup_inputs,
        max_cache_length_upper_bound=10,
        max_batch_size=2,
    )
    runner.warmup_pre_ready()
    assert engine.calls["capture"]

    inputs = cast(
        DeepseekV41Inputs,
        pipeline_model.prepare_initial_token_inputs(
            cast(Any, [[object(), object()]])
        ),
    )
    pipeline_model.execute(inputs)
    (executed,) = engine.calls["execute"]
    # The engine's input list: the batch's own inputs, then the tables.
    assert executed[-len(engram) :] == engram
    assert len(executed) == 3 + 1 + NUM_DEVICES + 3 * NUM_DEVICES + len(engram)

    for captured in engine.calls["capture"]:
        assert len(captured) == len(executed)
        assert captured[-len(engram) :] == engram

    characteristics = runner.align(
        BatchCharacteristics(
            batch_size=2, max_prompt_length=1, max_cache_valid_length=7
        )
    )
    runner.replay(model_inputs=inputs, batch_characteristics=characteristics)
    (replayed,) = engine.calls["replay"]
    assert len(replayed) == len(executed)
    assert replayed[-len(engram) :] == engram


def test_execute_rebinds_tables_on_processor_built_inputs() -> None:
    """Processor-built inputs, which lack the tables, execute with them."""
    pipeline_model, engram = _pipeline_model()
    engine = cast(_RecordingModel, pipeline_model.model)
    processor = cast(_StubProcessor, pipeline_model._batch_processor)
    inputs = processor.prepare_initial_token_inputs([[object()]])
    assert inputs.buffers[-1] is not engram[-1]
    pipeline_model.execute(inputs)
    (executed,) = engine.calls["execute"]
    assert executed[-len(engram) :] == engram


# --- DSpark (the speculative pairing's batches) ------------------------------

DRAFT_WIDTH = 5


def _context(active: int, position: int) -> TextContext:
    """A request with ``position`` tokens, the last ``active`` of them new."""
    ctx = TextContext(
        max_length=64, tokens=TokenBuffer(np.ones(position, dtype=np.int64))
    )
    ctx.tokens.skip_processing(position - active)
    return ctx


def test_dspark_dp4_batches_stage_each_replicas_bounds() -> None:
    """At DP4 the processor stages each replica's bounds and context length.

    The pairing's graph reads them beside the KV cache, including for an
    idle replica, and a marker as long as the prompt tokens plus every draft.
    """
    runtime = BatchProcessorRuntime(
        pipeline_config=cast(
            Any,
            SimpleNamespace(
                needs_bitmask_constraints=False, speculative=SimpleNamespace()
            ),
        ),
        devices=[CPU()],
        return_logits=ReturnLogits.VARIABLE,
        max_batch_size=8,
        max_seq_len=64,
        max_batch_input_tokens=64,
        data_parallel_degree=NUM_DEVICES,
    )
    processor = DeepseekV41BatchProcessor(
        cast(
            Any,
            SimpleNamespace(
                engram_layer_ids=(),
                data_parallel_degree=NUM_DEVICES,
                engram_host_buffers=[],
            ),
        ),
        runtime,
    )
    replicas = [
        [_context(3, 10), _context(1, 4)],
        [],
        [_context(2, 7)],
        [_context(5, 5)],
    ]
    inputs = processor.prepare_initial_token_inputs(
        cast(Any, replicas), return_n_logits=DRAFT_WIDTH + 1
    )

    offsets = inputs.host_input_row_offsets
    splits = inputs.data_parallel_splits
    assert offsets is not None and splits is not None
    assert offsets.dtype == DType.uint32
    assert offsets.to_numpy().tolist() == [0, 3, 4, 6, 11]
    assert splits.dtype == DType.int64
    assert splits.to_numpy().tolist() == [0, 2, 2, 3, 4]
    lengths = inputs.batch_context_lengths
    assert [b.dtype for b in lengths] == [DType.int32] * NUM_DEVICES
    assert [b.to_numpy().tolist() for b in lengths] == [[14], [0], [7], [5]]
    assert inputs.merged_rows is not None
    assert inputs.merged_rows.shape == (11 + 4 * DRAFT_WIDTH,)
