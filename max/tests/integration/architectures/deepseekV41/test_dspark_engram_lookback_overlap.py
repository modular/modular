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
"""Tests the DSpark verify step's Engram lookback under the overlap scheduler.

The overlap pipeline prepares step ``n + 1`` while step ``n`` still runs, so
the host does not yet know how many of step ``n``'s drafts the device
accepted, and a request's ``processed_length`` trails its device position by
that count. The batch processor must then take the request's window from the
history the previous step's graph recorded (``read_engram_lookback`` and
``record_engram_lookback``), not from ``processed_length``.

The test drives real contexts in the order ``OverlapTextGenerationPipeline``
runs a step: prepare and launch step ``n``, sync step ``n - 1``, then mark
step ``n`` in flight. The launch runs the real batch processor and a graph of
the two lookback functions, with the previous step's tokens and drafts
realized as ``RealizeFutureTokenProcessor`` does. Every row's window must be
the tokens just before the device's chunk.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import CPU, Accelerator, Buffer, Device, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, Graph, TensorType
from max.nn.kv_cache import BatchCharacteristics, MHAAttnKey
from max.nn.transformer import ReturnLogits
from max.pipelines.architectures.deepseekV41.model_config import (
    DeepseekV41Config,
)
from max.pipelines.architectures.unified_dspark_deepseekV41.batch_processor import (
    UnifiedDSparkDeepseekV41BatchProcessor,
    UnifiedDSparkDeepseekV41Inputs,
)
from max.pipelines.architectures.unified_dspark_deepseekV41.model import (
    read_engram_lookback,
    record_engram_lookback,
)
from max.pipelines.context import SpecDecodingState, TextContext, TokenBuffer
from max.pipelines.lib.graph_capture import ServeGraphCaptureRunner
from max.pipelines.lib.interfaces.batch_processor import BatchProcessorRuntime
from max.pipelines.lib.pipeline_variants.overlap_text_generation import (
    in_flight_draft_placeholders,
)
from max.pipelines.lib.pipeline_variants.utils import (
    update_spec_decode_context_and_prepare_responses,
)
from max.pipelines.modeling.types import RequestID
from max.pipelines.speculative.config import MAGIC_DRAFT_TOKEN_ID

DEPTH = 3
"""The checkpoint's ``engram_max_ngram_size - 1``."""

NUM_DRAFTS = 5
"""The checkpoint's ``dspark_block_size``, the drafts each step verifies."""

MAX_BATCH_SIZE = 8
MAX_LENGTH = 4096
LOW_TOKEN, HIGH_TOKEN = 100, 1000
"""Token ids the test samples; none is a placeholder id."""


def _lookback_graph(device: DeviceRef) -> Graph:
    """The verify graph's lookback handling with the model left out."""
    input_types: list[TensorType | BufferType] = [
        TensorType(DType.int64, ["batch", DEPTH], device=device),
        TensorType(DType.int64, ["batch"], device=device),
        BufferType(DType.int64, ["history_rows", DEPTH], device=device),
        TensorType(DType.int64, ["total_seq_len"], device=device),
        TensorType(DType.uint32, ["offsets_len"], device=device),
        TensorType(DType.int64, ["draft_batch", "num_drafts"], device=device),
        TensorType(DType.int64, ["accepted_batch"], device=device),
    ]
    with Graph("dspark_engram_lookback", input_types=input_types) as graph:
        host, source, history, tokens, offsets, drafts, accepted = graph.inputs
        window = read_engram_lookback(
            host.tensor, source.tensor, history.buffer
        )
        record_engram_lookback(
            history.buffer,
            window,
            tokens.tensor,
            offsets.tensor,
            drafts.tensor,
            accepted.tensor,
        )
        graph.output(window)
    return graph


@dataclass
class _Device:
    device: Device
    model: Model


_DEVICE_KINDS = ["cpu", *(["gpu"] if accelerator_count() else [])]


@pytest.fixture(scope="module", params=_DEVICE_KINDS)
def lookback_device(request: pytest.FixtureRequest) -> _Device:
    if request.param == "cpu":
        device: Device = CPU()
        device_ref = DeviceRef.CPU()
    else:
        device = Accelerator()
        device_ref = DeviceRef.GPU()
    session = InferenceSession(devices=[device])
    return _Device(device, session.load(_lookback_graph(device_ref)))


def _processor(
    device: Device, dp: int
) -> UnifiedDSparkDeepseekV41BatchProcessor:
    config = cast(
        DeepseekV41Config,
        SimpleNamespace(engram_layer_ids=(1, 14), engram_max_ngram_size=4),
    )
    runtime = BatchProcessorRuntime(
        pipeline_config=cast(
            Any, SimpleNamespace(needs_bitmask_constraints=False)
        ),
        devices=[device],
        return_logits=ReturnLogits.VARIABLE,
        max_batch_size=MAX_BATCH_SIZE,
        max_seq_len=MAX_LENGTH,
        max_batch_input_tokens=MAX_LENGTH,
        data_parallel_degree=dp,
    )
    return UnifiedDSparkDeepseekV41BatchProcessor(config, runtime)


@dataclass
class _Request:
    """A request and what the device holds for it."""

    ctx: TextContext
    truth: list[int]
    """The device's tokens, through the first token of its next chunk."""
    start: int = 0
    """The device position of the request's next chunk."""
    chunks: list[int] = field(default_factory=list)
    """Prompt chunk sizes still to run; empty once the request decodes."""

    @classmethod
    def prompt(
        cls, rng: np.random.Generator, chunks: Sequence[int]
    ) -> _Request:
        tokens = rng.integers(LOW_TOKEN, HIGH_TOKEN, size=sum(chunks))
        ctx = TextContext(
            request_id=RequestID(),
            max_length=MAX_LENGTH,
            tokens=TokenBuffer(tokens.astype(np.int64)),
        )
        return cls(ctx, tokens.tolist(), 0, list(chunks))

    @classmethod
    def padding(cls) -> _Request:
        """A data-parallel padding dummy, built as ``DPBatchPadder`` does."""
        ctx = TextContext.new_padding_context(
            max_length=MAX_LENGTH, model_name=""
        )
        ctx.update(0)
        return cls(ctx, [0, 0], 1)

    def preempt(self) -> None:
        """Resets the request; the device re-encodes what the host kept."""
        self.ctx.reset()
        self.truth = np.asarray(self.ctx.tokens.all).tolist()
        self.start = 0
        self.chunks = [len(self.truth)]

    def expected_window(self) -> list[int]:
        return [
            self.truth[self.start - 1 - j] if self.start - 1 - j >= 0 else -1
            for j in range(DEPTH)
        ]


@dataclass
class _Step:
    """What a launched step leaves for the next one and for its sync."""

    contexts: list[TextContext]
    drafts: npt.NDArray[np.int64]
    accepted: npt.NDArray[np.int64]
    bonus: npt.NDArray[np.int64]
    next_drafts: npt.NDArray[np.int64]


@dataclass
class _Stats:
    rows: int = 0
    from_history: int = 0
    host_window_stale: int = 0
    accepted_before_history_read: set[int] = field(default_factory=set)
    mixed_verify_steps: int = 0
    """Steps that verified drafts while some row still prefilled."""
    from_history_after_mixed: int = 0
    """Rows that read a window a mixed verify step recorded."""


class _Pipeline:
    """Runs steps in ``OverlapTextGenerationPipeline._execute`` order."""

    def __init__(
        self,
        lookback: _Device,
        *,
        overlap: bool,
        dp: int,
        rng: np.random.Generator,
        mixed: bool = False,
    ) -> None:
        self._lookback = lookback
        self._processor = _processor(lookback.device, dp)
        self._overlap = overlap
        self._mixed = mixed
        self._prev_was_mixed = False
        self._rng = rng
        self._prev: _Step | None = None
        self._num_decode_rows = 0
        self.stats = _Stats()

    def _to_device(self, array: npt.NDArray[Any]) -> Buffer:
        return Buffer.from_numpy(np.ascontiguousarray(array)).to(
            self._lookback.device
        )

    def step(self, replicas: Sequence[Sequence[_Request]]) -> None:
        flat = [r for replica in replicas for r in replica]
        contexts = [r.ctx for r in flat]
        for r in flat:
            if r.chunks:
                size = r.chunks.pop(0)
                if size < r.ctx.tokens.active_length:
                    r.ctx.tokens.chunk(size)
        # A pure decode step verifies drafts; with ``mixed``, so does a
        # prefill+decode step, whose prefill rows carry only placeholders.
        decoding = [bool(c.tokens.generated_length) for c in contexts]
        verifies = all(decoding) or (self._mixed and any(decoding))
        k = NUM_DRAFTS if verifies else 0
        is_mixed = verifies and not all(decoding)
        # `_execute_spec_decode`: the drafts the host has, or placeholders.
        for c in contexts:
            state = c.spec_decoding_state
            if k == 0:
                state.draft_tokens_to_verify = []
            elif not state.draft_tokens_to_verify:
                state.draft_tokens_to_verify = [MAGIC_DRAFT_TOKEN_ID] * k
            else:
                state.draft_tokens_to_verify = state.draft_tokens_to_verify[:k]
        drafts = np.array(
            [c.spec_decoding_state.draft_tokens_to_verify for c in contexts],
            dtype=np.int64,
        ).reshape(len(contexts), k)

        inputs = self._processor.prepare_initial_token_inputs(
            [[r.ctx for r in replica] for replica in replicas]
        )
        assert isinstance(inputs, UnifiedDSparkDeepseekV41Inputs)
        assert inputs.engram_lookback_ids is not None
        assert inputs.engram_lookback_source is not None
        assert inputs.engram_lookback_history is not None

        tokens = np.concatenate(
            [np.asarray(c.tokens.active, dtype=np.int64) for c in contexts]
        )
        offsets = np.cumsum(
            [0] + [c.tokens.active_length for c in contexts], dtype=np.uint32
        )
        # `RealizeFutureTokenProcessor`: rows the previous step also ran take
        # its sampled token, its drafts, and its accepted count.
        prev = self._prev
        prev_rows = (
            {c.request_id: i for i, c in enumerate(prev.contexts)}
            if prev is not None
            else {}
        )
        accepted_before = [-1] * len(flat)
        for row, (r, c) in enumerate(zip(flat, contexts, strict=True)):
            cache_length = c.tokens.processed_length + len(
                c.spec_decoding_state.maybe_accepted_draft_tokens
            )
            i = prev_rows.get(c.request_id)
            if prev is not None and i is not None and c.tokens.generated_length:
                tokens[offsets[row + 1] - 1] = prev.bonus[i]
                drafts[row] = prev.next_drafts[i, :k]
                accepted_before[row] = int(prev.accepted[i])
                cache_length += int(prev.accepted[i]) - prev.drafts.shape[1]
            # Guards the simulation itself: the device's chunk starts at the
            # cache length the pipeline realizes.
            assert cache_length == r.start
            chunk = tokens[offsets[row] : offsets[row + 1]].tolist()
            assert chunk == r.truth[r.start : r.start + len(chunk)]

        accepted = np.zeros(len(flat), dtype=np.int64)
        if k:
            for row in range(len(flat)):
                # The driver reports zero for rows of placeholders only.
                if not np.all(drafts[row] == MAGIC_DRAFT_TOKEN_ID):
                    accepted[row] = self._num_decode_rows % (k + 1)
                    self._num_decode_rows += 1

        expected = np.array([r.expected_window() for r in flat], np.int64)
        host_window = inputs.engram_lookback_ids.to(CPU()).to_numpy()
        source = inputs.engram_lookback_source.to(CPU()).to_numpy()
        (window_buffer,) = self._lookback.model.execute(
            inputs.engram_lookback_ids,
            inputs.engram_lookback_source,
            inputs.engram_lookback_history,
            self._to_device(tokens),
            self._to_device(offsets),
            self._to_device(drafts),
            self._to_device(accepted),
        )
        assert isinstance(window_buffer, Buffer)
        window = window_buffer.to(CPU()).to_numpy()
        np.testing.assert_array_equal(window, expected)

        stats = self.stats
        stats.rows += len(flat)
        stats.from_history += int(np.sum(source >= 0))
        stats.mixed_verify_steps += int(is_mixed)
        if self._prev_was_mixed:
            stats.from_history_after_mixed += int(np.sum(source >= 0))
        stats.host_window_stale += int(
            np.sum(np.any(host_window != expected, axis=1))
        )
        for row in np.flatnonzero(source >= 0):
            assert accepted_before[row] >= 0, "read a row not in flight"
            stats.accepted_before_history_read.add(accepted_before[row])

        bonus = self._rng.integers(LOW_TOKEN, HIGH_TOKEN, size=len(flat))
        next_drafts = self._rng.integers(
            LOW_TOKEN, HIGH_TOKEN, size=(len(flat), NUM_DRAFTS)
        )
        for row, r in enumerate(flat):
            if r.ctx.tokens.generated_length:
                r.truth.extend(drafts[row, : accepted[row]].tolist())
                r.truth.append(int(bonus[row]))
                r.start += 1 + int(accepted[row])
            else:
                r.start += r.ctx.tokens.active_length
                if not r.ctx.tokens.actively_chunked:
                    r.truth.append(int(bonus[row]))

        launched = _Step(contexts, drafts, accepted, bonus, next_drafts)
        self._prev_was_mixed = is_mixed
        if self._overlap:
            if prev is not None:
                self._sync(prev)
            self._mark_in_flight(contexts, k)
            self._prev = launched
        else:
            self._mark_in_flight(contexts, k)
            self._sync(launched)

    def drain(self) -> None:
        if self._prev is not None:
            self._sync(self._prev)
            self._prev = None

    def _mark_in_flight(self, contexts: Sequence[TextContext], k: int) -> None:
        for c in contexts:
            c.update_with_future_token()
            c.spec_decoding_state.maybe_accepted_draft_tokens = (
                in_flight_draft_placeholders(c, k)
            )
            if c.tokens.generated_length and self._overlap:
                c.spec_decoding_state.draft_tokens_to_verify = []

    @staticmethod
    def _sync(step: _Step) -> None:
        update_spec_decode_context_and_prepare_responses(
            draft_tokens=cast(Any, step.drafts),
            next_draft_tokens=cast(Any, step.next_drafts),
            num_accepted_draft_tokens=cast(Any, step.accepted),
            next_tokens=cast(Any, step.bonus),
            context_batch=step.contexts,
            max_seq_len=MAX_LENGTH,
        )


def _run(
    lookback: _Device,
    *,
    overlap: bool,
    dp: int,
    num_steps: int = 48,
    mixed: bool = False,
) -> _Stats:
    """Runs a schedule that joins, chunks, skips, reorders and preempts.

    Requests join mid-run with prompts chunked down to one token, a request
    sits out a step, every step shuffles the batch, one request is preempted
    while its step is in flight and runs again at once, and another finishes.
    """
    rng = np.random.default_rng(1234 + dp + 10 * overlap)
    pipeline = _Pipeline(lookback, overlap=overlap, dp=dp, rng=rng, mixed=mixed)
    joins = {
        0: [[7], [2], [4, 1, 6]],
        9: [[1, 1, 1, 5]],
        21: [[3]],
        30: [[2, 2]],
    }
    running: list[_Request] = []
    finished: set[int] = set()
    for t in range(num_steps):
        running.extend(
            _Request.prompt(rng, chunks) for chunks in joins.get(t, [])
        )
        if t == 26:
            running[2].preempt()
        if t == 36:
            finished.add(id(running[0]))
        batch = [
            r
            for i, r in enumerate(running)
            if id(r) not in finished and not (i == 1 and t % 7 == 3)
        ]
        order = rng.permutation(len(batch))
        batch = [batch[i] for i in order]
        replicas = [batch[i::dp] for i in range(dp)]
        decoding = all(r.ctx.tokens.generated_length for r in batch)
        if dp > 1 and decoding:
            # `DPBatchPadder` pads short replicas of decode steps at the end.
            widest = max(len(replica) for replica in replicas)
            for replica in replicas:
                replica.extend(
                    _Request.padding() for _ in range(widest - len(replica))
                )
        pipeline.step(replicas)
    pipeline.drain()
    return pipeline.stats


@pytest.mark.parametrize("dp", [1, 2])
def test_verify_lookback_follows_accepted_drafts(
    lookback_device: _Device, dp: int
) -> None:
    """Every window matches the device position, for any accepted count."""
    stats = _run(lookback_device, overlap=True, dp=dp)
    assert stats.accepted_before_history_read == set(range(NUM_DRAFTS + 1))
    # The schedule does exercise the stale host window.
    assert stats.host_window_stale > 0


@pytest.mark.parametrize("dp", [1, 2])
def test_mixed_verify_lookback_follows_accepted_drafts(
    lookback_device: _Device, dp: int
) -> None:
    """Mixed prefill+decode steps that verify keep every window exact.

    A prefill row in such a step carries placeholder drafts and accepts
    none, so the window it records ends at its chunk; its decode rows record
    the window past their accepted drafts, as in a pure decode step.
    """
    stats = _run(lookback_device, overlap=True, dp=dp, mixed=True)
    assert stats.mixed_verify_steps > 0
    assert stats.from_history_after_mixed > 0
    assert stats.accepted_before_history_read == set(range(NUM_DRAFTS + 1))


def test_without_overlap_the_host_window_is_used(
    lookback_device: _Device,
) -> None:
    """With overlap off nothing is in flight, so no row reads the history."""
    stats = _run(lookback_device, overlap=False, dp=1)
    assert stats.from_history == 0
    assert stats.host_window_stale == 0


class _HistoryWritingEngine:
    """Stands in for the compiled graph; replay writes the bound history."""

    def __init__(self, marker: Buffer) -> None:
        self.marker = marker
        self.output = Buffer.zeros((1,), DType.int64, device=marker.device)
        self.history_index = -1
        self.captured: list[list[Buffer]] = []
        self.replayed: list[list[Buffer]] = []

    def capture(self, graph_key: int, *buffers: Buffer) -> list[Buffer]:
        self.captured.append(list(buffers))
        return [self.output, self.output, self.output]

    def replay(self, graph_key: int, *buffers: Buffer) -> None:
        self.replayed.append(list(buffers))
        buffers[self.history_index].inplace_copy_from(self.marker)

    def _await_device_graphs(self) -> None:
        """Capture instantiates nothing asynchronously here."""


@pytest.mark.skipif(
    not accelerator_count(), reason="replay refreshes device inputs only"
)
def test_replay_records_into_the_history_the_next_step_reads() -> None:
    """Captured graphs bind the live history, so replayed writes persist."""
    device = Accelerator()
    processor = _processor(device, dp=1)
    kv_cache_inputs = cast(Any, [Buffer.zeros((2,), DType.int64, device)])

    def spec_inputs(
        contexts: list[TextContext],
    ) -> UnifiedDSparkDeepseekV41Inputs:
        inputs = processor.prepare_initial_token_inputs(
            [contexts], kv_cache_inputs, NUM_DRAFTS + 1
        )
        batch = len(contexts)
        inputs.draft_tokens = Buffer.zeros(
            (batch, NUM_DRAFTS), DType.int64, device
        )
        for name in ("temperature", "top_k", "top_p"):
            setattr(inputs, name, Buffer.zeros((batch,), DType.float32, device))
        inputs.max_k = Buffer.zeros((1,), DType.int64)
        inputs.min_top_p = Buffer.zeros((1,), DType.float32)
        return inputs

    @contextmanager
    def warmup_inputs(
        batch_size: int, batch_characteristics: BatchCharacteristics
    ) -> Iterator[UnifiedDSparkDeepseekV41Inputs]:
        # As `OverlapTextGenerationPipeline._warmup_model_inputs` builds them.
        yield spec_inputs(
            [
                TextContext(
                    max_length=MAX_LENGTH,
                    tokens=TokenBuffer(np.zeros(1, dtype=np.int64)),
                    _spec_decoding_state=SpecDecodingState(
                        draft_tokens_to_verify=[0] * NUM_DRAFTS
                    ),
                )
                for _ in range(batch_size)
            ]
        )

    marker_np = np.arange(MAX_BATCH_SIZE * DEPTH, dtype=np.int64).reshape(
        MAX_BATCH_SIZE, DEPTH
    )
    engine = _HistoryWritingEngine(Buffer.from_numpy(marker_np).to(device))
    kv_params = MagicMock()
    kv_params.num_draft_tokens_per_step = NUM_DRAFTS
    kv_params.graph_capture_probe_cache_lengths.return_value = [8]
    kv_params.resolve_attn_key.side_effect = lambda b, q, c: MHAAttnKey(
        batch_size=b, max_prompt_length=q, num_partitions=c
    )
    runner = ServeGraphCaptureRunner(
        model=cast(Model, engine),
        kv_params=kv_params,
        warmup_model_inputs=warmup_inputs,
        max_cache_length_upper_bound=8,
        max_batch_size=2,
        num_speculative_tokens=NUM_DRAFTS,
    )
    runner.warmup_pre_ready()
    history = processor._engram_history
    assert history is not None
    assert engine.captured
    for captured in engine.captured:
        bound = [i for i, b in enumerate(captured) if b is history]
        assert len(bound) == 1, "a capture did not bind the live history"
        engine.history_index = bound[0]

    rng = np.random.default_rng(7)
    requests = [_Request.prompt(rng, [4]) for _ in range(2)]
    for r in requests:
        r.ctx.update(int(rng.integers(LOW_TOKEN, HIGH_TOKEN)))
    inputs = spec_inputs([r.ctx for r in requests])
    assert inputs.engram_lookback_history is history
    runner.replay(
        model_inputs=inputs,
        batch_characteristics=runner.align(
            BatchCharacteristics(
                batch_size=2,
                max_prompt_length=NUM_DRAFTS + 1,
                max_cache_valid_length=8,
            )
        ),
    )
    (replayed,) = engine.replayed
    assert replayed[engine.history_index] is history
    np.testing.assert_array_equal(history.to(CPU()).to_numpy(), marker_np)
