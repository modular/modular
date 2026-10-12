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
"""Tests for PipelineModel device graph capture plumbing.

These exercise the ``ServeGraphCaptureRunner`` capture/record/align/replay flow
on CPU with mocked KV cache params (no real attention kernels): capture records
a ``BatchCharacteristics -> GraphKey`` map and the recorded cache lengths;
``align`` buckets a runtime cache length up to a recorded one and looks up the
captured graph; ``replay`` copies inputs into the captured buffers and replays.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import Model
from max.experimental.compilation import CompiledCallable
from max.graph import DeviceRef
from max.nn.kv_cache import AttnKeyInterface, BatchCharacteristics, MHAAttnKey
from max.nn.kv_cache.utils import MultiAttnKey
from max.pipelines.lib import MemoryPlan, ModelInputs, ModelOutputs
from max.pipelines.lib.graph_capture import (
    _HOST_INPUT_GUARD_ENV,
    ServeGraphCaptureRunner,
)
from max.pipelines.lib.interfaces import UnifiedEagleOutputs
from test_common.mocks.pipeline_config import (
    DummyPipelineConfig,
    mock_huggingface_config,
)
from test_common.mocks.pipeline_model import MockModelInputs, MockPipelineModel


class DummyModel:
    def __init__(self, output_buffer: Buffer) -> None:
        self.output_buffer = output_buffer
        self.input_devices = [CPU()]
        self.input_metadata = [
            SimpleNamespace(name=name)
            for name in ("tokens", "input_row_offsets", "return_n_logits")
        ]
        self.capture_calls: list[tuple[int, list[Buffer]]] = []
        self.replay_calls: list[tuple[int, list[Buffer]]] = []
        self.debug_verify_replay_calls: list[tuple[int, list[Buffer]]] = []

    def capture(self, graph_key: int, *buffers: Buffer) -> list[Buffer]:
        self.capture_calls.append((graph_key, list(buffers)))
        return [self.output_buffer]

    def replay(self, graph_key: int, *buffers: Buffer) -> None:
        self.replay_calls.append((graph_key, list(buffers)))

    def debug_verify_replay(self, graph_key: int, *buffers: Buffer) -> None:
        self.debug_verify_replay_calls.append((graph_key, list(buffers)))

    def _await_device_graphs(self) -> None:
        pass


class EagleDummyModel(DummyModel):
    """DummyModel that returns 3 output buffers for Eagle capture."""

    def capture(self, graph_key: int, *buffers: Buffer) -> list[Buffer]:
        self.capture_calls.append((graph_key, list(buffers)))
        return [self.output_buffer, self.output_buffer, self.output_buffer]


class CapturePipelineModel(MockPipelineModel):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.input_buffer = Buffer.zeros((4,), dtype=DType.float32)
        self.output_buffer = Buffer.zeros((4,), dtype=DType.float32)
        self.model = DummyModel(self.output_buffer)

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        return ModelOutputs(logits=self.output_buffer)


def _mock_kv_params(
    *,
    is_mla: bool = False,
    data_parallel_degree: int = 1,
    probe_lengths: list[int] | None = None,
    num_partitions_for_length: Any = None,
) -> MagicMock:
    """Builds a MagicMock ``KVCacheParams`` for runner tests.

    ``resolve_attn_key`` returns a real :class:`MHAAttnKey` so the runner can
    read ``num_partitions`` / ``batch_size`` off it. ``num_partitions_for_length``
    maps a probed cache length to a ``num_partitions`` (defaults to the length
    itself, so each probed length yields a distinct key).
    """
    if num_partitions_for_length is None:
        num_partitions_for_length = lambda length: length

    kv_params = MagicMock()
    kv_params.devices = [DeviceRef.CPU()]
    kv_params.n_kv_heads_per_device = 1
    kv_params.is_mla = is_mla
    kv_params.data_parallel_degree = data_parallel_degree
    kv_params.num_draft_tokens_per_step = 1
    kv_params.graph_capture_probe_cache_lengths.return_value = (
        probe_lengths if probe_lengths is not None else [1, 10]
    )

    def _resolve(batch_size: int, q: int, cache_length: int) -> MHAAttnKey:
        return MHAAttnKey(
            batch_size=batch_size,
            max_prompt_length=q,
            num_partitions=num_partitions_for_length(cache_length),
        )

    kv_params.resolve_attn_key.side_effect = _resolve
    return kv_params


def _make_runner(
    model: DummyModel,
    *,
    max_batch_size: int = 1,
    num_speculative_tokens: int = 0,
    kv_params: MagicMock | None = None,
    warmup_model_inputs: Any = None,
    widths_by_batch_size: Sequence[Sequence[int]] | None = None,
) -> ServeGraphCaptureRunner:
    return ServeGraphCaptureRunner(
        model=cast(Model, model),
        kv_params=kv_params if kv_params is not None else _mock_kv_params(),
        warmup_model_inputs=warmup_model_inputs or MagicMock(),
        max_cache_length_upper_bound=10,
        max_batch_size=max_batch_size,
        num_speculative_tokens=num_speculative_tokens,
        widths_by_batch_size=widths_by_batch_size,
    )


def _bc(batch_size: int, q: int, cache_length: int) -> BatchCharacteristics:
    return BatchCharacteristics(
        batch_size=batch_size,
        max_prompt_length=q,
        max_cache_valid_length=cache_length,
    )


def _gk(
    *,
    num_partitions: int,
    q_max_seq_len: int,
    batch_size: int = 1,
    spec: bool = False,
) -> MultiAttnKey:
    """Builds the capture key the way ``ServeGraphCaptureRunner`` does.

    The runner folds the resolved verify-width dispatch metadata and, under
    speculative decoding, the draft-width metadata into a ``MultiAttnKey``
    keyed by ``"verify"``/``"draft"``.
    """
    children: dict[str, AttnKeyInterface] = {
        "verify": MHAAttnKey(
            batch_size=batch_size,
            max_prompt_length=q_max_seq_len,
            num_partitions=num_partitions,
        )
    }
    if spec:
        children["draft"] = MHAAttnKey(
            batch_size=batch_size,
            max_prompt_length=1,
            num_partitions=num_partitions,
        )
    return MultiAttnKey.from_dict(children)


@pytest.fixture
@mock_huggingface_config
def capture_model() -> CapturePipelineModel:
    pipeline_config = DummyPipelineConfig(
        model_path="test/model",
        quantization_encoding=MagicMock(),
        max_batch_size=4,
        max_length=128,
        device_graph_capture=True,
    )
    return CapturePipelineModel(
        pipeline_config=pipeline_config,
        session=MagicMock(),
        devices=[CPU()],
        kv_cache_config=MagicMock(),
        weights=MagicMock(),
        adapter=None,
        return_logits=MagicMock(),
        # The value the mock's clamp derived before plans became required:
        # min(MOCK_MODEL_MAX_SEQ_LEN, max_length=128).
        memory_plan=MemoryPlan(
            planned_max_batch_size=4,
            footprint=0,
            planned_max_length=128,
        ),
    )


# ---------------------------------------------------------------------------
# replay()
# ---------------------------------------------------------------------------


def test_replay_copies_inputs_and_replays(
    capture_model: CapturePipelineModel,
) -> None:
    inputs = MockModelInputs(active_batch_size=1, eos_prob=0.0)
    runner = _make_runner(capture_model.model)

    key = _gk(num_partitions=1, q_max_seq_len=1)
    bc = _bc(1, 1, 1)
    runner._records[bc] = key
    trace_inputs = inputs.buffers
    runner.graph_entries[key] = (
        trace_inputs,
        ModelOutputs(logits=capture_model.output_buffer),
    )

    output = runner.replay(model_inputs=inputs, batch_characteristics=bc)
    assert capture_model.model.replay_calls
    assert output.logits is capture_model.output_buffer


def test_replay_miss_raises(capture_model: CapturePipelineModel) -> None:
    inputs = MockModelInputs(active_batch_size=1, eos_prob=0.0)
    runner = _make_runner(capture_model.model)
    # No record for these characteristics — replay raises RuntimeError.
    with pytest.raises(RuntimeError, match=r"No captured device graph for"):
        runner.replay(model_inputs=inputs, batch_characteristics=_bc(4, 1, 1))
    assert not capture_model.model.replay_calls


def _runner_with_captured_host_input(
    model: DummyModel,
) -> tuple[ServeGraphCaptureRunner, Buffer]:
    """Returns a runner holding one graph captured with ``return_n_logits=1``.

    On CPU every captured input is host-resident, so ``replay`` sends each one
    through the host-input guard.
    """
    runner = _make_runner(model)
    key = _gk(num_partitions=1, q_max_seq_len=1)
    runner._records[_bc(1, 1, 1)] = key
    captured = MockModelInputs(
        active_batch_size=1, eos_prob=0.0, return_n_logits=1
    )
    runner.graph_entries[key] = (
        captured.buffers,
        ModelOutputs(logits=model.output_buffer),
    )
    return runner, captured.return_n_logits


def test_replay_guard_abort_reads_the_captured_value_before_refresh(
    capture_model: CapturePipelineModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Abort mode raises on a changed host input and leaves the graph unrun.

    The guard only works while the captured buffer still holds its captured
    value; if the refresh copy ran first, it would compare the live value with
    itself and stay silent.
    """
    monkeypatch.setenv(_HOST_INPUT_GUARD_ENV, "abort")
    runner, captured = _runner_with_captured_host_input(capture_model.model)
    live = MockModelInputs(active_batch_size=1, eos_prob=0.0, return_n_logits=4)

    with pytest.raises(
        RuntimeError,
        match=r"'return_n_logits' \(index 2\) changed since capture",
    ):
        runner.replay(model_inputs=live, batch_characteristics=_bc(1, 1, 1))
    assert captured.to_numpy()[0] == 1
    assert not capture_model.model.replay_calls


def test_replay_guard_report_warns_once_and_keeps_replaying(
    capture_model: CapturePipelineModel,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv(_HOST_INPUT_GUARD_ENV, "report")
    runner, captured = _runner_with_captured_host_input(capture_model.model)

    with caplog.at_level(logging.WARNING, logger="max.pipelines"):
        for return_n_logits in (4, 8):
            live = MockModelInputs(
                active_batch_size=1,
                eos_prob=0.0,
                return_n_logits=return_n_logits,
            )
            runner.replay(model_inputs=live, batch_characteristics=_bc(1, 1, 1))

    reports = [
        r for r in caplog.records if "changed since capture" in r.message
    ]
    assert len(reports) == 1
    # Replay does not refresh host inputs, so the captured value stays.
    assert captured.to_numpy()[0] == 1
    assert len(capture_model.model.replay_calls) == 2


def test_replay_debug_verify_uses_verify_inputs(
    capture_model: CapturePipelineModel,
) -> None:
    replay_inputs = MockModelInputs(active_batch_size=1, eos_prob=0.0)
    debug_inputs = MockModelInputs(active_batch_size=1, eos_prob=0.0)
    runner = _make_runner(capture_model.model)

    key = _gk(num_partitions=1, q_max_seq_len=1)
    bc = _bc(1, 1, 1)
    runner._records[bc] = key
    runner.graph_entries[key] = (
        replay_inputs.buffers,
        ModelOutputs(logits=capture_model.output_buffer),
    )

    runner.replay(
        model_inputs=replay_inputs,
        batch_characteristics=bc,
        debug_verify_replay=True,
        debug_verify_model_inputs=debug_inputs,
    )

    assert capture_model.model.debug_verify_replay_calls
    _, verified_buffers = capture_model.model.debug_verify_replay_calls[-1]
    assert verified_buffers == list(debug_inputs.buffers)


def test_replay_returns_eagle_outputs(
    capture_model: CapturePipelineModel,
) -> None:
    runner = _make_runner(capture_model.model, num_speculative_tokens=1)
    buf = Buffer.zeros((4,), dtype=DType.float32)
    eagle_outputs = UnifiedEagleOutputs(
        num_accepted_draft_tokens=buf,
        next_tokens=buf,
        next_draft_tokens=buf,
    )
    inputs = MockModelInputs(active_batch_size=1, eos_prob=0.0)
    key = _gk(num_partitions=5, q_max_seq_len=2, spec=True)
    bc = _bc(1, 2, 5)
    runner._records[bc] = key
    runner.graph_entries[key] = (inputs.buffers, eagle_outputs)

    output = runner.replay(model_inputs=inputs, batch_characteristics=bc)
    assert isinstance(output, UnifiedEagleOutputs)
    assert output.num_accepted_draft_tokens is buf


# ---------------------------------------------------------------------------
# align()
# ---------------------------------------------------------------------------


def test_align_buckets_cache_length_up_and_looks_up_key() -> None:
    runner = _make_runner(DummyModel(Buffer.zeros((4,), dtype=DType.float32)))
    runner._recorded_cache_lengths = [1, 10]
    runner._records = {
        _bc(1, 1, 1): _gk(num_partitions=1, q_max_seq_len=1),
        _bc(1, 1, 10): _gk(num_partitions=10, q_max_seq_len=1),
    }

    # Runtime cache length 5 buckets up to recorded 10.
    aligned = runner.align(_bc(1, 1, 5))
    assert aligned == _bc(1, 1, 10)

    # Exact recorded length stays put.
    aligned = runner.align(_bc(1, 1, 1))
    assert aligned == _bc(1, 1, 1)


def test_align_above_max_recorded_raises() -> None:
    runner = _make_runner(DummyModel(Buffer.zeros((4,), dtype=DType.float32)))
    runner._recorded_cache_lengths = [1, 10]
    runner._records = {_bc(1, 1, 10): _gk(num_partitions=10, q_max_seq_len=1)}

    # A cache length beyond the largest captured length has no valid graph.
    with pytest.raises(RuntimeError, match=r"exceeds the largest captured"):
        runner.align(_bc(1, 1, 9999))


def test_align_q_mismatch_raises() -> None:
    runner = _make_runner(
        DummyModel(Buffer.zeros((4,), dtype=DType.float32)),
        num_speculative_tokens=0,
    )
    runner._recorded_cache_lengths = [10]
    runner._records = {_bc(1, 1, 10): _gk(num_partitions=10, q_max_seq_len=1)}

    with pytest.raises(
        RuntimeError, match=r"verify width 1, which is not captured"
    ):
        runner.align(_bc(1, 2, 5))


def test_align_rejects_a_width_not_captured_at_that_batch_size() -> None:
    """A width captured only at other batch sizes has no graph here."""
    runner = _make_runner(
        EagleDummyModel(Buffer.zeros((4,), dtype=DType.float32)),
        max_batch_size=2,
        num_speculative_tokens=3,
        widths_by_batch_size=[[3], [3], [1, 3]],
    )
    runner._recorded_cache_lengths = [10]

    assert runner.align(_bc(2, 2, 5)) == _bc(2, 2, 10)
    with pytest.raises(
        RuntimeError, match=r"not captured at batch size 1; captured widths"
    ):
        runner.align(_bc(1, 2, 5))


# ---------------------------------------------------------------------------
# warmup_pre_ready()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("num_spec", "expected_type"),
    [(1, UnifiedEagleOutputs), (0, ModelOutputs)],
)
def test_warmup_records_and_captures(
    num_spec: int, expected_type: type
) -> None:
    model: DummyModel = (
        EagleDummyModel(Buffer.zeros((4,), dtype=DType.float32))
        if num_spec > 0
        else DummyModel(Buffer.zeros((4,), dtype=DType.float32))
    )
    mock_inputs = MockModelInputs(active_batch_size=1, eos_prob=0.0)

    @contextmanager
    def _warmup_ctx(
        batch_size: int,
        batch_characteristics: BatchCharacteristics,
    ) -> Iterator[MockModelInputs]:
        yield mock_inputs

    # Distinct num_partitions per probed length -> two captured graphs.
    kv_params = _mock_kv_params(probe_lengths=[1, 10])
    runner = ServeGraphCaptureRunner(
        model=cast(Model, model),
        kv_params=kv_params,
        warmup_model_inputs=_warmup_ctx,
        max_cache_length_upper_bound=10,
        max_batch_size=1,
        num_speculative_tokens=num_spec,
    )
    runner.warmup_pre_ready()

    expected_q = 1 + num_spec
    spec = num_spec > 0
    assert runner._recorded_cache_lengths == [1, 10]
    # Both probed lengths recorded for batch_size=1, each yielding a distinct
    # key whose verify width is ``expected_q``.
    expected_keys = {
        _bc(1, expected_q, 1): _gk(
            num_partitions=1, q_max_seq_len=expected_q, spec=spec
        ),
        _bc(1, expected_q, 10): _gk(
            num_partitions=10, q_max_seq_len=expected_q, spec=spec
        ),
    }
    assert runner._records == expected_keys
    assert set(runner.graph_entries) == set(expected_keys.values())
    for _inputs, outputs in runner.graph_entries.values():
        assert isinstance(outputs, expected_type)


def test_warmup_dedups_shared_keys() -> None:
    """Multiple probed lengths that resolve to one key capture one graph."""
    model = DummyModel(Buffer.zeros((4,), dtype=DType.float32))
    mock_inputs = MockModelInputs(active_batch_size=1, eos_prob=0.0)

    @contextmanager
    def _warmup_ctx(
        batch_size: int,
        batch_characteristics: BatchCharacteristics,
    ) -> Iterator[MockModelInputs]:
        yield mock_inputs

    # Every probed length resolves to the same num_partitions -> one graph,
    # but all lengths are still recorded for bucketing.
    kv_params = _mock_kv_params(
        probe_lengths=[1, 5, 10], num_partitions_for_length=lambda length: 3
    )
    runner = ServeGraphCaptureRunner(
        model=cast(Model, model),
        kv_params=kv_params,
        warmup_model_inputs=_warmup_ctx,
        max_cache_length_upper_bound=10,
        max_batch_size=1,
    )
    runner.warmup_pre_ready()

    assert runner._recorded_cache_lengths == [1, 5, 10]
    assert len(runner.graph_entries) == 1
    assert {bc.max_cache_valid_length for bc in runner._records} == {1, 5, 10}
    assert all(
        key == _gk(num_partitions=3, q_max_seq_len=1)
        for key in runner._records.values()
    )


def test_release_graph_reaches_the_engine_model_of_a_compiled_callable() -> (
    None
):
    compiled = MagicMock(spec=CompiledCallable)
    runner = ServeGraphCaptureRunner(
        model=cast(Model, compiled),
        kv_params=_mock_kv_params(),
        warmup_model_inputs=MagicMock(),
        max_cache_length_upper_bound=10,
        max_batch_size=1,
    )
    runner.release_graph(_gk(num_partitions=1, q_max_seq_len=1))
    compiled.engine_model.release_captured_graph.assert_called_once()


def test_replay_appends_the_signal_buffers_of_a_compiled_callable() -> None:
    compiled = MagicMock(spec=CompiledCallable)
    signal = Buffer.zeros((1,), dtype=DType.float32)
    compiled.signal_buffers = [signal]
    runner = ServeGraphCaptureRunner(
        model=compiled,
        kv_params=_mock_kv_params(),
        warmup_model_inputs=MagicMock(),
        max_cache_length_upper_bound=10,
        max_batch_size=1,
    )
    inputs = MockModelInputs(active_batch_size=1, eos_prob=0.0)
    key = _gk(num_partitions=1, q_max_seq_len=1)
    runner._records[_bc(1, 1, 1)] = key
    runner.graph_entries[key] = (
        inputs.buffers,
        ModelOutputs(logits=signal),
    )

    runner.replay(
        model_inputs=inputs,
        batch_characteristics=_bc(1, 1, 1),
        debug_verify_replay=True,
    )

    engine_model = compiled.engine_model
    verify_args = engine_model.debug_verify_replay.call_args.args
    replay_args = engine_model.replay.call_args.args
    assert verify_args[-1] is signal
    assert replay_args[-1] is signal
    assert len(replay_args) == 1 + len(inputs.buffers) + 1


@pytest.mark.parametrize(
    ("widths_by_batch_size", "expected_widths_by_batch_size"),
    [
        (None, {1: {3}, 2: {3}, 3: {3}}),
        ([[1, 3]] * 4, {1: {1, 3}, 2: {1, 3}, 3: {1, 3}}),
        ([[0], [1], [3], [3]], {1: {1}, 2: {3}, 3: {3}}),
        ([[3], [3], [1, 3], [1, 3]], {1: {3}, 2: {1, 3}, 3: {1, 3}}),
    ],
    ids=["unset", "same-rows", "one-per-row", "mixed-rows"],
)
def test_warmup_probes_widths_per_batch_size(
    widths_by_batch_size: list[list[int]] | None,
    expected_widths_by_batch_size: dict[int, set[int]],
) -> None:
    """Each batch size probes exactly its own row of the table."""
    model = EagleDummyModel(Buffer.zeros((4,), dtype=DType.float32))
    mock_inputs = MockModelInputs(active_batch_size=1, eos_prob=0.0)

    @contextmanager
    def _warmup_ctx(
        batch_size: int, batch_characteristics: BatchCharacteristics
    ) -> Iterator[MockModelInputs]:
        yield mock_inputs

    runner = _make_runner(
        model,
        max_batch_size=3,
        num_speculative_tokens=3,
        kv_params=_mock_kv_params(probe_lengths=[10]),
        warmup_model_inputs=_warmup_ctx,
        widths_by_batch_size=widths_by_batch_size,
    )
    runner.warmup_pre_ready()

    probed: dict[int, set[int]] = {}
    for bc in runner._records:
        probed.setdefault(bc.batch_size, set()).add(bc.max_prompt_length - 1)
    assert probed == expected_widths_by_batch_size
