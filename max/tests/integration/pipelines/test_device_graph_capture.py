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
"""Tests for PipelineModel device graph capture plumbing."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import Model
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheInputs, KVCacheInputsPerDevice
from max.pipelines.lib import ModelInputs, ModelOutputs
from max.pipelines.lib.graph_capture import ServeGraphCaptureRunner
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


class CapturePipelineModel(MockPipelineModel):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.input_buffer = Buffer.zeros((4,), dtype=DType.float32)
        self.output_buffer = Buffer.zeros((4,), dtype=DType.float32)
        self.model = DummyModel(self.output_buffer)

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        return ModelOutputs(logits=self.output_buffer)


def _make_runner(
    model: CapturePipelineModel, max_batch_size: int
) -> ServeGraphCaptureRunner:
    kv_params = MagicMock()
    kv_params.devices = [DeviceRef.CPU()]
    kv_params.n_kv_heads_per_device = 1
    kv_params.is_mla = False
    kv_params.data_parallel_degree = 1
    return ServeGraphCaptureRunner(
        model=cast(Model, model.model),
        execute_model=model.execute,
        session=MagicMock(),
        kv_params=kv_params,
        warmup_model_inputs=MagicMock(),
        max_cache_length_upper_bound=1,
        max_batch_size=max_batch_size,
    )


@pytest.fixture
def _mock_graph_key() -> Any:
    """Patches ``_resolve_replay_key``.

    Used by tests that exercise replay/capture plumbing without real KV cache
    inputs.
    """
    with patch.object(
        ServeGraphCaptureRunner,
        "_resolve_replay_key",
        side_effect=lambda mi: (int(mi.buffers[0].shape[0]), 1, 1, 0),
    ):
        yield


@pytest.fixture
@mock_huggingface_config
def capture_model() -> CapturePipelineModel:
    pipeline_config = DummyPipelineConfig(
        model_path="test/model",
        quantization_encoding=MagicMock(),
        max_batch_size=4,
        max_length=128,
    )
    pipeline_config.runtime.device_graph_capture = True
    return CapturePipelineModel(
        pipeline_config=pipeline_config,
        session=MagicMock(),
        devices=[CPU()],
        kv_cache_config=MagicMock(),
        weights=MagicMock(),
        adapter=None,
        return_logits=MagicMock(),
    )


@pytest.mark.usefixtures("_mock_graph_key")
def test_pipeline_model_capture_replay(
    capture_model: CapturePipelineModel,
) -> None:
    inputs = MockModelInputs(active_batch_size=1, eos_prob=0.0)
    runner = _make_runner(capture_model, max_batch_size=1)

    trace_inputs = inputs.buffers
    runner.graph_entries[(1, 1, 1, 0)] = (
        trace_inputs,
        ModelOutputs(*capture_model.model.capture(1, *trace_inputs)),
    )
    assert capture_model.model.capture_calls

    output = runner.replay(model_inputs=inputs)
    assert capture_model.model.replay_calls
    assert output.logits is capture_model.output_buffer


@pytest.mark.usefixtures("_mock_graph_key")
def test_pipeline_model_replay_miss_raises(
    capture_model: CapturePipelineModel,
) -> None:
    inputs = MockModelInputs(active_batch_size=4, eos_prob=0.0)
    runner = _make_runner(capture_model, max_batch_size=1)
    # No graph entry for batch_size=4 — replay raises KeyError.
    with pytest.raises(KeyError):
        runner.replay(model_inputs=inputs)

    assert not capture_model.model.capture_calls
    assert not capture_model.model.replay_calls


@pytest.mark.usefixtures("_mock_graph_key")
def test_pipeline_model_debug_verify_uses_runtime_inputs(
    capture_model: CapturePipelineModel,
) -> None:
    replay_inputs = MockModelInputs(active_batch_size=1, eos_prob=0.0)
    debug_inputs = MockModelInputs(active_batch_size=1, eos_prob=0.0)
    runner = _make_runner(capture_model, max_batch_size=1)

    trace_inputs = replay_inputs.buffers
    runner.graph_entries[(1, 1, 1, 0)] = (
        trace_inputs,
        ModelOutputs(*capture_model.model.capture(1, *trace_inputs)),
    )

    runner.replay(
        model_inputs=replay_inputs,
        debug_verify_replay=True,
        debug_verify_model_inputs=debug_inputs,
    )

    assert capture_model.model.debug_verify_replay_calls
    _, verified_buffers = capture_model.model.debug_verify_replay_calls[-1]
    assert verified_buffers == list(debug_inputs.buffers)


@pytest.mark.usefixtures("_mock_graph_key")
def test_pipeline_model_debug_verify_rejects_mismatched_graph_keys(
    capture_model: CapturePipelineModel,
) -> None:
    replay_inputs = MockModelInputs(active_batch_size=1, eos_prob=0.0)
    debug_inputs = MockModelInputs(active_batch_size=2, eos_prob=0.0)
    runner = _make_runner(capture_model, max_batch_size=1)

    trace_inputs = replay_inputs.buffers
    runner.graph_entries[(1, 1, 1, 0)] = (
        trace_inputs,
        ModelOutputs(*capture_model.model.capture(1, *trace_inputs)),
    )

    with pytest.raises(
        ValueError,
        match=r"debug_verify_model_inputs must map to the same graph key",
    ):
        runner.replay(
            model_inputs=replay_inputs,
            debug_verify_replay=True,
            debug_verify_model_inputs=debug_inputs,
        )

    assert not capture_model.model.debug_verify_replay_calls


# ---------------------------------------------------------------------------
# Helpers and tests for _resolve_replay_key and eager fallback
# ---------------------------------------------------------------------------


def _make_kv_per_device(
    max_cache_len: int,
    num_partitions: int,
    q_max_seq_len: int = 1,
    *,
    is_mla: bool = False,
    draft_num_partitions: int | None = None,
    draft_q_max_seq_len: int = 1,
) -> KVCacheInputsPerDevice[Buffer, Buffer]:
    """Creates a ``KVCacheInputsPerDevice`` with configurable metadata."""
    max_lengths = np.array([[0, max_cache_len]], dtype=np.uint32)
    # attention_dispatch_metadata layout:
    # MLA: [batch_size, q_max_seq_len, num_partitions]
    # MHA: [batch_size, q_max_seq_len, num_partitions, max_cache_valid_length]
    dispatch = np.array(
        [1, q_max_seq_len, num_partitions]
        + ([] if is_mla else [max_cache_len]),
        dtype=np.int64,
    )
    draft_metadata: Buffer | None = None
    if draft_num_partitions is not None:
        draft_dispatch = np.array(
            [1, draft_q_max_seq_len, draft_num_partitions]
            + ([] if is_mla else [max_cache_len]),
            dtype=np.int64,
        )
        draft_metadata = Buffer.from_numpy(draft_dispatch)
    return KVCacheInputsPerDevice(
        kv_blocks=Buffer.zeros((1,), dtype=DType.float32),
        cache_lengths=Buffer.from_numpy(np.array([0], dtype=np.uint32)),
        lookup_table=Buffer.from_numpy(np.array([0], dtype=np.uint32)),
        max_lengths=Buffer.from_numpy(max_lengths),
        attention_dispatch_metadata=Buffer.from_numpy(dispatch),
        draft_attention_dispatch_metadata=draft_metadata,
    )


def _make_mock_inputs_with_kv(
    active_batch_size: int,
    kv_per_device_list: list[KVCacheInputsPerDevice[Buffer, Buffer]],
) -> MockModelInputs:
    """Creates ``MockModelInputs`` with proper ``KVCacheInputs``."""
    return MockModelInputs(
        active_batch_size=active_batch_size,
        eos_prob=0.0,
        kv_cache_inputs=KVCacheInputs(inputs=kv_per_device_list),
    )


def _make_runner_for_resolve(
    *,
    data_parallel_degree: int = 1,
    is_mla: bool = False,
    num_speculative_tokens: int = 0,
    draft_is_mla: bool | None = None,
) -> ServeGraphCaptureRunner:
    """Creates a runner suitable for ``_resolve_replay_key`` tests."""
    output_buffer = Buffer.zeros((4,), dtype=DType.float32)
    dummy_model = DummyModel(output_buffer)

    kv_params = MagicMock()
    kv_params.devices = [DeviceRef.CPU()]
    kv_params.n_kv_heads_per_device = 1
    kv_params.is_mla = is_mla
    kv_params.data_parallel_degree = data_parallel_degree

    draft_kv_params: MagicMock | None = None
    if draft_is_mla is not None:
        draft_kv_params = MagicMock()
        draft_kv_params.devices = [DeviceRef.CPU()]
        draft_kv_params.n_kv_heads_per_device = 1
        draft_kv_params.is_mla = draft_is_mla
        draft_kv_params.data_parallel_degree = data_parallel_degree

    return ServeGraphCaptureRunner(
        model=cast(Model, dummy_model),
        execute_model=lambda mi: ModelOutputs(logits=output_buffer),
        session=MagicMock(),
        kv_params=kv_params,
        warmup_model_inputs=MagicMock(),
        max_cache_length_upper_bound=1,
        max_batch_size=1,
        num_speculative_tokens=num_speculative_tokens,
        draft_kv_params=draft_kv_params,
    )


def test_resolve_replay_key_single_device() -> None:
    runner = _make_runner_for_resolve()
    dummy_buf = Buffer.zeros((1,), dtype=DType.float32)
    runner.graph_entries[(1, 5, 1, 0)] = ((), ModelOutputs(logits=dummy_buf))

    kv = _make_kv_per_device(max_cache_len=100, num_partitions=5)
    inputs = _make_mock_inputs_with_kv(1, [kv])
    assert runner._resolve_replay_key(inputs) == (1, 5, 1, 0)


def test_resolve_replay_key_dp_syncs_metadata() -> None:
    runner = _make_runner_for_resolve(data_parallel_degree=2)
    dummy_buf = Buffer.zeros((4,), dtype=DType.float32)
    runner.graph_entries[(1, 7, 1, 0)] = ((), ModelOutputs(logits=dummy_buf))

    kv0 = _make_kv_per_device(max_cache_len=50, num_partitions=3)
    kv1 = _make_kv_per_device(max_cache_len=100, num_partitions=7)
    inputs = _make_mock_inputs_with_kv(1, [kv0, kv1])

    key = runner._resolve_replay_key(inputs)

    # np should be max(3, 7) = 7
    assert key == (1, 7, 1, 0)


def test_resolve_replay_key_mha_miss_raises() -> None:
    runner = _make_runner_for_resolve()
    # No graph entries — MHA has no bucketing, so miss raises.
    kv = _make_kv_per_device(max_cache_len=100, num_partitions=99)
    inputs = _make_mock_inputs_with_kv(1, [kv])
    with pytest.raises(RuntimeError, match=r"No captured device graph for"):
        runner._resolve_replay_key(inputs)


def test_resolve_replay_key_mla_buckets_up() -> None:
    runner = _make_runner_for_resolve(is_mla=True)
    output_buf = Buffer.zeros((4,), dtype=DType.float32)
    # Capture graphs for np=5 and np=10.
    runner.graph_entries[(1, 5, 1, 0)] = ((), ModelOutputs(logits=output_buf))
    runner.graph_entries[(1, 10, 1, 0)] = ((), ModelOutputs(logits=output_buf))

    # Runtime np=7 should bucket up to 10.
    kv = _make_kv_per_device(max_cache_len=100, num_partitions=7, is_mla=True)
    inputs = _make_mock_inputs_with_kv(1, [kv])
    assert runner._resolve_replay_key(inputs) == (1, 10, 1, 0)


def test_replay_mla_buckets_to_captured_graph() -> None:
    runner = _make_runner_for_resolve(is_mla=True)
    output_buf = Buffer.zeros((4,), dtype=DType.float32)
    model = cast(DummyModel, runner._model)

    # Capture a graph for np=10.
    kv_capture = _make_kv_per_device(
        max_cache_len=100, num_partitions=10, is_mla=True
    )
    capture_inputs = _make_mock_inputs_with_kv(1, [kv_capture])
    runner.graph_entries[(1, 10, 1, 0)] = (
        capture_inputs.buffers,
        ModelOutputs(logits=output_buf),
    )

    # Replay with np=7 → buckets to 10 → replays captured graph.
    kv_replay = _make_kv_per_device(
        max_cache_len=100, num_partitions=7, is_mla=True
    )
    replay_inputs = _make_mock_inputs_with_kv(1, [kv_replay])
    output = runner.replay(model_inputs=replay_inputs)
    assert model.replay_calls
    assert output.logits is output_buf


def test_resolve_replay_key_mla_miss_raises() -> None:
    runner = _make_runner_for_resolve(is_mla=True)
    output_buf = Buffer.zeros((4,), dtype=DType.float32)
    # Only np=5 captured — runtime np=99 has no bucket >= 99.
    runner.graph_entries[(1, 5, 1, 0)] = ((), ModelOutputs(logits=output_buf))

    kv = _make_kv_per_device(max_cache_len=200, num_partitions=99, is_mla=True)
    inputs = _make_mock_inputs_with_kv(1, [kv])
    with pytest.raises(RuntimeError, match=r"No captured device graph for"):
        runner._resolve_replay_key(inputs)


def test_replay_mha_miss_raises() -> None:
    runner = _make_runner_for_resolve(is_mla=False)
    output_buf = Buffer.zeros((4,), dtype=DType.float32)
    kv_capture = _make_kv_per_device(max_cache_len=100, num_partitions=5)
    capture_inputs = _make_mock_inputs_with_kv(1, [kv_capture])
    runner.graph_entries[(1, 5, 1, 0)] = (
        capture_inputs.buffers,
        ModelOutputs(logits=output_buf),
    )

    kv_miss = _make_kv_per_device(max_cache_len=200, num_partitions=99)
    miss_inputs = _make_mock_inputs_with_kv(1, [kv_miss])

    # MHA captures all modes; a miss is a real error.
    with pytest.raises(
        RuntimeError,
        match=r"No captured device graph for",
    ):
        runner.replay(model_inputs=miss_inputs)


def test_resolve_replay_key_q_seq_len_skip() -> None:
    runner = _make_runner_for_resolve()
    dummy_buf = Buffer.zeros((1,), dtype=DType.float32)
    runner.graph_entries[(1, 5, 1, 0)] = ((), ModelOutputs(logits=dummy_buf))

    # q_max_seq_len=2 (anything != 1) raises RuntimeError.
    kv = _make_kv_per_device(
        max_cache_len=100, num_partitions=5, q_max_seq_len=2
    )
    inputs = _make_mock_inputs_with_kv(1, [kv])
    with pytest.raises(RuntimeError, match=r"q_max_seq_len=2 != 1"):
        runner._resolve_replay_key(inputs)


def test_resolve_replay_key_q_seq_len_only_1_captured() -> None:
    runner = _make_runner_for_resolve()
    dummy_buf = Buffer.zeros((1,), dtype=DType.float32)
    runner.graph_entries[(1, 5, 1, 0)] = ((), ModelOutputs(logits=dummy_buf))

    # q_max_seq_len=1 resolves successfully.
    kv = _make_kv_per_device(
        max_cache_len=100, num_partitions=5, q_max_seq_len=1
    )
    inputs = _make_mock_inputs_with_kv(1, [kv])
    assert runner._resolve_replay_key(inputs) == (1, 5, 1, 0)

    # q_max_seq_len > 1 raises RuntimeError.
    for q in (2, 3, 4, 5):
        kv = _make_kv_per_device(
            max_cache_len=100, num_partitions=5, q_max_seq_len=q
        )
        inputs = _make_mock_inputs_with_kv(1, [kv])
        with pytest.raises(RuntimeError, match=rf"q_max_seq_len={q} != 1"):
            runner._resolve_replay_key(inputs)


# ---------------------------------------------------------------------------
# Tests for speculative token support in graph capture
# ---------------------------------------------------------------------------


def test_resolve_replay_key_multi_spec_tokens() -> None:
    """Eagle with num_speculative_tokens>1 uses q_max_seq_len=1+N and draft key."""
    num_spec = 3
    runner = _make_runner_for_resolve(num_speculative_tokens=num_spec)
    dummy_buf = Buffer.zeros((1,), dtype=DType.float32)
    expected_q = 1 + num_spec
    draft_np = 11
    runner.graph_entries[(1, 5, expected_q, draft_np)] = (
        (),
        ModelOutputs(logits=dummy_buf),
    )

    kv_ok = _make_kv_per_device(
        max_cache_len=100,
        num_partitions=5,
        q_max_seq_len=expected_q,
        draft_num_partitions=draft_np,
    )
    inputs_ok = _make_mock_inputs_with_kv(1, [kv_ok])
    assert runner._resolve_replay_key(inputs_ok) == (1, 5, expected_q, draft_np)

    kv_bad_q = _make_kv_per_device(
        max_cache_len=100,
        num_partitions=5,
        q_max_seq_len=2,
        draft_num_partitions=draft_np,
    )
    inputs_bad_q = _make_mock_inputs_with_kv(1, [kv_bad_q])
    with pytest.raises(RuntimeError, match=r"q_max_seq_len=2 != 1"):
        runner._resolve_replay_key(inputs_bad_q)


def test_resolve_replay_key_with_spec_tokens() -> None:
    """With num_speculative_tokens=1, q_max_seq_len=2 resolves; =1 raises."""
    runner = _make_runner_for_resolve(num_speculative_tokens=1)
    dummy_buf = Buffer.zeros((1,), dtype=DType.float32)
    expected_q = 2  # 1 + num_speculative_tokens
    draft_np = 5
    runner.graph_entries[(1, 5, expected_q, draft_np)] = (
        (),
        ModelOutputs(logits=dummy_buf),
    )

    kv_ok = _make_kv_per_device(
        max_cache_len=100,
        num_partitions=5,
        q_max_seq_len=expected_q,
        draft_num_partitions=draft_np,
    )
    inputs_ok = _make_mock_inputs_with_kv(1, [kv_ok])
    assert runner._resolve_replay_key(inputs_ok) == (1, 5, expected_q, draft_np)

    kv_bad = _make_kv_per_device(
        max_cache_len=100,
        num_partitions=5,
        q_max_seq_len=1,
        draft_num_partitions=draft_np,
    )
    inputs_bad = _make_mock_inputs_with_kv(1, [kv_bad])
    with pytest.raises(RuntimeError, match=r"q_max_seq_len=1 != 1"):
        runner._resolve_replay_key(inputs_bad)


def test_resolve_replay_key_dp_syncs_draft_metadata() -> None:
    """DP replay syncs draft_num_partitions across shards (max + broadcast)."""
    runner = _make_runner_for_resolve(
        data_parallel_degree=2,
        num_speculative_tokens=1,
    )
    dummy_buf = Buffer.zeros((4,), dtype=DType.float32)
    expected_q = 2
    draft_hi = 7
    draft_lo = 3
    runner.graph_entries[(1, 5, expected_q, draft_hi)] = (
        (),
        ModelOutputs(logits=dummy_buf),
    )

    kv0 = _make_kv_per_device(
        max_cache_len=50,
        num_partitions=5,
        q_max_seq_len=expected_q,
        draft_num_partitions=draft_lo,
    )
    kv1 = _make_kv_per_device(
        max_cache_len=100,
        num_partitions=5,
        q_max_seq_len=expected_q,
        draft_num_partitions=draft_hi,
    )
    inputs = _make_mock_inputs_with_kv(1, [kv0, kv1])

    key = runner._resolve_replay_key(inputs)
    assert key == (1, 5, expected_q, draft_hi)

    assert kv0.draft_attention_dispatch_metadata is not None
    assert kv1.draft_attention_dispatch_metadata is not None
    lo_np = kv0.draft_attention_dispatch_metadata.to_numpy()
    hi_np = kv1.draft_attention_dispatch_metadata.to_numpy()
    assert int(lo_np[2]) == draft_hi
    assert int(hi_np[2]) == draft_hi


def test_resolve_replay_key_mla_buckets_respects_draft_partitions() -> None:
    """MLA bucketing only considers captures with the same draft_num_partitions."""
    runner = _make_runner_for_resolve(is_mla=True, num_speculative_tokens=1)
    output_buf = Buffer.zeros((4,), dtype=DType.float32)
    expected_q = 2
    draft_np = 5
    runner.graph_entries[(1, 10, expected_q, draft_np)] = (
        (),
        ModelOutputs(logits=output_buf),
    )

    kv_match = _make_kv_per_device(
        max_cache_len=100,
        num_partitions=7,
        q_max_seq_len=expected_q,
        is_mla=True,
        draft_num_partitions=draft_np,
    )
    inputs_match = _make_mock_inputs_with_kv(1, [kv_match])
    assert runner._resolve_replay_key(inputs_match) == (
        1,
        10,
        expected_q,
        draft_np,
    )

    kv_wrong_draft = _make_kv_per_device(
        max_cache_len=100,
        num_partitions=7,
        q_max_seq_len=expected_q,
        is_mla=True,
        draft_num_partitions=99,
    )
    inputs_wrong_draft = _make_mock_inputs_with_kv(1, [kv_wrong_draft])
    with pytest.raises(RuntimeError, match=r"No captured device graph for"):
        runner._resolve_replay_key(inputs_wrong_draft)


def test_resolve_replay_key_zero_spec_tokens_backward_compat() -> None:
    """Default (num_speculative_tokens=0) keeps original q_max_seq_len=1 behavior."""
    runner = _make_runner_for_resolve(num_speculative_tokens=0)
    dummy_buf = Buffer.zeros((1,), dtype=DType.float32)
    runner.graph_entries[(1, 5, 1, 0)] = ((), ModelOutputs(logits=dummy_buf))

    kv = _make_kv_per_device(
        max_cache_len=100, num_partitions=5, q_max_seq_len=1
    )
    inputs = _make_mock_inputs_with_kv(1, [kv])
    assert runner._resolve_replay_key(inputs) == (1, 5, 1, 0)

    kv_bad = _make_kv_per_device(
        max_cache_len=100, num_partitions=5, q_max_seq_len=2
    )
    inputs_bad = _make_mock_inputs_with_kv(1, [kv_bad])
    with pytest.raises(RuntimeError, match=r"q_max_seq_len=2 != 1"):
        runner._resolve_replay_key(inputs_bad)


class EagleDummyModel(DummyModel):
    """DummyModel that returns 3 output buffers for Eagle capture."""

    def capture(self, graph_key: int, *buffers: Buffer) -> list[Buffer]:
        self.capture_calls.append((graph_key, list(buffers)))
        return [
            self.output_buffer,
            self.output_buffer,
            self.output_buffer,
        ]


def test_warmup_eagle_outputs_vs_model_outputs() -> None:
    """Warmup produces UnifiedEagleOutputs when spec tokens > 0, else ModelOutputs."""
    for num_spec, expected_type in [
        (1, UnifiedEagleOutputs),
        (0, ModelOutputs),
    ]:
        model: DummyModel
        if num_spec > 0:
            model = EagleDummyModel(Buffer.zeros((4,), dtype=DType.float32))
        else:
            model = DummyModel(Buffer.zeros((4,), dtype=DType.float32))

        kv_params = MagicMock()
        kv_params.devices = [DeviceRef.CPU()]
        kv_params.n_kv_heads_per_device = 1
        kv_params.is_mla = False
        kv_params.data_parallel_degree = 1
        kv_params.num_q_heads_per_device = 1
        kv_params.is_fp8_kv_dtype = False

        mock_inputs = MockModelInputs(
            active_batch_size=1,
            eos_prob=0.0,
            kv_cache_inputs=KVCacheInputs(
                inputs=[_make_kv_per_device(max_cache_len=10, num_partitions=1)]
            ),
        )

        from collections.abc import Iterator
        from contextlib import contextmanager

        @contextmanager
        def _warmup_ctx(
            batch_size: int,
            _mi: MockModelInputs = mock_inputs,
        ) -> Iterator[MockModelInputs]:
            yield _mi

        runner = ServeGraphCaptureRunner(
            model=cast(Model, model),
            execute_model=lambda mi: ModelOutputs(logits=mi.buffers[0]),
            session=MagicMock(),
            kv_params=kv_params,
            warmup_model_inputs=_warmup_ctx,
            max_cache_length_upper_bound=10,
            max_batch_size=1,
            num_speculative_tokens=num_spec,
        )
        runner.warmup_pre_ready()

        assert len(runner.graph_entries) > 0
        for key, (_inputs, outputs) in runner.graph_entries.items():
            assert isinstance(outputs, expected_type), (
                f"num_spec={num_spec}: expected {expected_type.__name__}, "
                f"got {type(outputs).__name__}"
            )
            assert key[2] == 1 + num_spec
            assert key[3] >= 0


# ---------------------------------------------------------------------------
# Tests for asymmetric target/draft attention types (e.g. MLA target with
# GQA/MHA drafter, or the reverse).
# ---------------------------------------------------------------------------


def test_runner_uses_separate_resolvers_for_asymmetric_attention() -> None:
    """draft_kv_params builds a distinct resolver/probe strategy."""
    runner = _make_runner_for_resolve(
        is_mla=True, draft_is_mla=False, num_speculative_tokens=1
    )
    assert runner._target_is_mla is True
    assert runner._draft_is_mla is False
    assert runner._target_resolver is not runner._draft_resolver
    assert runner._target_probe_strategy is not runner._draft_probe_strategy


def test_runner_falls_back_to_target_when_draft_kv_params_omitted() -> None:
    """Without ``draft_kv_params`` the draft fields alias the target fields."""
    runner = _make_runner_for_resolve(is_mla=True, num_speculative_tokens=1)
    assert runner._target_is_mla is True
    assert runner._draft_is_mla is True
    assert runner._target_resolver is runner._draft_resolver
    assert runner._target_probe_strategy is runner._draft_probe_strategy


def test_resolve_replay_key_mla_target_mha_draft_buckets_target_only() -> None:
    """MLA target buckets up; MHA draft requires an exact match."""
    runner = _make_runner_for_resolve(
        is_mla=True, draft_is_mla=False, num_speculative_tokens=1
    )
    output_buf = Buffer.zeros((4,), dtype=DType.float32)
    expected_q = 2
    draft_np = 7
    runner.graph_entries[(1, 10, expected_q, draft_np)] = (
        (),
        ModelOutputs(logits=output_buf),
    )

    kv_ok = _make_kv_per_device(
        max_cache_len=100,
        num_partitions=5,
        q_max_seq_len=expected_q,
        is_mla=True,
        draft_num_partitions=draft_np,
    )
    inputs_ok = _make_mock_inputs_with_kv(1, [kv_ok])
    assert runner._resolve_replay_key(inputs_ok) == (
        1,
        10,
        expected_q,
        draft_np,
    )

    kv_bad_draft = _make_kv_per_device(
        max_cache_len=100,
        num_partitions=5,
        q_max_seq_len=expected_q,
        is_mla=True,
        draft_num_partitions=99,
    )
    inputs_bad_draft = _make_mock_inputs_with_kv(1, [kv_bad_draft])
    with pytest.raises(RuntimeError, match=r"No captured device graph for"):
        runner._resolve_replay_key(inputs_bad_draft)


def test_resolve_replay_key_mha_target_mla_draft_buckets_draft_only() -> None:
    """MHA target requires an exact match; MLA draft buckets up."""
    runner = _make_runner_for_resolve(
        is_mla=False, draft_is_mla=True, num_speculative_tokens=1
    )
    output_buf = Buffer.zeros((4,), dtype=DType.float32)
    expected_q = 2
    target_np = 5
    runner.graph_entries[(1, target_np, expected_q, 10)] = (
        (),
        ModelOutputs(logits=output_buf),
    )

    kv_ok = _make_kv_per_device(
        max_cache_len=100,
        num_partitions=target_np,
        q_max_seq_len=expected_q,
        draft_num_partitions=7,
    )
    inputs_ok = _make_mock_inputs_with_kv(1, [kv_ok])
    assert runner._resolve_replay_key(inputs_ok) == (
        1,
        target_np,
        expected_q,
        10,
    )

    kv_bad_target = _make_kv_per_device(
        max_cache_len=100,
        num_partitions=99,
        q_max_seq_len=expected_q,
        draft_num_partitions=7,
    )
    inputs_bad_target = _make_mock_inputs_with_kv(1, [kv_bad_target])
    with pytest.raises(RuntimeError, match=r"No captured device graph for"):
        runner._resolve_replay_key(inputs_bad_target)


def test_resolve_replay_key_mla_target_mla_draft_buckets_both_axes() -> None:
    """Both axes MLA: each is independently bucketed up."""
    runner = _make_runner_for_resolve(
        is_mla=True, draft_is_mla=True, num_speculative_tokens=1
    )
    output_buf = Buffer.zeros((4,), dtype=DType.float32)
    expected_q = 2
    runner.graph_entries[(1, 10, expected_q, 10)] = (
        (),
        ModelOutputs(logits=output_buf),
    )

    kv = _make_kv_per_device(
        max_cache_len=100,
        num_partitions=7,
        q_max_seq_len=expected_q,
        is_mla=True,
        draft_num_partitions=7,
    )
    inputs = _make_mock_inputs_with_kv(1, [kv])
    assert runner._resolve_replay_key(inputs) == (1, 10, expected_q, 10)


def test_replay_returns_correct_output_type() -> None:
    """replay() returns UnifiedEagleOutputs when num_speculative_tokens > 0."""
    runner = _make_runner_for_resolve(num_speculative_tokens=1)
    expected_q = 2
    buf = Buffer.zeros((4,), dtype=DType.float32)
    eagle_outputs = UnifiedEagleOutputs(
        num_accepted_draft_tokens=buf,
        next_tokens=buf,
        next_draft_tokens=buf,
    )

    draft_np = 5
    kv = _make_kv_per_device(
        max_cache_len=100,
        num_partitions=5,
        q_max_seq_len=expected_q,
        draft_num_partitions=draft_np,
    )
    capture_inputs = _make_mock_inputs_with_kv(1, [kv])
    runner.graph_entries[(1, 5, expected_q, draft_np)] = (
        capture_inputs.buffers,
        eagle_outputs,
    )

    replay_kv = _make_kv_per_device(
        max_cache_len=100,
        num_partitions=5,
        q_max_seq_len=expected_q,
        draft_num_partitions=draft_np,
    )
    replay_inputs = _make_mock_inputs_with_kv(1, [replay_kv])
    output = runner.replay(model_inputs=replay_inputs)

    assert isinstance(output, UnifiedEagleOutputs)
    assert output.num_accepted_draft_tokens is buf
