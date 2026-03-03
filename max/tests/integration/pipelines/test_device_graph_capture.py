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

import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import Model
from max.graph import DeviceRef
from max.pipelines.lib import ModelInputs, ModelOutputs
from max.pipelines.lib.graph_capture import ServeGraphCaptureRunner
from test_common.mocks.pipeline_config import (
    DummyPipelineConfig,
    mock_huggingface_config,
)
from test_common.mocks.pipeline_model import (
    MockModelInputs,
    MockPipelineModel,
)


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
    return ServeGraphCaptureRunner(
        model=cast(Model, model.model),
        execute_model=model.execute,
        session=MagicMock(),
        kv_params=kv_params,
        warmup_model_inputs=MagicMock(),
        max_cache_length_upper_bound=1,
        max_batch_size=max_batch_size,
    )


@pytest.fixture(autouse=True)
def _mock_graph_key() -> Any:
    with patch(
        "max.pipelines.lib.graph_capture.mha_graph_key_from_inputs",
        side_effect=lambda mi: (int(mi.buffers[0].shape[0]), 1),
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


def test_pipeline_model_capture_replay(
    capture_model: CapturePipelineModel,
) -> None:
    inputs = MockModelInputs(active_batch_size=1, eos_prob=0.0)
    runner = _make_runner(capture_model, max_batch_size=1)

    trace_inputs = inputs.buffers
    runner.graph_entries[(1, 1)] = (
        trace_inputs,
        ModelOutputs(*capture_model.model.capture(1, *trace_inputs)),
    )
    assert capture_model.model.capture_calls

    output = runner.replay(model_inputs=inputs)
    assert capture_model.model.replay_calls
    assert output.logits is capture_model.output_buffer


def test_pipeline_model_replay_miss_raises(
    capture_model: CapturePipelineModel,
) -> None:
    inputs = MockModelInputs(active_batch_size=4, eos_prob=0.0)
    runner = _make_runner(capture_model, max_batch_size=1)
    with pytest.raises(
        RuntimeError,
        match=r"No captured device graph found for key:",
    ):
        runner.replay(model_inputs=inputs)

    assert not capture_model.model.capture_calls
    assert not capture_model.model.replay_calls


def test_pipeline_model_debug_verify_uses_runtime_inputs(
    capture_model: CapturePipelineModel,
) -> None:
    replay_inputs = MockModelInputs(active_batch_size=1, eos_prob=0.0)
    debug_inputs = MockModelInputs(active_batch_size=1, eos_prob=0.0)
    runner = _make_runner(capture_model, max_batch_size=1)

    trace_inputs = replay_inputs.buffers
    runner.graph_entries[(1, 1)] = (
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


def test_pipeline_model_debug_verify_rejects_mismatched_graph_keys(
    capture_model: CapturePipelineModel,
) -> None:
    replay_inputs = MockModelInputs(active_batch_size=1, eos_prob=0.0)
    debug_inputs = MockModelInputs(active_batch_size=2, eos_prob=0.0)
    runner = _make_runner(capture_model, max_batch_size=1)

    trace_inputs = replay_inputs.buffers
    runner.graph_entries[(1, 1)] = (
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
