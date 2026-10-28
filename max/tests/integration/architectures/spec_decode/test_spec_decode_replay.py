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
"""Checks the optional replay graph of SpecDecodePipelineModel."""

from __future__ import annotations

from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, Module, TensorType
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    SpecDecodePipelineModel,
    _BatchProcessor,
    _Inputs,
)


def _model(
    replay: bool, settled: bool
) -> tuple[SpecDecodePipelineModel, Mock, Mock]:
    model = object.__new__(SpecDecodePipelineModel)
    full, replay_graph = Mock(), Mock()
    for graph in (full, replay_graph):
        graph.execute.return_value = [Mock(), Mock(), Mock()]
    model.model = full
    model.replay_model = replay_graph if replay else None
    batching = Mock(spec=_BatchProcessor)
    batching.settle_replay.return_value = settled
    model._batch_processor = batching
    return model, full, batching


@pytest.fixture(autouse=True)
def _no_graph_buffers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_Inputs, "buffers", property(lambda _: ()))


def _inputs(*replay_trailing: Buffer) -> _Inputs:
    inputs = object.__new__(_Inputs)
    inputs.replay_trailing = list(replay_trailing)
    return inputs


def test_without_a_replay_graph_execute_never_consults_the_batching() -> None:
    model, full, batching = _model(replay=False, settled=True)
    inputs = _inputs()
    model.execute(inputs)
    full.execute.assert_called_once_with()
    batching.settle_replay.assert_not_called()
    batching.record_submission.assert_not_called()


@pytest.mark.parametrize("settled", [False, True])
def test_the_batching_selects_the_graph(settled: bool) -> None:
    model, full, batching = _model(replay=True, settled=settled)
    inputs = _inputs()
    model.execute(inputs)
    assert model.replay_model is not None
    replay_graph = cast(Mock, model.replay_model)
    selected, unused = (replay_graph, full) if settled else (full, replay_graph)
    selected.execute.assert_called_once_with()
    unused.execute.assert_not_called()
    batching.record_submission.assert_called_once_with(inputs)


def test_replay_inputs_without_a_replay_graph_are_an_error() -> None:
    model, full, _ = _model(replay=False, settled=False)
    inputs = _inputs(cast(Buffer, Mock()))
    with pytest.raises(RuntimeError, match="no replay graph"):
        model.execute(inputs)
    full.execute.assert_not_called()


def test_replay_graph_compiles_with_the_full_graph_and_shares_weights() -> None:
    cpu = DeviceRef.CPU()
    module = Module()
    graphs = []
    for name, delta in (("full", 1), ("full_replay", 2)):
        with Graph(
            name,
            input_types=[TensorType(DType.float32, [2], cpu)],
            module=module,
        ) as graph:
            graph.output(graph.inputs[0].tensor + delta)
        graphs.append(graph)
    model = object.__new__(SpecDecodePipelineModel)
    model._weight_loads = []
    full, replay = model._compile_with_replay(
        InferenceSession(devices=[CPU()]), graphs[0], graphs[1], {}
    )
    assert replay is not None
    x = Buffer.from_numpy(np.array([10, 20], dtype=np.float32))
    np.testing.assert_array_equal(full.execute(x)[0].to_numpy(), [11, 21])
    np.testing.assert_array_equal(replay.execute(x)[0].to_numpy(), [12, 22])
