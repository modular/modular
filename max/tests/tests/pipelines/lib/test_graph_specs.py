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
"""Tests for compiling a pipeline's graphs by role."""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, TensorType, TensorValue, Weight
from max.nn import Module
from max.pipelines.lib import (
    CompiledArtifacts,
    GraphRole,
    GraphSpec,
    compile_graphs,
    init_graphs,
)

X = TensorType(DType.float32, [2], device=DeviceRef.CPU())


class Scale(Module):
    """Multiplies its input by a weight."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.weight = Weight(name, DType.float32, [2], DeviceRef.CPU())

    def __call__(self, x: TensorValue) -> TensorValue:
        return x * self.weight


def _spec(name: str, value: float, group: str | None = None) -> GraphSpec:
    return GraphSpec(
        Scale(f"{name}_weight"),
        name,
        [X],
        weights={f"{name}_weight": np.full(2, value, dtype=np.float32)},
        group=group,
    )


SPECS: dict[str, GraphSpec] = {
    GraphRole.DECODER: _spec("language", 2.0, group="model"),
    GraphRole.VISION: _spec("vision", 3.0, group="model"),
    GraphRole.SAMPLER: _spec("sampler", 5.0),
}


@pytest.fixture(scope="module")
def session() -> InferenceSession:
    return InferenceSession(devices=[CPU()])


@pytest.fixture(scope="module")
def artifacts(session: InferenceSession) -> CompiledArtifacts:
    return compile_graphs(session, SPECS)


def _run(model: Model) -> list[float]:
    (out,) = model.execute(Buffer.from_numpy(np.ones(2, dtype=np.float32)))
    assert isinstance(out, Buffer)
    return out.to_numpy().tolist()


def test_a_group_compiles_into_one_artifact(
    artifacts: CompiledArtifacts,
) -> None:
    assert len(artifacts.compiled) == 2
    decoder = artifacts.artifact(GraphRole.DECODER)
    assert artifacts.artifact(GraphRole.VISION) is decoder
    assert artifacts.artifact(GraphRole.SAMPLER) is not decoder


def test_unknown_role_raises(artifacts: CompiledArtifacts) -> None:
    assert GraphRole.EP_INIT not in artifacts
    with pytest.raises(KeyError, match="ep_init"):
        artifacts.artifact(GraphRole.EP_INIT)


def test_init_runs_each_role_with_its_weights(
    session: InferenceSession, artifacts: CompiledArtifacts
) -> None:
    models = init_graphs(session, artifacts, SPECS)
    assert _run(models[GraphRole.DECODER]) == [2.0, 2.0]
    assert _run(models[GraphRole.VISION]) == [3.0, 3.0]
    assert _run(models[GraphRole.SAMPLER]) == [5.0, 5.0]


def test_weights_default_to_the_state_dict(session: InferenceSession) -> None:
    module = Scale("weight")
    module.load_state_dict({"weight": np.full(2, 7.0, dtype=np.float32)})
    specs = {"only": GraphSpec(module, "s", [X])}
    models = init_graphs(session, compile_graphs(session, specs), specs)
    assert _run(models["only"]) == [7.0, 7.0]


def test_a_group_needs_distinct_graph_names(
    session: InferenceSession,
) -> None:
    with pytest.raises(ValueError, match="distinct graph names"):
        compile_graphs(
            session,
            {
                "a": _spec("same", 1.0, group="g"),
                "b": _spec("same", 1.0, group="g"),
            },
        )


def test_separate_artifacts_may_share_graph_names(
    session: InferenceSession,
) -> None:
    specs = {"a": _spec("same", 1.0), "b": _spec("same", 1.0)}
    models = init_graphs(session, compile_graphs(session, specs), specs)
    assert _run(models["a"]) == [1.0, 1.0]
    assert _run(models["b"]) == [1.0, 1.0]


def test_merging_rejects_a_role_registered_twice(
    artifacts: CompiledArtifacts,
) -> None:
    with pytest.raises(ValueError, match="registered twice"):
        _ = artifacts | artifacts


def test_merging_keeps_each_role_s_artifact(
    session: InferenceSession, artifacts: CompiledArtifacts
) -> None:
    other = compile_graphs(session, {"draft": _spec("draft", 1.0)})
    merged = artifacts | other
    assert len(merged.compiled) == 3
    assert merged.artifact("draft") is other.artifact("draft")
    assert merged.artifact(GraphRole.SAMPLER) is artifacts.artifact(
        GraphRole.SAMPLER
    )
