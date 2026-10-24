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

import json
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pytest
from max._artifact.value_types import ArtifactError, treedef_to_json
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.experimental.compilation import CompiledCallable
from max.experimental.nn import Module as ModuleV3
from max.experimental.nn import module_dataclass
from max.experimental.sharding.types import as_layout
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorType, TensorValue, Weight
from max.nn import Module
from max.nn.graph_groups import HiddenStates, Logits, RaggedBatch
from max.nn.transformer import DecoderOutputs
from max.pipelines.lib import GraphRole, GraphSpec, compile_and_init_graphs

X = TensorType(DType.float32, [2], device=DeviceRef.CPU())


class Scale(Module):
    """Multiplies its input by a weight."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.weight = Weight(name, DType.float32, [2], DeviceRef.CPU())

    def __call__(self, x: TensorValue) -> TensorValue:
        return x * self.weight


def _logits(x: TensorValue) -> DecoderOutputs:
    return DecoderOutputs(logits=Logits(next_token_logits=x))


class Unkinded(OrderedDict[str, object]):
    """A tree node no exported graph can name."""

    def __tree_flatten__(self) -> tuple[Mapping[str, object], None]:
        return self, None

    @classmethod
    def __tree_unflatten__(
        cls, meta: None, children: Mapping[str, object]
    ) -> Unkinded:
        return cls(children)


class Outputs(Module):
    """Traces ``wrap(scale(x))``, for a graph whose outputs are a tree."""

    def __init__(
        self, scale: Scale, wrap: Callable[[TensorValue], object]
    ) -> None:
        super().__init__()
        self.scale = scale
        self.wrap = wrap

    def __call__(self, x: TensorValue) -> object:
        return self.wrap(self.scale(x))


@module_dataclass
class ScaleV3(ModuleV3[[Tensor], Tensor]):
    """Multiplies its input by a weight, as a ModuleV3 module."""

    weight: Tensor

    def forward(self, x: Tensor) -> Tensor:
        return x * self.weight


def _v3_spec(name: str, value: float, group: str | None = None) -> GraphSpec:
    return GraphSpec(
        ScaleV3(Tensor.zeros([2], dtype=DType.float32, device=CPU())),
        name,
        [X],
        weights={"weight": np.full(2, value, dtype=np.float32)},
        group=group,
    )


def _spec(
    name: str,
    value: float,
    group: str | None = None,
    wrap: Callable[[TensorValue], object] | None = None,
) -> GraphSpec:
    scale = Scale(f"{name}_weight")
    return GraphSpec(
        scale if wrap is None else Outputs(scale, wrap),
        name,
        [X],
        weights={f"{name}_weight": np.full(2, value, dtype=np.float32)},
        group=group,
    )


SPECS: dict[str, GraphSpec] = {
    GraphRole.DECODER: _spec("language", 2.0, group="model", wrap=_logits),
    GraphRole.VISION: _spec("vision", 3.0, group="model"),
    GraphRole.SAMPLER: _spec("sampler", 5.0),
}


@pytest.fixture(scope="module")
def session() -> InferenceSession:
    return InferenceSession(devices=[CPU()])


@pytest.fixture(scope="module")
def graphs(session: InferenceSession) -> dict[str, CompiledCallable[..., Any]]:
    return compile_and_init_graphs(session, SPECS)


def _run(model: Model) -> list[float]:
    (out,) = model.execute(Buffer.from_numpy(np.ones(2, dtype=np.float32)))
    assert isinstance(out, Buffer)
    return out.to_numpy().tolist()


def test_a_group_shares_one_compiled_model(
    graphs: dict[str, CompiledCallable[..., Any]],
) -> None:
    assert set(graphs) == set(SPECS)
    decoder = graphs[GraphRole.DECODER]
    vision = graphs[GraphRole.VISION]
    sampler = graphs[GraphRole.SAMPLER]
    assert vision._artifact is decoder._artifact
    assert sampler._artifact is not decoder._artifact
    assert vision.engine_model is not decoder.engine_model


def test_each_role_runs_with_its_weights(
    graphs: dict[str, CompiledCallable[..., Any]],
) -> None:
    assert _run(graphs[GraphRole.DECODER].engine_model) == [2.0, 2.0]
    assert _run(graphs[GraphRole.VISION].engine_model) == [3.0, 3.0]
    assert _run(graphs[GraphRole.SAMPLER].engine_model) == [5.0, 5.0]


def test_a_role_is_callable_on_tensors(
    graphs: dict[str, CompiledCallable[..., Any]],
) -> None:
    sampler = graphs[GraphRole.SAMPLER]
    assert sampler.in_layouts == [as_layout(X)]
    assert [layout.dtype for layout in sampler.out_layouts] == [DType.float32]
    (out,) = sampler(Tensor(storage=Buffer.from_numpy(np.ones(2, np.float32))))
    assert out.to_numpy().tolist() == [5.0, 5.0]
    # The host weights stay releasable.
    assert dict(sampler.weights) == {}


def test_a_decoder_names_its_logits(
    graphs: dict[str, CompiledCallable[..., Any]],
) -> None:
    decoder = graphs[GraphRole.DECODER]
    assert decoder.out_tree.leaf_paths == ("logits.next_token_logits",)


def test_a_decoder_without_named_logits_fails_to_compile(
    session: InferenceSession,
) -> None:
    with pytest.raises(ValueError, match=r"decoder.*logits"):
        compile_and_init_graphs(
            session, {GraphRole.DECODER: _spec("positional", 1.0)}
        )


def _named_decoder(inputs: Mapping[str, object]) -> GraphSpec:
    scale = Scale("named_weight")
    return GraphSpec(
        Outputs(scale, _logits),
        "named",
        inputs,
        weights={"named_weight": np.full(2, 1.0, dtype=np.float32)},
    )


def test_a_decoder_records_its_named_input_tree(
    session: InferenceSession,
) -> None:
    graphs = compile_and_init_graphs(
        session,
        {
            GraphRole.DECODER: _named_decoder(
                OrderedDict(batch=RaggedBatch(x=X))
            )
        },
    )
    in_tree = graphs[GraphRole.DECODER].in_tree
    assert in_tree.leaf_paths == ("0.0.batch.x",)
    assert '"ragged_batch"' in json.dumps(treedef_to_json(in_tree))


def test_a_named_decoder_input_outside_a_kinded_group_fails_to_compile(
    session: InferenceSession,
) -> None:
    with pytest.raises(ValueError, match=r"0\.x .*kinded group"):
        compile_and_init_graphs(
            session, {GraphRole.DECODER: _named_decoder(OrderedDict(x=X))}
        )


def test_a_pipeline_stage_decoder_hands_on_hidden_states(
    session: InferenceSession,
) -> None:
    def boundary(x: TensorValue) -> DecoderOutputs:
        return DecoderOutputs(hidden_states=HiddenStates(stage_boundary=(x,)))

    graphs = compile_and_init_graphs(
        session, {GraphRole.DECODER: _spec("stage", 1.0, wrap=boundary)}
    )
    assert graphs[GraphRole.DECODER].out_tree.leaf_paths == (
        "hidden_states.stage_boundary.0",
    )


def test_an_output_tree_that_cannot_be_exported_fails_to_compile(
    session: InferenceSession,
) -> None:
    with pytest.raises(ArtifactError, match=r"VISION.*Unkinded"):
        compile_and_init_graphs(
            session,
            {
                GraphRole.VISION: _spec(
                    "unkinded", 1.0, wrap=lambda x: Unkinded(scaled=x)
                )
            },
        )


def test_weights_default_to_the_state_dict(session: InferenceSession) -> None:
    module = Scale("weight")
    module.load_state_dict({"weight": np.full(2, 7.0, dtype=np.float32)})
    graphs = compile_and_init_graphs(
        session, {"only": GraphSpec(module, "s", [X])}
    )
    assert _run(graphs["only"].engine_model) == [7.0, 7.0]


def test_a_group_needs_distinct_graph_names(
    session: InferenceSession,
) -> None:
    with pytest.raises(ValueError, match="distinct graph names"):
        compile_and_init_graphs(
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
    graphs = compile_and_init_graphs(session, specs)
    assert _run(graphs["a"].engine_model) == [1.0, 1.0]
    assert _run(graphs["b"].engine_model) == [1.0, 1.0]


def test_identical_graphs_with_different_names_each_init(
    session: InferenceSession,
) -> None:
    specs = {
        name: GraphSpec(
            Scale("shared_weight"),
            name,
            [X],
            weights={"shared_weight": np.full(2, 4.0, dtype=np.float32)},
        )
        for name in ("first", "second")
    }
    graphs = compile_and_init_graphs(session, specs)
    assert _run(graphs["first"].engine_model) == [4.0, 4.0]
    assert _run(graphs["second"].engine_model) == [4.0, 4.0]


def test_a_modulev3_graph_runs_as_its_forward(
    session: InferenceSession,
) -> None:
    graphs = compile_and_init_graphs(
        session, {GraphRole.DECODER: _v3_spec("scale_v3", 7.0)}
    )
    run = graphs[GraphRole.DECODER]
    assert _run(run.engine_model) == [7.0, 7.0]
    # ``forward`` returns one tensor, not a tuple of the graph's outputs.
    out = run(Buffer.from_numpy(np.ones(2, dtype=np.float32)))
    assert out.to_numpy().tolist() == [7.0, 7.0]
    assert set(run.weights) == {"weight"}


def test_a_modulev3_graph_takes_no_group(session: InferenceSession) -> None:
    with pytest.raises(ValueError, match="no group"):
        compile_and_init_graphs(
            session,
            {
                "a": _v3_spec("a", 1.0, group="g"),
                "b": _spec("b", 1.0, group="g"),
            },
        )


def test_a_modulev3_graph_casts_its_weights_only_when_asked() -> None:
    def spec(auto_cast: bool) -> GraphSpec:
        return GraphSpec(
            ScaleV3(Tensor.zeros([2], dtype=DType.bfloat16, device=CPU())),
            "scale_bf16",
            [TensorType(DType.bfloat16, [2], device=DeviceRef.CPU())],
            weights={"weight": np.full(2, 2.0, dtype=np.float32)},
            auto_cast=auto_cast,
        )

    assert set(spec(auto_cast=True).weights_registry()) == {"weight"}
    with pytest.raises(ValueError):
        spec(auto_cast=False).weights_registry()
