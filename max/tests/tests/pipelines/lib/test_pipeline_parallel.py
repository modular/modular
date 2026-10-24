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
"""Tests for the pipeline-parallel stage contract on a toy decoder."""

from __future__ import annotations

import pickle
from dataclasses import replace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    BufferType,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    Weight,
    ops,
)
from max.nn import LayerList, MissingLayer, Module
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    MAXModelConfig,
    ModelInputs,
    ModelManifest,
    ModelOutputs,
    PipelineConfig,
)
from max.pipelines.lib.pipeline_parallel import (
    ArchConfigWithPartitionSpec,
    PartitionSpec,
    PipelinedModel,
    PipelineStage,
    StageTraced,
    partition_spec,
    pipeline_stages,
    raise_stage_traced,
    split_layers,
    stage_graph,
    stage_layers,
    stage_outputs,
    stage_pipeline_config,
)
from max.tree import Tree

NUM_LAYERS = 4
HIDDEN = 4
VOCAB = 8
CPU_REF = DeviceRef.CPU()
INPUT_TYPES = (
    TensorType(DType.int64, ["seq"], CPU_REF),
    TensorType(DType.float32, [1], CPU_REF),
    TensorType(DType.float32, [1], CPU_REF),
)


class Block(Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = Weight("weight", DType.float32, [HIDDEN, HIDDEN], CPU_REF)

    def __call__(self, h: TensorValue) -> TensorValue:
        return h @ self.weight


class ToyDecoder(Module):
    """Embeds tokens, runs its layers, and projects to the vocabulary.

    What its layers hand on grows by one tensor per layer, so each cut
    carries a different tree.
    """

    def __init__(self, layer_ids: range) -> None:
        super().__init__()
        self.layer_ids = layer_ids
        self.embedding = Weight(
            "embedding", DType.float32, [VOCAB, HIDDEN], CPU_REF
        )
        self.layers = LayerList(
            [
                Block() if i in layer_ids else MissingLayer(i)
                for i in range(NUM_LAYERS)
            ]
        )
        self.head = Weight("head", DType.float32, [HIDDEN, VOCAB], CPU_REF)

    def __call__(
        self,
        tokens: TensorValue,
        scale: TensorValue,
        bias: TensorValue,
        *,
        boundary_inputs: dict[str, Any] | None = None,
    ) -> Any:
        if self.layer_ids.start == 0:
            x = ops.gather(self.embedding, tokens, axis=0)
            trail: list[TensorValue] = []
        else:
            assert boundary_inputs is not None
            x, trail = boundary_inputs["h"], list(boundary_inputs["trail"])
        double_scale = scale * 2
        held = list(self.layers)[self.layer_ids.start : self.layer_ids.stop]
        for layer in held:
            x = layer(x) * double_scale
            trail.append(ops.mean(x, axis=-1))
        if self.layer_ids.stop < NUM_LAYERS:
            return {"h": x, "trail": trail}
        total = trail[0]
        for t in trail[1:]:
            total = total + t
        return (x @ self.head + bias, total)


class ToyModel(PipelinedModel[TextContext]):
    """The pipeline model of one stage of the toy decoder."""

    def __init__(
        self,
        stage: int | None = None,
        num_stages: int = 1,
        boundary_types: Tree[TensorType] | None = None,
        trace_only: bool = False,
    ) -> None:
        config = PipelineConfig.model_construct(
            models=ModelManifest({"main": MAXModelConfig()})
        )
        if stage is not None:
            sizes = split_layers(
                PartitionSpec(num_layers=NUM_LAYERS), num_stages
            )
            config = stage_pipeline_config(
                config,
                replace(
                    pipeline_stages(sizes)[stage],
                    boundary_types=boundary_types,
                    trace_only=trace_only,
                ),
            )
        self.pipeline_config = config
        self.devices = [CPU()]
        self.decoder = _decoder(stage_layers(self.pipeline_config, NUM_LAYERS))

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        raise NotImplementedError

    def build(self) -> tuple[Graph, dict[str, Any]]:
        stage = self.pipeline_config.model.pipeline_stage
        index = stage.index if stage else 0
        with stage_graph(stage, f"toy_{index}", INPUT_TYPES) as (
            graph,
            inputs,
            boundary_inputs,
        ):
            tokens, scale, bias = (v.tensor for v in inputs)
            outputs = self.decoder(
                tokens, scale, bias, boundary_inputs=boundary_inputs
            )
            graph.output(stage_outputs(stage, outputs))
        return graph, dict(self.decoder.state_dict())


def _weights() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    weights = {
        "embedding": rng.standard_normal((VOCAB, HIDDEN)),
        "head": rng.standard_normal((HIDDEN, VOCAB)),
    }
    for i in range(NUM_LAYERS):
        weights[f"layers.{i}.weight"] = (
            rng.standard_normal((HIDDEN, HIDDEN)) / 2
        )
    return {k: v.astype(np.float32) for k, v in weights.items()}


def _decoder(layer_ids: range) -> ToyDecoder:
    decoder = ToyDecoder(layer_ids)
    declared = set(decoder.raw_state_dict())
    decoder.load_state_dict(
        {k: v for k, v in _weights().items() if k in declared}
    )
    return decoder


def _host(array: np.ndarray) -> Buffer:
    return Buffer.from_numpy(array)


def _traced(
    stage: int, num_stages: int, boundary_types: Tree[TensorType] | None = None
) -> Any:
    with pytest.raises(StageTraced) as traced:
        ToyModel(stage, num_stages, boundary_types, trace_only=True).build()
    return traced.value.boundary_types


def test_one_stage_is_the_decoder_called_directly() -> None:
    staged, _ = ToyModel().build()
    # A weight's graph value belongs to the first graph that uses it.
    decoder = _decoder(range(NUM_LAYERS))
    with Graph("toy_0", input_types=INPUT_TYPES) as direct:
        tokens, scale, bias = (v.tensor for v in direct.inputs)
        direct.output(*decoder(tokens, scale, bias))
    assert str(staged) == str(direct)


def test_stages_match_the_whole_model() -> None:
    session = InferenceSession(devices=[CPU()])
    tokens = np.array([1, 5, 2], dtype=np.int64)
    scale = np.array([0.5], dtype=np.float32)
    bias = np.array([0.25], dtype=np.float32)

    graph, weights = ToyModel().build()
    expected = session.load(graph, weights_registry=weights).execute(
        _host(tokens), _host(scale), _host(bias)
    )

    graph0, weights0 = ToyModel(0, 2).build()
    graph1, weights1 = ToyModel(1, 2, _traced(0, 2)).build()
    # Each stage's registry also holds weights its graph never references,
    # which loading ignores.
    assert "head" in weights0 and "embedding" in weights1
    sent = session.load(graph0, weights_registry=weights0).execute(
        _host(tokens), _host(scale), _host(bias)
    )
    # The cut carries the hidden state and two trail entries.
    assert len(sent) == 3
    # The last stage takes the cut's tensors after the whole model's inputs,
    # and the tokens it ignores keep the length the hidden state shares with
    # them.
    got = session.load(graph1, weights_registry=weights1).execute(
        _host(tokens), _host(scale), _host(bias), *sent
    )
    for want, have in zip(expected, got, strict=True):
        np.testing.assert_array_equal(want.to_numpy(), have.to_numpy())


def test_one_full_stage_is_the_decoder_called_directly() -> None:
    staged, _ = ToyModel(0, 1).build()
    assert str(staged) == str(ToyModel().build()[0])


def test_pipeline_stage_is_not_a_config_field() -> None:
    assert "pipeline_stage" not in MAXModelConfig.model_fields
    config = MAXModelConfig()
    stage = PipelineStage(1, 2, range(2, 4))
    config._pipeline_stage = stage
    assert pickle.loads(pickle.dumps(config)).pipeline_stage == stage


def test_even_split_puts_the_remainder_first() -> None:
    assert split_layers(PartitionSpec(num_layers=10), 4) == [3, 3, 2, 2]


def test_even_split_moves_off_forbidden_cuts() -> None:
    spec = PartitionSpec(num_layers=8, forbidden_cuts=frozenset({4}))
    assert split_layers(spec, 2) == [3, 5]


def test_even_split_chooses_its_cuts_together() -> None:
    # Moving the first cut alone to the nearest allowed layer, 2, leaves the
    # second no allowed layer.
    spec = PartitionSpec(num_layers=5, forbidden_cuts=frozenset({3, 4}))
    assert split_layers(spec, 3) == [1, 1, 3]


@pytest.mark.parametrize("num_stages", [0, -1])
def test_a_split_needs_a_stage(num_stages: int) -> None:
    with pytest.raises(ValueError, match="at least one stage"):
        split_layers(PartitionSpec(num_layers=8), num_stages)


def test_given_split_must_avoid_forbidden_cuts() -> None:
    spec = PartitionSpec(num_layers=8, forbidden_cuts=frozenset({4}))
    with pytest.raises(ValueError, match="forbids a cut"):
        split_layers(spec, 2, [4, 4])


def test_given_split_must_cover_every_layer() -> None:
    with pytest.raises(ValueError, match="covers 7"):
        split_layers(PartitionSpec(num_layers=8), 2, [3, 4])


def test_pipeline_stages_cover_the_partition_in_order() -> None:
    stages = pipeline_stages([3, 3, 2])
    assert [stage.layers for stage in stages] == [
        range(3),
        range(3, 6),
        range(6, 8),
    ]
    assert [(s.is_first, s.is_last) for s in stages] == [
        (True, False),
        (False, False),
        (False, True),
    ]


def test_stage_layers() -> None:
    whole = PipelineConfig.model_construct(
        models=ModelManifest({"main": MAXModelConfig()})
    )
    assert stage_layers(whole, 8) == range(8)
    layers = [
        stage_layers(stage_pipeline_config(whole, stage), 8)
        for stage in pipeline_stages([3, 3, 2])
    ]
    assert layers == [range(3), range(3, 6), range(6, 8)]
    # The config that builds the whole model is left as it was.
    assert whole.model.pipeline_stage is None


def test_stage_layers_must_fit_the_decoder() -> None:
    whole = PipelineConfig.model_construct(
        models=ModelManifest({"main": MAXModelConfig()})
    )
    last = stage_pipeline_config(whole, pipeline_stages([3, 3, 2])[2])
    with pytest.raises(ValueError, match="does not fit"):
        stage_layers(last, 9)


SPEC = PartitionSpec(num_layers=8, forbidden_cuts=frozenset({1}))


class _CutConfig(ArchConfigWithPartitionSpec):
    @classmethod
    def construct_partition_spec(cls, huggingface_config: Any) -> PartitionSpec:
        return SPEC


class _UncutConfig:
    pass


def test_partition_spec_comes_from_the_architecture() -> None:
    assert partition_spec(Mock(config=_CutConfig), Mock()) is SPEC
    with pytest.raises(TypeError, match="ArchConfigWithPartitionSpec"):
        partition_spec(Mock(config=_UncutConfig), Mock())


def test_a_traced_stage_carries_its_boundary_types() -> None:
    # Four layers over three stages split 2, 1, 1.
    first = _traced(0, 3)
    assert first["h"] == TensorType(DType.float32, ["seq", HIDDEN], CPU_REF)
    assert len(first["trail"]) == 2
    # A middle stage traces from the boundary types it receives.
    second = _traced(1, 3, first)
    assert len(second["trail"]) == 3
    graph, _ = ToyModel(2, 3, second).build()
    assert len(list(graph.inputs)) == 4 + len(INPUT_TYPES)


def test_a_later_stage_needs_boundary_types() -> None:
    with pytest.raises(ValueError, match="needs the types"):
        ToyModel(1, 2).build()


def test_an_algebraic_dim_cannot_cross_a_cut() -> None:
    stage = replace(pipeline_stages([2, 2])[0], trace_only=True)
    with Graph("cut", input_types=INPUT_TYPES[:1]) as graph:
        tokens = graph.inputs[0].tensor
        doubled = ops.concat([tokens, tokens])
        with pytest.raises(ValueError, match="algebraic dim"):
            stage_outputs(stage, doubled)


def test_only_tensors_cross_a_cut() -> None:
    with pytest.raises(ValueError, match="Only tensors"):
        raise_stage_traced((BufferType(DType.float32, [HIDDEN], CPU_REF),))
