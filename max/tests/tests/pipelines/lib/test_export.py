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
"""Tests for exporting a pipeline's graphs to a pipeline artifact."""

from __future__ import annotations

import itertools
import json
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from max._artifact import ArtifactError
from max._artifact.batch_processor import BatchProcessorDims
from max._artifact.export import (
    SupportsCompiledModules,
    batch_processor_dims,
    pipeline_modules,
)
from max._artifact.model_artifact import ModelArtifact
from max._artifact.pipeline_modules import PipelineModules
from max._artifact.serialization import deserialize, serialize
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.experimental.compilation import CompiledCallable
from max.experimental.sharding.types import as_layout
from max.graph import BufferType, DeviceRef, TensorType, TensorValue
from max.nn import Module
from max.nn.graph_groups import Logits
from max.nn.transformer import DecoderOutputs, ReturnLogits
from max.pipelines.lib import (
    GraphRole,
    GraphSpec,
    ModelInputs,
    ModelOutputs,
    PipelineModelWithKVCache,
    compile_and_init_graphs,
)
from max.pipelines.lib.graph_specs import compile_roles_separately
from max.pipelines.lib.interfaces import BatchProcessorRuntime

X = TensorType(DType.float32, ["batch", 4], device=DeviceRef.CPU())


class _Scale(Module):
    def __init__(self, factor: int) -> None:
        super().__init__()
        self.factor = factor

    def __call__(self, x: TensorValue) -> DecoderOutputs:
        return DecoderOutputs(logits=Logits(next_token_logits=x * self.factor))


_factors = itertools.count(2)


def _specs(groups: Mapping[GraphRole, str | None]) -> dict[str, GraphSpec]:
    # Distinct factors keep the compile cache from handing one graph's
    # artifact to an identical graph of another name.
    return {
        role: GraphSpec(
            _Scale(next(_factors)),
            f"{role.value}_graph",
            [X],
            weights={},
            group=group,
        )
        for role, group in groups.items()
    }


def _compile(
    specs: Mapping[str, GraphSpec],
) -> dict[str, CompiledCallable[..., Any]]:
    return compile_and_init_graphs(InferenceSession(devices=[CPU()]), specs)


@dataclass
class _Pipeline:
    compiled_modules: Mapping[str, CompiledCallable[..., Any]]


def test_each_role_becomes_one_mef(tmp_path: Path) -> None:
    pipeline = _Pipeline(
        _compile(_specs({GraphRole.DECODER: None, GraphRole.SAMPLER: None}))
    )
    graphs = pipeline_modules(pipeline, "Fake")
    assert set(graphs.roles) == {"decoder", "sampler"}

    dest = serialize(ModelArtifact(model=graphs), tmp_path / "a")
    section = json.loads((dest / "artifact.json").read_text())["model"]
    assert set(section) == {"decoder", "sampler"}
    decoder = section["decoder"]
    assert decoder["mef"] == {"$file": "decoder/model.mef"}
    assert decoder["in_layouts"] == [
        {
            "buffer": False,
            "dtype": "float32",
            "shape": ["batch", 4],
            "mesh": {
                "devices": [{"type": "cpu", "id": 0}],
                "shape": [1],
                "axis_names": ["_"],
            },
            "placements": [{"type": "replicated"}],
        }
    ]
    assert sorted(str(p.relative_to(dest)) for p in dest.glob("*/*.mef")) == [
        "decoder/model.mef",
        "sampler/model.mef",
    ]

    loaded = deserialize(dest).model
    assert loaded is not None
    for role, compiled in graphs.roles.items():
        assert loaded.roles[role].in_layouts == compiled.in_layouts
        assert loaded.roles[role].in_tree == compiled.in_tree
        assert loaded.roles[role].out_layouts == compiled.out_layouts
        assert loaded.roles[role].out_tree == compiled.out_tree
    with pytest.raises(RuntimeError, match="not initialized"):
        _ = loaded.roles["decoder"].engine_model


def test_export_compiles_a_group_as_one_mef_per_role(tmp_path: Path) -> None:
    specs = _specs({GraphRole.DECODER: "model", GraphRole.VISION: "model"})
    with compile_roles_separately():
        graphs = _compile(specs)
    assert graphs[GraphRole.DECODER]._artifact is not (
        graphs[GraphRole.VISION]._artifact
    )
    dest = serialize(
        ModelArtifact(model=pipeline_modules(_Pipeline(graphs), "Fake")),
        tmp_path / "a",
    )
    assert sorted(str(p.relative_to(dest)) for p in dest.glob("*/*.mef")) == [
        "decoder/model.mef",
        "vision/model.mef",
    ]


class _LoadsItsOwnGraphs(PipelineModelWithKVCache[Any, Any, Any]):
    """A model that compiles its graphs itself, as several ModuleV3 and
    unified spec-decode models do, and so registers none by role."""

    def load_graphs(self, session: InferenceSession) -> dict[str, Model]:
        graph = _Scale(next(_factors)).trace(X, name="own")
        return {GraphRole.DECODER: session.load(graph)}

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        raise NotImplementedError


def test_a_model_that_loads_its_own_graphs_is_refused() -> None:
    model = _LoadsItsOwnGraphs.__new__(_LoadsItsOwnGraphs)
    model.load_graphs(InferenceSession(devices=[CPU()]))
    pipeline = _Pipeline(
        {
            **model.compiled_modules,
            **_compile(_specs({GraphRole.SAMPLER: None})),
        }
    )
    with pytest.raises(
        ArtifactError, match="OwnGraphs registers no decoder graph by role"
    ):
        pipeline_modules(pipeline, "OwnGraphs")


def test_a_pipeline_without_compiled_modules_is_not_exportable() -> None:
    assert isinstance(_Pipeline({}), SupportsCompiledModules)
    assert not isinstance(object(), SupportsCompiledModules)


def test_spec_decoder_pipeline_is_exported() -> None:
    pipeline = _Pipeline(_compile(_specs({GraphRole.SPEC_DECODER: None})))
    assert set(pipeline_modules(pipeline, "Fake").roles) == {"spec_decoder"}


def test_no_roles_are_refused() -> None:
    with pytest.raises(ArtifactError, match="no graphs"):
        PipelineModules({})


def _fake(**attrs: object) -> Any:
    """Stands in for an object that exposes only ``attrs``."""
    return SimpleNamespace(**attrs)


def test_batch_processor_dims_bound_a_bound_decoders_per_step_dims() -> None:
    runtime = BatchProcessorRuntime(
        pipeline_config=_fake(),
        devices=[],
        return_logits=ReturnLogits.LAST_TOKEN,
        max_batch_size=4,
        max_seq_len=256,
        max_batch_input_tokens=64,
    )
    processor = _fake(input_tree=OrderedDict(), runtime=runtime)
    model = _fake(
        batch_processor=processor,
        kv_params=_fake(page_size=128, num_draft_tokens=0),
    )
    tokens = TensorType(DType.int64, ["total_seq_len"], device=DeviceRef.CPU())
    # Pages are sized by the KV cache, so their dim gets no bound here.
    pages = BufferType(
        DType.bfloat16, ["total_num_pages", 2], device=DeviceRef.CPU()
    )
    decoder = _fake(in_layouts=[as_layout(tokens), as_layout(pages)])

    assert batch_processor_dims(model, decoder) == BatchProcessorDims(
        {"total_seq_len": runtime.max_batch_active_tokens}
    )

    processor.input_tree = None
    assert batch_processor_dims(model, decoder) is None
    model.batch_processor = None
    assert batch_processor_dims(model, decoder) is None
