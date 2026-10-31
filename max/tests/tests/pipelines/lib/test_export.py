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
from max._artifact.engine_config import (
    EngineConfig,
    EngineSamplingConfig,
    EngineSpeculativeConfig,
)
from max._artifact.export import (
    SupportsCompiledModules,
    batch_processor_dims,
    checkpoint_format_spec,
    model_artifact,
    pipeline_engine_config,
    pipeline_kv_cache,
    pipeline_modules,
    vision_preprocess_modules,
)
from max._artifact.kv_cache import (
    KVCacheQuantization,
    KVPoolLeaf,
    MHAKVCache,
    MLAKVCache,
    MSAKVCache,
    RecurrentStateCache,
    StateRegion,
)
from max._artifact.model_artifact import ModelArtifact
from max._artifact.pipeline_modules import PipelineModules
from max._artifact.serialization import deserialize, serialize
from max._artifact.vision import (
    AspectRatioPreserving,
    ModalityLayout,
    PromptText,
    Span,
    VisionMetadata,
)
from max.driver import CPU, DeviceSpec
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, TensorType, TensorValue
from max.nn.legacy import Module
from max.nn.legacy.graph_groups import Logits
from max.nn.legacy.kv_cache import (
    KVCacheQuantizationConfig,
    MHAKVCacheParams,
    MLAKVCacheParams,
    MSAKVCacheParams,
    MultiKVCacheParams,
    RecurrentStateParams,
    RecurrentStateRegion,
)
from max.nn.legacy.transformer import DecoderOutputs, ReturnLogits
from max.pipelines import lib as pipelines_lib
from max.pipelines.lib import (
    GraphRole,
    GraphSpec,
    KVCacheConfig,
    MAXModelConfig,
    MemoryPlan,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModelWithKVCache,
    PipelineRuntimeConfig,
    compile_and_init_graphs,
)
from max.pipelines.lib.graph_specs import compile_roles_separately
from max.pipelines.lib.interfaces import BatchProcessorRuntime
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.sampling import SamplingConfig
from max.pipelines.speculative import SpeculativeConfig
from max.tensor.compilation import CompiledCallable
from max.tensor.sharding.types import as_layout

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


_VISION = VisionMetadata.from_layouts(
    image=ModalityLayout(
        patch_size=16,
        merge_size=3,
        strategy=AspectRatioPreserving(max_patches=2520),
        span=Span(image_token_id=1, begin_marker=2, end_marker=3),
    ),
    video=None,
    max_image_bytes=0,
    max_video_bytes=0,
    prompt=PromptText(),
)


def test_export_adds_the_vision_preprocess_roles(tmp_path: Path) -> None:
    """A pipeline with a vision graph exports the graphs its model returns
    for preprocessing images and video beside the roles it serves, and its
    vision metadata."""
    image, video = _specs(
        {
            GraphRole.VISION_PREPROCESS: None,
            GraphRole.VISION_PREPROCESS_VIDEO: None,
        }
    ).values()
    model = _fake(
        vision_preprocess_spec=lambda: image,
        vision_preprocess_video_spec=lambda: video,
        vision_metadata=lambda: _VISION,
    )
    graphs = pipeline_modules(
        _Pipeline(
            _compile(_specs({GraphRole.DECODER: None, GraphRole.VISION: None}))
        ),
        "Fake",
    )

    dest = serialize(model_artifact(graphs, model, None), tmp_path / "a")
    written = json.loads((dest / "artifact.json").read_text())
    assert set(written["model"]) == {
        "decoder",
        "vision",
        "vision_preprocess",
        "vision_preprocess_video",
    }
    for role in ("vision_preprocess", "vision_preprocess_video"):
        assert written["model"][role]["mef"] == {"$file": f"{role}/model.mef"}
    assert deserialize(dest).vision == _VISION

    text_only = pipeline_modules(
        _Pipeline(_compile(_specs({GraphRole.DECODER: None}))), "Fake"
    )
    artifact = model_artifact(text_only, model, None)
    assert artifact.model is not None
    assert set(artifact.model.roles) == {"decoder"}
    assert artifact.vision is None
    assert model_artifact(graphs, None, None).vision is None

    assert set(
        vision_preprocess_modules(
            _fake(
                vision_preprocess_spec=lambda: image,
                vision_preprocess_video_spec=lambda: None,
            )
        )
    ) == {"vision_preprocess"}
    assert (
        vision_preprocess_modules(
            _fake(
                vision_preprocess_spec=lambda: None,
                vision_preprocess_video_spec=lambda: None,
            )
        )
        == {}
    )


def test_spec_decoder_pipeline_is_exported() -> None:
    pipeline = _Pipeline(_compile(_specs({GraphRole.SPEC_DECODER: None})))
    assert set(pipeline_modules(pipeline, "Fake").roles) == {"spec_decoder"}


def test_no_roles_are_refused() -> None:
    with pytest.raises(ArtifactError, match="no graphs"):
        PipelineModules({})


def _fake(**attrs: object) -> Any:
    """Stands in for an object that exposes only ``attrs``."""
    return SimpleNamespace(**attrs)


def test_checkpoint_format_spec_is_the_architectures_document() -> None:
    repo = _fake(repo_id="org/model")
    assert checkpoint_format_spec(None, repo) is None
    assert checkpoint_format_spec(_fake(format_spec=None), repo) is None
    document = Path("spec.json")
    assert checkpoint_format_spec(_fake(format_spec=document), repo) == document


def test_checkpoint_format_spec_is_picked_per_checkpoint() -> None:
    document = Path("spec.json")
    arch = _fake(
        format_spec=lambda repo: document if repo.repo_id == "org/new" else None
    )
    assert checkpoint_format_spec(arch, _fake(repo_id="org/new")) == document
    assert checkpoint_format_spec(arch, _fake(repo_id="org/old")) is None


@pytest.mark.parametrize(
    "architecture",
    [
        "deepseekV41",
        "gemma4",
        "glm5_1",
        "laguna",
        "lfm3",
        "mimo_v2",
        "muse_glimmer",
        "qwen3_5",
    ],
)
def test_shipped_format_specs_are_packaged(architecture: str) -> None:
    architectures = Path(pipelines_lib.__file__).parent.parent / "architectures"
    assert (architectures / architecture / "formatspec.json").is_file()


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


_DEVICES = (DeviceSpec.accelerator(0), DeviceSpec.accelerator(1))
_PLAN = MemoryPlan(
    planned_max_batch_size=8,
    footprint=0,
    planned_max_length=4096,
    device_specs=_DEVICES,
    planned_max_batch_total_tokens=4096,
)


def _pipeline_config(
    *,
    sampling: SamplingConfig,
    speculative: SpeculativeConfig | None = None,
    reasoning_parser: str | None = None,
) -> PipelineConfig:
    model = MAXModelConfig.model_construct(
        model_path="org/model",
        device_specs=list(_DEVICES),
        data_parallel_degree=2,
        kv_cache=KVCacheConfig(),
    )
    return PipelineConfig.model_construct(
        runtime=PipelineRuntimeConfig.model_construct(
            # The configuration's, which the plan's values replace.
            max_batch_size=None,
            max_batch_input_tokens=2048,
            ep_size=2,
            reasoning_parser=reasoning_parser,
        ),
        models=ModelManifest({"main": model}),
        sampling=sampling,
        speculative=speculative,
    )


def test_engine_config_copies_the_planned_settings() -> None:
    config = _pipeline_config(
        sampling=SamplingConfig.model_construct(
            enable_structured_output=True, enable_penalties=True
        )
    )
    assert pipeline_engine_config(config, _PLAN) == EngineConfig(
        max_batch_size=8,
        max_length=4096,
        max_batch_input_tokens=2048,
        max_batch_total_tokens=4096,
        devices=_DEVICES,
        data_parallel_degree=2,
        ep_size=2,
        sampling=EngineSamplingConfig(
            needs_bitmask_constraints=True,
            enable_variable_logits=False,
            enable_penalties=True,
        ),
        speculative=None,
    )


def test_engine_config_copies_the_speculative_settings() -> None:
    config = _pipeline_config(
        sampling=SamplingConfig.model_construct(),
        speculative=SpeculativeConfig.model_construct(
            speculative_method="mtp", num_speculative_tokens=3
        ),
        reasoning_parser="gemma4",
    )
    assert pipeline_engine_config(config, _PLAN).speculative == (
        EngineSpeculativeConfig(
            speculative_method="mtp",
            num_speculative_tokens=3,
            draft_proposal="argmax",
            reasoning_parser="gemma4",
        )
    )


def _mha(num_layers: int, window_size: int | None = None) -> MHAKVCacheParams:
    return MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=4,
        head_dim=64,
        num_layers=num_layers,
        devices=[DeviceRef.CPU()],
        window_size=window_size,
    )


def test_kv_cache_records_every_cache_in_graph_order() -> None:
    quantized = MHAKVCacheParams(
        dtype=DType.float8_e4m3fn,
        n_kv_heads=4,
        head_dim=64,
        num_layers=1,
        devices=[DeviceRef.CPU()],
        kvcache_quant_config=KVCacheQuantizationConfig(
            scale_dtype=DType.float8_e8m0fnu, quantization_granularity=32
        ),
    )
    msa = MSAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=1,
        head_dim=128,
        num_layers=1,
        devices=[DeviceRef.CPU()],
        allow_kv_head_replication=True,
        stride_alignment=512,
    )
    mla = MLAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=576,
        num_layers=1,
        devices=[DeviceRef.CPU()],
        num_q_heads=16,
        sparse_indices_stride=64,
    )
    state = RecurrentStateParams(
        devices=[DeviceRef.CPU()],
        regions=(
            RecurrentStateRegion(
                leaf_id="conv",
                num_layers=2,
                row_shape=(64, 3),
                dtype=DType.bfloat16,
            ),
        ),
    )
    params = MultiKVCacheParams.from_params(
        {
            "target": MultiKVCacheParams.from_params(
                {
                    "sliding_attention": _mha(5, window_size=1024),
                    "full_attention": quantized,
                }
            ),
            "indexer": msa,
            "latent": mla,
            "state": state,
        }
    )
    layout = pipeline_kv_cache(params)
    assert layout.page_size == 128
    assert [(cache.name, cache.kind) for cache in layout.caches] == [
        ("target.sliding_attention", "mha"),
        ("target.full_attention", "mha"),
        ("indexer", "msa"),
        ("latent", "mla"),
        ("state", "recurrent"),
    ]
    sliding_id = "target.sliding_attention.sliding_window_group(1024)"
    assert layout.caches[0] == MHAKVCache(
        name="target.sliding_attention",
        dtype=DType.bfloat16,
        num_layers=5,
        n_kv_heads=4,
        head_dim=64,
        v_head_dim=None,
        window_size=1024,
        allow_kv_head_replication=False,
        slots_per_page=128,
        stride_alignment=None,
        scratch=False,
        kvcache_quant_config=None,
        leaves=(
            KVPoolLeaf(
                leaf_id=sliding_id,
                group="sliding_window",
                window_size=1024,
                bytes_per_page=params.leaves()[sliding_id].bytes_per_page,
                stride_alignment=None,
            ),
        ),
    )
    full = layout.caches[1]
    assert isinstance(full, MHAKVCache)
    assert (full.dtype, full.kvcache_quant_config) == (
        DType.float8_e4m3fn,
        KVCacheQuantization(
            scale_dtype=DType.float8_e8m0fnu,
            quantization_granularity=32,
            fp8_tail_dim=0,
        ),
    )

    def full_leaf(path: str) -> KVPoolLeaf:
        leaf_id = f"{path}.full_group"
        region = params.leaves()[leaf_id]
        return KVPoolLeaf(
            leaf_id=leaf_id,
            group="full",
            window_size=None,
            bytes_per_page=region.bytes_per_page,
            stride_alignment=region.stride_alignment,
        )

    assert layout.caches[2] == MSAKVCache(
        name="indexer",
        dtype=DType.bfloat16,
        num_layers=1,
        n_kv_heads=1,
        head_dim=128,
        v_head_dim=None,
        window_size=None,
        allow_kv_head_replication=True,
        slots_per_page=128,
        stride_alignment=512,
        scratch=False,
        kvcache_quant_config=None,
        leaves=(full_leaf("indexer"),),
    )
    assert layout.caches[3] == MLAKVCache(
        name="latent",
        dtype=DType.bfloat16,
        num_layers=1,
        num_q_heads=16,
        head_dim=576,
        v_head_dim=None,
        window_size=None,
        slots_per_page=128,
        sparse_indices_stride=64,
        stride_alignment=None,
        scratch=False,
        kvcache_quant_config=None,
        leaves=(full_leaf("latent"),),
    )
    assert layout.caches[4] == RecurrentStateCache(
        name="state",
        regions=(
            StateRegion(
                leaf_id="conv",
                num_layers=2,
                row_shape=(64, 3),
                dtype=DType.bfloat16,
                scratch=False,
            ),
        ),
        leaves=(
            KVPoolLeaf(
                leaf_id="conv",
                group="recurrent",
                window_size=None,
                bytes_per_page=2 * 64 * 3 * 2,
                stride_alignment=None,
            ),
        ),
    )


class _CustomKVCacheParams(MHAKVCacheParams):
    pass


def test_kv_cache_refuses_a_cache_type_it_does_not_record() -> None:
    custom = _CustomKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=4,
        head_dim=64,
        num_layers=1,
        devices=[DeviceRef.CPU()],
    )
    params = MultiKVCacheParams.from_params({"custom": custom})
    with pytest.raises(ArtifactError, match="_CustomKVCacheParams"):
        pipeline_kv_cache(params)
