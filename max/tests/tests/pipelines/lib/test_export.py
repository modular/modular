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
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, Mock, NonCallableMock, PropertyMock, patch

import huggingface_hub
import pytest
from click.testing import CliRunner
from max import tree
from max._artifact import ArtifactError, HuggingFaceRef
from max._artifact import export as export_module
from max._artifact.batch_processor import BatchProcessorDims
from max._artifact.engine_config import (
    CapturedHostInput,
    EngineCaptureConfig,
    EngineConfig,
    EngineParallelConfig,
    EnginePrefillContextParallel,
    EngineSamplingConfig,
    EngineSpeculativeConfig,
    SpecGraphFacts,
)
from max._artifact.export import (
    SupportsCompiledModules,
    _check_batch_context_window,
    _prefill_context_parallel,
    _takes_mrope_positions,
    _verify_facts,
    _warn_unhonored_context_parallel,
    batch_processor_dims,
    captured_host_inputs,
    checkpoint_format_spec,
    checkpoint_tokenizer,
    model_artifact,
    pipeline_engine_config,
    pipeline_kv_cache,
    pipeline_modules,
    pipeline_sampling_defaults,
    ships_overlap_helpers,
    verify_step_prefill_budget,
    vision_preprocess_modules,
)
from max._artifact.kv_cache import (
    ContextParallel,
    KVCacheLayout,
    KVCacheQuantization,
    KVPoolLeaf,
    MHAKVCache,
    MLAKVCache,
    MSAIndexKVCache,
    RecurrentStateCache,
    StateRegion,
)
from max._artifact.model_artifact import ModelArtifact
from max._artifact.pipeline_modules import PipelineModules
from max._artifact.sampling_defaults import SamplingDefaults
from max._artifact.serialization import deserialize, serialize
from max._artifact.tokenizer import Tokenizer
from max._artifact.vision import (
    AspectRatioPreserving,
    ModalityLayout,
    PromptText,
    Span,
    VisionMetadata,
)
from max.driver import CPU, Device, DeviceSpec
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, TensorType, TensorValue
from max.nn.kv_cache import (
    KVCacheContextParallel,
    KVCacheQuantizationConfig,
    MHAKVCacheParams,
    MLAKVCacheParams,
    MSAIndexKVCacheParams,
    MSAKVCacheParams,
    MultiKVCacheParams,
    RecurrentStateParams,
    RecurrentStateRegion,
)
from max.nn.legacy import Module
from max.nn.legacy.graph_groups import Logits, MRoPEPositions, RaggedBatch
from max.nn.legacy.transformer import DecoderOutputs
from max.nn.outputs import ReturnLogits
from max.pipelines import lib as pipelines_lib
from max.pipelines.architectures.gemma4.arch import gemma4_arch
from max.pipelines.architectures.llama3.arch import llama_arch
from max.pipelines.context import SamplingParamsGenerationConfigDefaults
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
    SupportedArchitecture,
    compile_and_init_graphs,
)
from max.pipelines.lib.graph_specs import compile_roles_separately
from max.pipelines.lib.interfaces import BatchProcessorRuntime
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.lib.module_registration import (
    DecoderBatchProcessor,
    decoder_input_types,
)
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    SpecDecodePipelineModel,
    _BatchProcessor,
)
from max.pipelines.lib.pipeline_variants.text_generation import (
    TextGenerationPipelineInterface,
)
from max.pipelines.sampling import SamplingConfig
from max.pipelines.speculative import SpeculativeConfig
from max.pipelines.speculative.config import VerifyWidthRange
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


def test_only_an_accelerator_export_ships_the_overlap_helpers() -> None:
    accelerator = Mock(spec=Device)
    accelerator.is_host = False
    assert not ships_overlap_helpers([CPU()])
    assert ships_overlap_helpers([accelerator])
    assert ships_overlap_helpers([CPU(), accelerator])


def test_only_a_graph_with_mrope_positions_takes_the_positions_feed() -> None:
    # The serving engine stages M-RoPE positions by this group's kind, so the
    # export ships the positions feed exactly when the signature holds it.
    tokens = TensorType(DType.int64, ["total_seq_len"], device=DeviceRef.CPU())
    positions = TensorType(
        DType.int64, [3, "total_seq_len"], device=DeviceRef.CPU()
    )

    def graph(inputs: object) -> CompiledCallable[..., Any]:
        signature: tuple[list[Any], tree.TreeDef] = (
            [],
            tree.flatten(inputs)[1],
        )
        return cast(
            CompiledCallable[..., Any], _fake(graph_signature=lambda: signature)
        )

    plain = OrderedDict(batch=RaggedBatch(tokens=tokens))
    assert not _takes_mrope_positions(graph(plain))
    # How a replay gets each host input is recorded for the group kinds the
    # graphs take, and only those.
    recorded = captured_host_inputs([graph(plain)])
    assert recorded["ragged_batch.return_n_logits"] == CapturedHostInput(
        "shape"
    )
    assert not any(
        name.startswith(("spec_decode.", "paged_kv.")) for name in recorded
    )
    nested = OrderedDict(
        batch=RaggedBatch(tokens=tokens),
        rope=OrderedDict(positions=MRoPEPositions(position_ids=positions)),
    )
    assert _takes_mrope_positions(graph(nested))


def test_a_decoder_taking_decoder_inputs_records_its_host_batch_fields() -> (
    None
):
    # A decoder registered by role (Llama's, for one) takes its batch fields
    # as direct leaves of its decoder_inputs root, which the serving engine
    # reads as ragged_batch fields. Each one it reads on the host needs a
    # replay mode, or the engine refuses device graph capture at boot.
    types = decoder_input_types(
        NonCallableMock(get_symbolic_inputs=lambda: (X,)),
        DeviceRef.CPU(),
        data_parallel_degree=2,
    )
    signature: tuple[list[Any], tree.TreeDef] = (
        [],
        tree.flatten(((types,), {}))[1],
    )
    decoder = cast(
        CompiledCallable[..., Any], _fake(graph_signature=lambda: signature)
    )
    recorded = captured_host_inputs([decoder])
    for name in (
        "return_n_logits",
        "host_input_row_offsets",
        "data_parallel_splits",
    ):
        assert name in types
        assert recorded[f"ragged_batch.{name}"] == CapturedHostInput("shape")


def test_a_model_that_loads_its_own_graphs_is_refused() -> None:
    model = _LoadsItsOwnGraphs.__new__(_LoadsItsOwnGraphs)
    model.load_graphs(InferenceSession(devices=[CPU()]))
    pipeline = _Pipeline(
        {
            **model.compiled_graphs,
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


def test_a_vision_graph_needs_its_metadata_and_preprocessing_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A serving engine needs both the media layout and the image
    preprocessing graph to serve a vision graph, so a model that declares
    only one of them is refused. A model that declares neither exports its
    vision graph alone."""
    graphs = PipelineModules({"decoder": _fake(), "vision": _fake()})
    no_preprocessing = _fake(
        vision_preprocess_spec=lambda: None,
        vision_preprocess_video_spec=lambda: None,
        vision_metadata=lambda: _VISION,
    )
    with pytest.raises(ArtifactError, match="no vision_preprocess graph"):
        model_artifact(graphs, no_preprocessing, None)

    no_metadata = _fake(vision_metadata=lambda: None)
    monkeypatch.setattr(
        export_module,
        "vision_preprocess_modules",
        lambda model: {"vision_preprocess": _fake()},
    )
    with pytest.raises(ArtifactError, match="no vision metadata"):
        model_artifact(graphs, no_metadata, None)

    monkeypatch.setattr(
        export_module, "vision_preprocess_modules", lambda model: {}
    )
    artifact = model_artifact(graphs, no_metadata, None)
    assert artifact.vision is None
    assert artifact.model is not None
    assert set(artifact.model.roles) == {"decoder", "vision"}


def test_spec_decoder_pipeline_is_exported() -> None:
    pipeline = _Pipeline(_compile(_specs({GraphRole.SPEC_DECODER: None})))
    assert set(pipeline_modules(pipeline, "Fake").roles) == {"spec_decoder"}


def test_no_roles_are_refused() -> None:
    with pytest.raises(ArtifactError, match="no graphs"):
        PipelineModules({})


def _fake(**attrs: object) -> Any:
    """Stands in for an object that exposes only ``attrs``."""
    return SimpleNamespace(**attrs)


@pytest.mark.parametrize(
    ("published", "server", "expected"),
    [
        # Nemotron: sampled, with MAX's default top-k (none).
        (
            {"do_sample": True, "temperature": 1.0, "top_p": 0.95},
            {},
            SamplingDefaults(
                temperature=1.0, top_k=None, top_p=0.95, repetition_penalty=1.0
            ),
        ),
        # Gemma 4: no do_sample, each value applies.
        (
            {"temperature": 1.0, "top_k": 64, "top_p": 0.95},
            {},
            SamplingDefaults(
                temperature=1.0, top_k=64, top_p=0.95, repetition_penalty=1.0
            ),
        ),
        # do_sample false is greedy whatever else the file sets.
        (
            {"do_sample": False, "temperature": 0.6},
            {},
            SamplingDefaults(
                temperature=0.0, top_k=1, top_p=1.0, repetition_penalty=1.0
            ),
        ),
        # A published repetition penalty carries through.
        (
            {"repetition_penalty": 1.05},
            {},
            SamplingDefaults(
                temperature=1.0, top_k=None, top_p=1.0, repetition_penalty=1.05
            ),
        ),
        # Nothing published: MAX's own defaults.
        (
            {},
            {},
            SamplingDefaults(
                temperature=1.0, top_k=None, top_p=1.0, repetition_penalty=1.0
            ),
        ),
        # The server's --temperature and --top-k override the checkpoint.
        (
            {"temperature": 1.0, "top_k": 64},
            {"temperature": 0.3, "top_k": 8},
            SamplingDefaults(
                temperature=0.3, top_k=8, top_p=1.0, repetition_penalty=1.0
            ),
        ),
    ],
)
def test_pipeline_sampling_defaults_resolve_as_max_serve_does(
    published: dict[str, Any],
    server: dict[str, Any],
    expected: SamplingDefaults,
) -> None:
    config = _fake(
        runtime=_fake(
            temperature=server.get("temperature"), top_k=server.get("top_k")
        ),
        model=_fake(
            sampling_params_defaults=SamplingParamsGenerationConfigDefaults(
                **published
            )
        ),
    )
    assert pipeline_sampling_defaults(config) == expected


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


def test_checkpoint_tokenizer_is_the_checkpoints_own(tmp_path: Path) -> None:
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "tokenizer_config.json").write_text("{}")
    (tmp_path / "chat_template.jinja").write_text("{{ checkpoint }}")
    repo = _fake(
        repo_type="local",
        local_path=str(tmp_path),
        repo_id=str(tmp_path),
        revision="main",
    )
    named = {"default": "{{ messages }}", "tool_use": "{{ tools }}"}
    tokenizer = _fake(
        delegate=_fake(get_vocab=lambda: {"a": 0, "b": 9}, chat_template=named),
        eos_token_ids={9, 3},
    )
    config = _fake(model=_fake(huggingface_config=_fake(vocab_size=16)))
    arch = _fake(tokenizer_family=lambda repo: ("gemma4", "thinking"))

    assert checkpoint_tokenizer(arch, repo, tokenizer, config) == Tokenizer(
        family="gemma4",
        flavor="thinking",
        files={
            name: HuggingFaceRef(str(tmp_path), "main", name)
            for name in ("tokenizer.json", "tokenizer_config.json")
        },
        chat_templates=named,
        vocab_size=16,
        sampleable_vocab_size=10,
        eos_token_ids=(3, 9),
    )

    served = _fake(
        delegate=_fake(
            get_vocab=lambda: {"a": 0}, chat_template="{{ served }}"
        ),
        eos_token_ids={0},
    )
    unnamed = checkpoint_tokenizer(None, repo, served, config)
    assert unnamed is not None
    assert (unnamed.family, unnamed.flavor) == (None, None)
    assert unnamed.chat_templates == {"default": "{{ served }}"}
    untemplated = _fake(
        delegate=_fake(get_vocab=lambda: {"a": 0}, chat_template=None),
        eos_token_ids={0},
    )
    result = checkpoint_tokenizer(None, repo, untemplated, config)
    assert result is not None and result.chat_templates == {}
    assert checkpoint_tokenizer(arch, repo, _fake(), config) is None


def test_an_offline_hub_snapshot_keeps_its_repo_id_at_its_commit(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshots" / "0123abcd"
    snapshot.mkdir(parents=True)
    (snapshot / "tokenizer.json").write_text("{}")
    repo = _fake(
        repo_type="local",
        local_path=str(snapshot),
        repo_id="org/model",
        revision="main",
    )
    tokenizer = _fake(
        delegate=_fake(get_vocab=lambda: {"a": 0}, chat_template=None),
        eos_token_ids={0},
    )
    config = _fake(model=_fake(huggingface_config=_fake(vocab_size=4)))
    result = checkpoint_tokenizer(None, repo, tokenizer, config)
    assert result is not None
    assert result.files == {
        "tokenizer.json": HuggingFaceRef(
            "org/model", "0123abcd", "tokenizer.json"
        )
    }


def test_checkpoint_tokenizer_files_are_referenced_at_the_cached_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cached = "a" * 40
    downloaded = "b" * 40
    fetched: list[str] = []

    def from_cache(repo_id: str, name: str, revision: str) -> str | None:
        assert (repo_id, revision) == ("org/model", "main")
        if name != "tokenizer.json":
            return None
        return str(tmp_path / "snapshots" / cached / name)

    def download(repo_id: str, name: str, revision: str) -> str:
        fetched.append(name)
        if name == "tiktoken.model":
            raise huggingface_hub.errors.EntryNotFoundError(name)
        return str(tmp_path / "snapshots" / downloaded / name)

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", from_cache)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    repo = _fake(repo_type="online", repo_id="org/model", revision="main")
    tokenizer = _fake(
        delegate=_fake(get_vocab=lambda: {"a": 0}, chat_template=None),
        eos_token_ids={0},
    )
    config = _fake(model=_fake(huggingface_config=_fake(vocab_size=1)))

    result = checkpoint_tokenizer(None, repo, tokenizer, config)

    assert result is not None
    assert result.files == {
        "tokenizer.json": HuggingFaceRef("org/model", cached, "tokenizer.json"),
        "tokenizer_config.json": HuggingFaceRef(
            "org/model", downloaded, "tokenizer_config.json"
        ),
    }
    assert fetched == ["tokenizer_config.json", "tiktoken.model"]


@pytest.mark.parametrize(
    ("arch", "repo_id", "family"),
    [
        (gemma4_arch, "google/gemma-4-31B-it", ("gemma4", "thinking")),
        (gemma4_arch, "google/gemma-4-E4B-it", ("gemma4", "nonthinking")),
        (
            llama_arch,
            "meta-llama/Llama-3.1-8B-Instruct",
            ("llama3", "standard"),
        ),
        (llama_arch, "meta-llama/Llama-3.2-1B-Instruct", ("llama3", "lite")),
        (llama_arch, "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", None),
        (
            llama_arch,
            "myorg/Llama-3.1-8B-Instruct-v3.2",
            ("llama3", "standard"),
        ),
        (llama_arch, "mlabonne/Meta-Llama-3-120B-Instruct", None),
        (
            gemma4_arch,
            "/hub/models--google--gemma-4-31B-it/snapshots/0e2b4e4b",
            ("gemma4", "thinking"),
        ),
        (
            gemma4_arch,
            "/hub/models--google--gemma-4-E2B-it/snapshots/0e2b4e4b",
            ("gemma4", "nonthinking"),
        ),
    ],
)
def test_architecture_names_the_checkpoints_tokenizer_family(
    arch: SupportedArchitecture,
    repo_id: str,
    family: tuple[str, str | None] | None,
) -> None:
    assert callable(arch.tokenizer_family)
    assert arch.tokenizer_family(_fake(repo_id=repo_id)) == family


@pytest.mark.parametrize(
    "architecture",
    [
        "beam_modulev3",
        "deepseekV41",
        "gemma4",
        "glm5_1",
        "inkling",
        "laguna",
        "lfm3",
        "mimo_v2_modulev3",
        "muse_glimmer",
        "nemotron_h_modulev3",
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
    processor = DecoderBatchProcessor(_fake(), runtime)
    processor.bind_input_tree(OrderedDict())
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
    mixed_batches: bool | None = False,
    ep_size: int = 2,
    **model_fields: Any,
) -> PipelineConfig:
    model = MAXModelConfig.model_construct(
        **{
            "model_path": "org/model",
            "device_specs": list(_DEVICES),
            "data_parallel_degree": 2,
            "kv_cache": KVCacheConfig(),
            **model_fields,
        }
    )
    runtime: dict[str, Any] = {
        # The configuration's, which the plan's values replace.
        "max_batch_size": None,
        "max_batch_input_tokens": 2048,
        "ep_size": ep_size,
        "reasoning_parser": reasoning_parser,
    }
    # ``None`` leaves the flag at its default, not set by name.
    if mixed_batches is not None:
        runtime["enable_spec_decode_mixed_batches"] = mixed_batches
    return PipelineConfig.model_construct(
        runtime=PipelineRuntimeConfig.model_construct(**runtime),
        models=ModelManifest({"main": model}),
        sampling=sampling,
        speculative=speculative,
    )


def test_export_builds_the_pipeline_for_an_artifact(tmp_path: Path) -> None:
    """The export builds with ``artifact_export`` set, on a copy."""
    config = _pipeline_config(sampling=SamplingConfig.model_construct())
    built: list[bool] = []

    class _Stop(Exception):
        pass

    def retrieve_factory(pipeline_config: PipelineConfig, **_: Any) -> None:
        built.append(pipeline_config.artifact_export)
        raise _Stop

    registry = export_module.PIPELINE_REGISTRY
    with (
        patch.object(
            ModelManifest,
            "main_architecture_name",
            new_callable=PropertyMock,
            return_value="LlamaForCausalLM",
        ),
        patch.object(registry, "retrieve_pipeline_task"),
        patch.object(registry, "architecture_for_config"),
        patch.object(registry, "retrieve_factory", retrieve_factory),
        pytest.raises(_Stop),
    ):
        export_module.export_pipeline(config, tmp_path / "pipeline")
    assert built == [True]
    assert not config.artifact_export


def test_export_refuses_a_window_that_holds_no_max_length_request() -> None:
    """The window's page-aligned floor must hold one request at max length."""
    plan = replace(_PLAN, planned_max_length=4100)
    for short in (0, 4096, 4100):
        with pytest.raises(ArtifactError, match="at least 4224"):
            _check_batch_context_window(
                replace(plan, planned_max_batch_total_tokens=short), 128
            )
    for fits in (None, 4224):
        _check_batch_context_window(
            replace(plan, planned_max_batch_total_tokens=fits), 128
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
        prefill_token_budget=None,
        max_batch_total_tokens=4096,
        devices=_DEVICES,
        parallel=EngineParallelConfig(
            dp=2, tp=1, ep=2, dcp=1, pcp=1, dcp_interleave=1
        ),
        capture=EngineCaptureConfig(max_batch_size=8, prefill_max_tokens=0),
        sampling=EngineSamplingConfig(
            needs_bitmask_constraints=True,
            enable_variable_logits=False,
            enable_penalties=True,
        ),
        speculative=None,
        prefix_caching=True,
    )
    off = pipeline_engine_config(config, _PLAN, prefix_caching=False)
    assert not off.prefix_caching


def test_a_speculative_expert_parallel_export_leaves_room_for_verify_rows() -> (
    None
):
    speculative = SpeculativeConfig.model_construct(
        speculative_method="dflash2",
        num_speculative_tokens=7,
        num_speculative_tokens_per_batch_size=None,
        adaptive_speculative_widths=None,
        adaptive_speculative_min_batch_size=1,
        num_speculative_tokens_mixed_batch=None,
    )
    config = _pipeline_config(
        sampling=SamplingConfig.model_construct(), speculative=speculative
    )
    # Eight rows of an anchor and seven drafts each beside the budget.
    assert pipeline_engine_config(config, _PLAN).prefill_token_budget == (
        2048 - 8 * 8
    )
    assert verify_step_prefill_budget(8448, 32, 7) == 8192
    named = pipeline_engine_config(config, _PLAN, prefill_token_budget=1024)
    assert named.prefill_token_budget == 1024
    # Without expert parallelism the capacity stands in, as before.
    unsharded = _pipeline_config(
        sampling=SamplingConfig.model_construct(),
        speculative=speculative,
        ep_size=1,
    )
    assert pipeline_engine_config(unsharded, _PLAN).prefill_token_budget is None
    with pytest.raises(ArtifactError, match="does not hold 8 verify rows"):
        verify_step_prefill_budget(64, 8, 7)


def _plan(n_devices: int) -> MemoryPlan:
    return MemoryPlan(
        planned_max_batch_size=8,
        footprint=0,
        planned_max_length=4096,
        device_specs=tuple(DeviceSpec.accelerator(i) for i in range(n_devices)),
        planned_max_batch_total_tokens=4096,
    )


@pytest.mark.parametrize(
    "n_devices, dp, ep, split, pcp, expected",
    [
        (4, 1, 4, None, False, (1, 4, 4, 1, 1, 1)),
        (4, 4, 4, None, False, (4, 1, 4, 1, 1, 1)),
        (8, 8, 8, None, False, (8, 1, 8, 1, 1, 1)),
        (8, 1, 1, (8, 128), False, (1, 8, 1, 8, 1, 128)),
        (8, 2, 8, (4, 1), False, (2, 4, 8, 4, 1, 1)),
    ],
)
def test_engine_config_records_the_parallel_layout(
    n_devices: int,
    dp: int,
    ep: int,
    split: tuple[int, int] | None,
    pcp: bool,
    expected: tuple[int, ...],
) -> None:
    # The decode context degree is the caches' split, not the flags'.
    config = _pipeline_config(
        sampling=SamplingConfig.model_construct(),
        ep_size=ep,
        data_parallel_degree=dp,
        decode_context_parallel_degree=split[0] if split else 1,
        decode_context_parallel_interleave=split[1] if split else 1,
        prefill_context_parallel=pcp,
    )
    parallel = pipeline_engine_config(
        config,
        _plan(n_devices),
        context_parallel=None
        if split is None
        else ContextParallel(degree=split[0], interleave=split[1]),
        prefill_context_parallel=EnginePrefillContextParallel(
            min_step_tokens=7168, min_rows_per_request=8, max_batch_context=1
        )
        if pcp
        else None,
    ).parallel
    assert (
        parallel.dp,
        parallel.tp,
        parallel.ep,
        parallel.dcp,
        parallel.pcp,
        parallel.dcp_interleave,
    ) == expected


def test_context_parallel_flags_without_split_caches_record_no_split() -> None:
    # An architecture that builds no DCP caches serves every position on
    # every rank, so its artifact records no decode context degree.
    config = _pipeline_config(
        sampling=SamplingConfig.model_construct(),
        decode_context_parallel_degree=2,
        prefill_context_parallel=True,
    )
    parallel = pipeline_engine_config(config, _plan(2)).parallel
    assert (parallel.dcp, parallel.pcp, parallel.dcp_interleave) == (1, 1, 1)


def test_engine_config_refuses_prefill_context_parallelism() -> None:
    config = _pipeline_config(
        sampling=SamplingConfig.model_construct(),
        ep_size=8,
        data_parallel_degree=2,
        decode_context_parallel_degree=4,
        decode_context_parallel_interleave=1,
        prefill_context_parallel=True,
    )
    with pytest.raises(
        ArtifactError, match="PCP is not supported on the artifact path"
    ):
        pipeline_engine_config(
            config,
            _plan(8),
            context_parallel=ContextParallel(degree=4, interleave=1),
            prefill_context_parallel=EnginePrefillContextParallel(
                min_step_tokens=7168,
                min_rows_per_request=8,
                max_batch_context=1,
            ),
        )


def test_engine_config_refuses_replicas_that_do_not_divide_the_devices() -> (
    None
):
    config = _pipeline_config(
        sampling=SamplingConfig.model_construct(), data_parallel_degree=3
    )
    with pytest.raises(ArtifactError, match="do not divide 2 devices"):
        pipeline_engine_config(config, _PLAN)


def test_engine_config_records_the_capture_limits() -> None:
    config = _pipeline_config(sampling=SamplingConfig.model_construct())
    # Decode capture covers the planned batch for every architecture: the
    # engine decides what it can capture.
    assert pipeline_engine_config(config, _PLAN).capture == (
        EngineCaptureConfig(max_batch_size=8, prefill_max_tokens=0)
    )
    # Prefill capture is capped at the prefill tokens a step takes: the
    # budget, or the compiled input capacity without one.
    assert pipeline_engine_config(
        config, _PLAN, prefill_capture_tokens=8192
    ).capture == EngineCaptureConfig(max_batch_size=8, prefill_max_tokens=2048)
    capped = pipeline_engine_config(
        config,
        _PLAN,
        prefill_token_budget=1024,
        prefill_capture_tokens=8192,
    )
    assert capped.prefill_token_budget == 1024
    assert capped.max_batch_input_tokens == 2048
    assert capped.capture.prefill_max_tokens == 1024


def test_engine_config_records_the_host_inputs() -> None:
    config = _pipeline_config(sampling=SamplingConfig.model_construct())
    pinned = {"spec_decode.max_k": CapturedHostInput("pinned", -1)}
    assert pipeline_engine_config(
        config, _PLAN, host_inputs=pinned
    ).capture == EngineCaptureConfig(
        max_batch_size=8, prefill_max_tokens=0, host_inputs=pinned
    )


def test_a_prefill_budget_past_the_input_capacity_is_refused() -> None:
    config = _pipeline_config(sampling=SamplingConfig.model_construct())
    with pytest.raises(ArtifactError, match="compiled input capacity 2048"):
        pipeline_engine_config(config, _PLAN, prefill_token_budget=4096)


def test_engine_config_records_the_capture_cache_bound() -> None:
    config = _pipeline_config(sampling=SamplingConfig.model_construct())
    assert pipeline_engine_config(
        config, _PLAN, capture_cache_bound="bucketed"
    ).capture == EngineCaptureConfig(
        max_batch_size=8, prefill_max_tokens=0, cache_bound="bucketed"
    )
    assert pipeline_engine_config(config, _PLAN).capture.cache_bound == (
        "aligned"
    )


@pytest.mark.parametrize("greedy", [False, True])
def test_engine_config_copies_the_speculative_settings(greedy: bool) -> None:
    config = _pipeline_config(
        sampling=SamplingConfig.model_construct(),
        speculative=SpeculativeConfig.model_construct(
            speculative_method="mtp",
            num_speculative_tokens=3,
            use_greedy_acceptance=greedy,
        ),
        reasoning_parser="gemma4",
    )
    assert pipeline_engine_config(config, _PLAN).speculative == (
        EngineSpeculativeConfig(
            speculative_method="mtp",
            num_speculative_tokens=3,
            draft_proposal="argmax",
            acceptance_mode="greedy" if greedy else "stochastic",
            reasoning_parser="gemma4",
            draft_query_len=1,
            proposer="sequential",
            verify_widths=(3,),
            mixed_verify_width=None,
        )
    )


@pytest.mark.parametrize(
    "method, proposer, draft_query_len",
    [
        ("mtp", "sequential", 1),
        ("eagle", "sequential", 1),
        ("dflash", "block", 7),
        ("dflash2", "block", 7),
        ("dspark", "block", 7),
    ],
)
def test_engine_config_records_the_proposer_facts(
    method: Any, proposer: str, draft_query_len: int
) -> None:
    config = _pipeline_config(
        sampling=SamplingConfig.model_construct(),
        speculative=SpeculativeConfig.model_construct(
            speculative_method=method, num_speculative_tokens=7
        ),
    )
    speculative = pipeline_engine_config(config, _PLAN).speculative
    assert speculative is not None
    assert speculative.speculative_method == method
    assert speculative.proposer == proposer
    assert speculative.draft_query_len == draft_query_len


def _ranges(small: int, large: int) -> list[VerifyWidthRange]:
    """Verifies ``small`` drafts up to batch size 4 and ``large`` past it."""
    return [
        VerifyWidthRange(batch_start=1, batch_end=4, num_tokens=small),
        VerifyWidthRange(batch_start=5, batch_end=8, num_tokens=large),
    ]


@pytest.mark.parametrize(
    "fields, mixed_batches, takes_prefill_rows, widths, mixed",
    [
        # Every step verifies the whole draft; no prefill rides a verify.
        ({}, False, False, (7,), None),
        # A graph that verifies beside prefill rows records the full mixed
        # width whatever the pipeline's flag: the serving engine decides.
        ({}, None, True, (7,), 7),
        ({}, False, True, (7,), 7),
        # The schedule narrows larger batches; a mixed step verifies up to
        # the depth.
        (
            {"num_speculative_tokens_per_batch_size": _ranges(7, 3)},
            False,
            True,
            (3, 7),
            7,
        ),
        # A mixed step that verifies its own width adds it.
        (
            {
                "num_speculative_tokens_per_batch_size": _ranges(7, 3),
                "num_speculative_tokens_mixed_batch": 2,
            },
            True,
            True,
            (2, 3, 7),
            2,
        ),
        # A mixed step that follows the schedule verifies up to the depth.
        ({}, True, True, (7,), 7),
        # A schedule that skips the drafter reaches width 0, which the
        # executor captures too.
        (
            {"num_speculative_tokens_per_batch_size": _ranges(5, 0)},
            False,
            False,
            (0, 5),
            None,
        ),
        # Adaptive widths are each reachable.
        (
            {"adaptive_speculative_widths": "2,4,7"},
            False,
            False,
            (2, 4, 7),
            None,
        ),
    ],
)
def test_engine_config_records_the_verify_widths_of_the_width_policy(
    fields: dict[str, Any],
    mixed_batches: bool | None,
    takes_prefill_rows: bool,
    widths: tuple[int, ...],
    mixed: int | None,
) -> None:
    speculative_fields: dict[str, Any] = {
        "speculative_method": "dflash2",
        "num_speculative_tokens": 7,
        "num_speculative_tokens_per_batch_size": None,
        "adaptive_speculative_widths": None,
        "adaptive_speculative_min_batch_size": 1,
        "num_speculative_tokens_mixed_batch": None,
        **fields,
    }
    config = _pipeline_config(
        sampling=SamplingConfig.model_construct(),
        speculative=SpeculativeConfig.model_construct(**speculative_fields),
        mixed_batches=mixed_batches,
    )
    facts = SpecGraphFacts(
        verify_takes_prefill_rows=takes_prefill_rows,
        verify_commits_long_rows=False,
        successor_replay=False,
    )
    speculative = pipeline_engine_config(
        config, _PLAN, verify_facts=facts
    ).speculative
    assert speculative is not None
    assert speculative.verify_widths == widths
    assert speculative.mixed_verify_width == mixed


@pytest.mark.parametrize(
    "flags, mixed",
    [
        # Unnamed: the graph's mixed verify width.
        ([], 7),
        # Off by name: still the graph's width, which the engine serves or
        # not by its own flag.
        (["--no-enable-spec-decode-mixed-batches"], 7),
    ],
)
def test_a_cli_export_records_the_graphs_mixed_width(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flags: list[str],
    mixed: int,
) -> None:
    from max._entrypoints.pipelines import cli_export

    args: list[Any] = []

    def from_args(pipeline_args: Any) -> Any:
        args.append(pipeline_args)
        return pipeline_args

    monkeypatch.setattr(PipelineConfig, "from_args", staticmethod(from_args))
    monkeypatch.setattr(
        export_module,
        "export_pipeline",
        lambda config, output, *_, **__: output,
    )
    # The placeholder repo is never fetched.
    with (
        patch("max.pipelines.lib.config.model_config.validate_hf_repo_access"),
        patch("max.pipelines.weights.hf_utils.validate_hf_repo_access"),
        patch(
            "max.pipelines.weights.hf_utils.generate_local_model_path",
            side_effect=lambda repo_id, revision=None: f"/fake/{repo_id}",
        ),
        patch("huggingface_hub.file_exists", return_value=False),
        patch(
            "max.pipelines.lib.config.model_config.GenerationConfig.from_pretrained",
            side_effect=OSError("placeholder repo"),
        ),
        patch(
            "max.pipelines.lib.config.model_config.load_huggingface_config",
            return_value=MagicMock(architectures=None),
        ),
    ):
        result = CliRunner().invoke(
            cli_export,
            [
                "--model-path",
                "org/model",
                "--output",
                str(tmp_path / "pipeline"),
                *flags,
            ],
        )
    assert result.exit_code == 0, result.output
    (pipeline_args,) = args
    config = PipelineConfig.model_construct(
        runtime=pipeline_args.runtime,
        models=ModelManifest(
            {
                "main": MAXModelConfig.model_construct(
                    model_path="org/model",
                    device_specs=list(_DEVICES),
                    data_parallel_degree=2,
                    kv_cache=KVCacheConfig(),
                )
            }
        ),
        sampling=SamplingConfig.model_construct(),
        speculative=SpeculativeConfig.model_construct(
            speculative_method="dflash2",
            num_speculative_tokens=7,
            num_speculative_tokens_per_batch_size=None,
            adaptive_speculative_widths=None,
            adaptive_speculative_min_batch_size=1,
            num_speculative_tokens_mixed_batch=None,
        ),
    )
    facts = SpecGraphFacts(
        verify_takes_prefill_rows=True,
        verify_commits_long_rows=False,
        successor_replay=False,
    )
    speculative = pipeline_engine_config(
        config, _PLAN, verify_facts=facts
    ).speculative
    assert speculative is not None
    assert speculative.mixed_verify_width == mixed


def test_a_cli_export_for_a_target_needs_a_batch_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import max.driver
    from max._entrypoints.pipelines import cli_export

    # Parsing --target switches the process to virtual devices.
    for name in (
        "set_virtual_device_api",
        "set_virtual_device_target_arch",
        "set_virtual_device_count",
    ):
        monkeypatch.setattr(max.driver, name, lambda *_: None)
    exported: list[Any] = []
    monkeypatch.setattr(
        export_module,
        "export_pipeline",
        lambda *args, **_: exported.append(args),
    )
    from max.pipelines import PipelineArgs

    def reached(**kwargs: Any) -> Any:
        raise RuntimeError(f"built with batch {kwargs['max_batch_size']}")

    monkeypatch.setattr(PipelineArgs, "from_flat_kwargs", staticmethod(reached))
    args = [
        "--model-path",
        "org/model",
        "--output",
        str(tmp_path / "pipeline"),
        "--target",
        "cuda:sm_100a",
    ]
    result = CliRunner().invoke(cli_export, args)
    assert result.exit_code != 0
    assert "--max-batch-size" in result.output
    # With a batch size the export goes on to build the pipeline.
    result = CliRunner().invoke(cli_export, [*args, "--max-batch-size", "8"])
    assert str(result.exception) == "built with batch 8", result.output
    assert not exported


def test_engine_config_records_the_verify_facts() -> None:
    config = _pipeline_config(
        sampling=SamplingConfig.model_construct(),
        speculative=SpeculativeConfig.model_construct(
            speculative_method="mtp", num_speculative_tokens=3
        ),
    )
    # Mixed values, so a swapped field cannot pass.
    facts = SpecGraphFacts(
        verify_takes_prefill_rows=True,
        verify_commits_long_rows=False,
        successor_replay=True,
    )
    speculative = pipeline_engine_config(
        config, _PLAN, verify_facts=facts
    ).speculative
    assert speculative is not None
    assert speculative.verify_takes_prefill_rows
    assert not speculative.verify_commits_long_rows
    assert speculative.successor_replay


def test_engine_config_records_the_reversed_verify_facts() -> None:
    config = _pipeline_config(
        sampling=SamplingConfig.model_construct(),
        speculative=SpeculativeConfig.model_construct(
            speculative_method="mtp", num_speculative_tokens=3
        ),
    )
    # The reverse mix, so the two keys cannot be swapped anywhere.
    facts = SpecGraphFacts(
        verify_takes_prefill_rows=False,
        verify_commits_long_rows=True,
        successor_replay=True,
    )
    speculative = pipeline_engine_config(
        config, _PLAN, verify_facts=facts
    ).speculative
    assert speculative is not None
    assert not speculative.verify_takes_prefill_rows
    assert speculative.verify_commits_long_rows
    assert speculative.successor_replay


def test_engine_config_without_verify_facts_reads_all_false() -> None:
    config = _pipeline_config(
        sampling=SamplingConfig.model_construct(),
        speculative=SpeculativeConfig.model_construct(
            speculative_method="mtp", num_speculative_tokens=3
        ),
    )
    speculative = pipeline_engine_config(config, _PLAN).speculative
    assert speculative is not None
    assert not speculative.verify_takes_prefill_rows
    assert not speculative.verify_commits_long_rows
    assert not speculative.successor_replay


def test_verify_facts_read_the_spec_pipeline_model() -> None:
    class _SpecModel(SpecDecodePipelineModel):
        def __init__(self) -> None:
            # The export reads only the three fact properties.
            pass

        @property
        def verify_takes_prefill_rows(self) -> bool:
            return True

        @property
        def verify_commits_long_rows(self) -> bool:
            return False

        @property
        def successor_replay(self) -> bool:
            return True

    class _SpecPipeline(TextGenerationPipelineInterface[Any]):
        _pipeline_model = _SpecModel()

        @property
        def max_batch_size(self) -> int:
            return 1

        @property
        def kv_manager(self) -> Any:
            raise NotImplementedError

        @property
        def pipeline_config(self) -> Any:
            raise NotImplementedError

        @property
        def tokenizer(self) -> Any:
            raise NotImplementedError

        def execute(self, inputs: Any) -> Any:
            raise NotImplementedError

        def release(self, request_id: Any) -> None:
            raise NotImplementedError

    class _PlainPipeline(TextGenerationPipelineInterface[Any]):
        _pipeline_model = cast(PipelineModelWithKVCache[Any], object())

        @property
        def max_batch_size(self) -> int:
            return 1

        @property
        def kv_manager(self) -> Any:
            raise NotImplementedError

        @property
        def pipeline_config(self) -> Any:
            raise NotImplementedError

        @property
        def tokenizer(self) -> Any:
            raise NotImplementedError

        def execute(self, inputs: Any) -> Any:
            raise NotImplementedError

        def release(self, request_id: Any) -> None:
            raise NotImplementedError

    assert _verify_facts(_SpecPipeline()) == SpecGraphFacts(
        verify_takes_prefill_rows=True,
        verify_commits_long_rows=False,
        successor_replay=True,
    )
    # A pipeline whose model is not a fused spec-decode one has no facts.
    assert _verify_facts(_PlainPipeline()) is None


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
    sparse = MSAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=4,
        head_dim=64,
        num_layers=1,
        devices=[DeviceRef.CPU()],
    )
    index = MSAIndexKVCacheParams(
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
            "sparse": sparse,
            "indexer": index,
            "latent": mla,
            "state": state,
        }
    )
    layout = pipeline_kv_cache(params)
    assert layout.page_size == 128
    assert [(cache.name, cache.kind) for cache in layout.caches] == [
        ("target.sliding_attention", "mha"),
        ("target.full_attention", "mha"),
        ("sparse", "msa"),
        ("indexer", "msa_index"),
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
        speculative_method=None,
        num_draft_tokens=0,
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

    assert layout.caches[3] == MSAIndexKVCache(
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
        speculative_method=None,
        num_draft_tokens=0,
        leaves=(full_leaf("indexer"),),
    )
    assert layout.caches[4] == MLAKVCache(
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
        speculative_method=None,
        num_draft_tokens=0,
        leaves=(full_leaf("latent"),),
        # An unscaled BF16 latent.
        sparse_kv_form="plain",
    )
    assert layout.caches[5] == RecurrentStateCache(
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


def test_kv_cache_records_the_unread_dispatch_of_named_mla_caches() -> None:
    mla = MLAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=576,
        num_layers=1,
        devices=[DeviceRef.CPU()],
        num_q_heads=16,
    )
    params = MultiKVCacheParams.from_params(
        {
            "target": MultiKVCacheParams.from_params(
                {"mla": mla, "indexer": mla}
            ),
            "draft": MultiKVCacheParams.from_params(
                {"mla": mla, "indexer": mla}
            ),
        }
    )
    layout = pipeline_kv_cache(params, ("indexer",))
    assert [
        (cache.name, isinstance(cache, MLAKVCache) and cache.dispatch_unread)
        for cache in layout.caches
    ] == [
        ("target.mla", False),
        ("target.indexer", True),
        ("draft.mla", False),
        ("draft.indexer", True),
    ]


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


def _dcp_latent(degree: int = 4) -> MLAKVCacheParams:
    return MLAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=576,
        num_layers=1,
        devices=[DeviceRef.GPU(i) for i in range(8)],
        num_q_heads=64,
        context_parallel=KVCacheContextParallel(degree=degree, interleave=128),
    )


def test_kv_cache_records_how_a_cache_splits_positions() -> None:
    unsplit = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=8,
        head_dim=64,
        num_layers=1,
        devices=[DeviceRef.GPU(i) for i in range(8)],
    )
    params = MultiKVCacheParams.from_params(
        {"latent": _dcp_latent(), "plain": unsplit}
    )
    latent, plain = pipeline_kv_cache(params).caches
    assert isinstance(latent, MLAKVCache) and isinstance(plain, MHAKVCache)
    assert latent.context_parallel == ContextParallel(degree=4, interleave=128)
    assert plain.context_parallel is None


@pytest.mark.parametrize("degree", [1, 3])
def test_a_context_parallel_degree_must_split_a_replica(degree: int) -> None:
    with pytest.raises(ValueError, match="context parallel degree"):
        _dcp_latent(degree)


def _pcp_config(
    prefill_context_parallel: bool, decode_context_parallel_degree: int = 1
) -> PipelineConfig:
    config = _pipeline_config(sampling=SamplingConfig.model_construct())
    model = config.model.model_copy(
        update={
            "prefill_context_parallel": prefill_context_parallel,
            "decode_context_parallel_degree": decode_context_parallel_degree,
        }
    )
    return config.model_copy(update={"models": ModelManifest({"main": model})})


def test_engine_config_records_prefill_context_parallel_over_split_caches() -> (
    None
):
    split = pipeline_kv_cache(_dcp_latent())
    unsplit = KVCacheLayout(page_size=128, caches=())
    assert _prefill_context_parallel(
        _pcp_config(True), split
    ) == EnginePrefillContextParallel(
        min_step_tokens=7168, min_rows_per_request=8, max_batch_context=262144
    )
    # An architecture that splits no cache builds no PCP path.
    assert _prefill_context_parallel(_pcp_config(True), unsplit) is None
    assert _prefill_context_parallel(_pcp_config(False), split) is None
    assert _prefill_context_parallel(_pcp_config(True), None) is None


@pytest.mark.parametrize(
    ("config", "warns"),
    [
        (_pcp_config(False, decode_context_parallel_degree=4), True),
        (_pcp_config(True), True),
        (_pcp_config(False), False),
    ],
)
def test_export_warns_when_no_cache_splits_the_positions_asked_for(
    config: PipelineConfig, warns: bool, caplog: pytest.LogCaptureFixture
) -> None:
    unsplit = KVCacheLayout(page_size=128, caches=())
    _warn_unhonored_context_parallel(config, "LlamaForCausalLM", unsplit)
    assert ("ignores the context parallel flags" in caplog.text) == warns
    caplog.clear()
    _warn_unhonored_context_parallel(
        config, "GlmMoeDsaForCausalLM", pipeline_kv_cache(_dcp_latent())
    )
    assert "ignores the context parallel flags" not in caplog.text


def _fused_bounds_processor(target: Any) -> _BatchProcessor:
    config = _fake(
        speculative=SpeculativeConfig.model_construct(
            speculative_method="mtp", num_speculative_tokens=3
        ),
        model=_fake(
            data_parallel_degree=1,
            huggingface_config=_fake(
                get_text_config=lambda: _fake(vocab_size=64)
            ),
        ),
    )
    runtime = BatchProcessorRuntime(
        pipeline_config=config,
        devices=[],
        return_logits=ReturnLogits.VARIABLE,
        max_batch_size=4,
        max_seq_len=256,
        max_batch_input_tokens=64,
    )
    processor = object.__new__(_BatchProcessor)
    processor.runtime = runtime
    processor.__dict__["_target_batching"] = target
    processor.bind_input_tree(OrderedDict())
    return processor


def test_fused_export_bounds_include_the_targets_model_specific_dims() -> None:
    target = _fake(max_dims=Mock(return_value={"model_rows": 37}))
    processor = _fused_bounds_processor(target)
    kv = _fake(page_size=128, num_draft_tokens=3)
    model = _fake(batch_processor=processor, kv_params=kv)
    layout = as_layout(
        TensorType(
            DType.float32,
            ["merged_total_seq_len", "model_rows"],
            device=DeviceRef.CPU(),
        )
    )
    decoder = _fake(in_layouts=[layout])
    assert batch_processor_dims(model, decoder) == BatchProcessorDims(
        {"merged_total_seq_len": 76, "model_rows": 37}
    )
    target.max_dims.assert_called_once_with(["model_rows"], kv)


def test_fused_export_preserves_the_targets_unbounded_dim_error() -> None:
    target = _fake(
        max_dims=Mock(side_effect=ValueError("target cannot bound model_rows"))
    )
    processor = _fused_bounds_processor(target)
    model = _fake(batch_processor=processor, kv_params=_fake(page_size=128))
    decoder = _fake(
        in_layouts=[
            as_layout(
                TensorType(
                    DType.float32, ["model_rows"], device=DeviceRef.CPU()
                )
            )
        ]
    )
    with pytest.raises(ValueError, match="target cannot bound model_rows"):
        batch_processor_dims(model, decoder)
