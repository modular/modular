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
"""Checks registered EAGLE construction and its initial and recurrent calls."""

from __future__ import annotations

import pickle
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from max import tree
from max.driver import CPU
from max.dtype import DType
from max.graph import BufferType, DeviceRef, Graph, TensorType
from max.graph.weights import WeightsFormat
from max.nn import Linear
from max.nn import Module as ModuleV3
from max.nn.kv_cache import MHAKVCacheParams
from max.nn.outputs import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.eagle_llama3.arch import (
    eagle3_llama_arch,
    eagle_llama_arch,
)
from max.pipelines.architectures.eagle_llama3.eagle_llama3 import (
    EagleLlama3,
    EagleLlama3Draft,
)
from max.pipelines.architectures.eagle_llama3.model_config import (
    EagleLlama3Config,
    eagle_proposer,
)
from max.pipelines.architectures.llama3.llama3 import Llama3
from max.pipelines.lib import GraphRole
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.lib.pipeline_variants import spec_decode_pipeline_model
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    SpecDecodePipelineModel,
    _builds_module,
    _module_builder,
    _proposer,
    spec_decode_architecture,
)
from max.pipelines.lib.registration_draft import (
    RegisteredDraftBuilder,
    TargetHiddenStates,
)
from max.pipelines.speculative import proposers, spec_adapters
from max.pipelines.speculative.config import SpeculativeConfig
from max.pipelines.speculative.draft_contracts import (
    DraftPhase,
    EagleDraftInputs,
    EagleDraftOutputs,
)
from max.pipelines.speculative.driver import DraftStepInput
from max.pipelines.speculative.eagle import EagleDraftAdapter
from max.tensor import Tensor, defaults


def _config() -> EagleLlama3Config:
    cpu = DeviceRef.CPU()
    return EagleLlama3Config(
        hidden_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=1,
        rope_theta=10000.0,
        rope_scaling_params=None,
        max_seq_len=64,
        intermediate_size=64,
        interleaved_rope_weights=False,
        vocab_size=48,
        dtype=DType.float32,
        model_quantization_encoding=None,
        quantization_config=None,
        kv_params=MHAKVCacheParams(
            dtype=DType.float32,
            n_kv_heads=1,
            head_dim=16,
            num_layers=1,
            devices=[cpu],
        ),
        attention_multiplier=0.25,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        rms_norm_eps=1e-5,
        clip_qkv=None,
        norm_method="rms_norm",
        devices=[cpu],
        return_logits=ReturnLogits.LAST_TOKEN,
        return_hidden_states=ReturnHiddenStates.LAST,
    )


def test_registered_eagle_pairs_without_a_config_builder() -> None:
    assert eagle_llama_arch.modules == {GraphRole.DRAFT: EagleLlama3Draft}
    assert eagle_llama_arch.pipeline_model is None
    assert not hasattr(eagle_llama_arch.config, "build_module")
    assert _builds_module(eagle_llama_arch, GraphRole.DRAFT)
    assert not _builds_module(eagle_llama_arch, GraphRole.DECODER)
    assert isinstance(
        _module_builder(eagle_llama_arch, GraphRole.DRAFT),
        RegisteredDraftBuilder,
    )
    assert (
        spec_decode_architecture(
            "LlamaForCausalLM", "eagle", "LlamaForCausalLM"
        )
        is not None
    )
    assert (
        spec_decode_architecture(
            "LlamaForCausalLM", "mtp", eagle_llama_arch.name
        )
        is None
    )
    assert (
        spec_decode_architecture(
            eagle_llama_arch.name, "eagle", eagle_llama_arch.name
        )
        is None
    )
    # EAGLE3 checkpoints construct through the same registration.
    assert eagle3_llama_arch.modules == eagle_llama_arch.modules
    assert eagle3_llama_arch.draft is eagle_llama_arch.draft
    assert _builds_module(eagle3_llama_arch, GraphRole.DRAFT)


def test_draft_role_requires_its_registration() -> None:
    with pytest.raises(ValueError, match="declared together"):
        replace(eagle_llama_arch, draft=None)
    with pytest.raises(ValueError, match="declared together"):
        replace(eagle_llama_arch, modules={})


def test_draft_role_takes_either_module_api() -> None:
    assert replace(
        eagle_llama_arch, modules={GraphRole.DRAFT: _ModuleV3Draft}
    ).modules == {GraphRole.DRAFT: _ModuleV3Draft}
    not_a_module: Any = object
    with pytest.raises(TypeError, match="module class"):
        replace(eagle_llama_arch, modules={GraphRole.DRAFT: not_a_module})


def test_registered_draft_reads_the_target_layers_its_config_names() -> None:
    assert eagle_llama_arch.draft is not None
    hidden_states = eagle_llama_arch.draft.target_hidden_states
    speculative = SpeculativeConfig(
        speculative_method="eagle", num_speculative_tokens=2
    )
    assert hidden_states(speculative, SimpleNamespace()) == TargetHiddenStates(
        ReturnHiddenStates.ALL_NORMALIZED
    )
    assert hidden_states(
        speculative, SimpleNamespace(target_layer_ids=[1, 3])
    ) == TargetHiddenStates(ReturnHiddenStates.SELECTED_LAYERS, (1, 3))


def test_registered_draft_survives_worker_serialization() -> None:
    restored = pickle.loads(pickle.dumps(eagle_llama_arch))
    assert restored.modules == eagle_llama_arch.modules
    assert restored.draft == eagle_llama_arch.draft
    assert restored.draft is not None
    assert restored.draft.strict_weights is True
    assert restored.pipeline_model is None


def test_registered_draft_binds_the_targets_actual_parameters() -> None:
    config = _config()
    target, draft = Llama3(config), EagleLlama3Draft(config)
    assert eagle_llama_arch.draft is not None
    eagle_llama_arch.draft.bind_shared_modules(target, draft)
    assert draft.embed_tokens is target.embed_tokens
    assert draft.lm_head is target.lm_head
    with pytest.raises(ValueError, match="shared module"):
        eagle_llama_arch.draft.bind_shared_modules(Mock(spec=[]), draft)


def test_builder_rejects_the_wrong_method_before_loading() -> None:
    context: Any = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            speculative=SimpleNamespace(speculative_method="mtp")
        )
    )
    with pytest.raises(ValueError, match="requires speculative method 'eagle'"):
        RegisteredDraftBuilder(eagle_llama_arch).build_module(context)


@pytest.mark.parametrize("strict_weights", [None, False, True])
def test_builder_validates_owned_weights_and_drops_target_aliases(
    monkeypatch: pytest.MonkeyPatch,
    strict_weights: bool | None,
) -> None:
    config = _config()
    template = EagleLlama3Draft(config)
    rng = np.random.default_rng(0)
    state: Any = {
        name: rng.normal(0, 0.1, weight.shape.static_dims).astype(np.float32)
        for name, weight in template.raw_state_dict().items()
        if name != "fc.bias"
    }
    assert eagle_llama_arch.draft is not None
    registration = replace(
        eagle_llama_arch.draft,
        configure=lambda context, weights: config,
    )
    if strict_weights is not None:
        registration = replace(registration, strict_weights=strict_weights)
    arch = replace(eagle_llama_arch, draft=registration)
    from_module = (
        "max.pipelines.lib.registration_draft.parse_state_dict_from_weights"
    )
    monkeypatch.setattr(from_module, lambda *args, **kwargs: state)
    checkpoint = SimpleNamespace(huggingface_config=object())
    context: Any = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            speculative=SimpleNamespace(speculative_method="eagle"),
            draft_model=checkpoint,
        ),
        weights=object(),
        adapter=None,
    )
    loaded: list[dict[str, object]] = []
    monkeypatch.setattr(
        EagleLlama3Draft,
        "load_state_dict",
        lambda self, weights, **kwargs: loaded.append(weights),
    )
    module, returned = RegisteredDraftBuilder(arch).build_module(context)
    assert isinstance(module, EagleLlama3Draft)
    assert returned is config
    assert loaded and not any(
        key.startswith(("embed_tokens.", "lm_head.")) for key in loaded[0]
    )
    state["unexpected.weight"] = object()
    if strict_weights is not False:
        with pytest.raises(
            ValueError, match=r"unexpected weights.*unexpected\.weight"
        ):
            RegisteredDraftBuilder(arch).build_module(context)
    else:
        RegisteredDraftBuilder(arch).build_module(context)
    del state["unexpected.weight"]
    del state["fc.weight"]
    with pytest.raises(ValueError, match=r"fc\.weight"):
        RegisteredDraftBuilder(arch).build_module(context)


def test_adapter_routes_initial_conditioning_then_draft_carry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    draft = EagleLlama3Draft(config)
    calls: list[tuple[EagleDraftInputs, DraftPhase]] = []

    def compute(
        inputs: EagleDraftInputs, *, phase: DraftPhase
    ) -> EagleDraftOutputs:
        calls.append((inputs, phase))
        return EagleDraftOutputs(
            logits=inputs.tokens, hidden_states=inputs.hidden_states
        )

    monkeypatch.setattr(draft, "draft_forward", compute)
    adapter = EagleDraftAdapter(draft, config.hidden_size)
    placeholders: list[Any] = [Mock() for _ in range(6)]
    tokens, offsets, count, target_hidden, carry, kv = placeholders
    batch: Any = SimpleNamespace(
        query_offsets_per_dev=[offsets],
        return_n_logits=count,
        draft_kv_collections=[kv],
    )
    adapter.prefill(batch, tokens, target_hidden)
    proposed = adapter.step(
        batch, DraftStepInput(tokens=tokens, hidden=[carry]), 1
    )
    assert calls[0][0].hidden_states is target_hidden
    assert calls[0][1] is DraftPhase.INITIAL
    assert calls[1][0].hidden_states is carry
    assert calls[1][1] is DraftPhase.STEP
    assert proposed.hidden == [carry]
    assert draft.return_logits is ReturnLogits.LAST_TOKEN
    assert draft.return_hidden_states is ReturnHiddenStates.LAST


@pytest.mark.parametrize("phase", [DraftPhase.INITIAL, DraftPhase.STEP])
def test_computation_matches_legacy_and_keeps_output_modes(
    phase: DraftPhase,
) -> None:
    """Each phase emits the logits and hidden states the legacy call does
    under that phase's output modes, and restores the draft's own."""
    config = _config()
    draft = EagleLlama3Draft(config)
    # Loading names each weight, which the graph needs to add them.
    draft.load_state_dict(
        {
            name: np.zeros(weight.shape.static_dims, dtype=np.float32)
            for name, weight in draft.raw_state_dict().items()
        },
        weight_alignment=1,
    )
    kv_types = config.kv_params.get_symbolic_inputs()
    cpu = DeviceRef.CPU()
    rows = 3 if phase is DraftPhase.INITIAL else 1
    types = [
        TensorType(DType.int64, [rows], cpu),
        TensorType(DType.uint32, [2], cpu),
        TensorType(DType.int64, [1], cpu),
        TensorType(DType.float32, [rows, 32], cpu),
        *tree.leaves(kv_types),
    ]
    with Graph("registered_eagle", input_types=types) as graph:
        tokens, offsets, count, hidden = (v.tensor for v in graph.inputs[:4])
        kv_values = [
            v.buffer if isinstance(t, BufferType) else v.tensor
            for v, t in zip(
                graph.inputs[4:], tree.leaves(kv_types), strict=True
            )
        ]
        (kv,) = config.kv_params.unflatten_kv_inputs(iter(kv_values))
        draft.return_logits = (
            ReturnLogits.VARIABLE
            if phase is DraftPhase.INITIAL
            else ReturnLogits.LAST_TOKEN
        )
        draft.return_hidden_states = (
            ReturnHiddenStates.ALL
            if phase is DraftPhase.INITIAL
            else ReturnHiddenStates.LAST
        )
        legacy = EagleLlama3.__call__(draft, tokens, kv, count, offsets, hidden)
        draft.return_logits = ReturnLogits.LAST_TOKEN
        draft.return_hidden_states = ReturnHiddenStates.LAST
        outputs = draft.draft_forward(
            EagleDraftInputs(
                tokens=tokens,
                input_row_offsets=offsets,
                return_n_logits=count,
                hidden_states=hidden,
                kv=kv,
            ),
            phase=phase,
        )
        (legacy_hidden,) = legacy.hidden_states.tensors()
        legacy_logits = legacy.logits[
            "logits" if phase is DraftPhase.INITIAL else "next_token_logits"
        ]
        assert outputs.logits.type == legacy_logits.type
        assert outputs.hidden_states.type == legacy_hidden.type
    assert draft.return_logits is ReturnLogits.LAST_TOKEN
    assert draft.return_hidden_states is ReturnHiddenStates.LAST


class _ModuleV3Draft(ModuleV3[[Tensor], Tensor]):
    """A ModuleV3 draft that records the mesh it was constructed on."""

    def __init__(self, config: object) -> None:
        super().__init__()
        del config
        self.mesh = defaults()[1]
        self.proj = Linear(4, 4)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


def _registered(arch: Any, **changes: Any) -> Any:
    assert arch.draft is not None
    return replace(arch, draft=replace(arch.draft, **changes))


@pytest.mark.parametrize("strict_weights", [None, False, True])
def test_registered_modulev3_draft_compiles_with_its_owned_weights(
    monkeypatch: pytest.MonkeyPatch,
    strict_weights: bool | None,
) -> None:
    config = object()
    arch = _registered(
        replace(eagle_llama_arch, modules={GraphRole.DRAFT: _ModuleV3Draft}),
        configure=lambda context, weights: config,
        shared_modules=("embed_tokens",),
        optional_weights=frozenset({"proj.bias"}),
    )
    if strict_weights is not None:
        arch = _registered(arch, strict_weights=strict_weights)
    weight = np.zeros((4, 4), dtype=np.float32)
    state: Any = {"proj.weight": weight, "embed_tokens.weight": weight}
    monkeypatch.setattr(
        "max.pipelines.lib.registration_draft.parse_state_dict_from_weights",
        lambda *args, **kwargs: state,
    )
    context: Any = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            speculative=SimpleNamespace(speculative_method="eagle"),
            draft_model=SimpleNamespace(huggingface_config=object()),
            model=SimpleNamespace(data_parallel_degree=1),
        ),
        weights=object(),
        adapter=None,
        kv_params=SimpleNamespace(devices=[DeviceRef.CPU()]),
    )
    builder = RegisteredDraftBuilder(arch)
    module, returned = builder.build_module(context)
    assert isinstance(module, _ModuleV3Draft)
    assert returned is config
    assert module.mesh.axis_names == ("tp",)
    assert module.mesh.devices == (CPU(),)
    assert builder.checkpoint_weights(context, config) == {
        "proj.weight": weight
    }
    state["unexpected.weight"] = object()
    if strict_weights is not False:
        with pytest.raises(
            ValueError, match=r"unexpected weights.*unexpected\.weight"
        ):
            RegisteredDraftBuilder(arch).build_module(context)
    else:
        RegisteredDraftBuilder(arch).build_module(context)
    del state["unexpected.weight"]
    del state["proj.weight"]
    with pytest.raises(ValueError, match=r"proj\.weight"):
        RegisteredDraftBuilder(arch).build_module(context)


def test_proposers_rank_target_then_registration_then_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from_adapter, from_draft, from_method = object(), object(), object()
    calls: list[dict[str, object]] = []

    def propose(module: object, config: object, **kwargs: object) -> Any:
        calls.append(kwargs)
        return from_draft

    adapter: list[object | None] = [None]
    monkeypatch.setattr(
        spec_adapters, "block_proposer", lambda *args: adapter[0]
    )
    monkeypatch.setattr(
        proposers, "block_proposer", lambda *args, **kwargs: from_method
    )
    arch = _registered(
        eagle_llama_arch, method="dspark", create_proposer=propose
    )
    speculative = SpeculativeConfig(
        speculative_method="dspark", num_speculative_tokens=3
    )
    text_config = object()
    hf_config: Any = SimpleNamespace(get_text_config=lambda: text_config)
    verify = SimpleNamespace(takes_distributed_inputs=True)

    def select(draft_arch: Any) -> object:
        return _proposer(
            "Target", draft_arch, verify, None, None, speculative, hf_config
        )

    assert select(arch) is from_draft
    assert calls == [
        {
            "speculative": speculative,
            "takes_distributed_inputs": True,
            "target_text_config": text_config,
        }
    ]
    assert select(_registered(arch, create_proposer=None)) is from_method
    adapter[0] = from_adapter
    assert select(arch) is from_adapter


def test_eagle_refuses_a_distributed_target() -> None:
    speculative = SpeculativeConfig(
        speculative_method="eagle", num_speculative_tokens=2
    )
    target_text_config: Any = object()
    with pytest.raises(ValueError, match="distributed target"):
        eagle_proposer(
            EagleLlama3Draft(_config()),
            _config(),
            speculative=speculative,
            takes_distributed_inputs=True,
            target_text_config=target_text_config,
        )


class _TargetBuilt(Exception):
    pass


def _build_until_the_target(
    monkeypatch: pytest.MonkeyPatch, **changes: Any
) -> tuple[SpecDecodePipelineModel, list[ModelLoadContext]]:
    """Builds a registered EAGLE pairing up to its target's construction.

    Returns the model and the load contexts its target was built from.
    """
    config = _config()
    arch = _registered(
        eagle_llama_arch,
        configure=lambda context, weights: config,
        **changes,
    )
    template = EagleLlama3Draft(config)
    state: Any = {
        name: np.zeros(weight.shape.static_dims, dtype=np.float32)
        for name, weight in template.raw_state_dict().items()
    }
    monkeypatch.setattr(
        "max.pipelines.lib.registration_draft.parse_state_dict_from_weights",
        lambda *args, **kwargs: state,
    )
    monkeypatch.setattr(
        EagleLlama3Draft, "load_state_dict", lambda self, *args, **kw: None
    )
    target_contexts: list[ModelLoadContext] = []

    def build_target(context: ModelLoadContext) -> Any:
        target_contexts.append(context)
        raise _TargetBuilt

    def module_builder(arch: Any, role: GraphRole = GraphRole.DECODER) -> Any:
        if role == GraphRole.DRAFT:
            return RegisteredDraftBuilder(arch)
        return SimpleNamespace(build_module=build_target)

    for name, value in {
        "load_weights": lambda paths: object(),
        "weights_format": lambda paths: WeightsFormat.safetensors,
        "_builds_modulev3": lambda arch: False,
        "_module_builder": module_builder,
    }.items():
        monkeypatch.setattr(spec_decode_pipeline_model, name, value)
    model = object.__new__(SpecDecodePipelineModel)
    text_config = SimpleNamespace(
        num_hidden_layers=2, hidden_size=32, vocab_size=48
    )
    pipeline_config: Any = SimpleNamespace(
        speculative=SpeculativeConfig(
            speculative_method="eagle", num_speculative_tokens=2
        ),
        draft_model=SimpleNamespace(
            resolved_weight_paths=list, huggingface_config=object()
        ),
        model=SimpleNamespace(
            huggingface_config=SimpleNamespace(
                get_text_config=lambda: text_config
            )
        ),
    )
    pairing: Any = SimpleNamespace(
        target=SimpleNamespace(pipeline_model=object), draft=arch
    )
    model.pipeline_config = pipeline_config
    model.pairing = pairing
    arch_config = SimpleNamespace(
        kv_params=SimpleNamespace(children={"target": config.kv_params})
    )
    monkeypatch.setattr(model, "arch_config_as", lambda cls: arch_config)
    monkeypatch.setattr(
        ModelLoadContext,
        "from_pipeline_model",
        classmethod(
            lambda cls, pipeline_model: ModelLoadContext(
                pipeline_config=pipeline_model.pipeline_config,
                weights=Mock(),
                adapter=None,
                memory_plan=Mock(),
                return_logits=ReturnLogits.LAST_TOKEN,
                return_hidden_states=ReturnHiddenStates.NONE,
            )
        ),
    )
    with pytest.raises(_TargetBuilt):
        model._build_modules(Mock())
    return model, target_contexts


def test_registered_drafter_export_reaches_the_pipeline_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exported = object()
    model, _ = _build_until_the_target(
        monkeypatch,
        drafter_export=lambda context, draft_config: (
            exported if isinstance(draft_config, EagleLlama3Config) else None
        ),
    )
    assert model.drafter_export is exported


def test_the_target_returns_and_captures_what_the_registration_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One value sets both the target's return mode and its captured
    layers, and the pairing checks those layers against the target."""
    reads = TargetHiddenStates(ReturnHiddenStates.SELECTED_LAYERS, (1,))
    _, (context,) = _build_until_the_target(
        monkeypatch, target_hidden_states=lambda speculative, config: reads
    )
    assert context.return_hidden_states is ReturnHiddenStates.SELECTED_LAYERS
    assert context.target_layer_ids == (1,)
    out_of_range = replace(reads, layer_ids=(2,))
    with pytest.raises(ValueError, match="must index the target's layers"):
        _build_until_the_target(
            monkeypatch,
            target_hidden_states=lambda speculative, config: out_of_range,
        )
