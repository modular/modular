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
"""Checks the DeepseekV3 NextN head pairs and drafts as a registered draft."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import NonCallableMock

import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightsFormat
from max.nn.kv_cache import MLAKVCacheParams
from max.nn.legacy.transformer import DecoderOutputs
from max.nn.outputs import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.deepseekV3_nextn.deepseekV3_nextn import (
    DeepseekV3NextN,
)
from max.pipelines.architectures.deepseekV3_nextn.model_config import (
    DeepseekV3NextNConfig,
)
from max.pipelines.architectures.mtp_deepseekV3 import (
    MTPDeepseekV3DraftArchConfig,
    mtp_deepseekV3_draft_arch,
    mtp_draft_registration,
)
from max.pipelines.lib import GraphRole
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    _module_builder,
    _proposer,
    spec_decode_architecture,
)
from max.pipelines.lib.registration_draft import (
    RegisteredDraftBuilder,
    TargetHiddenStates,
)
from max.pipelines.speculative import proposers
from max.pipelines.speculative.config import SpeculativeConfig
from max.pipelines.speculative.mla import MTPProposer

TARGET = "DeepseekV3ForCausalLM"
NUM_LAYERS = 61
NUM_EXPERTS = 32


def test_the_nextn_head_is_a_registered_mtp_draft() -> None:
    assert mtp_deepseekV3_draft_arch.modules == {
        GraphRole.DRAFT: DeepseekV3NextN
    }
    assert mtp_deepseekV3_draft_arch.draft is mtp_draft_registration
    assert mtp_draft_registration.method == "mtp"
    assert mtp_draft_registration.shared_modules == ("embed_tokens", "lm_head")
    assert mtp_draft_registration.create_proposer is None
    assert not hasattr(MTPDeepseekV3DraftArchConfig, "build_module")
    assert isinstance(
        _module_builder(mtp_deepseekV3_draft_arch, GraphRole.DRAFT),
        RegisteredDraftBuilder,
    )
    # The target's checkpoint ships the head, so it pairs without a draft.
    assert spec_decode_architecture(TARGET, "mtp", None) is not None


class _Head:
    """What the MLA proposer reads of a NextN head."""

    return_hidden_states = ReturnHiddenStates.LAST
    return_logits = ReturnLogits.LAST_TOKEN
    config = SimpleNamespace(hidden_size=64)

    def __call__(self, *args: object, **kwargs: object) -> DecoderOutputs:
        raise NotImplementedError


def test_the_head_drafts_through_the_mla_proposer() -> None:
    """The method's proposer picks the MLA head's from its KV params."""
    draft: Any = _Head()
    config = SimpleNamespace(
        kv_params=MLAKVCacheParams(
            dtype=DType.bfloat16,
            head_dim=576,
            num_layers=1,
            devices=[DeviceRef.GPU(i) for i in range(2)],
            data_parallel_degree=1,
            num_q_heads=128,
        )
    )
    hf_config: Any = SimpleNamespace(get_text_config=lambda: object())
    speculative = SpeculativeConfig(
        speculative_method="mtp", num_speculative_tokens=1
    )
    proposer = _proposer(
        TARGET,
        mtp_deepseekV3_draft_arch,
        SimpleNamespace(takes_distributed_inputs=True),
        draft,
        config,
        speculative,
        hf_config,
    )
    legacy = proposers.sequential_proposer(
        speculative, draft, config, hf_config, distributed=True
    )
    assert isinstance(proposer, MTPProposer)
    assert proposer.draft is draft
    assert type(legacy) is type(proposer)


def _config() -> DeepseekV3NextNConfig:
    """A one-device NextN config at DeepseekV3's own dims, but fewer
    experts."""
    devices = [DeviceRef.GPU()]
    return DeepseekV3NextNConfig(
        dtype=DType.bfloat16,
        kv_params=MLAKVCacheParams(
            dtype=DType.bfloat16,
            head_dim=576,
            num_layers=1,
            devices=devices,
            data_parallel_degree=1,
            num_q_heads=128,
        ),
        devices=devices,
        n_routed_experts=NUM_EXPERTS,
        topk_method="noaux_tc",
        correction_bias_dtype=DType.float32,
        max_seq_len=4096,
        rope_scaling={
            "type": "yarn",
            "factor": 40.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
    )


def test_the_target_returns_what_the_method_selected() -> None:
    """The registration asks the target for the hidden states and layers
    the method's helpers chose before it: the last normalized layer."""
    speculative = SpeculativeConfig(
        speculative_method="mtp", num_speculative_tokens=1
    )
    config = _config()
    reads = mtp_draft_registration.target_hidden_states(speculative, config)
    assert reads == TargetHiddenStates(
        proposers.target_hidden_states(speculative, config),
        proposers.capture_layers(speculative, config),
    )
    assert reads == TargetHiddenStates(ReturnHiddenStates.ALL_NORMALIZED)


def _checkpoint() -> dict[str, NonCallableMock]:
    """A DeepseekV3 checkpoint's names: a trunk weight, and the NextN
    layer at index ``num_hidden_layers``, as DeepSeek ships it."""
    projections = ("gate_proj", "up_proj", "down_proj")
    layer = [
        "enorm.weight",
        "hnorm.weight",
        "eh_proj.weight",
        "shared_head.norm.weight",
        "shared_head.head.weight",
        "embed_tokens.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_a_proj.weight",
        "self_attn.q_a_layernorm.weight",
        "self_attn.q_b_proj.weight",
        "self_attn.kv_a_proj_with_mqa.weight",
        "self_attn.kv_a_layernorm.weight",
        "self_attn.kv_b_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate.weight",
        "mlp.gate.e_score_correction_bias",
        *(f"mlp.shared_experts.{p}.weight" for p in projections),
        *(
            f"mlp.experts.{i}.{p}.weight"
            for i in range(NUM_EXPERTS)
            for p in projections
        ),
    ]
    names = [
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.norm.weight",
        "lm_head.weight",
        *(f"model.layers.{NUM_LAYERS}.{name}" for name in layer),
    ]
    return {name: NonCallableMock() for name in names}


def test_the_builder_loads_every_nextn_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Through the architecture's own weight adapter, the checkpoint feeds
    every weight the head owns and nothing else; the head reads its
    embedding and head from the target."""
    config = _config()
    assert mtp_deepseekV3_draft_arch.draft is not None
    arch = replace(
        mtp_deepseekV3_draft_arch,
        draft=replace(
            mtp_deepseekV3_draft_arch.draft,
            configure=lambda context, weights: config,
        ),
    )
    # Every allreduce opens its accelerators on construction, which this
    # CPU test has none of; nothing here runs one.
    monkeypatch.setattr(
        "max.nn.legacy.comm.allreduce.Accelerator", lambda id: object()
    )
    loaded: list[dict[str, object]] = []
    monkeypatch.setattr(
        DeepseekV3NextN,
        "load_state_dict",
        lambda self, weights, **kwargs: loaded.append(dict(weights)),
    )
    checkpoint = _checkpoint()
    context: Any = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            speculative=SpeculativeConfig(
                speculative_method="mtp", num_speculative_tokens=1
            ),
            draft_model=None,
            model=SimpleNamespace(
                huggingface_config=SimpleNamespace(num_hidden_layers=NUM_LAYERS)
            ),
        ),
        weights=SimpleNamespace(items=checkpoint.items),
        adapter=arch.weight_adapters[WeightsFormat.safetensors],
    )
    head, returned = RegisteredDraftBuilder(arch).build_module(context)
    assert isinstance(head, DeepseekV3NextN)
    assert returned is config
    owned = {
        name
        for name in head.raw_state_dict()
        if not name.startswith(("embed_tokens.", "lm_head."))
    }
    assert loaded and loaded[0].keys() == owned
