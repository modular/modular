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
"""GLM-5.x's NextN layer pairs with its ModuleV3 target as a draft."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import Weights
from max.nn.kv_cache import MHAKVCacheParams, MultiKVCacheParams
from max.pipelines.architectures.glm5_1_modulev3 import mtp
from max.pipelines.architectures.glm5_1_modulev3.arch import (
    glm5_1_modulev3_arch,
    glm5_1_mtp_draft_arch,
)
from max.pipelines.architectures.glm5_1_modulev3.spec_adapters import ADAPTER
from max.pipelines.architectures.glm5_1_modulev3.weight_adapters import (
    glm5_1_mtp_weight_rules,
)
from max.pipelines.lib import WeightTransform
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    _SpecDecodeModuleV3PipelineModel,
    spec_decode_architecture,
)
from max.pipelines.lib.registration_runtime import ModuleRuntime
from max.pipelines.speculative.config import SpeculativeConfig
from max.pipelines.speculative.spec_adapters import has_adapter
from transformers.models.deepseek_v3.configuration_deepseek_v3 import (
    DeepseekV3Config,
)

_TARGET = "GlmMoeDsaForCausalLM"
_LAYERS = 2
"""Target layers; the NextN layer sits at index ``_LAYERS``."""


def test_the_modulev3_target_pairs_with_its_checkpoint_nextn_layer() -> None:
    assert has_adapter(glm5_1_modulev3_arch.name)
    arch = spec_decode_architecture(_TARGET, "mtp", None, prefer_module_v3=True)
    assert arch is not None
    assert arch.name == f"{glm5_1_modulev3_arch.name}+mtp"
    assert arch.pipeline_model is _SpecDecodeModuleV3PipelineModel
    assert arch.multi_gpu_supported
    assert arch.requires_max_batch_context_length
    assert arch.reasoning_parser == "glm45"


def test_the_graph_api_target_keeps_its_unified_architecture() -> None:
    """The default lookup serves the graph-API target, whose MTP still runs
    as its own unified architecture."""
    assert spec_decode_architecture(_TARGET, "mtp", None) is None


def test_the_nextn_layer_drafts_only_under_mtp() -> None:
    registration = glm5_1_mtp_draft_arch.draft
    assert registration is not None
    assert registration.method == "mtp"
    # It reads the target's embedding and head through its proposer.
    assert registration.shared_modules == ()
    assert (
        spec_decode_architecture(_TARGET, "eagle", None, prefer_module_v3=True)
        is None
    )


def test_the_target_adapter_refuses_other_methods() -> None:
    ADAPTER.check(
        SpeculativeConfig(speculative_method="mtp", num_speculative_tokens=2),
        False,
    )
    with pytest.raises(ValueError, match="NextN layer"):
        ADAPTER.check(
            SpeculativeConfig(
                speculative_method="eagle", num_speculative_tokens=2
            ),
            False,
        )


def _weight(dtype: DType = DType.bfloat16) -> Any:
    weights = MagicMock(spec=Weights)
    weights.data.return_value = SimpleNamespace(dtype=dtype)
    return weights


def test_the_draft_takes_only_the_nextn_layer_under_its_own_names() -> None:
    nextn = f"model.layers.{_LAYERS}"
    transform = WeightTransform(
        glm5_1_mtp_weight_rules(DeepseekV3Config(num_hidden_layers=_LAYERS))
    )
    converted = transform(
        {
            "model.norm.weight": _weight(),
            "model.layers.0.input_layernorm.weight": _weight(),
            "lm_head.weight": _weight(),
            f"{nextn}.enorm.weight": _weight(),
            f"{nextn}.hnorm.weight": _weight(),
            f"{nextn}.eh_proj.weight": _weight(),
            f"{nextn}.shared_head.norm.weight": _weight(),
            f"{nextn}.input_layernorm.weight": _weight(),
        }
    )
    assert set(converted) == {
        "enorm.weight",
        "hnorm.weight",
        "eh_proj.weight",
        "shared_head_norm.weight",
        "decoder_layer.input_layernorm.weight",
    }


@dataclass(frozen=True)
class _Quant:
    mlp_quantized_layers: set[int]
    attn_quantized_layers: set[int]


@dataclass(frozen=True)
class _EP:
    dispatch_dtype: DType = DType.uint8
    dispatch_quant_config: object = "nvfp4"
    fused_shared_expert: bool = False


@dataclass(frozen=True)
class _Config:
    """Stands in for the target's finalized config."""

    quant_config: _Quant | None = field(
        default_factory=lambda: _Quant({0, 1}, {0, 1})
    )
    num_hidden_layers: int = _LAYERS
    first_k_dense_replace: int = 1
    n_shared_experts: int = 1
    kv_params: object = None
    indexer_types: tuple[str, ...] = ("full", "shared")
    ep_config: Any = None

    @classmethod
    def initialize(cls, pipeline_config: Any, *, max_seq_len: int) -> _Config:
        return cls()

    def finalize(self, *args: Any) -> _Config:
        return self

    def expert_parallel_config(self, runtime: ModuleRuntime) -> _EP:
        return _EP()

    def apply_runtime(self, runtime: ModuleRuntime) -> _Config:
        return replace(self, ep_config=runtime.ep)


def _configure(
    monkeypatch: pytest.MonkeyPatch,
    draft_weights: dict[str, Any],
    *,
    ep_size: int = 8,
) -> tuple[Any, list[object], Any]:
    monkeypatch.setattr(mtp, "Glm5_1Config", _Config)
    monkeypatch.setattr(mtp, "parse_state_dict_from_weights", lambda *args: {})
    started: list[object] = []
    context: Any = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            model=SimpleNamespace(data_parallel_degree=1),
            runtime=SimpleNamespace(
                max_batch_input_tokens=8,
                ep_size=ep_size,
                ep_use_allreduce=False,
            ),
        ),
        huggingface_config=SimpleNamespace(num_hidden_layers=_LAYERS),
        weights=object(),
        kv_params=MultiKVCacheParams.from_params(
            {
                "mla": MHAKVCacheParams(
                    dtype=DType.bfloat16,
                    n_kv_heads=1,
                    head_dim=16,
                    num_layers=1,
                    devices=[DeviceRef.CPU()],
                )
            }
        ),
        max_seq_len=16,
        max_batch_total_tokens=32,
        start_ep=started.append,
    )
    return (
        mtp.configure_mtp_draft(context, draft_weights),
        started,
        context,
    )


def test_a_bf16_nextn_layer_narrows_the_target_and_sizes_ep_for_bf16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NVFP4 checkpoints leave the layer in bf16: its experts dispatch wider
    than the target's, and it starts the buffers both share."""
    config, started, context = _configure(
        monkeypatch,
        {"eh_proj.weight": 0, "decoder_layer.mlp.experts.0.w.weight": 0},
    )
    assert config.kv_params is context.kv_params
    assert config.indexer_types == []
    assert config.quant_config == _Quant({0, 1}, {0, 1})
    assert config.ep_config == _EP(
        dispatch_dtype=DType.bfloat16,
        dispatch_quant_config=None,
        fused_shared_expert=True,
    )
    assert started == [config.ep_config]


def test_a_quantized_nextn_layer_takes_the_targets_layouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, started, _ = _configure(
        monkeypatch,
        {
            "eh_proj.weight": 0,
            "decoder_layer.mlp.experts.0.w.weight_scale_inv": 0,
            "decoder_layer.self_attn.q_a_proj.weight_scale_inv": 0,
        },
    )
    assert config.quant_config == _Quant({0, 1, _LAYERS}, {0, 1, _LAYERS})
    assert config.ep_config == _EP()
    assert started == [_EP()]


def test_a_single_device_nextn_layer_starts_no_ep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, started, _ = _configure(
        monkeypatch, {"eh_proj.weight": 0}, ep_size=1
    )
    assert config.ep_config is None
    assert started == []


def test_a_checkpoint_without_a_nextn_layer_does_not_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(KeyError, match="no NextN layer"):
        _configure(monkeypatch, {})
