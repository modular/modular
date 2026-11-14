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

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import NonCallableMock

from max.graph.weights import WeightsFormat
from max.pipelines.architectures.llama3.weight_adapters import (
    convert_gguf_state_dict,
)
from max.pipelines.architectures.llama3_modulev3 import llama_modulev3_arch
from max.pipelines.architectures.llama3_modulev3 import (
    weight_adapters as modulev3_weight_adapters,
)
from max.pipelines.lib import WeightTransform
from max.pipelines.modeling.config_enums import SupportedEncoding
from transformers import LlamaConfig

# Every tensor name of meta-llama/Llama-3.1-8B-Instruct, with one layer.
_CHECKPOINT_NAMES = (
    "model.embed_tokens.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.post_attention_layernorm.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.norm.weight",
    "lm_head.weight",
)


@dataclass
class _WeightRepoStub:
    # ``_select_quantization_encoding`` iterates this only when no
    # ``weight_path`` is set; an empty tuple means no float cast candidates,
    # so the given ``quantization_encoding`` passes through unchanged.
    supported_encodings: tuple[SupportedEncoding, ...] = ()


@dataclass
class _ModelConfigStub:
    quantization_encoding: SupportedEncoding | None
    weight_path: list[str] = field(default_factory=list)
    huggingface_weight_repo: _WeightRepoStub = field(
        default_factory=_WeightRepoStub
    )


@dataclass
class _PipelineConfigStub:
    model: _ModelConfigStub


def _weight_mock() -> NonCallableMock:
    weight = NonCallableMock()
    weight.data.return_value = "mock_weight_data"
    return weight


def test_convert_gguf_state_dict_non_quantized_uses_stacked_linear_keys() -> (
    None
):
    state_dict = {
        "blk.0.attn_q.weight": _weight_mock(),
        "blk.0.attn_k.weight": _weight_mock(),
        "blk.0.attn_v.weight": _weight_mock(),
        "rope_freqs.weight": _weight_mock(),
    }
    # A non-quantized encoding resolves to no ``QuantizationEncoding``, so the
    # adapter uses the unfused stacked-linear mapping.
    pipeline_config = _PipelineConfigStub(
        model=_ModelConfigStub(quantization_encoding="bfloat16")
    )

    converted = convert_gguf_state_dict(
        state_dict,  # type: ignore[arg-type]
        pipeline_config=pipeline_config,  # type: ignore[arg-type]
    )

    assert "layers.0.self_attn.q_proj.weight" in converted
    assert "layers.0.self_attn.k_proj.weight" in converted
    assert "layers.0.self_attn.v_proj.weight" in converted
    assert "rope_freqs.weight" not in converted


def test_convert_gguf_state_dict_quantized_keeps_legacy_qkv_keys() -> None:
    state_dict = {
        "blk.0.attn_q.weight": _weight_mock(),
        "blk.0.attn_k.weight": _weight_mock(),
        "blk.0.attn_v.weight": _weight_mock(),
    }
    pipeline_config = _PipelineConfigStub(
        model=_ModelConfigStub(quantization_encoding="q4_k")
    )

    converted = convert_gguf_state_dict(
        state_dict,  # type: ignore[arg-type]
        pipeline_config=pipeline_config,  # type: ignore[arg-type]
    )

    assert "layers.0.self_attn.q_proj.weight" in converted
    assert "layers.0.self_attn.k_proj.weight" in converted
    assert "layers.0.self_attn.v_proj.weight" in converted


def test_modulev3_rules_name_what_the_converter_did() -> None:
    state_dict = {name: _weight_mock() for name in _CHECKPOINT_NAMES}
    adapter = llama_modulev3_arch.checkpoint_adapters(LlamaConfig())[
        WeightsFormat.safetensors
    ]
    assert isinstance(adapter, WeightTransform)

    converted = modulev3_weight_adapters.convert_safetensor_state_dict(
        state_dict  # type: ignore[arg-type]
    )

    assert adapter(state_dict).keys() == converted.keys()


def test_gguf_checkpoint_keeps_its_converter() -> None:
    adapters = llama_modulev3_arch.checkpoint_adapters(LlamaConfig())

    assert (
        adapters[WeightsFormat.gguf]
        is llama_modulev3_arch.weight_adapters[WeightsFormat.gguf]
    )
