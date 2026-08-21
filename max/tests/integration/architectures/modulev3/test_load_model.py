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
"""load_model() integration tests for ModuleV3 architectures.

Models tested here use small synthetic configs and zero weights, so they
exercise module init (config parsing, weight adaptation, graph tracing)
without needing real checkpoints.
"""

from __future__ import annotations

import pytest
import torch
from max.graph.weights import SafetensorWeights
from max.pipelines.architectures.gemma4_modulev3.model import Gemma4Model
from max.pipelines.architectures.gemma4_modulev3.weight_adapters import (
    convert_safetensor_state_dict as convert_gemma4_state_dict,
)
from max.pipelines.architectures.llama3_modulev3.model import Llama3Model
from max.pipelines.architectures.llama3_modulev3.weight_adapters import (
    convert_safetensor_state_dict,
)
from max.pipelines.architectures.olmo_modulev3.model import OlmoModel
from max.pipelines.architectures.phi3_modulev3.model import Phi3Model
from test_common.load_model_helpers import (
    assert_load_model_succeeds,
    make_pipeline_config_factory,
    make_small_llama_config,
    make_zero_weights,
)
from transformers import PretrainedConfig


@pytest.mark.parametrize(
    "model_cls,repo_id",
    [
        (Llama3Model, "meta-llama/Llama-3.1-8B-Instruct"),
        (Phi3Model, "microsoft/phi-4"),
        (OlmoModel, "allenai/OLMo-1B-hf"),
        (Llama3Model, "ibm-granite/granite-3.1-8b-instruct"),
    ],
    ids=["llama3", "phi3", "olmo", "granite"],
)
def test_load_model(model_cls: type, repo_id: str) -> None:
    hf_config = make_small_llama_config()
    weights = make_zero_weights(hf_config)
    make_pipeline_config = make_pipeline_config_factory(hf_config, repo_id)
    assert_load_model_succeeds(
        model_cls, make_pipeline_config, weights, convert_safetensor_state_dict
    )


def make_small_gemma4_config() -> PretrainedConfig:
    """Synthetic 6-layer gemma4 config: one full 5:1 sliding:full period."""
    layer_types = ["sliding_attention"] * 5 + ["full_attention"]
    text_config = PretrainedConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        # Dims are chosen so no two geometries collide: sliding kv width
        # (4*16=64) != global kv width (1*32=32), and both q widths
        # (8*16=128 sliding, 8*32=256 global) differ from hidden_size (64).
        # num_attention_heads stays a multiple of both kv head counts.
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=16,
        num_global_key_value_heads=1,
        global_head_dim=32,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        attention_bias=False,
        sliding_window=128,
        final_logit_softcapping=30.0,
        attention_k_eq_v=True,
        num_kv_shared_layers=0,
        enable_moe_block=False,
        use_double_wide_mlp=False,
        num_experts=0,
        top_k_experts=0,
        moe_intermediate_size=0,
        vocab_size_per_layer_input=256,
        hidden_size_per_layer_input=0,
        layer_types=layer_types,
        rope_parameters={
            "full_attention": {
                "rope_type": "proportional",
                "partial_rotary_factor": 0.25,
                "rope_theta": 1_000_000.0,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10_000.0,
            },
        },
    )
    hf_config = PretrainedConfig(
        text_config=text_config,
        vision_config={},  # non-None sentinel; unified model_type skips it
        # Read by make_pipeline_config_factory to size max_length.
        max_position_embeddings=text_config.max_position_embeddings,
        image_token_id=255,
        tie_word_embeddings=True,
        architectures=["Gemma4UnifiedForConditionalGeneration"],
    )
    hf_config.model_type = "gemma4_unified"
    return hf_config


def make_gemma4_zero_weights(hf_config: PretrainedConfig) -> SafetensorWeights:
    t = hf_config.text_config

    def z(*shape: int) -> torch.Tensor:
        return torch.zeros(*shape, dtype=torch.bfloat16)

    wm: dict[str, torch.Tensor] = {}
    prefix = "model.language_model."
    wm[prefix + "embed_tokens.weight"] = z(t.vocab_size, t.hidden_size)
    wm[prefix + "norm.weight"] = z(t.hidden_size)
    for i, layer_type in enumerate(t.layer_types):
        lp = f"{prefix}layers.{i}."
        sliding = layer_type == "sliding_attention"
        hd = t.head_dim if sliding else t.global_head_dim
        n_kv = (
            t.num_key_value_heads if sliding else t.num_global_key_value_heads
        )
        wm[lp + "self_attn.q_proj.weight"] = z(
            t.num_attention_heads * hd, t.hidden_size
        )
        wm[lp + "self_attn.k_proj.weight"] = z(n_kv * hd, t.hidden_size)
        if sliding:
            wm[lp + "self_attn.v_proj.weight"] = z(n_kv * hd, t.hidden_size)
        wm[lp + "self_attn.o_proj.weight"] = z(
            t.hidden_size, t.num_attention_heads * hd
        )
        wm[lp + "self_attn.q_norm.weight"] = z(hd)
        wm[lp + "self_attn.k_norm.weight"] = z(hd)
        for norm in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            wm[lp + norm + ".weight"] = z(t.hidden_size)
        wm[lp + "mlp.gate_proj.weight"] = z(t.intermediate_size, t.hidden_size)
        wm[lp + "mlp.up_proj.weight"] = z(t.intermediate_size, t.hidden_size)
        wm[lp + "mlp.down_proj.weight"] = z(t.hidden_size, t.intermediate_size)
        wm[lp + "layer_scalar"] = z(1)
    return SafetensorWeights(
        [],
        tensors=set(wm.keys()),
        tensors_to_file_idx={},
        _st_weight_map=wm,
    )


def test_load_model_gemma4_modulev3() -> None:
    hf_config = make_small_gemma4_config()
    weights = make_gemma4_zero_weights(hf_config)
    make_pipeline_config = make_pipeline_config_factory(
        hf_config, "google/gemma-4-31B-it"
    )
    assert_load_model_succeeds(
        Gemma4Model, make_pipeline_config, weights, convert_gemma4_state_dict
    )
