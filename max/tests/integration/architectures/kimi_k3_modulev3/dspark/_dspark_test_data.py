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
"""Test data shared by the K3 DSpark tests.

The config dict is ``Inferact/Kimi-K3-DSpark``'s ``config.json``, verbatim;
the header dict is its ``model.safetensors`` header, read header-only (the
file itself is 7 GB), so the tests pin the real names and shapes without
loading them.
"""

from __future__ import annotations

# Inferact/Kimi-K3-DSpark config.json, verbatim.
DRAFT_CONFIG: dict[str, object] = {
    "architectures": ["K3DSparkModel"],
    "model_type": "k3_dspark",
    "hidden_size": 7168,
    "intermediate_size": 14336,
    "num_hidden_layers": 5,
    "num_attention_heads": 64,
    "num_key_value_heads": 64,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "mla_use_nope": False,
    "mla_use_output_gate": False,
    "vocab_size": 163840,
    "rms_norm_eps": 1e-05,
    "max_position_embeddings": 1048576,
    "rope_theta": 50000.0,
    "num_target_layers": 5,
    "target_hidden_size": 7168,
    "target_num_hidden_layers": 93,
    "target_layer_ids": [2, 23, 47, 71, 89],
    "mask_token_id": 163837,
    "markov_rank": 256,
    "markov_head_type": "vanilla",
    "enable_confidence_head": True,
    "confidence_head_with_markov": True,
    "tie_word_embeddings": False,
    "draft_vocab_size": 163840,
    "torch_dtype": "bfloat16",
    "rope_parameters": {
        "rope_type": "yarn",
        "factor": 32.0,
        "original_max_position_embeddings": 32768,
        "rope_theta": 50000.0,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
    },
}

# A tiny ``k3_dspark`` config mapping with the published checkpoint's key
# set, sized for kernel and numerics tests. The head geometry is the
# published one -- a 576-wide latent (kv_lora 512 + rope 64), nope 128, v
# 128 -- because the drafter serves only through the fused MLA kernels,
# whose latent width is fixed; only the widths no kernel reads shrink. Two
# taps off a four-layer target, so the ids stay in range beside the 4-layer
# fixture.
TINY_DRAFT_CONFIG: dict[str, object] = {
    "architectures": ["K3DSparkModel"],
    "model_type": "k3_dspark",
    "hidden_size": 64,
    "intermediate_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "q_lora_rank": 16,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "mla_use_nope": False,
    "mla_use_output_gate": False,
    "vocab_size": 64,
    "rms_norm_eps": 1e-05,
    "max_position_embeddings": 256,
    "num_target_layers": 2,
    "target_hidden_size": 64,
    "target_num_hidden_layers": 4,
    "target_layer_ids": [0, 1],
    "mask_token_id": 63,
    "markov_rank": 8,
    "markov_head_type": "vanilla",
    "enable_confidence_head": True,
    "confidence_head_with_markov": True,
    "block_size": 7,
    "tie_word_embeddings": False,
    "draft_vocab_size": 64,
    "torch_dtype": "bfloat16",
    "rope_parameters": {
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 64,
        "rope_theta": 10000.0,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
    },
}

# The checkpoint's safetensors header: tensor name -> (dtype, shape).
REAL_HEADER: dict[str, tuple[str, tuple[int, ...]]] = {
    "confidence_head.proj.bias": ("BF16", (1,)),
    "confidence_head.proj.weight": ("BF16", (1, 7424)),
    "context_norm.weight": ("BF16", (7168,)),
    "context_proj.weight": ("BF16", (7168, 35840)),
    "embed_tokens.weight": ("BF16", (163840, 7168)),
    "final_norm.weight": ("BF16", (7168,)),
    "markov_head.markov_w1.weight": ("BF16", (163840, 256)),
    "markov_head.markov_w2.weight": ("BF16", (163840, 256)),
}
for _layer in range(5):
    for _name, _shape in (
        ("input_layernorm.weight", (7168,)),
        ("mlp.down_proj.weight", (7168, 14336)),
        ("mlp.gate_proj.weight", (14336, 7168)),
        ("mlp.up_proj.weight", (14336, 7168)),
        ("post_attention_layernorm.weight", (7168,)),
        ("self_attn.kv_a_layernorm.weight", (512,)),
        ("self_attn.kv_a_proj_with_mqa.weight", (576, 7168)),
        ("self_attn.kv_b_proj.weight", (16384, 512)),
        ("self_attn.o_proj.weight", (7168, 8192)),
        ("self_attn.q_a_layernorm.weight", (1536,)),
        ("self_attn.q_a_proj.weight", (1536, 7168)),
        ("self_attn.q_b_proj.weight", (12288, 1536)),
    ):
        REAL_HEADER[f"layers.{_layer}.{_name}"] = ("BF16", _shape)
