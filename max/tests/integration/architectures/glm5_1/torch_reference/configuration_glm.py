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

"""GlmMoeDsa config for GLM-5.1 layer integration tests.

Adapted from HuggingFace transformers v5.8.1
``configuration_glm_moe_dsa.py`` for use with :meth:`PretrainedConfig.update`.
"""

from __future__ import annotations

from typing import Any

from transformers.configuration_utils import PretrainedConfig


class GlmMoeDsaConfig(PretrainedConfig):
    """Configuration for GlmMoeDsa (GLM-5.x MoE + DSA) models."""

    model_type = "glm_moe_dsa"

    attribute_map = {
        "num_local_experts": "n_routed_experts",
        "head_dim": "qk_rope_head_dim",
    }

    def __init__(
        self,
        vocab_size: int = 154880,
        hidden_size: int = 6144,
        intermediate_size: int = 12288,
        moe_intermediate_size: int = 2048,
        num_hidden_layers: int = 78,
        num_attention_heads: int = 64,
        num_key_value_heads: int | None = None,
        n_shared_experts: int = 1,
        n_routed_experts: int = 256,
        routed_scaling_factor: float = 2.5,
        kv_lora_rank: int = 512,
        q_lora_rank: int = 2048,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 256,
        qk_nope_head_dim: int = 192,
        n_group: int = 1,
        topk_group: int = 1,
        num_experts_per_tok: int = 8,
        norm_topk_prob: bool = True,
        hidden_act: str = "silu",
        max_position_embeddings: int = 202752,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_parameters: dict[str, Any] | None = None,
        rope_theta: float | None = None,
        rope_interleave: bool = True,
        indexer_rope_interleave: bool = True,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        index_topk: int = 2048,
        index_head_dim: int = 128,
        index_n_heads: int = 32,
        indexer_types: list[str] | None = None,
        mlp_layer_types: list[str] | None = None,
        first_k_dense_replace: int = 3,
        moe_layer_freq: int = 1,
        scoring_func: str = "sigmoid",
        topk_method: str = "noaux_tc",
        ep_size: int = 1,
        **kwargs: Any,
    ) -> None:
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        if rope_parameters is None:
            theta = rope_theta if rope_theta is not None else 10000.0
            rope_parameters = {"rope_type": "default", "rope_theta": theta}

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.num_local_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.head_dim = qk_rope_head_dim
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_parameters = rope_parameters
        self.rope_theta = rope_parameters.get("rope_theta", 10000.0)
        self.rope_interleave = rope_interleave
        self.indexer_rope_interleave = indexer_rope_interleave
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.index_topk = index_topk
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.ep_size = ep_size
        self.rope_scaling = None

        if mlp_layer_types is None:
            n_dense = min(first_k_dense_replace, num_hidden_layers)
            self.mlp_layer_types = ["dense"] * n_dense + ["sparse"] * (
                num_hidden_layers - n_dense
            )
        else:
            self.mlp_layer_types = mlp_layer_types

        if indexer_types is None:
            self.indexer_types = ["full"] * num_hidden_layers
        else:
            self.indexer_types = indexer_types

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
