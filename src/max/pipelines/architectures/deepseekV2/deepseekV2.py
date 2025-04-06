# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""Implements the DeepseekV2 model."""

from __future__ import annotations

from max.nn import (
    MLPV2,
    EmbeddingV2,
    LinearV2,
    Module,
    OptimizedRotaryEmbedding,
    RMSNormV2,
    Transformer,
    TransformerBlock,
)
from max.nn.attention.attention_with_rope import LatentAttentionWithRope
from max.pipelines.architectures.deepseekV2.layers.mix_of_experts import MoE
from max.pipelines.kv_cache import FetchContinuousBatchingKVCacheCollection

from .model_config import DeepseekV2Config


class DeepseekV2(Transformer):
    """Defines the DeepseekV2 transformer model."""

    def __init__(self, config: DeepseekV2Config):
        assert len(config.devices) == 1
        # TODO: Replace with YarnRope when integrated with LatentAttention
        rope = OptimizedRotaryEmbedding(
            dim=config.qk_rope_head_dim,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_position_embeddings,
        )

        layers = [
            TransformerBlock(
                attention=LatentAttentionWithRope(
                    rope=rope,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    hidden_size=config.hidden_size,
                    kv_params=config.kv_params,
                    layer_idx=i,
                    dtype=config.dtype,
                    q_lora_rank=config.q_lora_rank,
                    kv_lora_rank=config.kv_lora_rank,
                    qk_nope_head_dim=config.qk_nope_head_dim,
                    qk_rope_head_dim=config.qk_rope_head_dim,
                    v_head_dim=config.v_head_dim,
                ),
                mlp=self._get_mlp(config, i),
                attention_norm=RMSNormV2(
                    config.hidden_size,
                    config.rms_norm_eps,
                ),
                mlp_norm=RMSNormV2(
                    config.hidden_size,
                    config.rms_norm_eps,
                ),
            )
            for i in range(config.num_hidden_layers)
        ]

        # Create Embedding and output layers.
        embedding_output_dtype = config.dtype
        embedding_layer = EmbeddingV2(
            config.vocab_size,
            config.hidden_size,
            embedding_output_dtype,
            config.devices[0],
        )
        lm_head = self.lm_head = LinearV2(
            config.hidden_size, config.vocab_size, dtype=config.dtype
        )

        super().__init__(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            layers=layers,
            norm=RMSNormV2(
                config.hidden_size,
                config.rms_norm_eps,
            ),
            output=lm_head,
            embedding=embedding_layer,
            kv_params=config.kv_params,
            kv_collection_constructor=FetchContinuousBatchingKVCacheCollection(
                config.kv_params
            ),
        )

    def _get_mlp(self, config: DeepseekV2Config, i: int) -> Module:
        """Helper function to return a mixture of experts layer or traditional multi-layer perceptron layer
        for the TransformerBlock's mlp depending on the layer idx.

        Args:
            config: Configuration object containing model parameters
            i: Layer index

        Returns:
            Either a MoE or MLPV2 module depending on the layer index and config
        """
        if (
            config.n_routed_experts is not None
            and i >= config.first_k_dense_replace
            and i % config.moe_layer_freq == 0
        ):
            return MoE(
                num_experts_per_tok=config.num_experts_per_tok,
                ep_size=config.ep_size,
                experts_per_rank=config.n_routed_experts // config.ep_size,
                moe_intermediate_size=config.moe_intermediate_size,
                max_position_embeddings=config.max_position_embeddings,
                n_shared_experts=config.n_shared_experts,
                dtype=config.dtype,
            )
        else:
            return MLPV2(
                dtype=config.dtype,
                quantization_encoding=None,
                hidden_dim=config.hidden_size,
                feed_forward_length=config.intermediate_size,
                devices=config.devices,
            )
