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
    MLP,
    Embedding,
    Linear,
    Module,
    RMSNorm,
    Transformer,
    TransformerBlock,
)
from max.nn.attention.attention_with_rope import LatentAttentionWithRope
from max.nn.kv_cache import FetchPagedKVCacheCollection
from max.nn.rotary_embedding import (
    DeepseekYarnRopeScalingParams,
    DeepseekYarnRotaryEmbedding,
)
from max.pipelines.architectures.deepseekV2.layers.mix_of_experts import MoE

from .model_config import DeepseekV2Config


class DeepseekV2(Transformer):
    """Defines the DeepseekV2 transformer model."""

    def __init__(self, config: DeepseekV2Config):
        assert len(config.devices) == 1
        assert config.rope_scaling is not None
        scaling_params = DeepseekYarnRopeScalingParams(
            scaling_factor=config.rope_scaling["factor"],
            original_max_position_embeddings=config.rope_scaling[
                "original_max_position_embeddings"
            ],
            beta_fast=config.rope_scaling["beta_fast"],
            beta_slow=config.rope_scaling["beta_slow"],
            mscale=config.rope_scaling["mscale"],
            mscale_all_dim=config.rope_scaling["mscale_all_dim"],
        )
        rope = DeepseekYarnRotaryEmbedding(
            config.qk_rope_head_dim,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_position_embeddings,
            scaling_params=scaling_params,
            device=config.devices[0],
        )

        layers = [
            TransformerBlock(
                attention=LatentAttentionWithRope(
                    rope=rope,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    hidden_size=config.hidden_size,
                    kv_params=config.kv_params,
                    dtype=config.dtype,
                    q_lora_rank=config.q_lora_rank,
                    kv_lora_rank=config.kv_lora_rank,
                    qk_nope_head_dim=config.qk_nope_head_dim,
                    qk_rope_head_dim=config.qk_rope_head_dim,
                    v_head_dim=config.v_head_dim,
                    devices=config.devices,
                ),
                mlp=self._get_mlp(config, i),
                attention_norm=RMSNorm(
                    config.hidden_size,
                    config.dtype,
                    config.rms_norm_eps,
                ),
                mlp_norm=RMSNorm(
                    config.hidden_size,
                    config.dtype,
                    config.rms_norm_eps,
                ),
            )
            for i in range(config.num_hidden_layers)
        ]

        # Create Embedding and output layers.
        embedding_output_dtype = config.dtype
        embedding_layer = Embedding(
            config.vocab_size,
            config.hidden_size,
            embedding_output_dtype,
            config.devices[0],
        )
        lm_head = self.lm_head = Linear(
            config.hidden_size,
            config.vocab_size,
            dtype=config.dtype,
            device=config.devices[0],
        )

        super().__init__(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            layers=layers,
            norm=RMSNorm(
                config.hidden_size,
                config.dtype,
                config.rms_norm_eps,
            ),
            output=lm_head,
            embedding=embedding_layer,
            kv_params=config.kv_params,
            kv_collection_constructor=FetchPagedKVCacheCollection(
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
            Either a MoE or MLP module depending on the layer index and config
        """
        if (
            config.n_routed_experts is not None
            and i >= config.first_k_dense_replace
            and i % config.moe_layer_freq == 0
        ):
            assert len(config.devices) == 1, "Expect only one device"
            return MoE(
                num_experts_per_tok=config.num_experts_per_tok,
                ep_size=config.ep_size,
                experts_per_rank=config.n_routed_experts // config.ep_size,
                moe_intermediate_size=config.moe_intermediate_size,
                hidden_size=config.hidden_size,
                n_shared_experts=config.n_shared_experts,
                dtype=config.dtype,
                device=config.devices[0],
            )
        else:
            return MLP(
                dtype=config.dtype,
                quantization_encoding=None,
                hidden_dim=config.hidden_size,
                feed_forward_length=config.intermediate_size,
                devices=config.devices,
            )
