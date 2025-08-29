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
"""Implements the DeepseekV2 model that runs on multiple devices."""

from __future__ import annotations

import functools

from max.graph import ShardingStrategy
from max.nn import (
    MLP,
    ColumnParallelLinear,
    DistributedTransformer,
    DistributedTransformerBlock,
    RMSNorm,
    VocabParallelEmbedding,
)
from max.nn.attention.multi_latent_attention import (
    DistributedLatentAttentionWithRope,
)
from max.nn.kv_cache import FetchPagedKVCacheCollection
from max.nn.moe import MoE
from max.nn.rotary_embedding import (
    DeepseekYarnRopeScalingParams,
    DeepseekYarnRotaryEmbedding,
)
from max.pipelines.architectures.deepseekV2.layers.moe_gate import (
    DeepSeekV2MoEGate,
)

from .model_config import DeepseekV2Config


class DistributedDeepseekV2(DistributedTransformer):
    """Defines the DeepseekV2 transformer model."""

    def __init__(self, config: DeepseekV2Config) -> None:
        assert len(config.devices) > 1
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

        distributed_norm = functools.partial(
            RMSNorm,
            dim=config.hidden_size,
            dtype=config.dtype,
            eps=config.rms_norm_eps,
            multiply_before_cast=False,
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
            DistributedTransformerBlock(
                devices=config.devices,
                attention=DistributedLatentAttentionWithRope(
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
                mlp=self._get_mlp(config, idx),
                attention_norm=distributed_norm(),
                mlp_norm=distributed_norm(),
            )
            for idx in range(config.num_hidden_layers)
        ]

        # Create Embedding and output layers.
        embedding_output_dtype = config.dtype
        embedding_layer = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            embedding_output_dtype,
            config.devices,
        )
        lm_head = self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            dtype=config.dtype,
            devices=config.devices,
        )

        super().__init__(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            layers=layers,
            norm=distributed_norm(),
            output=lm_head,
            embedding=embedding_layer,
            kv_params=config.kv_params,
            kv_collection_constructor=FetchPagedKVCacheCollection(
                config.kv_params
            ),
            rope=rope,
            devices=config.devices,
            use_subgraphs=True,
            subgraph_layer_groups=[
                [
                    i
                    for i in range(
                        config.first_k_dense_replace, config.num_hidden_layers
                    )
                ]
            ],
        )

    def _get_mlp(self, config: DeepseekV2Config, idx: int) -> MLP | MoE:
        """Helper function to return a mixture of experts layer or traditional multi-layer perceptron layer
        for the TransformerBlock's mlp depending on the layer idx.

        Args:
            config: Configuration object containing model parameters
            i: Layer index

        Returns:
            List of MLP shards or MoE modules depending on the layer index and config
        """
        if (
            config.n_routed_experts is not None
            and idx >= config.first_k_dense_replace
            and idx % config.moe_layer_freq == 0
        ):
            moe = MoE(
                devices=config.devices,
                hidden_dim=config.hidden_size,
                num_experts=config.n_routed_experts,
                num_experts_per_token=config.num_experts_per_tok,
                moe_dim=config.moe_intermediate_size,
                gate_cls=DeepSeekV2MoEGate,
                has_shared_experts=True,
                shared_experts_dim=config.n_shared_experts
                * config.moe_intermediate_size,
                dtype=config.dtype,
            )
            moe.sharding_strategy = ShardingStrategy.tensor_parallel(
                len(config.devices)
            )
            return moe
        else:
            mlp = MLP(
                dtype=config.dtype,
                quantization_encoding=None,
                hidden_dim=config.hidden_size,
                feed_forward_length=config.intermediate_size,
                devices=config.devices,
            )
            mlp.sharding_strategy = ShardingStrategy.tensor_parallel(
                len(config.devices)
            )
            return mlp
