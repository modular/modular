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

"""Implements the GPT OSS transformer block."""

from __future__ import annotations

from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
)
from max.nn import Module
from max.nn.comm.allreduce import Allreduce
from max.nn.kv_cache import PagedKVCacheCollection
from max.nn.transformer.distributed_transformer import (
    ShardableCallable,
    forward_sharded_layers,
)
from max.pipelines.architectures.gpt_oss.layers.attention import GptOssAttention
from max.pipelines.architectures.gpt_oss.layers.moe import GptOssMoE


class GptOssTransformerBlock(Module):
    """Stack of Attention, MoE, and RMSNorm layers for GPT OSS.

    This is a distributed transformer block that uses a Mixture of Experts (MoE)
    layer instead of a standard feedforward network.
    Block's attention type (full or window) is specified in the model config.
    """

    def __init__(
        self,
        attention: GptOssAttention,
        mlp: GptOssMoE,
        input_layernorm: ShardableCallable,
        post_attention_layernorm: ShardableCallable,
        devices: list[DeviceRef],
    ) -> None:
        super().__init__()

        # TODO: Figure out a better way to indicate to the type checker that these
        # are Shardable Modules. (Probably need a protocol called ShardableModule)
        self.self_attn = attention
        self.self_attn.sharding_strategy = ShardingStrategy.tensor_parallel(
            len(devices)
        )
        self.self_attn_shards = attention.shard(devices)

        self.mlp = mlp
        self.mlp.sharding_strategy = ShardingStrategy.tensor_parallel(
            len(devices)
        )
        self.mlp_shards = mlp.shard(devices)

        self.input_layernorm = input_layernorm
        self.input_layernorm.sharding_strategy = ShardingStrategy.replicate(
            len(devices)
        )
        self.input_layernorm_shards = input_layernorm.shard(devices)

        self.post_attention_layernorm = post_attention_layernorm
        self.post_attention_layernorm.sharding_strategy = (
            ShardingStrategy.replicate(len(devices))
        )
        self.post_attention_layernorm_shards = post_attention_layernorm.shard(
            devices
        )

        self.devices = devices
        self.allreduce = Allreduce(num_accelerators=len(devices))

    def __call__(
        self,
        layer_idx: TensorValue,
        xs: list[TensorValue],
        signal_buffers: list[BufferValue],
        kv_collections: list[PagedKVCacheCollection],
        input_row_offsets: list[TensorValue],
        **kwargs,
    ) -> list[TensorValue]:
        residual = xs
        norm_xs = forward_sharded_layers(self.input_layernorm_shards, xs)
        attn_out = [
            shard(
                norm_xs[i],
                kv_collections[i],
                input_row_offsets=input_row_offsets[i],
                **kwargs,
            )
            for i, shard in enumerate(self.self_attn_shards)
        ]
        attn_out = self.allreduce(attn_out, signal_buffers)

        # Add residual connection after attention
        hidden_states = [
            residual[i] + attn_out[i] for i in range(len(attn_out))
        ]

        # Apply post-attention layer norm and then MoE
        residual = hidden_states
        norm_xs = forward_sharded_layers(
            self.post_attention_layernorm_shards, hidden_states
        )

        # Apply MoE - it returns (output, router_logits)
        mlp_results = [
            self.mlp_shards[i](norm_xs[i]) for i in range(len(norm_xs))
        ]

        # Separate outputs and router logits
        mlp_outputs = [result[0] for result in mlp_results]

        # Allreduce MoE outputs
        mlp_outputs = self.allreduce(mlp_outputs, signal_buffers)

        # Add residual connection
        hidden_states = [
            residual[i] + mlp_outputs[i] for i in range(len(mlp_outputs))
        ]
        return hidden_states
