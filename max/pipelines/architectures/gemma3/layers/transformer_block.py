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

"""Implements the Gemma3 model."""

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
from max.pipelines.architectures.gemma3.layers.attention import Gemma3Attention


class Gemma3TransformerBlock(Module):
    """Stack of Attention, FeedForward, and RMSNorm layers.

    Unlike the transformer block in the `max.nn` library, this class applies
    normalizations to the hidden states immediately after the attention, and
    before and after the feedforward layers.
    """

    def __init__(
        self,
        attention: Gemma3Attention,
        mlp: ShardableCallable,
        input_layernorm: ShardableCallable,
        post_attention_layernorm: ShardableCallable,
        pre_feedforward_layernorm: ShardableCallable,
        post_feedforward_layernorm: ShardableCallable,
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

        self.pre_feedforward_layernorm = pre_feedforward_layernorm
        self.pre_feedforward_layernorm.sharding_strategy = (
            ShardingStrategy.replicate(len(devices))
        )
        self.pre_feedforward_layernorm_shards = pre_feedforward_layernorm.shard(
            devices
        )

        self.post_feedforward_layernorm = post_feedforward_layernorm
        self.post_feedforward_layernorm.sharding_strategy = (
            ShardingStrategy.replicate(len(devices))
        )
        self.post_feedforward_layernorm_shards = (
            post_feedforward_layernorm.shard(devices)
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

        hidden_states = forward_sharded_layers(
            self.post_attention_layernorm_shards, attn_out
        )
        hidden_states = [
            residual[i] + hidden_states[i] for i in range(len(hidden_states))
        ]

        residual = hidden_states
        norm_xs = forward_sharded_layers(
            self.pre_feedforward_layernorm_shards, hidden_states
        )

        hidden_states = forward_sharded_layers(self.mlp_shards, norm_xs)
        hidden_states = self.allreduce(hidden_states, signal_buffers)

        hidden_states = forward_sharded_layers(
            self.post_feedforward_layernorm_shards, hidden_states
        )
        return [
            residual[i] + hidden_states[i] for i in range(len(hidden_states))
        ]
