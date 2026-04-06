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

"""Implements the Gemma3 model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
)
from max.nn.comm.allreduce import Allreduce
from max.nn.kv_cache import PagedCacheValues
from max.nn.layer import Module
from max.nn.transformer.distributed_transformer import (
    ShardableCallable,
    forward_sharded_layers,
)
from max.pipelines.architectures.gemma3.layers.attention import Gemma3Attention
from max.pipelines.architectures.gemma3.layers.rms_norm import (
    Gemma3RMSNorm,
    gemma3_rms_norm_fused_residual_add,
)


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
        kv_collections: list[PagedCacheValues],
        input_row_offsets: list[TensorValue],
        normalized_xs: Sequence[TensorValue] | None = None,
        next_input_layernorm_shards: Sequence[Gemma3RMSNorm] | None = None,
        **kwargs,
    ) -> tuple[list[TensorValue], list[TensorValue] | None]:
        residual = xs
        norm_xs = (
            list(normalized_xs)
            if normalized_xs is not None
            else forward_sharded_layers(self.input_layernorm_shards, xs)
        )
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

        fused_attn_norm = [
            gemma3_rms_norm_fused_residual_add(
                attn_out[i],
                residual[i],
                cast(
                    Gemma3RMSNorm, self.post_attention_layernorm_shards[i]
                ),
                cast(
                    Gemma3RMSNorm, self.pre_feedforward_layernorm_shards[i]
                ),
            )
            for i in range(len(attn_out))
        ]
        norm_xs = [fused_output for fused_output, _ in fused_attn_norm]
        residual = [fused_residual for _, fused_residual in fused_attn_norm]

        hidden_states = forward_sharded_layers(self.mlp_shards, norm_xs)
        hidden_states = self.allreduce(hidden_states, signal_buffers)

        if next_input_layernorm_shards is None:
            hidden_states = forward_sharded_layers(
                self.post_feedforward_layernorm_shards, hidden_states
            )
            return (
                [
                    residual[i] + hidden_states[i]
                    for i in range(len(hidden_states))
                ],
                None,
            )

        fused_mlp_norm = [
            gemma3_rms_norm_fused_residual_add(
                hidden_states[i],
                residual[i],
                cast(
                    Gemma3RMSNorm, self.post_feedforward_layernorm_shards[i]
                ),
                next_input_layernorm_shards[i],
            )
            for i in range(len(hidden_states))
        ]
        return (
            [fused_residual for _, fused_residual in fused_mlp_norm],
            [fused_output for fused_output, _ in fused_mlp_norm],
        )
