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

"""Gemma4 Decoder Layer."""

from __future__ import annotations

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    Weight,
)
from max.nn.activation import activation_function_from_name
from max.nn.comm.allreduce import Allreduce
from max.nn.kv_cache import PagedCacheValues
from max.nn.layer import Module
from max.nn.linear import Linear
from max.nn.moe.moe import MoE
from max.nn.quant_config import QuantConfig
from max.nn.transformer.distributed_transformer import (
    ShardableCallable,
    forward_sharded_layers,
)
from max.pipelines.architectures.gemma4.layers.attention import (
    Gemma4Attention,
)
from max.pipelines.architectures.gemma4.layers.rms_norm import Gemma4RMSNorm


class Gemma4TextDecoderLayer(Module):
    """Gemma4 decoder layer: Attention + FeedForward with pre/post norms.

    Mirrors the HuggingFace ``Gemma4TextDecoderLayer`` structure:

    1. ``input_layernorm`` -> attention -> ``post_attention_layernorm`` -> residual
    2. ``pre_feedforward_layernorm`` -> MLP -> ``post_feedforward_layernorm`` -> residual
       (or MoE variant with parallel shared-MLP + routed-expert branches)
    3. Per-layer input (when the checkpoint has one) -> residual
    4. Multiply by ``layer_scalar``
    """

    def __init__(
        self,
        attention: Gemma4Attention,
        mlp: ShardableCallable,
        hidden_size: int,
        rms_norm_eps: float,
        devices: list[DeviceRef],
        unquantized_dtype: DType = DType.bfloat16,
        enable_moe_block: bool = False,
        moe_block: MoE | None = None,
        hidden_size_per_layer_input: int = 0,
        hidden_activation: str = "gelu_tanh",
        quant_config: QuantConfig | None = None,
    ) -> None:
        super().__init__()

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

        self.input_layernorm = Gemma4RMSNorm(
            hidden_size, unquantized_dtype, eps=rms_norm_eps
        )
        self.input_layernorm.sharding_strategy = ShardingStrategy.replicate(
            len(devices)
        )
        self.input_layernorm_shards = self.input_layernorm.shard(devices)

        self.post_attention_layernorm = Gemma4RMSNorm(
            hidden_size, unquantized_dtype, eps=rms_norm_eps
        )
        self.post_attention_layernorm.sharding_strategy = (
            ShardingStrategy.replicate(len(devices))
        )
        self.post_attention_layernorm_shards = (
            self.post_attention_layernorm.shard(devices)
        )

        self.pre_feedforward_layernorm = Gemma4RMSNorm(
            hidden_size, unquantized_dtype, eps=rms_norm_eps
        )
        self.pre_feedforward_layernorm.sharding_strategy = (
            ShardingStrategy.replicate(len(devices))
        )
        self.pre_feedforward_layernorm_shards = (
            self.pre_feedforward_layernorm.shard(devices)
        )

        self.post_feedforward_layernorm = Gemma4RMSNorm(
            hidden_size, unquantized_dtype, eps=rms_norm_eps
        )
        self.post_feedforward_layernorm.sharding_strategy = (
            ShardingStrategy.replicate(len(devices))
        )
        self.post_feedforward_layernorm_shards = (
            self.post_feedforward_layernorm.shard(devices)
        )

        # Gemma 4 per-layer embeddings (PLE): the layer mixes its slice of
        # the packed per-layer embedding into its residual stream. The two
        # projections are quantized when the checkpoint is (a quantized
        # export ships ``weight_scale``/``weight_scale_2``/``input_scale``
        # next to their weights), and stay plain BF16 linears when it is not.
        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        if hidden_size_per_layer_input:
            self.per_layer_activation = activation_function_from_name(
                hidden_activation
            )

            self.per_layer_input_gate = Linear(
                in_dim=hidden_size,
                out_dim=hidden_size_per_layer_input,
                dtype=unquantized_dtype,
                device=devices[0],
                has_bias=False,
                quant_config=quant_config,
            )
            self.per_layer_input_gate.sharding_strategy = (
                ShardingStrategy.replicate(len(devices))
            )
            self.per_layer_input_gate_shards = self.per_layer_input_gate.shard(
                devices
            )

            self.per_layer_projection = Linear(
                in_dim=hidden_size_per_layer_input,
                out_dim=hidden_size,
                dtype=unquantized_dtype,
                device=devices[0],
                has_bias=False,
                quant_config=quant_config,
            )
            self.per_layer_projection.sharding_strategy = (
                ShardingStrategy.replicate(len(devices))
            )
            self.per_layer_projection_shards = self.per_layer_projection.shard(
                devices
            )

            self.post_per_layer_input_norm = Gemma4RMSNorm(
                hidden_size, unquantized_dtype, eps=rms_norm_eps
            )
            self.post_per_layer_input_norm.sharding_strategy = (
                ShardingStrategy.replicate(len(devices))
            )
            self.post_per_layer_input_norm_shards = (
                self.post_per_layer_input_norm.shard(devices)
            )

        self.enable_moe_block = enable_moe_block
        self.moe_block = moe_block

        if self.enable_moe_block:
            assert self.moe_block is not None
            self.moe_block.sharding_strategy = ShardingStrategy.tensor_parallel(
                len(devices)
            )
            self.moe_block_shards = self.moe_block.shard(devices)

            self.post_feedforward_layernorm_1 = Gemma4RMSNorm(
                hidden_size, unquantized_dtype, eps=rms_norm_eps
            )
            self.post_feedforward_layernorm_1.sharding_strategy = (
                ShardingStrategy.replicate(len(devices))
            )
            self.post_feedforward_layernorm_1_shards = (
                self.post_feedforward_layernorm_1.shard(devices)
            )

            self.post_feedforward_layernorm_2 = Gemma4RMSNorm(
                hidden_size, unquantized_dtype, eps=rms_norm_eps
            )
            self.post_feedforward_layernorm_2.sharding_strategy = (
                ShardingStrategy.replicate(len(devices))
            )
            self.post_feedforward_layernorm_2_shards = (
                self.post_feedforward_layernorm_2.shard(devices)
            )

        self.devices = devices
        self.allreduce = Allreduce(num_accelerators=len(devices))

        self.layer_scalar = Weight(
            "layer_scalar", unquantized_dtype, shape=[1], device=DeviceRef.CPU()
        )

    def __call__(
        self,
        layer_idx: TensorValue,
        xs: list[TensorValue],
        signal_buffers: list[BufferValue],
        kv_collections: list[PagedCacheValues],
        input_row_offsets: list[TensorValue],
        per_layer_inputs: list[TensorValue] | None = None,
        **kwargs: object,
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

        if self.enable_moe_block:
            mlp_normed = forward_sharded_layers(
                self.pre_feedforward_layernorm_shards, hidden_states
            )
            mlp_out = forward_sharded_layers(self.mlp_shards, mlp_normed)
            mlp_out = self.allreduce(mlp_out, signal_buffers)
            mlp_out = forward_sharded_layers(
                self.post_feedforward_layernorm_1_shards, mlp_out
            )

            # Experts are tensor-parallel sharded, so each device computes
            # a partial result; allreduce sums them into the full output.
            moe_out = forward_sharded_layers(
                self.moe_block_shards, hidden_states
            )
            moe_out = self.allreduce(moe_out, signal_buffers)
            moe_out = forward_sharded_layers(
                self.post_feedforward_layernorm_2_shards, moe_out
            )

            hidden_states = [
                mlp_out[i] + moe_out[i] for i in range(len(mlp_out))
            ]
        else:
            norm_xs = forward_sharded_layers(
                self.pre_feedforward_layernorm_shards, hidden_states
            )
            hidden_states = forward_sharded_layers(self.mlp_shards, norm_xs)
            hidden_states = self.allreduce(hidden_states, signal_buffers)

        hidden_states = forward_sharded_layers(
            self.post_feedforward_layernorm_shards, hidden_states
        )
        hidden_states = [
            residual[i] + hidden_states[i] for i in range(len(hidden_states))
        ]

        if self.hidden_size_per_layer_input:
            assert per_layer_inputs is not None, (
                "A model with per-layer embeddings always passes them."
            )
            gated = forward_sharded_layers(
                self.per_layer_input_gate_shards, hidden_states
            )
            gated = [self.per_layer_activation(g) for g in gated]
            gated = [
                g * p for g, p in zip(gated, per_layer_inputs, strict=True)
            ]
            projected = forward_sharded_layers(
                self.per_layer_projection_shards, gated
            )
            projected = forward_sharded_layers(
                self.post_per_layer_input_norm_shards, projected
            )
            hidden_states = [
                h + p for h, p in zip(hidden_states, projected, strict=True)
            ]

        hidden_states = [
            h * self.layer_scalar.to(h.device) for h in hidden_states
        ]

        return hidden_states
