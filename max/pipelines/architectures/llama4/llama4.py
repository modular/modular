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
"""Implements the Llama4 model."""

from __future__ import annotations

from typing import cast

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    TensorValueLike,
    ops,
)
from max.nn import (
    MLP,
    Allreduce,
    ColumnParallelLinear,
    Llama3RotaryEmbedding,
    Module,
    ReturnLogits,
    RMSNorm,
    VocabParallelEmbedding,
)
from max.nn.kv_cache import (
    FetchPagedKVCacheCollection,
    PagedKVCacheCollection,
)
from max.nn.layer import LayerList

from .layers.attention import Llama4TextAttention
from .layers.moe import DistributedLlama4MoE, Llama4MoEGate
from .model_config import Llama4Config


def distribute_value(
    v: TensorValue, devices: list[DeviceRef]
) -> list[TensorValue]:
    return [v.to(device) for device in devices]


class Llama4DecoderLayer(Module):
    """Llama4 decoder attention block."""

    def __init__(
        self,
        rope: Llama3RotaryEmbedding,
        config: Llama4Config,
        layer_idx: int,
        devices: list[DeviceRef],
    ) -> None:
        super().__init__()
        is_nope_layer = (layer_idx + 1) % config.no_rope_layer_interval == 0
        use_rope = not is_nope_layer
        self.self_attn = Llama4TextAttention(
            rope=rope,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            hidden_size=config.hidden_size,
            kv_params=config.kv_params,
            layer_idx=layer_idx,
            dtype=config.dtype,
            attn_temperature_tuning=config.attn_temperature_tuning,
            floor_scale=config.floor_scale,
            attn_scale=config.attn_scale,
            devices=config.devices,
            use_rope=use_rope,
            use_qk_norm=config.use_qk_norm,
            qk_norm_eps=config.rms_norm_eps,
        )
        self.is_moe_layer = layer_idx in config.moe_layers
        self.feed_forward: Module
        if self.is_moe_layer:
            self.feed_forward = DistributedLlama4MoE(
                devices=config.devices,
                hidden_dim=config.hidden_size,
                num_experts=config.num_local_experts,
                num_experts_per_token=config.num_experts_per_tok,
                moe_dim=config.intermediate_size,
                gate_cls=Llama4MoEGate,
                has_shared_experts=True,
                shared_experts_dim=config.intermediate_size,
                dtype=config.dtype,
                apply_router_weight_first=True,
            )
        else:
            self.feed_forward = MLP(
                config.dtype,
                quantization_encoding=None,
                hidden_dim=config.hidden_size,
                feed_forward_length=config.intermediate_size_mlp,
                devices=config.devices,
            )
            self.feed_forward.sharding_strategy = (
                ShardingStrategy.tensor_parallel(len(config.devices))
            )
            self.feed_forward_shards = self.feed_forward.shard(config.devices)
            self.feed_forward_allreduce = Allreduce(
                num_accelerators=len(config.devices)
            )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.dtype,
            multiply_before_cast=False,
        )
        self.input_layernorm.sharding_strategy = ShardingStrategy.replicate(
            len(config.devices)
        )
        self.input_layernorm_shards = self.input_layernorm.shard(config.devices)

        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.dtype,
            multiply_before_cast=False,
        )
        self.post_attention_layernorm.sharding_strategy = (
            ShardingStrategy.replicate(len(config.devices))
        )
        self.post_attention_layernorm_shards = (
            self.post_attention_layernorm.shard(config.devices)
        )
        self.devices = devices

    def __call__(
        self,
        xs: list[TensorValue],
        distributed_cache_positions: list[TensorValue],
        signal_buffers: list[BufferValue],
        kv_collections: list[PagedKVCacheCollection],
        **kwargs,
    ) -> list[TensorValue]:
        # Apply input layer norm to each shard
        norm_xs = [
            self.input_layernorm_shards[i](xs[i]) for i in range(len(xs))
        ]

        attn_outs = self.self_attn(
            norm_xs,
            distributed_cache_positions,
            kv_collections,
            signal_buffers=signal_buffers,
            **kwargs,
        )

        hidden_states = [x + attn_out for x, attn_out in zip(xs, attn_outs)]
        # Apply post attention layer norm to each shard
        post_norm_states = [
            self.post_attention_layernorm_shards[i](hidden_states[i])
            for i in range(len(hidden_states))
        ]

        if self.is_moe_layer:
            mlp_outs = self.feed_forward(
                post_norm_states, signal_buffers=signal_buffers
            )
        else:
            mlp_outs = [
                shard(x)
                for shard, x in zip(self.feed_forward_shards, post_norm_states)
            ]
            mlp_outs = self.feed_forward_allreduce(mlp_outs, signal_buffers)
        hidden_states = [
            h + mlp_out for h, mlp_out in zip(hidden_states, mlp_outs)
        ]
        return hidden_states


class Llama4TextModel(Module):
    """The Llama4 text transformer model."""

    def __init__(self, config: Llama4Config) -> None:
        super().__init__()
        self.rope = Llama3RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            interleaved=config.interleaved_rope_weights,
            scaling_params=config.rope_scaling_params,
            device=config.devices[0],
        )
        self.n_heads = config.num_attention_heads
        self.layers = LayerList(
            [
                Llama4DecoderLayer(self.rope, config, layer_idx, config.devices)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.dtype,
            multiply_before_cast=False,
        )
        self.norm.sharding_strategy = ShardingStrategy.replicate(
            len(config.devices)
        )
        self.norm_shards = self.norm.shard(config.devices)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            config.dtype,
            devices=config.devices,
            quantization_encoding=None,
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.dtype,
            config.devices,
            quantization_encoding=None,
        )
        self.kv_params = config.kv_params
        self.kv_collection_constructor = FetchPagedKVCacheCollection(
            config.kv_params, num_layers=config.num_hidden_layers
        )
        self.return_logits = config.return_logits
        self.devices = config.devices

        if config.return_logits == ReturnLogits.VARIABLE:
            raise ValueError(
                "llama4 does not currently support variable logits"
            )

    def __call__(
        self,
        tokens: TensorValueLike,
        cache_positions: TensorValueLike,
        signal_buffers: list[BufferValue],
        kv_cache_inputs_per_dev: list[tuple[TensorValue, ...]],
        **kwargs,
    ) -> tuple[TensorValue, ...]:
        h = self.embed_tokens(tokens, signal_buffers)

        kv_collections = [
            self.kv_collection_constructor(*kv_cache_inputs)
            for kv_cache_inputs in kv_cache_inputs_per_dev
        ]

        input_row_offsets = kwargs["input_row_offsets"]
        distributed_cache_positions = distribute_value(
            TensorValue(cache_positions), self.devices
        )
        for _, layer in enumerate(self.layers):
            h = layer(
                h,
                distributed_cache_positions,
                signal_buffers,
                kv_collections,
                **kwargs,
            )

        h0 = h[0]
        last_token_indices = input_row_offsets[1:] - 1
        last_token_h = ops.gather(h0, last_token_indices, axis=0)
        last_token_distributed = distribute_value(last_token_h, self.devices)
        # Apply norm to each shard
        norm_last_token = [
            self.norm_shards[i](last_token_distributed[i])
            for i in range(len(self.devices))
        ]
        last_logits = ops.cast(
            self.lm_head(norm_last_token, signal_buffers)[0],
            DType.float32,
        )

        logits = None
        offsets = None

        if self.return_logits == ReturnLogits.ALL:
            logits = ops.cast(
                self.lm_head(
                    [
                        self.norm_shards[i](h[i])
                        for i in range(len(self.devices))
                    ],
                    signal_buffers,
                )[0],
                DType.float32,
            )
            offsets = cast(TensorValue, kwargs["input_row_offsets"])

        if logits is not None and offsets is not None:
            return (last_logits, logits, offsets)
        else:
            return (last_logits,)


class Llama4(Module):
    """The Llama4 model (currently text-only)."""

    def __init__(self, config: Llama4Config) -> None:
        self.language_model = Llama4TextModel(config)

    def __call__(
        self,
        tokens: TensorValueLike,
        cache_positions: TensorValueLike,
        signal_buffers: list[BufferValue],
        kv_cache_inputs_per_dev: list[tuple[TensorValue, ...]],
        **kwargs,
    ) -> tuple[TensorValue, ...]:
        return self.language_model(
            tokens,
            cache_positions,
            signal_buffers,
            kv_cache_inputs_per_dev,
            **kwargs,
        )
