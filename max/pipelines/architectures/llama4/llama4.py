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
from max.graph import BufferValue, DeviceRef, TensorValue, TensorValueLike, ops
from max.nn import (
    ColumnParallelLinear,
    DistributedMLP,
    DistributedRMSNorm,
    Llama3RotaryEmbedding,
    Module,
    ReturnLogits,
    VocabParallelEmbedding,
)
from max.nn.kv_cache import (
    ContinuousBatchingKVCacheCollection,
    FetchPagedKVCacheCollection,
    PagedKVCacheCollection,
)
from max.nn.layer import LayerList

from .layers.attention import Llama4TextAttention
from .layers.moe import DistributedMoE
from .model_config import Llama4Config


def distribute_value(v, devices: list[DeviceRef]):
    return [v.to(device) for device in devices]


class Llama4DecoderLayer(Module):
    """Llama4 decoder attention block."""

    def __init__(
        self,
        rope: Llama3RotaryEmbedding,
        config: Llama4Config,
        layer_idx: int,
        devices: list[DeviceRef],
    ):
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
            self.feed_forward = DistributedMoE(
                hidden_dim=config.hidden_size,
                top_k=config.num_experts_per_tok,
                num_experts=config.num_local_experts,
                intermediate_size=config.intermediate_size,
                intermediate_size_mlp=config.intermediate_size_mlp,
                dtype=config.dtype,
                devices=config.devices,
            )
        else:
            self.feed_forward = DistributedMLP(
                config.dtype,
                quantization_encoding=None,
                hidden_dim=config.hidden_size,
                feed_forward_length=config.intermediate_size_mlp,
                devices=config.devices,
            )
        self.input_layernorm = DistributedRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.dtype,
            devices=config.devices,
        )
        self.post_attention_layernorm = DistributedRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.dtype,
            devices=config.devices,
        )
        self.devices = devices

    def __call__(
        self,
        xs: list[TensorValue],
        distributed_cache_positions: list[TensorValue],
        signal_buffers: list[BufferValue],
        kv_collections: list[
            ContinuousBatchingKVCacheCollection | PagedKVCacheCollection
        ],
        **kwargs,
    ) -> list[TensorValue]:
        attn_outs = self.self_attn(
            self.input_layernorm(xs),
            distributed_cache_positions,
            kv_collections,
            signal_buffers=signal_buffers,
            **kwargs,
        )

        hidden_states = [x + attn_out for x, attn_out in zip(xs, attn_outs)]
        post_norm_states = self.post_attention_layernorm(hidden_states)

        mlp_outs = self.feed_forward(
            post_norm_states, signal_buffers=signal_buffers
        )
        hidden_states = [
            h + mlp_out for h, mlp_out in zip(hidden_states, mlp_outs)
        ]
        return hidden_states


class Llama4TextModel(Module):
    """The Llama4 text transformer model."""

    def __init__(self, config: Llama4Config):
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
        self.norm = DistributedRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.dtype,
            devices=config.devices,
        )
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
            cache_positions, self.devices
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
        last_logits = ops.cast(
            self.lm_head(self.norm(last_token_distributed))[0], DType.float32
        )

        logits = None
        offsets = None

        if self.return_logits == ReturnLogits.ALL:
            logits = ops.cast(self.lm_head(self.norm(h))[0], DType.float32)
            offsets = cast(TensorValue, kwargs["input_row_offsets"])

        if logits is not None and offsets is not None:
            return (last_logits, logits, offsets)
        else:
            return (last_logits,)


class Llama4(Module):
    """The Llama4 model (currently text-only)."""

    def __init__(self, config: Llama4Config):
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
