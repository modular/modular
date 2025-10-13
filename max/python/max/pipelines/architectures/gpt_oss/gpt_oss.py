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

"""Implements the GPT OSS model."""

from __future__ import annotations

import functools
from collections.abc import Sequence

from max.dtype import DType
from max.graph import (
    BufferValue,
    ShardingStrategy,
    TensorValue,
    ops,
)
from max.nn import (
    ColumnParallelLinear,
    Embedding,
    LayerList,
    Module,
)
from max.nn.kv_cache import PagedCacheValues
from max.nn.norm.rms_norm import RMSNorm
from max.nn.rotary_embedding import (
    YarnRotaryEmbedding,
    YarnScalingParams,
)

from .layers.attention import GptOssAttention
from .layers.moe import GptOssMoE
from .layers.transformer_block import GptOssTransformerBlock
from .model_config import GptOssConfig


class GptOssTextModel(Module):
    """The GPT OSS language model.

    Decoder-only Transformer with MoE feed-forward, rotary embeddings (YARN),
    and mixed attention (full + sliding window).
    """

    def __init__(self, config: GptOssConfig) -> None:
        super().__init__()
        self.devices = config.devices

        # Create YARN scaling params if configured
        assert config.rope_scaling is not None, (
            "RoPE scaling is required for GPT-OSS models"
        )
        assert isinstance(config.rope_scaling, YarnScalingParams), (
            "Only YARN scaling is supported for GPT-OSS models"
        )
        yarn_scaling_params: YarnScalingParams = config.rope_scaling

        # RoPE with YARN scaling for full and window attention layers
        rope = YarnRotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_position_embeddings,
            device=config.devices[0],
            head_dim=config.head_dim,
            interleaved=False,
            scaling_params=yarn_scaling_params,
        )
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.dtype,
            device=config.devices[0],
        )

        self.norm = RMSNorm(
            config.hidden_size,
            config.dtype,
            config.rms_norm_eps,
            multiply_before_cast=True,
        )
        self.norm.sharding_strategy = ShardingStrategy.replicate(
            len(config.devices)
        )
        self.norm_shards = self.norm.shard(config.devices)

        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            dtype=config.dtype,
            devices=config.devices,
            tied_weight=(
                self.embed_tokens.weight if config.tie_word_embeddings else None
            ),
        )

        create_norm = functools.partial(
            RMSNorm,
            config.hidden_size,
            config.dtype,
            eps=config.rms_norm_eps,
            multiply_before_cast=True,
        )

        layers = [
            GptOssTransformerBlock(
                attention=GptOssAttention(
                    rope=rope,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    hidden_size=config.hidden_size,
                    kv_params=config.kv_params,
                    layer_idx=i,
                    dtype=config.dtype,
                    devices=config.devices,
                    local_window_size=config.sliding_window,
                    has_bias=config.attention_bias,
                    layer_type=config.layer_types[i]
                    if i < len(config.layer_types)
                    else "full_attention",
                ),
                mlp=GptOssMoE(config),
                input_layernorm=create_norm(),
                post_attention_layernorm=create_norm(),
                devices=config.devices,
            )
            for i in range(config.num_hidden_layers)
        ]

        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.layers = LayerList(layers)
        self.kv_params = config.kv_params
        self.return_logits = config.return_logits

    def __call__(
        self,
        tokens: TensorValue,
        signal_buffers: Sequence[BufferValue],
        kv_collections: Sequence[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: Sequence[TensorValue],
        **kwargs,
    ) -> tuple[TensorValue, ...]:
        h_embed = self.embed_tokens(tokens)
        # Replicate embedding output to all devices
        h = [h_embed.to(device) for device in self.devices]

        # Run through transformer layers
        for idx, layer in enumerate(self.layers):
            layer_idx_tensor = ops.constant(
                idx, DType.uint32, device=self.devices[0]
            )
            h = layer(
                layer_idx_tensor,
                h,
                signal_buffers,
                kv_collections,
                input_row_offsets=input_row_offsets,
                **kwargs,
            )

        # Get last token logits only (no variable logits support).
        last_token_indices = [offsets[1:] - 1 for offsets in input_row_offsets]
        last_token_h = []
        if h:
            last_token_h = [
                ops.gather(h_device, indices, axis=0)
                for h_device, indices in zip(h, last_token_indices, strict=True)
            ]
        last_logits = ops.cast(
            # Take only the device 0 logits to device-to-host transfer.
            self.lm_head(
                [
                    self.norm_shards[i](last_token_h[i])
                    for i in range(len(last_token_h))
                ],
                signal_buffers,
            )[0],
            DType.float32,
        )

        # For now, simplified to return last token only
        # TODO: Handle VARIABLE and ALL logits cases for distributed processing
        return (last_logits,)


class GptOss(Module):
    """The GPT OSS model."""

    def __init__(self, config: GptOssConfig) -> None:
        super().__init__()
        self.language_model = GptOssTextModel(config)

    def __call__(
        self,
        tokens: TensorValue,
        signal_buffers: Sequence[BufferValue],
        kv_cache_inputs_per_dev: Sequence[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: Sequence[TensorValue],
    ) -> tuple[TensorValue, ...]:
        return self.language_model(
            tokens,
            signal_buffers,
            kv_cache_inputs_per_dev,
            return_n_logits,
            input_row_offsets,
        )
