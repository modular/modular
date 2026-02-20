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
from max.nn.legacy.embedding import Embedding
from max.nn.legacy.kv_cache import PagedCacheValues
from max.nn.legacy.layer import LayerList, Module
from max.nn.legacy.linear import ColumnParallelLinear
from max.nn.legacy.norm.rms_norm import RMSNorm
from max.nn.legacy.rotary_embedding import (
    YarnRotaryEmbedding,
    YarnScalingParams,
)
from max.nn.legacy.transformer.distributed_transformer import (
    DistributedLogitsPostprocessMixin,
)

from .layers.attention import GptOssAttention
from .layers.moe import GptOssMoE
from .layers.transformer_block import GptOssTransformerBlock
from .model_config import GptOssConfig


class GptOssTextModel(DistributedLogitsPostprocessMixin, Module):
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

        return self._postprocess_logits(
            h, input_row_offsets, return_n_logits, signal_buffers
        )


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
