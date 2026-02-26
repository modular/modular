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

import functools
from collections.abc import Sequence

from max.dtype import DType
from max.graph import BufferValue, ShardingStrategy, TensorValue, ops
from max.nn.kv_cache import PagedCacheValues
from max.nn.layer import LayerList, Module
from max.nn.linear import MLP, ColumnParallelLinear
from max.nn.rotary_embedding import (
    Llama3RopeScalingParams,
    Llama3RotaryEmbedding,
)
from max.nn.transformer.distributed_transformer import (
    DistributedLogitsPostprocessMixin,
)

from .layers.attention import Gemma3Attention
from .layers.rms_norm import Gemma3RMSNorm
from .layers.scaled_word_embedding import ScaledWordEmbedding
from .layers.transformer_block import Gemma3TransformerBlock
from .model_config import Gemma3Config


class Gemma3TextModel(DistributedLogitsPostprocessMixin, Module):
    """The Gemma 3 language model."""

    def __init__(self, config: Gemma3Config) -> None:
        super().__init__()
        self.devices = config.devices
        # Use scaling_params for both cases (with and without scaling)
        scaling_params = (
            Llama3RopeScalingParams(
                factor=config.rope_scaling.factor,
                low_freq_factor=1e38,  # This degenerates to linear scaling
                high_freq_factor=1e38,
                orig_max_position=config.max_position_embeddings,
            )
            if config.rope_scaling is not None
            else None
        )

        rope_global = Llama3RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_position_embeddings,
            head_dim=config.head_dim,
            interleaved=False,
            scaling_params=scaling_params,
        )

        # rope_local doesn't use scaling
        rope_local = Llama3RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_local_base_freq,
            max_seq_len=config.max_position_embeddings,
            head_dim=config.head_dim,
            interleaved=False,
            scaling_params=None,  # No scaling
        )

        embedding_output_dtype = config.dtype
        if config.float8_config and config.float8_config.embedding_output_dtype:
            embedding_output_dtype = config.float8_config.embedding_output_dtype

        self.embed_tokens = ScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            embedding_output_dtype,
            config.devices,
            embed_scale=config.hidden_size**0.5,
        )

        self.norm = Gemma3RMSNorm(
            config.hidden_size,
            DType.bfloat16,
            config.rms_norm_eps,
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
            Gemma3RMSNorm,
            config.hidden_size,
            DType.bfloat16,
            eps=config.rms_norm_eps,
        )

        layers = [
            Gemma3TransformerBlock(
                attention=Gemma3Attention(
                    rope_global=rope_global,
                    rope_local=rope_local,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    hidden_size=config.hidden_size,
                    kv_params=config.kv_params,
                    layer_idx=i,
                    dtype=config.dtype,
                    devices=config.devices,
                    qk_norm_eps=config.rms_norm_eps,
                    local_window_size=config.sliding_window,
                    float8_config=config.float8_config,
                ),
                mlp=MLP(
                    dtype=config.dtype,
                    quantization_encoding=None,
                    hidden_dim=config.hidden_size,
                    feed_forward_length=config.intermediate_size,
                    devices=config.devices,
                    activation_function=config.hidden_activation,
                    float8_config=config.float8_config,
                ),
                input_layernorm=create_norm(),
                post_attention_layernorm=create_norm(),
                pre_feedforward_layernorm=create_norm(),
                post_feedforward_layernorm=create_norm(),
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
        h = self.embed_tokens(tokens, signal_buffers)

        # Create KV cache collections per device

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


class Gemma3(Module):
    """The Gemma model (currently text-only)."""

    def __init__(self, config: Gemma3Config) -> None:
        super().__init__()
        self.language_model = Gemma3TextModel(config)

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
