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

import functools
from collections.abc import Sequence

from max.dtype import DType
from max.graph import TensorValue, TensorValueLike, ops
from max.nn import MLP, LayerList, Linear, Module
from max.nn.kv_cache import FetchPagedKVCacheCollection
from max.nn.rotary_embedding import (
    Llama3RopeScalingParams,
    Llama3RotaryEmbedding,
)

from .layers.attention import _Gemma3Attention as Gemma3Attention
from .layers.rms_norm import Gemma3RMSNorm
from .layers.scaled_word_embedding import ScaledWordEmbedding
from .model_config import Gemma3Config


class TransformerBlock(Module):
    """Stack of Attention, FeedForward, and RMSNorm layers.

    Unlike the transformer block in the `max.nn` library, this class applies
    normalizations to the hidden states immediately after the attention, and
    before and after the feedforward layers.
    """

    def __init__(
        self,
        attention: Module,
        mlp: Module,
        input_layernorm: Module,
        post_attention_layernorm: Module,
        pre_feedforward_layernorm: Module,
        post_feedforward_layernorm: Module,
    ) -> None:
        super().__init__()
        self.self_attn = attention
        self.mlp = mlp
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.pre_feedforward_layernorm = pre_feedforward_layernorm
        self.post_feedforward_layernorm = post_feedforward_layernorm

    def __call__(
        self,
        x: TensorValue,
        kv_collection: FetchPagedKVCacheCollection,
        **kwargs,
    ) -> TensorValue:
        residual = x
        attn_out = self.self_attn(
            self.input_layernorm(x), kv_collection, **kwargs
        )
        hidden_states = self.post_attention_layernorm(attn_out)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return residual + hidden_states


class Gemma3TextModel(Module):
    """The Gemma 3 language model."""

    def __init__(self, config: Gemma3Config) -> None:
        assert len(config.devices) == 1, (
            "Only single-device configuration is supported."
        )

        # Use Llama3RotaryEmbedding for both cases (with and without scaling)
        scaling_params = (
            Llama3RopeScalingParams(
                factor=config.rope_scaling.factor,
                low_freq_factor=1.0,  # No special scaling for low frequencies
                high_freq_factor=1.0,  # No special scaling for high frequencies
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
            device=config.devices[0],
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
            device=config.devices[0],
            head_dim=config.head_dim,
            interleaved=False,
            scaling_params=None,  # No scaling
        )

        self.embed_tokens = ScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.dtype,
            config.devices[0],
            embed_scale=config.hidden_size**0.5,
        )

        self.norm = Gemma3RMSNorm(
            config.hidden_size, config.dtype, config.rms_norm_eps
        )

        self.lm_head = Linear(
            config.hidden_size,
            config.vocab_size,
            dtype=config.dtype,
            device=config.devices[0],
        )

        if config.tie_word_embeddings:
            self.lm_head.set_shared_weight("weight", self.embed_tokens.weight)

        create_norm = functools.partial(
            Gemma3RMSNorm,
            config.hidden_size,
            config.dtype,
            eps=config.rms_norm_eps,
        )

        layers = [
            TransformerBlock(
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
                ),
                mlp=MLP(
                    dtype=config.dtype,
                    quantization_encoding=None,
                    hidden_dim=config.hidden_size,
                    feed_forward_length=config.intermediate_size,
                    devices=config.devices,
                    activation_function=config.hidden_activation,
                ),
                input_layernorm=create_norm(),
                post_attention_layernorm=create_norm(),
                pre_feedforward_layernorm=create_norm(),
                post_feedforward_layernorm=create_norm(),
            )
            for i in range(config.num_hidden_layers)
        ]

        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.layers = LayerList(layers)
        self.norm = self.norm
        self.lm_head = self.lm_head
        self.embed_tokens = self.embed_tokens
        self.kv_params = config.kv_params
        self.kv_collection_constructor = FetchPagedKVCacheCollection(
            config.kv_params
        )

    def __call__(
        self,
        tokens: TensorValueLike,
        kv_cache_inputs: Sequence[TensorValue],
        **kwargs,
    ) -> tuple[TensorValue, ...]:
        h = self.embed_tokens(tokens)

        kv_collection = self.kv_collection_constructor(*kv_cache_inputs)
        input_row_offsets = kwargs["input_row_offsets"]

        for layer in self.layers:
            h = layer(h, kv_collection, **kwargs)

        # Retrieve a variable number of tokens
        last_h = ops.gather(h, input_row_offsets[1:] - 1, axis=0)
        last_logits = ops.cast(self.lm_head(self.norm(last_h)), DType.float32)

        return (last_logits,)


class Gemma3(Module):
    """The Gemma model (currently text-only)."""

    def __init__(self, config: Gemma3Config) -> None:
        super().__init__()
        self.language_model = Gemma3TextModel(config)

    def __call__(
        self,
        tokens: TensorValue,
        input_row_offsets: TensorValue,
        kv_cache_inputs: Sequence[TensorValue],
        return_n_logits: TensorValue,  # Not used for Gemma3 currently
    ) -> tuple[TensorValue, ...]:
        return self.language_model(
            tokens,
            input_row_offsets=input_row_offsets,
            kv_cache_inputs=kv_cache_inputs,
        )
