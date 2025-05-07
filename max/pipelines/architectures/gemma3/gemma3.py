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

from collections.abc import Sequence

from max.graph import TensorValue
from max.nn import (
    MLP,
    Linear,
    Module,
    Transformer,
    TransformerBlock,
)
from max.nn.kv_cache import FetchPagedKVCacheCollection
from max.nn.rotary_embedding import OptimizedRotaryEmbedding

from .layers.attention import _Gemma3Attention as Gemma3Attention
from .layers.rms_norm import Gemma3RMSNorm
from .layers.scaled_word_embedding import ScaledWordEmbedding
from .model_config import Gemma3Config


class Gemma3TextModel(Transformer):
    """The Gemma 3 language model."""

    def __init__(self, config: Gemma3Config):
        assert len(config.devices) == 1, (
            "Only single-device configuration is supported."
        )

        rope = OptimizedRotaryEmbedding(
            dim=config.head_dim,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_position_embeddings,
            device=config.devices[0],
        )

        self.embed_tokens = ScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.dtype,
            config.devices[0],
        )

        self.norm = Gemma3RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

        self.lm_head = Linear(
            config.hidden_size,
            config.vocab_size,
            dtype=config.dtype,
            device=config.devices[0],
        )

        if config.tie_word_embeddings:
            self.lm_head.set_shared_weight("weight", self.embed_tokens.weight)

        layers = [
            TransformerBlock(
                attention=Gemma3Attention(
                    rope=rope,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    hidden_size=config.hidden_size,
                    kv_params=config.kv_params,
                    layer_idx=i,
                    dtype=config.dtype,
                    devices=config.devices,
                ),
                mlp=MLP(
                    dtype=config.dtype,
                    quantization_encoding=None,
                    hidden_dim=config.hidden_size,
                    feed_forward_length=config.intermediate_size,
                    devices=config.devices,
                ),
                attention_norm=Gemma3RMSNorm(
                    config.hidden_size,
                    config.rms_norm_eps,
                ),
                mlp_norm=Gemma3RMSNorm(
                    config.hidden_size,
                    config.rms_norm_eps,
                ),
            )
            for i in range(config.num_hidden_layers)
        ]

        super().__init__(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            layers=layers,
            norm=self.norm,
            output=self.lm_head,
            embedding=self.embed_tokens,
            kv_params=config.kv_params,
            kv_collection_constructor=FetchPagedKVCacheCollection(
                config.kv_params
            ),
        )


class Gemma3(Module):
    """The Gemma model (currently text-only)."""

    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.language_model = Gemma3TextModel(config)

    def __call__(
        self,
        tokens: TensorValue,
        input_row_offsets: TensorValue,
        kv_cache_inputs: Sequence[TensorValue],
        return_n_logits: TensorValue,
    ) -> tuple[TensorValue, ...]:
        return self.language_model(
            tokens,
            input_row_offsets=input_row_offsets,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=return_n_logits,
        )
