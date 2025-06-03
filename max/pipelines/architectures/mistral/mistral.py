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
"""Implements the Mistral model."""

from __future__ import annotations

from max.graph import DeviceRef
from max.nn import (
    MLP,
    AttentionWithRope,
    Embedding,
    Linear,
    OptimizedRotaryEmbedding,
    RMSNorm,
    Transformer,
    TransformerBlock,
)
from max.nn.kv_cache import (
    FetchContinuousBatchingKVCacheCollection,
    FetchPagedKVCacheCollection,
    KVCacheStrategy,
)

from .model_config import MistralConfig


class Mistral(Transformer):
    """Defines the Mistral transformer model."""

    def __init__(self, config: MistralConfig):
        assert len(config.devices) == 1

        # hidden_size (5120) != num_attention_heads * head_dim (128 * 40 = 4,096)
        rope = OptimizedRotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            interleaved=False,
            device=DeviceRef.GPU(),
        )

        layers = [
            TransformerBlock(
                attention=AttentionWithRope(
                    rope=rope,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    hidden_size=config.hidden_size,
                    kv_params=config.kv_params,
                    dtype=config.dtype,
                    devices=config.devices,
                    scale=config.attention_multiplier,
                    stacked_qkv=False,
                    has_bias=False,
                ),
                mlp=MLP(
                    dtype=config.dtype,
                    quantization_encoding=None,
                    hidden_dim=config.hidden_size,
                    feed_forward_length=config.feed_forward_length,
                    devices=config.devices,
                ),
                attention_norm=RMSNorm(
                    config.hidden_size,
                    config.dtype,
                    config.rms_norm_eps,
                    multiply_before_cast=False,
                ),
                mlp_norm=RMSNorm(
                    config.hidden_size,
                    config.dtype,
                    config.rms_norm_eps,
                    multiply_before_cast=False,
                ),
            )
            for i in range(config.num_hidden_layers)
        ]

        # Create Embedding and output layers.
        embedding_output_dtype = config.dtype
        embedding_layer = Embedding(
            config.vocab_size,
            config.hidden_size,
            embedding_output_dtype,
            config.devices[0],
        )
        output = Linear(
            config.hidden_size,
            config.vocab_size,
            embedding_output_dtype,
            config.devices[0],
        )
        kv_collection_cls: (
            type[FetchContinuousBatchingKVCacheCollection]
            | type[FetchPagedKVCacheCollection]
        )
        if config.kv_params.cache_strategy == KVCacheStrategy.CONTINUOUS:
            kv_collection_cls = FetchContinuousBatchingKVCacheCollection
        elif config.kv_params.cache_strategy == KVCacheStrategy.PAGED:
            kv_collection_cls = FetchPagedKVCacheCollection
        else:
            raise ValueError(
                "Unsupported caching strategy "
                + str(config.kv_params.cache_strategy)
            )

        super().__init__(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            layers=layers,
            norm=RMSNorm(
                config.hidden_size,
                config.dtype,
                config.rms_norm_eps,
                multiply_before_cast=False,
            ),
            output=output,
            embedding=embedding_layer,
            kv_params=config.kv_params,
            kv_collection_constructor=kv_collection_cls(config.kv_params),
            rope=rope,
            return_logits=config.return_logits,
        )
