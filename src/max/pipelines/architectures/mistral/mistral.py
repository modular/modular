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

from max.nn import (
    MLPV2,
    AttentionWithRopeV2,
    EmbeddingV2,
    LinearV2,
    OptimizedRotaryEmbedding,
    RMSNormV2,
    Transformer,
    TransformerBlock,
)
from max.nn.kv_cache import FetchContinuousBatchingKVCacheCollection

from .model_config import MistralConfig


class Mistral(Transformer):
    """Defines the Mistral transformer model."""

    def __init__(self, config: MistralConfig):
        assert len(config.devices) == 1

        rope = OptimizedRotaryEmbedding(
            dim=config.num_attention_heads * config.head_dim,
            n_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            interleaved=False,
        )

        layers = [
            TransformerBlock(
                attention=AttentionWithRopeV2(
                    rope=rope,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    hidden_size=config.hidden_size,
                    kv_params=config.kv_params,
                    layer_idx=i,
                    dtype=config.dtype,
                    devices=config.devices,
                    scale=config.attention_multiplier,
                    stacked_qkv=False,
                    has_bias=False,
                ),
                mlp=MLPV2(
                    dtype=config.dtype,
                    quantization_encoding=None,
                    hidden_dim=config.hidden_size,
                    feed_forward_length=config.feed_forward_length,
                    devices=config.devices,
                ),
                attention_norm=RMSNormV2(
                    config.hidden_size,
                    config.rms_norm_eps,
                ),
                mlp_norm=RMSNormV2(
                    config.hidden_size,
                    config.rms_norm_eps,
                ),
            )
            for i in range(config.num_hidden_layers)
        ]

        # Create Embedding and output layers.
        embedding_output_dtype = config.dtype
        embedding_layer = EmbeddingV2(
            config.vocab_size,
            config.hidden_size,
            embedding_output_dtype,
            config.devices[0],
        )
        output = LinearV2(
            config.hidden_size,
            config.vocab_size,
            embedding_output_dtype,
            config.devices[0],
        )

        super().__init__(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            layers=layers,
            norm=RMSNormV2(
                config.hidden_size,
                config.rms_norm_eps,
            ),
            output=output,
            embedding=embedding_layer,
            kv_params=config.kv_params,
            kv_collection_constructor=FetchContinuousBatchingKVCacheCollection(
                config.kv_params
            ),
            return_logits=config.return_logits,
        )
