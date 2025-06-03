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
"""Build a Mistral model that runs on multiple devices."""

from __future__ import annotations

import functools
import logging

from max.nn import (
    ColumnParallelLinear,
    DistributedAttentionWithRope,
    DistributedMLP,
    DistributedRMSNorm,
    DistributedTransformer,
    DistributedTransformerBlock,
    OptimizedRotaryEmbedding,
    VocabParallelEmbedding,
)
from max.nn.kv_cache import (
    FetchPagedKVCacheCollection,
    KVCacheStrategy,
)

logger = logging.getLogger("max.pipelines")

from .model_config import MistralConfig


class DistributedMistral(DistributedTransformer):
    """The Mistral text transformer model."""

    def __init__(self, config: MistralConfig):
        assert len(config.devices) > 1

        rope = OptimizedRotaryEmbedding(
            dim=config.num_attention_heads * config.head_dim,
            n_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            interleaved=False,
            device=config.devices[0],
        )

        distributed_norm = functools.partial(
            DistributedRMSNorm,
            dim=config.hidden_size,
            dtype=config.dtype,
            eps=config.rms_norm_eps,
            devices=config.devices,
        )

        layers = [
            DistributedTransformerBlock(
                devices=config.devices,
                attention=DistributedAttentionWithRope(
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
                    clip_qkv=False,
                ),
                mlp=DistributedMLP(
                    dtype=config.dtype,
                    quantization_encoding=None,
                    hidden_dim=config.hidden_size,
                    feed_forward_length=config.feed_forward_length,
                    devices=config.devices,
                ),
                attention_norm=distributed_norm(),
                mlp_norm=distributed_norm(),
            )
            for i in range(config.num_hidden_layers)
        ]

        # Create Embedding and output layers.
        embedding_layer = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.dtype,
            config.devices,
            quantization_encoding=None,
        )

        output = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            config.dtype,
            devices=config.devices,
            quantization_encoding=None,
        )

        if config.kv_params.cache_strategy != KVCacheStrategy.PAGED:
            raise ValueError(
                "Unsupported caching strategy "
                + str(config.kv_params.cache_strategy)
            )

        super().__init__(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            layers=layers,
            norm=distributed_norm(),
            output=output,
            embedding=embedding_layer,
            kv_params=config.kv_params,
            kv_collection_constructor=FetchPagedKVCacheCollection(
                config.kv_params, num_layers=config.num_hidden_layers
            ),
            devices=config.devices,
            rope=rope,
            return_logits=config.return_logits,
        )
