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
"""Build a Llama3 model that uses continuous or paged kv-caching"""

from __future__ import annotations

import functools
from typing import Callable

from max.dtype import DType
from max.graph.quantization import QuantizationEncoding
from max.pipelines.kv_cache import (
    FetchContinuousBatchingKVCacheCollection,
    FetchPagedKVCacheCollection,
    KVCacheStrategy,
)
from max.pipelines.nn import (
    MLPV2,
    AttentionWithRopeV2,
    EmbeddingV2,
    GPTQAttentionWithRope,
    GPTQLinearV2,
    LayerV2,
    LinearV2,
    Llama3RotaryEmbedding,
    RMSNormV2,
    Transformer,
    TransformerBlock,
)

from .model_config import Llama3Config
from .naive_llama3 import ConstantLayerNorm, StackedMLP


class Llama3(Transformer):
    def __init__(self, config: Llama3Config):
        assert len(config.devices) == 1
        rope = Llama3RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            interleaved=config.interleaved_rope_weights,
            scaling_params=config.rope_scaling_params,
        )

        # Select norm layer class.
        create_norm: Callable[..., LayerV2]
        if config.norm_method == "rms_norm":
            if config.rms_norm_eps is None:
                raise ValueError(
                    "rms_norm_eps cannot be None for model that uses RMSNorm."
                )
            create_norm = functools.partial(
                RMSNormV2, config.hidden_size, config.rms_norm_eps
            )
        else:
            create_norm = functools.partial(
                ConstantLayerNorm, config.hidden_size
            )

        # Select linear layer class.
        linear_cls: Callable[..., LinearV2]
        if config.quantization_config:
            linear_cls = functools.partial(
                GPTQLinearV2, quantization_config=config.quantization_config
            )
        else:
            linear_cls = LinearV2
        mlp_cls = StackedMLP if config.stacked_mlp else MLPV2
        attention_cls: Callable[..., AttentionWithRopeV2]
        if config.quantization_config:
            attention_cls = functools.partial(
                GPTQAttentionWithRope,
                quantization_config=config.quantization_config,
                scale=config.attention_multiplier,
            )
        else:
            attention_cls = functools.partial(
                AttentionWithRopeV2,
                stacked_qkv=config.stacked_qkv,
                scale=config.attention_multiplier,
                clip_qkv=config.clip_qkv,
            )

        layers = [
            TransformerBlock(
                attention=attention_cls(
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    hidden_size=config.hidden_size,
                    kv_params=config.kv_params,
                    layer_idx=i,
                    dtype=config.dtype,
                    rope=rope,
                    linear_cls=linear_cls,
                    devices=config.devices,
                ),
                mlp=mlp_cls(
                    config.dtype,
                    config.quantization_encoding,
                    config.hidden_size,
                    config.intermediate_size,
                    linear_cls,
                    devices=config.devices,
                ),
                attention_norm=create_norm(),
                mlp_norm=create_norm(),
                residual_multiplier=config.residual_multiplier,
            )
            for i in range(config.num_hidden_layers)
        ]

        # Create Embedding and output layers.
        embedding_output_dtype = config.dtype
        embedding_output_quantization = config.quantization_encoding
        if config.quantization_encoding == QuantizationEncoding.GPTQ:
            embedding_output_dtype = DType.bfloat16
            embedding_output_quantization = None
        embedding_layer = EmbeddingV2(
            config.vocab_size,
            config.hidden_size,
            embedding_output_dtype,
            config.devices[0],
            quantization_encoding=embedding_output_quantization,
        )
        output = LinearV2(
            config.hidden_size,
            config.vocab_size,
            embedding_output_dtype,
            config.devices[0],
            quantization_encoding=embedding_output_quantization,
        )

        if config.tie_word_embeddings:
            output.set_shared_weight("weight", embedding_layer.weight)

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
            norm=create_norm(),
            output=output,
            embedding=embedding_layer,
            kv_params=config.kv_params,
            kv_collection_constructor=kv_collection_cls(config.kv_params),
            all_logits=config.all_logits,
            embedding_multiplier=config.embedding_multiplier,
            logits_postprocessor=config.logits_postprocessor,
        )
