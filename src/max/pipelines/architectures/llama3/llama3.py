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
from typing import Callable, Literal, Optional

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, TensorValue
from max.graph.quantization import QuantizationConfig, QuantizationEncoding
from max.pipelines.kv_cache import (
    FetchContinuousBatchingKVCacheCollection,
    FetchPagedKVCacheCollection,
    KVCacheParams,
    KVCacheStrategy,
)
from max.pipelines.nn import (
    AttentionWithRopeV2,
    EmbeddingV2,
    GPTQAttentionWithRope,
    GPTQLinearV2,
    LayerV2,
    LinearV2,
    OptimizedRotaryEmbedding,
    RMSNormV2,
    Transformer,
    TransformerBlock,
)

from .naive_llama3 import ConstantLayerNorm, Llama3MLP, StackedMLP


class Llama3(Transformer):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        num_hidden_layers: int,
        rope_theta: float,
        max_seq_len: int,
        intermediate_size: int,
        interleaved_rope_weights: bool,
        rope_scaling: Optional[np.ndarray],
        vocab_size: int,
        dtype: DType,
        quantization_encoding: Optional[QuantizationEncoding],
        quantization_config: Optional[QuantizationConfig],
        kv_params: KVCacheParams,
        all_logits: bool,
        norm_method: Literal["rms_norm"] | Literal["layer_norm"],
        rms_norm_eps: Optional[float],
        tie_word_embeddings: bool,
        stacked_mlp: bool,
        stacked_qkv: bool,
        logits_postprocessor: Callable[[TensorValue], TensorValue] | None,
        attention_multiplier: float,
        embedding_multiplier: float,
        residual_multiplier: float,
        devices: list[DeviceRef],
        clip_qkv: Optional[float],
    ):
        rope = OptimizedRotaryEmbedding(
            dim=hidden_size,
            n_heads=num_attention_heads,
            theta=rope_theta,
            max_seq_len=max_seq_len,
            rope_scaling=rope_scaling,
            interleaved=interleaved_rope_weights,
        )
        create_norm: Callable[..., LayerV2]
        if norm_method == "rms_norm":
            if rms_norm_eps is None:
                raise ValueError(
                    "rms_norm_eps cannot be None for model that uses RMSNorm."
                )
            create_norm = functools.partial(
                RMSNormV2, hidden_size, rms_norm_eps
            )
        else:
            create_norm = functools.partial(ConstantLayerNorm, hidden_size)

        linear_cls: Callable[..., LinearV2]
        if quantization_config:
            linear_cls = functools.partial(
                GPTQLinearV2, quantization_config=quantization_config
            )
        else:
            linear_cls = LinearV2
        mlp_cls = StackedMLP if stacked_mlp else Llama3MLP
        attention_cls: Callable[..., AttentionWithRopeV2]
        if quantization_config:
            attention_cls = functools.partial(
                GPTQAttentionWithRope,
                quantization_config=quantization_config,
                scale=attention_multiplier,
            )
        else:
            attention_cls = functools.partial(
                AttentionWithRopeV2,
                stacked_qkv=stacked_qkv,
                scale=attention_multiplier,
                clip_qkv=clip_qkv,
            )

        layers = [
            TransformerBlock(
                attention=attention_cls(
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    hidden_size=hidden_size,
                    kv_params=kv_params,
                    layer_idx=i,
                    dtype=dtype,
                    rope=rope,
                    linear_cls=linear_cls,
                    device=devices[0],
                ),
                mlp=mlp_cls(
                    dtype,
                    quantization_encoding,
                    hidden_size,
                    intermediate_size,
                    linear_cls,
                    # devices=devices,  # TODO(kathywu): setting devices causes issues
                ),
                attention_norm=create_norm(),
                mlp_norm=create_norm(),
                residual_multiplier=residual_multiplier,
            )
            for i in range(num_hidden_layers)
        ]

        # Create Embedding and output layers.
        embedding_output_dtype = dtype
        embedding_output_quantization = quantization_encoding
        if quantization_encoding == QuantizationEncoding.GPTQ:
            embedding_output_dtype = DType.bfloat16
            embedding_output_quantization = None

        embedding_layer = EmbeddingV2(
            vocab_size,
            hidden_size,
            embedding_output_dtype,
            None,  # TODO(kathywu): setting devices causes issues
            quantization_encoding=embedding_output_quantization,
        )
        output = LinearV2(
            hidden_size,
            vocab_size,
            embedding_output_dtype,
            None,  # TODO(kathywu): setting devices causes issues
            quantization_encoding=embedding_output_quantization,
        )

        if tie_word_embeddings:
            output.set_shared_weight("weight", embedding_layer.weight)

        kv_collection_cls: (
            type[FetchContinuousBatchingKVCacheCollection]
            | type[FetchPagedKVCacheCollection]
        )
        if kv_params.cache_strategy == KVCacheStrategy.CONTINUOUS:
            kv_collection_cls = FetchContinuousBatchingKVCacheCollection
        elif kv_params.cache_strategy == KVCacheStrategy.PAGED:
            kv_collection_cls = FetchPagedKVCacheCollection
        else:
            raise ValueError(
                "Unsupported caching strategy " + str(kv_params.cache_strategy)
            )

        super().__init__(
            dim=hidden_size,
            n_heads=num_attention_heads,
            layers=layers,
            norm=create_norm(),
            output=output,
            embedding=embedding_layer,
            kv_params=kv_params,
            kv_collection_constructor=kv_collection_cls(kv_params),
            all_logits=all_logits,
            embedding_multiplier=embedding_multiplier,
            logits_postprocessor=logits_postprocessor,
        )
