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
"""Build a Mistral model via Graph API from Safetensor weights."""

import math
from typing import Union

from max.dtype import DType
from max.graph import DeviceRef, Graph, ops
from max.graph.weights import Weights
from max.nn import (
    MLPV1,
    AttentionWithRopeV1,
    EmbeddingV1,
    LinearV1,
    OptimizedRotaryEmbedding,
    RMSNormV1,
    TransformerBlock,
)
from max.nn.kv_cache import (
    FetchContinuousBatchingKVCacheCollection,
    FetchPagedKVCacheCollection,
    KVCacheParams,
    KVCacheStrategy,
)
from max.pipelines.lib import PipelineConfig
from transformers import AutoConfig

from ..llava.llava_decoder import Transformer


def feed_forward(
    dtype: DType,
    hidden_dim: int,
    feed_forward_length: int,
    weights: Weights,
):
    return MLPV1(
        linear(dtype, feed_forward_length, hidden_dim, weights.mlp.gate_proj),
        linear(dtype, hidden_dim, feed_forward_length, weights.mlp.down_proj),
        linear(dtype, feed_forward_length, hidden_dim, weights.mlp.up_proj),
    )


def linear(
    dtype: DType,
    in_features: int,
    out_features: int,
    weights: Weights,
) -> LinearV1:
    return LinearV1(
        weights.weight.allocate(
            dtype, [in_features, out_features], device=DeviceRef.GPU()
        )
    )


def rms_norm(dims: int, eps: float, weights: Weights) -> RMSNormV1:
    return RMSNormV1(
        weights.weight.allocate(DType.bfloat16, [dims], device=DeviceRef.GPU()),
        eps,
    )


def embedding(
    params: PipelineConfig,
    vocab_size: int,
    hidden_dim: int,
    weights: Weights,
    dtype: DType,
):
    return EmbeddingV1(
        weights.weight.allocate(
            dtype, [vocab_size, hidden_dim], device=DeviceRef.GPU()
        ),
        device=DeviceRef.GPU(),
    )


def _attention_opaque(
    kv_params: KVCacheParams,
    params: PipelineConfig,
    rope: OptimizedRotaryEmbedding,
    weights: Weights,
    huggingface_config: AutoConfig,
    dtype: DType,
):
    kv_weight_dim = (
        huggingface_config.text_config.head_dim
        * huggingface_config.text_config.num_key_value_heads
    )

    wq = weights.self_attn.q_proj.weight.allocate(
        dtype,
        [
            huggingface_config.text_config.num_attention_heads
            * huggingface_config.text_config.head_dim,
            huggingface_config.text_config.hidden_size,
        ],
        device=DeviceRef.GPU(),
    )
    wk = weights.self_attn.k_proj.weight.allocate(
        dtype,
        [kv_weight_dim, huggingface_config.text_config.hidden_size],
        device=DeviceRef.GPU(),
    )
    wv = weights.self_attn.v_proj.weight.allocate(
        dtype,
        [kv_weight_dim, huggingface_config.text_config.hidden_size],
        device=DeviceRef.GPU(),
    )
    wqkv = ops.concat((wq, wk, wv))

    return AttentionWithRopeV1(
        n_heads=huggingface_config.text_config.num_attention_heads,
        kv_params=kv_params,
        wqkv=wqkv,
        wo=linear(
            dtype,
            huggingface_config.text_config.hidden_size,
            huggingface_config.text_config.num_attention_heads
            * huggingface_config.text_config.head_dim,
            weights.self_attn.o_proj,
        ),
        rope=rope,
        scale=math.sqrt(1 / kv_params.head_dim),
    )


def _transformer(
    graph: Graph,
    params: PipelineConfig,
    weights: Weights,
    max_seq_len: int,
    kv_params: KVCacheParams,
    huggingface_config: AutoConfig,
    dtype: DType,
):
    with graph:
        rope = OptimizedRotaryEmbedding(
            dim=huggingface_config.text_config.num_attention_heads
            * huggingface_config.text_config.head_dim,
            n_heads=huggingface_config.text_config.num_attention_heads,
            head_dim=huggingface_config.text_config.head_dim,
            theta=huggingface_config.text_config.rope_theta,
            max_seq_len=max_seq_len,
            interleaved=False,
            device=DeviceRef.GPU(),
        )

        layers = [
            TransformerBlock(
                attention=_attention_opaque(
                    kv_params,
                    params,
                    rope,
                    weights.language_model.model.layers[i],
                    huggingface_config=huggingface_config,
                    dtype=dtype,
                ),
                mlp=feed_forward(
                    dtype,
                    huggingface_config.text_config.hidden_size,
                    huggingface_config.text_config.intermediate_size,
                    weights.language_model.model.layers[i],
                ),
                attention_norm=rms_norm(
                    huggingface_config.text_config.hidden_size,
                    huggingface_config.text_config.rms_norm_eps,
                    weights.language_model.model.layers[i].input_layernorm,
                ),
                mlp_norm=rms_norm(
                    huggingface_config.text_config.hidden_size,
                    huggingface_config.text_config.rms_norm_eps,
                    weights.language_model.model.layers[
                        i
                    ].post_attention_layernorm,
                ),
            )
            for i in range(huggingface_config.text_config.num_hidden_layers)
        ]

        embedding_layer = embedding(
            params,
            huggingface_config.text_config.vocab_size,
            huggingface_config.text_config.hidden_size,
            weights.language_model.model.embed_tokens,
            dtype=dtype,
        )

        output = linear(
            dtype,
            huggingface_config.text_config.vocab_size,
            huggingface_config.text_config.hidden_size,
            weights.language_model.lm_head,
        )

        kv_collection_cls: Union[
            type[FetchContinuousBatchingKVCacheCollection],
            type[FetchPagedKVCacheCollection],
        ]
        if kv_params.cache_strategy == KVCacheStrategy.CONTINUOUS:
            kv_collection_cls = FetchContinuousBatchingKVCacheCollection
        elif kv_params.cache_strategy == KVCacheStrategy.PAGED:
            kv_collection_cls = FetchPagedKVCacheCollection
        else:
            raise ValueError(
                f"Unsupported caching strategy {kv_params.cache_strategy}"
            )

        return Transformer(
            dim=huggingface_config.text_config.hidden_size,
            n_heads=huggingface_config.text_config.num_attention_heads,
            layers=layers,
            norm=rms_norm(
                huggingface_config.text_config.hidden_size,
                huggingface_config.text_config.rms_norm_eps,
                weights.language_model.model.norm,
            ),
            output=output,
            embedding=embedding_layer,
            kv_params=kv_params,
            kv_collection_constructor=kv_collection_cls(kv_params),
        )
