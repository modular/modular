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

from max.dtype import DType
from max.graph import Graph, TensorType, ops
from max.graph.weights import SafetensorWeights
from max.pipelines import PipelineConfig
from max.pipelines.kv_cache import (
    FetchContinuousBatchingKVCacheCollection,
    KVCacheManager,
    KVCacheParams,
)
from max.pipelines.nn import (
    MLP,
    AttentionWithRope,
    Embedding,
    Linear,
    OptimizedRotaryEmbedding,
    RMSNorm,
    Transformer,
    TransformerBlock,
)


def feed_forward(
    dtype: DType,
    hidden_dim: int,
    feed_forward_length: int,
    weights: SafetensorWeights,
):
    return MLP(
        linear(
            dtype,
            feed_forward_length,
            hidden_dim,
            weights.gate_proj,
        ),
        linear(
            dtype,
            hidden_dim,
            feed_forward_length,
            weights.down_proj,
        ),
        linear(
            dtype,
            feed_forward_length,
            hidden_dim,
            weights.up_proj,
        ),
    )


def linear(
    dtype: DType,
    in_features: int,
    out_features: int,
    weights: SafetensorWeights,
) -> Linear:
    return Linear(
        weights.weight.allocate(dtype, [in_features, out_features], None)
    )


def rms_norm(dims: int, eps: float, weights: SafetensorWeights) -> RMSNorm:
    return RMSNorm(weights.weight.allocate(DType.bfloat16, [dims]), eps)


def embedding(
    params: PipelineConfig,
    vocab_size: int,
    hidden_dim: int,
    weights: SafetensorWeights,
):
    return Embedding(
        weights.weight.allocate(
            params.dtype,
            [vocab_size, hidden_dim],
        )
    )


def _attention_opaque(
    kv_params: KVCacheParams,
    params: PipelineConfig,
    rope: OptimizedRotaryEmbedding,
    weights: SafetensorWeights,
    layer_idx: int,
):
    kv_weight_dim = (
        params.huggingface_config.head_dim
        * params.huggingface_config.num_key_value_heads
    )

    wq = weights.self_attn.q_proj.weight.allocate(
        params.dtype,
        [
            params.huggingface_config.num_attention_heads
            * params.huggingface_config.head_dim,
            params.huggingface_config.hidden_size,
        ],
    )
    wk = weights.self_attn.k_proj.weight.allocate(
        params.dtype,
        [kv_weight_dim, params.huggingface_config.hidden_size],
    )
    wv = weights.self_attn.v_proj.weight.allocate(
        params.dtype,
        [kv_weight_dim, params.huggingface_config.hidden_size],
    )
    wqkv = ops.concat((wq, wk, wv))

    return AttentionWithRope(
        n_heads=params.huggingface_config.num_attention_heads,
        kv_params=kv_params,
        wqkv=wqkv,
        wo=linear(
            params.dtype,
            params.huggingface_config.hidden_size,
            params.huggingface_config.num_attention_heads
            * params.huggingface_config.head_dim,
            weights.self_attn.o_proj,
        ),
        rope=rope,
        layer_idx=layer_idx,  # type: ignore
        scale=math.sqrt(1 / kv_params.head_dim),
    )


def _transformer(
    graph: Graph,
    params: PipelineConfig,
    weights: SafetensorWeights,
    max_seq_len: int,
    kv_params: KVCacheParams,
):
    with graph:
        rope = OptimizedRotaryEmbedding(
            dim=params.huggingface_config.num_attention_heads
            * params.huggingface_config.head_dim,
            n_heads=params.huggingface_config.num_attention_heads,
            theta=params.huggingface_config.rope_theta,
            max_seq_len=max_seq_len,
            rope_scaling=None,
            interleaved=False,
        )

        layers = [
            TransformerBlock(
                attention=_attention_opaque(
                    kv_params,
                    params,
                    rope,
                    weights.model.layers[i],
                    layer_idx=ops.constant(i, DType.uint32),  # type: ignore
                ),
                mlp=feed_forward(
                    params.dtype,
                    params.huggingface_config.hidden_size,
                    params.huggingface_config.intermediate_size,
                    weights.model.layers[i].mlp,
                ),
                attention_norm=rms_norm(
                    params.huggingface_config.hidden_size,
                    params.huggingface_config.rms_norm_eps,
                    weights.model.layers[i].input_layernorm,
                ),
                mlp_norm=rms_norm(
                    params.huggingface_config.hidden_size,
                    params.huggingface_config.rms_norm_eps,
                    weights.model.layers[i].post_attention_layernorm,
                ),
            )
            for i in range(params.huggingface_config.num_hidden_layers)
        ]

        embedding_layer = embedding(
            params,
            params.huggingface_config.vocab_size,
            params.huggingface_config.hidden_size,
            weights.model.embed_tokens,
        )

        output = linear(
            params.dtype,
            params.huggingface_config.vocab_size,
            params.huggingface_config.hidden_size,
            weights.lm_head,
        )

        kv_collection_cls = FetchContinuousBatchingKVCacheCollection

        return Transformer(
            dim=params.huggingface_config.hidden_size,
            n_heads=params.huggingface_config.num_attention_heads,
            layers=layers,
            norm=rms_norm(
                params.huggingface_config.hidden_size,
                params.huggingface_config.rms_norm_eps,
                weights.model.norm,
            ),
            output=output,
            embedding=embedding_layer,
            kv_params=kv_params,
            kv_collection_constructor=kv_collection_cls(kv_params),
        )


def _build_graph(
    pipeline_config: PipelineConfig,
    weights: SafetensorWeights,
    max_seq_len: int,
    kv_params: KVCacheParams,
    kv_manager: KVCacheManager,
) -> Graph:
    tokens_type = TensorType(DType.int64, shape=["total_seq_len"])
    input_row_offsets_type = TensorType(
        DType.uint32, shape=["input_row_offsets_len"]
    )

    kv_cache_args = kv_manager.input_symbols()[0]

    with Graph(
        "mistral",
        input_types=[
            tokens_type,
            input_row_offsets_type,
            *kv_cache_args,
        ],
    ) as graph:
        model = _transformer(
            graph=graph,
            params=pipeline_config,
            weights=weights,
            max_seq_len=max_seq_len,
            kv_params=kv_params,
        )
        tokens, input_row_offsets, *kv_cache = graph.inputs
        outputs = model(tokens, kv_cache, input_row_offsets=input_row_offsets)
        graph.output(*outputs)
        return graph
