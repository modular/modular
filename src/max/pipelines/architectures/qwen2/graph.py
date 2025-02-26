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
"""Build a Qwen2 model via Graph API from GGUF weights."""

import math
from typing import List, Union, cast

from max.dtype import DType
from max.graph import (
    DeviceRef,
    Graph,
    TensorValue,
    TensorValueLike,
    Weight,
    ops,
)
from max.graph.weights import Weights
from max.pipelines import PipelineConfig, RopeType, WeightsFormat
from max.pipelines.kv_cache import (
    FetchContinuousBatchingKVCacheCollection,
    FetchPagedKVCacheCollection,
    KVCacheParams,
    KVCacheStrategy,
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


def distribute_value(
    v: TensorValue, devices: List[DeviceRef]
) -> List[TensorValue]:
    return [v.to(device) for device in devices]


def shard_col_value(
    x: TensorValueLike, devices: List[DeviceRef]
) -> List[TensorValue]:
    n_devices = len(devices)
    v = TensorValue(x)
    col_size = v.shape[1] // n_devices
    return [
        v[:, i * col_size : (i + 1) * col_size].to(device)
        for i, device in enumerate(devices)
    ]


def shard_row_value(
    x: TensorValueLike, devices: List[DeviceRef]
) -> List[TensorValue]:
    n_devices = len(devices)
    v = TensorValue(x)
    row_size = v.shape[0] // n_devices
    return [
        v[i * row_size : (i + 1) * row_size, :].to(device)
        for i, device in enumerate(devices)
    ]


def feed_forward(
    dtype: DType,
    hidden_dim: int,
    feed_forward_length: int,
    weights: Weights,
) -> MLP:
    return MLP(
        linear(
            dtype,
            feed_forward_length,
            hidden_dim,
            weights.mlp.gate_proj,
        ),
        linear(
            dtype,
            hidden_dim,
            feed_forward_length,
            weights.mlp.down_proj,
        ),
        linear(
            dtype,
            feed_forward_length,
            hidden_dim,
            weights.mlp.up_proj,
        ),
    )


def linear(
    dtype: DType,
    in_features: int,
    out_features: int,
    weights: Weights,
) -> Linear:
    return Linear(
        weights.weight.allocate(
            dtype,
            [in_features, out_features],
        )
    )


def rms_norm(dims: int, eps: float, weights: Weights) -> RMSNorm:
    return RMSNorm(weights.weight.allocate(DType.float32, [dims]), eps)


def embedding(
    pipeline_config: PipelineConfig,
    vocab_size: int,
    hidden_dim: int,
    weights: Weights,
) -> Embedding:
    return Embedding(
        weights.weight.allocate(
            pipeline_config.dtype,
            [vocab_size, hidden_dim],
        )
    )


def attention(
    kv_params: KVCacheParams,
    pipeline_config: PipelineConfig,
    rope: OptimizedRotaryEmbedding,
    weights: Weights,
    layer_idx: TensorValue,
) -> AttentionWithRope:
    kv_weight_dim = (
        pipeline_config.huggingface_config.hidden_size
        // pipeline_config.huggingface_config.num_attention_heads
    ) * pipeline_config.huggingface_config.num_key_value_heads

    wq = weights.self_attn.q_proj.weight.allocate(
        pipeline_config.dtype,
        [
            pipeline_config.huggingface_config.hidden_size,
            pipeline_config.huggingface_config.hidden_size,
        ],
    )
    wk = weights.self_attn.k_proj.weight.allocate(
        pipeline_config.dtype,
        [kv_weight_dim, pipeline_config.huggingface_config.hidden_size],
    )
    wv = weights.self_attn.v_proj.weight.allocate(
        pipeline_config.dtype,
        [kv_weight_dim, pipeline_config.huggingface_config.hidden_size],
    )

    wqkv = ops.concat((wq, wk, wv))

    bias_q = weights.self_attn.q_proj.bias.allocate(
        pipeline_config.dtype,
        [pipeline_config.huggingface_config.hidden_size],
    )

    bias_k = weights.self_attn.k_proj.bias.allocate(
        pipeline_config.dtype,
        [kv_weight_dim],
    )

    bias_v = weights.self_attn.v_proj.bias.allocate(
        pipeline_config.dtype,
        [kv_weight_dim],
    )

    bias_qkv = ops.concat((bias_q, bias_k, bias_v))

    return AttentionWithRope(
        n_heads=pipeline_config.huggingface_config.num_attention_heads,
        kv_params=kv_params,
        wqkv=wqkv,
        wo=linear(
            pipeline_config.dtype,
            pipeline_config.huggingface_config.hidden_size,
            pipeline_config.huggingface_config.hidden_size,
            weights.self_attn.o_proj,
        ),
        rope=rope,
        layer_idx=layer_idx,
        bias=bias_qkv,
        scale=math.sqrt(1.0 / kv_params.head_dim),
    )


def transformer(
    graph: Graph,
    pipeline_config: PipelineConfig,
    weights: Weights,
    kv_params: KVCacheParams,
) -> Transformer:
    with graph:
        if weights.rope_freqs.weight.exists():
            rope_scaling = weights.rope_freqs.weight.raw_tensor()
        else:
            rope_scaling = None

        interleaved_rope_weights = (
            pipeline_config.weights_format == WeightsFormat.gguf
            and pipeline_config.rope_type == RopeType.normal
        )
        rope = OptimizedRotaryEmbedding(
            dim=pipeline_config.huggingface_config.hidden_size,
            n_heads=pipeline_config.huggingface_config.num_attention_heads,
            theta=pipeline_config.huggingface_config.rope_theta,
            max_seq_len=pipeline_config.huggingface_config.max_position_embeddings,
            rope_scaling=rope_scaling,
            interleaved=interleaved_rope_weights,
        )

        rms_norm_eps = pipeline_config.huggingface_config.rms_norm_eps
        layers = [
            TransformerBlock(
                attention=attention(
                    kv_params,
                    pipeline_config,
                    rope,
                    weights.model.layers[i],
                    layer_idx=ops.constant(i, DType.uint32),
                ),
                mlp=feed_forward(
                    pipeline_config.dtype,
                    pipeline_config.huggingface_config.hidden_size,
                    pipeline_config.huggingface_config.intermediate_size,
                    weights.model.layers[i],
                ),
                attention_norm=rms_norm(
                    pipeline_config.huggingface_config.hidden_size,
                    rms_norm_eps,
                    weights.model.layers[i].input_layernorm,
                ),
                mlp_norm=rms_norm(
                    pipeline_config.huggingface_config.hidden_size,
                    rms_norm_eps,
                    weights.model.layers[i].post_attention_layernorm,
                ),
            )
            for i in range(pipeline_config.huggingface_config.num_hidden_layers)
        ]

        embedding_layer = embedding(
            pipeline_config,
            pipeline_config.huggingface_config.vocab_size,
            pipeline_config.huggingface_config.hidden_size,
            weights.model.embed_tokens,
        )

        # Some model variants lack dedicated weights for a final linear
        # layer, and share the embedding layer.
        if weights.lm_head.weight.exists():
            output = Linear.create(
                pipeline_config.dtype,
                pipeline_config.graph_quantization_encoding,
                pipeline_config.huggingface_config.vocab_size,
                pipeline_config.huggingface_config.hidden_size,
                weights.lm_head,
            )
        else:
            output = Linear.create(
                pipeline_config.dtype,
                pipeline_config.graph_quantization_encoding,
                pipeline_config.huggingface_config.vocab_size,
                pipeline_config.huggingface_config.hidden_size,
                cast(Weight, embedding_layer.weights),
            )

        kv_collection: Union[
            FetchContinuousBatchingKVCacheCollection,
            FetchPagedKVCacheCollection,
        ]
        if kv_params.cache_strategy == KVCacheStrategy.CONTINUOUS:
            kv_collection = FetchContinuousBatchingKVCacheCollection(kv_params)
        elif kv_params.cache_strategy == KVCacheStrategy.PAGED:
            kv_collection = FetchPagedKVCacheCollection(kv_params)
        else:
            raise ValueError(
                "Unsupported caching strategy " + str(kv_params.cache_strategy)
            )

        return Transformer(
            dim=pipeline_config.huggingface_config.hidden_size,
            n_heads=pipeline_config.huggingface_config.num_attention_heads,
            layers=layers,
            norm=rms_norm(
                pipeline_config.huggingface_config.hidden_size,
                rms_norm_eps,
                weights.model.norm,
            ),
            output=output,
            embedding=embedding_layer,
            kv_params=kv_params,
            kv_collection_constructor=kv_collection,
            all_logits=pipeline_config.enable_echo,
        )
