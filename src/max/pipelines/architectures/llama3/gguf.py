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
"""Build a Llama3 model via Graph API from GGUF weights."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
from max.dtype import DType
from max.graph import (
    DeviceRef,
    Graph,
    TensorValue,
    TensorValueLike,
    ops,
)
from max.graph.quantization import QuantizationEncoding
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
    DistributedAttentionWithRope,
    DistributedMLP,
    DistributedRMSNorm,
    DistributedTransformer,
    DistributedTransformerBlock,
    Embedding,
    Linear,
    LinearV2,
    OptimizedRotaryEmbedding,
    RMSNorm,
    VocabParallelEmbedding,
)
from max.pipelines.nn.layer import Layer


def distribute_value(
    v: TensorValue, devices: List[DeviceRef]
) -> List[TensorValue]:
    return [v.to(device) for device in devices]


def shard_col_value(
    x: TensorValueLike, devices: List[DeviceRef]
) -> List[TensorValue]:
    n_devices = len(devices)
    v = TensorValue(x)
    col_size = int(v.shape[1]) // n_devices
    return [
        v[:, i * col_size : (i + 1) * col_size].to(device)
        for i, device in enumerate(devices)
    ]


def shard_row_value(
    x: TensorValueLike, devices: List[DeviceRef]
) -> List[TensorValue]:
    n_devices = len(devices)
    v = TensorValue(x)
    row_size = int(v.shape[0]) // n_devices
    return [
        v[i * row_size : (i + 1) * row_size, :].to(device)
        for i, device in enumerate(devices)
    ]


@dataclass
class Phi3MLP(Layer):
    """
    A multi-layer perceptron composed of two linear layers. Where the
    gate_up_proj is a stacked linear layer which contains the up_proj and
    gate_proj. This is used by the Phi3 models.
    """

    gate_up_proj: Linear
    down_proj: Linear

    def __call__(self, x: TensorValue) -> TensorValue:
        up_states = self.gate_up_proj(x)

        gate = up_states[:, : up_states.shape.static_dims[0] // 2]
        up_states = up_states[:, up_states.shape.static_dims[0] // 2 :]

        return self.down_proj(ops.silu(gate) * up_states)


def rms_norm(dims: int, eps: float, weights: Weights) -> RMSNorm:
    return RMSNorm(weights.weight.allocate(DType.float32, [dims]), eps)


@dataclass
class LayerNorm(Layer):
    """Layer normalization block."""

    gamma: TensorValue
    beta: TensorValue
    eps: float = 1e-5

    def __call__(self, input: TensorValue):
        return ops.cast(
            ops.layer_norm(
                ops.cast(input, DType.float32),
                gamma=ops.cast(self.gamma, DType.float32),
                beta=ops.cast(self.beta, DType.float32),
                epsilon=self.eps,
            ),
            input.dtype,
        )


def layer_norm(dims: int, eps: float = 1e-5) -> LayerNorm:
    gamma = ops.constant(np.ones(dims), DType.float32)
    beta = ops.constant(np.zeros(dims), DType.float32)
    return LayerNorm(gamma, beta, eps=eps)


def norm(
    norm_method: Literal["rms_norm"] | Literal["layer_norm"],
    pipeline_config: PipelineConfig,
    weights: Weights,
    weight_name: str,
) -> RMSNorm | LayerNorm:
    if norm_method == "rms_norm":
        if pipeline_config.huggingface_config.model_type == "exaone":
            rms_norm_eps = pipeline_config.huggingface_config.layer_norm_epsilon
        else:
            rms_norm_eps = pipeline_config.huggingface_config.rms_norm_eps

        return rms_norm(
            pipeline_config.huggingface_config.hidden_size,
            rms_norm_eps,
            weights[weight_name],
        )
    else:
        return layer_norm(pipeline_config.huggingface_config.hidden_size)


def distributed_norm(
    norm_method: Literal["rms_norm"] | Literal["layer_norm"],
    pipeline_config: PipelineConfig,
    weights: Weights,
    weight_name: str,
    devices: List[DeviceRef],
) -> DistributedRMSNorm:
    assert norm_method == "rms_norm"

    weights_ = TensorValue(
        weights[weight_name].weight.allocate(
            DType.float32,
            [pipeline_config.huggingface_config.hidden_size],
        )
    )
    weights_devs = distribute_value(weights_, devices)

    rms_norms = [
        RMSNorm(weights_dev, pipeline_config.huggingface_config.rms_norm_eps)
        for weights_dev in weights_devs
    ]

    return DistributedRMSNorm(rms_norms, devices)


def embedding(
    pipeline_config: PipelineConfig,
    vocab_size: int,
    hidden_dim: int,
    weights: Weights,
) -> Embedding:
    if pipeline_config.quantization_encoding == "gptq":
        return Embedding(
            weights.weight.allocate(
                DType.bfloat16,
                [vocab_size, hidden_dim],
                None,
            )
        )

    else:
        return Embedding(
            weights.weight.allocate(
                pipeline_config.dtype,
                [vocab_size, hidden_dim],
                pipeline_config.graph_quantization_encoding,
            )
        )


def distributed_feed_forward(
    dtype: DType,
    quantization_encoding: Optional[QuantizationEncoding],
    hidden_dim: int,
    feed_forward_length: int,
    weights: Weights,
    devices: List[DeviceRef],
) -> DistributedMLP:
    w_ffn_down_full = weights.ffn_down.weight.allocate(
        dtype, [hidden_dim, feed_forward_length], quantization_encoding
    )
    ffn_down_sharded = shard_col_value(w_ffn_down_full, devices)
    w_ffn_gate_full = weights.ffn_gate.weight.allocate(
        dtype, [feed_forward_length, hidden_dim], quantization_encoding
    )
    ffn_gate_sharded = shard_row_value(w_ffn_gate_full, devices)
    w_ffn_up_full = weights.ffn_up.weight.allocate(
        dtype, [feed_forward_length, hidden_dim], quantization_encoding
    )
    ffn_up_sharded = shard_row_value(w_ffn_up_full, devices)

    mlps = [
        MLP(
            Linear(ffn_gate_sharded[rank]),
            Linear(ffn_down_sharded[rank]),
            Linear(ffn_up_sharded[rank]),
        )
        for rank in range(len(devices))
    ]

    return DistributedMLP(mlps, len(devices))


def distributed_attention_opaque(
    kv_params: KVCacheParams,
    pipeline_config: PipelineConfig,
    rope: OptimizedRotaryEmbedding,
    weights: Weights,
    layer_idx: TensorValue,
    devices: List[DeviceRef],
) -> DistributedAttentionWithRope:
    kv_weight_dim = (
        pipeline_config.huggingface_config.hidden_size
        // pipeline_config.huggingface_config.num_attention_heads
    ) * pipeline_config.huggingface_config.num_key_value_heads
    wq_full = weights.attn_q.weight.allocate(
        pipeline_config.dtype,
        [
            pipeline_config.huggingface_config.hidden_size,
            pipeline_config.huggingface_config.hidden_size,
        ],
        pipeline_config.graph_quantization_encoding,
    )
    wk_full = weights.attn_k.weight.allocate(
        pipeline_config.dtype,
        [kv_weight_dim, pipeline_config.huggingface_config.hidden_size],
        pipeline_config.graph_quantization_encoding,
    )
    wv_full = weights.attn_v.weight.allocate(
        pipeline_config.dtype,
        [kv_weight_dim, pipeline_config.huggingface_config.hidden_size],
        pipeline_config.graph_quantization_encoding,
    )

    wo_full = weights.attn_output.weight.allocate(
        pipeline_config.dtype,
        [
            pipeline_config.huggingface_config.hidden_size,
            pipeline_config.huggingface_config.hidden_size,
        ],
        pipeline_config.graph_quantization_encoding,
    )
    wq_shards = shard_row_value(wq_full, devices)
    wk_shards = shard_row_value(wk_full, devices)
    wv_shards = shard_row_value(wv_full, devices)

    # Didn't transpose here since linear will transpose so shard on col instead
    # of row
    wo_shards = shard_col_value(wo_full, devices)
    attns = [
        AttentionWithRope(
            n_heads=pipeline_config.huggingface_config.num_attention_heads
            // len(devices),
            kv_params=kv_params,
            wqkv=ops.concat(
                (wq_shards[rank], wk_shards[rank], wv_shards[rank])
            ),
            wo=Linear(wo_shards[rank]),
            rope=rope,
            layer_idx=layer_idx,
            scale=math.sqrt(1.0 / kv_params.head_dim),
        )
        for rank in range(len(devices))
    ]

    return DistributedAttentionWithRope(attns, devices)


def _kv_collection_constructor(
    kv_params: KVCacheParams,
) -> FetchContinuousBatchingKVCacheCollection | FetchPagedKVCacheCollection:
    """Gets the fetch KV collection based on the KV cache strategy.

    Returns:
        Callable that stages an op to fetch a KV cache collection.

    Raises:
        ValueError: If the cache strategy is unsupported.
    """
    if kv_params.cache_strategy == KVCacheStrategy.CONTINUOUS:
        return FetchContinuousBatchingKVCacheCollection(kv_params)
    elif kv_params.cache_strategy == KVCacheStrategy.PAGED:
        return FetchPagedKVCacheCollection(kv_params)

    msg = f"Unsupported caching strategy {kv_params.cache_strategy}"
    raise ValueError(msg)


def distributed_transformer_opaque(
    graph: Graph,
    pipeline_config: PipelineConfig,
    weights: Weights,
    max_seq_len: int,
    kv_params: KVCacheParams,
    norm_method: Literal["rms_norm"] | Literal["layer_norm"],
) -> DistributedTransformer:
    devices = [
        DeviceRef(spec.device_type, spec.id)
        for spec in pipeline_config.device_specs
    ]
    with graph:
        if weights.rope_freqs.weight.exists():
            rope_scaling = weights.rope_freqs.weight.raw_tensor()
        else:
            rope_scaling = None

        interleaved_rope_weights = (
            pipeline_config.weights_format == WeightsFormat.gguf
            and pipeline_config.rope_type == RopeType.normal
        )
        partial_rotary_factor = getattr(
            pipeline_config.huggingface_config, "partial_rotary_factor", 1
        )
        rope = OptimizedRotaryEmbedding(
            dim=pipeline_config.huggingface_config.hidden_size,
            n_heads=partial_rotary_factor
            * pipeline_config.huggingface_config.num_attention_heads,
            theta=pipeline_config.huggingface_config.rope_theta,
            max_seq_len=max_seq_len,
            rope_scaling=rope_scaling,
            interleaved=interleaved_rope_weights,
        )

        layers = [
            DistributedTransformerBlock(
                attention=distributed_attention_opaque(
                    kv_params,
                    pipeline_config,
                    rope,
                    weights.blk[i],
                    layer_idx=ops.constant(i, DType.uint32),
                    devices=devices,
                ),
                mlp=distributed_feed_forward(
                    pipeline_config.dtype,
                    pipeline_config.graph_quantization_encoding,
                    pipeline_config.huggingface_config.hidden_size,
                    pipeline_config.huggingface_config.intermediate_size,
                    weights.blk[i],
                    devices=devices,
                ),
                attention_norm=distributed_norm(
                    norm_method,
                    pipeline_config,
                    weights.blk[i],
                    "attn_norm",
                    devices=devices,
                ),
                mlp_norm=distributed_norm(
                    norm_method,
                    pipeline_config,
                    weights.blk[i],
                    "ffn_norm",
                    devices=devices,
                ),
                devices=devices,
            )
            for i in range(pipeline_config.huggingface_config.num_hidden_layers)
        ]

        # TODO(max.nn.Model): We still rely on the `Weights` mechanism for
        # constructing the Python-side weights registry.
        # So we have to "allocate" a spot in the weights registry for
        # output/embedding weights here.
        embedding_weight = weights.token_embd.weight.allocate(
            pipeline_config.dtype,
            [
                pipeline_config.huggingface_config.vocab_size,
                pipeline_config.huggingface_config.hidden_size,
            ],
        )
        weights.output.weight.allocate(
            pipeline_config.dtype,
            [
                pipeline_config.huggingface_config.vocab_size,
                pipeline_config.huggingface_config.hidden_size,
            ],
        )

        embedding_layer = VocabParallelEmbedding(
            vocab_size=pipeline_config.huggingface_config.vocab_size,
            hidden_dim=pipeline_config.huggingface_config.hidden_size,
            dtype=pipeline_config.dtype,
            devices=devices,
            # Use the embedding weight's name, which mismatches between
            # Safetensors and GGUF Llama 3.
            name=embedding_weight.name,
        )

        output = LinearV2(
            in_dim=pipeline_config.huggingface_config.hidden_size,
            out_dim=pipeline_config.huggingface_config.vocab_size,
            dtype=pipeline_config.dtype,
            # Only compute output embedding on device 0 for now.
            # TODO(MODELS-378): More optimal would be to:
            # - Shard embedding table across devices.
            # - Compute output on all devices for multistep.
            device=devices[0],
            # Smaller model variants lack dedicated weights for a final linear
            # layer, and share the embedding layer.
            name=(
                weights.output.name
                if weights.output.weight.exists()
                else embedding_layer.weight.name
            ),
        )

        return DistributedTransformer(
            dim=pipeline_config.huggingface_config.hidden_size,
            n_heads=pipeline_config.huggingface_config.num_attention_heads,
            layers=layers,
            norm=norm(norm_method, pipeline_config, weights, "output_norm"),  # type:ignore
            output=output,
            embedding=embedding_layer,
            kv_params=kv_params,
            kv_collection_constructor=_kv_collection_constructor(kv_params),
            devices=devices,
            all_logits=pipeline_config.enable_echo,
        )
