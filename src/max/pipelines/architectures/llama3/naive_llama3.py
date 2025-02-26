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
"""Builds a Llama3 model that uses naive KV-caching."""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Callable, Literal, Optional, Union

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.graph.quantization import QuantizationConfig, QuantizationEncoding
from max.pipelines.kv_cache import KVCacheParams
from max.pipelines.nn import (
    MLPV2,
    EmbeddingV2,
    GPTQLinearV2,
    LayerV2,
    LinearV2,
    NaiveAttentionWithRope,
    NaiveTransformer,
    NaiveTransformerBlock,
    OptimizedRotaryEmbedding,
    RMSNormV2,
    RotaryEmbedding,
)


class ConstantLayerNorm(LayerV2):
    """Layer normalization block with constant gamma and beta values."""

    gamma: np.ndarray
    beta: np.ndarray
    eps: float = 1e-5

    def __init__(self, dims, eps: float = 1e-5):
        super().__init__()
        self.gamma = np.ones(dims)
        self.beta = np.zeros(dims)
        self.eps = eps

    def __call__(self, input: TensorValue):
        gamma = ops.constant(self.gamma, DType.float32)
        beta = ops.constant(self.beta, DType.float32)
        return ops.cast(
            ops.layer_norm(
                ops.cast(input, DType.float32),
                gamma=gamma,
                beta=beta,
                epsilon=self.eps,
            ),
            input.dtype,
        )


class NaiveLlama3(NaiveTransformer):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        num_hidden_layers: int,
        rope_theta: float,
        max_seq_len: int,
        rms_norm_eps: Optional[float],
        intermediate_size: int,
        interleaved_rope_weights: bool,
        rope_scaling: Optional[np.ndarray],
        vocab_size: int,
        dtype: DType,
        quantization_encoding: Optional[QuantizationEncoding],
        quantization_config: Optional[QuantizationConfig],
        kv_params: KVCacheParams,
        norm_method: Literal["rms_norm"] | Literal["layer_norm"],
        tie_word_embeddings: bool,
        stacked_mlp: bool,
        stacked_qkv: bool,
        logits_postprocessor: Callable[[TensorValue], TensorValue] | None,
        attention_multiplier: float,
        embedding_multiplier: float,
        residual_multiplier: float,
        devices: list[DeviceRef],
        clip_qkv: float | None,
    ):
        if stacked_qkv:
            raise ValueError(
                "Stacked QKV is not supported with naive caching strategy."
            )
        rope = RotaryEmbedding(
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
        layers = [
            NaiveTransformerBlock(
                attention=NaiveLLama3Attention(
                    kv_params,
                    hidden_size,
                    num_attention_heads,
                    num_key_value_heads,
                    rope,
                    dtype,
                    quantization_encoding,
                    linear_cls,
                    scale=attention_multiplier,
                    device=devices[0],
                    clip_qkv=clip_qkv,
                ),
                mlp=mlp_cls(
                    dtype,
                    quantization_encoding,
                    hidden_size,
                    intermediate_size,
                    linear_cls,
                    devices=devices,
                ),
                attention_norm=create_norm(),
                mlp_norm=create_norm(),
                residual_multiplier=residual_multiplier,
            )
            for i in range(num_hidden_layers)
        ]

        embedding_layer = EmbeddingV2(
            vocab_size,
            hidden_size,
            dtype,
            devices[0],
            quantization_encoding=quantization_encoding,
        )

        output = LinearV2(
            hidden_size,
            vocab_size,
            dtype,
            devices[0],
            quantization_encoding=quantization_encoding,
        )
        if tie_word_embeddings:
            output.set_shared_weight("weight", embedding_layer.weight)

        super().__init__(
            dim=hidden_size,
            n_heads=num_attention_heads,
            layers=layers,
            norm=create_norm(),
            output=output,
            theta=rope_theta,
            embedding=embedding_layer,
            embedding_multiplier=embedding_multiplier,
            logits_postprocessor=logits_postprocessor,
        )


class NaiveLLama3Attention(NaiveAttentionWithRope):
    def __init__(
        self,
        kv_params: KVCacheParams,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        rope: Union[OptimizedRotaryEmbedding, RotaryEmbedding],
        dtype: DType,
        quantization_encoding: Optional[QuantizationEncoding],
        linear_cls: Callable[..., LinearV2],
        scale: float | None,
        device: DeviceRef,
        clip_qkv: float | None,
    ):
        kv_weight_dim = (
            hidden_size // num_attention_heads
        ) * num_key_value_heads

        super().__init__(
            n_heads=num_attention_heads,
            kv_params=kv_params,
            dim=hidden_size,
            wk=linear_cls(
                in_dim=hidden_size,
                out_dim=kv_weight_dim,
                dtype=dtype,
                device=device,
                quantization_encoding=quantization_encoding,
                clip_weight=clip_qkv,
            ),
            wv=linear_cls(
                in_dim=hidden_size,
                out_dim=kv_weight_dim,
                dtype=dtype,
                device=device,
                quantization_encoding=quantization_encoding,
                clip_weight=clip_qkv,
            ),
            wq=linear_cls(
                in_dim=hidden_size,
                out_dim=hidden_size,
                dtype=dtype,
                device=device,
                quantization_encoding=quantization_encoding,
                clip_weight=clip_qkv,
            ),
            wo=linear_cls(
                in_dim=hidden_size,
                out_dim=hidden_size,
                dtype=dtype,
                device=device,
                quantization_encoding=quantization_encoding,
            ),
            rope=rope,
            scale=scale,
        )


class Llama3MLP(MLPV2):
    def __init__(
        self,
        dtype: DType,
        quantization_encoding: Optional[QuantizationEncoding],
        hidden_dim: int,
        feed_forward_length: int,
        linear_cls: Callable[..., LinearV2],
        devices: Sequence[DeviceRef] = (),
    ):
        super().__init__(
            gate_proj=linear_cls(
                in_dim=hidden_dim,
                out_dim=feed_forward_length,
                dtype=dtype,
                device=devices[0] if devices else None,
                quantization_encoding=quantization_encoding,
            ),
            down_proj=linear_cls(
                in_dim=feed_forward_length,
                out_dim=hidden_dim,
                dtype=dtype,
                device=devices[0] if devices else None,
                quantization_encoding=quantization_encoding,
            ),
            up_proj=linear_cls(
                in_dim=hidden_dim,
                out_dim=feed_forward_length,
                dtype=dtype,
                device=devices[0] if devices else None,
                quantization_encoding=quantization_encoding,
            ),
        )


class StackedMLP(LayerV2):
    def __init__(
        self,
        dtype: DType,
        quantization_encoding: Optional[QuantizationEncoding],
        hidden_dim: int,
        feed_forward_length: int,
        linear_cls: Callable[..., LinearV2],
        devices: Sequence[DeviceRef] = (),
    ):
        super().__init__()
        self.gate_up_proj = linear_cls(
            in_dim=hidden_dim,
            out_dim=feed_forward_length * 2,
            dtype=dtype,
            device=devices[0] if devices else None,
            quantization_encoding=quantization_encoding,
        )
        self.down_proj = linear_cls(
            in_dim=feed_forward_length,
            out_dim=hidden_dim,
            dtype=dtype,
            device=devices[0] if devices else None,
            quantization_encoding=quantization_encoding,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        up_states = self.gate_up_proj(x)

        gate = up_states[:, : up_states.shape.static_dims[0] // 2]
        up_states = up_states[:, up_states.shape.static_dims[0] // 2 :]

        return self.down_proj(ops.silu(gate) * up_states)
