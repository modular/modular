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
from typing import Callable, Optional

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.graph.quantization import QuantizationEncoding
from max.nn import (
    MLPV2,
    EmbeddingV2,
    GPTQLinearV2,
    LinearV2,
    Llama3RotaryEmbedding,
    Module,
    NaiveAttentionWithRope,
    NaiveTransformer,
    NaiveTransformerBlock,
    RMSNormV2,
    RotaryEmbedding,
)
from max.nn.kv_cache import KVCacheParams

from .model_config import Llama3Config


class ConstantLayerNorm(Module):
    """Layer normalization block with constant gamma and beta values."""

    gamma: np.ndarray
    beta: np.ndarray
    eps: float = 1e-5
    device: DeviceRef

    def __init__(
        self,
        dims,
        device: DeviceRef,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.gamma = np.ones(dims)
        self.beta = np.zeros(dims)
        self.eps = eps
        self.device = device

    def __call__(self, input: TensorValue):
        gamma = ops.constant(self.gamma, DType.float32, self.device)
        beta = ops.constant(self.beta, DType.float32, self.device)
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
    def __init__(self, config: Llama3Config):
        if config.stacked_qkv:
            raise ValueError(
                "Stacked QKV is not supported with naive caching strategy."
            )
        rope = Llama3RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            interleaved=config.interleaved_rope_weights,
            scaling_params=config.rope_scaling_params,
            device=config.devices[0],
        )

        create_norm: Callable[..., Module]
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
                ConstantLayerNorm, config.hidden_size, config.devices[0]
            )

        linear_cls: Callable[..., LinearV2]
        if config.quantization_config:
            linear_cls = functools.partial(
                GPTQLinearV2, quantization_config=config.quantization_config
            )
        else:
            linear_cls = LinearV2

        mlp_cls = StackedMLP if config.stacked_mlp else MLPV2
        layers = [
            NaiveTransformerBlock(
                attention=NaiveLLama3Attention(
                    config.kv_params,
                    config.hidden_size,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    rope,
                    config.dtype,
                    config.model_quantization_encoding,
                    linear_cls,
                    scale=config.attention_multiplier,
                    device=config.devices[0],
                    clip_qkv=config.clip_qkv,
                ),
                mlp=mlp_cls(
                    config.dtype,
                    config.model_quantization_encoding,
                    config.hidden_size,
                    config.intermediate_size,
                    config.devices,
                    linear_cls,
                ),
                attention_norm=create_norm(),
                mlp_norm=create_norm(),
                residual_multiplier=config.residual_multiplier,
            )
            for i in range(config.num_hidden_layers)
        ]

        embedding_layer = EmbeddingV2(
            config.vocab_size,
            config.hidden_size,
            config.dtype,
            config.devices[0],
            quantization_encoding=config.model_quantization_encoding,
        )

        output = LinearV2(
            config.hidden_size,
            config.vocab_size,
            config.dtype,
            config.devices[0],
            quantization_encoding=config.model_quantization_encoding,
        )
        if config.tie_word_embeddings:
            output.set_shared_weight("weight", embedding_layer.weight)

        super().__init__(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            layers=layers,
            norm=create_norm(),
            output=output,
            theta=config.rope_theta,
            embedding=embedding_layer,
            embedding_multiplier=config.embedding_multiplier,
            logits_postprocessor=config.logits_postprocessor,
            return_logits=config.return_logits,
        )


class NaiveLLama3Attention(NaiveAttentionWithRope):
    def __init__(
        self,
        kv_params: KVCacheParams,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        rope: RotaryEmbedding,
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


class StackedMLP(Module):
    def __init__(
        self,
        dtype: DType,
        quantization_encoding: Optional[QuantizationEncoding],
        hidden_dim: int,
        feed_forward_length: int,
        devices: Sequence[DeviceRef],
        linear_cls: Callable[..., LinearV2],
    ):
        super().__init__()
        self.gate_up_proj = linear_cls(
            in_dim=hidden_dim,
            out_dim=feed_forward_length * 2,
            dtype=dtype,
            device=devices[0],
            quantization_encoding=quantization_encoding,
        )
        self.down_proj = linear_cls(
            in_dim=feed_forward_length,
            out_dim=hidden_dim,
            dtype=dtype,
            device=devices[0],
            quantization_encoding=quantization_encoding,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        up_states = self.gate_up_proj(x)

        gate = up_states[:, : up_states.shape.static_dims[0] // 2]
        up_states = up_states[:, up_states.shape.static_dims[0] // 2 :]

        return self.down_proj(ops.silu(gate) * up_states)
