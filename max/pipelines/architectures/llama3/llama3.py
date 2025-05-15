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
from collections.abc import Sequence
from typing import Callable, Optional

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.graph.quantization import QuantizationEncoding
from max.nn import (
    MLP,
    AttentionWithRope,
    Embedding,
    GGUFQAttentionWithRope,
    GPTQAttentionWithRope,
    GPTQLinear,
    Linear,
    Llama3RotaryEmbedding,
    Module,
    RMSNorm,
    Transformer,
    TransformerBlock,
)
from max.nn.kv_cache import (
    FetchContinuousBatchingKVCacheCollection,
    FetchPagedKVCacheCollection,
    KVCacheStrategy,
)

from .model_config import Llama3Config


class StackedMLP(Module):
    def __init__(
        self,
        dtype: DType,
        quantization_encoding: Optional[QuantizationEncoding],
        hidden_dim: int,
        feed_forward_length: int,
        devices: Sequence[DeviceRef],
        linear_cls: Callable[..., Linear],
        has_scale: bool = False,
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


class ConstantLayerNorm(Module):
    """Layer normalization block with constant gamma and beta values."""

    gamma: np.ndarray
    beta: np.ndarray
    eps: float = 1e-5
    device: DeviceRef
    dtype: DType

    def __init__(
        self,
        dims,
        device: DeviceRef,
        eps: float = 1e-5,
        dtype: DType = DType.float32,
    ):
        super().__init__()
        self.gamma = np.ones(dims)
        self.beta = np.zeros(dims)
        self.eps = eps
        self.device = device
        self.dtype = dtype

    def __call__(self, input: TensorValue):
        gamma = ops.constant(self.gamma, self.dtype, self.device)
        beta = ops.constant(self.beta, self.dtype, self.device)
        return ops.cast(
            ops.layer_norm(
                ops.cast(input, DType.float32),
                gamma=gamma,
                beta=beta,
                epsilon=self.eps,
            ),
            input.dtype,
        )


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
            device=config.devices[0],
        )

        # Select norm layer class.
        create_norm: Callable[..., Module]
        if config.norm_method == "rms_norm":
            if config.rms_norm_eps is None:
                raise ValueError(
                    "rms_norm_eps cannot be None for model that uses RMSNorm."
                )
            create_norm = functools.partial(
                RMSNorm,
                config.hidden_size,
                config.rms_norm_eps,
                dtype=config.norm_dtype or DType.float32,
                multiply_before_cast=False,  # disable Gemma3-style scaling
            )
        else:
            create_norm = functools.partial(
                ConstantLayerNorm,
                config.hidden_size,
                config.devices[0],
                dtype=config.norm_dtype or DType.float32,
            )

        # Select linear layer class.
        linear_cls: Callable[..., Linear]
        if config.quantization_config:
            linear_cls = functools.partial(
                GPTQLinear, quantization_config=config.quantization_config
            )
        else:
            linear_cls = functools.partial(
                Linear, float8_config=config.float8_config
            )
        if config.stacked_mlp and config.float8_config:
            msg = "StackedMLP and float8 are not compatible"
            raise ValueError(msg)
        mlp_cls = (
            StackedMLP
            if config.stacked_mlp
            else functools.partial(MLP, float8_config=config.float8_config)
        )
        attention_cls: Callable[..., AttentionWithRope]
        if config.model_quantization_encoding == QuantizationEncoding.GPTQ:
            assert config.quantization_config is not None
            assert not config.attention_bias, (
                "Attention bias is not supported for GPTQAttentionWithRope."
            )
            attention_cls = functools.partial(
                GPTQAttentionWithRope,
                quantization_config=config.quantization_config,
                scale=config.attention_multiplier,
            )
        elif config.model_quantization_encoding is not None:
            assert not config.attention_bias, (
                "Attention bias is not supported for GGUFQAttentionWithRope."
            )
            attention_cls = functools.partial(
                GGUFQAttentionWithRope,
                quantization_encoding=config.model_quantization_encoding,
                scale=config.attention_multiplier,
            )
        else:
            attention_cls = functools.partial(
                AttentionWithRope,
                stacked_qkv=config.stacked_qkv,
                scale=config.attention_multiplier,
                clip_qkv=config.clip_qkv,
                has_bias=config.attention_bias,
                float8_config=config.float8_config,
            )

        layers = [
            TransformerBlock(
                attention=attention_cls(
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    hidden_size=config.hidden_size,
                    kv_params=config.kv_params,
                    dtype=config.dtype,
                    rope=rope,
                    linear_cls=linear_cls,
                    devices=config.devices,
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

        # Create Embedding and output layers.
        embedding_output_dtype = config.dtype
        embedding_output_quantization = config.model_quantization_encoding
        if config.model_quantization_encoding == QuantizationEncoding.GPTQ:
            embedding_output_dtype = DType.bfloat16
            embedding_output_quantization = None
        if config.float8_config and config.float8_config.embedding_output_dtype:
            embedding_output_dtype = config.float8_config.embedding_output_dtype
        embedding_layer = Embedding(
            config.vocab_size,
            config.hidden_size,
            embedding_output_dtype,
            config.devices[0],
            quantization_encoding=embedding_output_quantization,
        )
        output = Linear(
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
            kv_collection_constructor=kv_collection_cls(
                config.kv_params, num_layers=config.num_hidden_layers
            ),
            return_logits=config.return_logits,
            embedding_multiplier=config.embedding_multiplier,
            logits_postprocessor=config.logits_postprocessor,
        )
