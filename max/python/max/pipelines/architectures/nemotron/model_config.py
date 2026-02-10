# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
"""Config for Nemotron models (NemotronForCausalLM)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.quantization import QuantizationEncoding
from max.graph.weights import WeightData, WeightsFormat, weights_format
from max.nn.legacy.float8_config import Float8Config
from max.nn.legacy.kv_cache import KVCacheParams
from max.nn.legacy.transformer import ReturnLogits
from max.pipelines.lib import (
    KVCacheConfig,
    PipelineConfig,
    RopeType,
    parse_float8_config,
    upper_bounded_default,
)
from max.pipelines.lib.interfaces.arch_config import ArchConfigWithKVCache
from transformers import AutoConfig
from typing_extensions import Self, override


@dataclass(kw_only=True)
class NemotronConfig(ArchConfigWithKVCache):
    """Model configuration for Nemotron graph construction and execution.

    Nemotron differs from LLaMA-family models in several ways:
    - Uses LayerNorm (not RMSNorm) with a configurable ``norm_eps``.
    - Uses a non-gated MLP with squared ReLU activation.
    - Applies partial rotary positional embeddings (typically 50 % of head_dim).
    - Specifies an explicit ``kv_channels`` / ``head_dim``.
    """

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    rope_theta: float
    max_seq_len: int
    intermediate_size: int
    vocab_size: int
    dtype: DType
    head_dim: int
    partial_rotary_factor: float
    norm_eps: float
    model_quantization_encoding: QuantizationEncoding | None
    kv_params: KVCacheParams
    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN
    norm_dtype: DType | None = None
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    stacked_qkv: bool = False
    attention_multiplier: float = 1.0
    embedding_multiplier: float = 1.0
    residual_multiplier: float = 1.0
    devices: list[DeviceRef]
    float8_config: Float8Config | None = None
    interleaved_rope_weights: bool = False

    def get_kv_params(self) -> KVCacheParams:
        return self.kv_params

    def get_max_seq_len(self) -> int:
        return self.max_seq_len

    @staticmethod
    def get_head_dim(huggingface_config: AutoConfig) -> int:
        """Returns head dimension, preferring explicit ``kv_channels``."""
        if hasattr(huggingface_config, "kv_channels"):
            return huggingface_config.kv_channels
        if hasattr(huggingface_config, "head_dim"):
            return huggingface_config.head_dim
        return (
            huggingface_config.hidden_size
            // huggingface_config.num_attention_heads
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        return huggingface_config.num_hidden_layers

    @staticmethod
    def construct_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return KVCacheParams(
            dtype=cache_dtype,
            n_kv_heads=huggingface_config.num_key_value_heads,
            head_dim=NemotronConfig.get_head_dim(huggingface_config),
            num_layers=NemotronConfig.get_num_layers(huggingface_config),
            page_size=kv_cache_config.kv_cache_page_size,
            cache_strategy=kv_cache_config.cache_strategy,
            enable_prefix_caching=kv_cache_config.enable_prefix_caching,
            enable_kvcache_swapping_to_host=kv_cache_config.enable_kvcache_swapping_to_host,
            host_kvcache_swap_space_gb=kv_cache_config.host_kvcache_swap_space_gb,
            devices=devices,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
        )

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
    ) -> int:
        try:
            return upper_bounded_default(
                upper_bound=huggingface_config.max_position_embeddings,
                default=pipeline_config.max_length,
            )
        except ValueError as e:
            raise ValueError(
                "Unable to infer max_length for Nemotron, the provided "
                f"max_length ({pipeline_config.max_length}) exceeds the "
                f"model's max_position_embeddings "
                f"({huggingface_config.max_position_embeddings})."
            ) from e

    @override
    @classmethod
    def initialize(cls, pipeline_config: PipelineConfig) -> Self:
        huggingface_config = pipeline_config.model.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required for "
                f"'{pipeline_config.model.model_path}', but config could not "
                f"be loaded. Please ensure the model repository contains a "
                f"valid config.json file."
            )

        kv_cache_config = pipeline_config.model.kv_cache
        quantization_encoding = pipeline_config.model.quantization_encoding
        if quantization_encoding is None:
            raise ValueError("quantization_encoding must not be None")
        dtype = quantization_encoding.dtype
        cache_dtype = pipeline_config.model.kv_cache.cache_dtype
        n_devices = len(pipeline_config.model.device_specs)

        _weights_format = weights_format(pipeline_config.model.weight_path)
        interleaved_rope_weights = (
            _weights_format == WeightsFormat.gguf
            and pipeline_config.model.rope_type == RopeType.normal
        )

        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in pipeline_config.model.device_specs[:n_devices]
        ]

        head_dim = cls.get_head_dim(huggingface_config)
        attention_multiplier = getattr(
            huggingface_config,
            "attention_multiplier",
            math.sqrt(1.0 / float(head_dim)),
        )
        embedding_multiplier = getattr(
            huggingface_config, "embedding_multiplier", 1.0
        )
        residual_multiplier = getattr(
            huggingface_config, "residual_multiplier", 1.0
        )

        return cls(
            hidden_size=huggingface_config.hidden_size,
            num_attention_heads=huggingface_config.num_attention_heads,
            num_key_value_heads=huggingface_config.num_key_value_heads,
            num_hidden_layers=huggingface_config.num_hidden_layers,
            rope_theta=huggingface_config.rope_theta,
            intermediate_size=huggingface_config.intermediate_size,
            vocab_size=huggingface_config.vocab_size,
            dtype=dtype,
            head_dim=head_dim,
            partial_rotary_factor=getattr(
                huggingface_config, "partial_rotary_factor", 0.5
            ),
            norm_eps=getattr(huggingface_config, "norm_eps", 1e-5),
            model_quantization_encoding=pipeline_config.model.graph_quantization_encoding,
            max_seq_len=cls.calculate_max_seq_len(
                pipeline_config, huggingface_config=huggingface_config
            ),
            kv_params=cls.construct_kv_params(
                huggingface_config=huggingface_config,
                pipeline_config=pipeline_config,
                devices=device_refs,
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
            ),
            attention_multiplier=attention_multiplier,
            embedding_multiplier=embedding_multiplier,
            residual_multiplier=residual_multiplier,
            devices=device_refs,
            interleaved_rope_weights=interleaved_rope_weights,
        )

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: dict[str, WeightData],
        return_logits: ReturnLogits,
    ) -> None:
        """Complete configuration that requires introspection of the weights."""

        # Strip common prefixes so downstream key checks work uniformly.
        has_model_prefix = any(k.startswith("model.") for k in state_dict)
        if has_model_prefix:
            normalized = {
                k.removeprefix("model."): v
                for k, v in state_dict.items()
                if k.startswith("model.")
            }
        else:
            normalized = dict(state_dict)

        # Float8 support.
        float8_config = parse_float8_config(
            huggingface_config, normalized, self.dtype
        )

        # Norm dtype from weights (only used when float8 needs a specific norm dtype).
        norm_dtype = None
        if "layers.0.input_layernorm.weight" in normalized:
            norm_dtype = normalized["layers.0.input_layernorm.weight"].dtype

        # Tie word embeddings.
        if "tie_word_embeddings" in huggingface_config:
            tie_word_embeddings = huggingface_config.tie_word_embeddings
        else:
            tie_word_embeddings = (
                getattr(huggingface_config, "tie_word_embeddings", False)
                or "lm_head.weight" not in normalized
            )

        self.tie_word_embeddings = tie_word_embeddings
        self.float8_config = float8_config
        self.norm_dtype = norm_dtype
        self.return_logits = return_logits

        # Detect stacked QKV weights.
        self.stacked_qkv = (
            "layers.0.self_attn.qkv_proj.weight" in normalized
        )

        # Nemotron does not use attention bias by default.
        self.attention_bias = False
