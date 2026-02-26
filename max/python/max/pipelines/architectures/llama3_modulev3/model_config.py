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
"""Config for Llama3 models (ModuleV3)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData, WeightsFormat, weights_format
from max.nn.kv_cache import KVCacheParams
from max.nn.rotary_embedding import (
    Llama3RopeScalingParams,
    LongRoPEScalingParams,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib import (
    KVCacheConfig,
    PipelineConfig,
    upper_bounded_default,
)
from max.pipelines.lib.config.config_enums import supported_encoding_dtype
from max.pipelines.lib.interfaces.arch_config import ArchConfigWithKVCache
from transformers import AutoConfig
from typing_extensions import Self, override

from .layers.rotary_embedding import LongRoPERotaryEmbedding


@dataclass(kw_only=True)
class Llama3Config(ArchConfigWithKVCache):
    """Model configuration for Llama3 graph construction/execution."""

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    rope_theta: float
    rope_scaling_params: Llama3RopeScalingParams | None
    max_seq_len: int
    intermediate_size: int
    interleaved_rope_weights: bool
    vocab_size: int
    dtype: DType
    kv_params: KVCacheParams
    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN
    norm_method: Literal["rms_norm"] | Literal["layer_norm"] = "rms_norm"
    attention_bias: bool = False
    rms_norm_eps: float | None = None
    tie_word_embeddings: bool = False
    stacked_mlp: bool = False
    stacked_qkv: bool = False
    attention_multiplier: float
    embedding_multiplier: float
    residual_multiplier: float
    devices: list[DeviceRef]
    clip_qkv: float | None = None
    norm_elementwise_affine: bool = True
    longrope_scaling_params: LongRoPEScalingParams | None = None
    logits_scaling: float = 1.0
    return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE

    def get_kv_params(self) -> KVCacheParams:
        return self.kv_params

    def get_max_seq_len(self) -> int:
        return self.max_seq_len

    @staticmethod
    def get_head_dim(huggingface_config: AutoConfig) -> int:
        if hasattr(huggingface_config, "head_dim"):
            return huggingface_config.head_dim
        else:
            return (
                huggingface_config.hidden_size
                // huggingface_config.num_attention_heads
            )

    @staticmethod
    def get_head_dim_from_config(config: Llama3Config) -> int:
        return config.kv_params.head_dim

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        return huggingface_config.num_hidden_layers

    @staticmethod
    def calculate_attention_multiplier(
        huggingface_config: AutoConfig,
    ) -> float:
        return getattr(
            huggingface_config,
            "attention_multiplier",
            math.sqrt(
                1.0 / float(Llama3Config.get_head_dim(huggingface_config))
            ),
        )

    @staticmethod
    def construct_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return kv_cache_config.to_params(
            dtype=cache_dtype,
            n_kv_heads=huggingface_config.num_key_value_heads,
            head_dim=Llama3Config.get_head_dim(huggingface_config),
            num_layers=Llama3Config.get_num_layers(huggingface_config),
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
                default=pipeline_config.model.max_length,
            )
        except ValueError as e:
            raise ValueError(
                "Unable to infer max_length for Llama3, the provided "
                f"max_length ({pipeline_config.model.max_length}) exceeds the "
                f"model's max_position_embeddings "
                f"({huggingface_config.max_position_embeddings})."
            ) from e

    @override
    @classmethod
    def initialize(cls, pipeline_config: PipelineConfig) -> Self:
        huggingface_config = pipeline_config.model.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required for '{pipeline_config.model.model_path}', "
                "but config could not be loaded. "
                "Please ensure the model repository contains a valid config.json file."
            )

        kv_cache_config = pipeline_config.model.kv_cache
        quantization_encoding = pipeline_config.model.quantization_encoding
        if quantization_encoding is None:
            raise ValueError("quantization_encoding must not be None")
        dtype = supported_encoding_dtype(quantization_encoding)
        cache_dtype = pipeline_config.model.kv_cache.cache_dtype

        _weights_format = weights_format(pipeline_config.model.weight_path)
        interleaved_rope_weights = (
            _weights_format == WeightsFormat.gguf
            and pipeline_config.model.rope_type == "normal"
        )

        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in pipeline_config.model.device_specs
        ]

        embedding_multiplier = getattr(
            huggingface_config, "embedding_multiplier", 1.0
        )
        residual_multiplier = getattr(
            huggingface_config, "residual_multiplier", 1.0
        )
        rope_scaling_params: Llama3RopeScalingParams | None = None
        longrope_scaling_params: LongRoPEScalingParams | None = None
        rope_scaling = huggingface_config.rope_scaling

        if rope_scaling is not None:
            rope_type = rope_scaling.get("type")
            rope_type_alt = rope_scaling.get("rope_type")
            if rope_type is None and rope_type_alt is None:
                raise ValueError(
                    "Neither 'type' nor 'rope_type' found in rope_scaling huggingface config"
                )
            if rope_type == "llama3" or rope_type_alt == "llama3":
                rope_scaling_params = Llama3RopeScalingParams(
                    factor=rope_scaling["factor"],
                    low_freq_factor=rope_scaling["low_freq_factor"],
                    high_freq_factor=rope_scaling["high_freq_factor"],
                    orig_max_position=rope_scaling[
                        "original_max_position_embeddings"
                    ],
                )
            elif rope_type == "longrope" or rope_type_alt == "longrope":
                longrope_scaling_params = LongRoPEScalingParams(
                    short_factor=rope_scaling["short_factor"],
                    long_factor=rope_scaling["long_factor"],
                    original_max_position=huggingface_config.original_max_position_embeddings,
                    max_position_embeddings=huggingface_config.max_position_embeddings,
                )
                rope_scaling_params = None

        # Calculate base attention multiplier.
        base_attention_multiplier = Llama3Config.calculate_attention_multiplier(
            huggingface_config
        )

        # Apply LongRoPE attention scaling if needed.
        attention_multiplier = base_attention_multiplier
        if longrope_scaling_params is not None:
            rope_embedding = LongRoPERotaryEmbedding(
                dim=huggingface_config.hidden_size,
                n_heads=huggingface_config.num_attention_heads,
                theta=huggingface_config.rope_theta,
                max_seq_len=Llama3Config.calculate_max_seq_len(
                    pipeline_config, huggingface_config=huggingface_config
                ),
                device=device_refs[0].to_device(),
                interleaved=interleaved_rope_weights,
                scaling_params=longrope_scaling_params,
            )
            attention_multiplier = rope_embedding.compute_scale()

        return cls(
            hidden_size=huggingface_config.hidden_size,
            num_attention_heads=huggingface_config.num_attention_heads,
            num_key_value_heads=huggingface_config.num_key_value_heads,
            num_hidden_layers=huggingface_config.num_hidden_layers,
            rope_theta=huggingface_config.rope_theta,
            rope_scaling_params=rope_scaling_params,
            longrope_scaling_params=longrope_scaling_params,
            intermediate_size=huggingface_config.intermediate_size,
            interleaved_rope_weights=interleaved_rope_weights,
            vocab_size=huggingface_config.vocab_size,
            dtype=dtype,
            max_seq_len=Llama3Config.calculate_max_seq_len(
                pipeline_config, huggingface_config=huggingface_config
            ),
            kv_params=Llama3Config.construct_kv_params(
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
            clip_qkv=getattr(huggingface_config, "clip_qkv", None),
            logits_scaling=getattr(huggingface_config, "logits_scaling", 1.0),
        )

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: dict[str, WeightData],
        return_logits: ReturnLogits,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
        norm_method: Literal["rms_norm"] | Literal["layer_norm"] = "rms_norm",
        attention_bias: bool = False,
    ) -> None:
        """Define parameters that can't be determined just from the pipeline config."""

        # Normalize the LLM state dict so downstream introspection sees canonical
        # Llama-style keys (no "language_model." or "model." prefix).
        def _strip_prefix(s: str, prefix: str) -> str:
            return s.removeprefix(prefix)

        has_lm_prefix = any(k.startswith("language_model.") for k in state_dict)
        has_model_prefix = any(k.startswith("model.") for k in state_dict)

        if has_lm_prefix:
            normalized_state_dict: dict[str, WeightData] = {
                _strip_prefix(k, "language_model."): v
                for k, v in state_dict.items()
                if k.startswith("language_model.")
            }
        elif has_model_prefix:
            normalized_state_dict = {
                _strip_prefix(k, "model."): v
                for k, v in state_dict.items()
                if k.startswith("model.")
            }
        else:
            normalized_state_dict = dict(state_dict)

        # When tie_word_embeddings=True, the embedding weights are shared with
        # the output weights.
        if "tie_word_embeddings" in huggingface_config:
            tie_word_embeddings = huggingface_config.tie_word_embeddings
        else:
            tie_word_embeddings = (
                getattr(huggingface_config, "tie_word_embeddings", False)
                or "lm_head.weight" not in normalized_state_dict
            )

        rms_norm_eps = None
        if norm_method == "rms_norm":
            if huggingface_config.model_type == "exaone":
                rms_norm_eps = huggingface_config.layer_norm_epsilon
            else:
                rms_norm_eps = huggingface_config.rms_norm_eps

        self.norm_method = norm_method
        self.rms_norm_eps = rms_norm_eps
        self.tie_word_embeddings = tie_word_embeddings
        self.stacked_mlp = (
            "layers.0.mlp.gate_up_proj.weight" in normalized_state_dict
        )
        self.stacked_qkv = (
            "layers.0.self_attn.qkv_proj.weight" in normalized_state_dict
        )
        self.attention_bias = attention_bias
        self.return_logits = return_logits
        self.return_hidden_states = return_hidden_states
