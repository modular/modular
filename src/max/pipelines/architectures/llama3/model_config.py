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
"""Config for Llama3 models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal, Optional

from max.dtype import DType
from max.graph import DeviceRef, TensorValue
from max.graph.quantization import QuantizationConfig, QuantizationEncoding
from max.graph.weights import WeightData, WeightsFormat, weights_format
from max.nn import Llama3RopeScalingParams
from max.pipelines import upper_bounded_default
from max.pipelines.config import (
    KVCacheConfig,
    MAXConfig,
    PipelineConfig,
    RopeType,
)
from max.pipelines.kv_cache import KVCacheParams
from transformers import AutoConfig


# TODO(zheng): Move this under MAXModelConfig. The challenge here is that
# MAXModelConfig has optional fields, and Llama3Config has required fields.
# We can work around this by having a superclass of MAXModelConfig that has
# the abstract methods, and then having Llama3Config extend that.
@dataclass
class Llama3Config(MAXConfig):
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    rope_theta: float
    rope_scaling_params: Optional[Llama3RopeScalingParams]
    max_seq_len: int
    intermediate_size: int
    interleaved_rope_weights: bool
    vocab_size: int
    dtype: DType
    quantization_encoding: Optional[QuantizationEncoding]
    quantization_config: Optional[QuantizationConfig]
    kv_params: KVCacheParams
    all_logits: bool
    norm_method: Literal["rms_norm"] | Literal["layer_norm"]
    rms_norm_eps: Optional[float]
    tie_word_embeddings: bool
    stacked_mlp: bool
    stacked_qkv: bool
    logits_postprocessor: Callable[[TensorValue], TensorValue] | None
    attention_multiplier: float
    embedding_multiplier: float
    residual_multiplier: float
    devices: list[DeviceRef]
    clip_qkv: Optional[float]

    @staticmethod
    def help() -> dict[str, str]:
        return {}

    @staticmethod
    def calculate_attention_multiplier(
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> float:
        """The attention multiplier is a scalar that scales the attention scores.
        It is used to control the variance of the attention scores.

        This function is used to get the attention multiplier from the
        huggingface config. If the attention multiplier is not set, it will be
        calculated as the square root of 1.0 divided by the head dimension.
        """
        return getattr(
            huggingface_config,
            "attention_multiplier",
            math.sqrt(
                1.0
                / float(
                    Llama3Config.get_kv_params(
                        huggingface_config=huggingface_config,
                        n_devices=n_devices,
                        kv_cache_config=kv_cache_config,
                        cache_dtype=cache_dtype,
                    ).head_dim
                )
            ),
        )

    # TODO(zheng): Figure out a scalable abstract method for all MAXModelConfigs.
    @staticmethod
    def get_kv_params(
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return KVCacheParams(
            dtype=cache_dtype,
            n_kv_heads=huggingface_config.num_key_value_heads,
            head_dim=(
                huggingface_config.hidden_size
                // huggingface_config.num_attention_heads
            ),
            page_size=kv_cache_config.kv_cache_page_size,
            cache_strategy=kv_cache_config.cache_strategy,
            enable_prefix_caching=kv_cache_config.enable_prefix_caching,
            n_devices=n_devices,
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        return huggingface_config.num_hidden_layers

    # TODO(zheng): Figure out a scalable abstract method for all MAXModelConfigs.
    # Also, these should just be class properties since they're already made
    # unique as a model config.
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
            msg = (
                "Unable to infer max_length for Llama3, the provided "
                f"max_length ({pipeline_config.max_length}) exceeds the "
                f"model's max_position_embeddings "
                f"({huggingface_config.max_position_embeddings})."
            )
            raise ValueError(msg) from e

    # TODO(zheng): Figure out a scalable abstract method for all MAXModelConfigs.
    @staticmethod
    def generate(
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        state_dict: dict[str, WeightData],
        dtype: DType,
        n_devices: int,
        logits_postprocessor: Callable[[TensorValue], TensorValue] | None,
        cache_dtype: DType,
        kv_cache_config: KVCacheConfig,
        norm_method: Literal["rms_norm"] | Literal["layer_norm"] = "rms_norm",
    ) -> Llama3Config:
        _weights_format = weights_format(
            pipeline_config.model_config.weight_path
        )
        interleaved_rope_weights = (
            _weights_format == WeightsFormat.gguf
            and pipeline_config.rope_type == RopeType.normal
        )
        rms_norm_eps = None
        if norm_method == "rms_norm":
            if huggingface_config.model_type == "exaone":
                rms_norm_eps = huggingface_config.layer_norm_epsilon
            else:
                rms_norm_eps = huggingface_config.rms_norm_eps

        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in pipeline_config.model_config.device_specs
        ]

        # When tie_word_embeddings=True, the embedding weights are shared with
        # the output weights.
        tie_word_embeddings = (
            getattr(huggingface_config, "tie_word_embeddings", False)
            or "lm_head.weight" not in state_dict
        )
        embedding_multiplier = getattr(
            huggingface_config, "embedding_multiplier", 1.0
        )
        residual_multiplier = getattr(
            huggingface_config, "residual_multiplier", 1.0
        )
        rope_scaling_params = None
        rope_scaling = huggingface_config.rope_scaling

        if rope_scaling is not None:
            # Since "rope_type" huggingface config is not standardized, we need
            # to check for both "type" and "rope_type" keys.
            # TODO: A better solution would be for those family of models to
            # create their own subclass of MAXModelConfig or Llama3Config, then
            # parts of it like rope_scaling to account for such differences.
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

        return Llama3Config(
            hidden_size=huggingface_config.hidden_size,
            num_attention_heads=huggingface_config.num_attention_heads,
            num_key_value_heads=huggingface_config.num_key_value_heads,
            num_hidden_layers=huggingface_config.num_hidden_layers,
            rope_theta=huggingface_config.rope_theta,
            rope_scaling_params=rope_scaling_params,
            rms_norm_eps=rms_norm_eps,
            intermediate_size=huggingface_config.intermediate_size,
            interleaved_rope_weights=interleaved_rope_weights,
            vocab_size=huggingface_config.vocab_size,
            dtype=dtype,
            quantization_encoding=pipeline_config.graph_quantization_encoding,
            quantization_config=pipeline_config.model_config._quant_config,
            all_logits=pipeline_config.enable_echo,
            max_seq_len=Llama3Config.calculate_max_seq_len(
                pipeline_config, huggingface_config=huggingface_config
            ),
            kv_params=Llama3Config.get_kv_params(
                huggingface_config=huggingface_config,
                n_devices=n_devices,
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
            ),
            norm_method=norm_method,
            tie_word_embeddings=tie_word_embeddings,
            stacked_mlp="layers.0.mlp.gate_up_proj.weight" in state_dict,
            stacked_qkv="layers.0.self_attn.qkv_proj.weight" in state_dict,
            logits_postprocessor=logits_postprocessor,
            attention_multiplier=Llama3Config.calculate_attention_multiplier(
                huggingface_config,
                n_devices,
                kv_cache_config,
                cache_dtype,
            ),
            embedding_multiplier=embedding_multiplier,
            residual_multiplier=residual_multiplier,
            devices=device_refs,
            clip_qkv=getattr(huggingface_config, "clip_qkv", None),
        )
