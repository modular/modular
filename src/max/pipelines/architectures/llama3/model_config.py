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
from max.nn import (
    Float8Config,
    Float8Scaling,
    Llama3RopeScalingParams,
    ReturnLogits,
)
from max.nn.kv_cache import KVCacheParams
from max.pipelines.lib import (
    KVCacheConfig,
    MAXModelConfig,
    MAXModelConfigBase,
    PipelineConfig,
    RopeType,
    upper_bounded_default,
)
from transformers import AutoConfig


@dataclass
class Llama3ConfigBase(MAXModelConfigBase):
    """Base configuration for Llama3 models."""

    # Required fields
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
    model_quantization_encoding: Optional[QuantizationEncoding]
    quantization_config: Optional[QuantizationConfig]
    kv_params: KVCacheParams
    return_logits: ReturnLogits
    norm_method: Literal["rms_norm"] | Literal["layer_norm"]
    norm_dtype: DType | None
    attention_bias: bool
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
    float8_config: Float8Config | None

    @staticmethod
    def help() -> dict[str, str]:
        return {}


@dataclass
class Llama3Config(MAXModelConfig, Llama3ConfigBase):
    """Implementation of MAXModelConfig for Llama3 models."""

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
            enable_kvcache_swapping_to_host=kv_cache_config.enable_kvcache_swapping_to_host,
            host_kvcache_swap_space_gb=kv_cache_config.host_kvcache_swap_space_gb,
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
        return_logits: ReturnLogits,
        norm_method: Literal["rms_norm"] | Literal["layer_norm"] = "rms_norm",
        attention_bias: bool = False,
    ) -> Llama3Config:
        _weights_format = weights_format(
            pipeline_config.model_config.weight_path
        )
        interleaved_rope_weights = (
            _weights_format == WeightsFormat.gguf
            and pipeline_config.model_config.rope_type == RopeType.normal
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

        norm_dtype = None
        float8_config = None
        if dtype == DType.float8_e4m3fn:
            input_scale_name = "layers.0.mlp.down_proj.input_scale"
            weight_scale_name = "layers.0.mlp.down_proj.weight_scale"

            has_static_input_scale = input_scale_name in state_dict
            input_scale_scalar = False
            if has_static_input_scale:
                input_scale_scalar = state_dict[input_scale_name].shape in [
                    [],
                    [1],
                ]
            weight_scale_scalar = state_dict[weight_scale_name].shape in [
                [],
                [1],
            ]

            attn_float8 = (
                state_dict["layers.0.self_attn.k_proj.weight"].dtype
                == DType.float8_e4m3fn
            )

            if (
                not has_static_input_scale
                or not input_scale_scalar
                or not weight_scale_scalar
            ):
                msg = "float8 models only support static scaling input and weight scaling currently"
                raise ValueError(msg)

            float8_config = Float8Config(
                input_scaling=Float8Scaling.TENSOR,
                input_scale_dtype=state_dict[input_scale_name].dtype,
                weight_scaling=Float8Scaling.TENSOR,
                weight_scale_dtype=state_dict[weight_scale_name].dtype,
                attn_in_float8=attn_float8,
                embedding_output_dtype=(
                    state_dict["lm_head.weight"].dtype
                    if "lm_head.weight" in state_dict
                    else None
                ),
            )
            norm_dtype = (
                state_dict["layers.0.input_layernorm.weight"].dtype
                if "layers.0.input_layernorm.weight" in state_dict
                else None
            )

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
            # TODO: Since pipeline_config.model_config is a MAXModelConfig, these
            # fields should not have to reinstantiated. Once we roll out the final
            # iteration of MAXModelConfig, it will automatically instantiate based
            # on the underlying model repo id.
            model_quantization_encoding=pipeline_config.model_config.graph_quantization_encoding,
            quantization_config=pipeline_config.model_config._quant_config,
            return_logits=return_logits,
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
            norm_dtype=norm_dtype,
            attention_bias=attention_bias,
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
            float8_config=float8_config,
        )
