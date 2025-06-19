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
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable, Literal

from max.dtype import DType
from max.graph import DeviceRef, TensorValue
from max.graph.quantization import QuantizationConfig, QuantizationEncoding
from max.graph.weights import WeightData, WeightsFormat, weights_format
from max.nn import (
    Float8Config,
    Float8InputScaleSpec,
    Float8ScaleGranularity,
    Float8ScaleOrigin,
    Float8WeightScaleSpec,
    Llama3RopeScalingParams,
    Llama3RotaryEmbedding,
    LongRoPERotaryEmbedding,
    LongRoPEScalingParams,
    ReturnLogits,
    RotaryEmbedding,
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


def create_rope_embedding(
    hidden_size: int,
    num_attention_heads: int,
    rope_theta: float,
    max_seq_len: int,
    interleaved_rope_weights: bool,
    rope_scaling_params: Llama3RopeScalingParams | None,
    longrope_scaling_params: LongRoPEScalingParams | None,
    device: DeviceRef,
) -> RotaryEmbedding:
    """Create appropriate RoPE embedding based on scaling parameters.

    Args:
        hidden_size: Model hidden dimension
        num_attention_heads: Number of attention heads
        rope_theta: RoPE theta parameter (typically 10000.0)
        max_seq_len: Maximum sequence length
        interleaved_rope_weights: Whether to use interleaved RoPE weights
        rope_scaling_params: Llama3 RoPE scaling parameters (if any)
        longrope_scaling_params: LongRoPE scaling parameters (if any)
        device: Device to place tensors on

    Returns:
        Configured RoPE embedding instance
    """
    if longrope_scaling_params is not None:
        return LongRoPERotaryEmbedding(
            dim=hidden_size,
            n_heads=num_attention_heads,
            theta=rope_theta,
            max_seq_len=max_seq_len,
            interleaved=interleaved_rope_weights,
            scaling_params=longrope_scaling_params,
            device=device,
        )
    else:
        return Llama3RotaryEmbedding(
            dim=hidden_size,
            n_heads=num_attention_heads,
            theta=rope_theta,
            max_seq_len=max_seq_len,
            interleaved=interleaved_rope_weights,
            scaling_params=rope_scaling_params,
            device=device,
        )


def _quantized_layers_and_embedding_dtype(
    huggingface_config: AutoConfig,
    ignored_modules: set[str],
    state_dict: Mapping[str, WeightData],
) -> tuple[set[int], set[int], DType | None]:
    """Helper to determine quantized MLP/Attention layers and embedding output dtype."""
    num_hidden_layers = huggingface_config.num_hidden_layers
    mlp_in_float8: set[int] = set()
    attn_qkv_in_float8: set[int] = set()

    for i in range(num_hidden_layers):
        # Check MLP components (gate_proj, up_proj, down_proj).
        not_converted_mlp_modules = [
            f"model.layers.{i}.mlp.{proj}" in ignored_modules
            for proj in ["gate_proj", "up_proj", "down_proj"]
        ]
        is_mlp_not_converted = any(not_converted_mlp_modules)
        if not is_mlp_not_converted:
            mlp_in_float8.add(i)
        elif not all(not_converted_mlp_modules):
            raise ValueError(
                "float8 quantization currently assumes uniform quantization for MLPs"
            )

        # Check Attention QKV components (q_proj, k_proj, v_proj, o_proj)
        not_converted_attn_qkv_modules = [
            f"model.layers.{i}.self_attn.{proj}" in ignored_modules
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]
        ]
        is_attn_qkv_not_converted = any(not_converted_attn_qkv_modules)
        if not is_attn_qkv_not_converted:
            attn_qkv_in_float8.add(i)
        elif not all(not_converted_attn_qkv_modules):
            raise ValueError(
                "float8 quantization currently assumes uniform quantization for attention QKV and output projections"
            )

    # Determine embedding_output_dtype
    embedding_output_dtype: DType | None
    if "lm_head.weight" in state_dict:
        embedding_output_dtype = state_dict["lm_head.weight"].dtype
    elif "model.embed_tokens.weight" in state_dict:
        # Handle tied embeddings.
        embedding_output_dtype = state_dict["model.embed_tokens.weight"].dtype
    elif "lm_head" in ignored_modules:
        # If `lm_head` is in ignored_modules, but its weight isn't in the
        # checkpoint, and neither are the embedding weights, consider that a
        # buggy checkpoint.
        raise ValueError("cannot determine original type from checkpoint")
    else:
        # Default to `lm_head` being quantized to float8.
        embedding_output_dtype = DType.float8_e4m3fn

    return mlp_in_float8, attn_qkv_in_float8, embedding_output_dtype


def _parse_compressed_tensors_float8_config(
    huggingface_config: AutoConfig,
    state_dict: Mapping[str, WeightData],
    dtype: DType,
) -> Float8Config:
    """Parses a Float8Config in the compressed-tensors format."""

    # This function specifically handles "compressed-tensors" style.
    # It assumes hf_quant_config and its structure are present.

    hf_quant_config = getattr(huggingface_config, "quantization_config", None)
    # Verification should done by the caller.
    assert hf_quant_config and (
        hf_quant_config.get("quant_method") == "compressed-tensors"
        # compressed-tensors might have a missing quant_method if it's the default.
        or not hf_quant_config.get("quant_method")
    )

    # Extract group config.
    # Assume only one group 'group_0' matters for now.
    group_config = hf_quant_config["config_groups"]["group_0"]
    input_act_config = group_config["input_activations"]
    weight_config = group_config["weights"]

    # Parse input scaling spec.
    input_origin = (
        Float8ScaleOrigin.DYNAMIC
        if input_act_config["dynamic"]
        else Float8ScaleOrigin.STATIC
    )
    input_strategy_str = input_act_config["strategy"]
    if input_strategy_str == "tensor":
        input_granularity = Float8ScaleGranularity.TENSOR
    elif input_strategy_str == "channel":
        input_granularity = Float8ScaleGranularity.ROWWISE
    elif input_strategy_str == "token":
        input_granularity = Float8ScaleGranularity.COLWISE
    else:
        raise ValueError(
            f"unsupported FP8 input activation strategy: {input_strategy_str}"
        )

    input_scale_name = "layers.0.mlp.down_proj.input_scale"
    has_input_scale = input_scale_name in state_dict
    input_spec = Float8InputScaleSpec(
        granularity=input_granularity,
        origin=input_origin,
        # Set reasonable defaults if the static input scale isn't present.
        dtype=state_dict[input_scale_name].dtype if has_input_scale else dtype,
        activation_scale_ub=None,
    )

    # Parse weight spec.
    weight_strategy_str = weight_config["strategy"]
    if weight_strategy_str == "tensor":
        weight_granularity = Float8ScaleGranularity.TENSOR
    elif weight_strategy_str == "channel":
        weight_granularity = Float8ScaleGranularity.ROWWISE
    elif weight_strategy_str == "token":
        weight_granularity = Float8ScaleGranularity.COLWISE
    else:
        raise ValueError(
            f"unsupported FP8 weight strategy: {weight_strategy_str}"
        )

    # Validate weight config, which shouldn't dynamically quantize.
    if weight_config["dynamic"]:
        # This method uses static weight scaling according to the examples provided.
        raise ValueError(
            "dynamic weight scaling is not supported for compressed-tensors FP8 method"
        )

    weight_scale = state_dict["layers.0.mlp.down_proj.weight_scale"]
    weight_spec = Float8WeightScaleSpec(
        granularity=weight_granularity, dtype=weight_scale.dtype
    )

    # Determine which layers have MLP and QKV in float8.
    # Modules listed in `ignore` are not converted to float8.
    ignore_modules = set(hf_quant_config.get("ignore", []))
    mlp_in_float8, attn_qkv_in_float8, embedding_output_dtype = (
        _quantized_layers_and_embedding_dtype(
            huggingface_config, ignore_modules, state_dict
        )
    )

    return Float8Config(
        input_scale=input_spec,
        weight_scale=weight_spec,
        mlp_in_float8=mlp_in_float8,
        attn_qkv_in_float8=attn_qkv_in_float8,
        embedding_output_dtype=embedding_output_dtype,
        quant_method="compressed-tensors",
    )


def _weight_scale_dtype(state_dict: Mapping[str, WeightData]) -> DType:
    """Determines the weight scale dtype from the state dict.

    Verifies the expected weight scale quantization along the way:
    - row-wise,
    - uniform weight scale dtype.
    """
    weight_scale_dtype: DType | None = None
    for weight_name, weight in state_dict.items():
        if "weight_scale" not in weight_name:
            continue

        if (
            (len(weight.shape) != 2)
            or (weight.shape[1] != 1)
            or (weight_scale_dtype and (weight.dtype != weight_scale_dtype))
        ):
            raise ValueError(
                "only row-wise weight quantization with uniform weight scale "
                "dtype is supported for FBGEMM FP8"
            )

        weight_scale_dtype = weight.dtype
    if not weight_scale_dtype:
        raise ValueError(
            "could not find weight scale dtype for FBGEMM FP8 quantized weights"
        )

    return weight_scale_dtype


def _parse_fbgemm_float8_config(
    huggingface_config: AutoConfig,
    state_dict: Mapping[str, WeightData],
    dtype: DType,
) -> Float8Config:
    """Parses a Float8Config in the FBGEMM FP8 format."""
    if dtype != DType.float8_e4m3fn:
        raise TypeError(
            "`_parse_fbgemm_float8_config` only supports float8 dtype"
        )

    hf_quant_config = getattr(huggingface_config, "quantization_config", None)
    assert (
        hf_quant_config and hf_quant_config.get("quant_method") == "fbgemm_fp8"
    )

    quant_method = hf_quant_config.get("quant_method")
    # Get the original Hugging Face module names.
    modules_to_not_convert_hf = set(
        hf_quant_config.get("modules_to_not_convert", [])
    )
    activation_scale_ub = hf_quant_config.get("activation_scale_ub")

    # For fbgemm_fp8, assume input is dynamic and column-wise.
    input_spec = Float8InputScaleSpec(
        granularity=Float8ScaleGranularity.COLWISE,
        origin=Float8ScaleOrigin.DYNAMIC,
        dtype=dtype,
        activation_scale_ub=activation_scale_ub,
    )

    # For fbgemm_fp8, weight is static, row-wise.
    weight_spec = Float8WeightScaleSpec(
        granularity=Float8ScaleGranularity.ROWWISE,
        dtype=_weight_scale_dtype(state_dict),
    )

    # Determine which layers have MLP and QKV in float8.
    # Modules listed in `modules_to_not_convert` are not converted to float8.
    mlp_in_float8, attn_qkv_in_float8, embedding_output_dtype = (
        _quantized_layers_and_embedding_dtype(
            huggingface_config, modules_to_not_convert_hf, state_dict
        )
    )

    return Float8Config(
        input_scale=input_spec,
        weight_scale=weight_spec,
        mlp_in_float8=mlp_in_float8,
        attn_qkv_in_float8=attn_qkv_in_float8,
        embedding_output_dtype=embedding_output_dtype,
        quant_method=quant_method,
    )


def parse_float8_config(
    huggingface_config: AutoConfig,
    state_dict: Mapping[str, WeightData],
    dtype: DType,
) -> Float8Config | None:
    """Parses Float8Config from HuggingFace config by dispatching to
    format-specific parsers.
    """
    if dtype != DType.float8_e4m3fn:
        return None

    hf_quant_config = getattr(huggingface_config, "quantization_config", None)
    if not hf_quant_config:
        raise ValueError(
            "expected a `quantization_config` field in Hugging Face config when "
            "the dtype is float8"
        )

    quant_method = hf_quant_config.get("quant_method")

    if quant_method == "compressed-tensors":
        return _parse_compressed_tensors_float8_config(
            huggingface_config, state_dict, dtype
        )
    elif quant_method == "fbgemm_fp8":
        return _parse_fbgemm_float8_config(
            huggingface_config, state_dict, dtype
        )

    raise ValueError(
        "FP8 dtype specified, but an unsupported or incompatible 'quantization_config' "
        f"was found. Quant method: '{quant_method}'. "
        "Supported methods are 'compressed-tensors' and 'fbgemm_fp8'."
    )


@dataclass
class Llama3ConfigBase(MAXModelConfigBase):
    """Base configuration for Llama3 models."""

    # Required fields
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
    model_quantization_encoding: QuantizationEncoding | None
    quantization_config: QuantizationConfig | None
    kv_params: KVCacheParams
    return_logits: ReturnLogits
    norm_method: Literal["rms_norm"] | Literal["layer_norm"]
    norm_dtype: DType | None
    attention_bias: bool
    rms_norm_eps: float | None
    tie_word_embeddings: bool
    stacked_mlp: bool
    stacked_qkv: bool
    logits_postprocessor: Callable[[TensorValue], TensorValue] | None
    attention_multiplier: float
    embedding_multiplier: float
    residual_multiplier: float
    devices: list[DeviceRef]
    clip_qkv: float | None
    float8_config: Float8Config | None
    longrope_scaling_params: LongRoPEScalingParams | None = None

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
        # Base attention multiplier
        base_multiplier = getattr(
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

        return base_multiplier

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

        # Parse the float8 config from compressed-tensors or FBGEMM.
        float8_config = parse_float8_config(
            huggingface_config, state_dict, dtype
        )

        # Determine norm_dtype.
        # Note: due to automatic weight dtype casting, norm dtype is not always
        # correct. To avoid any issue, only set norm_dtype for float8 models
        # for now.
        norm_dtype = None
        if "layers.0.input_layernorm.weight" in state_dict:
            norm_dtype = state_dict["layers.0.input_layernorm.weight"].dtype

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
        rope_scaling_params: Llama3RopeScalingParams | None = None
        longrope_scaling_params: LongRoPEScalingParams | None = None
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
            elif rope_type == "longrope" or rope_type_alt == "longrope":
                longrope_scaling_params = LongRoPEScalingParams(
                    short_factor=rope_scaling["short_factor"],
                    long_factor=rope_scaling["long_factor"],
                    original_max_position=huggingface_config.original_max_position_embeddings,
                    max_position_embeddings=huggingface_config.max_position_embeddings,
                )
                rope_scaling_params = None

        # Calculate base attention multiplier
        base_attention_multiplier = Llama3Config.calculate_attention_multiplier(
            huggingface_config, n_devices, kv_cache_config, cache_dtype
        )

        # Apply LongRoPE attention scaling if needed
        attention_multiplier = base_attention_multiplier
        if longrope_scaling_params is not None:
            # Create temporary RoPE embedding to get proper attention scale
            rope_embedding = create_rope_embedding(
                hidden_size=huggingface_config.hidden_size,
                num_attention_heads=huggingface_config.num_attention_heads,
                rope_theta=huggingface_config.rope_theta,
                max_seq_len=Llama3Config.calculate_max_seq_len(
                    pipeline_config, huggingface_config=huggingface_config
                ),
                interleaved_rope_weights=interleaved_rope_weights,
                rope_scaling_params=rope_scaling_params,
                longrope_scaling_params=longrope_scaling_params,
                device=DeviceRef.CPU(),  # temporary device, not used for scale computation
            )
            attention_multiplier = rope_embedding.compute_scale()

        return Llama3Config(
            hidden_size=huggingface_config.hidden_size,
            num_attention_heads=huggingface_config.num_attention_heads,
            num_key_value_heads=huggingface_config.num_key_value_heads,
            num_hidden_layers=huggingface_config.num_hidden_layers,
            rope_theta=huggingface_config.rope_theta,
            rope_scaling_params=rope_scaling_params,
            longrope_scaling_params=longrope_scaling_params,
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
            attention_multiplier=attention_multiplier,
            embedding_multiplier=embedding_multiplier,
            residual_multiplier=residual_multiplier,
            devices=device_refs,
            clip_qkv=getattr(huggingface_config, "clip_qkv", None),
            float8_config=float8_config,
            use_subgraphs=pipeline_config.model_config.use_subgraphs,
        )
