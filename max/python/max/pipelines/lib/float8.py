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

"""Float8 configuration parsing utilities for Hugging Face models."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from typing import Any

import huggingface_hub
from max.dtype import DType
from max.graph.weights import WeightData
from max.nn.float8_config import (
    Float8Config,
    Float8InputScaleSpec,
    Float8ScaleGranularity,
    Float8ScaleOrigin,
    Float8WeightScaleSpec,
)
from transformers import AutoConfig


def _quantized_layers_and_embedding_dtype(
    huggingface_config: AutoConfig,
    ignored_modules: set[str],
    state_dict: Mapping[str, WeightData],
    state_dict_name_prefix: str = "",
    ignored_modules_prefix: str = "model.",
) -> tuple[set[int], set[int], DType | None]:
    """Helper to determine quantized MLP/Attention layers and embedding output dtype.

    # TODO: For llama3, the layer name re-mapping is not applied to the `ignore`
    # list in quantization config, hence the two prefixes are needed here.
    """
    # Handle multimodal configs where num_hidden_layers is in text_config
    if hasattr(huggingface_config, "text_config"):
        num_hidden_layers = huggingface_config.text_config.num_hidden_layers
    else:
        num_hidden_layers = huggingface_config.num_hidden_layers
    mlp_in_float8: set[int] = set()
    attn_qkv_in_float8: set[int] = set()

    for i in range(num_hidden_layers):
        # Check MLP components (gate_proj, up_proj, down_proj).
        not_converted_mlp_modules = [
            f"{ignored_modules_prefix}layers.{i}.mlp.{proj}" in ignored_modules
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
            f"{ignored_modules_prefix}layers.{i}.self_attn.{proj}"
            in ignored_modules
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
    if f"{state_dict_name_prefix}embed_tokens.weight" in state_dict:
        # Check embed_tokens first since it's the actual embedding layer dtype
        embedding_output_dtype = state_dict[
            f"{state_dict_name_prefix}embed_tokens.weight"
        ].dtype
    elif f"{state_dict_name_prefix}lm_head.weight" in state_dict:
        # Fall back to lm_head dtype (for tied embeddings or when embed_tokens is missing)
        embedding_output_dtype = state_dict[
            f"{state_dict_name_prefix}lm_head.weight"
        ].dtype
    elif f"{state_dict_name_prefix}lm_head" in ignored_modules:
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
    state_dict_name_prefix: str = "",
    ignored_modules_prefix: str = "model.",
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

    input_scale_name = (
        f"{state_dict_name_prefix}layers.0.mlp.down_proj.input_scale"
    )
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

    weight_scale = state_dict[
        f"{state_dict_name_prefix}layers.0.mlp.down_proj.weight_scale"
    ]
    weight_spec = Float8WeightScaleSpec(
        granularity=weight_granularity, dtype=weight_scale.dtype
    )

    # Determine which layers have MLP and QKV in float8.
    # Modules listed in `ignore` are not converted to float8.
    ignore_modules = set(hf_quant_config.get("ignore", []))
    mlp_in_float8, attn_qkv_in_float8, embedding_output_dtype = (
        _quantized_layers_and_embedding_dtype(
            huggingface_config,
            ignore_modules,
            state_dict,
            state_dict_name_prefix=state_dict_name_prefix,
            ignored_modules_prefix=ignored_modules_prefix,
        )
    )

    bias_dtype = _bias_dtype(state_dict)

    return Float8Config(
        input_scale=input_spec,
        weight_scale=weight_spec,
        mlp_in_float8=mlp_in_float8,
        attn_qkv_in_float8=attn_qkv_in_float8,
        embedding_output_dtype=embedding_output_dtype,
        bias_dtype=bias_dtype,
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


def _bias_dtype(state_dict: Mapping[str, WeightData]) -> DType | None:
    """Determines the bias dtype from the state dict.

    Looks for any weight ending with `.bias` and returns its dtype.
    Assumes all bias weights have the same dtype.

    Args:
        state_dict: The state dictionary containing weights.

    Returns:
        The dtype of bias weights if found, None otherwise.
    """
    bias_dtype: DType | None = None
    for weight_name, weight in state_dict.items():
        if weight_name.endswith(".bias"):
            if bias_dtype is None:
                bias_dtype = weight.dtype
            elif weight.dtype != bias_dtype:
                raise ValueError(
                    f"Inconsistent bias dtypes found: {bias_dtype} vs {weight.dtype} "
                    f"in {weight_name}"
                )
    return bias_dtype


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

    bias_dtype = _bias_dtype(state_dict)

    return Float8Config(
        input_scale=input_spec,
        weight_scale=weight_spec,
        mlp_in_float8=mlp_in_float8,
        attn_qkv_in_float8=attn_qkv_in_float8,
        embedding_output_dtype=embedding_output_dtype,
        bias_dtype=bias_dtype,
        quant_method=quant_method,
    )


def _parse_fp8_float8_config(
    huggingface_config: AutoConfig,
    state_dict: Mapping[str, WeightData],
    dtype: DType,
) -> Float8Config:
    """Parses a Float8Config when quant_method="fp8"."""
    if dtype != DType.float8_e4m3fn:
        raise TypeError("`_parse_fp8_float8_config` only supports float8 dtype")

    hf_quant_config = getattr(huggingface_config, "quantization_config", None)

    # The quant method is checked by the caller.
    assert hf_quant_config and hf_quant_config.get("quant_method") == "fp8"

    if "activation_scheme" not in hf_quant_config:
        raise ValueError("activation_scheme must be specified")
    if hf_quant_config["activation_scheme"] != "dynamic":
        raise ValueError("activation_scheme must be dynamic")
    if "weight_block_size" not in hf_quant_config:
        raise ValueError("weight_block_size must be specified")
    if hf_quant_config["weight_block_size"] != [128, 128]:
        raise ValueError("weight_block_size must be [128, 128]")

    quant_method = hf_quant_config.get("quant_method")

    weight_scale_dtype: DType | None = None
    for weight_name, weight in state_dict.items():
        if "weight_scale" not in weight_name:
            continue
        weight_scale_dtype = weight.dtype
    if not weight_scale_dtype:
        raise ValueError(
            "could not find weight scale dtype for FP8 quantized weights"
        )

    input_spec = Float8InputScaleSpec(
        granularity=Float8ScaleGranularity.BLOCK,
        origin=Float8ScaleOrigin.DYNAMIC,
        dtype=weight_scale_dtype,
        block_size=(1, 128),
    )
    weight_spec = Float8WeightScaleSpec(
        granularity=Float8ScaleGranularity.BLOCK,
        dtype=weight_scale_dtype,
        block_size=(128, 128),
    )

    bias_dtype = _bias_dtype(state_dict)

    return Float8Config(
        input_scale=input_spec,
        weight_scale=weight_spec,
        mlp_in_float8=set(),
        attn_qkv_in_float8=set(),
        embedding_output_dtype=DType.bfloat16,
        bias_dtype=bias_dtype,
        quant_method=quant_method,
    )


def _parse_float8_config(
    huggingface_config: AutoConfig,
    state_dict: Mapping[str, WeightData],
    dtype: DType,
    state_dict_name_prefix: str = "",
    ignored_modules_prefix: str = "model.",
) -> Float8Config | None:
    """Parses Float8Config from Hugging Face config (if exists) by dispatching to format-specific parsers.

    Dispatches to the appropriate format-specific parser based on the
    quantization method in the Hugging Face config. Returns None if the dtype is not float8_e4m3fn.
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
            huggingface_config,
            state_dict,
            dtype,
            state_dict_name_prefix=state_dict_name_prefix,
            ignored_modules_prefix=ignored_modules_prefix,
        )
    elif quant_method == "fbgemm_fp8":
        return _parse_fbgemm_float8_config(
            huggingface_config, state_dict, dtype
        )
    elif quant_method == "fp8":  # DeepSeekV3
        return _parse_fp8_float8_config(huggingface_config, state_dict, dtype)

    raise ValueError(
        "FP8 dtype specified, but an unsupported or incompatible 'quantization_config' "
        f"was found. Quant method: '{quant_method}'. "
        "Supported methods are 'compressed-tensors' and 'fbgemm_fp8'."
    )


logger = logging.getLogger("max.pipelines")


_FP4_DTYPES = (DType.uint8, DType.float4_e2m1fn)


def _load_standalone_quant_config(
    huggingface_config: AutoConfig,
) -> dict[str, Any] | None:
    """Try to load quantization config from a standalone hf_quant_config.json file.

    Some models (e.g., nvidia/Llama-3.1-405B-Instruct-NVFP4) store their quantization
    config in a separate hf_quant_config.json file rather than in the main config.json.

    The standalone file has a different structure:
    {
        "producer": {"name": "modelopt", "version": "..."},
        "quantization": {"quant_algo": "NVFP4", ...}
    }

    This function maps it to the standard format:
    {"quant_method": "modelopt", "quant_algo": "NVFP4", ...}

    Returns:
        dict with quant_method and quant_algo if found, None otherwise.
    """
    model_path = getattr(huggingface_config, "_name_or_path", None)
    if not model_path:
        return None

    quant_config_filename = "hf_quant_config.json"

    try:
        # Try local path first
        if os.path.isdir(model_path):
            local_path = os.path.join(model_path, quant_config_filename)
            if os.path.exists(local_path):
                with open(local_path) as f:
                    standalone_config = json.load(f)
            else:
                return None
        else:
            # Try to download from HuggingFace
            try:
                downloaded_path = huggingface_hub.hf_hub_download(
                    repo_id=model_path,
                    filename=quant_config_filename,
                    local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                )
                with open(downloaded_path) as f:
                    standalone_config = json.load(f)
            except Exception:
                # File doesn't exist or can't be downloaded
                return None

        # Map the standalone format to the standard format
        producer = standalone_config.get("producer", {})
        quantization = standalone_config.get("quantization", {})

        quant_method = producer.get("name")
        quant_algo = quantization.get("quant_algo")

        if quant_method and quant_algo:
            return {"quant_method": quant_method, "quant_algo": quant_algo}
        return None

    except Exception as e:
        logger.debug(f"Failed to load standalone quant config: {e}")
        return None


def _resolve_quant_config(
    huggingface_config: AutoConfig,
    state_dict: Mapping[str, WeightData],
) -> dict[str, Any] | None:
    if hf_quant_config := getattr(
        huggingface_config, "quantization_config", None
    ):
        return hf_quant_config

    standalone_config = _load_standalone_quant_config(huggingface_config)
    if standalone_config:
        return standalone_config

    if any("weight_scale_2" in name for name in state_dict):
        return {"quant_method": "modelopt", "quant_algo": "NVFP4"}

    return None


def _parse_modelopt_float4_config(
    huggingface_config: AutoConfig,
    state_dict: Mapping[str, WeightData],
    dtype: DType,
    *,
    quant_method_override: str | None = None,
    quant_algo_override: str | None = None,
) -> Float8Config | None:
    hf_quant_config = getattr(huggingface_config, "quantization_config", None)
    quant_method = quant_method_override
    quant_algo = quant_algo_override
    if hf_quant_config:
        quant_method = hf_quant_config.get("quant_method", quant_method)
        quant_algo = hf_quant_config.get("quant_algo", quant_algo)
    if not quant_method or not quant_algo:
        return None

    input_spec = Float8InputScaleSpec(
        granularity=Float8ScaleGranularity.BLOCK,
        origin=Float8ScaleOrigin.STATIC,
        dtype=DType.float32,
        block_size=(1, 16),
    )
    weight_spec = Float8WeightScaleSpec(
        granularity=Float8ScaleGranularity.BLOCK,
        dtype=DType.float8_e4m3fn,
        block_size=(1, 16 // 2),
    )

    bias_dtype = _bias_dtype(state_dict)

    # All layers use float4 in modelopt NVFP4 checkpoints.
    all_layers = set(range(huggingface_config.num_hidden_layers))

    return Float8Config(
        input_scale=input_spec,
        weight_scale=weight_spec,
        mlp_in_float8=all_layers,
        attn_qkv_in_float8=all_layers,
        embedding_output_dtype=DType.bfloat16,
        bias_dtype=bias_dtype,
        quant_method=quant_method,
        quant_algo=quant_algo,
    )


def _parse_float4_config(
    huggingface_config: AutoConfig,
    state_dict: Mapping[str, WeightData],
    dtype: DType,
    state_dict_name_prefix: str = "",
    ignored_modules_prefix: str = "model.",
) -> Float8Config | None:
    # Accept both uint8 (fp4-e2m1fnX2 format) and float4_e2m1fn
    if dtype not in _FP4_DTYPES:
        return None

    quant_config = _resolve_quant_config(huggingface_config, state_dict)
    if not quant_config:
        return None

    quant_method = quant_config.get("quant_method")
    if quant_method == "modelopt":
        return _parse_modelopt_float4_config(
            huggingface_config,
            state_dict,
            dtype,
            quant_method_override=quant_method,
            quant_algo_override=quant_config.get("quant_algo"),
        )

    logger.debug(
        "Skipping FP4 parsing for unsupported quant method: %s",
        quant_method,
    )
    return None


def parse_float8_config(  # TODO: rename to generic
    huggingface_config: AutoConfig,
    state_dict: Mapping[str, WeightData],
    dtype: DType,
    state_dict_name_prefix: str = "",
    ignored_modules_prefix: str = "model.",
) -> Float8Config | None:
    """Parses Float8 or Float4 config from HuggingFace config and state dict.

    Args:
        huggingface_config: HuggingFace model configuration.
        state_dict: Weight state dict to inspect for scales.
        dtype: Target dtype (e.g. float8_e4m3fn or packed fp4).
        state_dict_name_prefix: Optional prefix for state dict keys.
        ignored_modules_prefix: Prefix of modules to ignore when parsing.

    Returns:
        Float8Config or Float4 config if supported, otherwise None.
    """
    # uint8 is packed fp4 (float4_e2m1fnx2) in NVFP4 checkpoints.
    if dtype in _FP4_DTYPES:
        return _parse_float4_config(
            huggingface_config,
            state_dict,
            dtype,
            state_dict_name_prefix,
            ignored_modules_prefix,
        )
    elif dtype == DType.float8_e4m3fn:
        return _parse_float8_config(
            huggingface_config,
            state_dict,
            dtype,
            state_dict_name_prefix,
            ignored_modules_prefix,
        )
    else:
        return None
