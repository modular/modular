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

"""Float8 configuration parsing utilities for models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from max.dtype import DType
from max.graph.weights import WeightData
from transformers import AutoConfig


class Float8ScaleGranularity(Enum):
    """Specifies the granularity of the quantization scale factor.

    Determines whether a scale factor applies per-tensor, per-row (often for
    weights), per-column, or per-block within a tensor.
    """

    TENSOR = "tensor"
    """Per-tensor scaling."""

    ROWWISE = "rowwise"
    """Per-row scaling."""

    COLWISE = "colwise"
    """Per-column scaling."""

    BLOCK = "block"
    """Per-block scaling."""

    def __str__(self):
        return self.value


class Float8ScaleOrigin(Enum):
    """Specifies whether the quantization scale is determined statically or dynamically."""

    STATIC = "static"
    """Scales are pre-computed and loaded with the model weights."""

    DYNAMIC = "dynamic"
    """Scales are computed at runtime based on the input data."""


@dataclass
class Float8WeightScaleSpec:
    """Specifies how weights are scaled for float8 quantization."""

    granularity: Float8ScaleGranularity
    """The :obj:`Float8ScaleGranularity` of the weight scale factor application."""

    dtype: DType
    """The :obj:`DType` of the weight scale factor(s)."""

    block_size: tuple[int, int] | None = None
    """The :obj:`tuple[int, int]` of the block size for block-wise scaling."""

    def __post_init__(self):
        if self.granularity == Float8ScaleGranularity.BLOCK:
            if self.block_size is None:
                raise ValueError(
                    "block_size must be specified for block-wise scaling"
                )
            if len(self.block_size) != 2:
                raise ValueError("block_size must be a tuple of two integers")

    @property
    def is_tensor(self) -> bool:
        """Whether the weight scale granularity is per-tensor."""
        return self.granularity == Float8ScaleGranularity.TENSOR

    @property
    def is_rowwise(self) -> bool:
        """Whether the weight scale granularity is row-wise."""
        return self.granularity == Float8ScaleGranularity.ROWWISE

    @property
    def is_colwise(self) -> bool:
        """Whether the weight scale granularity is column-wise."""
        return self.granularity == Float8ScaleGranularity.COLWISE

    @property
    def is_block(self) -> bool:
        """Whether the weight scale granularity is block-wise."""
        return self.granularity == Float8ScaleGranularity.BLOCK


@dataclass
class Float8InputScaleSpec:
    """Specifies how input activations are scaled for float8 quantization."""

    granularity: Float8ScaleGranularity
    """The :obj:`Float8ScaleGranularity` of the input scale factor application."""

    origin: Float8ScaleOrigin
    """The :obj:`Float8ScaleOrigin` (static or dynamic) of the input scale factor."""

    dtype: DType
    """The :obj:`DType` of the input scale factor(s)."""

    activation_scale_ub: float | None = None
    """An optional upper bound for dynamic activation scaling."""

    block_size: tuple[int, int] | None = None
    """The :obj:`tuple[int, int]` of the block size for block-wise scaling."""

    def __post_init__(self):
        if self.granularity == Float8ScaleGranularity.BLOCK:
            if self.block_size is None:
                raise ValueError(
                    "block_size must be specified for block-wise scaling"
                )
            if len(self.block_size) != 2:
                raise ValueError("block_size must be a tuple of two integers")

    @property
    def is_tensor(self) -> bool:
        """Whether the input scale granularity is per-tensor."""
        return self.granularity == Float8ScaleGranularity.TENSOR

    @property
    def is_rowwise(self) -> bool:
        """Whether the input scale granularity is row-wise."""
        return self.granularity == Float8ScaleGranularity.ROWWISE

    @property
    def is_colwise(self) -> bool:
        """Whether the input scale granularity is column-wise."""
        return self.granularity == Float8ScaleGranularity.COLWISE

    @property
    def is_block(self) -> bool:
        """Whether the input scale granularity is block-wise."""
        return self.granularity == Float8ScaleGranularity.BLOCK


@dataclass
class Float8Config:
    """Configures float8 quantization settings for a layer or model section."""

    input_scale: Float8InputScaleSpec
    """:obj:`Float8InputScaleSpec` for input activation scaling."""

    weight_scale: Float8WeightScaleSpec
    """:obj:`Float8WeightScaleSpec` for weight scaling."""

    mlp_in_float8: set[int]
    """Set of layer indices with MLPs in float8.

    MLPs are considered to be either "all quantized" or all not quantized per
    layer.
    So either all of gate proj, down proj, and up proj are float8, or all bfloat16.
    """

    attn_qkv_in_float8: set[int]
    """Set of layer indices with attention QKV projections in float8.

    QKV projections are considered to be either "all quantized" or all not
    quantized per layer.
    So either all of {q,k,v,o}_proj are float8, or all bfloat16.
    """

    embedding_output_dtype: DType | None = None
    """The :obj:`DType` of the output from the embedding layer."""

    quant_method: str | None = None
    """The quantization method used (e.g., "fbgemm_fp8")."""

    @property
    def is_static(self) -> bool:
        """Returns ``True`` if this input scale is static."""
        return self.input_scale.origin == Float8ScaleOrigin.STATIC

    @property
    def is_dynamic(self) -> bool:
        """Returns ``True`` if this input scale is dynamic."""
        return self.input_scale.origin == Float8ScaleOrigin.DYNAMIC


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

    return Float8Config(
        input_scale=input_spec,
        weight_scale=weight_spec,
        mlp_in_float8=set(),
        attn_qkv_in_float8=set(),
        embedding_output_dtype=DType.bfloat16,
        quant_method=quant_method,
    )


def parse_float8_config(
    huggingface_config: AutoConfig,
    state_dict: Mapping[str, WeightData],
    dtype: DType,
    state_dict_name_prefix: str = "",
    ignored_modules_prefix: str = "model.",
) -> Float8Config | None:
    """Parses Float8Config from HuggingFace config by dispatching to format-specific parsers.

    Dispatches to the appropriate format-specific parser based on the
    quantization method in the HuggingFace config.
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
