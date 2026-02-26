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
"""Tests for parse_float8_config in Llama3 model_config."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from max.dtype import DType
from max.graph import Shape
from max.graph.weights import WeightData
from max.nn.float8_config import (
    Float8ScaleGranularity,
    Float8ScaleOrigin,
)
from max.pipelines.lib.float8 import parse_float8_config
from transformers import AutoConfig

# Define a base path for test data.
# Note: Bazel runs tests from the workspace root, but puts datafiles
# in a path relative to the test file's location.
# tests/pipelines/test_parse_float8_config.py
# tests/pipelines/testdata/llama_3_1_405b_fp8.json
TEST_DATA_PATH = Path(__file__).parent / "testdata"


@pytest.fixture
def hf_config_instruct_fbgemm() -> AutoConfig:
    """Loads the Llama-3.1-405B-Instruct-FP8 HuggingFace config."""
    config_path = TEST_DATA_PATH / "llama_3_1_405b_instruct_fp8.json"
    return AutoConfig.from_pretrained(str(config_path), trust_remote_code=True)


@pytest.fixture
def hf_config_base_fbgemm() -> AutoConfig:
    """Loads the Llama-3.1-405B-FP8 (base) HuggingFace config."""
    config_path = TEST_DATA_PATH / "llama_3_1_405b_fp8.json"
    return AutoConfig.from_pretrained(str(config_path), trust_remote_code=True)


@pytest.fixture
def state_dict_with_lm_head_and_fbgemm_scales() -> dict[str, WeightData]:
    """Mock state_dict with lm_head.weight and a valid FBGEMM weight_scale."""
    return {
        "layers.0.mlp.down_proj.weight_scale": WeightData(
            name="layers.0.mlp.down_proj.weight_scale",
            shape=Shape((1, 1)),
            dtype=DType.float8_e4m3fn,
            data=torch.zeros((1, 1), dtype=DType.float8_e4m3fn.to_torch()),
        ),
        "lm_head.weight": WeightData(
            name="lm_head.weight",
            shape=Shape((1, 1)),
            dtype=DType.bfloat16,
            data=torch.zeros((1, 1), dtype=DType.bfloat16.to_torch()),
        ),
    }


@pytest.fixture
def state_dict_fbgemm_scales_only() -> dict[str, WeightData]:
    """Mock state_dict with only a valid FBGEMM weight_scale (no lm_head)."""
    return {
        "layers.0.mlp.down_proj.weight_scale": WeightData(
            name="layers.0.mlp.down_proj.weight_scale",
            shape=Shape((1, 1)),
            dtype=DType.float8_e4m3fn,
            data=torch.zeros((1, 1), dtype=DType.float8_e4m3fn.to_torch()),
        ),
    }


def test_parse_fbgemm_instruct_config(
    hf_config_instruct_fbgemm: AutoConfig,
    state_dict_with_lm_head_and_fbgemm_scales: dict[str, WeightData],
) -> None:
    """Tests parsing Llama-3.1-405B-Instruct-FP8 (fbgemm).
    - "lm_head" is NOT in modules_to_not_convert.
    - Layer 0 MLP and Attn ARE in modules_to_not_convert.
    """
    hf_config = hf_config_instruct_fbgemm
    state_dict = state_dict_with_lm_head_and_fbgemm_scales
    dtype = DType.float8_e4m3fn

    float8_config = parse_float8_config(hf_config, state_dict, dtype)

    assert float8_config is not None
    assert float8_config.quant_method == "fbgemm_fp8"
    assert (
        float8_config.input_scale.granularity == Float8ScaleGranularity.COLWISE
    )
    assert float8_config.input_scale.origin == Float8ScaleOrigin.DYNAMIC
    assert float8_config.input_scale.dtype == dtype
    assert (
        float8_config.input_scale.activation_scale_ub
        == hf_config.quantization_config["activation_scale_ub"]
    )
    assert (
        float8_config.weight_scale.granularity == Float8ScaleGranularity.ROWWISE
    )

    # Check dtypes originating from the state dict.
    assert float8_config.weight_scale.dtype == DType.float8_e4m3fn
    assert float8_config.embedding_output_dtype == DType.bfloat16

    # Layer 0 MLP and Attn are in modules_to_not_convert
    expected_mlp_in_float8 = set(range(1, hf_config.num_hidden_layers))
    expected_attn_qkv_in_float8 = set(range(1, hf_config.num_hidden_layers))
    assert float8_config.mlp_in_float8 == expected_mlp_in_float8
    assert float8_config.attn_qkv_in_float8 == expected_attn_qkv_in_float8


def test_parse_fbgemm_base_config(
    hf_config_base_fbgemm: AutoConfig,
    state_dict_with_lm_head_and_fbgemm_scales: dict[str, WeightData],
) -> None:
    """Tests parsing Llama-3.1-405B-FP8 (base, fbgemm).
    - "lm_head" IS in modules_to_not_convert.
    - Layer 0 MLP and Attn ARE in modules_to_not_convert.
    - embedding_output_dtype still comes from state_dict as lm_head.weight is present.
    """
    hf_config = hf_config_base_fbgemm
    state_dict = state_dict_with_lm_head_and_fbgemm_scales
    dtype = DType.float8_e4m3fn

    float8_config = parse_float8_config(hf_config, state_dict, dtype)

    assert float8_config is not None
    assert float8_config.quant_method == "fbgemm_fp8"

    # Check output dtype from the state dict.
    assert float8_config.embedding_output_dtype == DType.bfloat16

    expected_mlp_in_float8 = set(range(1, hf_config.num_hidden_layers))
    expected_attn_qkv_in_float8 = set(range(1, hf_config.num_hidden_layers))
    assert float8_config.mlp_in_float8 == expected_mlp_in_float8
    assert float8_config.attn_qkv_in_float8 == expected_attn_qkv_in_float8


# --- Error Cases --- #


def test_error_wrong_dtype(hf_config_instruct_fbgemm: AutoConfig) -> None:
    """Tests parse_float8_config returns None if dtype is not float8_e4m3fn."""
    assert (
        parse_float8_config(hf_config_instruct_fbgemm, {}, DType.float16)
        is None
    )


def test_error_missing_quantization_config(
    hf_config_instruct_fbgemm: AutoConfig,
) -> None:
    """Tests error if quantization_config is missing."""
    hf_config_no_quant = deepcopy(hf_config_instruct_fbgemm)
    del hf_config_no_quant.quantization_config
    with pytest.raises(
        ValueError, match="expected a `quantization_config` field"
    ):
        parse_float8_config(hf_config_no_quant, {}, DType.float8_e4m3fn)


def test_error_unsupported_quant_method(
    hf_config_instruct_fbgemm: AutoConfig,
    state_dict_with_lm_head_and_fbgemm_scales: dict[str, WeightData],
) -> None:
    """Tests error if quant_method is unsupported."""
    hf_config_bad_method = deepcopy(hf_config_instruct_fbgemm)
    hf_config_bad_method.quantization_config["quant_method"] = (
        "unsupported_method"
    )
    with pytest.raises(ValueError, match="unsupported or incompatible"):
        parse_float8_config(
            hf_config_bad_method,
            state_dict_with_lm_head_and_fbgemm_scales,
            DType.float8_e4m3fn,
        )


def test_error_fbgemm_missing_weight_scale(
    hf_config_instruct_fbgemm: AutoConfig,
) -> None:
    """Tests _weight_scale_dtype error: missing weight_scale for fbgemm."""
    # State dict with lm_head but no weight_scales for fbgemm path
    state_dict_no_scales = {
        "lm_head.weight": WeightData(
            name="lm_head.weight",
            shape=Shape((1, 1)),
            dtype=DType.bfloat16,
            data=torch.zeros((1, 1), dtype=DType.bfloat16.to_torch()),
        )
    }
    with pytest.raises(ValueError, match="could not find weight scale dtype"):
        parse_float8_config(
            hf_config_instruct_fbgemm, state_dict_no_scales, DType.float8_e4m3fn
        )


def test_error_fbgemm_inconsistent_weight_scale_dtype(
    hf_config_instruct_fbgemm: AutoConfig,
) -> None:
    """Tests _weight_scale_dtype error: inconsistent weight_scale dtypes for fbgemm."""
    state_dict_inconsistent_ws_dtype = {
        "layers.0.mlp.down_proj.weight_scale": WeightData(
            name="layers.0.mlp.down_proj.weight_scale",
            shape=Shape((1, 1)),
            dtype=DType.float8_e4m3fn,
            data=torch.zeros((1, 1), dtype=DType.float8_e4m3fn.to_torch()),
        ),
        "layers.1.self_attn.k_proj.weight_scale": WeightData(
            name="layers.1.self_attn.k_proj.weight_scale",
            shape=Shape((1, 1)),
            dtype=DType.float8_e5m2,  # Different float8 dtype
            data=torch.zeros((1, 1), dtype=DType.float8_e5m2.to_torch()),
        ),
        "lm_head.weight": WeightData(
            name="lm_head.weight",
            shape=Shape((1, 1)),
            dtype=DType.bfloat16,
            data=torch.zeros((1, 1), dtype=DType.bfloat16.to_torch()),
        ),
    }
    hf_config = deepcopy(hf_config_instruct_fbgemm)
    hf_config.quantization_config["quant_method"] = "fbgemm_fp8"

    with pytest.raises(
        ValueError, match="uniform weight scale dtype is supported"
    ):
        parse_float8_config(
            hf_config, state_dict_inconsistent_ws_dtype, DType.float8_e4m3fn
        )


def test_error_fbgemm_bad_weight_scale_shape(
    hf_config_instruct_fbgemm: AutoConfig,
) -> None:
    """Tests _weight_scale_dtype error: bad weight_scale shape for fbgemm."""
    state_dict_bad_ws_shape = {
        "layers.0.mlp.down_proj.weight_scale": WeightData(
            name="layers.0.mlp.down_proj.weight_scale",
            shape=Shape((1, 2)),  # Not col-vector (shape[1] != 1)
            dtype=DType.float8_e4m3fn,
            data=torch.zeros((1, 2), dtype=DType.float8_e4m3fn.to_torch()),
        ),
        "lm_head.weight": WeightData(
            name="lm_head.weight",
            shape=Shape((1, 1)),
            dtype=DType.bfloat16,
            data=torch.zeros((1, 1), dtype=DType.bfloat16.to_torch()),
        ),
    }
    hf_config = deepcopy(hf_config_instruct_fbgemm)
    hf_config.quantization_config["quant_method"] = "fbgemm_fp8"

    with pytest.raises(ValueError, match="only row-wise weight quantization"):
        parse_float8_config(
            hf_config, state_dict_bad_ws_shape, DType.float8_e4m3fn
        )


def test_error_partial_mlp_quantization(
    hf_config_instruct_fbgemm: AutoConfig,
    state_dict_with_lm_head_and_fbgemm_scales: dict[str, WeightData],
) -> None:
    """Tests _quantized_layers_and_embedding_dtype error: partial MLP quantization."""
    hf_config_partial_mlp = deepcopy(hf_config_instruct_fbgemm)
    # Keep gate_proj, up_proj, but remove down_proj from not_convert for layer 0
    hf_config_partial_mlp.quantization_config["modules_to_not_convert"] = [
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
        # "model.layers.0.mlp.down_proj", # This one is now considered converted
        "model.layers.0.self_attn.k_proj",  # Keep attn fully not converted
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.v_proj",
    ]
    with pytest.raises(ValueError, match="uniform quantization for MLPs"):
        parse_float8_config(
            hf_config_partial_mlp,
            state_dict_with_lm_head_and_fbgemm_scales,
            DType.float8_e4m3fn,
        )


def test_error_partial_attn_qkv_quantization(
    hf_config_instruct_fbgemm: AutoConfig,
    state_dict_with_lm_head_and_fbgemm_scales: dict[str, WeightData],
) -> None:
    """Tests _quantized_layers_and_embedding_dtype error: partial Attention QKV quantization."""
    hf_config_partial_attn = deepcopy(hf_config_instruct_fbgemm)
    # Keep k_proj, o_proj, v_proj but remove q_proj from not_convert for layer 0
    hf_config_partial_attn.quantization_config["modules_to_not_convert"] = [
        "model.layers.0.mlp.down_proj",  # Keep mlp fully not converted
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.o_proj",
        # "model.layers.0.self_attn.q_proj", # This one is now considered converted
        "model.layers.0.self_attn.v_proj",
    ]
    with pytest.raises(
        ValueError, match="uniform quantization for attention QKV"
    ):
        parse_float8_config(
            hf_config_partial_attn,
            state_dict_with_lm_head_and_fbgemm_scales,
            DType.float8_e4m3fn,
        )


def test_error_lm_head_ignored_no_weights(
    hf_config_base_fbgemm: AutoConfig,  # This config has lm_head in modules_to_not_convert
    state_dict_fbgemm_scales_only: dict[
        str, WeightData
    ],  # No lm_head or embed_tokens
) -> None:
    """Tests error if lm_head is ignored and no embedding weights in state_dict."""
    with pytest.raises(
        ValueError, match="cannot determine original type from checkpoint"
    ):
        parse_float8_config(
            hf_config_base_fbgemm,
            state_dict_fbgemm_scales_only,
            DType.float8_e4m3fn,
        )


@pytest.fixture
def hf_config_compressed_tensors(
    hf_config_instruct_fbgemm: AutoConfig,
) -> AutoConfig:
    """Creates a base compressed-tensors config from an fbgemm config."""
    config = deepcopy(hf_config_instruct_fbgemm)
    config.quantization_config["quant_method"] = "compressed-tensors"
    # Compressed-tensors uses 'ignore' instead of 'modules_to_not_convert'
    if "modules_to_not_convert" in config.quantization_config:
        config.quantization_config["ignore"] = config.quantization_config.pop(
            "modules_to_not_convert"
        )
    config.quantization_config["config_groups"] = {
        "group_0": {
            "input_activations": {"dynamic": True, "strategy": "tensor"},
            "weights": {"dynamic": False, "strategy": "tensor"},
        }
    }
    return config


def test_error_ct_unsupported_input_strategy(
    hf_config_compressed_tensors: AutoConfig,
    state_dict_with_lm_head_and_fbgemm_scales: dict[str, WeightData],
) -> None:
    """Tests compressed-tensors error: unsupported input activation strategy."""
    config = deepcopy(hf_config_compressed_tensors)
    config.quantization_config["config_groups"]["group_0"]["input_activations"][
        "strategy"
    ] = "unsupported_input_strat"
    with pytest.raises(
        ValueError, match="unsupported FP8 input activation strategy"
    ):
        parse_float8_config(
            config,
            state_dict_with_lm_head_and_fbgemm_scales,
            DType.float8_e4m3fn,
        )


def test_error_ct_unsupported_weight_strategy(
    hf_config_compressed_tensors: AutoConfig,
    state_dict_with_lm_head_and_fbgemm_scales: dict[str, WeightData],
) -> None:
    """Tests compressed-tensors error: unsupported weight strategy."""
    config = deepcopy(hf_config_compressed_tensors)
    config.quantization_config["config_groups"]["group_0"]["weights"][
        "strategy"
    ] = "unsupported_weight_strat"
    with pytest.raises(ValueError, match="unsupported FP8 weight strategy"):
        parse_float8_config(
            config,
            state_dict_with_lm_head_and_fbgemm_scales,
            DType.float8_e4m3fn,
        )


def test_error_ct_dynamic_weight_scaling(
    hf_config_compressed_tensors: AutoConfig,
    state_dict_with_lm_head_and_fbgemm_scales: dict[str, WeightData],
) -> None:
    """Tests compressed-tensors error: dynamic weight scaling not supported."""
    config = deepcopy(hf_config_compressed_tensors)
    config.quantization_config["config_groups"]["group_0"]["weights"][
        "dynamic"
    ] = True
    with pytest.raises(
        ValueError,
        match="dynamic weight scaling is not supported for compressed-tensors",
    ):
        parse_float8_config(
            config,
            state_dict_with_lm_head_and_fbgemm_scales,
            DType.float8_e4m3fn,
        )


# --- Gemma3 FP8 Tests --- #


@pytest.fixture
def hf_config_gemma3_compressed_tensors() -> AutoConfig:
    """Loads the Gemma3 FP8 compressed-tensors HuggingFace config."""
    config_path = TEST_DATA_PATH / "gemma-3-4b-it-FP8-dynamic.json"
    huggingface_config = AutoConfig.from_pretrained(
        str(config_path), trust_remote_code=True
    )
    hf_quant_config = getattr(huggingface_config, "quantization_config", None)
    # To the language model section of the config (`text_config`), add a
    # reference to the top level `quantization_config` for compatibility
    # with the base Gemma3Model
    if hf_quant_config:
        huggingface_config.text_config.quantization_config = hf_quant_config
    return huggingface_config


@pytest.fixture
def state_dict_gemma3_with_language_model_prefix() -> dict[str, WeightData]:
    """Mock state_dict with Gemma3's language_model prefix and scales."""
    return {
        "language_model.layers.0.mlp.down_proj.input_scale": WeightData(
            name="language_model.layers.0.mlp.down_proj.input_scale",
            shape=Shape((1, 1)),
            dtype=DType.float32,
            data=torch.zeros((1, 1), dtype=DType.float32.to_torch()),
        ),
        "language_model.layers.0.mlp.down_proj.weight_scale": WeightData(
            name="language_model.layers.0.mlp.down_proj.weight_scale",
            shape=Shape((1, 1)),
            dtype=DType.float32,
            data=torch.zeros((1, 1), dtype=DType.float32.to_torch()),
        ),
        "language_model.embed_tokens.weight": WeightData(
            name="language_model.embed_tokens.weight",
            shape=Shape((1, 1)),
            dtype=DType.bfloat16,
            data=torch.zeros((1, 1), dtype=DType.bfloat16.to_torch()),
        ),
        "language_model.lm_head.weight": WeightData(
            name="language_model.lm_head.weight",
            shape=Shape((1, 1)),
            dtype=DType.float8_e4m3fn,
            data=torch.zeros((1, 1), dtype=DType.float8_e4m3fn.to_torch()),
        ),
    }


def test_parse_gemma3_compressed_tensors(
    hf_config_gemma3_compressed_tensors: AutoConfig,
    state_dict_gemma3_with_language_model_prefix: dict[str, WeightData],
) -> None:
    """Tests parsing Gemma3 FP8 compressed-tensors config with language_model prefix.
    - Uses "language_model." prefix for both state_dict and ignored modules
    - Only vision_tower modules and language_model.lm_head are in ignore list
    - All language model layers should be quantized to float8
    """
    hf_config = hf_config_gemma3_compressed_tensors
    state_dict = state_dict_gemma3_with_language_model_prefix
    dtype = DType.float8_e4m3fn

    # Call with Gemma3's language_model prefix
    float8_config = parse_float8_config(
        hf_config.text_config,
        state_dict,
        dtype,
        state_dict_name_prefix="language_model.",
        ignored_modules_prefix="language_model.",
    )

    assert float8_config is not None
    assert float8_config.quant_method == "compressed-tensors"
    assert (
        float8_config.input_scale.granularity == Float8ScaleGranularity.COLWISE
    )
    assert float8_config.input_scale.origin == Float8ScaleOrigin.DYNAMIC
    assert float8_config.input_scale.dtype == DType.float32

    # Check dtypes from state dict - embed_tokens takes priority
    assert float8_config.embedding_output_dtype == DType.bfloat16

    # All language model layers should be quantized (lm_head is ignored, but layers are not)
    expected_mlp_in_float8 = set(range(hf_config.text_config.num_hidden_layers))
    expected_attn_qkv_in_float8 = set(
        range(hf_config.text_config.num_hidden_layers)
    )
    assert float8_config.mlp_in_float8 == expected_mlp_in_float8
    assert float8_config.attn_qkv_in_float8 == expected_attn_qkv_in_float8


def test_gemma3_layer_prefix_handling(
    hf_config_gemma3_compressed_tensors: AutoConfig,
) -> None:
    """Tests that Gemma3 prefix handling works correctly for both standard and prefixed modules."""
    # Test with just lm_head.weight (no embed_tokens)
    state_dict_lm_head_only = {
        "language_model.layers.0.mlp.down_proj.input_scale": WeightData(
            name="language_model.layers.0.mlp.down_proj.input_scale",
            shape=Shape((1, 1)),
            dtype=DType.float32,
            data=torch.zeros((1, 1), dtype=DType.float32.to_torch()),
        ),
        "language_model.layers.0.mlp.down_proj.weight_scale": WeightData(
            name="language_model.layers.0.mlp.down_proj.weight_scale",
            shape=Shape((1, 1)),
            dtype=DType.float32,
            data=torch.zeros((1, 1), dtype=DType.float32.to_torch()),
        ),
        "language_model.lm_head.weight": WeightData(
            name="language_model.lm_head.weight",
            shape=Shape((1, 1)),
            dtype=DType.float8_e4m3fn,
            data=torch.zeros((1, 1), dtype=DType.float8_e4m3fn.to_torch()),
        ),
    }

    float8_config = parse_float8_config(
        hf_config_gemma3_compressed_tensors.text_config,
        state_dict_lm_head_only,
        DType.float8_e4m3fn,
        state_dict_name_prefix="language_model.",
        ignored_modules_prefix="language_model.",
    )

    assert float8_config is not None
    # Should fall back to lm_head dtype when embed_tokens is missing
    assert float8_config.embedding_output_dtype == DType.float8_e4m3fn


def test_gemma3_vs_llama3_prefix_difference(
    hf_config_gemma3_compressed_tensors: AutoConfig,
) -> None:
    """Tests that Gemma3 and Llama3 handle prefixes differently."""
    # Create state dict with both prefixes for comparison
    state_dict_with_both_prefixes = {
        # Gemma3-style prefixes
        "language_model.layers.0.mlp.down_proj.input_scale": WeightData(
            name="language_model.layers.0.mlp.down_proj.input_scale",
            shape=Shape((1, 1)),
            dtype=DType.float32,
            data=torch.zeros((1, 1), dtype=DType.float32.to_torch()),
        ),
        "language_model.layers.0.mlp.down_proj.weight_scale": WeightData(
            name="language_model.layers.0.mlp.down_proj.weight_scale",
            shape=Shape((1, 1)),
            dtype=DType.float32,
            data=torch.zeros((1, 1), dtype=DType.float32.to_torch()),
        ),
        "language_model.embed_tokens.weight": WeightData(
            name="language_model.embed_tokens.weight",
            shape=Shape((1, 1)),
            dtype=DType.bfloat16,
            data=torch.zeros((1, 1), dtype=DType.bfloat16.to_torch()),
        ),
        # Llama3-style prefixes (should be ignored when using language_model prefix)
        "layers.0.mlp.down_proj.input_scale": WeightData(
            name="layers.0.mlp.down_proj.input_scale",
            shape=Shape((1, 1)),
            dtype=DType.float16,  # Different dtype
            data=torch.zeros((1, 1), dtype=DType.float16.to_torch()),
        ),
        "embed_tokens.weight": WeightData(
            name="embed_tokens.weight",
            shape=Shape((1, 1)),
            dtype=DType.float16,  # Different dtype
            data=torch.zeros((1, 1), dtype=DType.float16.to_torch()),
        ),
    }

    # Parse with Gemma3 prefixes - should use bfloat16 from language_model.embed_tokens.weight
    float8_config = parse_float8_config(
        hf_config_gemma3_compressed_tensors.text_config,
        state_dict_with_both_prefixes,
        DType.float8_e4m3fn,
        state_dict_name_prefix="language_model.",
        ignored_modules_prefix="language_model.",
    )

    assert float8_config is not None
    assert float8_config.embedding_output_dtype == DType.bfloat16


def test_parse_fbgemm_bias_dtype(
    hf_config_instruct_fbgemm: AutoConfig,
    state_dict_with_lm_head_and_fbgemm_scales: dict[str, WeightData],
) -> None:
    """Tests that parse_float8_config correctly extracts bias_dtype from state dict."""
    hf_config = hf_config_instruct_fbgemm
    state_dict = state_dict_with_lm_head_and_fbgemm_scales.copy()

    # Add bias weights to state dict
    state_dict["layers.0.mlp.down_proj.bias"] = WeightData(
        name="layers.0.mlp.down_proj.bias",
        shape=Shape((1,)),
        dtype=DType.float32,
        data=torch.zeros((1,), dtype=DType.float32.to_torch()),
    )
    state_dict["layers.0.self_attn.q_proj.bias"] = WeightData(
        name="layers.0.self_attn.q_proj.bias",
        shape=Shape((1,)),
        dtype=DType.float32,
        data=torch.zeros((1,), dtype=DType.float32.to_torch()),
    )

    dtype = DType.float8_e4m3fn
    float8_config = parse_float8_config(hf_config, state_dict, dtype)

    assert float8_config is not None
    assert float8_config.bias_dtype == DType.float32


def test_parse_fbgemm_bias_dtype_none(
    hf_config_instruct_fbgemm: AutoConfig,
    state_dict_with_lm_head_and_fbgemm_scales: dict[str, WeightData],
) -> None:
    """Tests that parse_float8_config returns None for bias_dtype when no bias weights exist."""
    hf_config = hf_config_instruct_fbgemm
    state_dict = state_dict_with_lm_head_and_fbgemm_scales
    dtype = DType.float8_e4m3fn

    float8_config = parse_float8_config(hf_config, state_dict, dtype)

    assert float8_config is not None
    assert float8_config.bias_dtype is None


def test_parse_fbgemm_bias_dtype_inconsistent(
    hf_config_instruct_fbgemm: AutoConfig,
    state_dict_with_lm_head_and_fbgemm_scales: dict[str, WeightData],
) -> None:
    """Tests that parse_float8_config raises error for inconsistent bias dtypes."""
    hf_config = hf_config_instruct_fbgemm
    state_dict = state_dict_with_lm_head_and_fbgemm_scales.copy()

    # Add bias weights with inconsistent dtypes
    state_dict["layers.0.mlp.down_proj.bias"] = WeightData(
        name="layers.0.mlp.down_proj.bias",
        shape=Shape((1,)),
        dtype=DType.float32,
        data=torch.zeros((1,), dtype=DType.float32.to_torch()),
    )
    state_dict["layers.0.self_attn.q_proj.bias"] = WeightData(
        name="layers.0.self_attn.q_proj.bias",
        shape=Shape((1,)),
        dtype=DType.bfloat16,  # Different dtype
        data=torch.zeros((1,), dtype=DType.bfloat16.to_torch()),
    )

    dtype = DType.float8_e4m3fn
    with pytest.raises(ValueError, match="Inconsistent bias dtypes found"):
        parse_float8_config(hf_config, state_dict, dtype)


def test_parse_compressed_tensors_bias_dtype(
    hf_config_compressed_tensors: AutoConfig,
    state_dict_with_lm_head_and_fbgemm_scales: dict[str, WeightData],
) -> None:
    """Tests that parse_float8_config correctly extracts bias_dtype for compressed-tensors."""
    config = hf_config_compressed_tensors
    state_dict = state_dict_with_lm_head_and_fbgemm_scales.copy()

    # Add input_scale and weight_scale for compressed-tensors
    state_dict["layers.0.mlp.down_proj.input_scale"] = WeightData(
        name="layers.0.mlp.down_proj.input_scale",
        shape=Shape((1, 1)),
        dtype=DType.float32,
        data=torch.zeros((1, 1), dtype=DType.float32.to_torch()),
    )
    state_dict["layers.0.mlp.down_proj.weight_scale"] = WeightData(
        name="layers.0.mlp.down_proj.weight_scale",
        shape=Shape((1, 1)),
        dtype=DType.float32,
        data=torch.zeros((1, 1), dtype=DType.float32.to_torch()),
    )

    # Add bias weights
    state_dict["layers.0.mlp.down_proj.bias"] = WeightData(
        name="layers.0.mlp.down_proj.bias",
        shape=Shape((1,)),
        dtype=DType.bfloat16,
        data=torch.zeros((1,), dtype=DType.bfloat16.to_torch()),
    )

    dtype = DType.float8_e4m3fn
    float8_config = parse_float8_config(config, state_dict, dtype)

    assert float8_config is not None
    assert float8_config.bias_dtype == DType.bfloat16


@pytest.fixture
def hf_config_fp8(hf_config_instruct_fbgemm: AutoConfig) -> AutoConfig:
    """Creates a base fp8 config from an fbgemm config."""
    config = deepcopy(hf_config_instruct_fbgemm)
    config.quantization_config["quant_method"] = "fp8"
    config.quantization_config["activation_scheme"] = "dynamic"
    config.quantization_config["weight_block_size"] = [128, 128]
    return config


def test_parse_fp8_bias_dtype(
    hf_config_fp8: AutoConfig,
    state_dict_with_lm_head_and_fbgemm_scales: dict[str, WeightData],
) -> None:
    """Tests that parse_float8_config correctly extracts bias_dtype for fp8 method."""
    config = hf_config_fp8
    state_dict = state_dict_with_lm_head_and_fbgemm_scales.copy()

    # Add weight_scale for fp8 (required)
    state_dict["layers.0.mlp.down_proj.weight_scale"] = WeightData(
        name="layers.0.mlp.down_proj.weight_scale",
        shape=Shape((1, 1)),
        dtype=DType.float32,
        data=torch.zeros((1, 1), dtype=DType.float32.to_torch()),
    )

    # Add bias weights
    state_dict["layers.0.mlp.down_proj.bias"] = WeightData(
        name="layers.0.mlp.down_proj.bias",
        shape=Shape((1,)),
        dtype=DType.float16,
        data=torch.zeros((1,), dtype=DType.float16.to_torch()),
    )

    dtype = DType.float8_e4m3fn
    float8_config = parse_float8_config(config, state_dict, dtype)

    assert float8_config is not None
    assert float8_config.quant_method == "fp8"
    assert float8_config.bias_dtype == DType.float16


def test_parse_float4_from_standalone_hf_quant_config(
    hf_config_instruct_fbgemm: AutoConfig,
    tmp_path: Path,
) -> None:
    """Tests parsing FP4 config from standalone hf_quant_config.json.

    Emulates nvidia/DeepSeek-R1-0528-NVFP4-v2.
    """
    repo_id = "nvidia/DeepSeek-R1-0528-NVFP4-v2"
    repo_dir = tmp_path / "nvidia/DeepSeek-R1-0528-NVFP4-v2"
    repo_dir.mkdir(parents=True)
    hf_quant_config = {
        "producer": {"name": "modelopt", "version": "0.0"},
        "quantization": {"quant_algo": "NVFP4"},
    }
    (repo_dir / "hf_quant_config.json").write_text(json.dumps(hf_quant_config))

    hf_config = deepcopy(hf_config_instruct_fbgemm)
    if hasattr(hf_config, "quantization_config"):
        del hf_config.quantization_config
    hf_config._name_or_path = str(repo_dir)

    float8_config = parse_float8_config(hf_config, {}, DType.uint8)

    assert float8_config is not None
    assert float8_config.quant_method == "modelopt"
    assert float8_config.quant_algo == "NVFP4"


def test_parse_float4_skips_gptq_quant_method(
    hf_config_instruct_fbgemm: AutoConfig,
) -> None:
    """Tests FP4 parsing ignores GPTQ quantization configs."""
    hf_config = deepcopy(hf_config_instruct_fbgemm)
    hf_config.quantization_config = {"quant_method": "gptq"}

    float8_config = parse_float8_config(hf_config, {}, DType.uint8)

    assert float8_config is None
