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
"""Tests for the modelopt ``quantized_layers`` reader and the QuantConfig presets."""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.nn.quant_config import (
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
)
from max.pipelines.weights.quant import (
    ModelOptModuleQuant,
    read_modelopt_quantized_layers,
)

_ALLOWED = {("FP8", None), ("W4A16_NVFP4", 16)}


def _config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "kv_cache_quant_algo": None,
        "quantized_layers": {
            "model.layers.0.mlp.gate_proj": {
                "quant_algo": "W4A16_NVFP4",
                "group_size": 16,
            },
            "model.layers.0.self_attn.qkv_proj": {"quant_algo": "FP8"},
        },
    }
    config.update(overrides)
    return config


def test_reads_each_module() -> None:
    modules = read_modelopt_quantized_layers(
        _config(),
        allowed=_ALLOWED,
        expected_modules={
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.self_attn.qkv_proj",
        },
        kv_cache_quant_algos=(None,),
    )
    assert modules == {
        "model.layers.0.mlp.gate_proj": ModelOptModuleQuant("W4A16_NVFP4", 16),
        "model.layers.0.self_attn.qkv_proj": ModelOptModuleQuant("FP8", None),
    }


def test_the_standalone_file_needs_no_quant_method() -> None:
    config = _config()
    del config["quant_method"]
    assert read_modelopt_quantized_layers(config, allowed=_ALLOWED)


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"quant_algo": "NVFP4"}, "Model: .*cannot read quant_algo 'NVFP4'"),
        ({"quant_method": "fp8"}, "quant_method='fp8'"),
        ({"quantized_layers": {}}, "no 'quantized_layers' map"),
        ({"quantized_layers": None}, "no 'quantized_layers' map"),
        ({"kv_cache_quant_algo": "FP8"}, "kv_cache_quant_algo is 'FP8'"),
        (
            {"quantized_layers": {"lm_head": {"quant_algo": "NVFP4"}}},
            r"quantized_layers\['lm_head'\] .* 'NVFP4' with group_size None",
        ),
        (
            {
                "quantized_layers": {
                    "lm_head": {"quant_algo": "W4A16_NVFP4", "group_size": 32}
                }
            },
            "group_size 32",
        ),
        (
            {
                "quantized_layers": {
                    "lm_head": {"quant_algo": "FP8", "group_size": "x"}
                }
            },
            r"quantized_layers\['lm_head'\]",
        ),
        ({"quantized_layers": {"lm_head": "FP8"}}, r"\['lm_head'\] is 'FP8'"),
    ],
)
def test_refuses_what_the_caller_cannot_read(
    overrides: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        read_modelopt_quantized_layers(
            _config(**overrides),
            allowed=_ALLOWED,
            kv_cache_quant_algos=(None,),
            model_name="Model",
        )


def test_accepts_the_top_level_algorithms_the_caller_names() -> None:
    read_modelopt_quantized_layers(
        _config(quant_algo="NVFP4"),
        allowed=_ALLOWED,
        quant_algos=("NVFP4", "MIXED_PRECISION"),
    )


def test_kv_cache_algorithm_is_unchecked_by_default() -> None:
    read_modelopt_quantized_layers(
        _config(kv_cache_quant_algo="FP8"), allowed=_ALLOWED
    )


def test_expected_modules_must_match_exactly() -> None:
    with pytest.raises(ValueError, match=r"outside the 1 expected.*qkv_proj"):
        read_modelopt_quantized_layers(
            _config(),
            allowed=_ALLOWED,
            expected_modules={"model.layers.0.mlp.gate_proj"},
        )
    with pytest.raises(
        ValueError, match=r"lists 2 of the 3 expected .*down_proj"
    ):
        read_modelopt_quantized_layers(
            _config(),
            allowed=_ALLOWED,
            expected_modules={
                "model.layers.0.mlp.gate_proj",
                "model.layers.0.mlp.down_proj",
                "model.layers.0.self_attn.qkv_proj",
            },
        )


def test_blockscaled_fp8_preset() -> None:
    config = QuantConfig.blockscaled_fp8(
        mlp_quantized_layers=range(2),
        attn_quantized_layers=[1],
        embedding_output_dtype=DType.bfloat16,
    )
    assert config.format == QuantFormat.BLOCKSCALED_FP8
    assert config.weight_scale.granularity == ScaleGranularity.BLOCK
    assert config.weight_scale.block_size == (128, 128)
    assert config.weight_scale.dtype == DType.float32
    assert config.input_scale.origin == ScaleOrigin.DYNAMIC
    assert config.input_scale.block_size == (1, 128)
    assert config.input_scale.dtype == DType.float32
    assert config.mlp_quantized_layers == {0, 1}
    assert config.attn_quantized_layers == {1}
    assert config.embedding_output_dtype == DType.bfloat16
    assert config.scales_granularity_mnk == (1, 128, 128)

    narrow = QuantConfig.blockscaled_fp8(
        mlp_quantized_layers=(),
        attn_quantized_layers=(),
        block_size=64,
        scale_dtype=DType.bfloat16,
    )
    assert narrow.weight_scale.block_size == (64, 64)
    assert narrow.input_scale.block_size == (1, 64)
    assert narrow.input_scale.dtype == DType.bfloat16


def test_mxfp4_preset() -> None:
    config = QuantConfig.mxfp4(mlp_quantized_layers={3})
    assert config.is_mxfp4
    assert config.weight_scale.dtype == DType.float8_e8m0fnu
    assert config.weight_scale.block_size == (1, 32)
    assert config.input_scale.origin == ScaleOrigin.DYNAMIC
    assert config.input_scale.dtype == DType.float32
    assert config.input_scale.block_size == (1, 32)
    assert config.mlp_quantized_layers == {3}
    assert config.attn_quantized_layers == set()


def test_nvfp4_preset() -> None:
    config = QuantConfig.nvfp4(
        mlp_quantized_layers={0},
        attn_quantized_layers={0},
        shared_experts_weight_dtype=DType.bfloat16,
    )
    assert config.is_nvfp4
    assert config.weight_scale.dtype == DType.float8_e4m3fn
    assert config.weight_scale.block_size == (1, 16)
    assert config.input_scale.origin == ScaleOrigin.STATIC
    assert config.input_scale.dtype == DType.float32
    assert config.input_scale.block_size == (1, 16)
    assert config.shared_experts_weight_dtype == DType.bfloat16
