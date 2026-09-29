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
"""Nemotron-H's per-module quantization scheme and its BF16 dequantization."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.graph import Shape
from max.graph.weights import WeightData
from max.pipelines.architectures.nemotron_h_modulev3.quantization import (
    ModuleFormat,
    parse_quant_scheme,
)
from max.pipelines.architectures.nemotron_h_modulev3.weight_adapters import (
    dequantize_to_bf16,
)

# E4M3 encodings of 0.5, 1, 2 and 4.
_E4M3 = {0.5: 0x30, 1.0: 0x38, 2.0: 0x40, 4.0: 0x48}


def _weight(array: npt.NDArray[np.generic], dtype: DType) -> WeightData:
    buffer = Buffer.from_numpy(np.ascontiguousarray(array))
    return WeightData(
        data=buffer.view(dtype, array.shape),
        name="",
        dtype=dtype,
        shape=Shape(array.shape),
    )


def _values(weight: WeightData) -> npt.NDArray[np.float32]:
    assert weight.dtype == DType.bfloat16
    bits = np.from_dlpack(weight.to_buffer().view(DType.uint16))
    return (bits.astype(np.uint32) << 16).view(np.float32)


def test_mixed_precision_names_each_module_format() -> None:
    scheme = parse_quant_scheme(
        {
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_scheme": {
                "dynamic": False,
                "num_bits": 8,
                "type": "float",
            },
            "quantized_layers": {
                "backbone.layers.0.mixer.in_proj": {"quant_algo": "FP8"},
                "lm_head": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
            },
        }
    )
    assert scheme.format_of("backbone.layers.0.mixer.in_proj") is (
        ModuleFormat.FP8_STATIC_TENSOR
    )
    assert scheme.format_of("lm_head") is ModuleFormat.NVFP4_WEIGHT_ONLY
    assert scheme.format_of("backbone.embeddings") is ModuleFormat.BF16

    names = {
        "backbone.layers.0.mixer.in_proj.weight",
        "backbone.layers.0.mixer.in_proj.weight_scale",
        "backbone.layers.0.mixer.in_proj.input_scale",
        "lm_head.weight",
        "lm_head.weight_scale",
        "lm_head.weight_scale_2",
    }
    scheme.check_weights(names)
    with pytest.raises(ValueError, match="lm_head"):
        scheme.check_weights(names - {"lm_head.weight_scale_2"})


def test_an_unknown_algorithm_is_refused() -> None:
    config = {
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": {"lm_head": {"quant_algo": "NVFP4"}},
    }
    with pytest.raises(NotImplementedError, match="lm_head"):
        parse_quant_scheme(config)


def test_nvfp4_matches_the_reference_dequantization() -> None:
    codes = np.random.default_rng(0).integers(0, 16, size=(2, 32))
    # Two E2M1 codes per byte, the even column in the low nibble.
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
    block_values = np.array([[1.0, 2.0], [0.5, 4.0]], dtype=np.float32)
    block_scales = np.vectorize(_E4M3.get)(block_values).astype(np.uint8)
    state_dict = {
        "m.weight": _weight(packed, DType.uint8),
        "m.weight_scale": _weight(block_scales, DType.float8_e4m3fn),
        "m.weight_scale_2": _weight(
            np.array(0.25, dtype=np.float32), DType.float32
        ),
    }

    out = dequantize_to_bf16(state_dict, {"m": ModuleFormat.NVFP4_WEIGHT_ONLY})

    e2m1 = np.array(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=np.float32,
    )
    expected = e2m1[codes] * np.repeat(block_values, 16, axis=1) * 0.25
    assert list(out) == ["m.weight"]
    np.testing.assert_array_equal(_values(out["m.weight"]), expected)


def test_fp8_drops_the_input_scale() -> None:
    values = np.array([[1.0, 2.0], [0.5, 4.0]], dtype=np.float32)
    codes = np.vectorize(_E4M3.get)(values).astype(np.uint8)
    state_dict = {
        "m.weight": _weight(codes, DType.float8_e4m3fn),
        "m.weight_scale": _weight(
            np.array(0.5, dtype=np.float32), DType.float32
        ),
        "m.input_scale": _weight(
            np.array([3.0], dtype=np.float32), DType.float32
        ),
    }

    out = dequantize_to_bf16(state_dict, {"m": ModuleFormat.FP8_STATIC_TENSOR})

    assert list(out) == ["m.weight"]
    np.testing.assert_array_equal(_values(out["m.weight"]), values * 0.5)
