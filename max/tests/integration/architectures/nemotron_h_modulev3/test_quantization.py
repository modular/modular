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
    interleave_nvfp4_scales,
    stack_nvfp4_experts,
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


def test_interleave_puts_each_scale_where_the_kernel_reads_it() -> None:
    rows, cols = 130, 8
    scales = (
        (np.arange(rows * cols) % 251 + 1).astype(np.uint8).reshape(rows, cols)
    )

    out = interleave_nvfp4_scales(scales)

    assert out.shape == (2, 2, 32, 4, 4)
    for r in range(rows):
        for c in range(cols):
            got = out[r // 128, c // 4, r % 32, (r % 128) // 32, c % 4]
            assert got == scales[r, c]
    # The two real rows of the second granule, and zero padding after them.
    assert np.count_nonzero(out[1]) == 2 * cols


def test_stacking_keeps_the_routed_experts_in_nvfp4() -> None:
    rng = np.random.default_rng(0)
    mixer = "backbone.layers.1.mixer"
    modules = {
        f"{mixer}.experts.{e}.{proj}": ModuleFormat.NVFP4_WEIGHT_ONLY
        for e in range(2)
        for proj in ("up_proj", "down_proj")
    }
    modules["lm_head"] = ModuleFormat.NVFP4_WEIGHT_ONLY
    state_dict = {}
    for module in modules:
        state_dict[f"{module}.weight"] = _weight(
            rng.integers(0, 256, (128, 32), dtype=np.uint8), DType.uint8
        )
        state_dict[f"{module}.weight_scale"] = _weight(
            np.full((128, 4), _E4M3[1.0], dtype=np.uint8), DType.float8_e4m3fn
        )
        state_dict[f"{module}.weight_scale_2"] = _weight(
            np.array(0.25, dtype=np.float32), DType.float32
        )

    out, remaining = stack_nvfp4_experts(state_dict, modules, {mixer})

    assert remaining == {"lm_head": ModuleFormat.NVFP4_WEIGHT_ONLY}
    assert sorted(n for n in out if n.startswith(mixer)) == [
        f"{mixer}.{p}_{s}"
        for p in ("down", "up")
        for s in ("block_scale", "scale", "weight")
    ]
    up = out[f"{mixer}.up_weight"]
    assert up.dtype == DType.uint8 and tuple(up.shape) == (2, 128, 32)
    np.testing.assert_array_equal(
        np.from_dlpack(up.to_buffer())[1],
        np.from_dlpack(
            state_dict[f"{mixer}.experts.1.up_proj.weight"].to_buffer()
        ),
    )
    block = out[f"{mixer}.up_block_scale"]
    assert block.dtype == DType.float8_e4m3fn
    assert tuple(block.shape) == (2, 1, 1, 32, 4, 4)
    scale = out[f"{mixer}.up_scale"]
    np.testing.assert_array_equal(
        np.from_dlpack(scale.to_buffer()), [0.25, 0.25]
    )
