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
from _tiny_config import TINY, TINY_LAYERS, WIDE, model_config
from max.driver import Buffer
from max.dtype import DType
from max.graph import Shape
from max.graph.weights import WeightData
from max.pipelines.architectures.nemotron_h_modulev3.model_config import (
    LayerKind,
)
from max.pipelines.architectures.nemotron_h_modulev3.quantization import (
    ModuleFormat,
    parse_quant_scheme,
)
from max.pipelines.architectures.nemotron_h_modulev3.weight_adapters import (
    _bytes,
    dequantize_to_bf16,
    interleave_nvfp4_scales,
    permute_mamba_for_tp,
    repeat_kv_heads_for_tp,
    stack_bf16_experts,
    stack_nvfp4_experts,
)
from max.pipelines.weights._fp8 import e4m3fn_lut
from max.pipelines.weights.fp4_quantization import (
    FP4Format,
    e2m1_decode_table,
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


def _strip_device_padding(
    stack: npt.NDArray[np.generic], axis: int, n: int, channels: int
) -> npt.NDArray[np.generic]:
    """Returns ``stack`` without its per-device padding, which must be 0."""
    real = []
    for block in np.split(stack, n, axis=axis):
        kept, padding = np.split(block, [channels // n], axis=axis)
        assert not padding.any()
        real.append(kept)
    return np.concatenate(real, axis=axis)


@pytest.mark.parametrize("n", [1, 2])
def test_bf16_stacking_dequantizes_each_expert_into_its_slice(n: int) -> None:
    """NVFP4 experts on GPUs without the W4A4 matmul, and BF16 experts.

    Under tensor parallelism each device's share of the channels is padded
    with zeros to 64.
    """
    rng = np.random.default_rng(0)
    nvfp4_mixer, bf16_mixer = (
        "backbone.layers.1.mixer",
        "backbone.layers.3.mixer",
    )
    modules = {
        f"{nvfp4_mixer}.experts.{e}.{proj}": ModuleFormat.NVFP4_WEIGHT_ONLY
        for e in range(2)
        for proj in ("up_proj", "down_proj")
    }
    modules["lm_head"] = ModuleFormat.NVFP4_WEIGHT_ONLY
    state_dict = {}
    for module in modules:
        state_dict[f"{module}.weight"] = _weight(
            rng.integers(0, 256, (4, 16), dtype=np.uint8), DType.uint8
        )
        state_dict[f"{module}.weight_scale"] = _weight(
            np.full((4, 2), _E4M3[2.0], dtype=np.uint8), DType.float8_e4m3fn
        )
        state_dict[f"{module}.weight_scale_2"] = _weight(
            np.array(0.25, dtype=np.float32), DType.float32
        )
    for e in range(2):
        for proj in ("up_proj", "down_proj"):
            bits = rng.integers(0, 2**15, (4, 32), dtype=np.uint16)
            state_dict[f"{bf16_mixer}.experts.{e}.{proj}.weight"] = _weight(
                bits, DType.bfloat16
            )

    out, remaining = stack_bf16_experts(
        state_dict, modules, {nvfp4_mixer, bf16_mixer}, num_devices=n
    )

    assert remaining == {"lm_head": ModuleFormat.NVFP4_WEIGHT_ONLY}
    assert not any(".experts." in name for name in out)
    expected = dequantize_to_bf16(
        state_dict, {m: f for m, f in modules.items() if m != "lm_head"}
    )
    padded = {1: {"up": (2, 4, 32), "down": (2, 4, 32)}}.get(
        n, {"up": (2, 128, 32), "down": (2, 4, 128)}
    )
    for mixer, source in ((nvfp4_mixer, expected), (bf16_mixer, state_dict)):
        for proj, axis, channels in (("up", 1, 4), ("down", 2, 32)):
            stack = out[f"{mixer}.{proj}_weight"]
            assert tuple(stack.shape) == padded[proj]
            values = _strip_device_padding(_values(stack), axis, n, channels)
            for e in range(2):
                np.testing.assert_array_equal(
                    values[e],
                    _values(source[f"{mixer}.experts.{e}.{proj}_proj.weight"]),
                )


def _deinterleave(block: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Inverts :func:`interleave_nvfp4_scales`, padding rows included."""
    granules, atoms = block.shape[:2]
    return block.transpose(0, 3, 2, 1, 4).reshape(granules * 128, atoms * 4)


def _nvfp4_values(
    codes: npt.NDArray[np.uint8], scales: npt.NDArray[np.uint8], scale: float
) -> npt.NDArray[np.float64]:
    e2m1 = e2m1_decode_table(FP4Format.NVFP4).astype(np.float64)
    values = np.stack([e2m1[codes & 0xF], e2m1[codes >> 4]], axis=-1)
    values = values.reshape(codes.shape[0], -1, 16)
    values *= e4m3fn_lut()[scales][..., None]
    return values.reshape(codes.shape[0], -1) * scale


def _relu2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.maximum(x, 0) ** 2


def test_tp_stacking_pads_each_devices_nvfp4_channels_with_zeros() -> None:
    """Two devices each hold 48 of the 96 channels, padded to 64.

    The padding's codes and block scales are 0x00, which dequantize to zero,
    so the devices' partial MLPs sum to the whole one.
    """
    rng = np.random.default_rng(0)
    mixer = "backbone.layers.1.mixer"
    hidden, inner, n = 64, 96, 2
    share, padded = inner // n, 64
    shapes = {"up_proj": (inner, hidden), "down_proj": (hidden, inner)}
    modules = {
        f"{mixer}.experts.{e}.{proj}": ModuleFormat.NVFP4_WEIGHT_ONLY
        for e in range(2)
        for proj in shapes
    }
    state_dict = {}
    for module in modules:
        rows, cols = shapes[module.rpartition(".")[2]]
        state_dict[f"{module}.weight"] = _weight(
            rng.integers(0, 256, (rows, cols // 2), dtype=np.uint8),
            DType.uint8,
        )
        # Nonzero scales, 2^-2 to 2^2, so zero padding stands out.
        state_dict[f"{module}.weight_scale"] = _weight(
            rng.integers(0x28, 0x48, (rows, cols // 16), dtype=np.uint8),
            DType.float8_e4m3fn,
        )
        state_dict[f"{module}.weight_scale_2"] = _weight(
            np.array(0.25, dtype=np.float32), DType.float32
        )

    out, _ = stack_nvfp4_experts(state_dict, modules, {mixer}, num_devices=n)

    stacked = {
        name: _bytes(out[f"{mixer}.{name}"])
        for name in (
            "up_weight",
            "up_block_scale",
            "down_weight",
            "down_block_scale",
        )
    }
    assert {k: v.shape for k, v in stacked.items()} == {
        "up_weight": (2, n * padded, hidden // 2),
        "up_block_scale": (2, n, hidden // 64, 32, 4, 4),
        "down_weight": (2, hidden, n * padded // 2),
        "down_block_scale": (2, 1, n * padded // 64, 32, 4, 4),
    }
    x = rng.standard_normal((5, hidden))
    for e in range(2):
        up, down = (
            _nvfp4_values(
                _bytes(state_dict[f"{mixer}.experts.{e}.{proj}.weight"]),
                _bytes(state_dict[f"{mixer}.experts.{e}.{proj}.weight_scale"]),
                0.25,
            )
            for proj in ("up_proj", "down_proj")
        )
        got = np.zeros((5, hidden))
        for d in range(n):
            up_codes = stacked["up_weight"][e, d * padded : (d + 1) * padded]
            up_scales = _deinterleave(stacked["up_block_scale"][e, d : d + 1])
            assert not up_codes[share:].any() and not up_scales[share:].any()
            up_d = _nvfp4_values(up_codes, up_scales[:padded], 0.25)
            np.testing.assert_array_equal(
                up_d[:share], up[d * share : (d + 1) * share]
            )

            packed = slice(d * padded // 2, (d + 1) * padded // 2)
            down_codes = stacked["down_weight"][e, :, packed]
            down_scales = _deinterleave(
                stacked["down_block_scale"][e, :, d : d + 1]
            )[:hidden]
            assert not down_codes[:, share // 2 :].any()
            assert not down_scales[:, share // 16 :].any()
            down_d = _nvfp4_values(down_codes, down_scales, 0.25)
            np.testing.assert_array_equal(
                down_d[:, :share], down[:, d * share : (d + 1) * share]
            )
            got += _relu2(x @ up_d.T) @ down_d.T
        np.testing.assert_allclose(got, _relu2(x @ up.T) @ down.T, rtol=1e-10)


def _bits(weight: WeightData) -> npt.NDArray[np.generic]:
    return np.from_dlpack(weight.to_buffer().view(DType.uint16))


def test_mamba_rows_are_regrouped_by_device() -> None:
    """Device 0's rows are the first half of each fused part."""
    config = model_config(TINY_LAYERS, TINY, n_devices=2)
    rng = np.random.default_rng(0)
    # The tiny mixer's in_proj stacks gate 32, x 32, B 16, C 16 and dt 4
    # rows, and its conv stacks x, B and C.
    in_proj = rng.integers(0, 2**15, (100, 32), dtype=np.uint16)
    conv_bias = rng.integers(0, 2**15, (64,), dtype=np.uint16)
    state_dict: dict[str, WeightData] = {}
    for mixer in config.mixers(LayerKind.MAMBA):
        for name, array in (
            ("in_proj.weight", in_proj),
            ("conv1d.weight", np.zeros((64, 4), dtype=np.uint16)),
            ("conv1d.bias", conv_bias),
        ):
            state_dict[f"{mixer}.{name}"] = _weight(array, DType.bfloat16)

    out = permute_mamba_for_tp(state_dict, config, 2)

    mixer = "backbone.layers.4.mixer"
    halves = [(0, 16), (32, 48), (64, 72), (80, 88), (96, 98)]
    np.testing.assert_array_equal(
        _bits(out[f"{mixer}.in_proj.weight"])[:50],
        np.concatenate([in_proj[a:b] for a, b in halves]),
    )
    np.testing.assert_array_equal(
        _bits(out[f"{mixer}.conv1d.bias"])[32:],
        np.concatenate([conv_bias[16:32], conv_bias[40:48], conv_bias[56:64]]),
    )


def test_each_device_gets_a_copy_of_its_kv_head() -> None:
    """With four devices and two KV heads, devices 0 and 1 get head 0."""
    config = model_config(TINY_LAYERS, WIDE, n_devices=4)
    rng = np.random.default_rng(0)
    # Two heads of eight rows each.
    k_proj = rng.integers(0, 2**15, (16, 32), dtype=np.uint16)
    mixer = "backbone.layers.3.mixer"
    state_dict = {
        f"{mixer}.{proj}.weight": _weight(k_proj, DType.bfloat16)
        for proj in ("k_proj", "v_proj")
    }

    out = repeat_kv_heads_for_tp(state_dict, config, 4)

    for proj in ("k_proj", "v_proj"):
        np.testing.assert_array_equal(
            _bits(out[f"{mixer}.{proj}.weight"]),
            np.concatenate([k_proj[:8], k_proj[:8], k_proj[8:], k_proj[8:]]),
        )
    # Two devices split the heads without repeating them.
    assert repeat_kv_heads_for_tp(state_dict, config, 2) == state_dict
