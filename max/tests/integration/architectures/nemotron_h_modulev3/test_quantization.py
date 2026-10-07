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
"""Nemotron-H's per-module quantization scheme and its weight layouts."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from _tiny_config import NVFP4_DIMS, TINY, TINY_LAYERS, WIDE, model_config
from max.driver import Buffer
from max.dtype import DType
from max.graph import Shape
from max.graph.weights import WeightData
from max.pipelines.architectures.nemotron_h_modulev3.layers.quantized import (
    nvfp4_block_scale_shape,
)
from max.pipelines.architectures.nemotron_h_modulev3.model_config import (
    LayerKind,
)
from max.pipelines.architectures.nemotron_h_modulev3.quantization import (
    ModuleFormat,
    NemotronHQuantScheme,
    Parallelism,
    linear_parallelism,
    parse_quant_scheme,
)
from max.pipelines.architectures.nemotron_h_modulev3.weight_adapters import (
    _bytes,
    interleave_nvfp4_scales,
    permute_mamba_for_tp,
    prepare_nvfp4_linears,
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
    with pytest.raises(ValueError, match="lm_head"):
        parse_quant_scheme(config)


def test_a_wrong_nvfp4_group_size_is_refused() -> None:
    config = {
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": {
            "lm_head": {"quant_algo": "W4A16_NVFP4", "group_size": 32}
        },
    }
    with pytest.raises(ValueError, match="group_size 32"):
        parse_quant_scheme(config)


def test_a_uniform_algorithm_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot read quant_algo 'NVFP4'"):
        parse_quant_scheme({"quant_method": "modelopt", "quant_algo": "NVFP4"})


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

    out = stack_nvfp4_experts(state_dict, 2, {mixer: 0}, num_devices=1)

    assert sorted(out) == [
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
    assert tuple(block.shape) == (2, 128, 4)
    scale = out[f"{mixer}.up_scale"]
    np.testing.assert_array_equal(
        np.from_dlpack(scale.to_buffer()), [0.25, 0.25]
    )


@pytest.mark.parametrize("num_devices", [1, 2])
def test_shared_expert_slices_stack_with_the_routed_experts(
    num_devices: int,
) -> None:
    """The shared up projection splits by rows and the down projection by
    columns, each slice shaped as a routed expert, after the routed experts.
    At two devices each holds 64 of a slice's 128 channels, which needs no
    padding, so the device blocks lay the slice out as stored."""
    rng = np.random.default_rng(0)
    mixer = "backbone.layers.1.mixer"
    hidden, inner, experts, slices = 128, 128, 2, 2

    def nvfp4(module: str, n: int, k: int, global_scale: float) -> None:
        state_dict[f"{module}.weight"] = _weight(
            rng.integers(0, 256, (n, k // 2), dtype=np.uint8), DType.uint8
        )
        state_dict[f"{module}.weight_scale"] = _weight(
            rng.integers(0x28, 0x48, (n, k // 16), dtype=np.uint8),
            DType.float8_e4m3fn,
        )
        state_dict[f"{module}.weight_scale_2"] = _weight(
            np.array(global_scale, dtype=np.float32), DType.float32
        )

    state_dict: dict[str, WeightData] = {}
    for e in range(experts):
        nvfp4(f"{mixer}.experts.{e}.up_proj", inner, hidden, 0.5)
        nvfp4(f"{mixer}.experts.{e}.down_proj", hidden, inner, 0.5)
    shared = f"{mixer}.shared_experts"
    nvfp4(f"{shared}.up_proj", slices * inner, hidden, 0.25)
    nvfp4(f"{shared}.down_proj", hidden, slices * inner, 0.25)

    def array(name: str) -> npt.NDArray[np.uint8]:
        return np.from_dlpack(
            state_dict[name].to_buffer().view(DType.uint8)
        ).view(np.uint8)

    up_codes = array(f"{shared}.up_proj.weight")
    up_scales = array(f"{shared}.up_proj.weight_scale")
    down_codes = array(f"{shared}.down_proj.weight")
    down_scales = array(f"{shared}.down_proj.weight_scale")

    out = stack_nvfp4_experts(
        state_dict, experts, {mixer: slices}, num_devices=num_devices
    )

    assert not any(name.startswith(shared) for name in out)
    positions = [experts, experts + 1]

    def stacked(name: str) -> npt.NDArray[np.uint8]:
        return np.from_dlpack(
            out[f"{mixer}.{name}"].to_buffer().view(DType.uint8)
        ).view(np.uint8)

    for i, at in enumerate(positions):
        rows = slice(i * inner, (i + 1) * inner)
        np.testing.assert_array_equal(stacked("up_weight")[at], up_codes[rows])
        np.testing.assert_array_equal(
            stacked("up_block_scale")[at], up_scales[rows]
        )
        cols = slice(i * inner // 2, (i + 1) * inner // 2)
        np.testing.assert_array_equal(
            stacked("down_weight")[at], down_codes[:, cols]
        )
        scale_cols = slice(i * inner // 16, (i + 1) * inner // 16)
        np.testing.assert_array_equal(
            stacked("down_block_scale")[at], down_scales[:, scale_cols]
        )
    expected_scales = [0.25 if p in positions else 0.5 for p in range(4)]
    for proj in ("up", "down"):
        np.testing.assert_array_equal(
            np.from_dlpack(out[f"{mixer}.{proj}_scale"].to_buffer()),
            expected_scales,
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
def test_bf16_stacking_copies_each_expert_into_its_slice(n: int) -> None:
    """Under tensor parallelism each device's share of the channels is
    padded with zeros to 64."""
    rng = np.random.default_rng(0)
    mixer = "backbone.layers.1.mixer"
    state_dict = {
        f"{mixer}.experts.{e}.{proj}.weight": _weight(
            rng.integers(0, 2**15, (4, 32), dtype=np.uint16), DType.bfloat16
        )
        for e in range(2)
        for proj in ("up_proj", "down_proj")
    }

    out = stack_bf16_experts(state_dict, 2, {mixer}, num_devices=n)

    assert sorted(out) == [f"{mixer}.down_weight", f"{mixer}.up_weight"]
    padded = {1: {"up": (2, 4, 32), "down": (2, 4, 32)}}.get(
        n, {"up": (2, 128, 32), "down": (2, 4, 128)}
    )
    for proj, axis, channels in (("up", 1, 4), ("down", 2, 32)):
        stack = out[f"{mixer}.{proj}_weight"]
        assert tuple(stack.shape) == padded[proj]
        values = _strip_device_padding(_values(stack), axis, n, channels)
        for e in range(2):
            np.testing.assert_array_equal(
                values[e],
                _values(state_dict[f"{mixer}.experts.{e}.{proj}_proj.weight"]),
            )


def test_bf16_stacking_refuses_a_quantized_expert() -> None:
    mixer = "backbone.layers.1.mixer"
    state_dict = {
        f"{mixer}.experts.{e}.{proj}.weight": _weight(
            np.zeros((4, 16), dtype=np.uint8), DType.uint8
        )
        for e in range(2)
        for proj in ("up_proj", "down_proj")
    }
    with pytest.raises(ValueError, match=r"experts\.0\.up_proj"):
        stack_bf16_experts(state_dict, 2, {mixer})


@pytest.mark.parametrize(
    "module, n, shares",
    [
        # Column parallel: each device interleaves its own output rows.
        ("backbone.layers.1.mixer.shared_experts.up_proj", 2, 2),
        # Row parallel: each device interleaves its own input columns.
        ("backbone.layers.1.mixer.shared_experts.down_proj", 2, 2),
        # Replicated: one share that every device holds whole.
        ("lm_head", 2, 1),
        ("backbone.layers.1.mixer.shared_experts.up_proj", 1, 1),
    ],
)
def test_nvfp4_linear_scales_are_interleaved_per_device(
    module: str, n: int, shares: int
) -> None:
    config = model_config(TINY_LAYERS, NVFP4_DIMS, n_devices=n)
    in_dim, out_dim = config.linear_shape(module)
    config.quant_scheme = NemotronHQuantScheme(
        {module: ModuleFormat.NVFP4_WEIGHT_ONLY}
    )
    scales = (
        (np.arange(out_dim * in_dim // 16) % 251 + 1)
        .astype(np.uint8)
        .reshape(out_dim, in_dim // 16)
    )
    weight = _weight(
        np.zeros((out_dim, in_dim // 2), dtype=np.uint8), DType.uint8
    )
    state_dict = {
        f"{module}.weight": weight,
        f"{module}.weight_scale": _weight(scales, DType.float8_e4m3fn),
        f"{module}.weight_scale_2": _weight(
            np.array(0.25, dtype=np.float32), DType.float32
        ),
    }

    out = prepare_nvfp4_linears(state_dict, config, n)

    assert out[f"{module}.weight"] is weight
    parallelism = linear_parallelism(module)
    block = out[f"{module}.weight_scale"]
    assert list(block.shape) == nvfp4_block_scale_shape(
        out_dim, in_dim, parallelism, n
    )
    got = np.from_dlpack(block.to_buffer().view(DType.uint8))
    if parallelism is Parallelism.COLUMN:
        parts = np.split(scales, shares, axis=0)
    elif parallelism is Parallelism.ROW:
        parts = np.split(scales, shares, axis=1)
    else:
        parts = [scales]
    for share, part in zip(got, parts, strict=True):
        np.testing.assert_array_equal(share, interleave_nvfp4_scales(part))
    global_scale = out[f"{module}.weight_scale_2"]
    assert list(global_scale.shape) == [1]
    np.testing.assert_array_equal(
        np.from_dlpack(global_scale.to_buffer()), [0.25]
    )


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
    state_dict = {}
    for e in range(2):
        for proj, (rows, cols) in shapes.items():
            module = f"{mixer}.experts.{e}.{proj}"
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

    out = stack_nvfp4_experts(state_dict, 2, {mixer: 0}, num_devices=n)

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
        "up_block_scale": (2, n * padded, hidden // 16),
        "down_weight": (2, hidden, n * padded // 2),
        "down_block_scale": (2, hidden, n * padded // 16),
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
            block = slice(d * padded, (d + 1) * padded)
            up_codes = stacked["up_weight"][e, block]
            up_scales = stacked["up_block_scale"][e, block]
            assert not up_codes[share:].any() and not up_scales[share:].any()
            up_d = _nvfp4_values(up_codes, up_scales, 0.25)
            np.testing.assert_array_equal(
                up_d[:share], up[d * share : (d + 1) * share]
            )

            packed = slice(d * padded // 2, (d + 1) * padded // 2)
            down_codes = stacked["down_weight"][e, :, packed]
            groups = slice(d * padded // 16, (d + 1) * padded // 16)
            down_scales = stacked["down_block_scale"][e, :, groups]
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


def test_nvfp4_in_proj_scales_move_with_their_rows() -> None:
    """Each device's share of an NVFP4 in_proj's block scales holds the
    scales of the rows it holds."""
    config = model_config(TINY_LAYERS, NVFP4_DIMS, n_devices=2)
    mixer = "backbone.layers.4.mixer"
    module = f"{mixer}.in_proj"
    config.quant_scheme = NemotronHQuantScheme(
        {module: ModuleFormat.NVFP4_WEIGHT_ONLY}
    )
    in_dim, out_dim = config.linear_shape(module)
    # Each row's first two code bytes hold its checkpoint row index.
    codes = np.zeros((out_dim, in_dim // 2), dtype=np.uint8)
    codes[:, 0] = np.arange(out_dim) % 256
    codes[:, 1] = np.arange(out_dim) // 256
    scales = np.random.default_rng(0).integers(
        0, 256, (out_dim, in_dim // 16), dtype=np.uint8
    )
    conv_dim = config.conv_dim
    state_dict = {
        f"{module}.weight": _weight(codes, DType.uint8),
        f"{module}.weight_scale": _weight(scales, DType.float8_e4m3fn),
        f"{module}.weight_scale_2": _weight(
            np.array(1.0, dtype=np.float32), DType.float32
        ),
        f"{mixer}.conv1d.weight": _weight(
            np.zeros((conv_dim, 4), dtype=np.uint16), DType.bfloat16
        ),
        f"{mixer}.conv1d.bias": _weight(
            np.zeros((conv_dim,), dtype=np.uint16), DType.bfloat16
        ),
    }
    for other in config.mixers(LayerKind.MAMBA) - {mixer}:
        for name in ("in_proj.weight", "conv1d.weight", "conv1d.bias"):
            state_dict[f"{other}.{name}"] = state_dict[f"{mixer}.{name}"]

    out = prepare_nvfp4_linears(
        permute_mamba_for_tp(state_dict, config, 2), config, 2
    )

    placed = np.from_dlpack(out[f"{module}.weight"].to_buffer())
    shares = np.from_dlpack(
        out[f"{module}.weight_scale"].to_buffer().view(DType.uint8)
    )
    for share, device_codes in zip(
        shares, np.split(placed, 2, axis=0), strict=True
    ):
        rows = device_codes[:, 0] + 256 * device_codes[:, 1].astype(np.int64)
        np.testing.assert_array_equal(
            share, interleave_nvfp4_scales(scales[rows])
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
