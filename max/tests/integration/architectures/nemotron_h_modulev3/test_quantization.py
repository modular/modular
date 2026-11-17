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

from collections.abc import Callable, Mapping
from typing import Any
from unittest.mock import Mock

import numpy as np
import numpy.typing as npt
import pytest
from _tiny_config import (
    NVFP4_DIMS,
    TINY,
    TINY_LAYERS,
    WIDE,
    model_config,
)
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph import Shape
from max.graph.weights import WeightData
from max.nn import Module
from max.nn.kv_cache import MHAKVCacheParams
from max.pipelines.architectures.nemotron_h_modulev3.layers.attention import (
    NemotronHAttention,
)
from max.pipelines.architectures.nemotron_h_modulev3.layers.mamba2 import (
    NemotronHMamba2Mixer,
)
from max.pipelines.architectures.nemotron_h_modulev3.layers.moe import (
    NemotronHMoE,
    NemotronHRouter,
)
from max.pipelines.architectures.nemotron_h_modulev3.layers.quantized import (
    RowGroupedLinear,
)
from max.pipelines.architectures.nemotron_h_modulev3.model_config import (
    ATTN_CACHE_KEY,
    LayerKind,
    NemotronHConfig,
)
from max.pipelines.architectures.nemotron_h_modulev3.quantization import (
    ModuleFormat,
    NemotronHQuantScheme,
    parse_quant_scheme,
)
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.realization_context import ensure_context
from max.tensor.sharding import DeviceMesh

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


def _raw(weight: WeightData) -> npt.NDArray[np.uint8]:
    """Returns a weight's bytes, one row per row of the weight."""
    rows = int(weight.shape[0]) if weight.shape else 1
    raw = np.from_dlpack(weight.to_buffer().view(DType.uint8)).view(np.uint8)
    return raw.reshape(rows, -1)


def _mesh(n: int) -> DeviceMesh:
    return DeviceMesh((CPU(),) * n, (n,), ("tp",))


def _build(n: int, build: Callable[[], Module[..., Any]]) -> Module[..., Any]:
    """Builds a layer on a mesh of ``n`` simulated devices."""
    with default_dtype(DType.bfloat16), default_device(_mesh(n)):
        return build()


def _assign(module: Module[..., Any], values: Mapping[str, WeightData]) -> None:
    """Sets the module's parameters by path, placed as it declares them."""
    for path, value in values.items():
        *parents, attr = path.split(".")
        owner: Any = module
        for part in parents:
            owner = owner[int(part)] if part.isdigit() else getattr(owner, part)
        param = getattr(owner, attr)
        tensor = Tensor.from_dlpack(value.to_buffer())
        assert tensor.dtype == param.dtype, path
        setattr(owner, attr, tensor.to(param.mapping))


def _bytes(tensor: Tensor) -> npt.NDArray[np.uint8]:
    raw = np.from_dlpack(tensor.driver_tensor.view(DType.uint8))
    return raw.view(np.uint8).reshape(-1)


def _random(shape: tuple[int, ...], seed: int) -> WeightData:
    """Returns finite normal BF16 values, as weights are: the top halves of
    float32s."""
    values = np.random.default_rng(seed).standard_normal(shape, np.float32)
    bits = (values.view(np.uint32) >> 16).astype(np.uint16)
    return _weight(bits, DType.bfloat16)


def _scalar(value: float) -> WeightData:
    return _weight(np.array(value, dtype=np.float32), DType.float32)


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


def test_the_graph_stacks_the_nvfp4_routed_experts() -> None:
    """Each mixer's per-expert tensors become the grouped matmul's
    operands: the packed weights, the block scales and the global scales,
    each with one leading expert axis."""
    rng = np.random.default_rng(0)
    mixer = "backbone.layers.1.mixer"
    # NVFP4 shapes, so each projection's block scales span whole atoms.
    config = model_config(TINY_LAYERS, NVFP4_DIMS)
    e = config.num_experts
    hidden, channels = config.hidden_size, config.moe_intermediate_size

    values: dict[str, WeightData] = {}
    stored: dict[str, list[npt.NDArray[np.generic]]] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    for proj, out_dim, in_dim in (
        ("up", channels, hidden),
        ("down", hidden, channels),
    ):
        shapes[proj] = (out_dim, in_dim // 2)
        stored[proj] = []
        for expert in range(e):
            module = f"experts.{expert}.{proj}_proj"
            values[f"{module}.weight"] = _weight(
                rng.integers(0, 256, (out_dim, in_dim // 2), dtype=np.uint8),
                DType.uint8,
            )
            values[f"{module}.weight_scale"] = _weight(
                np.full((out_dim, in_dim // 16), _E4M3[1.0], dtype=np.uint8),
                DType.float8_e4m3fn,
            )
            values[f"{module}.weight_scale_2"] = _scalar(0.25)
            stored[proj].append(_raw(values[f"{module}.weight"]))

    with ensure_context():
        moe = _build(1, lambda: NemotronHMoE(config, mixer, w4a4_experts=True))
        assert isinstance(moe, NemotronHMoE)
        _assign(moe, values)
        placed = {}
        for proj in stored:
            weight, block_scale, global_scale = moe.placed_params(
                proj, _mesh(1)
            )
            # An x86 host compiles no float8_e4m3fn op, so the block scales
            # are never realized: only their type is checked.
            assert block_scale.dtype == DType.float8_e4m3fn
            del block_scale
            placed[proj] = (weight, global_scale)

    for proj, (weight, global_scale) in placed.items():
        assert weight.dtype == DType.uint8
        assert tuple(int(d) for d in weight.shape) == (
            e,
            *shapes[proj],
        )
        np.testing.assert_array_equal(
            _bytes(weight.local_shards[0]),
            np.stack(stored[proj]).reshape(-1),
        )
        assert global_scale.dtype == DType.float32
        np.testing.assert_array_equal(
            _bytes(global_scale.local_shards[0]),
            np.full(e, 0.25, dtype=np.float32).view(np.uint8),
        )


def test_the_graph_stacks_the_bf16_routed_experts_as_stored() -> None:
    """Each mixer's per-expert BF16 weights become one stack, expert by
    expert in router order."""
    mixer = "backbone.layers.1.mixer"
    config = model_config(TINY_LAYERS, TINY)
    e = config.num_experts
    hidden, channels = config.hidden_size, config.moe_intermediate_size

    values: dict[str, WeightData] = {}
    stored: dict[str, list[npt.NDArray[np.generic]]] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    for proj, shape in (
        ("up", (channels, hidden)),
        ("down", (hidden, channels)),
    ):
        shapes[proj] = shape
        stored[proj] = []
        for expert in range(e):
            weight = _random(shape, seed=expert)
            values[f"experts.{expert}.{proj}_proj.weight"] = weight
            stored[proj].append(_raw(weight))

    with ensure_context():
        moe = _build(1, lambda: NemotronHMoE(config, mixer))
        assert isinstance(moe, NemotronHMoE)
        _assign(moe, values)
        placed = {proj: moe.placed_params(proj, _mesh(1)) for proj in stored}

    for proj, (stack,) in placed.items():
        assert stack.dtype == DType.bfloat16
        assert tuple(int(d) for d in stack.shape) == (e, *shapes[proj])
        np.testing.assert_array_equal(
            _bytes(stack.local_shards[0]),
            np.stack(stored[proj]).reshape(-1),
        )


def _device_block(
    stored: npt.NDArray[np.uint8], axis: int, d: int, n: int, padded: int
) -> npt.NDArray[np.uint8]:
    """Returns device ``d``'s share of ``stored``'s ``axis``, zero-padded to
    ``padded`` columns of ``stored``'s units."""
    share = np.split(stored, n, axis=axis)[d]
    pad = [(0, 0)] * stored.ndim
    pad[axis] = (0, padded - share.shape[axis])
    return np.pad(share, pad)


def test_each_device_gets_its_padded_share_of_every_expert() -> None:
    """Up projections split their rows and down projections their columns.

    A device's share of each expert's channels is zero-padded to a multiple
    of 64: each of two devices pads its 48 of 96 channels to 64.
    """
    mixer = "backbone.layers.1.mixer"
    n = 2
    config = model_config(
        TINY_LAYERS,
        NVFP4_DIMS | {"moe_intermediate_size": 96},
        n_devices=n,
    )
    hidden, inner = config.hidden_size, config.moe_intermediate_size
    padded = config.moe_intermediate_size_per_device

    shapes = {"up": (inner, hidden), "down": (hidden, inner)}
    values: dict[str, WeightData] = {}
    stored: dict[str, list[npt.NDArray[np.uint8]]] = {}
    for proj, shape in shapes.items():
        stored[proj] = []
        for e in range(config.num_experts):
            weight = _random(shape, seed=e)
            values[f"experts.{e}.{proj}_proj.weight"] = weight
            stored[proj].append(_raw(weight))
    # One realization, so the test compiles one graph.
    with ensure_context():
        moe = _build(n, lambda: NemotronHMoE(config, mixer))
        assert isinstance(moe, NemotronHMoE)
        _assign(moe, values)
        placed = {proj: moe.placed_params(proj, _mesh(n)) for proj in shapes}

    for proj in shapes:
        # Expert channels are the rows of up and the columns of down, whose
        # rows hold two bytes per channel.
        axis, size = (0, padded) if proj == "up" else (1, 2 * padded)
        for d in range(n):
            want = np.stack(
                [_device_block(w, axis, d, n, size) for w in stored[proj]]
            )
            np.testing.assert_array_equal(
                _bytes(placed[proj][0].local_shards[d]), want.reshape(-1)
            )


def _mamba(config: NemotronHConfig, mixer: str) -> NemotronHMamba2Mixer:
    n = len(config.devices)
    mamba = _build(n, lambda: NemotronHMamba2Mixer(config, mixer))
    assert isinstance(mamba, NemotronHMamba2Mixer)
    return mamba


def test_mamba_rows_are_regrouped_by_device() -> None:
    """Device 0's rows are the first half of each fused part."""
    config = model_config(TINY_LAYERS, TINY, n_devices=2)
    # The tiny mixer's in_proj stacks gate 32, x 32, B 16, C 16 and dt 4
    # rows, and its conv stacks x, B and C.
    in_proj = _random((100, 32), seed=0)
    conv_bias = _random((64,), seed=1)
    with ensure_context():
        mamba = _mamba(config, "backbone.layers.0.mixer")
        _assign(mamba, {"in_proj.weight": in_proj, "conv1d.bias": conv_bias})
        assert isinstance(mamba.in_proj, RowGroupedLinear)
        placed = mamba.in_proj.placed_weight(_mesh(2))
        _, bias = mamba.conv1d.placed(_mesh(2))

    halves = [(0, 16), (32, 48), (64, 72), (80, 88), (96, 98)]
    rows = _raw(in_proj)
    np.testing.assert_array_equal(
        _bytes(placed.local_shards[0]),
        np.concatenate([rows[a:b] for a, b in halves]).reshape(-1),
    )
    bias_bytes = _raw(conv_bias).reshape(64, 2)
    np.testing.assert_array_equal(
        _bytes(bias.local_shards[1]),
        np.concatenate(
            [bias_bytes[16:32], bias_bytes[40:48], bias_bytes[56:64]]
        ).reshape(-1),
    )


def test_each_device_gets_a_copy_of_its_kv_head() -> None:
    """With four devices and two KV heads, devices 0 and 1 get head 0."""
    config = model_config(TINY_LAYERS, WIDE, n_devices=4)
    # Eight query heads, and two KV heads, of eight rows each.
    q_proj = _random((64, 32), seed=0)
    k_proj = _random((16, 32), seed=1)
    v_proj = _random((16, 32), seed=2)
    with ensure_context():
        attention = _build(
            4,
            lambda: NemotronHAttention(
                config,
                config.kv_params.child(ATTN_CACHE_KEY, MHAKVCacheParams),
                0,
            ),
        )
        assert isinstance(attention, NemotronHAttention)
        _assign(
            attention,
            {
                "qkv_proj.q_proj.weight": q_proj,
                "qkv_proj.k_proj.weight": k_proj,
                "qkv_proj.v_proj.weight": v_proj,
            },
        )
        qkv = attention.qkv_weight(_mesh(4))

    for d, shard in enumerate(qkv.local_shards):
        head = slice(d // 2 * 8, (d // 2 + 1) * 8)
        np.testing.assert_array_equal(
            _bytes(shard),
            np.concatenate(
                [
                    _raw(q_proj)[d * 16 : (d + 1) * 16],
                    _raw(k_proj)[head],
                    _raw(v_proj)[head],
                ]
            ).reshape(-1),
        )


@pytest.mark.parametrize("quantized", [False, True])
def test_the_router_loads_the_gate_as_stored(quantized: bool) -> None:
    """The gate weight loads in the dtype its checkpoint stores it in, and
    the router casts it to its own dtype in the graph.

    The BF16 checkpoint stores the gate in BF16 and the modelopt-quantized
    one in float32; the score bias is float32 in both.
    """
    config = model_config(TINY_LAYERS, TINY)
    if quantized:
        config.quant_scheme = NemotronHQuantScheme(
            {"lm_head": ModuleFormat.NVFP4_WEIGHT_ONLY}
        )
    router = _build(1, lambda: NemotronHRouter(config))
    assert isinstance(router, NemotronHRouter)
    assert router.weight.dtype == config.router_gate_stored_dtype
    assert router.weight.dtype == (
        DType.float32 if quantized else DType.bfloat16
    )
    assert router.e_score_correction_bias.dtype == DType.float32
    # The router's own dtype, which the graph casts the gate to.
    assert router.gate_dtype == config.router_gate_dtype


@pytest.mark.parametrize("stored", [DType.bfloat16, DType.float32])
def test_finalize_reads_the_gate_dtype_from_the_checkpoint(
    stored: DType,
) -> None:
    """A quantized checkpoint can store its gate in BF16, as one that
    quantizes only ``lm_head`` may. The router declares the gate as the
    checkpoint stores it, so a loader that reads the bytes as stored, with
    no cast, loads it."""
    config = model_config(TINY_LAYERS, TINY)
    config.quant_scheme = NemotronHQuantScheme(
        {"lm_head": ModuleFormat.NVFP4_WEIGHT_ONLY}
    )
    lm_head = (
        "lm_head.weight",
        "lm_head.weight_scale",
        "lm_head.weight_scale_2",
    )
    state_dict = {name: Mock(spec=WeightData) for name in lm_head}
    for mixer in config.mixers(LayerKind.MOE):
        state_dict[f"{mixer}.gate.weight"] = Mock(spec=WeightData, dtype=stored)
    config.finalize(Mock(), state_dict)
    assert config.router_gate_stored_dtype == stored
    router = _build(1, lambda: NemotronHRouter(config))
    assert isinstance(router, NemotronHRouter)
    assert router.weight.dtype == stored
