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
"""Tests that converted checkpoint tensors strict-load into model layers."""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.graph import DeviceRef, Graph, ShardingStrategy
from max.graph.weights import WeightData
from max.pipelines.architectures.deepseekV41.layers.attention import (
    wo_a_linear,
)
from max.pipelines.architectures.deepseekV41.layers.moe import (
    DeepseekV41Gate,
    DeepseekV41SharedExpert,
)
from max.pipelines.architectures.deepseekV41.weight_adapters import (
    convert_weight_data,
    rewire_trunk_state_dict,
    with_dtype,
)


def _f32(
    name: str, shape: tuple[int, ...] = (4,), value: float = 1.0
) -> WeightData:
    return with_dtype(
        np.full(shape, value, dtype=np.float32), DType.float32, name
    )


def _bf16(
    name: str, shape: tuple[int, ...] = (4,), value: float = 1.0
) -> WeightData:
    f32 = np.full(shape, value, dtype=np.float32)
    bits = (f32.view(np.uint32) >> 16).astype(np.uint16)
    return with_dtype(bits, DType.bfloat16, name)


_N_EXPERTS = 4
_GATE = "layers.0.ffn.gate"


def _trunk_layer_with_gate(
    bias_vl: np.ndarray | None,
) -> dict[str, WeightData]:
    """Returns one trunk layer's gate and attention projections.

    The attention projections are included because rewiring fuses the query
    projections on every layer.
    """
    attn = "layers.0.attn"
    state_dict = {
        f"{attn}.wq_a.weight": _bf16(f"{attn}.wq_a.weight", (2, 4)),
        f"{attn}.wkv.weight": _bf16(f"{attn}.wkv.weight", (2, 4)),
        f"{attn}.wq_b.weight": _bf16(f"{attn}.wq_b.weight", (4, 2)),
        f"{attn}.q_norm.weight": _f32(f"{attn}.q_norm.weight", (2,)),
        f"{_GATE}.weight": _f32(f"{_GATE}.weight", (_N_EXPERTS, 4)),
        f"{_GATE}.bias": _f32(f"{_GATE}.bias", (_N_EXPERTS,), 0.5),
    }
    if bias_vl is not None:
        state_dict[f"{_GATE}.bias_vl"] = with_dtype(
            bias_vl, DType.float32, f"{_GATE}.bias_vl"
        )
    return state_dict


def _load_gate_bias_vl(state_dict: dict[str, WeightData]) -> np.ndarray:
    """Strict-loads the rewired gate tensors into a ``DeepseekV41Gate``."""
    prefix = f"trunk.{_GATE}."
    gate = DeepseekV41Gate(
        n_routed_experts=_N_EXPERTS,
        num_experts_per_tok=2,
        routed_scaling_factor=1.0,
        hidden_size=4,
        dtype=DType.float32,
        device=DeviceRef.CPU(),
    )
    gate.load_state_dict(
        {
            name.removeprefix(prefix): data
            for name, data in rewire_trunk_state_dict(state_dict).items()
            if name.startswith(prefix)
        },
        strict=True,
    )
    return np.from_dlpack(gate.state_dict(auto_initialize=False)["bias_vl"])


def test_gate_bias_vl_loads_checkpoint_value_when_present() -> None:
    bias_vl = np.arange(1, _N_EXPERTS + 1, dtype=np.float32)
    loaded = _load_gate_bias_vl(_trunk_layer_with_gate(bias_vl))
    np.testing.assert_array_equal(loaded, bias_vl)


def test_gate_bias_vl_loads_zeros_when_absent() -> None:
    loaded = _load_gate_bias_vl(_trunk_layer_with_gate(None))
    np.testing.assert_array_equal(loaded, np.zeros(_N_EXPERTS, np.float32))


@pytest.mark.parametrize("required", ["weight", "bias"])
def test_gate_weights_other_than_bias_vl_stay_required(required: str) -> None:
    state_dict = _trunk_layer_with_gate(None)
    del state_dict[f"{_GATE}.{required}"]
    with pytest.raises(
        ValueError, match=rf"Missing required weights: {required}\b"
    ):
        _load_gate_bias_vl(state_dict)


_SHARED_EXPERT = "layers.0.ffn.shared_experts"


_SE_HIDDEN = 32
_SE_INTER = 32


def _mxfp8_weight_and_scale(
    base: str, out_dim: int, in_dim: int
) -> dict[str, WeightData]:
    """Builds MXFP8 ``{base}.weight``/``{base}.scale`` tensors.

    Every weight byte is ``0x38`` (1.0) and every e8m0 scale byte is ``127``
    (``2**0``), so dequantized values are all 1.0.
    """
    weight_bytes = np.full((out_dim, in_dim), 0x38, dtype=np.uint8)
    scale_bytes = np.full((out_dim // 32, in_dim // 32), 127, dtype=np.uint8)
    return {
        f"{base}.weight": with_dtype(
            weight_bytes, DType.float8_e4m3fn, f"{base}.weight"
        ),
        f"{base}.scale": with_dtype(
            scale_bytes, DType.float8_e8m0fnu, f"{base}.scale"
        ),
    }


def test_shared_expert_tensors_load_strict() -> None:
    """Converted shared-expert tensors strict-load into the module.

    The MXFP8 weight loads with a ``(out_dim, in_dim // 32)``
    ``weight_scale``.
    """
    state_dict = _trunk_layer_with_gate(None)
    for proj, out_dim, in_dim in (
        ("w1", _SE_INTER, _SE_HIDDEN),
        ("w3", _SE_INTER, _SE_HIDDEN),
        ("w2", _SE_HIDDEN, _SE_INTER),
    ):
        state_dict.update(
            _mxfp8_weight_and_scale(f"{_SHARED_EXPERT}.{proj}", out_dim, in_dim)
        )

    rewired = rewire_trunk_state_dict(convert_weight_data(state_dict))
    prefix = f"trunk.{_SHARED_EXPERT}."
    expert = DeepseekV41SharedExpert(
        hidden_size=_SE_HIDDEN,
        moe_intermediate_size=_SE_INTER,
        swiglu_limit=10.0,
        device=DeviceRef.CPU(),
    )
    expert.load_state_dict(
        {
            name.removeprefix(prefix): data
            for name, data in rewired.items()
            if name.startswith(prefix)
        },
        strict=True,
    )

    loaded = expert.state_dict(auto_initialize=False)
    # Check `.shape`: numpy cannot DLPack-import float8 buffers.
    # w1 and w3 arrive fused, rows [w1; w3].
    assert not any(name.startswith(("w1.", "w3.")) for name in loaded)
    weight = loaded["gate_up_proj.weight"]
    assert isinstance(weight, Buffer)
    assert tuple(weight.shape) == (2 * _SE_INTER, _SE_HIDDEN)
    assert weight.dtype == DType.float8_e4m3fn
    weight_scale = loaded["gate_up_proj.weight_scale"]
    assert isinstance(weight_scale, Buffer)
    assert tuple(weight_scale.shape) == (2 * _SE_INTER, _SE_HIDDEN // 32)


# Real per-group shapes are [8 * 1024, 4096]; small here, same structure.
_WOA_GROUPS, _WOA_RANK, _WOA_WIDTH = 4, 64, 128


def _woa_checkpoint(prefix: str) -> dict[str, WeightData]:
    """Returns ``<prefix>.attn.wo_a`` as stored in the checkpoint.

    The e4m3 weight is 1.0 everywhere, and the compact 32x32 e8m0 scale has a
    distinct byte per block.
    """
    n, k = _WOA_GROUPS * _WOA_RANK, _WOA_WIDTH
    scale = (np.arange((n // 32) * (k // 32)) % 8 + 120).astype(np.uint8)
    return {
        f"{prefix}.attn.wo_a.weight": with_dtype(
            np.full((n, k), 0x38, np.uint8), DType.float8_e4m3fn, "w"
        ),
        f"{prefix}.attn.wo_a.scale": with_dtype(
            scale.reshape(n // 32, k // 32), DType.float8_e8m0fnu, "s"
        ),
    }


@pytest.mark.parametrize("woa_mxfp8", [True, False])
def test_woa_converted_tensors_strict_load_and_shard(woa_mxfp8: bool) -> None:
    """Converted ``wo_a`` strict-loads and shards rowwise with its scale."""
    linear = wo_a_linear(
        _WOA_WIDTH,
        _WOA_GROUPS * _WOA_RANK,
        DeviceRef.CPU(),
        mxfp8=woa_mxfp8,
    )
    out = convert_weight_data(_woa_checkpoint("layers.3"), woa_mxfp8=woa_mxfp8)
    linear.load_state_dict(
        {
            name.removeprefix("layers.3.attn.wo_a."): data
            for name, data in out.items()
        },
        weight_alignment=1,
        strict=True,
    )
    for name, weight in linear.raw_state_dict().items():
        assert weight.dtype == out[f"layers.3.attn.wo_a.{name}"].dtype
    linear.sharding_strategy = ShardingStrategy.rowwise(2)
    devices = [DeviceRef.CPU(), DeviceRef.CPU()]
    with Graph("woa_shard", input_types=[]):
        shards = linear.weight.shard(devices)
        assert [tuple(int(d) for d in s.shape) for s in shards] == [
            (_WOA_GROUPS * _WOA_RANK // 2, _WOA_WIDTH)
        ] * 2
        if woa_mxfp8:
            assert linear.weight_scale is not None
            scale_shards = linear.weight_scale.shard(devices)
            assert [tuple(int(d) for d in s.shape) for s in scale_shards] == [
                (_WOA_GROUPS * _WOA_RANK // 2, _WOA_WIDTH // 32)
            ] * 2
        else:
            assert linear.weight_scale is None
