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
"""Tests for the MX expert-weight preshuffle helpers (KERN-3393).

The float8 cases back each ``WeightData`` with a MAX ``Buffer`` whose DLPack
dtype is a float8 numpy cannot import — the exact shape of the production
state dict. numpy's ``from_dlpack`` error type for those changed from
``RuntimeError`` to ``BufferError`` in numpy 2.5.0 (numpy gh-30937); the
helpers used to dispatch on that type and, under numpy >= 2.5, silently
skipped every MXFP8 expert while the caller still flipped
``block_scaled_preshuffled_b`` — serving row-major weights to the preb kernel.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.graph.type import Shape
from max.graph.weights import WeightData
from max.nn.moe import MoE
from max.pipelines.weights.block_scaled_preshuffle import (
    preshuffle_block_scaled_b_experts,
    preshuffle_block_scaled_b_scales,
    preshuffle_block_scaled_b_stacked,
    sigma_interleave_gate_up,
)

_N, _K_BYTES = 16, 64
_MN, _K_SCALES = 32, 8


def _weight_bytes(seed: int, shape: tuple[int, int]) -> np.ndarray:
    return (
        np.random.default_rng(seed)
        .integers(0, 256, size=shape)
        .astype(np.uint8)
    )


def _f8_weight_data(name: str, raw: np.ndarray, dtype: DType) -> WeightData:
    """WeightData backed by a MAX Buffer whose DLPack dtype is ``dtype``."""
    buf = Buffer.from_dlpack(raw).view(dtype, raw.shape)
    return WeightData(buf, name, dtype, Shape(raw.shape))


def _expected_b_5d(src: np.ndarray) -> np.ndarray:
    n, k_bytes = src.shape
    return (
        src.reshape(n // 16, 16, k_bytes // 64, 4, 16)
        .transpose(0, 2, 3, 1, 4)
        .reshape(n, k_bytes)
    )


def _expected_scale_4d(src: np.ndarray) -> np.ndarray:
    mn, k_scales = src.shape
    return (
        src.reshape(mn // 32, 2, 16, k_scales // 8, 2, 4)
        .transpose(0, 3, 5, 2, 4, 1)
        .reshape(mn, k_scales)
    )


def _result_bytes(wd: WeightData) -> np.ndarray:
    return np.from_dlpack(
        wd.to_buffer().view(DType.uint8, wd.shape.static_dims)
    )


def test_preshuffle_b_experts_float8_buffer_backed() -> None:
    """MXFP8 experts behind a float8 DLPack producer must be preshuffled."""
    names = [
        f"layers.0.mlp.experts.{i}.{proj}.weight"
        for i in range(2)
        for proj in ("gate_proj", "up_proj", "down_proj")
    ]
    raws = {n: _weight_bytes(i, (_N, _K_BYTES)) for i, n in enumerate(names)}
    state_dict = {
        n: _f8_weight_data(n, raw, DType.float8_e4m3fn)
        for n, raw in raws.items()
    }

    preshuffle_block_scaled_b_experts(state_dict)

    for n, raw in raws.items():
        assert state_dict[n].dtype == DType.float8_e4m3fn
        np.testing.assert_array_equal(
            _result_bytes(state_dict[n]), _expected_b_5d(raw)
        )


def test_preshuffle_b_experts_uint8() -> None:
    """MXFP4-packed uint8 experts (Kimi K2.5 path) keep working."""
    name = "language_model.layers.3.mlp.experts.7.up_proj.weight"
    raw = _weight_bytes(7, (_N, _K_BYTES))
    state_dict = {name: WeightData.from_numpy(raw.copy(), name)}

    preshuffle_block_scaled_b_experts(state_dict)

    assert state_dict[name].dtype == DType.uint8
    np.testing.assert_array_equal(
        _result_bytes(state_dict[name]), _expected_b_5d(raw)
    )


def test_preshuffle_b_scales_e8m0_buffer_backed() -> None:
    """E8M0 scales behind a float8 DLPack producer must be preshuffled."""
    name = "layers.0.mlp.experts.0.gate_proj.weight_scale"
    raw = _weight_bytes(11, (_MN, _K_SCALES))
    state_dict = {name: _f8_weight_data(name, raw, DType.float8_e8m0fnu)}

    preshuffle_block_scaled_b_scales(state_dict)

    assert state_dict[name].dtype == DType.float8_e8m0fnu
    np.testing.assert_array_equal(
        _result_bytes(state_dict[name]), _expected_scale_4d(raw)
    )


def test_preshuffle_b_experts_rejects_unshuffleable_group() -> None:
    """A matched group with no shuffleable weight raises instead of no-oping."""
    name = "layers.0.mlp.experts.0.gate_proj.weight"
    raw = np.zeros((_N, _K_BYTES // 2), dtype=np.float32)
    state_dict = {name: WeightData.from_numpy(raw, name)}

    with pytest.raises(ValueError, match="preshuffle skipped"):
        preshuffle_block_scaled_b_experts(state_dict)


def test_preshuffle_b_experts_rejects_partial_group() -> None:
    """A group mixing shuffleable and unshuffleable weights raises."""
    good = "layers.0.mlp.experts.0.gate_proj.weight"
    bad = "layers.0.mlp.experts.1.gate_proj.weight"
    state_dict = {
        good: WeightData.from_numpy(_weight_bytes(0, (_N, _K_BYTES)), good),
        bad: WeightData.from_numpy(
            np.zeros((_N, _K_BYTES), dtype=np.float32), bad
        ),
    }

    with pytest.raises(ValueError, match="preshuffle skipped"):
        preshuffle_block_scaled_b_experts(state_dict)


def test_preshuffle_b_scales_rejects_unshuffleable_group() -> None:
    """A matched scale group with a non-E8M0 scale raises."""
    name = "layers.0.mlp.experts.0.gate_proj.weight_scale"
    raw = np.zeros((_MN, _K_SCALES), dtype=np.float32)
    state_dict = {name: WeightData.from_numpy(raw, name)}

    with pytest.raises(ValueError, match="preshuffle skipped"):
        preshuffle_block_scaled_b_scales(state_dict)


def test_preshuffle_b_experts_counts_matches_under_virtual_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Virtual devices skip the byte copy, so identity cannot report a match.

    The graph-dump tools compile against virtual devices only. Reporting 0
    there would make a clean dump indistinguishable from a naming regression.
    """
    monkeypatch.setattr(
        "max.pipelines.weights.block_scaled_preshuffle.is_virtual_device_mode",
        lambda: True,
    )
    name = "layers.0.mlp.experts.0.gate_proj.weight"
    state_dict = {
        name: WeightData.from_numpy(_weight_bytes(3, (_N, _K_BYTES)), name)
    }
    before = state_dict[name]

    assert preshuffle_block_scaled_b_experts(state_dict) == 1
    assert state_dict[name] is before, "virtual mode must not permute bytes"


def _sigma_fused(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    """The sigma-permuted gate_up: row 2i is gate row i, row 2i+1 is up row i."""
    d, k = gate.shape
    out = np.empty((2 * d, k), dtype=gate.dtype)
    out[0::2] = gate
    out[1::2] = up
    return out


def test_sigma_interleave_before_preshuffle_matches_fused_reference() -> None:
    """Interleave-then-preshuffle equals preshuffling one sigma-ordered N=2D
    weight, which is what the fused SwiGLU epilogue reads. Also pins that two
    N=D blobs concatenate to the N=2D blob, which is how the graph reassembles
    them. The companion test below covers the wrong order.
    """
    gate = _weight_bytes(21, (_N, _K_BYTES))
    up = _weight_bytes(22, (_N, _K_BYTES))
    g_name = "layers.0.mlp.experts.0.gate_proj.weight"
    u_name = "layers.0.mlp.experts.0.up_proj.weight"
    state_dict = {
        g_name: WeightData.from_numpy(gate.copy(), g_name),
        u_name: WeightData.from_numpy(up.copy(), u_name),
    }

    assert sigma_interleave_gate_up(state_dict) == 1
    preshuffle_block_scaled_b_experts(state_dict)

    # `MoE.gate_up_proj` stacks gate then up per expert, so the kernel sees the
    # two halves concatenated.
    got = np.concatenate(
        [_result_bytes(state_dict[g_name]), _result_bytes(state_dict[u_name])],
        axis=0,
    )
    np.testing.assert_array_equal(got, _expected_b_5d(_sigma_fused(gate, up)))


def test_preshuffle_then_graph_permute_is_not_the_sigma_layout() -> None:
    """Pins the bug: permuting after the preshuffle does not produce the layout.

    Reproduces what a graph-level ``reshape([E, 2, D, K]) -> permute`` does to
    already-preshuffled blobs, and asserts it differs from the real thing --
    so anyone who moves the interleave back after the preshuffle fails here
    instead of shipping a model that silently computes wrong values.
    """
    gate = _weight_bytes(23, (_N, _K_BYTES))
    up = _weight_bytes(24, (_N, _K_BYTES))

    # Wrong order: preshuffle each projection, then interleave the rows of the
    # shuffled buffers as though they were still logical N rows.
    wrong = _sigma_fused(_expected_b_5d(gate), _expected_b_5d(up))
    # Right order: interleave the raw rows, then preshuffle as one N=2D weight.
    right = _expected_b_5d(_sigma_fused(gate, up))

    assert wrong.shape == right.shape, (
        "shapes match, which is why this is silent"
    )
    assert not np.array_equal(wrong, right), (
        "if these are equal the preshuffle no longer interleaves N with K and"
        " the ordering constraint this test guards has gone away"
    )


def test_sigma_interleave_scales() -> None:
    """Scales take the same treatment, or gemm1 pairs a row with a wrong scale."""
    gate = _weight_bytes(31, (_MN, _K_SCALES))
    up = _weight_bytes(32, (_MN, _K_SCALES))
    g_name = "layers.0.mlp.experts.0.gate_proj.weight_scale"
    u_name = "layers.0.mlp.experts.0.up_proj.weight_scale"
    state_dict = {
        g_name: WeightData.from_numpy(gate.copy(), g_name),
        u_name: WeightData.from_numpy(up.copy(), u_name),
    }

    assert sigma_interleave_gate_up(state_dict) == 1

    fused = _sigma_fused(gate, up)
    np.testing.assert_array_equal(
        _result_bytes(state_dict[g_name]), fused[:_MN]
    )
    np.testing.assert_array_equal(
        _result_bytes(state_dict[u_name]), fused[_MN:]
    )


def test_sigma_interleave_rejects_missing_sibling() -> None:
    """A gate without its up raises: interleaving half pairs the wrong rows."""
    g_name = "layers.0.mlp.experts.0.gate_proj.weight"
    state_dict = {
        g_name: WeightData.from_numpy(_weight_bytes(41, (_N, _K_BYTES)), g_name)
    }

    with pytest.raises(ValueError, match="both gate_proj and up_proj"):
        sigma_interleave_gate_up(state_dict)


def test_sigma_interleave_leaves_down_proj_alone() -> None:
    """down_proj has no sibling to pair with and must pass through untouched."""
    d_name = "layers.0.mlp.experts.0.down_proj.weight"
    raw = _weight_bytes(51, (_N, _K_BYTES))
    state_dict = {d_name: WeightData.from_numpy(raw.copy(), d_name)}

    assert sigma_interleave_gate_up(state_dict) == 0
    np.testing.assert_array_equal(_result_bytes(state_dict[d_name]), raw)


def _moe_layout_stub(
    *, fused: bool, preshuffled: bool, interleaved: bool
) -> SimpleNamespace:
    """The surface `MoE._needs_graph_sigma_permute` reads, without a graph."""
    return SimpleNamespace(
        _uses_fused_swiglu_layout=lambda: fused,
        _weights_preshuffled=preshuffled,
        quant_config=SimpleNamespace(gate_up_sigma_interleaved=interleaved),
    )


@pytest.mark.parametrize(
    ("fused", "preshuffled", "interleaved", "expected"),
    [
        (False, False, False, False),
        (False, True, False, False),
        (True, False, False, True),
        (True, True, True, False),
    ],
)
def test_graph_sigma_permute_decision(
    fused: bool, preshuffled: bool, interleaved: bool, expected: bool
) -> None:
    """Raw weights are permuted in the graph; interleaved preshuffled ones are not."""
    stub = _moe_layout_stub(
        fused=fused, preshuffled=preshuffled, interleaved=interleaved
    )
    assert MoE._needs_graph_sigma_permute(cast(MoE, stub)) is expected


def test_preshuffled_fused_layout_needs_the_load_time_interleave() -> None:
    """A loader that preshuffles without interleaving must not build.

    The graph cannot permute preshuffled rows, so skipping the permute here
    would feed the fused epilogue split gate/up halves -- it runs and computes
    wrong activations with no error.
    """
    stub = _moe_layout_stub(fused=True, preshuffled=True, interleaved=False)
    with pytest.raises(ValueError, match="sigma_interleave_gate_up"):
        MoE._needs_graph_sigma_permute(cast(MoE, stub))


# Kimi K3's routed gate/up shard at TP8: `[E, N, K_BYTES]` stacked by the
# weight adapter rather than stored per expert, with gate and up stacked along
# N so each half splits on its own.
_K3_E, _K3_N, _K3_KB, _K3_KS = 2, 6144, 1792, 112
_K3_TP, _K3_INTER = 8, 3072


def _k3_gate_up_rows(device: int) -> np.ndarray:
    """The two disjoint N runs `ShardingStrategy.gate_up(axis=1)` hands device i."""
    half = _K3_INTER // _K3_TP
    return np.r_[
        device * half : (device + 1) * half,
        _K3_INTER + device * half : _K3_INTER + (device + 1) * half,
    ]


def _stacked_state_dict(name: str, seed: int) -> dict[str, WeightData]:
    weight = _weight_bytes(seed, (_K3_E * _K3_N, _K3_KB)).reshape(
        _K3_E, _K3_N, _K3_KB
    )
    scale_raw = _weight_bytes(seed + 1, (_K3_E * _K3_N, _K3_KS)).reshape(
        _K3_E, _K3_N, _K3_KS
    )
    return {
        name: WeightData.from_numpy(weight, name),
        f"{name}_scale": _f8_weight_data(
            f"{name}_scale", scale_raw, DType.float8_e8m0fnu
        ),
    }


def test_preshuffle_b_stacked_matches_per_expert_layout() -> None:
    """Each expert slice of a stacked tensor gets the same bytes as a lone one."""
    name = "layers.0.block_sparse_moe.experts_gate_up_proj"
    state_dict = _stacked_state_dict(name, 10)
    src = _result_bytes(state_dict[name]).copy()
    src_scale = _result_bytes(state_dict[f"{name}_scale"]).copy()

    assert preshuffle_block_scaled_b_stacked(state_dict, [name]) == 2 * _K3_E

    got = _result_bytes(state_dict[name])
    got_scale = _result_bytes(state_dict[f"{name}_scale"])
    for e in range(_K3_E):
        np.testing.assert_array_equal(got[e], _expected_b_5d(src[e]))
        np.testing.assert_array_equal(
            got_scale[e], _expected_scale_4d(src_scale[e])
        )
    assert state_dict[f"{name}_scale"].dtype == DType.float8_e8m0fnu


def test_preshuffle_b_stacked_commutes_with_the_gate_up_shard() -> None:
    """Permuting then sharding on N must equal sharding then permuting.

    This is what makes the K3 gate/up flip safe: the model permutes whole
    `[E, N, K]` tensors at load and `ShardingStrategy.gate_up(axis=1)` slices
    them afterwards. The 5D layout puts `N0` outermost, so an N run on a
    16-row boundary stays a contiguous run of whole tiles -- but only on N,
    which is why the down projection cannot follow (see the K-axis case below).
    """
    name = "layers.0.block_sparse_moe.experts_gate_up_proj"
    state_dict = _stacked_state_dict(name, 20)
    src = _result_bytes(state_dict[name]).copy()

    preshuffle_block_scaled_b_stacked(state_dict, [name])
    permuted = _result_bytes(state_dict[name])

    for device in range(_K3_TP):
        rows = _k3_gate_up_rows(device)
        for e in range(_K3_E):
            np.testing.assert_array_equal(
                permuted[e][rows],
                _expected_b_5d(src[e][rows]),
                err_msg=f"gate/up shard {device} does not commute",
            )


def test_preshuffle_b_stacked_does_not_commute_with_a_k_axis_shard() -> None:
    """The negative control, without which the test above proves nothing.

    K3's down projection splits on the packed-K axis, and a K slice of
    `(N0, K0, KLane, NLane, KPack)` is strided rather than contiguous. If this
    ever starts passing, the layout changed and the gate/up-only restriction in
    `_preshuffle_gate_up_for_amd` needs revisiting rather than trusting.

    Sliced in half rather than at K3's real TP8 boundary, because 1536/8 = 192
    packed bytes is not a whole 64-byte MFMA K tile and would fail the reshape
    before it could fail the comparison -- a second, independent reason the down
    projection cannot take this path.
    """
    name = "layers.0.block_sparse_moe.experts_gate_up_proj"
    state_dict = _stacked_state_dict(name, 30)
    src = _result_bytes(state_dict[name]).copy()

    preshuffle_block_scaled_b_stacked(state_dict, [name])
    permuted = _result_bytes(state_dict[name])

    k_shard = slice(0, _K3_KB // 2)
    assert not np.array_equal(
        permuted[0][:, k_shard], _expected_b_5d(src[0][:, k_shard])
    )


def test_preshuffle_b_stacked_rejects_a_missing_scale() -> None:
    """A weight permuted without its scale is a wrong-logits bug, not a warning."""
    name = "layers.0.block_sparse_moe.experts_gate_up_proj"
    state_dict = _stacked_state_dict(name, 40)
    del state_dict[f"{name}_scale"]

    with pytest.raises(ValueError, match="permuted together"):
        preshuffle_block_scaled_b_stacked(state_dict, [name])


def test_preshuffle_b_stacked_raises_before_replacing_anything() -> None:
    """A later bad name must not leave earlier weights half permuted.

    The first name is valid and the second is missing its scale, so the
    error comes after one weight/scale pair and the second weight have
    already passed validation.
    """
    good = "layers.0.block_sparse_moe.experts_gate_up_proj"
    bad = "layers.1.block_sparse_moe.experts_gate_up_proj"
    state_dict = _stacked_state_dict(good, 60) | _stacked_state_dict(bad, 62)
    del state_dict[f"{bad}_scale"]
    before = dict(state_dict)

    with pytest.raises(ValueError, match="permuted together"):
        preshuffle_block_scaled_b_stacked(state_dict, [good, bad])

    for tensor_name, wd in before.items():
        assert state_dict[tensor_name] is wd, f"{tensor_name!r} was replaced"


def test_preshuffle_b_stacked_rejects_an_unstacked_weight() -> None:
    """The per-expert layout must not silently fall through this entry point."""
    name = "layers.0.block_sparse_moe.experts_gate_up_proj"
    raw = _weight_bytes(50, (_N, _K_BYTES))
    state_dict = {
        name: WeightData.from_numpy(raw, name),
        f"{name}_scale": WeightData.from_numpy(
            _weight_bytes(51, (_MN, _K_SCALES)), f"{name}_scale"
        ),
    }

    with pytest.raises(ValueError, match="rank-3"):
        preshuffle_block_scaled_b_stacked(state_dict, [name])
