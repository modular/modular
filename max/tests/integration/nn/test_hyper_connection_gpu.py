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
"""Validates ``max.nn.legacy.HyperConnection`` against the PyTorch reference.

Covers the whole ModuleV2 layer: the unweighted RMSNorm, the stream
projection, the fused gate op, and the stream collapse under each ``pre_mix``
state. At DeepSeek-V4 widths on bf16 streams, the site without ``pre_mix`` runs
as the fused ``mo.mhc.pre`` op on gfx950 and as the op chain elsewhere.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pytest
import torch
from max.driver import accelerator_api
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.legacy import HyperConnection
from max.nn.legacy.kernels import mhc_pre_supports
from test_common.hyper_connection_reference import hyper_connection_reference
from torch.utils.dlpack import from_dlpack

_HC_MULT = 4
_SINKHORN_ITERS = 20
_HIDDEN_SIZE = 128
_HC_EPS = 1e-6
_RMS_NORM_EPS = 1e-6
_MIX_WIDTH = 2 * _HC_MULT + _HC_MULT**2

# Row counts that straddle the kernel's four-warp block boundary.
_ROWS = [1, 3, 4, 7, 128, 4096]

# The `pre_mix` collapse is elementwise over rows, so one row count that is
# not a multiple of the gate kernel's block suffices.
_PRE_MIX_ROWS = 7

# Keep the reference projection at full float32 so the comparison is against
# the exact layer, not against torch's own reduced-precision matmul.
torch.set_float32_matmul_precision("highest")

# MAX runs the float32 stream projection at TF32 precision, which lands ~2e-3
# of absolute error in `hc_proj` and carries it through the sigmoids and the
# Sinkhorn projection, so the projection -- not the gate math -- sets what this
# test can resolve. These bounds are measured with ~2x headroom at the largest
# row count.
_GATE_TOL = {"rtol": 5e-3, "atol": 2e-3}
_COLLAPSED_TOL = {"rtol": 5e-3, "atol": 5e-3}


@pytest.fixture(scope="module")
def hc_weights() -> dict[str, torch.Tensor]:
    """The layer's three weights, on CPU for the weights registry."""
    torch.manual_seed(0)
    fan_in = _HC_MULT * _HIDDEN_SIZE
    return {
        # Scaled by 1/sqrt(fan_in) so the projection of an RMS-normalized
        # stream lands in the sigmoid's and softmax's responsive range.
        "hc_fn": torch.randn(_MIX_WIDTH, fan_in, dtype=torch.float32)
        / math.sqrt(fan_in),
        "hc_base": torch.randn(_MIX_WIDTH, dtype=torch.float32),
        "hc_scale": 2 * torch.rand(3, dtype=torch.float32),
    }


def _load_hc(
    session: InferenceSession,
    weights: dict[str, torch.Tensor],
    pre_mix: Literal["omitted", "carried", "one_hot"],
) -> Model:
    device_ref = DeviceRef.GPU()
    input_types = [
        TensorType(
            DType.float32,
            ["total_seq_len", _HC_MULT, _HIDDEN_SIZE],
            device=device_ref,
        )
    ]
    if pre_mix == "carried":
        input_types.append(
            TensorType(
                DType.float32, ["total_seq_len", _HC_MULT], device=device_ref
            )
        )
    with Graph(f"hyper_connection_{pre_mix}", input_types=input_types) as graph:
        layer = HyperConnection(
            hidden_size=_HIDDEN_SIZE,
            hc_mult=_HC_MULT,
            device=device_ref,
            hc_sinkhorn_iters=_SINKHORN_ITERS,
            hc_eps=_HC_EPS,
            rms_norm_eps=_RMS_NORM_EPS,
        )
        streams = graph.inputs[0].tensor
        if pre_mix == "omitted":
            outputs = layer(streams)
        elif pre_mix == "carried":
            outputs = layer(streams, graph.inputs[1].tensor)
        else:
            one_hot = np.eye(_HC_MULT, dtype=np.float32)[0]
            outputs = layer(
                streams,
                ops.constant(one_hot, DType.float32, device=device_ref),
            )
        graph.output(*outputs)

    return session.load(graph, weights_registry=weights)


@pytest.fixture(scope="module")
def hc_model(
    gpu_session: InferenceSession, hc_weights: dict[str, torch.Tensor]
) -> Model:
    """One compiled graph for the whole module; the row count stays symbolic."""
    return _load_hc(gpu_session, hc_weights, "omitted")


def _reference(
    streams: torch.Tensor, hc_weights: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return hyper_connection_reference(
        streams,
        hc_weights["hc_fn"].cuda(),
        hc_weights["hc_base"].cuda(),
        hc_weights["hc_scale"].cuda(),
        hc_mult=_HC_MULT,
        hc_sinkhorn_iters=_SINKHORN_ITERS,
        hc_eps=_HC_EPS,
        rms_norm_eps=_RMS_NORM_EPS,
    )


def _random_streams(rows: int) -> torch.Tensor:
    torch.manual_seed(rows)
    return torch.randn(
        rows, _HC_MULT, _HIDDEN_SIZE, dtype=torch.float32, device="cuda"
    )


@pytest.mark.parametrize("rows", _ROWS)
def test_gates_match_torch(
    hc_model: Model, hc_weights: dict[str, torch.Tensor], rows: int
) -> None:
    streams = _random_streams(rows)

    post, comb, collapsed, pre = (
        from_dlpack(t) for t in hc_model.execute(streams)
    )
    ref_post, ref_comb, ref_collapsed, ref_pre = _reference(streams, hc_weights)

    assert tuple(post.shape) == (rows, _HC_MULT)
    assert tuple(comb.shape) == (rows, _HC_MULT, _HC_MULT)
    assert tuple(collapsed.shape) == (rows, _HIDDEN_SIZE)
    assert tuple(pre.shape) == (rows, _HC_MULT)

    torch.testing.assert_close(post, ref_post, **_GATE_TOL)
    torch.testing.assert_close(comb, ref_comb, **_GATE_TOL)
    torch.testing.assert_close(pre, ref_pre, **_GATE_TOL)
    torch.testing.assert_close(collapsed, ref_collapsed, **_COLLAPSED_TOL)


def test_carried_pre_mix_weights_collapse(
    gpu_session: InferenceSession, hc_weights: dict[str, torch.Tensor]
) -> None:
    model = _load_hc(gpu_session, hc_weights, "carried")
    streams = _random_streams(_PRE_MIX_ROWS)
    pre_mix = torch.rand(
        _PRE_MIX_ROWS, _HC_MULT, dtype=torch.float32, device="cuda"
    )

    _, _, collapsed, pre = (
        from_dlpack(t) for t in model.execute(streams, pre_mix)
    )
    *_, ref_pre = _reference(streams, hc_weights)

    torch.testing.assert_close(pre, ref_pre, **_GATE_TOL)
    torch.testing.assert_close(
        collapsed, (pre_mix.unsqueeze(-1) * streams).sum(dim=1)
    )


def test_one_hot_pre_mix_selects_first_copy(
    gpu_session: InferenceSession, hc_weights: dict[str, torch.Tensor]
) -> None:
    model = _load_hc(gpu_session, hc_weights, "one_hot")
    streams = _random_streams(_PRE_MIX_ROWS)

    _, _, collapsed, _ = (from_dlpack(t) for t in model.execute(streams))

    torch.testing.assert_close(collapsed, streams[:, 0], rtol=0, atol=0)


# The fused site's gates are held to 1e-4 of float64. The op chain may run its
# float32 projection at TF32 and keeps the bounds above instead.
_FUSED_GATE_ATOL = 1e-4

# The collapsed row is rounded to bf16 on both paths: the excess over
# ``rtol * |ref|`` must stay within 1e-2 of the reference's RMS.
_FUSED_COLLAPSED_RTOL = 2.0**-7
_FUSED_COLLAPSED_ATOL_FRAC = 1e-2


@dataclass(frozen=True)
class _FusedConfig:
    hidden_size: int
    fn_dtype: DType
    rms_norm_eps: float


# DeepSeek-V4 stores the projection in float32 with rms_norm_eps 1e-6;
# DeepSeek-V4.1 uses 1e-20, which must reach the kernel unflushed.
_FUSED_CONFIGS = [
    _FusedConfig(hidden_size, DType.float32, rms_norm_eps)
    for hidden_size in (4096, 5120)
    for rms_norm_eps in (1e-6, 1e-20)
] + [_FusedConfig(4096, DType.bfloat16, 1e-6)]


def _uses_fused(cfg: _FusedConfig) -> bool:
    return accelerator_api() == "hip" and mhc_pre_supports(
        _HC_MULT,
        cfg.hidden_size,
        DType.bfloat16,
        cfg.fn_dtype,
        DeviceRef.GPU(),
    )


def _fused_weights(cfg: _FusedConfig) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(cfg.hidden_size)
    fan_in = _HC_MULT * cfg.hidden_size
    hc_fn = torch.randn(_MIX_WIDTH, fan_in, generator=gen) / math.sqrt(fan_in)
    if cfg.fn_dtype == DType.float32:
        # A projection that kept only the top 16 bits of each weight would
        # miss the 1e-4 gate.
        low_bits = hc_fn.view(torch.int32) & 0xFFFF
        assert (low_bits != 0).float().mean() > 0.99
    torch_dtype = (
        torch.float32 if cfg.fn_dtype == DType.float32 else torch.bfloat16
    )
    return {
        "hc_fn": hc_fn.to(torch_dtype),
        "hc_base": (0.85 * (2 * torch.rand(_MIX_WIDTH, generator=gen) - 1)).to(
            torch_dtype
        ),
        "hc_scale": torch.tensor([0.5, 0.5, 1.0]).to(torch_dtype),
    }


def _fused_layer(cfg: _FusedConfig, device_ref: DeviceRef) -> HyperConnection:
    return HyperConnection(
        hidden_size=cfg.hidden_size,
        hc_mult=_HC_MULT,
        device=device_ref,
        dtype=cfg.fn_dtype,
        hc_sinkhorn_iters=_SINKHORN_ITERS,
        hc_eps=_HC_EPS,
        rms_norm_eps=cfg.rms_norm_eps,
    )


def _fused_graph(cfg: _FusedConfig, with_pre_mix: bool) -> Graph:
    """The bf16 site, then the op chain on the same values as float32.

    Float32 streams never take the fused path, so the second call is the op
    chain on every GPU.
    """
    device_ref = DeviceRef.GPU()
    input_types = [
        TensorType(
            DType.bfloat16,
            ["total_seq_len", _HC_MULT, cfg.hidden_size],
            device=device_ref,
        )
    ]
    if with_pre_mix:
        input_types.append(
            TensorType(
                DType.float32, ["total_seq_len", _HC_MULT], device=device_ref
            )
        )
    name = (
        f"hyper_connection_bf16_{cfg.hidden_size}_{cfg.fn_dtype}"
        f"_{cfg.rms_norm_eps:g}_{int(with_pre_mix)}"
    )
    with Graph(name, input_types=input_types) as graph:
        layer = _fused_layer(cfg, device_ref)
        streams = graph.inputs[0].tensor
        pre_mix = graph.inputs[1].tensor if with_pre_mix else None
        site = layer(streams, pre_mix)
        post, comb, collapsed, pre = layer(
            ops.cast(streams, DType.float32), pre_mix
        )
        graph.output(
            *site, post, comb, ops.cast(collapsed, DType.bfloat16), pre
        )
    return graph


@pytest.mark.parametrize("with_pre_mix", [False, True])
@pytest.mark.parametrize("cfg", _FUSED_CONFIGS, ids=str)
def test_fused_site_in_graph_only_without_pre_mix(
    cfg: _FusedConfig, with_pre_mix: bool
) -> None:
    graph = _fused_graph(cfg, with_pre_mix)
    assert ("mo.mhc.pre" in str(graph)) == (
        _uses_fused(cfg) and not with_pre_mix
    )


@pytest.fixture(scope="module")
def fused_model(
    gpu_session: InferenceSession,
) -> Callable[[_FusedConfig], Model]:
    """Compiles each config once; the row count stays symbolic."""
    cache: dict[_FusedConfig, Model] = {}

    def get(cfg: _FusedConfig) -> Model:
        if cfg not in cache:
            cache[cfg] = gpu_session.load(
                _fused_graph(cfg, with_pre_mix=False),
                weights_registry=_fused_weights(cfg),
            )
        return cache[cfg]

    return get


def _fused_streams(cfg: _FusedConfig, rows: int) -> torch.Tensor:
    """bf16 streams whose odd rows have a mean square of ``rms_norm_eps``.

    On those rows a flushed or altered epsilon moves the RMS scale by up to
    ``sqrt(2)``, which the gates resolve.
    """
    gen = torch.Generator().manual_seed(rows)
    streams = torch.randn(rows, _HC_MULT, cfg.hidden_size, generator=gen)
    streams[1::2] *= math.sqrt(cfg.rms_norm_eps)
    return streams.to(torch.bfloat16)


def _float64_site(
    cfg: _FusedConfig,
    streams: torch.Tensor,
    weights: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The site in float64; returns post, comb, collapsed, pre.

    Runs on the host: at D = 5120 it would not fit in torch's share of the
    device memory.
    """
    hc = _HC_MULT
    hc_fn, base, scale = (
        weights[name].double() for name in ("hc_fn", "hc_base", "hc_scale")
    )
    x = streams.double()
    flat = x.flatten(start_dim=1)
    rms = torch.rsqrt(flat.square().mean(-1, keepdim=True) + cfg.rms_norm_eps)
    mix = (flat @ hc_fn.T) * rms
    pre_w, post_w, comb_w = mix.split([hc, hc, hc * hc], dim=-1)
    pre_b, post_b, comb_b = base.split([hc, hc, hc * hc])
    pre = torch.sigmoid(pre_w * scale[0] + pre_b) + _HC_EPS
    post = 2 * torch.sigmoid(post_w * scale[1] + post_b)
    comb = (comb_w * scale[2] + comb_b).view(-1, hc, hc)
    comb = torch.softmax(comb, dim=-1) + _HC_EPS
    comb = comb / (comb.sum(dim=-2, keepdim=True) + _HC_EPS)
    for _ in range(_SINKHORN_ITERS - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + _HC_EPS)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + _HC_EPS)
    collapsed = (pre.unsqueeze(-1) * x).sum(dim=1)
    return post, comb, collapsed, pre


def _gate_error(
    actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...]
) -> float:
    """The worst gate error, a NaN counting as infinite.

    The builtin ``max`` keeps a NaN only when it comes first.
    """
    errors = [
        (a.double() - e.double()).abs().max().item()
        for a, e in zip(actual, expected, strict=True)
    ]
    return math.inf if any(math.isnan(e) for e in errors) else max(errors)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_nan_gate_fails_the_check(index: int) -> None:
    """One NaN in any gate output fails the gate, wherever it sits."""
    expected = tuple(torch.rand(2, _HC_MULT) for _ in range(3))
    actual = tuple(t.clone() for t in expected)
    actual[index].view(-1)[-1] = math.nan
    assert not _gate_error(actual, expected) <= _FUSED_GATE_ATOL


def _collapsed_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """The collapsed row's excess over ``rtol * |ref|``, in units of atol."""
    assert torch.isfinite(actual).all()
    ref = expected.double()
    atol = _FUSED_COLLAPSED_ATOL_FRAC * ref.square().mean().sqrt().item()
    excess = (actual.double() - ref).abs() - _FUSED_COLLAPSED_RTOL * ref.abs()
    return excess.max().item() / atol


@pytest.mark.parametrize("rows", [1, 8, 128, 1024])
@pytest.mark.parametrize("cfg", _FUSED_CONFIGS, ids=str)
def test_fused_site_matches_float64_and_op_chain(
    fused_model: Callable[[_FusedConfig], Model],
    cfg: _FusedConfig,
    rows: int,
) -> None:
    streams = _fused_streams(cfg, rows)
    outputs = [
        from_dlpack(t).cpu() for t in fused_model(cfg).execute(streams.cuda())
    ]
    post, comb, collapsed, pre = outputs[:4]
    chain_post, chain_comb, chain_collapsed, chain_pre = outputs[4:]
    ref_post, ref_comb, ref_collapsed, ref_pre = _float64_site(
        cfg, streams, _fused_weights(cfg)
    )

    assert tuple(post.shape) == (rows, _HC_MULT)
    assert tuple(comb.shape) == (rows, _HC_MULT, _HC_MULT)
    assert tuple(collapsed.shape) == (rows, cfg.hidden_size)
    assert collapsed.dtype == torch.bfloat16
    assert tuple(pre.shape) == (rows, _HC_MULT)

    gates = (post, comb, pre)
    ref_gates = (ref_post, ref_comb, ref_pre)
    chain_gates = (chain_post, chain_comb, chain_pre)
    gate_error = _gate_error(gates, ref_gates)
    chain_gate_error = _gate_error(gates, chain_gates)
    collapsed_error = _collapsed_error(collapsed, ref_collapsed)
    chain_collapsed_error = _collapsed_error(collapsed, chain_collapsed)
    print(
        f"{cfg} rows={rows}: gates vs float64 {gate_error:.3e}, vs chain"
        f" {chain_gate_error:.3e}; collapsed/atol vs float64"
        f" {collapsed_error:.3e}, vs chain {chain_collapsed_error:.3e}"
    )

    if _uses_fused(cfg):
        assert gate_error <= _FUSED_GATE_ATOL
    else:
        for actual, expected in zip(gates, ref_gates, strict=True):
            torch.testing.assert_close(actual.double(), expected, **_GATE_TOL)
    assert chain_gate_error <= _FUSED_GATE_ATOL
    assert collapsed_error <= 1.0
    assert chain_collapsed_error <= 1.0
