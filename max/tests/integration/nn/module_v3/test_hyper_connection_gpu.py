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
"""Validates ``max.nn.HyperConnection`` against PyTorch.

Covers the whole ModuleV3 layer -- the unweighted RMSNorm, the stream
projection, the fused gate op, and the stream collapse -- compiled ahead of
time and measured against the same reference the ModuleV2 layer uses. At the
widths ``mo.mhc.pre`` takes, the layer is also held to a float64 reference and,
where the op runs, to the op chain it replaces. GPU-only.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Literal

import pytest
import torch
from max import functional as F
from max.driver import (
    CPU,
    Accelerator,
    Device,
    accelerator_api,
    accelerator_count,
)
from max.dtype import DType
from max.nn import GemmaRMSNorm, HyperConnection, Module, RMSNorm
from max.nn import hyper_connection as hyper_connection_module
from max.nn.legacy.kernels import mhc_pre_supports
from max.tensor import default_device, default_dtype
from max.tensor.compilation import CompiledCallable
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    Replicated,
    TensorLayout,
)
from max.tensor.tensor import Tensor, TensorType
from test_common.hyper_connection_reference import hyper_connection_reference

_HC_MULT = 4
_SINKHORN_ITERS = 20
_HIDDEN_SIZE = 128
_HC_EPS = 1e-6
_RMS_NORM_EPS = 1e-6
_MIX_WIDTH = 2 * _HC_MULT + _HC_MULT**2

# Row counts that straddle the kernel's four-warp block boundary.
_ROWS = [1, 3, 4, 7, 128, 4096]

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
    """The layer's three weights, on CPU."""
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


@pytest.fixture(scope="module")
def hc_compiled(
    hc_weights: dict[str, torch.Tensor],
) -> tuple[Device, CompiledCallable[[Tensor], tuple[Tensor, Tensor, Tensor]]]:
    """One ahead-of-time compile; the row count stays symbolic."""
    device = Accelerator()
    layer = HyperConnection(
        hidden_size=_HIDDEN_SIZE,
        hc_mult=_HC_MULT,
        hc_sinkhorn_iters=_SINKHORN_ITERS,
        hc_eps=_HC_EPS,
        rms_norm_eps=_RMS_NORM_EPS,
    ).to(device)

    layer.hc_fn = Tensor(hc_weights["hc_fn"], device=device)
    layer.hc_base = Tensor(hc_weights["hc_base"], device=device)
    layer.hc_scale = Tensor(hc_weights["hc_scale"], device=device)

    return device, layer.compile(
        TensorType(
            DType.float32,
            ["total_seq_len", _HC_MULT, _HIDDEN_SIZE],
            device=device,
        )
    )


def test_parameters(hc_weights: dict[str, torch.Tensor]) -> None:
    layer = HyperConnection(hidden_size=_HIDDEN_SIZE, hc_mult=_HC_MULT)
    assert set(dict(layer.parameters)) == set(hc_weights)


@pytest.mark.parametrize("rows", _ROWS)
def test_gates_match_torch(
    hc_compiled: tuple[
        Device, CompiledCallable[[Tensor], tuple[Tensor, Tensor, Tensor]]
    ],
    hc_weights: dict[str, torch.Tensor],
    rows: int,
) -> None:
    device, compiled = hc_compiled

    torch.manual_seed(rows)
    streams = torch.randn(
        rows, _HC_MULT, _HIDDEN_SIZE, dtype=torch.float32, device="cuda"
    )

    post, comb, collapsed = (
        torch.from_dlpack(t.to(CPU()))
        for t in compiled(Tensor(streams, device=device))
    )

    ref_post, ref_comb, ref_collapsed, _ = (
        t.cpu()
        for t in hyper_connection_reference(
            streams,
            hc_weights["hc_fn"].cuda(),
            hc_weights["hc_base"].cuda(),
            hc_weights["hc_scale"].cuda(),
            hc_mult=_HC_MULT,
            hc_sinkhorn_iters=_SINKHORN_ITERS,
            hc_eps=_HC_EPS,
            rms_norm_eps=_RMS_NORM_EPS,
        )
    )

    assert tuple(post.shape) == (rows, _HC_MULT)
    assert tuple(comb.shape) == (rows, _HC_MULT, _HC_MULT)
    assert tuple(collapsed.shape) == (rows, _HIDDEN_SIZE)

    torch.testing.assert_close(post, ref_post, **_GATE_TOL)
    torch.testing.assert_close(comb, ref_comb, **_GATE_TOL)
    torch.testing.assert_close(collapsed, ref_collapsed, **_COLLAPSED_TOL)


_FUSED_HIDDEN = 4096
_FUSED_ROWS = [1, 8, 128, 1024]
# Distinct from each other and from _HC_EPS, so passing one in place of another
# moves the outputs on the near-zero stream rows of ``_streams``.
_RMS_EPS = 1e-4
_NORM_EPS = 3e-4
# The fused op is held to 1e-4 of float64 on every gate, and the bf16
# collapsed row to ``rtol * |ref|`` plus 1% of the row RMS, as the op's own
# test allows; the normed row gets twice the rtol.
_FUSED_GATE_ATOL = 1e-4
_COLLAPSED_RTOL = 2.0**-7
_NORMED_RTOL = 2.0**-6
_COLLAPSED_ATOL_FRAC = 1e-2
# The chain's rms_norm op; the fused op folds it in.
_RMS_NORM_OP = re.compile(r"\bmo\.[\w.]*rms_norm\b")

_Layer = CompiledCallable[[Tensor], tuple[Tensor, Tensor, Tensor]]


# "rms" is the plain bf16 RMSNorm the op folds. "gemma" and "rms_f32_weight"
# must stay a separate norm op after it: the op cannot apply GemmaRMSNorm's
# offset weight and multiply-before-cast, or a weight in another dtype.
_Norm = Literal["rms", "gemma", "rms_f32_weight"]


@dataclass(frozen=True)
class _Variant:
    fn_dtype: DType
    norm: _Norm | None

    @property
    def has_norm(self) -> bool:
        return self.norm is not None


_VARIANTS = [
    _Variant(fn_dtype, norm)
    for fn_dtype in (DType.float32, DType.bfloat16)
    for norm in (None, "rms")
] + [_Variant(DType.bfloat16, norm) for norm in ("gemma", "rms_f32_weight")]
# The rms_norm kernel takes its weight in the row's dtype, so a float32 weight
# on the bf16 collapsed row does not compile on any path; that variant checks
# only that the op leaves the norm alone.
_NUMERIC_VARIANTS = [v for v in _VARIANTS if v.norm != "rms_f32_weight"]


@dataclass(frozen=True)
class _Outputs:
    post: torch.Tensor
    comb: torch.Tensor
    collapsed: torch.Tensor


class _PreSite(Module[[Tensor], tuple[Tensor, Tensor, Tensor]]):
    """A site and the sublayer input norm it is handed, as a decoder holds."""

    def __init__(self, variant: _Variant) -> None:
        self.hc = HyperConnection(
            hidden_size=_FUSED_HIDDEN,
            hc_mult=_HC_MULT,
            hc_sinkhorn_iters=_SINKHORN_ITERS,
            hc_eps=_HC_EPS,
            rms_norm_eps=_RMS_EPS,
            hc_fn_dtype=variant.fn_dtype,
        )
        self.norm: RMSNorm | None = None
        weight_dtype = (
            DType.float32
            if variant.norm == "rms_f32_weight"
            else DType.bfloat16
        )
        norm_type = GemmaRMSNorm if variant.norm == "gemma" else RMSNorm
        if variant.has_norm:
            with default_dtype(weight_dtype):
                self.norm = norm_type(_FUSED_HIDDEN, eps=_NORM_EPS)

    def forward(self, streams: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.hc(streams, self.norm)


def _uses_mhc_pre(fn_dtype: DType, device: Device) -> bool:
    return accelerator_api() == "hip" and mhc_pre_supports(
        _HC_MULT, _FUSED_HIDDEN, DType.bfloat16, fn_dtype, device
    )


def _fused_weights(variant: _Variant) -> dict[str, torch.Tensor]:
    """Weights whose float32 projection keeps nonzero low bits."""
    gen = torch.Generator().manual_seed(1)
    fan_in = _HC_MULT * _FUSED_HIDDEN
    hc_fn = torch.randn(_MIX_WIDTH, fan_in, generator=gen) / math.sqrt(fan_in)
    weights = {
        "hc.hc_fn": hc_fn.to(
            torch.bfloat16
            if variant.fn_dtype == DType.bfloat16
            else torch.float32
        ),
        "hc.hc_base": 0.85 * (2 * torch.rand(_MIX_WIDTH, generator=gen) - 1),
        "hc.hc_scale": torch.tensor([0.5, 0.5, 1.0]),
    }
    if variant.has_norm:
        gamma = 1 + 0.085 * (2 * torch.rand(_FUSED_HIDDEN, generator=gen) - 1)
        if variant.norm == "gemma":
            gamma -= 1
        weights["norm.weight"] = (
            gamma
            if variant.norm == "rms_f32_weight"
            else gamma.to(torch.bfloat16)
        )
    return weights


def _input_type(device: Device) -> TensorType:
    return TensorType(
        DType.bfloat16,
        ["total_seq_len", _HC_MULT, _FUSED_HIDDEN],
        device=device,
    )


def _streams(rows: int) -> torch.Tensor:
    """Streams on the host: the test's torch GPU budget is smaller than them.

    Odd rows have a mean square of ``_RMS_EPS`` and their collapse one of the
    order of ``_NORM_EPS``, so a swapped or altered epsilon moves the RMS scale
    or the norm well past the gates. Row 2, if present, is all zero.
    """
    gen = torch.Generator().manual_seed(rows)
    streams = torch.randn(rows, _HC_MULT, _FUSED_HIDDEN, generator=gen)
    streams[1::2] *= math.sqrt(_RMS_EPS)
    streams[2:3] = 0
    return streams.to(torch.bfloat16)


def _float64_site(
    streams: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    norm_offset: float = 0.0,
) -> _Outputs:
    """The site in float64; the norm reads the bf16-rounded collapse."""
    hc = _HC_MULT
    x = streams.double()
    rows = x.shape[0]
    flat = x.reshape(rows, -1)
    rms = torch.rsqrt(flat.square().mean(-1, keepdim=True) + _RMS_EPS)
    mix = (flat @ weights["hc.hc_fn"].double().T) * rms
    pre_w, post_w, comb_w = mix.split([hc, hc, hc * hc], dim=-1)
    pre_b, post_b, comb_b = (
        weights["hc.hc_base"].double().split([hc, hc, hc * hc])
    )
    scale = weights["hc.hc_scale"].double()

    pre = torch.sigmoid(pre_w * scale[0] + pre_b) + _HC_EPS
    post = 2 * torch.sigmoid(post_w * scale[1] + post_b)
    comb = (comb_w * scale[2] + comb_b).view(rows, hc, hc)
    comb = torch.softmax(comb, dim=-1) + _HC_EPS
    comb = comb / (comb.sum(dim=-2, keepdim=True) + _HC_EPS)
    for _ in range(_SINKHORN_ITERS - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + _HC_EPS)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + _HC_EPS)
    collapsed = (pre.unsqueeze(-1) * x).sum(dim=1)
    if "norm.weight" in weights:
        collapsed = collapsed.to(torch.bfloat16).double()
        collapsed = collapsed * torch.rsqrt(
            collapsed.square().mean(-1, keepdim=True) + _NORM_EPS
        )
        collapsed = collapsed * (weights["norm.weight"].double() + norm_offset)
    return _Outputs(post, comb, collapsed)


def _outputs(outputs: tuple[Tensor, ...]) -> _Outputs:
    post, comb, collapsed = (
        torch.from_dlpack(t.to(CPU())).double() for t in outputs
    )
    return _Outputs(post, comb, collapsed)


def _worst(*errors: float) -> float:
    """Returns the largest error, a NaN counting as infinite.

    The builtin ``max`` keeps a NaN only when it comes first.
    """
    return math.inf if any(math.isnan(e) for e in errors) else max(errors)


def _errors(
    variant: _Variant, actual: _Outputs, expected: _Outputs
) -> tuple[float, float]:
    """Returns the worst gate error and the collapsed error over its atol."""
    gate = _worst(
        (actual.post - expected.post).abs().max().item(),
        (actual.comb - expected.comb).abs().max().item(),
    )
    rtol = _NORMED_RTOL if variant.has_norm else _COLLAPSED_RTOL
    ref = expected.collapsed
    atol = _COLLAPSED_ATOL_FRAC * ref.square().mean().sqrt().item()
    excess = (actual.collapsed - ref).abs() - rtol * ref.abs()
    if not torch.isfinite(actual.collapsed).all():
        return gate, math.inf
    return gate, excess.max().item() / atol


@pytest.mark.parametrize("output", ["post", "comb", "collapsed"])
def test_nan_output_fails_the_check(output: str) -> None:
    """One NaN in any output fails the gates, wherever it sits."""
    expected = _Outputs(
        post=torch.rand(2, _HC_MULT, dtype=torch.float64),
        comb=torch.rand(2, _HC_MULT, _HC_MULT, dtype=torch.float64),
        collapsed=torch.rand(2, 8, dtype=torch.float64),
    )
    actual = _Outputs(**{k: v.clone() for k, v in vars(expected).items()})
    getattr(actual, output).view(-1)[-1] = math.nan
    gate, collapsed = _errors(_VARIANTS[0], actual, expected)
    assert not (gate <= _FUSED_GATE_ATOL and collapsed <= 1.0)


@contextmanager
def _op_chain_only() -> Iterator[None]:
    """Builds the op chain whatever the device, for the fused path's parity."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            hyper_connection_module, "mhc_pre_supports", lambda *args: False
        )
        yield


@pytest.fixture(scope="module")
def fused_compiled() -> Callable[[_Variant, bool], _Layer]:
    """Compiles each variant once, by default and as the op chain."""
    device = Accelerator()
    cache: dict[tuple[_Variant, bool], _Layer] = {}

    def get(variant: _Variant, op_chain: bool) -> _Layer:
        if (variant, op_chain) not in cache:
            site = _PreSite(variant).to(device)
            with _op_chain_only() if op_chain else nullcontext():
                cache[variant, op_chain] = site.compile(
                    _input_type(device), weights=_fused_weights(variant)
                )
        return cache[variant, op_chain]

    return get


@pytest.mark.parametrize("variant", _VARIANTS, ids=str)
def test_fused_site_in_graph(variant: _Variant) -> None:
    device = Accelerator()
    site = _PreSite(variant)
    graph = str(site.trace(_input_type(device)))
    uses_kernel = _uses_mhc_pre(variant.fn_dtype, device)
    assert ("mo.mhc.pre" in graph) == uses_kernel
    # The op replaces the projection and the gate op; the chain keeps both.
    assert ("mo.hyper_connection.gates" in graph) != uses_kernel
    # The op applies a plain RMSNorm itself; any other norm runs after it.
    has_rms_norm_op = _RMS_NORM_OP.search(graph) is not None
    assert has_rms_norm_op == (
        variant.has_norm and not (uses_kernel and variant.norm == "rms")
    )

    with _op_chain_only():
        chain = str(site.trace(_input_type(device)))
    assert "mo.mhc.pre" not in chain
    assert (_RMS_NORM_OP.search(chain) is not None) == variant.has_norm


@pytest.mark.parametrize("rows", _FUSED_ROWS)
@pytest.mark.parametrize("variant", _NUMERIC_VARIANTS, ids=str)
def test_fused_site_matches_float64_and_op_chain(
    fused_compiled: Callable[[_Variant, bool], _Layer],
    variant: _Variant,
    rows: int,
) -> None:
    device = Accelerator()
    weights = _fused_weights(variant)
    streams = _streams(rows)
    expected = _float64_site(
        streams, weights, norm_offset=1.0 if variant.norm == "gemma" else 0.0
    )
    streams_in = Tensor(streams, device=device)
    actual = _outputs(fused_compiled(variant, False)(streams_in))
    assert tuple(actual.comb.shape) == (rows, _HC_MULT, _HC_MULT)
    assert tuple(actual.collapsed.shape) == (rows, _FUSED_HIDDEN)

    gate, collapsed = _errors(variant, actual, expected)
    label = f"{variant} T={rows}"
    print(f"{label} vs fp64: gates {gate:.3e} collapsed/atol {collapsed:.3e}")
    uses_kernel = _uses_mhc_pre(variant.fn_dtype, device)
    # The op chain's float32 projection may run at TF32.
    assert gate <= (_FUSED_GATE_ATOL if uses_kernel else _GATE_TOL["atol"])
    assert collapsed <= 1.0
    if not uses_kernel:
        return

    chain = _outputs(fused_compiled(variant, True)(streams_in))
    gate, collapsed = _errors(variant, actual, chain)
    print(f"{label} vs chain: gates {gate:.3e} collapsed/atol {collapsed:.3e}")
    assert gate <= 2 * _FUSED_GATE_ATOL
    assert collapsed <= 1.0


@pytest.mark.skipif(accelerator_count() < 2, reason="needs two GPUs")
def test_fused_site_on_replicated_mesh() -> None:
    """Each device runs the site on its replica and keeps its placement."""
    mesh = DeviceMesh((Accelerator(0), Accelerator(1)), (2,), ("tp",))
    replicated = DeviceMapping(mesh, (Replicated(),))
    variant = _Variant(DType.bfloat16, norm="rms")
    weights = _fused_weights(variant)
    with default_device(mesh):
        site = _PreSite(variant)
    input_type = TensorLayout(
        DType.bfloat16, ["total_seq_len", _HC_MULT, _FUSED_HIDDEN], replicated
    )
    uses_kernel = _uses_mhc_pre(variant.fn_dtype, mesh.devices[0])
    assert str(site.trace(input_type)).count("mo.mhc.pre") == (
        2 if uses_kernel else 0
    )

    compiled = site.compile(input_type, weights=weights)
    tolerance = _FUSED_GATE_ATOL if uses_kernel else _GATE_TOL["atol"]
    for rows in (8, 1024):
        streams = _streams(rows)
        expected = _float64_site(streams, weights)
        outputs = compiled(
            F.transfer_to(Tensor(streams, device=mesh.devices[0]), replicated)
        )
        for out in outputs:
            assert out.placements == (Replicated(),)
        for shard in range(mesh.num_devices):
            actual = _outputs(tuple(out.local_shards[shard] for out in outputs))
            gate, collapsed = _errors(variant, actual, expected)
            print(
                f"mesh device {shard} T={rows}: gates {gate:.3e}"
                f" collapsed/atol {collapsed:.3e}"
            )
            assert gate <= tolerance
            assert collapsed <= 1.0
