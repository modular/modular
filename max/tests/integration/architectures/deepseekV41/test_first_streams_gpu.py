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
"""Tests the fused mHC steps at the sublayers that used to run unfused.

The first sublayer's identical embedding copies (``first_streams``) and an
Engram layer's gated copies (``_replace_copies``) are held as a pending
identity write-back, so the next sublayer runs the fused step. It must leave
the copies unchanged and give the gates the unfused ``_hc_pre`` gives, up to
the fused step's TF32 projection. Both paths' normalized inputs must round
the float64 RMSNorm of the carried collapse once. The write-back before the
Engram gate and the final collapse runs ``mhc_post``, which must match the
graph-op ``hc_post`` up to one rounding, and the DSpark tap mean and the
final collapse must round the float64 result once.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.graph.weights import WeightData
from max.nn.hyper_connection import HyperConnection
from max.nn.layer import Module
from max.nn.norm.rms_norm import RMSNorm
from max.pipelines.architectures.deepseekV4.layers import (
    expand_copies,
    hc_post,
)
from max.pipelines.architectures.deepseekV41.deepseekV41 import (
    HCStreams,
    _hc_enter,
    _hc_mean,
    _replace_copies,
    first_streams,
)
from max.pipelines.architectures.deepseekV41.layers import final_collapse
from max.pipelines.architectures.deepseekV41.weight_adapters import with_dtype

HIDDEN = 5120
HC = 4
EPS = 1e-20

# Faults the comparisons must catch: each breaks the identity write-back.
FAULTS = ("post", "comb")


def _faulty(streams: HCStreams, fault: str | None) -> HCStreams:
    assert streams.post is not None and streams.comb is not None
    if fault == "post":
        return dataclasses.replace(streams, post=streams.post + 1.0)
    if fault == "comb":
        return dataclasses.replace(streams, comb=streams.comb * 0.5)
    return streams


def _gates(streams: HCStreams) -> list[TensorValue]:
    assert streams.pre is not None and streams.post is not None
    assert streams.comb is not None
    return [streams.residual, streams.pre, streams.post, streams.comb]


class _FusedAndUnfused(Module):
    def __init__(self, device: DeviceRef, fault: str | None) -> None:
        super().__init__()
        self.hc = HyperConnection(
            HIDDEN,
            HC,
            device,
            DType.float32,
            hc_sinkhorn_iters=20,
            hc_eps=1e-6,
            rms_norm_eps=EPS,
        )
        self.norm = RMSNorm(HIDDEN, DType.float32, EPS)
        self.fault = fault

    def __call__(
        self,
        x: TensorValue,
        copies: TensorValue,
        pre: TensorValue,
        post: TensorValue,
        comb: TensorValue,
    ) -> tuple[TensorValue, ...]:
        outs: list[TensorValue] = []
        # First sublayer: unfused, then fused from first_streams.
        h_ref, ref = _hc_enter(
            HCStreams(expand_copies(x, HC)), self.hc, self.norm
        )
        h, new = _hc_enter(
            _faulty(first_streams(x, HC), self.fault), self.hc, self.norm
        )
        outs += [h_ref, h, *_gates(ref), *_gates(new)]
        # Engram layer: gated copies entered unfused, then fused.
        pending = HCStreams(copies, pre=pre, out=x, post=post, comb=comb)
        h_ref, ref = _hc_enter(HCStreams(copies, pre=pre), self.hc, self.norm)
        h, new = _hc_enter(
            _faulty(_replace_copies(pending, copies), self.fault),
            self.hc,
            self.norm,
        )
        outs += [h_ref, h, *_gates(ref), *_gates(new)]
        # The write-back: graph ops, then mhc_post.
        outs += [hc_post(x, copies, post, comb), pending.materialize()]
        # The DSpark tap and the final collapse.
        outs += [_hc_mean(copies), final_collapse(copies, pre)]
        return tuple(outs)


def _norm_weight() -> torch.Tensor:
    # bf16-exact: the fused step rounds the weight to bf16.
    rng = np.random.default_rng(11)
    weight = torch.from_numpy(rng.uniform(0.5, 1.5, HIDDEN).astype(np.float32))
    return weight.to(torch.bfloat16).float()


def _weights(rng: np.random.Generator) -> dict[str, WeightData]:
    mix = 2 * HC + HC * HC

    def f32(values: np.ndarray, name: str) -> WeightData:
        return with_dtype(values.astype(np.float32), DType.float32, name)

    return {
        "hc.hc_fn": f32(
            rng.standard_normal((mix, HC * HIDDEN)) / np.sqrt(HC * HIDDEN),
            "hc.hc_fn",
        ),
        "hc.hc_base": f32(rng.uniform(-0.5, 0.5, mix), "hc.hc_base"),
        "hc.hc_scale": f32(rng.uniform(0.5, 1.5, 3), "hc.hc_scale"),
        "norm.weight": f32(_norm_weight().numpy(), "norm.weight"),
    }


def _build(
    session: InferenceSession, device: DeviceRef, fault: str | None
) -> Model:
    layer = _FusedAndUnfused(device, fault)
    layer.load_state_dict(_weights(np.random.default_rng(7)))
    graph = Graph(
        "first_streams",
        layer,
        input_types=(
            TensorType(DType.bfloat16, [1, "t", HIDDEN], device),
            TensorType(DType.bfloat16, [1, "t", HC, HIDDEN], device),
            TensorType(DType.float32, [1, "t", HC], device),
            TensorType(DType.float32, [1, "t", HC], device),
            TensorType(DType.float32, [1, "t", HC, HC], device),
        ),
    )
    return session.load(graph, weights_registry=layer.state_dict())


def _run(
    model: Model, accel: Accelerator, t: int
) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
    rng = np.random.default_rng(t)

    def normal(*shape: int) -> torch.Tensor:
        return torch.from_numpy(rng.standard_normal(shape).astype(np.float32))

    x = normal(1, t, HIDDEN).to(torch.bfloat16)
    copies = (3 * normal(1, t, HC, HIDDEN)).to(torch.bfloat16)
    pre = torch.softmax(normal(1, t, HC), -1)
    post = 2 * torch.sigmoid(normal(1, t, HC))
    comb = torch.softmax(normal(1, t, HC, HC), -2)
    outs = model.execute(
        *(Buffer.from_dlpack(v).to(accel) for v in (x, copies, pre, post, comb))
    )
    inputs = {"x": x, "copies": copies, "pre": pre}
    return [torch.from_dlpack(o).cpu() for o in outs], inputs


def _bf16_ulp(v: torch.Tensor) -> torch.Tensor:
    exponent = torch.frexp(v.abs().clamp_min(2.0**-126))[1]
    return torch.ldexp(torch.ones_like(v), exponent - 8)


def _assert_normed(
    h: torch.Tensor, streams: torch.Tensor, pre_mix: torch.Tensor
) -> None:
    """One rounding of the collapse and of ``h``, plus float32 slack, from
    the float64 RMSNorm of ``sum(pre_mix * streams)``."""
    terms = pre_mix.double().unsqueeze(-1) * streams.double()
    collapse = terms.sum(-2)
    rounded = collapse.to(torch.bfloat16).double()
    rstd = torch.rsqrt(rounded.square().mean(-1, keepdim=True) + EPS)
    scale = rstd * _norm_weight().double()
    want = collapse * scale
    tol = _bf16_ulp(want) + scale.abs() * (
        _bf16_ulp(collapse) + 2e-6 * terms.abs().sum(-2)
    )
    ratio = (h.double() - want).abs() / tol
    worst = float(ratio.max())
    if worst > 1:
        for flat in ratio.flatten().topk(4).indices.tolist():
            i = np.unravel_index(flat, ratio.shape)
            print(
                f"at {tuple(int(k) for k in i)}: got {float(h[i]):.6g},"
                f" want {float(want[i]):.6g}, collapse"
                f" {float(collapse[i]):.6g}, error/tol {float(ratio[i]):.2f}"
            )
    assert worst <= 1, f"error reaches {worst:.2f}x the tolerance"


def _check_entry(
    outs: list[torch.Tensor],
    t: int,
    streams: torch.Tensor,
    pre_mix: torch.Tensor,
) -> None:
    h_ref, h, res_ref, pre_ref, post_ref, comb_ref, res, pre, post, comb = outs
    assert res.shape == res_ref.shape == (1, t, HC, HIDDEN)
    # The identity write-back is exact: 0 * out + 1 * copy.
    assert torch.equal(res.view(torch.int16), res_ref.view(torch.int16))
    for got, want in ((pre, pre_ref), (post, post_ref), (comb, comb_ref)):
        assert got.shape == want.shape
        torch.testing.assert_close(got, want, rtol=1e-2, atol=2e-3)
    for normed in (h_ref, h):
        assert normed.shape == (1, t, HIDDEN)
        _assert_normed(normed, streams, pre_mix)
    # The comparisons must be able to fail: nothing is all zero.
    for v in (h_ref, res_ref, post_ref, comb_ref):
        assert float(v.float().abs().max()) > 0


def _first_entry(inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    x = inputs["x"]
    first = torch.zeros(*x.shape[:-1], HC)
    first[..., 0] = 1
    return x.unsqueeze(-2).expand(*x.shape[:-1], HC, HIDDEN), first


def _check(
    outs: list[torch.Tensor], inputs: dict[str, torch.Tensor], t: int
) -> None:
    copies, pre = inputs["copies"], inputs["pre"]
    _check_entry(outs[:10], t, *_first_entry(inputs))
    _check_entry(outs[10:20], t, copies, pre)
    want, got = outs[20].float(), outs[21].float()
    assert got.shape == want.shape == (1, t, HC, HIDDEN)
    torch.testing.assert_close(got, want, rtol=2**-7, atol=2**-20)
    assert float(want.abs().max()) > 0
    # Rounded once from the float64 mean and weighted sum.
    mean, collapsed = outs[22:]
    exact = copies.double()
    for got, want in (
        (mean, exact.mean(2)),
        (collapsed, (pre.double().unsqueeze(-1) * exact).sum(2)),
    ):
        assert got.shape == (1, t, HIDDEN)
        torch.testing.assert_close(got.double(), want, rtol=2**-8, atol=2**-16)


@pytest.fixture(scope="module")
def accel() -> Accelerator:
    return Accelerator()


@pytest.fixture(scope="module")
def session(accel: Accelerator) -> InferenceSession:
    return InferenceSession(devices=[accel])


@pytest.fixture(scope="module")
def device(accel: Accelerator) -> DeviceRef:
    return DeviceRef(accel.label, accel.id)


@pytest.fixture(scope="module")
def model(session: InferenceSession, device: DeviceRef) -> Model:
    return _build(session, device, None)


@pytest.mark.parametrize("t", [1, 6, 7, 48, 144, 577, 1031, 8168])
def test_fused_entries_match_unfused(
    model: Model, accel: Accelerator, t: int
) -> None:
    _check(*_run(model, accel, t), t)


def test_normed_check_sees_the_mixer(model: Model, accel: Accelerator) -> None:
    """Negative control: the normalized input must not also match the
    collapse with the copies' weights rotated."""
    outs, inputs = _run(model, accel, 129)
    rotated = inputs["pre"].roll(1, -1)
    for h in outs[10:12]:
        with pytest.raises(AssertionError):
            _assert_normed(h, inputs["copies"], rotated)


@pytest.mark.parametrize("fault", FAULTS)
def test_planted_fault_fails(
    session: InferenceSession,
    device: DeviceRef,
    accel: Accelerator,
    fault: str,
) -> None:
    faulty = _build(session, device, fault)
    outs, inputs = _run(faulty, accel, 129)
    entries = (
        (outs[:10], _first_entry(inputs)),
        (outs[10:20], (inputs["copies"], inputs["pre"])),
    )
    for entry, collapse_inputs in entries:
        with pytest.raises(AssertionError):
            _check_entry(entry, 129, *collapse_inputs)
