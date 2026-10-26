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
"""Tests the MXFP8 copy of the FFN input that the fused mHC step writes.

With ``mxfp8=True`` the step must write the same bf16 input as without, and
an MXFP8 copy byte-equal to ``quantize_dynamic_block_scaled`` of that input,
with zero scales on the padding rows. The shared expert must give the same
bytes from the copy as from quantizing its input itself.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.graph.weights import WeightData
from max.nn.hyper_connection import HyperConnection
from max.nn.kernels import quantize_dynamic_block_scaled
from max.nn.layer import Module
from max.nn.norm.rms_norm import RMSNorm
from max.pipelines.architectures.deepseekV41.layers import fused_hc_step
from max.pipelines.architectures.deepseekV41.layers.hyper_connection import (
    Mxfp8Rows,
)
from max.pipelines.architectures.deepseekV41.layers.moe import (
    DeepseekV41SharedExpert,
)
from max.pipelines.architectures.deepseekV41.weight_adapters import with_dtype

HIDDEN = 5120
HC = 4
INTER = 512
BLOCK = 32


def _quantize(x: TensorValue) -> Mxfp8Rows:
    values, scales = quantize_dynamic_block_scaled(
        x,
        sf_vector_size=BLOCK,
        scales_type=DType.float8_e8m0fnu,
        out_type=DType.float8_e4m3fn,
    )
    return Mxfp8Rows(values, scales)


class _Step(Module):
    def __init__(self, device: DeviceRef, fault: bool) -> None:
        super().__init__()
        self.hc = HyperConnection(
            HIDDEN,
            HC,
            device,
            DType.float32,
            hc_sinkhorn_iters=20,
            hc_eps=1e-6,
            rms_norm_eps=1e-20,
        )
        self.norm = RMSNorm(HIDDEN, DType.float32, 1e-6)
        self.shared = DeepseekV41SharedExpert(HIDDEN, INTER, 10.0, device)
        self.fault = fault

    def __call__(
        self,
        out: TensorValue,
        residual: TensorValue,
        post: TensorValue,
        comb: TensorValue,
        pre: TensorValue,
    ) -> tuple[TensorValue, ...]:
        args = (out, residual, post, comb, pre, self.hc, self.norm)
        h, *_, none = fused_hc_step(*args)
        assert none is None
        h_q, *_, q = fused_hc_step(*args, mxfp8=True)
        assert q is not None
        flat = ops.reshape(h, [-1, HIDDEN])
        if self.fault:
            # Twice the input: the same e4m3 values, every scale one binade up.
            q = _quantize(
                ops.cast(ops.cast(flat, DType.float32) * 2.0, h.dtype)
            )
        ref = _quantize(flat)
        return (
            h,
            h_q,
            q.values,
            q.scales,
            ref.values,
            ref.scales,
            self.shared(h),
            self.shared(h, q),
        )


def _e4m3(rng: np.random.Generator, *shape: int) -> np.ndarray:
    values = torch.from_numpy(rng.standard_normal(shape).astype(np.float32))
    return values.to(torch.float8_e4m3fn).view(torch.uint8).numpy()


def _weights(rng: np.random.Generator) -> dict[str, WeightData]:
    def fp8(name: str, n: int, k: int, scale_exp: int) -> dict[str, WeightData]:
        scales = np.full((n, k // BLOCK), 127 + scale_exp, np.uint8)
        scales += rng.integers(0, 3, scales.shape, dtype=np.uint8)
        return {
            f"{name}.weight": with_dtype(
                _e4m3(rng, n, k), DType.float8_e4m3fn, f"{name}.weight"
            ),
            f"{name}.weight_scale": with_dtype(
                scales, DType.float8_e8m0fnu, f"{name}.weight_scale"
            ),
        }

    def f32(values: np.ndarray, name: str) -> WeightData:
        return with_dtype(values.astype(np.float32), DType.float32, name)

    mix = 2 * HC + HC * HC
    return {
        "hc.hc_fn": f32(
            rng.standard_normal((mix, HC * HIDDEN)) / np.sqrt(HC * HIDDEN),
            "hc.hc_fn",
        ),
        "hc.hc_base": f32(rng.uniform(-0.5, 0.5, mix), "hc.hc_base"),
        "hc.hc_scale": f32(rng.uniform(0.5, 1.5, 3), "hc.hc_scale"),
        "norm.weight": f32(rng.uniform(0.5, 1.5, HIDDEN), "norm.weight"),
        **fp8("shared.gate_up_proj", 2 * INTER, HIDDEN, -8),
        **fp8("shared.w2", HIDDEN, INTER, -6),
    }


def _build(session: InferenceSession, device: DeviceRef, fault: bool) -> Model:
    layer = _Step(device, fault)
    layer.load_state_dict(_weights(np.random.default_rng(5)))
    graph = Graph(
        "mhc_mxfp8",
        layer,
        input_types=(
            TensorType(DType.bfloat16, [1, "t", HIDDEN], device),
            TensorType(DType.bfloat16, [1, "t", HC, HIDDEN], device),
            TensorType(DType.float32, [1, "t", HC], device),
            TensorType(DType.float32, [1, "t", HC, HC], device),
            TensorType(DType.float32, [1, "t", HC], device),
        ),
    )
    return session.load(graph, weights_registry=layer.state_dict())


def _run(model: Model, accel: Accelerator, t: int) -> list[torch.Tensor]:
    rng = np.random.default_rng(t)

    def normal(*shape: int) -> torch.Tensor:
        return torch.from_numpy(rng.standard_normal(shape).astype(np.float32))

    inputs = (
        normal(1, t, HIDDEN).to(torch.bfloat16),
        (3 * normal(1, t, HC, HIDDEN)).to(torch.bfloat16),
        2 * torch.sigmoid(normal(1, t, HC)),
        torch.softmax(normal(1, t, HC, HC), -2),
        torch.softmax(normal(1, t, HC), -1),
    )
    outs = model.execute(*(Buffer.from_dlpack(v).to(accel) for v in inputs))
    # Compare bits: bf16 as int16, fp8 and E8M0 as bytes.
    as_bits = {DType.bfloat16: DType.int16}
    results = []
    for out in outs:
        assert isinstance(out, Buffer)
        bits = as_bits.get(out.dtype, DType.uint8)
        results.append(torch.from_dlpack(out.view(bits)).cpu())
    return results


def _valid_rows(t: int, groups: int) -> torch.Tensor:
    """``[groups, 1, 32, 4, 1]``: whether each scale belongs to a row < t.

    Row ``g * 128 + m1 * 32 + m0`` sits at ``[g, :, m0, m1, :]``.
    """
    g = torch.arange(groups).view(-1, 1, 1, 1, 1)
    m0 = torch.arange(32).view(1, 1, -1, 1, 1)
    m1 = torch.arange(4).view(1, 1, 1, -1, 1)
    return g * 128 + m1 * 32 + m0 < t


def _check_copy(outs: list[torch.Tensor], t: int) -> None:
    h, h_q, values, scales, ref_values, ref_scales = outs[:6]
    assert h.shape == (1, t, HIDDEN)
    assert torch.equal(h_q, h)
    assert values.shape == ref_values.shape == (t, HIDDEN)
    assert torch.equal(values, ref_values)
    groups = (t + 127) // 128
    assert scales.shape == ref_scales.shape == (groups, HIDDEN // 128, 32, 4, 4)
    valid = _valid_rows(t, groups).expand_as(scales)
    assert torch.equal(scales[valid], ref_scales[valid])
    assert not scales[~valid].any(), "padding rows need zero scales"
    # The compares must be able to fail: the scales are not all one value.
    assert scales[valid].unique().numel() > 1


def _check_shared(outs: list[torch.Tensor], t: int) -> None:
    want, got = outs[6:]
    assert got.shape == want.shape == (1, t, HIDDEN)
    assert torch.equal(got, want)
    assert want.view(torch.bfloat16).float().abs().max() > 0


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
    return _build(session, device, fault=False)


@pytest.mark.parametrize("t", [1, 7, 48, 129, 577, 1031, 8168])
def test_mxfp8_copy_matches_quantize(
    model: Model, accel: Accelerator, t: int
) -> None:
    outs = _run(model, accel, t)
    _check_copy(outs, t)
    _check_shared(outs, t)


def test_planted_fault_fails(
    session: InferenceSession, device: DeviceRef, accel: Accelerator
) -> None:
    outs = _run(_build(session, device, fault=True), accel, 129)
    with pytest.raises(AssertionError):
        _check_copy(outs, 129)
    with pytest.raises(AssertionError):
        _check_shared(outs, 129)
