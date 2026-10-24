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
"""Tests ``DeepseekV41Query.call_fused`` against ``__call__``.

One graph runs both on the same MXFP8 weights: ``__call__`` then the layer's
``kv_norm`` and a quantize of ``qr``, and ``call_fused``. The fused kernel's
sum of squares runs in a different order, so a rare byte or element may round
the other way. Everything else must agree.
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
from max.nn.kernels import quantize_dynamic_block_scaled
from max.nn.layer import Module
from max.nn.norm.rms_norm import RMSNorm
from max.pipelines.architectures.deepseekV41.layers.dspark import (
    dspark_main_proj_quant_config,
)
from max.pipelines.architectures.deepseekV41.layers.query import (
    DeepseekV41Query,
)
from max.pipelines.architectures.deepseekV41.weight_adapters import with_dtype

HIDDEN = 1024
Q_LORA = 1280
HEAD_DIM = 512
HEADS = 4
ROPE_DIM = 64
EPS = 1e-20


class _FusedAndUnfused(Module):
    def __init__(self, kv_fault: bool = False) -> None:
        super().__init__()
        self.query = DeepseekV41Query(
            HIDDEN,
            Q_LORA,
            HEAD_DIM,
            HEADS,
            EPS,
            DType.bfloat16,
            DeviceRef.CPU(),
            quant_config=dspark_main_proj_quant_config(),
        )
        self.kv_norm = RMSNorm(HEAD_DIM, DType.float32, EPS)
        self.kv_fault = kv_fault
        if kv_fault:
            # A planted fault: the fused path normalizes kv with another weight.
            self.fused_kv_norm = RMSNorm(HEAD_DIM, DType.float32, EPS)

    def __call__(
        self, x: TensorValue, freqs_cis: TensorValue
    ) -> tuple[TensorValue, ...]:
        qr, q, kv_raw = self.query(x, freqs_cis)
        ref_q8, ref_scales = quantize_dynamic_block_scaled(
            ops.reshape(qr, [-1, Q_LORA]),
            sf_vector_size=32,
            scales_type=DType.float8_e8m0fnu,
            out_type=DType.float8_e4m3fn,
        )
        ref_kv = ops.cast(self.kv_norm(kv_raw), DType.bfloat16)
        fused = self.query.call_fused(
            x,
            freqs_cis,
            self.fused_kv_norm if self.kv_fault else self.kv_norm,
        )
        return (
            ref_q8,
            ref_scales,
            q,
            ops.reshape(ref_kv, [-1, HEAD_DIM]),
            fused.qr,
            fused.qr_scales,
            fused.q,
            fused.kv,
        )


def _weights(
    rng: np.random.Generator, kv_fault: bool = False
) -> dict[str, WeightData]:
    def e4m3(shape: tuple[int, int], name: str) -> WeightData:
        # Small positive and negative E4M3 values; no NaN encodings.
        mag = rng.integers(0x30, 0x40, shape, dtype=np.uint8)
        sign = rng.integers(0, 2, shape, dtype=np.uint8) << 7
        return with_dtype(mag | sign, DType.float8_e4m3fn, name)

    def e8m0(shape: tuple[int, int], name: str) -> WeightData:
        scales = rng.integers(118, 127, shape, dtype=np.uint8)
        return with_dtype(scales, DType.float8_e8m0fnu, name)

    def f32(shape: tuple[int, ...], name: str) -> WeightData:
        values = rng.uniform(0.5, 1.5, shape).astype(np.float32)
        return with_dtype(values, DType.float32, name)

    weights: dict[str, WeightData] = {}
    for name, n, k in (
        ("query.fused_wqa_wkv", Q_LORA + HEAD_DIM, HIDDEN),
        ("query.wq_b", HEADS * HEAD_DIM, Q_LORA),
    ):
        weights[f"{name}.weight"] = e4m3((n, k), f"{name}.weight")
        weights[f"{name}.weight_scale"] = e8m0(
            (n, k // 32), f"{name}.weight_scale"
        )
    weights["query.q_norm.weight"] = f32((Q_LORA,), "query.q_norm.weight")
    weights["kv_norm.weight"] = f32((HEAD_DIM,), "kv_norm.weight")
    if kv_fault:
        weights["fused_kv_norm.weight"] = f32(
            (HEAD_DIM,), "fused_kv_norm.weight"
        )
    return weights


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    # float64: a float32 dot over millions of elements drifts by 1e-4.
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


def _build(
    session: InferenceSession, device: DeviceRef, kv_fault: bool = False
) -> Model:
    layer = _FusedAndUnfused(kv_fault)
    layer.load_state_dict(_weights(np.random.default_rng(0), kv_fault))
    graph = Graph(
        "query_fused",
        layer,
        input_types=(
            TensorType(DType.bfloat16, [1, "m", HIDDEN], device),
            TensorType(DType.float32, ["m", ROPE_DIM // 2, 2], device),
        ),
    )
    return session.load(graph, weights_registry=layer.state_dict())


def _run(model: Model, accel: Accelerator, m: int) -> list[torch.Tensor]:
    rng = np.random.default_rng(m)
    x = torch.from_numpy(
        rng.standard_normal((1, m, HIDDEN)).astype(np.float32)
    ).to(torch.bfloat16)
    angles = rng.uniform(-np.pi, np.pi, (m, ROPE_DIM // 2))
    freqs = np.stack([np.cos(angles), np.sin(angles)], -1).astype(np.float32)
    outs = model.execute(
        Buffer.from_dlpack(x).to(accel), Buffer.from_numpy(freqs).to(accel)
    )
    return [
        torch.from_dlpack(
            o.view(DType.uint8)
            if o.dtype in (DType.float8_e4m3fn, DType.float8_e8m0fnu)
            else o
        ).cpu()
        for o in outs
    ]


def _check(outs: list[torch.Tensor]) -> None:
    ref_q8, ref_scales, ref_q, ref_kv, q8, scales, q, kv = outs
    assert q8.shape == ref_q8.shape and scales.shape == ref_scales.shape
    # The same pre-rounding sum order flips about one byte in a million.
    assert (q8 != ref_q8).float().mean() < 1e-3
    assert (scales != ref_scales).float().mean() < 1e-3
    assert q.shape == ref_q.shape and kv.shape == ref_kv.shape
    assert _cos(q, ref_q) > 0.9999
    torch.testing.assert_close(kv.float(), ref_kv.float(), rtol=2**-6, atol=0)
    # The comparison must be able to fail: the outputs are not all zero.
    assert float(ref_q.float().abs().max()) > 0
    assert float(ref_kv.float().abs().max()) > 0


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
    return _build(session, device)


# Decode rows (6 a request) at 8 to 24 requests, prefill chunks, and prime,
# ragged and single-row edges around the 128-row scale tiles.
@pytest.mark.parametrize(
    "m", [1, 6, 7, 48, 96, 127, 129, 144, 577, 1031, 4093, 8168]
)
def test_call_fused_matches_call(
    model: Model, accel: Accelerator, m: int
) -> None:
    _check(_run(model, accel, m))


def test_planted_faults_fail(
    model: Model,
    session: InferenceSession,
    device: DeviceRef,
    accel: Accelerator,
) -> None:
    with pytest.raises(AssertionError):
        _check(_run(_build(session, device, kv_fault=True), accel, 129))
    # A query row moved to its neighbor's place.
    outs = _run(model, accel, 129)
    outs[6] = outs[6].roll(1, dims=1)
    with pytest.raises(AssertionError):
        _check(outs)
