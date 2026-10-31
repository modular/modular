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
"""The non-EP FP8 MoE runs its grouped matmuls without a host sync on SM100.

`MoEQuantized` hands the grouped FP8 matmul host-side shape bounds in place of
the device-resident stats from `moe_create_indices`. On SM100 that must give
the same result as the old host copy, enqueue no device-to-host copy, and be
capturable into a device graph. The grouped matmul itself must still honour a
device stats tensor that reports fewer active groups than there are slots.
"""

from __future__ import annotations

import pytest
import torch
from max import driver
from max.driver import Accelerator, Buffer, LaunchTraceEntry, accelerator_count
from max.driver.driver import accelerator_architecture_name
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, Shape, TensorType
from max.graph.weights import WeightData
from max.nn.legacy import kernels
from max.nn.legacy.moe import MoEQuantized
from max.nn.legacy.moe.quant_strategy import Fp8Strategy
from max.nn.legacy.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)

_K = 512
_N = 384
_BLOCK = 128
_FP8_MAX = 448.0

# Row counts of the active expert groups, and how many routing slots are
# allocated. The totals are 40 and 19. These scales are hand-built and
# unpadded, so 19 rows break the 16-byte A-scale stride and send the persistent
# kernel to its naive fallback, which does read the host bound. Real callers
# cannot reach it: the quantize op pads the scales' row dimension.
_ACTIVE_COUNTS = {"aligned": [16, 24], "prime_total": [7, 12]}
_SLOTS = 8
_POISON_OFFSET = 0xFFFFFF00
_POISON_ID = 1 << 30


pytestmark = pytest.mark.skipif(
    accelerator_count() == 0
    or not accelerator_architecture_name().startswith("sm_10"),
    reason="requires an SM100-class (Blackwell datacenter) GPU",
)


def _fp8_config() -> QuantConfig:
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.DYNAMIC,
            dtype=DType.float32,
            block_size=(1, _BLOCK),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float32,
            block_size=(_BLOCK, _BLOCK),
        ),
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        embedding_output_dtype=None,
        format=QuantFormat.BLOCKSCALED_FP8,
    )


def _dtoh_copies(entries: list[LaunchTraceEntry]) -> list[str]:
    return [
        e.name
        for e in entries
        if e.kind == LaunchTraceEntry.OperationKind.MEMCPY
        and e.memcpy_kind == LaunchTraceEntry.MemcpyKind.DTOH
    ]


# --- Grouped matmul with fewer active groups than routing slots -------------


def _build_grouped_matmul(
    session: InferenceSession, total_m: int, host_stats: bool
) -> Model:
    gpu = DeviceRef.GPU(0)
    stats_device = DeviceRef.CPU() if host_stats else gpu
    with Graph(
        "fp8_grouped_matmul_fewer_active_groups",
        input_types=[
            TensorType(DType.float8_e4m3fn, [total_m, _K], gpu),
            TensorType(DType.float32, [_K // _BLOCK, total_m], gpu),
            TensorType(DType.float8_e4m3fn, [_SLOTS, _N, _K], gpu),
            TensorType(
                DType.float32, [_SLOTS, _N // _BLOCK, _K // _BLOCK], gpu
            ),
            TensorType(DType.uint32, [_SLOTS + 1], gpu),
            TensorType(DType.int32, [_SLOTS], gpu),
            TensorType(DType.uint32, [2], stats_device),
        ],
    ) as graph:
        hidden, scales, weight, weight_scales, starts, ids, stats = (
            v.tensor for v in graph.inputs
        )
        graph.output(
            Fp8Strategy(_fp8_config(), DType.float8_e4m3fn).grouped_matmul(
                weight,
                weight_scales,
                expert_inputs=(hidden, scales, starts, ids, stats),
            )
        )
    return session.load(graph)


def _problem(counts: list[int]) -> dict[str, torch.Tensor]:
    """Builds `len(counts)` active groups in `_SLOTS` slots; the rest is poison."""
    generator = torch.Generator().manual_seed(11)
    total_m = sum(counts)
    activations = torch.randn(total_m, _K, generator=generator)
    groups = activations.view(total_m, _K // _BLOCK, _BLOCK)
    a_scales = groups.abs().amax(-1).clamp(min=1e-4) / _FP8_MAX
    a_fp8 = (groups / a_scales.unsqueeze(-1)).to(torch.float8_e4m3fn)
    weight = (torch.randn(_SLOTS, _N, _K, generator=generator) * 0.5).to(
        torch.float8_e4m3fn
    )
    weight_scales = (
        torch.rand(_SLOTS, _N // _BLOCK, _K // _BLOCK, generator=generator)
        * 0.02
        + 0.005
    )
    active = len(counts)
    starts = torch.full((_SLOTS + 1,), _POISON_OFFSET, dtype=torch.int64)
    starts[: active + 1] = torch.tensor(
        [0, *torch.tensor(counts).cumsum(0).tolist()]
    )
    ids = torch.full((_SLOTS,), _POISON_ID, dtype=torch.int32)
    ids[:active] = torch.tensor([3, 5][:active], dtype=torch.int32)
    return {
        "a": a_fp8.view(total_m, _K),
        "a_scales": a_scales.transpose(0, 1).contiguous(),
        "w": weight,
        "w_scales": weight_scales,
        "starts": starts,
        "ids": ids,
        "active": torch.tensor(active),
        "stats": torch.tensor([max(counts), active], dtype=torch.int64),
    }


def _inputs(
    p: dict[str, torch.Tensor], device: Accelerator, host: bool
) -> list[Buffer]:
    def gpu(t: torch.Tensor, dtype: DType | None = None) -> Buffer:
        buf = Buffer.from_dlpack(t)
        if dtype is not None:
            buf = buf.view(dtype)
        return buf.to(device)

    stats = Buffer.from_numpy(p["stats"].numpy().astype("uint32"))
    return [
        gpu(p["a"].view(torch.uint8), DType.float8_e4m3fn),
        gpu(p["a_scales"]),
        gpu(p["w"].view(torch.uint8), DType.float8_e4m3fn),
        gpu(p["w_scales"]),
        Buffer.from_numpy(p["starts"].numpy().astype("uint32")).to(device),
        Buffer.from_numpy(p["ids"].numpy()).to(device),
        stats if host else stats.to(device),
    ]


def _reference(p: dict[str, torch.Tensor]) -> torch.Tensor:
    a = p["a"].float()
    out = torch.zeros(a.shape[0], _N)
    for group in range(int(p["active"])):
        lo, hi = int(p["starts"][group]), int(p["starts"][group + 1])
        expert = int(p["ids"][group])
        weight = p["w"][expert].float()
        acc = torch.zeros(hi - lo, _N)
        for kb in range(_K // _BLOCK):
            cols = slice(kb * _BLOCK, (kb + 1) * _BLOCK)
            part = a[lo:hi, cols] @ weight[:, cols].T
            row_scale = p["w_scales"][expert][:, kb].repeat_interleave(_BLOCK)
            acc += part * p["a_scales"][kb, lo:hi, None] * row_scale[None, :]
        out[lo:hi] = acc
    return out


@pytest.mark.parametrize("case", list(_ACTIVE_COUNTS))
def test_device_stats_with_fewer_active_groups_than_slots(case: str) -> None:
    device = Accelerator()
    session = InferenceSession(devices=[device])
    problem = _problem(_ACTIVE_COUNTS[case])
    total_m = int(problem["a"].shape[0])

    device_model = _build_grouped_matmul(session, total_m, host_stats=False)
    host_model = _build_grouped_matmul(session, total_m, host_stats=True)
    got = device_model.execute(*_inputs(problem, device, host=False))[0]
    want = host_model.execute(*_inputs(problem, device, host=True))[0]
    got_t = torch.from_dlpack(got).cpu()
    assert torch.equal(got_t, torch.from_dlpack(want).cpu())
    torch.testing.assert_close(
        got_t.float(), _reference(problem), rtol=3e-2, atol=3e-2
    )


# --- The non-EP FP8 MoE layer ------------------------------------------------

_HIDDEN = 256
_MOE_DIM = 128
_EXPERTS = 4
_TOP_K = 2


def _moe_weights() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(3)
    raw: dict[str, torch.Tensor] = {
        "gate.gate_score.weight": (
            torch.randn(_EXPERTS, _HIDDEN, generator=generator) * 0.5
        ).to(torch.bfloat16)
    }
    for expert in range(_EXPERTS):
        for name, (out, inp) in {
            "gate_proj": (_MOE_DIM, _HIDDEN),
            "up_proj": (_MOE_DIM, _HIDDEN),
            "down_proj": (_HIDDEN, _MOE_DIM),
        }.items():
            prefix = f"experts.{expert}.{name}"
            raw[f"{prefix}.weight"] = (
                torch.randn(out, inp, generator=generator) * 0.2
            ).to(torch.float8_e4m3fn)
            raw[f"{prefix}.weight_scale"] = (
                torch.rand(out // _BLOCK, inp // _BLOCK, generator=generator)
                * 0.02
                + 0.005
            )
    return raw


def _build_moe(
    session: InferenceSession, tokens: int | str, force_host_copy: bool
) -> Model:
    moe = MoEQuantized(
        devices=[DeviceRef.GPU()],
        hidden_dim=_HIDDEN,
        num_experts=_EXPERTS,
        num_experts_per_token=_TOP_K,
        moe_dim=_MOE_DIM,
        dtype=DType.float8_e4m3fn,
        quant_config=_fp8_config(),
    )
    moe.load_state_dict(
        {
            key: WeightData(
                Buffer.from_dlpack(value.view(torch.uint8)).view(
                    DType.float8_e4m3fn
                ),
                key,
                DType.float8_e4m3fn,
                Shape(value.shape),
            )
            if value.dtype == torch.float8_e4m3fn
            else value
            for key, value in _moe_weights().items()
        }
    )
    saved = kernels._is_sm10x_gpu
    if force_host_copy:
        kernels._is_sm10x_gpu = lambda: False
    try:
        with Graph(
            "fp8_moe_no_host_sync",
            input_types=[
                TensorType(
                    DType.bfloat16, [tokens, _HIDDEN], device=DeviceRef.GPU()
                )
            ],
        ) as graph:
            graph.output(moe(graph.inputs[0].tensor))
    finally:
        kernels._is_sm10x_gpu = saved
    return session.load(graph, weights_registry=moe.state_dict())


# 8 tokens give 16 routed rows and 7 give 14, which is not a multiple of the
# quantize op's 4-row scale padding.
@pytest.mark.parametrize("tokens", [8, 7])
def test_moe_has_no_device_to_host_copy_and_matches_the_host_copy(
    tokens: int,
) -> None:
    device = Accelerator()
    session = InferenceSession(devices=[device])
    # Compile before the first device allocation so the graphs can be
    # precompiled on a CPU worker.
    model = _build_moe(session, tokens, force_host_copy=False)
    copying = _build_moe(session, tokens, force_host_copy=True)
    x = Buffer.from_dlpack(
        torch.randn(tokens, _HIDDEN, dtype=torch.bfloat16)
    ).to(device)

    got = torch.from_dlpack(model.execute(x)[0]).cpu().clone()
    want = torch.from_dlpack(copying.execute(x)[0]).cpu()
    assert torch.isfinite(got.float()).all()
    assert torch.equal(got, want)

    with driver.launch_trace() as entries:
        model.execute(x)
    assert _dtoh_copies(entries) == []
    # The control: the old arm does copy the stats back.
    with driver.launch_trace() as entries:
        copying.execute(x)
    assert _dtoh_copies(entries) != []


def test_moe_captures_into_a_device_graph() -> None:
    device = Accelerator()
    session = InferenceSession(devices=[device])
    model = _build_moe(session, 8, force_host_copy=False)
    x = Buffer.from_dlpack(torch.randn(8, _HIDDEN, dtype=torch.bfloat16)).to(
        device
    )
    eager = torch.from_dlpack(model.execute(x)[0]).cpu().clone()
    captured = model.capture(8, x)
    model.replay(8, x)
    device.synchronize()
    assert torch.equal(torch.from_dlpack(captured[0]).cpu(), eager)


def test_moe_with_a_symbolic_token_dim_executes_and_captures() -> None:
    """The decode graphs are built over a symbolic token dim."""
    device = Accelerator()
    session = InferenceSession(devices=[device])
    model = _build_moe(session, "tokens", force_host_copy=False)
    copying = _build_moe(session, "tokens", force_host_copy=True)

    for tokens in (8, 7, 1):
        x = Buffer.from_dlpack(
            torch.randn(tokens, _HIDDEN, dtype=torch.bfloat16)
        ).to(device)
        got = torch.from_dlpack(model.execute(x)[0]).cpu().clone()
        assert torch.equal(got, torch.from_dlpack(copying.execute(x)[0]).cpu())
        with driver.launch_trace() as entries:
            model.execute(x)
        assert _dtoh_copies(entries) == []

    x = Buffer.from_dlpack(torch.randn(8, _HIDDEN, dtype=torch.bfloat16)).to(
        device
    )
    eager = torch.from_dlpack(model.execute(x)[0]).cpu().clone()
    captured = model.capture(8, x)
    model.replay(8, x)
    device.synchronize()
    assert torch.equal(torch.from_dlpack(captured[0]).cpu(), eager)
