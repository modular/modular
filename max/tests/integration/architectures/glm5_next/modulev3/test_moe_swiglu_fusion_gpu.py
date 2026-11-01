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
"""GLM-5.3-Flash's EP expert chain fuses its clamped SwiGLU into the quantize.

The FP8 gate/up grouped matmul feeds the clamp and SwiGLU, which must fuse
into the row-bounded FP8 quantize of the down-projection input. Anything
between them runs over the whole EP receive buffer, not just the rows the
dispatch received, so the chain must be exactly two kernels.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max import driver
from max.driver import (
    CPU,
    Accelerator,
    Buffer,
    LaunchTraceEntry,
    accelerator_count,
)
from max.dtype import DType
from max.nn.common_layers import quant_ops
from max.nn.common_layers.quant_tensor import (
    FP8BlockActivation,
    FP8BlockTensor,
)
from max.pipelines.architectures.glm5_next_modulev3.layers.mlp import (
    Glm5NextExpertParallelMoE,
)
from max.tensor import Tensor
from max.tensor.executor import CompilingExecutor, set_default_executor
from max.tensor.realization_context import lazy

_BUFFER_ROWS = 8192
_HIDDEN = 1024
_MOE_DIM = 2048
_GROUP = 128
_WEIGHT_BLOCK = 128
_SWIGLU_LIMIT = 10.0
_FP8_MAX = 448.0
# Two decode tokens on each of four ranks, top-8 routed: 64 routed rows,
# 16 of them on this rank.
_DECODE_ROWS_PER_EXPERT = [1, 3, 2, 2, 1, 4, 2, 1]
_PREFILL_ROWS_PER_EXPERT = [512] * len(_DECODE_ROWS_PER_EXPERT)
_N_EXPERTS = len(_DECODE_ROWS_PER_EXPERT)
# The model's decode grid.y cap; only the MI355 FP8 kernel reads it.
_DECODE_GRID_M_CAP = 1


def _fp8(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.float8_e4m3fn)


def _to_device_fp8(t: torch.Tensor, device: Accelerator) -> Tensor:
    data = Buffer.from_numpy(t.view(torch.uint8).numpy()).to(device)
    return Tensor(storage=data.view(DType.float8_e4m3fn))


def _to_device(t: np.ndarray, device: Accelerator | CPU) -> Tensor:
    return Tensor(storage=Buffer.from_numpy(t).to(device))


@pytest.mark.skipif(accelerator_count() == 0, reason="requires a GPU")
@pytest.mark.parametrize(
    "rows_per_expert",
    [_DECODE_ROWS_PER_EXPERT, _PREFILL_ROWS_PER_EXPERT],
    ids=["decode", "prefill"],
)
def test_ep_gate_up_swiglu_quantize_is_two_kernels(
    rows_per_expert: list[int],
) -> None:
    # A float32 gate/up output is read through two slices by the quantize
    # and cannot fuse into it.
    assert Glm5NextExpertParallelMoE.gate_up_accum_dtype is None
    device = Accelerator()
    generator = torch.Generator().manual_seed(3881)
    live_rows = sum(rows_per_expert)
    starts = np.cumsum([0, *rows_per_expert]).astype(np.uint32)

    hidden = _fp8(torch.randn(_BUFFER_ROWS, _HIDDEN, generator=generator))
    hidden_scales = 0.5 + 0.5 * torch.rand(
        _HIDDEN // _GROUP, _BUFFER_ROWS, generator=generator
    )
    weight = _fp8(
        torch.randn(_N_EXPERTS, 2 * _MOE_DIM, _HIDDEN, generator=generator)
        * 0.5
    )
    weight_scales = 0.5 + 0.5 * torch.rand(
        _N_EXPERTS,
        2 * _MOE_DIM // _WEIGHT_BLOCK,
        _HIDDEN // _WEIGHT_BLOCK,
        generator=generator,
    )

    tokens = FP8BlockActivation(
        data=_to_device_fp8(hidden, device),
        scales=_to_device(hidden_scales.numpy(), device),
    )
    gate_up = FP8BlockTensor(
        weight=_to_device_fp8(weight, device),
        weight_scale_inv=_to_device(weight_scales.numpy(), device),
    )
    # Only the type matters: it selects the FP8 requant of the SwiGLU output.
    down = FP8BlockTensor.zeros((256, _MOE_DIM)).to(device)
    expert_start = _to_device(starts, device)
    expert_ids = _to_device(np.arange(_N_EXPERTS, dtype=np.int32), device)
    usage_stats = _to_device(
        np.array([max(rows_per_expert), _N_EXPERTS], dtype=np.uint32), device
    )
    estimated_total_m = _to_device(np.array(live_rows, dtype=np.uint32), CPU())

    def run() -> FP8BlockActivation:
        # Compiled like the model graph. Only the result outlives the lazy
        # scope, so the gate/up output is an intermediate the compiler is
        # free to fuse away.
        with set_default_executor(CompilingExecutor()), lazy():
            out = quant_ops.grouped_matmul_silu(
                tokens,
                gate_up,
                down,
                expert_start,
                expert_ids,
                usage_stats,
                quant_config=None,
                estimated_total_m=estimated_total_m,
                accum_dtype=Glm5NextExpertParallelMoE.gate_up_accum_dtype,
                decode_grid_m_cap=_DECODE_GRID_M_CAP,
                swiglu_limit=_SWIGLU_LIMIT,
            )
        assert isinstance(out, FP8BlockActivation)
        # Realizes the result inside the launch-trace window.
        torch.from_dlpack(out.scales)
        return out

    run()
    with driver.launch_trace() as entries:
        result = run()
    kernels = [
        e.name
        for e in entries
        if e.kind == LaunchTraceEntry.OperationKind.KERNEL_LAUNCH
    ]
    assert len(kernels) == 2, kernels
    # The grouped matmul: AMD names it grouped_matmul, SM100 gmm.
    assert any(n in kernels[0] for n in ("grouped_matmul", "_gmm_")), kernels
    assert "fp8_quantize_" in kernels[1], kernels

    hidden_deq = hidden[:live_rows].double() * hidden_scales[
        :, :live_rows
    ].T.double().repeat_interleave(_GROUP, dim=1)
    weight_deq = weight.double() * weight_scales.double().repeat_interleave(
        _WEIGHT_BLOCK, dim=1
    ).repeat_interleave(_WEIGHT_BLOCK, dim=2)
    gate_up_ref = torch.cat(
        [
            hidden_deq[starts[e] : starts[e + 1]] @ weight_deq[e].T
            for e in range(_N_EXPERTS)
        ]
    ).to(torch.bfloat16)
    gate = gate_up_ref[:, :_MOE_DIM].float().clamp(max=_SWIGLU_LIMIT)
    up = gate_up_ref[:, _MOE_DIM:].float().clamp(-_SWIGLU_LIMIT, _SWIGLU_LIMIT)
    groups = (torch.nn.functional.silu(gate) * up).view(
        live_rows, _MOE_DIM // _GROUP, _GROUP
    )

    codes = torch.from_dlpack(result.data.cast(DType.float32)).cpu()
    scales = torch.from_dlpack(result.scales).cpu()
    got_scales = scales[:, :live_rows].T
    group_amax = groups.abs().amax(-1)
    torch.testing.assert_close(
        got_scales, group_amax / _FP8_MAX, rtol=2e-2, atol=1e-5
    )
    dequantized = codes[:live_rows].view(groups.shape) * got_scales.unsqueeze(
        -1
    )
    # The e4m3 rounding step plus slack, as in test_fp8_silu_quantize_gpu.py.
    # This checks the quantize contract; it cannot tell a bf16 SwiGLU from a
    # float32 one, since the reference rounds gate/up to bf16 too.
    bound = group_amax.unsqueeze(-1) * (1 / 16 + 2e-2)
    assert ((dequantized - groups).abs() <= bound).all()
