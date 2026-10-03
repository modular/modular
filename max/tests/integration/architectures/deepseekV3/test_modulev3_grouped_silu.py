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
"""ModuleV3 `grouped_silu` on an FP8 down projection is one bounded kernel.

The SwiGLU is plain graph ops feeding the row-bounded FP8 quantize and must
fuse into it, stopping at the rows the EP dispatch received.
"""

from __future__ import annotations

import pytest
import torch
from max import driver
from max.driver import Accelerator, LaunchTraceEntry, accelerator_count
from max.dtype import DType
from max.experimental.tensor import Tensor
from max.pipelines.architectures.deepseekV3_modulev3.layers import quant_ops
from max.pipelines.architectures.deepseekV3_modulev3.layers.quant_tensor import (
    FP8BlockTensor,
)

_MOE_DIM = 512
_GROUP = 128
_BUFFER_ROWS = 1024
_N_EXPERTS = 8
_FP8_MAX = 448.0


@pytest.mark.skipif(accelerator_count() == 0, reason="requires a GPU")
@pytest.mark.parametrize("live_rows", [64, 640])
def test_grouped_silu_fp8_is_one_bounded_kernel(live_rows: int) -> None:
    device = Accelerator()
    generator = torch.Generator().manual_seed(11)
    gate_up_host = (
        torch.randn(_BUFFER_ROWS, 2 * _MOE_DIM, generator=generator) * 2.0
    ).to(torch.bfloat16)
    starts = torch.tensor(
        [live_rows * i // _N_EXPERTS for i in range(_N_EXPERTS + 1)],
        dtype=torch.int64,
    )
    gate_up = Tensor.from_dlpack(gate_up_host).to(device)
    expert_start = Tensor.from_dlpack(starts.numpy().astype("uint32")).to(
        device
    )
    down = FP8BlockTensor.zeros((256, _MOE_DIM)).to(device)

    def run() -> FP8BlockTensor:
        out = quant_ops.grouped_silu(gate_up, expert_start, down)
        assert isinstance(out, FP8BlockTensor)
        # Reading a result realizes the pending eager graph.
        torch.from_dlpack(out.weight_scale_inv)
        return out

    result = run()
    with driver.launch_trace() as entries:
        result = run()
    kernels = [
        e.name
        for e in entries
        if e.kind == LaunchTraceEntry.OperationKind.KERNEL_LAUNCH
    ]
    assert sum("fp8_quantization" in name for name in kernels) == 1, kernels
    assert not any("silu" in name.lower() for name in kernels), kernels

    codes = torch.from_dlpack(result.data.cast(DType.float32)).cpu()
    scales = torch.from_dlpack(result.weight_scale_inv).cpu()
    gate = gate_up_host[:live_rows, :_MOE_DIM].float()
    up = gate_up_host[:live_rows, _MOE_DIM:].float()
    groups = (torch.nn.functional.silu(gate) * up).view(
        live_rows, _MOE_DIM // _GROUP, _GROUP
    )
    got_scales = scales[:, :live_rows].T
    torch.testing.assert_close(
        got_scales, groups.abs().amax(-1) / _FP8_MAX, rtol=2e-2, atol=1e-5
    )
    dequantized = codes[:live_rows].view(groups.shape) * got_scales.unsqueeze(
        -1
    )
    bound = groups.abs().amax(-1, keepdim=True) * (1 / 16 + 2e-2)
    assert ((dequantized - groups).abs() <= bound).all()
