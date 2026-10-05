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
"""The block-scaled FP8 EP SwiGLU is one fused quantize kernel.

`Fp8Strategy.fused_silu_quantize` emits plain graph ops (SiLU of the gate half
times the up half) feeding the row-bounded FP8 quantize. The graph compiler
must fuse them into a single kernel that stops at the rows the EP dispatch
actually received, with the scales laid out for the grouped matmul.
"""

from __future__ import annotations

import pytest
import torch
from max import driver
from max.driver import Accelerator, Buffer, LaunchTraceEntry, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.moe.quant_strategy import Fp8Strategy
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)

_MOE_DIM = 512
_GROUP = 128
_BUFFER_ROWS = 1024
_N_EXPERTS = 8
_FP8_MAX = 448.0


def _fp8_config() -> QuantConfig:
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.DYNAMIC,
            dtype=DType.float32,
            block_size=(1, _GROUP),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float32,
            block_size=(128, 128),
        ),
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        embedding_output_dtype=None,
        format=QuantFormat.BLOCKSCALED_FP8,
    )


@pytest.mark.skipif(accelerator_count() == 0, reason="requires a GPU")
@pytest.mark.parametrize("live_rows", [64, 640, _BUFFER_ROWS])
@pytest.mark.parametrize("dtype", [DType.bfloat16, DType.float32])
def test_fused_silu_quantize_is_one_bounded_kernel(
    dtype: DType, live_rows: int
) -> None:
    device = Accelerator()
    device_ref = DeviceRef.GPU(0)
    strategy = Fp8Strategy(_fp8_config(), DType.float8_e4m3fn)

    with Graph(
        "fp8_silu_quantize",
        input_types=[
            TensorType(dtype, (_BUFFER_ROWS, 2 * _MOE_DIM), device_ref),
            TensorType(DType.uint32, (_N_EXPERTS + 1,), device_ref),
            TensorType(DType.int32, (_N_EXPERTS,), device_ref),
        ],
    ) as graph:
        gate_up, expert_start, expert_ids = (v.tensor for v in graph.inputs)
        quantized, scales = strategy.fused_silu_quantize(
            gate_up,
            # Only the row prefix sum and expert ids are read.
            expert_inputs=(gate_up, gate_up, expert_start, expert_ids, gate_up),
        )
        graph.output(quantized, scales)

    model = InferenceSession(devices=[device]).load(graph)

    torch_dtype = torch.bfloat16 if dtype == DType.bfloat16 else torch.float32
    generator = torch.Generator().manual_seed(7)
    gate_up_host = (
        torch.randn(_BUFFER_ROWS, 2 * _MOE_DIM, generator=generator) * 2.0
    ).to(torch_dtype)
    # An all-zero group exercises the zero-amax case.
    gate_up_host[::7, :_GROUP] = 0
    starts = torch.tensor(
        [live_rows * i // _N_EXPERTS for i in range(_N_EXPERTS + 1)],
        dtype=torch.int64,
    )
    inputs = [
        Buffer.from_dlpack(gate_up_host).to(device),
        Buffer.from_numpy(starts.numpy().astype("uint32")).to(device),
        Buffer.from_numpy(torch.arange(_N_EXPERTS).numpy().astype("int32")).to(
            device
        ),
    ]
    model.execute(*inputs)
    with driver.launch_trace() as entries:
        outputs = model.execute(*inputs)
    kernels = [
        e.name
        for e in entries
        if e.kind == LaunchTraceEntry.OperationKind.KERNEL_LAUNCH
    ]
    assert len(kernels) == 1, kernels

    codes = (
        torch.from_dlpack(outputs[0].view(DType.uint8))
        .cpu()
        .view(torch.float8_e4m3fn)
        .float()
    )
    scales_host = torch.from_dlpack(outputs[1]).cpu()
    assert scales_host.shape[0] == _MOE_DIM // _GROUP
    assert scales_host.shape[1] >= _BUFFER_ROWS

    gate = gate_up_host[:live_rows, :_MOE_DIM].float()
    up = gate_up_host[:live_rows, _MOE_DIM:].float()
    activated = torch.nn.functional.silu(gate) * up
    groups = activated.view(live_rows, _MOE_DIM // _GROUP, _GROUP)
    expected_scales = groups.abs().amax(-1) / _FP8_MAX
    got_scales = scales_host[:, :live_rows].T
    # The bf16 intermediate of the unfused graph ops costs under 1% in a scale.
    torch.testing.assert_close(
        got_scales, expected_scales, rtol=2e-2, atol=1e-5
    )
    dequantized = codes[:live_rows].view(groups.shape) * got_scales.unsqueeze(
        -1
    )
    # E4M3 keeps 3 mantissa bits: half an ulp is at most 1/16 of the group max.
    bound = groups.abs().amax(-1, keepdim=True) * (1 / 16 + 2e-2)
    assert ((dequantized - groups).abs() <= bound).all()
