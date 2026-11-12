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
"""Tests the Laguna router's GPU kernel route against its CPU top-k chain."""

from __future__ import annotations

import torch
from max import functional as F
from max.driver import CPU, Accelerator, Buffer, Device
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.pipelines.architectures.laguna_modulev3.layers.moe_gate import (
    LagunaTopKRouter,
)
from max.tensor import default_dtype

NUM_EXPERTS = 256
TOP_K = 10
NUM_TOKENS = 64


def _route(
    device: Device, logits: torch.Tensor, bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    with F.lazy(), default_dtype(DType.bfloat16):
        router = LagunaTopKRouter(
            hidden_dim=NUM_EXPERTS,
            num_experts=NUM_EXPERTS,
            num_experts_per_token=TOP_K,
            norm_topk_prob=True,
            gate_dtype=DType.bfloat16,
            correction_bias_dtype=DType.bfloat16,
            routed_scaling_factor=2.5,
        )
        router.to(device)
    # An identity gate makes the router's input its own logits.
    compiled = router.compile(
        TensorType(
            DType.bfloat16,
            (NUM_TOKENS, NUM_EXPERTS),
            device=DeviceRef.from_device(device),
        ),
        weights={
            "gate_score.weight": torch.eye(NUM_EXPERTS, dtype=torch.bfloat16),
            "e_score_correction_bias": bias,
        },
    )
    idx, weight = compiled.execute_raw(Buffer.from_dlpack(logits).to(device))
    assert isinstance(idx, Buffer) and isinstance(weight, Buffer)
    return (
        torch.from_dlpack(idx.to(CPU())),
        torch.from_dlpack(weight.to(CPU())).float(),
    )


def test_kernel_route_matches_the_top_k_chain() -> None:
    torch.manual_seed(0)
    # Coarse bfloat16 logits and a two-level bias make exact ties in the
    # selection score common; the last row ties every expert.
    logits = (torch.randn(NUM_TOKENS, NUM_EXPERTS) * 2).to(torch.bfloat16)
    logits[-1] = 0.5
    bias = 0.0625 * torch.randint(0, 2, (NUM_EXPERTS,)).to(torch.bfloat16)

    gpu_idx, gpu_weight = _route(Accelerator(), logits, bias)
    cpu_idx, cpu_weight = _route(CPU(), logits, bias)

    # Both rank by the selection score, high first, ties to the lower expert.
    assert torch.equal(gpu_idx, cpu_idx.to(gpu_idx.dtype))
    tied = (bias == bias.max()).nonzero().flatten()[:TOP_K]
    assert torch.equal(gpu_idx[-1], tied.to(gpu_idx.dtype))
    # The kernel weights by the score itself where the chain subtracts the
    # bias back out, so the two can round to neighbouring bfloat16 values.
    torch.testing.assert_close(gpu_weight, cpu_weight, rtol=2**-7, atol=0)
