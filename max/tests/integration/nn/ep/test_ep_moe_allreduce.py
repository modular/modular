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

from __future__ import annotations

import pytest
import torch
from conftest import CompiledAllreduceEPModels
from max.driver import Buffer

MOE_DIM = 2048
HIDDEN_DIM = 7168
NUM_EXPERTS = 64


def torch_moe(
    input_token: torch.Tensor,
    moe_weights: dict[str, torch.Tensor],
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor,
) -> torch.Tensor:
    assert input_token.shape[0] == 1, (
        "The naive MoE implementation only supports a single token at a time"
    )

    top_k = topk_indices.shape[1]

    result = torch.zeros_like(input_token)

    for i in range(top_k):
        scores = topk_scores[0, i]
        expert_idx = topk_indices[0, i].item()
        gate_weight = moe_weights[f"experts.{expert_idx}.gate_proj.weight"]
        up_weight = moe_weights[f"experts.{expert_idx}.up_proj.weight"]
        down_weight = moe_weights[f"experts.{expert_idx}.down_proj.weight"]

        expert_gate = torch.nn.functional.silu(input_token @ gate_weight.T)
        expert_up = input_token @ up_weight.T
        expert_output = (expert_gate * expert_up) @ down_weight.T

        result += expert_output * scores

    shared_gate_weight = moe_weights["shared_experts.gate_proj.weight"]
    shared_up_weight = moe_weights["shared_experts.up_proj.weight"]
    shared_down_weight = moe_weights["shared_experts.down_proj.weight"]
    shared_expert_gate = torch.nn.functional.silu(
        input_token @ shared_gate_weight.T
    )
    shared_expert_up = input_token @ shared_up_weight.T
    shared_expert_output = (
        shared_expert_gate * shared_expert_up
    ) @ shared_down_weight.T
    result += shared_expert_output

    return result


@pytest.mark.parametrize("input_length", [128, 64, 16])
def test_ep_moe_allreduce(
    compiled_allreduce_ep_models: CompiledAllreduceEPModels | None,
    input_length: int,
    moe_weights: dict[str, torch.Tensor],
) -> None:
    """Test EP MoE with use_allreduce=True.

    A single input on device 0 is broadcast to all devices. Each device runs
    EP dispatch/combine locally (no cross-device A2A). An allreduce sum
    collects the partial MoE outputs, which should match the full torch MoE
    reference.
    """
    if compiled_allreduce_ep_models is None:
        pytest.skip("NVSHMEM library requires H100 or H200 or B200")

    models = compiled_allreduce_ep_models

    input_torch = torch.randn(
        input_length, HIDDEN_DIM, dtype=torch.bfloat16, device="cpu"
    )
    input_buf = Buffer.from_dlpack(input_torch).to(models.devices[0])

    result = models.moe_model.execute(
        input_buf,
        *models.ep_comm_init.model_inputs(),
        *models.signal_buffers,
    )
    # All devices hold the same allreduced result; pick the first.
    output = torch.from_dlpack(result[0]).to("cpu")

    gate_result = models.gate_model.execute(
        input_buf,
        *models.signal_buffers,
    )
    topk_idxs_weights = [torch.from_dlpack(x).to("cpu") for x in gate_result]

    # All devices ran the same gate on the same broadcast input, so topk
    # indices and weights are identical across devices. Use device 0's output.
    topk_idxs = topk_idxs_weights[0]
    topk_weights = topk_idxs_weights[1]

    gpu = next(iter(moe_weights.values())).device
    output = output.to(gpu)
    input_gpu = input_torch.to(gpu)
    topk_idxs = topk_idxs.to(gpu)
    topk_weights = topk_weights.to(gpu)

    for tok_idx in range(input_length):
        torch_output = torch_moe(
            input_gpu[tok_idx : tok_idx + 1],
            moe_weights,
            topk_idxs[tok_idx : tok_idx + 1],
            topk_weights[tok_idx : tok_idx + 1],
        )
        torch.testing.assert_close(
            output[tok_idx : tok_idx + 1],
            torch_output,
            rtol=3e-2,
            atol=4e-2,
        )
