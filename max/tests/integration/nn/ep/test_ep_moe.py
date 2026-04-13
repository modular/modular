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
from conftest import N_DEVICES, CompiledEPModels
from max.driver import Buffer

MOE_DIM = 2048
HIDDEN_DIM = 7168
NUM_EXPERTS = 64


def torch_moe(
    input_token: torch.Tensor,
    moe_weights: dict[str, torch.Tensor],
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor,
    swiglu_limit: float = 0.0,
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
        if swiglu_limit > 0:
            expert_gate = torch.clamp(expert_gate, max=swiglu_limit)
            expert_up = torch.clamp(
                expert_up, min=-swiglu_limit, max=swiglu_limit
            )
        expert_output = (expert_gate * expert_up) @ down_weight.T

        result += expert_output * scores

    shared_gate_weight = moe_weights["shared_experts.gate_proj.weight"]
    shared_up_weight = moe_weights["shared_experts.up_proj.weight"]
    shared_down_weight = moe_weights["shared_experts.down_proj.weight"]
    shared_expert_gate = torch.nn.functional.silu(
        input_token @ shared_gate_weight.T
    )
    shared_expert_up = input_token @ shared_up_weight.T
    if swiglu_limit > 0:
        shared_expert_gate = torch.clamp(shared_expert_gate, max=swiglu_limit)
        shared_expert_up = torch.clamp(
            shared_expert_up, min=-swiglu_limit, max=swiglu_limit
        )
    shared_expert_output = (
        shared_expert_gate * shared_expert_up
    ) @ shared_down_weight.T
    result += shared_expert_output

    return result


@pytest.mark.parametrize(
    "input_lengths",
    [
        [128, 128, 128, 128],  # Equal distribution
        [128, 0, 0, 0],  # All tokens on first device
        [64, 32, 16, 8],  # Decreasing distribution
        [0, 0, 0, 128],  # All tokens on last device
        [0, 0, 0, 0],  # All zero tokens (needed for multi-node EP)
    ],
)
def test_ep_moe(
    compiled_ep_models: CompiledEPModels | None,
    input_lengths: list[int],
    moe_weights: dict[str, torch.Tensor],
) -> None:
    if compiled_ep_models is None:
        pytest.skip("NVSHMEM library requires H100 or H200 or B200")

    n_devices = N_DEVICES
    devices = compiled_ep_models.devices

    assert len(input_lengths) == n_devices, (
        f"input_lengths length {len(input_lengths)} must match n_devices {n_devices}"
    )
    per_device_inputs_torch = [
        torch.randn(
            input_lengths[i],
            HIDDEN_DIM,
            dtype=torch.bfloat16,
            device="cpu",
        )
        for i in range(n_devices)
    ]

    per_device_inputs = [
        Buffer.from_dlpack(input).to(devices[i])
        for i, input in enumerate(per_device_inputs_torch)
    ]

    result = compiled_ep_models.moe_model.execute(
        *per_device_inputs,
        *compiled_ep_models.ep_comm_init.model_inputs(),
    )
    torch_result = [torch.from_dlpack(x).to("cpu") for x in result]

    gate_result = compiled_ep_models.gate_model.execute(*per_device_inputs)
    topk_idxs_weights = [torch.from_dlpack(x).to("cpu") for x in gate_result]

    gpu = next(iter(moe_weights.values())).device
    all_outputs = torch.cat(torch_result, dim=0).to(gpu)
    all_inputs = torch.cat(per_device_inputs_torch, dim=0).to(gpu)
    all_topk_idxs = torch.cat(topk_idxs_weights[::2], dim=0).to(gpu)
    all_topk_weights = torch.cat(topk_idxs_weights[1::2], dim=0).to(gpu)

    for tok_idx in range(all_inputs.shape[0]):
        torch_output = torch_moe(
            all_inputs[tok_idx : tok_idx + 1],
            moe_weights,
            all_topk_idxs[tok_idx : tok_idx + 1],
            all_topk_weights[tok_idx : tok_idx + 1],
        )
        torch.testing.assert_close(
            all_outputs[tok_idx : tok_idx + 1],
            torch_output,
            rtol=3e-2,
            atol=4e-2,
        )


@pytest.mark.parametrize(
    "input_lengths",
    [
        [128, 128, 128, 128],  # Equal distribution
        [64, 32, 16, 8],  # Decreasing distribution
    ],
)
def test_ep_moe_swiglu_limit(
    compiled_ep_models_swiglu: CompiledEPModels | None,
    input_lengths: list[int],
    moe_weights: dict[str, torch.Tensor],
) -> None:
    if compiled_ep_models_swiglu is None:
        pytest.skip("NVSHMEM library requires H100 or H200 or B200")

    n_devices = N_DEVICES
    models = compiled_ep_models_swiglu
    devices = models.devices

    assert len(input_lengths) == n_devices, (
        f"input_lengths length {len(input_lengths)} must match n_devices {n_devices}"
    )
    per_device_inputs_torch = [
        torch.randn(
            input_lengths[i],
            HIDDEN_DIM,
            dtype=torch.bfloat16,
            device="cpu",
        )
        for i in range(n_devices)
    ]

    per_device_inputs = [
        Buffer.from_dlpack(input).to(devices[i])
        for i, input in enumerate(per_device_inputs_torch)
    ]

    result = models.moe_model.execute(
        *per_device_inputs,
        *models.ep_comm_init.model_inputs(),
    )
    torch_result = [torch.from_dlpack(x).to("cpu") for x in result]

    gate_result = models.gate_model.execute(*per_device_inputs)
    topk_idxs_weights = [torch.from_dlpack(x).to("cpu") for x in gate_result]

    gpu = next(iter(moe_weights.values())).device
    all_outputs = torch.cat(torch_result, dim=0).to(gpu)
    all_inputs = torch.cat(per_device_inputs_torch, dim=0).to(gpu)
    all_topk_idxs = torch.cat(topk_idxs_weights[::2], dim=0).to(gpu)
    all_topk_weights = torch.cat(topk_idxs_weights[1::2], dim=0).to(gpu)

    for tok_idx in range(all_inputs.shape[0]):
        torch_output = torch_moe(
            all_inputs[tok_idx : tok_idx + 1],
            moe_weights,
            all_topk_idxs[tok_idx : tok_idx + 1],
            all_topk_weights[tok_idx : tok_idx + 1],
            swiglu_limit=models.swiglu_limit,
        )
        torch.testing.assert_close(
            all_outputs[tok_idx : tok_idx + 1],
            torch_output,
            rtol=3e-2,
            atol=4e-2,
        )
