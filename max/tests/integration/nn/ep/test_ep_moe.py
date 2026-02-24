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

import os

import pytest
import torch
from max.driver import Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    DeviceRef,
    Graph,
    ShardingStrategy,
    TensorType,
    TensorValue,
)
from max.nn.legacy.comm.ep import EPBatchManager, EPCommInitializer, EPConfig
from max.nn.legacy.moe import MoE, MoEGate
from max.nn.legacy.transformer.distributed_transformer import (
    forward_sharded_layers,
)
from test_common.graph_utils import is_b100_b200

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

        expert_gate = input_token @ gate_weight.T
        expert_up = input_token @ up_weight.T
        expert_output = (
            torch.nn.functional.silu(expert_gate) * expert_up
        ) @ down_weight.T

        result += expert_output * scores

    shared_gate_weight = moe_weights["shared_experts.gate_proj.weight"]
    shared_up_weight = moe_weights["shared_experts.up_proj.weight"]
    shared_down_weight = moe_weights["shared_experts.down_proj.weight"]
    shared_expert_gate = input_token @ shared_gate_weight.T
    shared_expert_up = input_token @ shared_up_weight.T
    shared_expert_output = (
        torch.nn.functional.silu(shared_expert_gate) * shared_expert_up
    ) @ shared_down_weight.T
    result += shared_expert_output

    return result


@pytest.mark.skipif(
    not is_b100_b200(),
    reason="NVSHMEM library requires H100 or H200 or B200",
)
@pytest.mark.parametrize("n_devices", [4])
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
    n_devices: int,
    input_lengths: list[int],
    moe_weights: dict[str, torch.Tensor],
) -> None:
    assert n_devices <= accelerator_count(), (
        "Devices are not enough to run EP test"
    )

    # Configuration parameters
    top_k = 8
    max_tokens_per_rank = 128
    dtype = DType.bfloat16

    # Initialize devices
    devices = [Accelerator(id) for id in range(n_devices)]
    devices_ref = [DeviceRef(d.label, d.id) for d in devices]
    session = InferenceSession(devices=devices)

    # Create EP configuration
    ep_config = EPConfig(
        dispatch_dtype=dtype,
        combine_dtype=dtype,
        hidden_size=HIDDEN_DIM,
        top_k=top_k,
        n_experts=NUM_EXPERTS,
        max_tokens_per_rank=max_tokens_per_rank,
        n_gpus_per_node=n_devices,
        n_nodes=int(os.environ.get("SHMEM_TOTAL_NODES", "1")),
    )

    # Initialize EP communication
    ep_comm_init = EPCommInitializer(ep_config)
    ep_batch_manager = EPBatchManager(ep_config)

    # Create MoE module with EP support
    moe = MoE(
        devices=devices_ref,
        hidden_dim=HIDDEN_DIM,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=top_k,
        moe_dim=MOE_DIM,
        has_shared_experts=True,
        shared_experts_dim=MOE_DIM,
        ep_size=n_devices,
        dtype=dtype,
        apply_router_weight_first=False,
        ep_batch_manager=ep_batch_manager,
    )
    moe.sharding_strategy = ShardingStrategy.expert_parallel(n_devices)
    moe_shards = moe.shard(devices_ref)

    # Load weights
    moe.load_state_dict(moe_weights)

    # Initialize EP communication infrastructure
    ep_comm_init.ep_init(session)

    per_device_input_types: list[TensorType] = [
        TensorType(
            DType.bfloat16,
            (f"input_len_{i}", HIDDEN_DIM),
            DeviceRef.GPU(i),
        )
        for i in range(n_devices)
    ]
    # Convert input_lengths list to torch tensor for validation
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

    with Graph(
        "EPMoE",
        input_types=[
            *per_device_input_types,
            *ep_batch_manager.input_types(),
        ],
    ) as graph:
        inputs_tensors = [x.tensor for x in graph.inputs[:n_devices]]

        ep_batch_manager.fetch_buffers(graph.inputs[n_devices:])

        # Run MoE with EP
        outputs = forward_sharded_layers(moe_shards, inputs_tensors)

        graph.output(*outputs)

    # Compile and execute
    compiled = session.load(graph, weights_registry=moe.state_dict())
    result = compiled.execute(*per_device_inputs, *ep_comm_init.model_inputs())
    torch_result = [torch.from_dlpack(x).to("cpu") for x in result]

    # We use MoEGate to get the topk indices and scores, as the output of MoE
    # is sensitive to the expert selection.
    moe_gate = MoEGate(
        devices=devices_ref,
        hidden_dim=HIDDEN_DIM,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=top_k,
        dtype=dtype,
    )
    moe_gate.sharding_strategy = ShardingStrategy.replicate(n_devices)
    moe_gate_shards = moe_gate.shard(devices_ref)

    gate_weight_dict = {
        "gate_score.weight": moe_weights["gate.gate_score.weight"]
    }
    moe_gate.load_state_dict(gate_weight_dict)

    with Graph(
        "MoEGate",
        input_types=per_device_input_types,
    ) as gate_graph:
        gate_inputs = [x.tensor for x in gate_graph.inputs[:n_devices]]

        gate_outputs: list[TensorValue] = []
        for moe_gate_shard, input in zip(
            moe_gate_shards, gate_inputs, strict=False
        ):
            gate_outputs.extend(moe_gate_shard(input))

        gate_graph.output(*gate_outputs)

    gate_compiled = session.load(
        gate_graph, weights_registry=moe_gate.state_dict()
    )
    gate_result = gate_compiled.execute(*per_device_inputs)
    topk_idxs_weights = [torch.from_dlpack(x).to("cpu") for x in gate_result]

    all_outputs = torch.cat(torch_result, dim=0)
    all_inputs = torch.cat(per_device_inputs_torch, dim=0)
    all_topk_idxs = torch.cat(topk_idxs_weights[::2], dim=0)
    all_topk_weights = torch.cat(topk_idxs_weights[1::2], dim=0)

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
