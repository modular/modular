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
from max.graph import BufferType, DeviceRef, Graph, TensorType, TensorValue, ops
from max.nn.legacy.comm.ep import EPBatchManager, EPCommInitializer, EPConfig
from max.nn.legacy.kernels import grouped_matmul_ragged
from test_common.graph_utils import is_b100_b200, is_h100_h200


@pytest.mark.skipif(
    not (is_h100_h200() or is_b100_b200()),
    reason="NVSHMEM library requires H100 or H200 or B200",
)
@pytest.mark.parametrize("n_devices", [4])
def test_ep_comm(n_devices: int) -> None:
    assert n_devices <= accelerator_count(), (
        "Devices are not enough to run EP test"
    )

    top_k = 8

    # Initialize the device-contexts
    devices = [Accelerator(id) for id in range(n_devices)]
    session = InferenceSession(devices=devices)

    config = EPConfig(
        dispatch_dtype=DType.bfloat16,
        combine_dtype=DType.bfloat16,
        hidden_size=7168,
        top_k=top_k,
        n_experts=min(256, n_devices * 32),
        max_tokens_per_rank=128,
        n_gpus_per_node=n_devices,
        n_nodes=int(os.environ.get("SHMEM_TOTAL_NODES", "1")),
    )
    ep_initializer = EPCommInitializer(config)
    ep_initializer.ep_init(session)

    torch.manual_seed(42)

    per_device_input_types: list[TensorType | BufferType] = [
        TensorType(
            DType.bfloat16,
            (f"input_len_{i}", config.hidden_size),
            DeviceRef.GPU(i),
        )
        for i in range(n_devices)
    ]
    per_device_topk_ids_types: list[TensorType | BufferType] = [
        TensorType(
            DType.int32, (f"input_len_{i}", config.top_k), DeviceRef.GPU(i)
        )
        for i in range(n_devices)
    ]

    input_lengths = torch.randint(1, config.max_tokens_per_rank, (n_devices,))
    per_device_inputs_torch = [
        torch.randn(
            input_lengths[i],
            config.hidden_size,
            dtype=torch.bfloat16,
            device="cpu",
        )
        for i in range(n_devices)
    ]

    per_device_inputs = [
        Buffer.from_dlpack(input).to(devices[i])
        for i, input in enumerate(per_device_inputs_torch)
    ]

    # create topk_ids
    all_topk_ids = []
    all_topk_ids_torch = []
    for i in range(n_devices):
        expert_scores = torch.randn(
            input_lengths[i], config.n_experts, device="cpu"
        )
        topk_ids = torch.topk(expert_scores, config.top_k, dim=1).indices.to(
            torch.int32
        )
        all_topk_ids_torch.append(topk_ids)
        all_topk_ids.append(Buffer.from_dlpack(topk_ids).to(devices[i]))

    # create expert weights
    expert_fingerprints = torch.randn(
        config.n_experts, dtype=torch.bfloat16, device="cpu"
    )
    weights_list = list()
    n_experts_per_device = config.n_experts // n_devices
    for i in range(n_devices):
        curr_device_expert_weights = list()
        for j in range(n_experts_per_device):
            curr_device_expert_weights.append(
                expert_fingerprints[i * n_experts_per_device + j]
                * torch.eye(
                    config.hidden_size, dtype=torch.bfloat16, device="cpu"
                )
            )
        weights_list.append(
            Buffer.from_dlpack(
                torch.stack(curr_device_expert_weights, dim=0)
            ).to(devices[i])
        )

    weights_types = [
        TensorType(
            DType.bfloat16,
            (n_experts_per_device, config.hidden_size, config.hidden_size),
            DeviceRef.GPU(i),
        )
        for i in range(n_devices)
    ]

    ep_manager = EPBatchManager(config)

    def build_ep_dispatch_graph() -> Graph:
        with Graph(
            "ep_comm",
            input_types=weights_types
            + per_device_input_types
            + per_device_topk_ids_types
            + ep_manager.input_types(),
        ) as graph:
            expert_weights = [val.tensor for val in graph.inputs[:n_devices]]
            xs = [val.tensor for val in graph.inputs[n_devices : n_devices * 2]]
            topk_ids = [
                val.tensor
                for val in graph.inputs[n_devices * 2 : n_devices * 3]
            ]

            ep_manager.fetch_buffers(graph.inputs[n_devices * 3 :])

            # flatten the outputs
            outputs: list[TensorValue] = []

            for dev_idx in range(n_devices):
                ep_manager.ep_dispatch_async(
                    xs[dev_idx], topk_ids[dev_idx], dev_idx
                )
                expert_inputs = ep_manager.ep_dispatch_wait(dev_idx)

                expert_outputs = grouped_matmul_ragged(
                    expert_inputs[0],
                    expert_weights[dev_idx],
                    *expert_inputs[1:],
                )

                ep_manager.ep_combine_async(expert_outputs, dev_idx)
                one = ops.constant(
                    1.0, dtype=DType.float32, device=DeviceRef.GPU(dev_idx)
                )
                router_weight = ops.broadcast_to(
                    one, (xs[dev_idx].shape[0], top_k)
                )
                outputs.append(
                    ep_manager.ep_combine_wait(router_weight, dev_idx)
                )

            graph.output(*outputs)

            return graph

    graph = build_ep_dispatch_graph()
    compiled = session.load(graph)
    results = compiled.execute(
        *weights_list,
        *per_device_inputs,
        *all_topk_ids,
        *ep_initializer.model_inputs(),
    )

    for dev_i in range(n_devices):
        top_k_expert_fingerprints = expert_fingerprints[
            all_topk_ids_torch[dev_i]
        ]
        ref_output = (
            (
                top_k_expert_fingerprints[:, :, None]
                * per_device_inputs_torch[dev_i][:, None, :]
            )
            .to(torch.float32)
            .sum(dim=1)
        ).to(torch.bfloat16)
        max_output = results[dev_i]
        torch.testing.assert_close(
            ref_output, torch.from_dlpack(max_output).to("cpu")
        )
