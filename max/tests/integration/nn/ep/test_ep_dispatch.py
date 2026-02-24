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
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, TensorValue
from max.nn.legacy.comm.ep import EPBatchManager, EPCommInitializer, EPConfig
from test_common.graph_utils import gpu_warp_size

# EP_DATA_READY_FLAG constant from ep_comm.mojo
EP_DATA_READY_FLAG = 1 << 10  # 1024

MAX_GPUS_PER_NODE = 8


def verify_ep_dispatch_results(
    results: list,
    per_device_inputs_torch: list[torch.Tensor],
    all_topk_ids_torch: list[torch.Tensor],
    atomic_counters: list[Buffer],
    config: EPConfig,
    n_devices: int,
) -> None:
    n_local_experts = config.n_experts // n_devices

    atomic_counters_torch = [
        torch.from_dlpack(counter) for counter in atomic_counters
    ]
    # dispatch_wait counters start at offset dispatch_size = 2 * n_experts + MAX_GPUS_PER_NODE
    dispatch_size = 2 * config.n_experts + MAX_GPUS_PER_NODE
    dispatch_wait_counters_torch = [
        counter[dispatch_size:] for counter in atomic_counters_torch
    ]

    # Parse results according to layout:
    # (tokens, row_offsets, expert_ids, dummy, src_token_info) * num_devices
    for device_idx in range(n_devices):
        base_idx = device_idx * 5

        # Extract outputs for this device
        output_tokens = results[base_idx]
        row_offsets = results[base_idx + 1]
        expert_ids = results[base_idx + 2]
        # Skip dummy value at base_idx + 3
        src_token_info = results[base_idx + 4]

        # Convert to torch tensors
        output_tokens_torch = torch.from_dlpack(output_tokens).to("cpu")
        row_offsets_torch = torch.from_dlpack(row_offsets)
        expert_ids_torch = torch.from_dlpack(expert_ids)
        src_token_info_torch = torch.from_dlpack(src_token_info)
        dispatch_wait_counter_torch = dispatch_wait_counters_torch[device_idx]

        # Verify outputs for each expert on this device
        for expert_idx in range(n_local_experts):
            curr_local_expert = expert_ids_torch[expert_idx]
            curr_expert = n_local_experts * device_idx + curr_local_expert

            token_start = row_offsets_torch[expert_idx]
            token_end = row_offsets_torch[expert_idx + 1]

            # Verify each token for this expert
            remote_rank = 0
            for token_idx in range(token_start, token_end):
                # Find which remote rank this token came from using atomic counters
                while (
                    remote_rank < n_devices
                    and dispatch_wait_counter_torch[
                        2 * (curr_local_expert * n_devices + remote_rank)
                    ]
                    <= token_idx + EP_DATA_READY_FLAG
                ):
                    remote_rank += 1

                assert remote_rank < n_devices, (
                    f"Cannot find the remote rank for token {token_idx} of device {device_idx}"
                )

                # Extract source token information
                remote_loc = src_token_info_torch[token_idx, 0]
                remote_topk_id = src_token_info_torch[token_idx, 1]

                # Verify that curr_expert is in remote rank's topk_ids at the expected position
                expected_expert = all_topk_ids_torch[remote_rank][
                    remote_loc, remote_topk_id
                ]
                assert expected_expert == curr_expert

                # Verify that the received token matches the remote rank's input token
                expected_token = per_device_inputs_torch[remote_rank][
                    remote_loc
                ]
                received_token = output_tokens_torch[token_idx]

                # Compare tokens with some tolerance for floating point precision
                assert torch.equal(expected_token, received_token), (
                    f"Device {device_idx}, Token {token_idx}: Token data mismatch"
                )


@pytest.mark.parametrize("n_devices", [4])
def test_ep_dispatch(n_devices: int) -> None:
    assert n_devices <= accelerator_count(), (
        "Devices are not enough to run EP test"
    )

    # Initialize the device-contexts
    host = CPU(0)
    devices = [Accelerator(id) for id in range(n_devices)]
    devices_wit_host = [host, *devices]
    session = InferenceSession(devices=devices_wit_host)

    config = EPConfig(
        dispatch_dtype=DType.bfloat16,
        combine_dtype=DType.bfloat16,
        hidden_size=7168,
        top_k=8,
        n_experts=min(256, n_devices * (1024 // gpu_warp_size())),
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

    ep_manager = EPBatchManager(config)

    def build_ep_dispatch_graph() -> Graph:
        with Graph(
            "ep_dispatch",
            input_types=per_device_input_types
            + per_device_topk_ids_types
            + ep_manager.input_types(),
        ) as graph:
            xs = [val.tensor for val in graph.inputs[:n_devices]]
            topk_ids = [
                val.tensor for val in graph.inputs[n_devices : n_devices * 2]
            ]

            ep_manager.fetch_buffers(graph.inputs[n_devices * 2 :])

            # flatten the outputs
            flattened_outputs: list[TensorValue] = []

            for dev_idx in range(n_devices):
                ep_manager.ep_dispatch_async(
                    xs[dev_idx], topk_ids[dev_idx], dev_idx
                )
                outputs = ep_manager.ep_dispatch_wait(dev_idx)
                flattened_outputs.extend(outputs)

                src_info = ep_manager._src_info[dev_idx]
                assert src_info is not None
                flattened_outputs.append(src_info)

            graph.output(*flattened_outputs)

            return graph

    graph = build_ep_dispatch_graph()
    compiled = session.load(graph)
    results = compiled.execute(
        *per_device_inputs, *all_topk_ids, *ep_initializer.model_inputs()
    )

    atomic_counters = ep_initializer.atomic_counters

    # Verify the EP dispatch results
    verify_ep_dispatch_results(
        results=results,
        per_device_inputs_torch=per_device_inputs_torch,
        all_topk_ids_torch=all_topk_ids_torch,
        atomic_counters=atomic_counters,
        config=config,
        n_devices=n_devices,
    )
