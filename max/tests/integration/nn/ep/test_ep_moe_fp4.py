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
from max.driver import Accelerator, Buffer, accelerator_api, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, Shape, ShardingStrategy, TensorType
from max.graph.weights import WeightData
from max.nn.legacy.comm.ep import EPBatchManager, EPCommInitializer, EPConfig
from max.nn.legacy.float8_config import (
    Float8Config,
    Float8InputScaleSpec,
    Float8ScaleGranularity,
    Float8ScaleOrigin,
    Float8WeightScaleSpec,
)
from max.nn.legacy.moe import MoEQuantized
from max.nn.legacy.transformer.distributed_transformer import (
    forward_sharded_layers,
)
from test_common.graph_utils import is_b100_b200

MOE_DIM = 2048
HIDDEN_DIM = 7168
NUM_EXPERTS = 64


@pytest.mark.skipif(
    accelerator_api() == "hip", reason="FP4 kernel only supports Nvidia GPUs"
)
@pytest.mark.skipif(
    not is_b100_b200(),
    reason="FP4 kernel requires B100 or B200",
)
@pytest.mark.parametrize("n_devices", [4])
def test_ep_moe_fp4(
    n_devices: int,
    moe_weights_fp4: dict[str, torch.Tensor],
) -> None:
    assert n_devices <= accelerator_count(), (
        "Devices are not enough to run EP test"
    )

    # Configuration parameters
    top_k = 8
    max_tokens_per_rank = 128
    dtype = DType.uint8

    # Wrap fp4 torch tensors as WeightData
    wrapped_moe_weights_fp4: dict[str, WeightData | torch.Tensor] = {}
    for key, value in moe_weights_fp4.items():
        if value.dtype == torch.float8_e4m3fn:
            wrapped_moe_weights_fp4[key] = WeightData(
                Buffer.from_dlpack(value.view(torch.uint8)).view(
                    DType.float8_e4m3fn
                ),
                key,
                DType.float8_e4m3fn,
                Shape(value.shape),
            )
        else:
            wrapped_moe_weights_fp4[key] = value

    # Initialize devices
    devices = [Accelerator(id) for id in range(n_devices)]
    devices_ref = [DeviceRef(d.label, d.id) for d in devices]
    session = InferenceSession(devices=devices)

    # Create fp4 config (NVFP4)
    fp4_input_config = Float8InputScaleSpec(
        granularity=Float8ScaleGranularity.BLOCK,
        origin=Float8ScaleOrigin.STATIC,
        dtype=DType.float32,
        block_size=(1, 16),
    )
    fp4_weight_config = Float8WeightScaleSpec(
        granularity=Float8ScaleGranularity.BLOCK,
        dtype=DType.float8_e4m3fn,
        block_size=(1, 8),
    )
    fp4_config = Float8Config(
        input_scale=fp4_input_config,
        weight_scale=fp4_weight_config,
        mlp_in_float8=set(),
        attn_qkv_in_float8=set(),
        embedding_output_dtype=None,
        quant_method="modelopt",
        quant_algo="NVFP4",
    )

    # Create EP configuration
    ep_config = EPConfig(
        dispatch_dtype=dtype,
        combine_dtype=DType.bfloat16,
        hidden_size=HIDDEN_DIM,
        top_k=top_k,
        n_experts=NUM_EXPERTS,
        max_tokens_per_rank=max_tokens_per_rank,
        n_gpus_per_node=n_devices,
        n_nodes=int(os.environ.get("SHMEM_TOTAL_NODES", "1")),
        dispatch_fp8_config=fp4_config,
        fused_shared_expert=True,
    )

    # Initialize EP communication
    ep_comm_init = EPCommInitializer(ep_config)
    ep_batch_manager = EPBatchManager(ep_config)

    # Create MoE module with EP support
    moe = MoEQuantized(
        devices=[DeviceRef.CPU()] + devices_ref,
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
        float8_config=fp4_config,
    )
    moe.sharding_strategy = ShardingStrategy.expert_parallel(n_devices)
    moe_shards = moe.shard(devices_ref)

    # Load weights
    moe.load_state_dict(wrapped_moe_weights_fp4)

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
    input_lengths = torch.randint(1, max_tokens_per_rank, (n_devices,))
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
        "EPMoE_FP4",
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

    all_outputs = torch.cat(torch_result, dim=0)
    assert not torch.any(torch.isnan(all_outputs)), (
        "MoE output should not contain NaN"
    )
