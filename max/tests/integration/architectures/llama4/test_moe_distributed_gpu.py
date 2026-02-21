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

import pytest
import torch
from max._core.engine import PrintStyle
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.graph.weight import ShardingStrategy
from max.nn.legacy import Signals
from max.nn.legacy.comm import Allreduce
from max.pipelines.architectures.llama4.layers.moe import (
    Llama4MoE,
    Llama4MoEGate,
)
from transformers.models.llama4.configuration_llama4 import (
    Llama4Config,
    Llama4TextConfig,
)
from transformers.models.llama4.modeling_llama4 import Llama4TextMoe


def generate_torch_outputs(
    text_config: Llama4TextConfig,
    input_tensor: torch.Tensor,
    dummy_router_weight: torch.Tensor,
    expert_weights: dict[str, torch.Tensor],
    shared_expert_weights: dict[str, torch.Tensor],
    dtype: DType,
    device: torch.device,
) -> torch.Tensor:
    layer = Llama4TextMoe(text_config).to(dtype).to(device)
    layer.training = False

    input_tensor = input_tensor.to(device)
    # Update expert weights
    for name, param in layer.experts.named_parameters():
        param.data = expert_weights[name].to(dtype).to(device)

    # Update shared expert weights
    for name, param in layer.shared_expert.named_parameters():
        param.data = shared_expert_weights[name].to(dtype).to(device)

    layer.router.weight.data = dummy_router_weight.to(device)
    return layer(input_tensor)[0]


def generate_max_outputs(
    text_config: Llama4TextConfig,
    input_tensor: torch.Tensor,
    dummy_router_weight: torch.Tensor,
    expert_weights: dict[str, torch.Tensor],
    shared_expert_weights: dict[str, torch.Tensor],
    dtype: DType,
    n_devices: int,
) -> torch.Tensor:
    # Initialize the device-contexts
    host = CPU(0)
    # Check we are parallelizing over legal amounts of devices and create contexts.
    assert n_devices <= accelerator_count()
    devices = [Accelerator(id) for id in range(n_devices)]
    devices_ref = [DeviceRef(d.label, d.id) for d in devices]

    input_tensor = input_tensor.reshape([-1, text_config.hidden_size])
    input_tensor = input_tensor.cuda()

    state_dict = {"gate.gate_score.weight": dummy_router_weight.cpu()}

    state_dict["experts.gate_up_proj"] = expert_weights["gate_up_proj"].cpu()
    state_dict["experts.down_proj"] = expert_weights["down_proj"].cpu()

    state_dict["shared_experts.gate_proj.weight"] = shared_expert_weights[
        "gate_proj.weight"
    ].cpu()
    state_dict["shared_experts.down_proj.weight"] = shared_expert_weights[
        "down_proj.weight"
    ].cpu()
    state_dict["shared_experts.up_proj.weight"] = shared_expert_weights[
        "up_proj.weight"
    ].cpu()

    moe = Llama4MoE(
        dtype=dtype,
        devices=devices_ref,
        hidden_dim=text_config.hidden_size,
        num_experts=text_config.num_local_experts,
        num_experts_per_token=text_config.num_experts_per_tok,
        moe_dim=text_config.intermediate_size,
        gate_cls=Llama4MoEGate,
        has_shared_experts=True,
        shared_experts_dim=text_config.intermediate_size,
        apply_router_weight_first=True,
    )
    moe.load_state_dict(state_dict)
    moe.sharding_strategy = ShardingStrategy.tensor_parallel(n_devices)
    moe_shards = moe.shard(devices_ref)
    allreduce = Allreduce(num_accelerators=n_devices)

    session = InferenceSession(devices=devices)
    session.set_debug_print_options(style=PrintStyle.COMPACT)

    signals = Signals(devices=(DeviceRef(d.label, d.id) for d in devices))
    signal_buffers = [
        Buffer.zeros(
            shape=(Signals.NUM_BYTES,),
            dtype=DType.uint8,
            device=dev,
        )
        for dev in devices
    ]

    with Graph(
        "DistributedMoE",
        input_types=[
            TensorType(
                dtype,
                (
                    input_tensor.shape[0],
                    text_config.hidden_size,
                ),
                device=DeviceRef.GPU(0),
            ),
            *signals.input_types(),
        ],
    ) as graph:
        inputs_tensor = graph.inputs[0].tensor
        graph_signal_buffers = [v.buffer for v in graph.inputs[1:]]

        inputs_per_device = [inputs_tensor.to(device) for device in devices_ref]

        moe_outs = [
            shard(x)
            for shard, x in zip(moe_shards, inputs_per_device, strict=True)
        ]
        outputs = allreduce(moe_outs, graph_signal_buffers)

        graph.output(outputs[0])

    compiled = session.load(graph, weights_registry=moe.state_dict())
    return compiled.execute(input_tensor, *signal_buffers)


@pytest.mark.skip(
    reason="TODO(MODELS-728): Fix MOE bugs introduced after transformers version bump"
)
def test_mix_of_experts(
    text_config: Llama4TextConfig,
    config: Llama4Config,
    input_tensor: torch.Tensor,
    dummy_router_weight: torch.Tensor,
    expert_weights: dict[str, torch.Tensor],
    shared_expert_weights: dict[str, torch.Tensor],
) -> None:
    torch_dtype = torch.bfloat16
    max_dtype = DType.bfloat16
    torch_output = generate_torch_outputs(
        text_config,
        input_tensor,
        dummy_router_weight,
        expert_weights,
        shared_expert_weights,
        torch_dtype,
        "cuda",
    )

    max_output = generate_max_outputs(
        text_config,
        input_tensor,
        dummy_router_weight,
        expert_weights,
        shared_expert_weights,
        max_dtype,
        2,
    )

    torch.testing.assert_close(
        torch_output.squeeze(),
        torch.from_dlpack(max_output[0]).to(torch_dtype).squeeze(),
        rtol=1e-3,
        atol=2 * torch.finfo(torch.bfloat16).eps,
    )
