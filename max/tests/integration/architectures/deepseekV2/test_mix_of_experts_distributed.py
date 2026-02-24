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

import functools
from collections.abc import Sequence

import pytest
import torch
from max._core.engine import PrintStyle
from max.driver import Accelerator, Device, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    DeviceRef,
    Graph,
    ShardingStrategy,
    TensorType,
    TensorValue,
)
from max.nn import Allreduce, Signals
from max.nn.moe import MoE
from max.pipelines.architectures.deepseekV2.layers.moe_gate import (
    DeepSeekV2MoEGate,
)
from torch_reference.configuration_deepseek import DeepseekV2Config
from torch_reference.modeling_deepseek import DeepseekV2MoE


def _distribute_value(
    v: TensorValue, devices: Sequence[Device]
) -> Sequence[TensorValue]:
    return [v.to(DeviceRef(device.label, device.id)) for device in devices]


def generate_torch_outputs(
    config: DeepseekV2Config,
    input_tensor: torch.Tensor,
    dummy_moe_weight: torch.Tensor,
    expert_weights: list[dict[str, torch.Tensor]],
    shared_expert_weights: dict[str, torch.Tensor],
    dtype: DType,
    device: torch.device,
) -> torch.Tensor:
    layer = DeepseekV2MoE(config).to(dtype).to(device)
    layer.training = False

    input_tensor = input_tensor.to(device)
    # Update expert weights
    for i, expert in enumerate(layer.experts):
        if expert is not None:
            for name, param in expert.named_parameters():
                param.data = expert_weights[i][name].to(dtype).to(device)

    # Update shared expert weights
    if layer.config.n_shared_experts is not None:
        for name, param in layer.shared_experts.named_parameters():
            param.data = shared_expert_weights[name].to(dtype).to(device)

    layer.gate.weight.data = dummy_moe_weight.to(device)
    return layer(input_tensor)


def generate_max_outputs(
    config: DeepseekV2Config,
    input_tensor: torch.Tensor,
    dummy_moe_weight: torch.Tensor,
    expert_weights: list[dict[str, torch.Tensor]],
    shared_expert_weights: dict[str, torch.Tensor],
    dtype: DType,
) -> torch.Tensor:
    input_tensor = input_tensor.cuda()

    state_dict = {"gate.gate_score.weight": dummy_moe_weight.cpu()}

    for i, expert_weight in enumerate(expert_weights):
        state_dict[f"experts.{i}.down_proj.weight"] = expert_weight[
            "down_proj.weight"
        ].cpu()
        state_dict[f"experts.{i}.gate_proj.weight"] = expert_weight[
            "gate_proj.weight"
        ].cpu()
        state_dict[f"experts.{i}.up_proj.weight"] = expert_weight[
            "up_proj.weight"
        ].cpu()

    state_dict["shared_experts.gate_proj.weight"] = shared_expert_weights[
        "gate_proj.weight"
    ].cpu()
    state_dict["shared_experts.down_proj.weight"] = shared_expert_weights[
        "down_proj.weight"
    ].cpu()
    state_dict["shared_experts.up_proj.weight"] = shared_expert_weights[
        "up_proj.weight"
    ].cpu()

    n_devices = 2
    if accelerator_count() < n_devices:
        pytest.skip("Not enough GPUs to run test.")

    devices = [Accelerator(id) for id in range(n_devices)]
    device_refs = [DeviceRef(device.label, device.id) for device in devices]
    signals = Signals(device_refs)

    tp_moe = MoE(
        dtype=dtype,
        devices=device_refs,
        hidden_dim=config.hidden_size,
        num_experts=config.n_routed_experts,
        num_experts_per_token=config.num_experts_per_tok,
        moe_dim=config.moe_intermediate_size,
        gate_cls=functools.partial(
            DeepSeekV2MoEGate,
            topk_method=config.topk_method,
            n_group=config.n_group,
            topk_group=config.topk_group,
            routed_scaling_factor=config.routed_scaling_factor,
        ),
        has_shared_experts=True,
        shared_experts_dim=(
            config.n_shared_experts * config.moe_intermediate_size
        ),
    )
    tp_moe.sharding_strategy = ShardingStrategy.tensor_parallel(n_devices)
    tp_moe_shards = tp_moe.shard(device_refs)

    tp_moe.load_state_dict(state_dict)

    allreduce = Allreduce(num_accelerators=n_devices)

    session = InferenceSession(devices=devices)
    session.set_debug_print_options(style=PrintStyle.COMPACT)
    with Graph(
        "distributed_tp_moe",
        input_types=[
            TensorType(
                dtype,
                (input_tensor.shape[0], config.hidden_size),
                device=DeviceRef.GPU(),
            ),
            *signals.input_types(),
        ],
    ) as graph:
        assert isinstance(graph.inputs[0], TensorValue)
        inputs = _distribute_value(graph.inputs[0], devices)
        signal_buffers = [inp.buffer for inp in graph.inputs[1:]]
        outputs = [
            tp_moe_shard(input)
            for tp_moe_shard, input in zip(tp_moe_shards, inputs, strict=True)
        ]
        outputs = allreduce(outputs, signal_buffers)
        graph.output(*outputs)

        compiled = session.load(graph, weights_registry=tp_moe.state_dict())
        return compiled.execute(input_tensor, *signals.buffers())


def test_mix_of_experts(
    config: DeepseekV2Config,
    input_tensor: torch.Tensor,
    dummy_moe_weight: torch.Tensor,
    expert_weights: list[dict[str, torch.Tensor]],
    shared_expert_weights: dict[str, torch.Tensor],
) -> None:
    torch_dtype = torch.bfloat16
    max_dtype = DType.bfloat16
    torch_output = generate_torch_outputs(
        config,
        input_tensor,
        dummy_moe_weight,
        expert_weights,
        shared_expert_weights,
        torch_dtype,
        torch.device("cuda"),
    )

    max_output = generate_max_outputs(
        config,
        input_tensor.squeeze(),  # remove batch dimension
        dummy_moe_weight,
        expert_weights,
        shared_expert_weights,
        max_dtype,
    )

    torch.testing.assert_close(
        torch_output.squeeze(),
        torch.from_dlpack(max_output[0]).to(torch_dtype).squeeze(),
        rtol=1e-3,
        atol=2 * torch.finfo(torch.bfloat16).eps,
    )
