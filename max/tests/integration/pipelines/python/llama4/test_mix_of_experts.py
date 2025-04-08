# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest
import torch
from max._core.engine import PrintStyle
from max.driver import Accelerator, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.llama4.mix_of_experts import MoE
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
    device: Device,
) -> torch.Tensor:
    is_gpu = isinstance(device, Accelerator)
    input_tensor = input_tensor.cuda() if is_gpu else input_tensor.cpu()

    # TODO: .cpu()s added as workaround for GEX-1967
    state_dict = {"router.weight": dummy_router_weight.cpu()}

    state_dict["experts.gate_up_proj.weight"] = expert_weights[
        "gate_up_proj"
    ].cpu()
    state_dict["experts.down_proj.weight"] = expert_weights["down_proj"].cpu()

    state_dict["shared_expert_gate_proj.weight"] = shared_expert_weights[
        "gate_proj.weight"
    ].cpu()
    state_dict["shared_expert_down_proj.weight"] = shared_expert_weights[
        "down_proj.weight"
    ].cpu()
    state_dict["shared_expert_up_proj.weight"] = shared_expert_weights[
        "up_proj.weight"
    ].cpu()

    moe = MoE(dtype=dtype)
    moe.load_state_dict(state_dict)

    session = InferenceSession(devices=[Accelerator(0)])
    session.set_debug_print_options(style=PrintStyle.COMPACT)
    graph = Graph(
        "MoE",
        moe,
        input_types=(
            TensorType(
                dtype,
                (
                    input_tensor.shape[0],
                    input_tensor.shape[1],
                    text_config.hidden_size,
                ),
                device=DeviceRef.GPU() if is_gpu else DeviceRef.CPU(),
            ),
        ),
    )

    compiled = session.load(graph, weights_registry=moe.state_dict())
    return compiled.execute(input_tensor)


@pytest.mark.skip(
    reason="Accuracy debugging in progress. Likely to pass when tested using actual weights from the checkpoint."
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
        Accelerator(),
    )

    torch.testing.assert_close(
        torch_output.squeeze(),
        torch.from_dlpack(max_output[0]).to(torch_dtype).squeeze(),
        rtol=1e-3,
        atol=2 * torch.finfo(torch.bfloat16).eps,
    )
