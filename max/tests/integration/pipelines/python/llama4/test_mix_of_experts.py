# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest
import torch
from max.dtype import DType
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
    return layer(input_tensor)


@pytest.mark.skip(reason="TODO: Working through torch shape errors.")
def test_mix_of_experts(
    text_config: Llama4TextConfig,
    config: Llama4Config,
    input_tensor: torch.Tensor,
    dummy_router_weight: torch.Tensor,
    expert_weights: dict[str, torch.Tensor],
    shared_expert_weights: dict[str, torch.Tensor],
) -> None:
    torch_dtype = torch.bfloat16
    torch_output = generate_torch_outputs(
        text_config,
        input_tensor,
        dummy_router_weight,
        expert_weights,
        shared_expert_weights,
        torch_dtype,
        "cuda",
    )

    print(torch_output)

    # TODO: Generate max outputs and compare with torch outputs once layer is implemented.
