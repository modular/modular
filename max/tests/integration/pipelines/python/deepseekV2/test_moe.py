# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest
import torch
from max._core.engine import PrintStyle
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType
from max.pipelines.architectures.deepseekV2.layers.moe import MoE
from torch_reference.configuration_deepseek import (
    DeepseekV2Config,
)
from torch_reference.modeling_deepseek import DeepseekV2MoE


def generate_torch_outputs(
    config: DeepseekV2Config,
    input_tensor: torch.Tensor,
    dummy_moe_weight: torch.Tensor,
    expert_weights: list[dict[str, torch.Tensor]],
    shared_expert_weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    layer = DeepseekV2MoE(config).to(torch.bfloat16)
    layer.training = False

    # Update expert weights
    for i, expert in enumerate(layer.experts):
        if expert is not None:
            for name, param in expert.named_parameters():
                param.data = expert_weights[i][name].to(torch.bfloat16)

    # Update shared expert weights
    if layer.config.n_shared_experts is not None:
        for name, param in expert.named_parameters():
            param.data = shared_expert_weights[name].to(torch.bfloat16)

    layer.gate.weight.data = dummy_moe_weight
    return layer(input_tensor)


def generate_max_outputs(
    config: DeepseekV2Config,
    input_tensor: torch.Tensor,
    dummy_moe_weight: torch.Tensor,
    expert_weights: list[dict[str, torch.Tensor]],
    shared_expert_weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    state_dict = {"gate.gate_score.weight": dummy_moe_weight}

    for i in range(len(expert_weights)):
        state_dict[f"gate_proj{i}.weight"] = expert_weights[i][
            "gate_proj.weight"
        ]
        state_dict[f"down_proj{i}.weight"] = expert_weights[i][
            "down_proj.weight"
        ]
        state_dict[f"up_proj{i}.weight"] = expert_weights[i]["up_proj.weight"]

    state_dict["shared_expert_gate_proj.weight"] = shared_expert_weights[
        "gate_proj.weight"
    ]
    state_dict["shared_expert_down_proj.weight"] = shared_expert_weights[
        "down_proj.weight"
    ]
    state_dict["shared_expert_up_proj.weight"] = shared_expert_weights[
        "up_proj.weight"
    ]

    moe = MoE()
    moe.load_state_dict(state_dict)

    session = InferenceSession()
    session.set_debug_print_options(style=PrintStyle.COMPACT)
    graph = Graph(
        "MoE",
        moe,
        input_types=(
            TensorType(
                DType.bfloat16,
                (
                    input_tensor.shape[0],
                    input_tensor.shape[1],
                    config.hidden_size,
                ),
            ),
        ),
    )

    compiled = session.load(graph, weights_registry=moe.state_dict())
    return compiled.execute(input_tensor)


@pytest.mark.skip(reason="Accuracy debugging in progress")
def test_moe(
    config: DeepseekV2Config,
    input_tensor: torch.Tensor,
    dummy_moe_weight: torch.Tensor,
    expert_weights: list[dict[str, torch.Tensor]],
    shared_expert_weights: dict[str, torch.Tensor],
) -> None:
    torch_output = generate_torch_outputs(
        config,
        input_tensor,
        dummy_moe_weight,
        expert_weights,
        shared_expert_weights,
    )

    max_output = generate_max_outputs(
        config,
        input_tensor,
        dummy_moe_weight,
        expert_weights,
        shared_expert_weights,
    )

    torch.testing.assert_close(
        torch_output,
        torch.from_dlpack(max_output[0]).to(torch.bfloat16),
        rtol=1e-3,
        atol=1e-6,
    )
