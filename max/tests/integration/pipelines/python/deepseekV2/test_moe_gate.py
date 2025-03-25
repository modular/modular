# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import torch
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.deepseekV2.layers.moe_gate import MaxMoEGate
from torch.utils.dlpack import from_dlpack
from torch_reference.configuration_deepseek import (
    DeepseekV2Config,
)
from torch_reference.modeling_deepseek import MoEGate


def generate_torch_outputs(
    config: DeepseekV2Config,
    input_tensor: torch.Tensor,
    dummy_moe_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    layer = MoEGate(config).to(torch.bfloat16)
    layer.weight.data = dummy_moe_weight
    torch_output = layer(input_tensor)
    torch_topk_idxs = torch_output[0].to(torch.bfloat16)
    torch_topk_weights = torch_output[1].to(torch.bfloat16)
    return torch_topk_idxs, torch_topk_weights


def generate_max_outputs(
    config: DeepseekV2Config,
    input_tensor: torch.Tensor,
    dummy_moe_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    state_dict = {"gate_score.weight": dummy_moe_weight.to(torch.float32).cpu()}
    model = MaxMoEGate()
    model.load_state_dict(state_dict)
    session = InferenceSession()
    graph = Graph(
        "MoEGate",
        model,
        input_types=(
            TensorType(
                DType.bfloat16,
                (
                    input_tensor.shape[0],
                    input_tensor.shape[1],
                    config.hidden_size,
                ),
                DeviceRef.CPU(),
            ),
        ),
    )

    compiled = session.load(graph, weights_registry=model.state_dict())
    max_output = compiled.execute(input_tensor)
    max_topk_idxs = from_dlpack(max_output[0]).to(torch.bfloat16)
    max_topk_weights = from_dlpack(max_output[1]).to(torch.bfloat16)
    return max_topk_idxs, max_topk_weights


def test_moe_gate(
    config: DeepseekV2Config,
    input_tensor: torch.Tensor,
    dummy_moe_weight: torch.Tensor,
) -> None:
    torch_topk_idxs, torch_topk_weights = generate_torch_outputs(
        config, input_tensor, dummy_moe_weight
    )
    max_topk_idxs, max_topk_weights = generate_max_outputs(
        config, input_tensor, dummy_moe_weight
    )

    # Top_k does not return values in sorted order, so we sort outputs here to compare.
    torch.testing.assert_close(
        torch.sort(torch_topk_idxs)[0],
        torch.sort(max_topk_idxs)[0],
        rtol=1e-3,
        atol=1e-6,
    )

    torch.testing.assert_close(
        torch.sort(torch_topk_weights)[0],
        torch.sort(max_topk_weights)[0],
        rtol=1e-3,
        atol=1e-6,
    )
