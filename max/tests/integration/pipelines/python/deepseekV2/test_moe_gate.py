# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import json
import os
from pathlib import Path

import pytest
import torch
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, Weight
from max.pipelines.architectures.deepseekV2.layers.moe_gate import MaxMoEGate
from max.pipelines.nn import Linear
from torch.utils.dlpack import from_dlpack
from torch_reference.configuration_deepseek import (
    DeepseekV2Config,
)
from torch_reference.modeling_deepseek import MoEGate


@pytest.fixture
def config() -> DeepseekV2Config:
    config = DeepseekV2Config()
    path = os.getenv("PIPELINES_TESTDATA")
    config_path = Path(path) / "config.json"  # type: ignore
    with open(config_path, "r") as file:
        data = json.load(file)
    config.update(data)
    return config


# TODO: Replace with real weights using indexing included in the upcoming Model API.
@pytest.fixture
def dummy_moe_weight(config: DeepseekV2Config) -> torch.Tensor:
    """
    Fixture to create dummy weights for an MLP layer.
    Returns tensors in bfloat16 format.
    """
    return torch.randn((64, config.hidden_size), dtype=torch.float32)


@pytest.fixture
def input_tensor(config: DeepseekV2Config) -> torch.Tensor:
    return torch.randn(
        1,
        1,
        config.hidden_size,
        dtype=torch.bfloat16,
    )


def test_moe_gate(
    config: DeepseekV2Config,
    input_tensor: torch.Tensor,
    dummy_moe_weight: torch.Tensor,
) -> None:
    # Generate torch outputs

    layer = MoEGate(config).to(torch.bfloat16)
    layer.weight.data = dummy_moe_weight
    torch_output = layer(input_tensor)
    torch_topk_idxs = torch_output[0].to(torch.bfloat16)
    torch_topk_weights = torch_output[1].to(torch.bfloat16)

    # Create MAX Weights
    weights_registry = {}
    weights_registry["gate_weight"] = dummy_moe_weight.to(torch.float32)

    gate_weight = Weight(
        name="gate_weight",
        dtype=DType.float32,
        shape=weights_registry["gate_weight"].shape,
    )

    # Generate MAX outputs
    session = InferenceSession()
    graph = Graph(
        "MoEGate",
        MaxMoEGate(Linear(gate_weight)),
        input_types=(
            TensorType(
                DType.bfloat16,
                (1, 1, config.hidden_size),
            ),
        ),
    )

    compiled = session.load(graph, weights_registry=weights_registry)

    max_output = compiled.execute(input_tensor)
    max_topk_idxs = from_dlpack(max_output[0]).to(torch.bfloat16)
    max_topk_weights = from_dlpack(max_output[1]).to(torch.bfloat16)

    # Top_k does not return values in sorted order, so we sort outputs here to compare.
    torch.testing.assert_close(
        torch.sort(torch_topk_idxs.squeeze(0))[0],
        torch.sort(max_topk_idxs)[0],
        rtol=1e-3,
        atol=1e-6,
    )

    torch.testing.assert_close(
        torch.sort(torch_topk_weights.squeeze(0))[0],
        torch.sort(max_topk_weights)[0],
        rtol=1e-3,
        atol=1e-6,
    )
