# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test script comparing MAX Idefics3VisionMLP vs PyTorch implementation."""

from dataclasses import dataclass

import pytest
import torch
from max.driver import Accelerator, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.idefics3.vision_model.encoder import (
    Idefics3VisionMLP as MAXIdefics3VisionMLP,
)
from torch.utils.dlpack import from_dlpack
from transformers.models.idefics3.modeling_idefics3 import (
    Idefics3VisionMLP as PyTorchIdefics3VisionMLP,
)

# Idefics3 Vision Config (from HuggingFace)
IDEFICS3_VISION_CONFIG = {
    "hidden_size": 1152,
    "num_attention_heads": 16,
    "layer_norm_eps": 1e-06,
    "hidden_act": "gelu_pytorch_tanh",
    "intermediate_size": 4304,
    "image_size": 364,
    "patch_size": 14,
    "num_channels": 3,
    "num_hidden_layers": 27,
    "initializer_range": 0.02,
}


@dataclass
class Idefics3VisionConfig:
    """Configuration for Idefics3 Vision Model.

    This dataclass encapsulates all the configuration parameters needed
    for the Idefics3 vision encoder, including model dimensions, attention
    settings, and layer specifications.
    """

    hidden_size: int = 1152
    num_attention_heads: int = 16
    layer_norm_eps: float = 1e-06
    hidden_act: str = "gelu_pytorch_tanh"
    intermediate_size: int = 4304
    image_size: int = 364
    patch_size: int = 14
    num_channels: int = 3
    num_hidden_layers: int = 27
    initializer_range: float = 0.02


def create_mlp_weights(
    hidden_size: int,
    intermediate_size: int,
    has_bias: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Create shared MLP weights for both PyTorch and MAX models."""
    torch.manual_seed(42)
    std = IDEFICS3_VISION_CONFIG["initializer_range"]  # 0.02

    weights = {}

    # First linear layer weights
    weights["fc1.weight"] = std * torch.randn(
        intermediate_size, hidden_size, dtype=dtype, device=device
    )

    # Second linear layer weights
    weights["fc2.weight"] = std * torch.randn(
        hidden_size, intermediate_size, dtype=dtype, device=device
    )

    # Add bias tensors if requested
    if has_bias:
        weights["fc1.bias"] = std * torch.randn(
            intermediate_size, dtype=dtype, device=device
        )
        weights["fc2.bias"] = std * torch.randn(
            hidden_size, dtype=dtype, device=device
        )

    return weights


@torch.no_grad()
def generate_pytorch_outputs(
    hidden_size: int,
    intermediate_size: int,
    mlp_weights: dict[str, torch.Tensor],
    input_tensor: torch.Tensor,
    has_bias: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Generate outputs using PyTorch MLP."""
    # Create config object for PyTorch implementation
    config = Idefics3VisionConfig()
    torch_mlp = PyTorchIdefics3VisionMLP(config).cuda().to(torch_dtype)

    # Load weights
    torch_mlp.fc1.weight.data = mlp_weights["fc1.weight"]
    torch_mlp.fc2.weight.data = mlp_weights["fc2.weight"]

    # Load bias if present
    if has_bias:
        torch_mlp.fc1.bias.data = mlp_weights["fc1.bias"]
        torch_mlp.fc2.bias.data = mlp_weights["fc2.bias"]

    return torch_mlp(input_tensor)


def generate_max_outputs(
    hidden_size: int,
    intermediate_size: int,
    mlp_weights: dict[str, torch.Tensor],
    input_tensor: torch.Tensor,
    has_bias: bool = False,
    dtype: DType = DType.bfloat16,
) -> torch.Tensor:
    """Generate outputs using MAX Idefics3VisionMLP."""
    cuda = Accelerator()
    session = InferenceSession(devices=[cuda])

    mlp = MAXIdefics3VisionMLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=dtype,
        device=DeviceRef.GPU(),
        has_bias=has_bias,
    )

    # Prepare state dict for MAX
    state_dict = {
        "fc1.weight": mlp_weights["fc1.weight"].cpu(),
        "fc2.weight": mlp_weights["fc2.weight"].cpu(),
    }

    if has_bias:
        state_dict.update(
            {
                "fc1.bias": mlp_weights["fc1.bias"].cpu(),
                "fc2.bias": mlp_weights["fc2.bias"].cpu(),
            }
        )

    mlp.load_state_dict(state_dict)

    # Get input dimensions - handle both 2D and 3D tensors
    input_dims = input_tensor.shape
    if len(input_dims) == 2:
        batch_size, hidden_dim = input_dims
        seq_len = None
    else:  # 3D tensor
        batch_size, seq_len, hidden_dim = input_dims

    def build_mlp_model() -> Graph:
        with Graph(
            "idefics3_mlp_test",
            input_types=[
                TensorType(
                    dtype=dtype,
                    shape=list(input_dims),
                    device=DeviceRef.GPU(),
                )
            ],
        ) as graph:
            inputs = graph.inputs[0]
            output = mlp(inputs.tensor)
            graph.output(output)
        return graph

    graph = build_mlp_model()
    model = session.load(graph, weights_registry=mlp.state_dict())

    x_tensor = Tensor.from_dlpack(input_tensor).to(cuda)
    max_output = model.execute(x_tensor)[0]

    return max_output


@pytest.mark.parametrize(
    "batch_size,seq_len,has_bias,dtype",
    [
        (2, 16, False, DType.bfloat16),
        (2, 32, False, DType.bfloat16),
        # Add test cases with bias
        (2, 16, True, DType.bfloat16),
        (2, 32, True, DType.bfloat16),
    ],
)
@torch.no_grad()
def test_idefics3_vision_mlp_3d(
    batch_size: int,
    seq_len: int,
    has_bias: bool,
    dtype: DType,
) -> None:
    """Test MLP with 3D input tensors (batch_size, seq_len, hidden_size)."""
    print(
        f"Testing Idefics3VisionMLP 3D: batch_size={batch_size}, seq_len={seq_len}, has_bias={has_bias}, dtype={dtype}"
    )

    config = IDEFICS3_VISION_CONFIG
    hidden_size = config["hidden_size"]  # 1152
    intermediate_size = config["intermediate_size"]  # 4304

    torch_dtype = torch.bfloat16 if dtype == DType.bfloat16 else torch.float32

    # Create shared weights
    mlp_weights = create_mlp_weights(
        hidden_size=hidden_size,  # type: ignore[arg-type]
        intermediate_size=intermediate_size,  # type: ignore[arg-type]
        has_bias=has_bias,
        dtype=torch_dtype,
        device="cuda",
    )

    # Create 3D input tensor (batch_size, seq_len, hidden_size)
    torch.manual_seed(42)
    input_tensor = torch.randn(
        batch_size, seq_len, hidden_size, dtype=torch_dtype, device="cuda"
    )

    # Generate outputs
    torch_output = generate_pytorch_outputs(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        mlp_weights=mlp_weights,
        input_tensor=input_tensor,
        has_bias=has_bias,
        torch_dtype=torch_dtype,
    )

    max_output = generate_max_outputs(
        hidden_size=hidden_size,  # type: ignore[arg-type]
        intermediate_size=intermediate_size,  # type: ignore[arg-type]
        mlp_weights=mlp_weights,
        input_tensor=input_tensor,
        has_bias=has_bias,
        dtype=dtype,
    )

    # Compare outputs
    max_output_torch = from_dlpack(max_output)

    print(f"PyTorch output shape: {torch_output.shape}")
    print(f"MAX output shape: {max_output_torch.shape}")
    print(f"PyTorch output mean: {torch_output.mean().item():.6f}")
    print(f"MAX output mean: {max_output_torch.mean().item():.6f}")

    # Verify shapes match
    assert torch_output.shape == max_output_torch.shape, (
        f"Shape mismatch: {torch_output.shape} vs {max_output_torch.shape}"
    )

    # Check if outputs are close with appropriate tolerance

    rtol = 4 * torch.finfo(torch.bfloat16).eps
    atol = 16 * torch.finfo(torch.bfloat16).eps

    print(f"rtol: {rtol}")
    print(f"atol: {atol}")

    torch.testing.assert_close(
        torch_output,
        max_output_torch,
        rtol=rtol,
        atol=atol,
    )


if __name__ == "__main__":
    pytest.main([__file__])
