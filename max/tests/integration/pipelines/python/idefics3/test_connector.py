# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test script comparing MAX Idefics3Connector vs PyTorch implementation."""

import math
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from max.driver import Accelerator, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.idefics3.vision_model.connector import (
    Idefics3Connector as MAXIdefics3Connector,
)
from torch.utils.dlpack import from_dlpack
from transformers.models.idefics3.modeling_idefics3 import (
    Idefics3Connector as PyTorchIdefics3Connector,
)


@dataclass
class Idefics3VisionConfig:
    """Configuration for Idefics3 Vision Model."""

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


@dataclass
class Idefics3TextConfig:
    """Configuration for Idefics3 Text Model."""

    hidden_size: int = 4096
    num_attention_heads: int = 32
    hidden_act: str = "silu"
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    initializer_range: float = 0.02
    vocab_size: int = 128259


@dataclass
class Idefics3Config:
    """Complete configuration for Idefics3 Model."""

    scale_factor: int = 2
    vision_config: Idefics3VisionConfig = None  # type: ignore[assignment]
    text_config: Idefics3TextConfig = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.vision_config is None:
            self.vision_config = Idefics3VisionConfig()
        if self.text_config is None:
            self.text_config = Idefics3TextConfig()


def create_connector_weights(
    vision_hidden_size: int,
    text_hidden_size: int,
    scale_factor: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Create shared connector weights for both PyTorch and MAX models."""
    torch.manual_seed(42)
    std = 0.02  # initializer_range

    # Calculate input size after pixel shuffle
    input_size = vision_hidden_size * (scale_factor**2)

    weights = {}

    # Modality projection weights (no bias)
    weights["modality_projection.proj.weight"] = std * torch.randn(
        text_hidden_size, input_size, dtype=dtype, device=device
    )

    return weights


@torch.no_grad()
def generate_pytorch_outputs(
    config: Idefics3Config,
    connector_weights: dict[str, torch.Tensor],
    input_tensor: torch.Tensor,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Generate outputs using PyTorch Connector."""
    # Create PyTorch config object
    pytorch_config = SimpleNamespace()
    pytorch_config.scale_factor = config.scale_factor
    pytorch_config.vision_config = SimpleNamespace()
    pytorch_config.vision_config.hidden_size = config.vision_config.hidden_size
    pytorch_config.text_config = SimpleNamespace()
    pytorch_config.text_config.hidden_size = config.text_config.hidden_size

    torch_connector = (
        PyTorchIdefics3Connector(pytorch_config).cuda().to(torch_dtype)
    )

    # Load weights
    torch_connector.modality_projection.proj.weight.data = connector_weights[
        "modality_projection.proj.weight"
    ]

    return torch_connector(input_tensor)


def generate_max_outputs(
    config: Idefics3Config,
    connector_weights: dict[str, torch.Tensor],
    input_tensor: torch.Tensor,
    dtype: DType = DType.bfloat16,
) -> torch.Tensor:
    """Generate outputs using MAX Idefics3Connector."""
    cuda = Accelerator()
    session = InferenceSession(devices=[cuda])

    # Create a config object that matches what Idefics3SimpleMLP expects
    # The MLP constructor expects: config.hidden_size, config.scale_factor, and config.text_config_hidden_size
    mlp_config = SimpleNamespace()
    mlp_config.hidden_size = (
        config.vision_config.hidden_size
    )  # What the MLP expects as input size base
    mlp_config.scale_factor = config.scale_factor
    mlp_config.text_config_hidden_size = (
        config.text_config.hidden_size
    )  # Output size

    connector = MAXIdefics3Connector(
        config=mlp_config,  # type: ignore[arg-type]
        dtype=dtype,
        device=DeviceRef.GPU(),
    )

    # Prepare state dict for MAX
    state_dict = {
        "modality_projection.proj.weight": connector_weights[
            "modality_projection.proj.weight"
        ].cpu(),
    }

    connector.load_state_dict(state_dict)

    # Get input dimensions
    input_dims = input_tensor.shape
    batch_size, seq_len, vision_hidden_size = input_dims

    def build_connector_model() -> Graph:
        with Graph(
            "idefics3_connector_test",
            input_types=[
                TensorType(
                    dtype=dtype,
                    shape=list(input_dims),
                    device=DeviceRef.GPU(),
                )
            ],
        ) as graph:
            inputs = graph.inputs[0]
            output = connector(inputs.tensor)
            graph.output(output)
        return graph

    graph = build_connector_model()
    model = session.load(graph, weights_registry=connector.state_dict())

    x_tensor = Tensor.from_dlpack(input_tensor).to(cuda)
    max_output = model.execute(x_tensor)[0]

    return max_output


@pytest.mark.parametrize(
    "batch_size,seq_len,dtype",
    [
        # 8x8 patch grid -> 16x16 tokens after shuffle
        (20, 676, DType.bfloat16),
        # 4x4 patch grid -> 4x4 tokens after shuffle
        (100, 676, DType.bfloat16),
        # 12x12 patch grid -> 36x36 tokens after shuffle
        (140, 676, DType.bfloat16),
    ],
)
@torch.no_grad()
def test_idefics3_connector(
    batch_size: int,
    seq_len: int,
    dtype: DType,
) -> None:
    """Test Idefics3 Connector with various input sizes."""
    print(
        f"Testing Idefics3Connector: batch_size={batch_size}, seq_len={seq_len}, dtype={dtype}"
    )

    # Verify seq_len is a perfect square
    sqrt_seq_len = int(math.sqrt(seq_len))
    assert sqrt_seq_len * sqrt_seq_len == seq_len, (
        f"seq_len {seq_len} must be a perfect square"
    )

    # Create config
    config = Idefics3Config()
    vision_hidden_size = config.vision_config.hidden_size  # 1152
    text_hidden_size = config.text_config.hidden_size  # 4096
    scale_factor = config.scale_factor  # 2

    torch_dtype = torch.bfloat16 if dtype == DType.bfloat16 else torch.float32

    # Create shared weights
    connector_weights = create_connector_weights(
        vision_hidden_size=vision_hidden_size,
        text_hidden_size=text_hidden_size,
        scale_factor=scale_factor,
        dtype=torch_dtype,
        device="cuda",
    )

    # Create input tensor (batch_size, seq_len, vision_hidden_size)
    torch.manual_seed(42)
    input_tensor = torch.randn(
        batch_size,
        seq_len,
        vision_hidden_size,
        dtype=torch_dtype,
        device="cuda",
    )

    # Generate outputs
    torch_output = generate_pytorch_outputs(
        config=config,
        connector_weights=connector_weights,
        input_tensor=input_tensor,
        torch_dtype=torch_dtype,
    )

    max_output = generate_max_outputs(
        config=config,
        connector_weights=connector_weights,
        input_tensor=input_tensor,
        dtype=dtype,
    )

    # Compare outputs
    max_output_torch = from_dlpack(max_output)

    print(f"Input shape: {input_tensor.shape}")
    print(f"PyTorch output shape: {torch_output.shape}")
    print(f"MAX output shape: {max_output_torch.shape}")
    print(f"PyTorch output mean: {torch_output.mean().item():.6f}")
    print(f"MAX output mean: {max_output_torch.mean().item():.6f}")

    # Expected output shape: (batch_size, seq_len/(scale_factor^2), text_hidden_size)
    expected_seq_len = seq_len // (scale_factor**2)
    expected_shape = (batch_size, expected_seq_len, text_hidden_size)

    print(f"Expected output shape: {expected_shape}")

    # Verify shapes match
    assert torch_output.shape == expected_shape, (
        f"PyTorch output shape mismatch: {torch_output.shape} vs {expected_shape}"
    )
    assert max_output_torch.shape == expected_shape, (
        f"MAX output shape mismatch: {max_output_torch.shape} vs {expected_shape}"
    )
    assert torch_output.shape == max_output_torch.shape, (
        f"Shape mismatch: {torch_output.shape} vs {max_output_torch.shape}"
    )

    # Check if outputs are close with appropriate tolerance
    rtol = torch.finfo(torch.bfloat16).eps
    atol = torch.finfo(torch.bfloat16).eps

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
