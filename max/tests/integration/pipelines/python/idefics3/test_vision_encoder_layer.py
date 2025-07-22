# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Test script comparing MAX Idefics3VisionEncoderLayer vs PyTorch implementation."""

from dataclasses import dataclass

import pytest
import torch
from max.driver import Accelerator, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.idefics3.vision_model.encoder import (
    Idefics3VisionEncoderLayer as MAXIdefics3VisionEncoderLayer,
)
from torch.utils.dlpack import from_dlpack
from transformers.models.idefics3.modeling_idefics3 import (
    Idefics3EncoderLayer as PyTorchIdefics3VisionEncoderLayer,
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
    "_attn_implementation": "eager",
    "attention_dropout": 0.0,
}


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
    _attn_implementation: str = "eager"
    attention_dropout: float = 0.0  # Add missing attention_dropout


def create_encoder_layer_weights(
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Create shared encoder layer weights for both PyTorch and MAX models."""
    torch.manual_seed(42)
    std = IDEFICS3_VISION_CONFIG["initializer_range"]  # 0.02
    head_dim = hidden_size // num_attention_heads

    weights = {}

    # Attention layer weights - separate for PyTorch
    weights["self_attn.q_proj.weight"] = std * torch.randn(
        hidden_size, hidden_size, dtype=dtype, device=device
    )
    weights["self_attn.q_proj.bias"] = torch.zeros(
        hidden_size, dtype=dtype, device=device
    )

    weights["self_attn.k_proj.weight"] = std * torch.randn(
        hidden_size, hidden_size, dtype=dtype, device=device
    )
    weights["self_attn.k_proj.bias"] = torch.zeros(
        hidden_size, dtype=dtype, device=device
    )

    weights["self_attn.v_proj.weight"] = std * torch.randn(
        hidden_size, hidden_size, dtype=dtype, device=device
    )
    weights["self_attn.v_proj.bias"] = torch.zeros(
        hidden_size, dtype=dtype, device=device
    )

    weights["self_attn.out_proj.weight"] = std * torch.randn(
        hidden_size, hidden_size, dtype=dtype, device=device
    )
    weights["self_attn.out_proj.bias"] = torch.zeros(
        hidden_size, dtype=dtype, device=device
    )

    # MLP weights
    weights["mlp.fc1.weight"] = std * torch.randn(
        intermediate_size, hidden_size, dtype=dtype, device=device
    )
    weights["mlp.fc2.weight"] = std * torch.randn(
        hidden_size, intermediate_size, dtype=dtype, device=device
    )

    # Add missing MLP bias weights
    weights["mlp.fc1.bias"] = torch.zeros(
        intermediate_size, dtype=dtype, device=device
    )
    weights["mlp.fc2.bias"] = torch.zeros(
        hidden_size, dtype=dtype, device=device
    )

    # Layer normalization weights
    weights["layer_norm1.weight"] = torch.ones(
        hidden_size, dtype=dtype, device=device
    )
    weights["layer_norm2.weight"] = torch.ones(
        hidden_size, dtype=dtype, device=device
    )

    # Add missing LayerNorm bias weights
    weights["layer_norm1.bias"] = torch.zeros(
        hidden_size, dtype=dtype, device=device
    )
    weights["layer_norm2.bias"] = torch.zeros(
        hidden_size, dtype=dtype, device=device
    )

    return weights


@torch.no_grad()
def generate_pytorch_outputs(
    vision_config: Idefics3VisionConfig,
    encoder_weights: dict[str, torch.Tensor],
    input_tensor: torch.Tensor,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Generate outputs using PyTorch encoder layer."""
    torch_encoder = PyTorchIdefics3VisionEncoderLayer(vision_config).cuda()

    # Convert model to correct dtype BEFORE loading weights
    torch_encoder = torch_encoder.to(torch_dtype)

    # Load weights into PyTorch model
    torch_encoder.load_state_dict(encoder_weights, strict=True)

    # Set to eval mode
    torch_encoder.eval()

    # PyTorch encoder layer returns (hidden_states,) tuple, extract the first element
    outputs = torch_encoder(input_tensor, attention_mask=None)
    return outputs[0]  # Extract hidden_states from tuple


def generate_max_outputs(
    vision_config: Idefics3VisionConfig,
    encoder_weights: dict[str, torch.Tensor],
    input_tensor: torch.Tensor,
    dtype: DType = DType.bfloat16,
) -> torch.Tensor:
    """Generate outputs using MAX Idefics3VisionEncoderLayer."""
    cuda = Accelerator()
    session = InferenceSession(devices=[cuda])

    max_encoder = MAXIdefics3VisionEncoderLayer(
        vision_config=vision_config,  # type: ignore[arg-type]
        dtype=dtype,
        device=DeviceRef.GPU(),
    )

    # Convert weights to CPU before loading (MAX expects CPU weights)
    cpu_weights = {key: weight.cpu() for key, weight in encoder_weights.items()}
    max_encoder.load_state_dict(cpu_weights)

    # Get input dimensions
    input_dims = input_tensor.shape
    batch_size, seq_len, hidden_size = input_dims

    def build_encoder_model() -> Graph:
        with Graph(
            "idefics3_encoder_layer_test",
            input_types=[
                TensorType(
                    dtype=dtype,
                    shape=list(input_dims),
                    device=DeviceRef.GPU(),
                ),
            ],
        ) as graph:
            hidden_states = graph.inputs[0]
            output = max_encoder(hidden_states.tensor)
            graph.output(output)
        return graph

    graph = build_encoder_model()
    model = session.load(graph, weights_registry=max_encoder.state_dict())

    x_tensor = Tensor.from_dlpack(input_tensor).to(cuda)
    max_output = model.execute(x_tensor)[0]

    return max_output


@pytest.mark.parametrize(
    "batch_size,seq_len,dtype",
    [
        (1, 16, DType.bfloat16),  # Small sequence
        (2, 32, DType.bfloat16),  # Medium sequence
        (1, 64, DType.bfloat16),  # Larger sequence
    ],
)
@torch.no_grad()
def test_idefics3_vision_encoder_layer(
    batch_size: int,
    seq_len: int,
    dtype: DType,
) -> None:
    """Test encoder layer with 3D input tensors (batch_size, seq_len, hidden_size)."""
    print(
        f"Testing Idefics3VisionEncoderLayer: batch_size={batch_size}, seq_len={seq_len}, dtype={dtype}"
    )

    config = IDEFICS3_VISION_CONFIG
    hidden_size = config["hidden_size"]  # 1152
    intermediate_size = config["intermediate_size"]  # 4304
    num_attention_heads = config["num_attention_heads"]  # 16

    torch_dtype = torch.bfloat16 if dtype == DType.bfloat16 else torch.float32

    # Create vision config object
    vision_config = Idefics3VisionConfig()

    # Create shared weights
    encoder_weights = create_encoder_layer_weights(
        hidden_size=hidden_size,  # type: ignore[arg-type]
        intermediate_size=intermediate_size,  # type: ignore[arg-type]
        num_attention_heads=num_attention_heads,  # type: ignore[arg-type]
        dtype=torch_dtype,
        device="cuda",
    )

    # Create input tensor (batch_size, seq_len, hidden_size)
    torch.manual_seed(42)
    input_tensor = torch.randn(
        batch_size, seq_len, hidden_size, dtype=torch_dtype, device="cuda"
    )

    # Generate outputs
    torch_output = generate_pytorch_outputs(
        vision_config=vision_config,
        encoder_weights=encoder_weights,
        input_tensor=input_tensor,
        torch_dtype=torch_dtype,
    )

    max_output = generate_max_outputs(
        vision_config=vision_config,
        encoder_weights=encoder_weights,
        input_tensor=input_tensor,
        dtype=dtype,
    )

    # Compare outputs
    max_output_torch = from_dlpack(max_output)

    print(f"PyTorch output shape: {torch_output.shape}")
    print(f"MAX output shape: {max_output_torch.shape}")
    print(f"PyTorch output mean: {torch_output.mean().item():.6f}")
    print(f"MAX output mean: {max_output_torch.mean().item():.6f}")
    print(f"PyTorch output std: {torch_output.std().item():.6f}")
    print(f"MAX output std: {max_output_torch.std().item():.6f}")

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
