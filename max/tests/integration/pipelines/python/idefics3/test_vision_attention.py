# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test script comparing MAX Idefics3VisionAttention vs PyTorch implementation."""

from types import SimpleNamespace
from typing import Optional

import pytest
import torch
from max.driver import Accelerator, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.idefics3.vision_model.attention import (
    Idefics3VisionAttention,
)
from torch.utils.dlpack import from_dlpack
from transformers.models.idefics3.modeling_idefics3 import (
    Idefics3VisionAttention as PyTorchIdefics3VisionAttention,
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
    "_attn_implementation": "eager",  # Use eager attention for testing
    "attention_dropout": 0.0,  # Add missing attention_dropout
}


def create_config_object(config_dict: dict) -> SimpleNamespace:
    """Create a config object from dictionary that HuggingFace expects."""
    config = SimpleNamespace()
    for key, value in config_dict.items():
        setattr(config, key, value)
    return config


def create_idefics3_attention_weights(
    hidden_size: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Create shared attention weights and biases for both PyTorch and MAX models."""
    torch.manual_seed(42)
    std = IDEFICS3_VISION_CONFIG["initializer_range"]  # 0.02

    weights = {}

    # Create separate Q, K, V weights (since stacked_qkv=False)
    weights["q_proj.weight"] = std * torch.randn(
        hidden_size, hidden_size, dtype=dtype, device=device
    )
    weights["k_proj.weight"] = std * torch.randn(
        hidden_size, hidden_size, dtype=dtype, device=device
    )
    weights["v_proj.weight"] = std * torch.randn(
        hidden_size, hidden_size, dtype=dtype, device=device
    )

    # Create separate Q, K, V biases (since qkv_has_bias=True)
    weights["q_proj.bias"] = std * torch.randn(
        hidden_size, dtype=dtype, device=device
    )
    weights["k_proj.bias"] = std * torch.randn(
        hidden_size, dtype=dtype, device=device
    )
    weights["v_proj.bias"] = std * torch.randn(
        hidden_size, dtype=dtype, device=device
    )

    # Output projection weight and bias (both PyTorch and MAX now use out_proj)
    weights["out_proj.weight"] = std * torch.randn(
        hidden_size, hidden_size, dtype=dtype, device=device
    )
    weights["out_proj.bias"] = std * torch.randn(
        hidden_size, dtype=dtype, device=device
    )

    return weights


@torch.no_grad()
def generate_pytorch_outputs(
    config: dict,
    attention_weights: dict[str, torch.Tensor],
    input_tensor: torch.Tensor,
    output_attentions: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Generate outputs using HuggingFace PyTorch Idefics3VisionAttention."""
    config_obj = create_config_object(config)
    torch_attention = PyTorchIdefics3VisionAttention(config_obj).cuda()

    # Load weights and biases
    torch_attention.q_proj.weight.data = attention_weights["q_proj.weight"]
    torch_attention.k_proj.weight.data = attention_weights["k_proj.weight"]
    torch_attention.v_proj.weight.data = attention_weights["v_proj.weight"]
    torch_attention.out_proj.weight.data = attention_weights["out_proj.weight"]

    torch_attention.q_proj.bias.data = attention_weights["q_proj.bias"]
    torch_attention.k_proj.bias.data = attention_weights["k_proj.bias"]
    torch_attention.v_proj.bias.data = attention_weights["v_proj.bias"]
    torch_attention.out_proj.bias.data = attention_weights["out_proj.bias"]

    torch_attention.eval()

    return torch_attention(input_tensor, output_attentions=output_attentions)


def generate_max_outputs(
    config: dict,
    attention_weights: dict[str, torch.Tensor],
    input_tensor: torch.Tensor,
    dtype: DType = DType.bfloat16,
) -> torch.Tensor:
    """Generate outputs using MAX Idefics3VisionAttention."""
    cuda = Accelerator()
    session = InferenceSession(devices=[cuda])

    attention = Idefics3VisionAttention(
        hidden_size=config["hidden_size"],
        num_attention_heads=config["num_attention_heads"],
        devices=[DeviceRef.GPU()],
        dtype=dtype,
    )

    # Prepare state dict for MAX (now using out_proj to match PyTorch naming)
    state_dict = {
        "q_proj.weight": attention_weights["q_proj.weight"].cpu(),
        "k_proj.weight": attention_weights["k_proj.weight"].cpu(),
        "v_proj.weight": attention_weights["v_proj.weight"].cpu(),
        "out_proj.weight": attention_weights["out_proj.weight"].cpu(),
        "q_proj.bias": attention_weights["q_proj.bias"].cpu(),
        "k_proj.bias": attention_weights["k_proj.bias"].cpu(),
        "v_proj.bias": attention_weights["v_proj.bias"].cpu(),
        "out_proj.bias": attention_weights["out_proj.bias"].cpu(),
    }

    attention.load_state_dict(state_dict)

    batch_size, seq_len, hidden_size = input_tensor.shape

    def build_attention_model() -> Graph:
        with Graph(
            "idefics3_attention_test",
            input_types=[
                TensorType(
                    dtype=dtype,
                    shape=[batch_size, seq_len, hidden_size],
                    device=DeviceRef.GPU(),
                )
            ],
        ) as graph:
            inputs = graph.inputs[0]
            output = attention(inputs.tensor)
            graph.output(output)
        return graph

    graph = build_attention_model()
    model = session.load(graph, weights_registry=attention.state_dict())

    x_tensor = Tensor.from_dlpack(input_tensor).to(cuda)
    max_output = model.execute(x_tensor)[0]

    return max_output


@pytest.mark.parametrize(
    "batch_size,seq_len,dtype",
    [
        (1, 64, DType.bfloat16),  # Small sequence, bfloat16
    ],
)
@torch.no_grad()
def test_idefics3_attention_basic(
    batch_size: int,
    seq_len: int,
    dtype: DType,
) -> None:
    """Test basic functionality with Idefics3 config parameters."""
    print(
        f"Testing Idefics3VisionAttention: batch_size={batch_size}, seq_len={seq_len}, dtype={dtype}"
    )

    config = IDEFICS3_VISION_CONFIG
    hidden_size = config["hidden_size"]  # 1152

    torch_dtype = torch.bfloat16 if dtype == DType.bfloat16 else torch.float32

    # Create shared weights
    attention_weights = create_idefics3_attention_weights(
        hidden_size=hidden_size,  # type: ignore[arg-type]
        dtype=torch_dtype,
        device="cuda",
    )

    # Create input tensor
    torch.manual_seed(42)
    input_tensor = torch.randn(
        batch_size, seq_len, hidden_size, dtype=torch_dtype, device="cuda"
    )

    # Generate outputs
    torch_output, _ = generate_pytorch_outputs(
        config=config,
        attention_weights=attention_weights,
        input_tensor=input_tensor,
        output_attentions=False,
    )

    max_output = generate_max_outputs(
        config=config,
        attention_weights=attention_weights,
        input_tensor=input_tensor,
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

    torch.testing.assert_close(
        torch_output,
        max_output_torch,
        rtol=2 * torch.finfo(torch.bfloat16).eps,
        atol=8 * torch.finfo(torch.bfloat16).eps,
    )
    print("PASSED")


if __name__ == "__main__":
    pytest.main([__file__])
