# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test script comparing MAX Qwen2.5VL PatchMerger vs PyTorch implementation."""

import pytest
import torch
from max.driver import Accelerator, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.qwen2_5vl.nn.visual_transformer import (
    PatchMerger as MAXPatchMerger,
)
from torch.utils.dlpack import from_dlpack
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLPatchMerger as PyTorchQwen2_5VLPatchMerger,
)
from utils.config_loader import ConfigNames, get_config_loader
from utils.weight_converter import patch_merger_MAX_to_HF
from utils.weight_generator import get_weight_generator


@torch.no_grad()
def generate_pytorch_outputs(
    hidden_size: int,
    out_hidden_size: int,
    spatial_merge_size: int,
    mlp_weights: dict[str, torch.Tensor],
    input_tensor: torch.Tensor,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Generate outputs using PyTorch PatchMerger."""
    # Create PyTorch Qwen2.5VL PatchMerger
    torch_patch_merger = PyTorchQwen2_5VLPatchMerger(
        dim=out_hidden_size,
        context_dim=hidden_size,
        spatial_merge_size=spatial_merge_size,
    )

    state_dict = patch_merger_MAX_to_HF(mlp_weights)

    # Load weights using load_state_dict
    torch_patch_merger.load_state_dict(state_dict, strict=True)

    # Move to CUDA and convert to target dtype
    torch_patch_merger = torch_patch_merger.cuda().to(torch_dtype)

    return torch_patch_merger(input_tensor)


def generate_max_outputs(
    hidden_size: int,
    out_hidden_size: int,
    spatial_merge_size: int,
    mlp_weights: dict[str, torch.Tensor],
    input_tensor: torch.Tensor,
    dtype: DType = DType.bfloat16,
) -> torch.Tensor:
    """Generate outputs using MAX Qwen2.5VL PatchMerger."""
    cuda = Accelerator()
    session = InferenceSession(devices=[cuda])

    patch_merger = MAXPatchMerger(
        hidden_size=hidden_size,
        out_hidden_size=out_hidden_size,
        spatial_merge_size=spatial_merge_size,
        dtype=dtype,
        device=DeviceRef.GPU(),
    )

    # Prepare state dict for MAX (always includes bias)
    state_dict = {
        "norm.weight": mlp_weights["norm.weight"].cpu(),
        "linear1.weight": mlp_weights["mlp.0.weight"].cpu(),
        "linear2.weight": mlp_weights["mlp.2.weight"].cpu(),
        "linear1.bias": mlp_weights["mlp.0.bias"].cpu(),
        "linear2.bias": mlp_weights["mlp.2.bias"].cpu(),
    }

    patch_merger.load_state_dict(state_dict)

    # Get input dimensions for 2D tensor (total_tokens, hidden_size)
    input_dims = input_tensor.shape

    def build_patch_merger_model() -> Graph:
        with Graph(
            "qwen25vl_patch_merger_test",
            input_types=[
                TensorType(
                    dtype=dtype,
                    shape=list(input_dims),
                    device=DeviceRef.GPU(),
                )
            ],
        ) as graph:
            inputs = graph.inputs[0]
            output = patch_merger(inputs.tensor)
            graph.output(output)
        return graph

    graph = build_patch_merger_model()
    model = session.load(graph, weights_registry=patch_merger.state_dict())

    x_tensor = Tensor.from_dlpack(input_tensor).to(cuda)
    max_output = model.execute(x_tensor)[0]

    return max_output


@pytest.mark.parametrize(
    "config_name",
    [
        pytest.param(ConfigNames.QWEN2_5VL_3B),
    ],
)
@pytest.mark.parametrize(
    "total_tokens,dtype",
    [
        (1024, DType.bfloat16),
        (4096, DType.bfloat16),
        (22972, DType.bfloat16),  # Real-world size from log
    ],
)
@torch.no_grad()
def test_qwen25vl_patch_merger_2d(
    config_name: ConfigNames,
    total_tokens: int,
    dtype: DType,
) -> None:
    """Test PatchMerger with 2D input tensors (total_tokens, hidden_size)."""
    print(
        f"Testing Qwen2.5VL PatchMerger 2D: total_tokens={total_tokens}, dtype={dtype}"
    )

    # Load config and generate weights
    loader = get_config_loader()
    hf_vision_config = loader.load_hf_vision_config(config_name)
    qwen2_5vl_config = loader.create_qwen2_5vl_config(config_name)

    torch_dtype = torch.bfloat16 if dtype == DType.bfloat16 else torch.float32

    # Create shared weights (MAX PatchMerger always has bias)
    mlp_weights = get_weight_generator(
        config_name
    ).generate_vision_patch_merger_weights()

    hidden_size = qwen2_5vl_config["vision_config"]["hidden_size"]
    out_hidden_size = qwen2_5vl_config["vision_config"]["out_hidden_size"]
    spatial_merge_size = qwen2_5vl_config["vision_config"]["spatial_merge_size"]

    # Create 2D input tensor (total_tokens, hidden_size) - matches real usage
    torch.manual_seed(42)
    input_tensor = torch.randn(
        total_tokens, hidden_size, dtype=torch_dtype, device="cuda"
    )

    # Generate outputs (both implementations always use bias)
    torch_output = generate_pytorch_outputs(
        hidden_size=hidden_size,
        out_hidden_size=out_hidden_size,
        spatial_merge_size=spatial_merge_size,
        mlp_weights=mlp_weights,
        input_tensor=input_tensor,
        torch_dtype=torch_dtype,
    )

    max_output = generate_max_outputs(
        hidden_size=hidden_size,
        out_hidden_size=out_hidden_size,
        spatial_merge_size=spatial_merge_size,
        mlp_weights=mlp_weights,
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

    # Verify shapes match
    assert torch_output.shape == max_output_torch.shape, (
        f"Shape mismatch: {torch_output.shape} vs {max_output_torch.shape}"
    )

    # Verify output dimensions are correct
    expected_output_tokens = total_tokens // (spatial_merge_size**2)
    expected_output_shape = (expected_output_tokens, out_hidden_size)
    assert torch_output.shape == expected_output_shape, (
        f"Unexpected output shape: {torch_output.shape}, expected {expected_output_shape}"
    )

    # Check if outputs are close with appropriate tolerance

    rtol = 2 * torch.finfo(torch.bfloat16).eps
    atol = 8 * torch.finfo(torch.bfloat16).eps

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
