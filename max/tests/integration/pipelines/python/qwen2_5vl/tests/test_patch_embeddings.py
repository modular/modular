# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Tests for Qwen2.5VL vision patch embeddings."""

import pytest
import torch
from max.driver import Accelerator, Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.qwen2_5vl.nn.visual_transformer import (
    VisionPatchEmbed,
)
from torch.utils.dlpack import from_dlpack
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionPatchEmbed as HFQwen2_5VisionPatchEmbed,
)
from utils.assert_tensors import assert_tensors_close
from utils.config_loader import ConfigNames, get_config_loader
from utils.weight_converter import convert_hf_to_max_weights
from utils.weight_generator import get_weight_generator


@torch.no_grad()
def generate_torch_outputs(
    pixel_values: torch.Tensor,
    hf_vision_config: dict,
    embeddings_weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Generate reference outputs using the HuggingFace implementation."""

    # Create the reference model
    ref_model = (
        HFQwen2_5VisionPatchEmbed(
            patch_size=hf_vision_config["patch_size"],
            temporal_patch_size=hf_vision_config["temporal_patch_size"],
            in_channels=hf_vision_config.get("in_channels", 3),
            embed_dim=hf_vision_config["hidden_size"],
        )
        .to(torch.bfloat16)
        .to("cuda")
    )

    # Load weights
    ref_model.load_state_dict(embeddings_weights, strict=True)
    ref_model.eval()

    # Forward pass
    with torch.no_grad():
        output = ref_model(pixel_values)

    return output


def generate_max_outputs(
    pixel_values: torch.Tensor,
    qwen2_5vl_config: dict,
    embeddings_weights: dict[str, torch.Tensor],
    dtype: DType,
    device: Device,
) -> torch.Tensor:
    """Generate outputs using MAX VisionPatchEmbed implementation."""
    is_gpu = isinstance(device, Accelerator)
    device_ref = DeviceRef.GPU() if is_gpu else DeviceRef.CPU()
    pixel_values = pixel_values.cuda() if is_gpu else pixel_values.cpu()

    vision_config = qwen2_5vl_config["vision_config"]

    # Convert weights to MAX format
    max_weights = convert_hf_to_max_weights(embeddings_weights)

    # Create VisionPatchEmbed with new constructor pattern
    patch_embed_module = VisionPatchEmbed(
        dtype=dtype,
        device=device_ref,
        patch_size=vision_config["patch_size"],
        temporal_patch_size=vision_config["temporal_patch_size"],
        in_channels=vision_config.get("in_channels", 3),
        embed_dim=vision_config["hidden_size"],
        spatial_merge_unit=vision_config["spatial_merge_size"],
    )

    # Load weights using state_dict
    patch_embed_module.load_state_dict(max_weights, strict=True)

    session = InferenceSession(devices=[device])

    # Define input types
    pixel_values_type = TensorType(
        dtype, shape=pixel_values.shape, device=device_ref
    )

    seq_len = pixel_values.shape[0]
    spatial_merge_unit = vision_config["spatial_merge_size"]
    # window_index should have length seq_len // spatial_merge_unit
    window_index_len = seq_len // spatial_merge_unit
    window_index = torch.arange(
        window_index_len, dtype=torch.int64, device=pixel_values.device
    )
    window_index_type = TensorType(
        DType.int64, shape=window_index.shape, device=device_ref
    )

    with Graph(
        "VisionPatchEmbed", input_types=(pixel_values_type, window_index_type)
    ) as graph:
        x, window_idx = graph.inputs
        output = patch_embed_module(x.tensor, window_idx.tensor)
        graph.output(output)

    compiled = session.load(
        graph, weights_registry=patch_embed_module.state_dict()
    )

    # Execute the model
    result = compiled.execute(
        Tensor.from_dlpack(pixel_values).to(device),
        Tensor.from_dlpack(window_index).to(device),
    )
    max_tensor = result[0]
    return from_dlpack(max_tensor)


@pytest.mark.parametrize(
    "config_name",
    [
        pytest.param(ConfigNames.QWEN2_5VL_3B),
    ],
)
@pytest.mark.parametrize(
    "target_size",
    [
        224,  # Small size (16x16 patches)
        448,  # Medium size (32x32 patches)
        672,  # Large size (48x48 patches)
    ],
)
def test_vision_patch_embed(config_name: ConfigNames, target_size: int) -> None:
    """Test patch embedding for different image resolutions."""
    torch.manual_seed(42)

    # Load config and generate weights
    loader = get_config_loader()
    hf_vision_config = loader.load_hf_vision_config(config_name)
    qwen2_5vl_config = loader.create_qwen2_5vl_config(config_name)
    embeddings_weights = get_weight_generator(
        config_name
    ).generate_vision_patch_embed_weights()

    # Create test inputs
    in_channels = hf_vision_config.get("in_channels", 3)
    temporal_patch_size = hf_vision_config["temporal_patch_size"]
    patch_size = hf_vision_config["patch_size"]

    # For Qwen2.5VL, input is already in patch format
    # Shape: (seq_len, in_channels * temporal_patch_size * patch_size * patch_size)
    num_patches_h = target_size // patch_size
    num_patches_w = target_size // patch_size
    seq_len = num_patches_h * num_patches_w
    input_dim = in_channels * temporal_patch_size * patch_size * patch_size

    pixel_values = torch.randn(seq_len, input_dim, dtype=torch.bfloat16).to(
        "cuda"
    )

    # Generate reference output
    torch_output = generate_torch_outputs(
        pixel_values, hf_vision_config, embeddings_weights
    )

    # Generate MAX output
    max_output = generate_max_outputs(
        pixel_values,
        qwen2_5vl_config,
        embeddings_weights,
        DType.bfloat16,
        Accelerator(),
    )

    # Verify output shape
    expected_shape = (seq_len, hf_vision_config["hidden_size"])
    assert max_output.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {max_output.shape}"
    )

    # Compare outputs
    assert_tensors_close(
        torch_output,
        max_output,
        rtol=1e-2,
        atol=5e-4,
        message="Vision patch embedding outputs do not match",
    )


@pytest.mark.parametrize(
    "config_name",
    [
        pytest.param(ConfigNames.QWEN2_5VL_3B),
    ],
)
@pytest.mark.parametrize(
    "height,width",
    [
        (224, 336),  # Non-square, 16x24 patches
        (448, 224),  # Non-square, 32x16 patches
    ],
)
def test_vision_patch_embed_non_square(
    config_name: ConfigNames, height: int, width: int
) -> None:
    """Test patch embedding for non-square images."""
    torch.manual_seed(42)

    # Load config and generate weights
    loader = get_config_loader()
    hf_vision_config = loader.load_hf_vision_config(config_name)
    qwen2_5vl_config = loader.create_qwen2_5vl_config(config_name)
    embeddings_weights = get_weight_generator(
        config_name
    ).generate_vision_patch_embed_weights()

    # Create test inputs
    in_channels = hf_vision_config.get("in_channels", 3)
    temporal_patch_size = hf_vision_config["temporal_patch_size"]
    patch_size = hf_vision_config["patch_size"]

    # Calculate number of patches
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    seq_len = num_patches_h * num_patches_w
    input_dim = in_channels * temporal_patch_size * patch_size * patch_size

    pixel_values = torch.randn(seq_len, input_dim, dtype=torch.bfloat16).to(
        "cuda"
    )

    # Generate reference output
    torch_output = generate_torch_outputs(
        pixel_values, hf_vision_config, embeddings_weights
    )

    # Generate MAX output
    max_output = generate_max_outputs(
        pixel_values,
        qwen2_5vl_config,
        embeddings_weights,
        DType.bfloat16,
        Accelerator(),
    )

    # Verify output shape
    expected_shape = (seq_len, hf_vision_config["hidden_size"])
    assert max_output.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {max_output.shape}"
    )

    # Compare outputs
    assert_tensors_close(
        torch_output,
        max_output,
        rtol=1e-2,
        atol=5e-4,
        message="Vision patch embedding non-square outputs do not match",
    )


@pytest.mark.parametrize(
    "config_name",
    [
        pytest.param(ConfigNames.QWEN2_5VL_3B),
    ],
)
def test_vision_patch_embed_video(config_name: ConfigNames) -> None:
    """Test patch embedding for video inputs with temporal dimension."""
    torch.manual_seed(42)

    # Load config and generate weights
    loader = get_config_loader()
    hf_vision_config = loader.load_hf_vision_config(config_name)
    qwen2_5vl_config = loader.create_qwen2_5vl_config(config_name)
    embeddings_weights = get_weight_generator(
        config_name
    ).generate_vision_patch_embed_weights()

    # Create test inputs for video
    in_channels = hf_vision_config.get("in_channels", 3)
    temporal_patch_size = hf_vision_config["temporal_patch_size"]
    patch_size = hf_vision_config["patch_size"]

    # Video dimensions
    num_frames = 4  # 4 frames in the video
    height = 224
    width = 224

    # Calculate patches
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    num_temporal_patches = num_frames // temporal_patch_size
    seq_len = num_temporal_patches * num_patches_h * num_patches_w
    input_dim = in_channels * temporal_patch_size * patch_size * patch_size

    pixel_values = torch.randn(seq_len, input_dim, dtype=torch.bfloat16).to(
        "cuda"
    )

    # Generate reference output
    torch_output = generate_torch_outputs(
        pixel_values, hf_vision_config, embeddings_weights
    )

    # Generate MAX output
    max_output = generate_max_outputs(
        pixel_values,
        qwen2_5vl_config,
        embeddings_weights,
        DType.bfloat16,
        Accelerator(),
    )

    # Verify output shape
    expected_shape = (seq_len, hf_vision_config["hidden_size"])
    assert max_output.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {max_output.shape}"
    )

    # Compare outputs
    assert_tensors_close(
        torch_output,
        max_output,
        rtol=1e-2,
        atol=5e-4,
        message="Vision patch embedding video outputs do not match",
    )
