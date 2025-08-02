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
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLVisionConfig,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionPatchEmbed as HFQwen2_5VisionPatchEmbed,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel as HFQwen2_5VisionTransformer,
)
from utils.assert_tensors import assert_tensors_close
from utils.config_loader import ConfigNames, get_config_loader
from utils.weight_converter import convert_hf_to_max_weights
from utils.weight_generator import get_weight_generator

RTOL = 1e-2
ATOL = 1e-2


@torch.no_grad()
def generate_torch_outputs(
    pixel_values: torch.Tensor,
    window_index: torch.Tensor,
    hf_vision_config: dict,
    embeddings_weights: dict[str, torch.Tensor],
    device: torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
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
        .to(dtype)
        .to(device)
    )

    # Load weights
    ref_model.load_state_dict(embeddings_weights, strict=True)
    ref_model.eval()

    # Forward pass
    with torch.no_grad():
        hidden_states = ref_model(pixel_values)
        seq_len, _ = hidden_states.size()
        spatial_merge_size = hf_vision_config["spatial_merge_size"]
        spatial_merge_unit = spatial_merge_size * spatial_merge_size
        hidden_states = hidden_states.reshape(
            seq_len // spatial_merge_unit,
            spatial_merge_unit,
            -1,
        )
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)

    return hidden_states


def generate_max_outputs(
    pixel_values: torch.Tensor,
    window_index: torch.Tensor,
    qwen2_5vl_config: dict,
    embeddings_weights: dict[str, torch.Tensor],
    dtype: DType,
    device: Device,
) -> torch.Tensor:
    """Generate outputs using MAX VisionPatchEmbed implementation."""
    is_gpu = isinstance(device, Accelerator)
    device_ref = DeviceRef.GPU() if is_gpu else DeviceRef.CPU()
    pixel_values = pixel_values.cuda() if is_gpu else pixel_values.cpu()
    window_index = window_index.cuda() if is_gpu else window_index.cpu()

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
        spatial_merge_unit=vision_config["spatial_merge_size"]
        * vision_config["spatial_merge_size"],
    )

    # Load weights using state_dict
    patch_embed_module.load_state_dict(max_weights, strict=True)

    session = InferenceSession(devices=[device])

    # Define input types
    pixel_values_type = TensorType(
        dtype, shape=pixel_values.shape, device=device_ref
    )

    seq_len = pixel_values.shape[0]
    spatial_merge_size = vision_config["spatial_merge_size"]
    spatial_merge_unit = spatial_merge_size * spatial_merge_size
    # window_index should have length seq_len // spatial_merge_unit
    window_index_len = seq_len // spatial_merge_unit
    window_index_type = TensorType(
        DType.int64, shape=(window_index_len,), device=device_ref
    )

    with Graph(
        "VisionPatchEmbed",
        input_types=(pixel_values_type, window_index_type),
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
    "target_size",
    # Small (16x16 patches), Medium (32x32 patches), Large (48x48 patches)
    [224, 448, 672],
)
def test_vision_patch_embed(target_size: int) -> None:
    """Test patch embedding for different image resolutions."""
    torch.manual_seed(42)

    # Load config and generate weights
    loader = get_config_loader()
    hf_vision_config = loader.load_hf_vision_config(ConfigNames.QWEN2_5VL_3B)
    qwen2_5vl_config = loader.create_qwen2_5vl_config(ConfigNames.QWEN2_5VL_3B)
    embeddings_weights = get_weight_generator(
        ConfigNames.QWEN2_5VL_3B
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

    pixel_values = torch.normal(
        mean=1.3,
        std=0.832,
        size=(seq_len, input_dim),
        dtype=torch.bfloat16,
    ).to("cuda")

    vision_transformer = HFQwen2_5VisionTransformer(
        config=Qwen2_5_VLVisionConfig()
    )

    grid_thw = torch.tensor(
        [[1, num_patches_h, num_patches_w]], dtype=torch.int64
    )
    window_index, _ = vision_transformer.get_window_index(grid_thw)

    # Generate reference output
    torch_output = generate_torch_outputs(
        pixel_values=pixel_values,
        window_index=window_index,
        hf_vision_config=hf_vision_config,
        embeddings_weights=embeddings_weights,
        device="cuda",
        dtype=torch.bfloat16,
    )

    # Generate MAX output
    max_output = generate_max_outputs(
        pixel_values=pixel_values,
        window_index=window_index,
        qwen2_5vl_config=qwen2_5vl_config,
        embeddings_weights=embeddings_weights,
        dtype=DType.bfloat16,
        device=Accelerator(),
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
        rtol=RTOL,
        atol=ATOL,
        message="Vision patch embedding outputs do not match",
    )


@pytest.mark.parametrize(
    "height,width",
    # 16x24 patches, 32x16 patches, 146x98 patches
    [(224, 336), (448, 224), (2044, 1372)],
)
def test_vision_patch_embed_non_square(height: int, width: int) -> None:
    """Test patch embedding for non-square images."""
    torch.manual_seed(42)

    # Load config and generate weights
    loader = get_config_loader()
    hf_vision_config = loader.load_hf_vision_config(ConfigNames.QWEN2_5VL_3B)
    qwen2_5vl_config = loader.create_qwen2_5vl_config(ConfigNames.QWEN2_5VL_3B)
    embeddings_weights = get_weight_generator(
        ConfigNames.QWEN2_5VL_3B
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

    pixel_values = torch.normal(
        mean=1.3,
        std=0.832,
        size=(seq_len, input_dim),
        dtype=torch.bfloat16,
    ).to("cuda")

    vision_transformer = HFQwen2_5VisionTransformer(
        config=Qwen2_5_VLVisionConfig()
    )

    grid_thw = torch.tensor(
        [[1, num_patches_h, num_patches_w]], dtype=torch.int64
    )
    window_index, _ = vision_transformer.get_window_index(grid_thw)

    # Generate reference output
    torch_output = generate_torch_outputs(
        pixel_values=pixel_values,
        window_index=window_index,
        hf_vision_config=hf_vision_config,
        embeddings_weights=embeddings_weights,
        device="cuda",
        dtype=torch.bfloat16,
    )

    # Generate MAX output
    max_output = generate_max_outputs(
        pixel_values=pixel_values,
        window_index=window_index,
        qwen2_5vl_config=qwen2_5vl_config,
        embeddings_weights=embeddings_weights,
        dtype=DType.bfloat16,
        device=Accelerator(),
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
        rtol=RTOL,
        atol=ATOL,
        message="Vision patch embedding non-square outputs do not match",
    )


def test_vision_patch_embed_video() -> None:
    """Test patch embedding for video inputs with temporal dimension."""
    torch.manual_seed(42)

    # Load config and generate weights
    loader = get_config_loader()
    hf_vision_config = loader.load_hf_vision_config(ConfigNames.QWEN2_5VL_3B)
    qwen2_5vl_config = loader.create_qwen2_5vl_config(ConfigNames.QWEN2_5VL_3B)
    embeddings_weights = get_weight_generator(
        ConfigNames.QWEN2_5VL_3B
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

    pixel_values = torch.normal(
        mean=1.3,
        std=0.832,
        size=(seq_len, input_dim),
        dtype=torch.bfloat16,
    ).to("cuda")

    vision_transformer = HFQwen2_5VisionTransformer(
        config=Qwen2_5_VLVisionConfig()
    )
    grid_thw = torch.tensor(
        [[num_temporal_patches, num_patches_h, num_patches_w]],
        dtype=torch.int64,
    )
    window_index, _ = vision_transformer.get_window_index(grid_thw)

    # Generate reference output
    torch_output = generate_torch_outputs(
        pixel_values=pixel_values,
        window_index=window_index,
        hf_vision_config=hf_vision_config,
        embeddings_weights=embeddings_weights,
        device="cuda",
        dtype=torch.bfloat16,
    )

    # Generate MAX output
    max_output = generate_max_outputs(
        pixel_values=pixel_values,
        window_index=window_index,
        qwen2_5vl_config=qwen2_5vl_config,
        embeddings_weights=embeddings_weights,
        dtype=DType.bfloat16,
        device=Accelerator(),
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
        rtol=RTOL,
        atol=ATOL,
        message="Vision patch embedding video outputs do not match",
    )


@pytest.mark.parametrize(
    "image_sizes",
    [  # Two square images of different sizes
        [(224, 224), (448, 448)],
        # Three images: non-square, non-square, square
        [(224, 336), (448, 224), (672, 672)],
        # Four images of various sizes
        [(224, 224), (448, 672), (336, 224), (672, 448)],
    ],
)
def test_vision_patch_embed_multiple_images(
    image_sizes: list[tuple[int, int]],
) -> None:
    """Test patch embedding for multiple images with different sizes."""
    torch.manual_seed(42)

    # Load config and generate weights
    loader = get_config_loader()
    hf_vision_config = loader.load_hf_vision_config(ConfigNames.QWEN2_5VL_3B)
    qwen2_5vl_config = loader.create_qwen2_5vl_config(ConfigNames.QWEN2_5VL_3B)
    embeddings_weights = get_weight_generator(
        ConfigNames.QWEN2_5VL_3B
    ).generate_vision_patch_embed_weights()

    # Create test inputs for multiple images
    in_channels = hf_vision_config.get("in_channels", 3)
    temporal_patch_size = hf_vision_config["temporal_patch_size"]
    patch_size = hf_vision_config["patch_size"]

    # Calculate patches for each image and create grid_thw
    grid_thw_list = []
    total_seq_len = 0
    pixel_values_list = []

    for height, width in image_sizes:
        # Calculate number of patches for this image
        num_patches_h = height // patch_size
        num_patches_w = width // patch_size
        seq_len = num_patches_h * num_patches_w
        input_dim = in_channels * temporal_patch_size * patch_size * patch_size

        # Add to grid (temporal_patches=1 for images)
        grid_thw_list.append([1, num_patches_h, num_patches_w])
        total_seq_len += seq_len

        # Generate pixel values for this image. mean and std simulate a real image.
        image_pixel_values = torch.normal(
            mean=1.3,
            std=0.832,
            size=(seq_len, input_dim),
            dtype=torch.bfloat16,
        )
        pixel_values_list.append(image_pixel_values)

    # Concatenate all pixel values
    pixel_values = torch.cat(pixel_values_list, dim=0).to("cuda")

    # Create grid_thw tensor [n_images, 3]
    grid_thw = torch.tensor(grid_thw_list, dtype=torch.int64)

    vision_transformer = HFQwen2_5VisionTransformer(
        config=Qwen2_5_VLVisionConfig()
    )
    window_index, _ = vision_transformer.get_window_index(grid_thw)

    # Generate reference output
    torch_output = generate_torch_outputs(
        pixel_values=pixel_values,
        window_index=window_index,
        hf_vision_config=hf_vision_config,
        embeddings_weights=embeddings_weights,
        device="cuda",
        dtype=torch.bfloat16,
    )

    # Generate MAX output
    max_output = generate_max_outputs(
        pixel_values=pixel_values,
        window_index=window_index,
        qwen2_5vl_config=qwen2_5vl_config,
        embeddings_weights=embeddings_weights,
        dtype=DType.bfloat16,
        device=Accelerator(),
    )

    # Verify output shape
    expected_shape = (total_seq_len, hf_vision_config["hidden_size"])
    assert max_output.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {max_output.shape}"
    )

    # Compare outputs
    assert_tensors_close(
        torch_output,
        max_output,
        rtol=RTOL,
        atol=ATOL,
        message=f"Vision patch embedding multiple images of size {image_sizes} outputs do not match",
    )
