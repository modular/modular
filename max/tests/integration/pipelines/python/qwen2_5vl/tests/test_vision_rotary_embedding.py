# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Tests for Qwen2.5VL vision rotary embedding."""

import pytest
import torch
from max.driver import Accelerator, Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.pipelines.architectures.qwen2_5vl.nn.visual_transformer import (
    VisionRotaryEmbedding,
)
from torch.utils.dlpack import from_dlpack
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLVisionConfig,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionRotaryEmbedding as HFQwen2_5VisionRotaryEmbedding,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel as HFQwen2_5VisionTransformer,
)
from utils.assert_tensors import assert_tensors_close
from utils.config_loader import ConfigNames, get_config_loader

RTOL = 1e-4
ATOL = 1e-4


def max_rot_pos_emb(
    grid_thw: torch.Tensor, spatial_merge_size: int
) -> tuple[torch.Tensor, int]:
    """Generate position IDs for MAX VisionRotaryEmbedding.

    Args:
        grid_thw: Tensor of shape (num_grids, 3) containing (temporal, height, width) dimensions
        spatial_merge_size: Size of spatial merging (e.g., 2 means 2x2 patches are merged)

    Returns:
        tuple of (pos_ids, max_grid_size) where pos_ids has shape (seq_len, 2)
    """
    pos_ids = []
    for t, h, w in grid_thw:
        hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
        hpos_ids = hpos_ids.reshape(
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        hpos_ids = hpos_ids.permute(0, 2, 1, 3)
        hpos_ids = hpos_ids.flatten()

        wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
        wpos_ids = wpos_ids.reshape(
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        wpos_ids = wpos_ids.permute(0, 2, 1, 3)
        wpos_ids = wpos_ids.flatten()
        pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
    pos_ids = torch.cat(pos_ids, dim=0)
    max_grid_size = int(grid_thw[:, 1:].max().item())
    return pos_ids, max_grid_size


def generate_torch_rotary_embeddings(
    hidden_size: int,
    n_heads: int,
    grid_thw: torch.Tensor,
    window_index: torch.Tensor,
    spatial_merge_unit: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate rotary position embeddings using HuggingFace implementation."""
    # Create HF Vision Config
    config = Qwen2_5_VLVisionConfig()
    config.hidden_size = hidden_size
    config.num_heads = n_heads

    # Initialize HF vision transformer and rotary embedding
    vision_transformer = HFQwen2_5VisionTransformer(config=config)
    head_dim = config.hidden_size // config.num_heads
    rotary_pos_emb_layer = HFQwen2_5VisionRotaryEmbedding(head_dim // 2).to(
        "cuda"
    )
    vision_transformer.rotary_pos_emb = rotary_pos_emb_layer

    # Generate position embeddings using HF implementation
    rotary_pos_emb = vision_transformer.rot_pos_emb(grid_thw)

    # Reshape and reorder as per the user's specification
    rotary_pos_emb = rotary_pos_emb.reshape(
        seq_len // spatial_merge_unit, spatial_merge_unit, -1
    )
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    return position_embeddings


def generate_torch_outputs(
    dim: int,
    n_heads: int,
    grid_thw: torch.Tensor,
    window_index: torch.Tensor,
    spatial_merge_unit: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate reference outputs using HuggingFace implementation."""
    with torch.no_grad():
        # Generate rotary position embeddings using HF implementation
        cos_emb, sin_emb = generate_torch_rotary_embeddings(
            hidden_size=dim,
            n_heads=n_heads,
            grid_thw=grid_thw,
            window_index=window_index,
            spatial_merge_unit=spatial_merge_unit,
            seq_len=seq_len,
        )

        return cos_emb, sin_emb


def generate_max_outputs(
    dim: int,
    n_heads: int,
    theta: float,
    rot_pos_ids: torch.Tensor,
    window_index: torch.Tensor,
    spatial_merge_unit: int,
    max_grid_size: int,
    seq_len: int,
    dtype: DType,
    device: Device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate outputs using MAX VisionRotaryEmbedding."""
    is_gpu = isinstance(device, Accelerator)
    device_ref = DeviceRef.GPU() if is_gpu else DeviceRef.CPU()

    session = InferenceSession(devices=[device])

    rot_pos_ids_type = TensorType(
        DType.int64, shape=(seq_len, 2), device=device_ref
    )
    window_index_type = TensorType(
        DType.int64, shape=(seq_len // spatial_merge_unit,), device=device_ref
    )

    # Create VisionRotaryEmbedding
    rotary_emb = VisionRotaryEmbedding(
        dim=dim,
        n_heads=n_heads,
        theta=theta,
    )

    # Create MAX graph
    with Graph(
        "VisionRotaryEmbedding",
        input_types=(rot_pos_ids_type, window_index_type),
    ) as graph:
        rot_pos_ids_tensor, window_index_tensor = graph.inputs

        # Generate rotary position embeddings
        cos_emb, sin_emb = rotary_emb.generate_rot_pos_embeddings(
            rot_pos_ids=rot_pos_ids_tensor.tensor,
            window_index=window_index_tensor.tensor,
            spatial_merge_unit=spatial_merge_unit,
            max_grid_size=ops.constant(
                max_grid_size, DType.int32, device=DeviceRef.CPU()
            ),
            seq_len=rot_pos_ids_tensor.tensor.shape[0],
        )

        graph.output(cos_emb, sin_emb)

    # Compile and run
    compiled = session.load(graph, weights_registry={})
    results = compiled.execute(
        Tensor.from_dlpack(rot_pos_ids).to(device),
        Tensor.from_dlpack(window_index).to(device),
    )

    # Convert back to torch
    cos_emb_torch = from_dlpack(results[0])
    sin_emb_torch = from_dlpack(results[1])

    return cos_emb_torch, sin_emb_torch


@pytest.mark.parametrize(
    "image_size",
    [
        224,  # Small image
        448,  # Medium image
        672,  # Large image
    ],
)
def test_vision_rotary_embedding(image_size: int) -> None:
    """Test VisionRotaryEmbedding against PyTorch reference implementation."""
    torch.manual_seed(42)

    # Load config
    config_loader = get_config_loader()
    qwen2_5vl_config = config_loader.create_qwen2_5vl_config(
        ConfigNames.QWEN2_5VL_3B
    )
    vision_config = qwen2_5vl_config["vision_config"]

    # Extract parameters
    dim = vision_config["hidden_size"]
    n_heads = vision_config["num_heads"]
    patch_size = vision_config["patch_size"]
    spatial_merge_size = vision_config["spatial_merge_size"]  # Get from config
    theta = 10000.0

    # Calculate dimensions
    patches_per_side = image_size // patch_size
    spatial_merge_unit = spatial_merge_size * spatial_merge_size
    seq_len = patches_per_side * patches_per_side

    # Create grid_thw - represents (temporal, height, width) dimensions
    # For single image: temporal=1, height and width in patches
    grid_thw = torch.tensor(
        [[1, patches_per_side, patches_per_side]],
        dtype=torch.int64,
    ).to("cuda")

    # Create window_index - simulate window ordering
    # For simplicity, create sequential indices
    vision_transformer = HFQwen2_5VisionTransformer(
        config=Qwen2_5_VLVisionConfig()
    )
    window_index_len = seq_len // spatial_merge_unit
    window_index, _ = vision_transformer.get_window_index(grid_thw)

    window_index = window_index.to("cuda")
    grid_thw = grid_thw.to("cuda")

    # Generate reference outputs
    torch_cos_emb, torch_sin_emb = generate_torch_outputs(
        dim=dim,
        n_heads=n_heads,
        grid_thw=grid_thw,
        window_index=window_index,
        spatial_merge_unit=spatial_merge_unit,
        seq_len=seq_len,
    )

    # Generate MAX outputs
    # Generate proper 2D position ids using the helper function
    pos_coords_2d, max_grid_size = max_rot_pos_emb(grid_thw, spatial_merge_size)
    pos_ids = pos_coords_2d.to("cuda")
    max_cos_emb, max_sin_emb = generate_max_outputs(
        dim=dim,
        n_heads=n_heads,
        theta=theta,
        rot_pos_ids=pos_ids,
        window_index=window_index,
        spatial_merge_unit=spatial_merge_unit,
        max_grid_size=max_grid_size,
        seq_len=seq_len,
        dtype=DType.float32,
        device=Accelerator(),
    )

    # Compare results

    assert_tensors_close(
        torch_cos_emb,
        max_cos_emb,
        rtol=RTOL,
        atol=ATOL,
        message="Vision rotary embedding (cos) non-square outputs do not match",
    )

    assert_tensors_close(
        torch_sin_emb,
        max_sin_emb,
        rtol=RTOL,
        atol=ATOL,
        message="Vision rotary embedding (sin) non-square outputs do not match",
    )


@pytest.mark.parametrize(
    "image_sizes",
    [
        # Two square images of different sizes
        [(224, 224), (448, 448)],
        # Three images: non-square, non-square, square
        [(224, 336), (448, 224), (672, 672)],
        # Four images of various sizes
        [(224, 224), (448, 672), (336, 224), (672, 448)],
    ],
)
def test_vision_rotary_embedding_multiple_images(
    image_sizes: list[tuple[int, int]],
) -> None:
    """Test VisionRotaryEmbedding with multiple images of different sizes."""
    torch.manual_seed(42)

    # Load config
    config_loader = get_config_loader()
    qwen2_5vl_config = config_loader.create_qwen2_5vl_config(
        ConfigNames.QWEN2_5VL_3B
    )
    vision_config = qwen2_5vl_config["vision_config"]

    # Extract parameters
    dim = vision_config["hidden_size"]
    n_heads = vision_config["num_heads"]
    patch_size = vision_config["patch_size"]
    spatial_merge_size = vision_config["spatial_merge_size"]
    theta = 10000.0

    # Calculate dimensions for multiple images
    spatial_merge_unit = spatial_merge_size * spatial_merge_size
    grid_thw_list = []
    total_seq_len = 0

    for height, width in image_sizes:
        # Calculate number of patches for this image
        patches_h = height // patch_size
        patches_w = width // patch_size
        seq_len = patches_h * patches_w

        # Add to grid (temporal_patches=1 for images)
        grid_thw_list.append([1, patches_h, patches_w])
        total_seq_len += seq_len

    # Create grid_thw tensor [n_images, 3]
    grid_thw = torch.tensor(grid_thw_list, dtype=torch.int64).to("cuda")

    # Generate window index for multiple images
    vision_transformer = HFQwen2_5VisionTransformer(
        config=Qwen2_5_VLVisionConfig()
    )
    window_index, _ = vision_transformer.get_window_index(grid_thw)
    window_index = window_index.to("cuda")

    # Generate reference outputs using HuggingFace implementation
    torch_cos_emb, torch_sin_emb = generate_torch_outputs(
        dim=dim,
        n_heads=n_heads,
        grid_thw=grid_thw,
        window_index=window_index,
        spatial_merge_unit=spatial_merge_unit,
        seq_len=total_seq_len,
    )

    # Generate MAX outputs
    # Generate 2D position IDs for all images combined
    pos_coords_2d, max_grid_size = max_rot_pos_emb(grid_thw, spatial_merge_size)
    pos_ids = pos_coords_2d.to("cuda")

    max_cos_emb, max_sin_emb = generate_max_outputs(
        dim=dim,
        n_heads=n_heads,
        theta=theta,
        rot_pos_ids=pos_ids,
        window_index=window_index,
        spatial_merge_unit=spatial_merge_unit,
        max_grid_size=max_grid_size,
        seq_len=total_seq_len,
        dtype=DType.float32,
        device=Accelerator(),
    )

    # Verify output shapes
    expected_shape = (total_seq_len, dim // n_heads)
    assert torch_cos_emb.shape == expected_shape, (
        f"Expected torch cos shape {expected_shape}, got {torch_cos_emb.shape}"
    )
    assert torch_sin_emb.shape == expected_shape, (
        f"Expected torch sin shape {expected_shape}, got {torch_sin_emb.shape}"
    )
    assert max_cos_emb.shape == expected_shape, (
        f"Expected MAX cos shape {expected_shape}, got {max_cos_emb.shape}"
    )
    assert max_sin_emb.shape == expected_shape, (
        f"Expected MAX sin shape {expected_shape}, got {max_sin_emb.shape}"
    )

    # Compare results
    assert_tensors_close(
        torch_cos_emb,
        max_cos_emb,
        rtol=RTOL,
        atol=ATOL,
        message="Vision rotary embedding (cos) multiple images outputs do not match",
    )

    assert_tensors_close(
        torch_sin_emb,
        max_sin_emb,
        rtol=RTOL,
        atol=ATOL,
        message="Vision rotary embedding (sin) multiple images outputs do not match",
    )
