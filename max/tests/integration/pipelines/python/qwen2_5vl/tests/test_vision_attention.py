# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Tests for Qwen2.5VL vision attention layer."""

import pytest
import torch
from max.driver import Accelerator, Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.qwen2_5vl.nn.visual_transformer import (
    VisionWindowSdpaAttention,
)
from torch.utils.dlpack import from_dlpack
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLVisionAttention as HFQwen2_5VLVisionAttention,
)
from utils.assert_tensors import assert_tensors_close
from utils.config_loader import ConfigNames, get_config_loader
from utils.weight_converter import convert_hf_to_max_weights
from utils.weight_generator import get_weight_generator


def generate_rot_pos_emb(
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    head_dim: int,
    theta: float = 10000.0,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """Generate rotary position embeddings based on grid dimensions.

    This function implements M-RoPE (Multimodal Rotary Position Embedding) for Qwen2.5VL,
    handling spatial structure with temporal, height, and width dimensions.

    Args:
        grid_thw: Tensor of shape (num_grids, 3) containing (temporal, height, width) dimensions
        spatial_merge_size: Size of spatial merging (e.g., 2 means 2x2 patches are merged)
        head_dim: Dimension of attention heads
        theta: Base for frequency computation
        device: Device to create tensors on

    Returns:
        Rotary position embeddings of shape (total_patches, head_dim // 2)
    """
    pos_ids = []

    for t, h, w in grid_thw:
        t, h, w = int(t), int(h), int(w)

        # Generate height position IDs
        hpos_ids = torch.arange(h, device=device).unsqueeze(1).expand(-1, w)
        hpos_ids = hpos_ids.reshape(
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        hpos_ids = hpos_ids.permute(0, 2, 1, 3)
        hpos_ids = hpos_ids.flatten()

        # Generate width position IDs
        wpos_ids = torch.arange(w, device=device).unsqueeze(0).expand(h, -1)
        wpos_ids = wpos_ids.reshape(
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        wpos_ids = wpos_ids.permute(0, 2, 1, 3)
        wpos_ids = wpos_ids.flatten()

        # Stack and repeat for temporal dimension
        pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))

    pos_ids = torch.cat(pos_ids, dim=0)

    # Get maximum grid size
    max_grid_size = int(grid_thw[:, 1:].max().item())

    # Generate inverse frequencies
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(
                0,
                head_dim // 2,
                dtype=torch.float32,
                device=device,
            )
            / (head_dim // 2)
        )
    )

    # Generate full rotary embeddings
    t = torch.arange(max_grid_size, dtype=torch.float32, device=device)
    rotary_pos_emb_full = torch.outer(t, inv_freq)

    # The HuggingFace model expects embeddings after spatial merging
    # Calculate the actual sequence length after merging
    total_patches = pos_ids.shape[0]
    seq_len_after_merge = total_patches // (spatial_merge_size**2)

    # For now, generate simple sequential embeddings for the merged sequence
    # This matches what the HuggingFace model expects
    pos_range = torch.arange(
        seq_len_after_merge, dtype=torch.float32, device=device
    )
    rotary_pos_emb = torch.outer(pos_range, inv_freq)

    return rotary_pos_emb


@torch.no_grad()
def generate_torch_outputs(
    hf_config: dict,
    input_tensor: torch.Tensor,
    vision_attention_weights: dict[str, torch.Tensor],
    grid_thw: torch.Tensor,
    spatial_merge_size: int = 2,
) -> torch.Tensor:
    """Generate reference outputs using HuggingFace Qwen2.5VL implementation."""
    # Create the HuggingFace attention layer - it expects dim and num_heads as direct arguments
    hidden_size = hf_config["hidden_size"]
    num_heads = hf_config["num_heads"]
    layer = (
        HFQwen2_5VLVisionAttention(dim=hidden_size, num_heads=num_heads)
        .to(torch.bfloat16)
        .to("cuda")
    )

    # Load weights using state_dict
    layer.load_state_dict(vision_attention_weights, strict=True)
    layer.eval()

    # Create dummy cu_seqlens for attention
    batch_size, seq_len, hidden_size = input_tensor.shape
    cu_seqlens = torch.tensor(
        [0, seq_len], dtype=torch.int32, device=input_tensor.device
    )

    # Generate rotary embeddings using the helper function
    head_dim = hidden_size // hf_config["num_heads"]
    rotary_pos_emb = generate_rot_pos_emb(
        grid_thw=grid_thw,
        spatial_merge_size=spatial_merge_size,
        head_dim=head_dim,
        theta=10000.0,
        device=input_tensor.device,
    ).to(torch.bfloat16)

    # Flatten input to (seq_len, hidden_size) as HF expects
    input_flattened = input_tensor.reshape(-1, hidden_size)

    with torch.no_grad():
        output = layer(
            input_flattened,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
        )

    return output


def generate_max_outputs(
    max_config: dict,
    input_tensor: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
    dtype: DType,
    device: Device,
    grid_thw: torch.Tensor,
    spatial_merge_size: int = 2,
) -> torch.Tensor:
    """Generate outputs using MAX Qwen2.5VL vision attention implementation."""
    is_gpu = isinstance(device, Accelerator)
    input_tensor = input_tensor.cuda() if is_gpu else input_tensor.cpu()
    device_ref = DeviceRef.GPU() if is_gpu else DeviceRef.CPU()

    vision_config = max_config["vision_config"]

    # Convert HuggingFace weights to MAX format using weight converter
    max_weights = convert_hf_to_max_weights(attention_weights)

    # Create the attention layer
    attention = VisionWindowSdpaAttention(
        dtype=dtype,
        device=device_ref,
        dim=vision_config["hidden_size"],
        n_heads=vision_config["num_heads"],
    )

    # Load weights using state_dict
    attention.load_state_dict(max_weights, strict=True)

    session = InferenceSession(devices=[device])

    # Build the graph
    batch_size, seq_len, hidden_size = input_tensor.shape
    input_type = TensorType(
        dtype, [batch_size, seq_len, hidden_size], device=device_ref
    )

    # Create consistent position embeddings matching HuggingFace implementation
    total_seq_len = batch_size * seq_len
    num_heads = vision_config["num_heads"]
    head_dim = hidden_size // num_heads

    # Generate rotary embeddings using the helper function
    rotary_pos_emb = generate_rot_pos_emb(
        grid_thw=grid_thw,
        spatial_merge_size=spatial_merge_size,
        head_dim=head_dim,
        theta=10000.0,
        device=input_tensor.device,
    )

    # Create cos/sin embeddings by concatenating (same as HF's emb = cat(rotary_pos_emb, rotary_pos_emb))
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    dummy_cos = emb.cos().to(torch.bfloat16)
    dummy_sin = emb.sin().to(torch.bfloat16)
    dummy_attention_mask = torch.zeros(
        1,
        total_seq_len,
        total_seq_len,
        dtype=torch.bfloat16,
        device=input_tensor.device,
    )

    cos_type = TensorType(dtype, shape=dummy_cos.shape, device=device_ref)
    sin_type = TensorType(dtype, shape=dummy_sin.shape, device=device_ref)
    mask_type = TensorType(
        dtype, shape=dummy_attention_mask.shape, device=device_ref
    )

    with Graph(
        "Qwen2_5VLVisionAttention",
        input_types=(input_type, cos_type, sin_type, mask_type),
    ) as graph:
        x, cos, sin, mask = graph.inputs
        # Flatten input to (seq_len, hidden_size) for MAX
        x_flattened = x.tensor.reshape((-1, hidden_size))
        position_embeddings = (cos.tensor, sin.tensor)
        output = attention(x_flattened, position_embeddings, mask.tensor)
        graph.output(output)

    compiled = session.load(graph, weights_registry=attention.state_dict())

    # Execute the model and get the first result
    result = compiled.execute(
        Tensor.from_dlpack(input_tensor).to(device),
        Tensor.from_dlpack(dummy_cos).to(device),
        Tensor.from_dlpack(dummy_sin).to(device),
        Tensor.from_dlpack(dummy_attention_mask).to(device),
    )
    # Convert result back to torch tensor
    max_tensor = result[0]
    return from_dlpack(max_tensor)


@pytest.mark.parametrize(
    "config_name",
    [
        pytest.param(ConfigNames.QWEN2_5VL_3B),
    ],
)
def test_vision_attention(config_name: ConfigNames) -> None:
    """Test Qwen2.5VL vision attention against PyTorch reference."""
    # Create test instance and load config
    config_loader = get_config_loader()
    weight_generator = get_weight_generator(config_name)

    # Create config-specific fixtures
    qwen2_5vl_config = config_loader.create_qwen2_5vl_config(config_name)
    vision_config = qwen2_5vl_config["vision_config"]

    hf_config = config_loader.load_hf_vision_config(config_name)

    # Create vision input tensor
    torch.manual_seed(42)
    # Use typical vision transformer dimensions
    patch_size = vision_config["patch_size"]
    image_size = vision_config["image_size"]
    spatial_merge_size = 2  # Common value for Qwen2.5VL

    # Calculate grid dimensions for a single image
    # For a single image: temporal=1, height and width in patches
    patches_per_side = image_size // patch_size
    # Ensure patches_per_side is divisible by spatial_merge_size
    patches_per_side = (
        patches_per_side // spatial_merge_size
    ) * spatial_merge_size
    grid_thw = torch.tensor(
        [[1, patches_per_side, patches_per_side]],  # [temporal, height, width]
        dtype=torch.long,
        device="cuda",
    )

    # Calculate sequence length after spatial merging
    # The formula should be: (patches_per_side // spatial_merge_size)^2
    seq_len = (patches_per_side // spatial_merge_size) ** 2

    batch_size = 1
    vision_input_tensor = torch.randn(
        batch_size,
        seq_len,  # sequence length after spatial merging
        vision_config["hidden_size"],
        dtype=torch.bfloat16,
    ).to("cuda")

    # Generate vision attention weights
    vision_attention_weights = (
        weight_generator.generate_vision_attention_weights()
    )

    # Generate reference output
    torch_output = generate_torch_outputs(
        hf_config,
        vision_input_tensor,
        vision_attention_weights,
        grid_thw=grid_thw,
        spatial_merge_size=spatial_merge_size,
    )

    # Generate MAX output
    max_output = generate_max_outputs(
        max_config=qwen2_5vl_config,
        input_tensor=vision_input_tensor,
        attention_weights=vision_attention_weights,
        dtype=DType.bfloat16,
        device=Accelerator(),
        grid_thw=grid_thw,
        spatial_merge_size=spatial_merge_size,
    )

    # Compare outputs using the base class method
    assert_tensors_close(
        torch_output,
        max_output,
        rtol=5e-2,
        atol=2e-2,
        message="Vision attention outputs do not match",
    )


@pytest.mark.parametrize(
    "config_name",
    [
        pytest.param(ConfigNames.QWEN2_5VL_3B),
    ],
)
@pytest.mark.parametrize(
    "seq_len",
    [
        16,  # Small sequence (4x4 after merging, from 8x8 patches)
        64,  # Medium sequence (8x8 after merging, from 16x16 patches)
    ],
)
def test_vision_attention_variable_seq_len(
    config_name: ConfigNames, seq_len: int
) -> None:
    """Test vision attention with different sequence lengths."""
    torch.manual_seed(42)

    # Load config and generate weights
    config_loader = get_config_loader()
    weight_generator = get_weight_generator(config_name)

    qwen2_5vl_config = config_loader.create_qwen2_5vl_config(config_name)
    vision_config = qwen2_5vl_config["vision_config"]
    hf_config = config_loader.load_hf_vision_config(config_name)

    # Create test input tensor with variable sequence length
    batch_size = 1
    spatial_merge_size = 2

    # For variable sequence length tests, create appropriate grid dimensions
    # Determine patch dimensions that result in the desired sequence length
    # seq_len should be divisible by spatial_merge_size^2
    patches_per_side = int((seq_len * spatial_merge_size**2) ** 0.5)
    # Ensure patches_per_side is divisible by spatial_merge_size
    patches_per_side = (
        patches_per_side // spatial_merge_size
    ) * spatial_merge_size
    grid_thw = torch.tensor(
        [[1, patches_per_side, patches_per_side]],  # [temporal, height, width]
        dtype=torch.long,
        device="cuda",
    )

    vision_input_tensor = torch.randn(
        batch_size,
        seq_len,
        vision_config["hidden_size"],
        dtype=torch.bfloat16,
    ).to("cuda")

    # Generate vision attention weights
    vision_attention_weights = (
        weight_generator.generate_vision_attention_weights()
    )

    # Generate reference output
    torch_output = generate_torch_outputs(
        hf_config,
        vision_input_tensor,
        vision_attention_weights,
        grid_thw=grid_thw,
        spatial_merge_size=spatial_merge_size,
    )

    # Generate MAX output
    max_output = generate_max_outputs(
        max_config=qwen2_5vl_config,
        input_tensor=vision_input_tensor,
        attention_weights=vision_attention_weights,
        dtype=DType.bfloat16,
        device=Accelerator(),
        grid_thw=grid_thw,
        spatial_merge_size=spatial_merge_size,
    )

    # Verify output shape
    expected_shape = (seq_len, vision_config["hidden_size"])
    assert max_output.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {max_output.shape}"
    )

    # Compare outputs
    assert_tensors_close(
        torch_output,
        max_output,
        rtol=2e-2,
        atol=5e-3,
        message=f"Vision attention outputs do not match for seq_len={seq_len}",
    )
