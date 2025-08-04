# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Tests for Qwen2.5VL vision attention layer."""

import pytest
import torch
import torch.nn.functional as F
from max.driver import Accelerator, Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.qwen2_5vl.nn.visual_transformer import (
    VisionWindowSdpaAttention,
)
from torch.utils.dlpack import from_dlpack
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLVisionConfig,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel as HFQwen2_5VisionTransformer,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLVisionAttention as HFQwen2_5VLVisionAttention,
)
from utils.assert_tensors import assert_tensors_close
from utils.config_loader import ConfigNames, get_config_loader
from utils.weight_converter import convert_hf_to_max_weights
from utils.weight_generator import get_weight_generator

RTOL = 2e-2
ATOL = 5e-3


def generate_cu_seqlens_full_attention(grid_thw: torch.Tensor) -> torch.Tensor:
    """Generate cu_seqlens for full attention layers following PyTorch implementation."""
    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
    return cu_seqlens


def generate_cu_seqlens_window_attention(
    cu_window_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Generate cu_seqlens for window attention layers following PyTorch implementation."""
    cu_window_seqlens = torch.tensor(cu_window_seqlens, dtype=torch.int32)
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
    return cu_window_seqlens


def generate_torch_rotary_embeddings(
    vision_transformer: HFQwen2_5VisionTransformer,
    grid_thw: torch.Tensor,
    window_index: torch.Tensor,
    spatial_merge_unit: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate rotary position embeddings using HuggingFace implementation."""
    # Generate position embeddings using HF implementation
    rotary_pos_emb = vision_transformer.rot_pos_emb(grid_thw)

    # Reshape and reorder as per the specification
    rotary_pos_emb = rotary_pos_emb.reshape(
        seq_len // spatial_merge_unit, spatial_merge_unit, -1
    )
    # Ensure rotary_pos_emb is on the same device as window_index
    rotary_pos_emb = rotary_pos_emb.to(window_index.device)
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)

    return emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)


@torch.no_grad()
def generate_torch_outputs(
    input_tensor: torch.Tensor,
    vision_attention_weights: dict[str, torch.Tensor],
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    grid_thw: torch.Tensor,
    cu_window_seqlens: torch.Tensor,
    hf_config: dict,
    use_window_attention: bool = False,
) -> torch.Tensor:
    """Generate reference outputs using HuggingFace Qwen2.5VL implementation."""
    # Create the HuggingFace attention layer
    attention_layer = (
        HFQwen2_5VLVisionAttention(
            dim=hf_config["hidden_size"],
            num_heads=hf_config["num_heads"],
        )
        .to(torch.bfloat16)
        .to("cuda")
    )

    # Load weights using state_dict
    attention_layer.load_state_dict(vision_attention_weights, strict=True)
    attention_layer.eval()

    # Generate appropriate cu_seqlens based on attention type
    seq_len, hidden_size = input_tensor.shape

    if use_window_attention:
        cu_seqlens = generate_cu_seqlens_window_attention(cu_window_seqlens)
    else:
        cu_seqlens = generate_cu_seqlens_full_attention(grid_thw)

    cu_seqlens = cu_seqlens.to(input_tensor.device)

    # Create attention layer
    with torch.no_grad():
        output = attention_layer(
            hidden_states=input_tensor,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )

    return output


def generate_max_outputs(
    input_tensor: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    grid_thw: torch.Tensor,
    cu_window_seqlens: torch.Tensor,
    qwen2_5vl_config: dict,
    dtype: DType,
    device: Device,
    use_window_attention: bool = False,
) -> torch.Tensor:
    """Generate outputs using MAX Qwen2.5VL vision attention implementation."""
    is_gpu = isinstance(device, Accelerator)
    input_tensor = input_tensor.cuda() if is_gpu else input_tensor.cpu()
    device_ref = DeviceRef.GPU() if is_gpu else DeviceRef.CPU()

    vision_config = qwen2_5vl_config["vision_config"]

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
    seq_len, hidden_size = input_tensor.shape
    input_type = TensorType(dtype, [seq_len, hidden_size], device=device_ref)

    # Generate proper attention mask based on cu_seqlens
    if use_window_attention:
        cu_seqlens = generate_cu_seqlens_window_attention(cu_window_seqlens)
    else:
        cu_seqlens = generate_cu_seqlens_full_attention(grid_thw)

    # Create attention mask from cu_seqlens
    attention_mask = torch.full(
        (1, seq_len, seq_len),
        -10000.0,
        dtype=torch.bfloat16,
        device=input_tensor.device,
    )

    # Set valid attention regions based on cu_seqlens
    for i in range(1, len(cu_seqlens)):
        attention_mask[
            ...,
            cu_seqlens[i - 1] : cu_seqlens[i],
            cu_seqlens[i - 1] : cu_seqlens[i],
        ] = 0.0

    cos_type = TensorType(
        dtype, shape=position_embeddings[0].shape, device=device_ref
    )
    sin_type = TensorType(
        dtype, shape=position_embeddings[1].shape, device=device_ref
    )
    mask_type = TensorType(dtype, shape=attention_mask.shape, device=device_ref)

    with Graph(
        "Qwen2_5VLVisionAttention",
        input_types=(input_type, cos_type, sin_type, mask_type),
    ) as graph:
        x, cos, sin, mask = graph.inputs
        output = attention(
            x=x.tensor,
            position_embeddings=(cos.tensor, sin.tensor),
            attention_mask=mask.tensor,
        )
        graph.output(output)

    compiled = session.load(graph, weights_registry=attention.state_dict())

    # Execute the model and get the first result
    result = compiled.execute(
        Tensor.from_dlpack(input_tensor).to(device),
        Tensor.from_dlpack(position_embeddings[0]).to(device),
        Tensor.from_dlpack(position_embeddings[1]).to(device),
        Tensor.from_dlpack(attention_mask).to(device),
    )
    # Convert result back to torch tensor
    max_tensor = result[0]
    return from_dlpack(max_tensor)


@pytest.mark.parametrize(
    "image_sizes",
    [
        # The quadratic attention computation (seq_len²) in the HuggingFace
        # reference creates large intermediate tensors, so test cases are
        # carefully sized to avoid OOM in CI.
        [(224, 224)],
        [(336, 224)],
        [(224, 224), (224, 224)],
    ],
)
@pytest.mark.parametrize("use_window_attention", [False, True])
def test_vision_attention_multiple_images(
    image_sizes: list[tuple[int, int]],
    use_window_attention: bool,
) -> None:
    """Test vision attention with multiple images of different sizes."""
    torch.manual_seed(42)

    # Load config and generate weights
    config_loader = get_config_loader()
    hf_config = config_loader.load_hf_vision_config(ConfigNames.QWEN2_5VL_3B)
    qwen2_5vl_config = config_loader.create_qwen2_5vl_config(
        ConfigNames.QWEN2_5VL_3B
    )

    weight_generator = get_weight_generator(ConfigNames.QWEN2_5VL_3B)

    # Create config-specific fixtures
    vision_config = qwen2_5vl_config["vision_config"]
    patch_size = vision_config["patch_size"]
    spatial_merge_size = vision_config["spatial_merge_size"]
    spatial_merge_unit = spatial_merge_size * spatial_merge_size

    # Calculate patches for each image and create grid_thw
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
    grid_thw = torch.tensor(grid_thw_list, dtype=torch.long).to("cuda")

    # Generate window index using HF Vision Transformer
    vision_transformer = HFQwen2_5VisionTransformer._from_config(
        Qwen2_5_VLVisionConfig(**vision_config)
    )
    window_index, cu_window_seqlens = vision_transformer.get_window_index(
        grid_thw
    )
    window_index = window_index.to("cuda")

    # Create vision input tensor
    vision_input_tensor = torch.randn(
        total_seq_len,
        vision_config["hidden_size"],
        dtype=torch.bfloat16,
    ).to("cuda")

    # Generate vision attention weights
    vision_attention_weights = (
        weight_generator.generate_vision_attention_weights()
    )

    # Generate rotary embeddings using the helper function
    cos_emb, sin_emb = generate_torch_rotary_embeddings(
        vision_transformer=vision_transformer,
        grid_thw=grid_thw,
        window_index=window_index,
        spatial_merge_unit=spatial_merge_unit,
        seq_len=total_seq_len,
    )

    # Use the new position_embeddings format (cos, sin tuple) that PyTorch expects
    position_embeddings = (cos_emb, sin_emb)

    # Generate reference output
    torch_output = generate_torch_outputs(
        input_tensor=vision_input_tensor,
        vision_attention_weights=vision_attention_weights,
        position_embeddings=position_embeddings,
        grid_thw=grid_thw,
        cu_window_seqlens=cu_window_seqlens,
        hf_config=hf_config,
        use_window_attention=use_window_attention,
    )

    # Generate MAX output
    max_output = generate_max_outputs(
        input_tensor=vision_input_tensor,
        attention_weights=vision_attention_weights,
        position_embeddings=position_embeddings,
        grid_thw=grid_thw,
        cu_window_seqlens=cu_window_seqlens,
        qwen2_5vl_config=qwen2_5vl_config,
        dtype=DType.bfloat16,
        device=Accelerator(),
        use_window_attention=use_window_attention,
    )

    # Verify output shape
    expected_shape = (total_seq_len, vision_config["hidden_size"])
    assert max_output.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {max_output.shape}"
    )

    # Compare outputs using the base class method
    attention_type = "window" if use_window_attention else "full"
    assert_tensors_close(
        torch_output,
        max_output,
        rtol=RTOL,
        atol=ATOL,
        message=f"Vision attention ({attention_type}) multiple images outputs do not match",
    )

    # Explicit cleanup to help with memory constraints
    del vision_transformer, vision_input_tensor, torch_output, max_output
    del vision_attention_weights, position_embeddings, cos_emb, sin_emb
    torch.cuda.empty_cache()
