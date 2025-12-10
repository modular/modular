# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Tests for Qwen3VL vision position patch embeddings."""

import pytest
import torch
from max.driver import Accelerator, Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.qwen3vl_moe.nn.data_processing import (
    get_bilinear_interpolation_weights_and_indices,
)
from max.pipelines.architectures.qwen3vl_moe.nn.visual_transformer import (
    BilinearInterpolationPositionEmbedding,
)
from torch.utils.dlpack import from_dlpack
from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeVisionConfig,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeVisionModel as HFQwen3VLMoeVisionModel,
)
from utils.assert_tensors import assert_tensors_close
from utils.config_loader import ConfigNames, get_config_loader
from utils.weight_generator import get_weight_generator

RTOL = 8e-3
ATOL = 8e-3


@torch.no_grad()
def generate_torch_outputs(
    grid_thw: torch.Tensor,
    embeddings_weights: dict[str, torch.Tensor],
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Generate reference outputs using the HuggingFace implementation."""
    # Create the reference model
    ref_model = HFQwen3VLMoeVisionModel._from_config(Qwen3VLMoeVisionConfig())
    ref_model.pos_embed.to(dtype=dtype).to(device=grid_thw.device)
    # Load weights
    ref_model.pos_embed.weight.data = (
        (embeddings_weights["pos_embed.weight"])
        .to(dtype=dtype)
        .to(device=grid_thw.device)
    )
    ref_model.eval()

    # Forward pass
    with torch.no_grad():
        hidden_states = ref_model.fast_pos_embed_interpolate(grid_thw)

    return hidden_states


def generate_max_outputs(
    grid_thw: torch.Tensor,
    qwen3vl_config: dict,
    embeddings_weights: dict[str, torch.Tensor],
    dtype: DType,
    device: Device,
) -> torch.Tensor:
    """Generate outputs using MAX VisionPatchEmbed implementation."""
    is_gpu = isinstance(device, Accelerator)
    device_ref = DeviceRef.GPU() if is_gpu else DeviceRef.CPU()
    grid_thw = grid_thw.cuda() if is_gpu else grid_thw.cpu()

    vision_config = qwen3vl_config["vision_config"]

    # Create VisionPatchEmbed with new constructor pattern
    patch_embed_module = BilinearInterpolationPositionEmbedding(
        dtype=dtype,
        device=device_ref,
        num_position_embeddings=vision_config["num_position_embeddings"],
        hidden_size=vision_config["hidden_size"],
        spatial_merge_size=vision_config["spatial_merge_size"],
    )

    embeddings_weights = {
        "embedding.weight": embeddings_weights["pos_embed.weight"]
    }

    # Load weights using state_dict
    patch_embed_module.load_state_dict(embeddings_weights, strict=True)

    session = InferenceSession(devices=[device])

    num_grid_per_side = int(vision_config["num_position_embeddings"] ** 0.5)
    np_idx, np_weights = get_bilinear_interpolation_weights_and_indices(
        grid_thw, num_grid_per_side
    )

    # Define input types
    idx_type = TensorType(DType.int64, shape=np_idx.shape, device=device_ref)
    weights_type = TensorType(
        DType.float64, shape=np_weights.shape, device=device_ref
    )

    grid_thw_type = TensorType(
        DType.int64, shape=grid_thw.shape, device=device_ref
    )

    with Graph(
        "VisionPatchPositionEmbedding",
        input_types=(idx_type, weights_type, grid_thw_type),
    ) as graph:
        idx, weights, grid = graph.inputs
        output = patch_embed_module(idx.tensor, weights.tensor, grid.tensor)
        graph.output(output)

    compiled = session.load(
        graph, weights_registry=patch_embed_module.state_dict()
    )

    # Execute the model

    result = compiled.execute(
        Tensor.from_dlpack(np_idx).to(device),
        Tensor.from_dlpack(np_weights).to(device),
        Tensor.from_dlpack(grid_thw).to(device),
    )
    max_tensor = result[0]
    return from_dlpack(max_tensor)


@pytest.mark.parametrize(
    "height,width",
    # 16x24 patches, 32x16 patches, 146x98 patches
    [(224, 320), (224, 224), (2336, 1568)],
)
def test_vision_patch_position_embedding_non_square(
    height: int, width: int
) -> None:
    """Test patch embedding for non-square images."""
    torch.manual_seed(42)

    # Load config and generate weights
    loader = get_config_loader()
    hf_full_config = loader.load_config(ConfigNames.QWEN3VL_30B)
    hf_vision_config = hf_full_config["vision_config"]
    qwen3vl_config = loader.create_qwen3vl_config(ConfigNames.QWEN3VL_30B)
    embeddings_weights = get_weight_generator(
        ConfigNames.QWEN3VL_30B
    ).generate_position_embedding_weights()

    # Create test inputs
    in_channels = hf_vision_config.get("in_channels", 3)
    temporal_patch_size = hf_vision_config["temporal_patch_size"]
    patch_size = hf_vision_config["patch_size"]

    # Calculate number of patches
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    seq_len = num_patches_h * num_patches_w
    _ = in_channels  # unused but kept for clarity of derivation
    _ = temporal_patch_size  # unused but kept for clarity of derivation

    grid_thw = torch.tensor(
        [[1, num_patches_h, num_patches_w]], dtype=torch.int64
    )

    # Generate reference output
    torch_output = generate_torch_outputs(
        grid_thw=grid_thw,
        embeddings_weights=embeddings_weights,
        dtype=torch.bfloat16,
    ).to("cpu")

    # Generate MAX output
    max_output = generate_max_outputs(
        grid_thw=grid_thw,
        qwen3vl_config=qwen3vl_config,
        embeddings_weights=embeddings_weights,
        dtype=DType.bfloat16,
        device=Accelerator(),
    ).to("cpu")

    # Verify output shape
    expected_shape = (seq_len, hf_vision_config["hidden_size"])
    assert torch_output.shape == expected_shape, (
        f"Expected shape {expected_shape}, got Torch: {torch_output.shape}, Max: {max_output.shape}"
    )
    assert max_output.shape == expected_shape, (
        f"Expected shape {expected_shape}, got Torch: {torch_output.shape}, Max: {max_output.shape}"
    )

    # Compare outputs
    assert_tensors_close(
        torch_output,
        max_output,
        rtol=RTOL,
        atol=ATOL,
        message="Vision patch embedding non-square outputs do not match",
    )


@pytest.mark.parametrize(
    "image_sizes",
    [  # Two square images of different sizes
        [(224, 224), (448, 448)],
        # Three images: non-square, non-square, square
        [(224, 352), (448, 224), (672, 672)],
        # Four images of various sizes
        [(224, 224), (448, 672), (352, 224), (672, 448)],
    ],
)
def test_vision_patch_position_embedding_multiple_images(
    image_sizes: list[tuple[int, int]],
) -> None:
    """Test patch embedding for multiple images with different sizes."""
    torch.manual_seed(42)

    # Load config and generate weights
    loader = get_config_loader()
    hf_full_config = loader.load_config(ConfigNames.QWEN3VL_30B)
    hf_vision_config = hf_full_config["vision_config"]
    qwen3vl_config = loader.create_qwen3vl_config(ConfigNames.QWEN3VL_30B)
    embeddings_weights = get_weight_generator(
        ConfigNames.QWEN3VL_30B
    ).generate_position_embedding_weights()

    # Create test inputs for multiple images
    in_channels = hf_vision_config.get("in_channels", 3)
    temporal_patch_size = hf_vision_config["temporal_patch_size"]
    patch_size = hf_vision_config["patch_size"]

    # Calculate patches for each image and create grid_thw
    grid_thw_list = []
    total_seq_len = 0

    for height, width in image_sizes:
        # Calculate number of patches for this image
        num_patches_h = height // patch_size
        num_patches_w = width // patch_size
        seq_len = num_patches_h * num_patches_w

        # Add to grid (temporal_patches=1 for images)
        grid_thw_list.append([1, num_patches_h, num_patches_w])
        total_seq_len += seq_len

    # Create grid_thw tensor [n_images, 3]
    grid_thw = torch.tensor(grid_thw_list, dtype=torch.int64)

    # Generate reference output
    torch_output = generate_torch_outputs(
        grid_thw=grid_thw,
        embeddings_weights=embeddings_weights,
        dtype=torch.bfloat16,
    ).to("cpu")

    # Generate MAX output
    max_output = generate_max_outputs(
        grid_thw=grid_thw,
        qwen3vl_config=qwen3vl_config,
        embeddings_weights=embeddings_weights,
        dtype=DType.bfloat16,
        device=Accelerator(),
    ).to("cpu")

    # Verify output shape
    expected_shape = (total_seq_len, hf_vision_config["hidden_size"])
    assert max_output.shape == expected_shape, (
        f"Expected shape {expected_shape}, got MAX output shape: {max_output.shape}"
    )
    assert torch_output.shape == expected_shape, (
        f"Expected shape {expected_shape}, got Torch output shape: {torch_output.shape}"
    )

    # Compare outputs
    assert_tensors_close(
        torch_output,
        max_output,
        rtol=RTOL,
        atol=ATOL,
        message=f"Vision patch embedding multiple images of size {image_sizes} outputs do not match",
    )
