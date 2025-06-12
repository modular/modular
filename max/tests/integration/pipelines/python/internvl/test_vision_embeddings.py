# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest
import torch
from max.driver import Accelerator, Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.internvl.internvl import InternVisionEmbeddings
from max.pipelines.architectures.internvl.model_config import VisionConfig
from torch.utils.dlpack import from_dlpack


# vLLM reference implementation
class InternVisionEmbeddingsReference(torch.nn.Module):
    """Reference implementation from vLLM."""

    def __init__(self, hidden_size: int, image_size: int, patch_size: int):
        super().__init__()
        self.embed_dim = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size

        self.class_embedding = torch.nn.Parameter(
            torch.randn(1, 1, self.embed_dim)
        )

        self.patch_embedding = torch.nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.position_embedding = torch.nn.Parameter(
            torch.randn(1, self.num_positions, self.embed_dim)
        )

    def _get_pos_embed(self, pos_embed: torch.Tensor, H: int, W: int):
        target_dtype = pos_embed.dtype
        pos_embed = (
            pos_embed.float()
            .reshape(
                1,
                self.image_size // self.patch_size,
                self.image_size // self.patch_size,
                -1,
            )
            .permute(0, 3, 1, 2)
        )
        pos_embed = torch.nn.functional.interpolate(
            pos_embed, size=(H, W), mode="bicubic", align_corners=False
        )
        return pos_embed.reshape(1, -1, H * W).permute(0, 2, 1).to(target_dtype)

    def _get_position_embedding(self, H: int, W: int) -> torch.Tensor:
        position_embedding = self.position_embedding
        if self.num_patches == H * W:
            return position_embedding

        return torch.cat(
            [
                position_embedding[:, :1, :],
                self._get_pos_embed(position_embedding[:, 1:, :], H, W),
            ],
            dim=1,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(
            pixel_values.to(target_dtype)
        )  # shape = [*, channel, width, height]
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(
            target_dtype
        )
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        position_embedding = self._get_position_embedding(height, width)
        embeddings = embeddings + position_embedding.to(target_dtype)
        return embeddings


@torch.no_grad()
def generate_torch_outputs(
    vision_config: VisionConfig,
    pixel_values: torch.Tensor,
    embeddings_weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Generate reference outputs using the vLLM InternVision implementation."""

    # Create reference model
    ref_embeddings = (
        InternVisionEmbeddingsReference(
            hidden_size=vision_config.hidden_size,
            image_size=vision_config.image_size,
            patch_size=vision_config.patch_size,
        )
        .to(torch.bfloat16)
        .to("cuda")
    )

    # Load weights
    ref_embeddings.patch_embedding.weight.data = (
        embeddings_weights["patch_embedding.weight"]
        .to(torch.bfloat16)
        .to("cuda")
    )
    if "patch_embedding.bias" in embeddings_weights:
        ref_embeddings.patch_embedding.bias.data = (
            embeddings_weights["patch_embedding.bias"]
            .to(torch.bfloat16)
            .to("cuda")
        )
    ref_embeddings.class_embedding.data = (
        embeddings_weights["class_embedding"].to(torch.bfloat16).to("cuda")
    )
    ref_embeddings.position_embedding.data = (
        embeddings_weights["position_embedding"].to(torch.bfloat16).to("cuda")
    )

    # Forward pass
    return ref_embeddings(pixel_values)


def generate_max_outputs(
    vision_config: VisionConfig,
    pixel_values: torch.Tensor,
    embeddings_weights: dict[str, torch.Tensor],
    dtype: DType,
    device: Device,
) -> torch.Tensor:
    """Generate outputs using MAX InternVisionEmbeddings implementation."""
    is_gpu = isinstance(device, Accelerator)
    pixel_values = pixel_values.cuda() if is_gpu else pixel_values.cpu()
    device_ref = DeviceRef.GPU() if is_gpu else DeviceRef.CPU()

    # Create a minimal config object for InternVisionEmbeddings
    class MinimalConfig:
        def __init__(self, vision_config: VisionConfig):
            self.vision_config = vision_config

    config = MinimalConfig(vision_config)

    # Prepare state dict.
    state_dict = {}
    for weight_name, value in embeddings_weights.items():
        state_dict[weight_name] = value.cpu()

    # Create the embeddings layer.
    embeddings = InternVisionEmbeddings(config, device_ref)  # type: ignore[arg-type]
    embeddings.load_state_dict(
        state_dict=state_dict,
        override_quantization_encoding=True,
        weight_alignment=1,
    )

    # Build the graph with symbolic dimensions to test the TypeError fix
    input_type = TensorType(
        dtype, shape=("batch", 3, "height", "width"), device=device_ref
    )

    graph = Graph(
        "InternVisionEmbeddings", forward=embeddings, input_types=(input_type,)
    )

    session = InferenceSession(devices=[Accelerator(0)])
    compiled = session.load(graph, weights_registry=state_dict)

    # Execute the model
    result = compiled.execute(Tensor.from_dlpack(pixel_values).to(device))
    # Convert result back to torch tensor
    max_tensor = result[0]
    return from_dlpack(max_tensor)


@pytest.mark.parametrize(
    "original_size,target_size",
    [
        (448, 224),  # Downscale
        (224, 448),  # Upscale
        (448, 896),  # 2x upscale
    ],
)
def test_vision_embeddings(
    original_size: int,
    target_size: int,
) -> None:
    """Test position embedding interpolation for different resolutions."""
    # Use fixed parameters
    patch_size = 14
    hidden_size = 768

    # Create vision config with original size
    vision_config = VisionConfig(
        dtype=DType.bfloat16,
        hidden_size=hidden_size,
        intermediate_size=4 * hidden_size,
        norm_type="layer_norm",
        image_size=original_size,
        patch_size=patch_size,
        num_attention_heads=hidden_size // 64,
        head_dim=64,
        layer_norm_eps=1e-6,
        qk_normalization=True,
    )

    # Create test inputs with target size
    torch.manual_seed(42)
    batch_size = 1
    # Pixel values after ImageNet normalization typically have std ~1.04
    pixel_values = torch.randn(
        batch_size, 3, target_size, target_size, dtype=torch.bfloat16
    ).to("cuda")

    # Create test weights for original size
    num_patches_original = (original_size // patch_size) ** 2
    num_positions_original = num_patches_original + 1

    # Weight initialization based on real InternVL model statistics:
    # ┌─────────────────────┬──────────┬───────────────┬───────────────┐
    # │ Weight              │ Std Dev  │ Range Min     │ Range Max     │
    # ├─────────────────────┼──────────┼───────────────┼───────────────┤
    # │ patch_embedding     │ 0.0134   │ -0.1328       │ 0.1162        │
    # │ patch_embedding.bias│ 0.0078   │ -0.0317       │ 0.0264        │
    # │ class_embedding     │ 0.1494   │ -1.8594       │ 0.8320        │
    # │ position_embedding  │ 0.0177   │ -0.1318       │ 0.1387        │
    # └─────────────────────┴──────────┴───────────────┴───────────────┘
    # Note: pixel_values after ImageNet normalization have std ~1.04

    # Use realistic weight initialization based on real InternVL model
    embeddings_weights = {
        "patch_embedding.weight": torch.randn(
            hidden_size, 3, patch_size, patch_size, dtype=torch.bfloat16
        )
        * 0.0134,  # Real model std: 0.0134
        "patch_embedding.bias": torch.randn(hidden_size, dtype=torch.bfloat16)
        * 0.0078,  # Real model std: 0.0078
        "class_embedding": torch.randn(1, 1, hidden_size, dtype=torch.bfloat16)
        * 0.1494,  # Real model std: 0.1494
        "position_embedding": torch.randn(
            1, num_positions_original, hidden_size, dtype=torch.bfloat16
        )
        * 0.0177,  # Real model std: 0.0177
    }

    # Generate reference output
    torch_output = generate_torch_outputs(
        vision_config, pixel_values, embeddings_weights
    )

    # Generate MAX output
    max_output = generate_max_outputs(
        vision_config=vision_config,
        pixel_values=pixel_values,
        embeddings_weights=embeddings_weights,
        dtype=DType.bfloat16,
        device=Accelerator(),
    )

    # Verify output shape
    expected_num_patches = (target_size // patch_size) ** 2
    expected_shape = (batch_size, expected_num_patches + 1, hidden_size)
    assert max_output.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {max_output.shape}"
    )

    # Compare outputs
    torch.testing.assert_close(
        torch_output.to(torch.bfloat16),
        max_output.to(torch.bfloat16),
        rtol=1e-2,
        atol=1e-2,
    )
