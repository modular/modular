# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import numpy as np
import pytest
import torch
from max.driver import CPU, Accelerator, Device, Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.internvl.internvl import InternVisionEmbeddings
from max.pipelines.architectures.internvl.model_config import VisionConfig
from max.pipelines.architectures.internvl.tokenizer import (
    extract_patches_from_image,
)
from torch.utils.dlpack import from_dlpack


# vLLM reference implementation
class InternVisionEmbeddingsReference(torch.nn.Module):
    """Reference implementation from vLLM."""

    def __init__(
        self, *, hidden_size: int, image_size: int, patch_size: int
    ) -> None:
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
        if ref_embeddings.patch_embedding.bias is not None:
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


def prepare_embeddings_weights_for_max(
    embeddings_weights: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Prepare embedding weights for MAX by converting Conv2D to Linear format.

    MAX expects Linear weights, but the reference uses Conv2D weights.
    This helper converts patch_embedding.weight from Conv2D format
    (out_channels, in_channels, kernel_h, kernel_w) to Linear format
    (out_channels, in_channels * kernel_h * kernel_w) and moves all
    weights to CPU.

    Args:
        embeddings_weights: Dictionary of embedding weights.

    Returns:
        Dictionary with converted weights on CPU.
    """
    state_dict = {}
    for weight_name, value in embeddings_weights.items():
        if weight_name == "patch_embedding.weight":
            # Convert Conv2D weights to Linear format as expected by MAX
            # Conv2D shape: (out_channels, in_channels, kernel_h, kernel_w)
            # Linear shape: (out_channels, in_channels * kernel_h * kernel_w)
            out_channels, in_channels, kernel_h, kernel_w = value.shape
            value = value.reshape(
                out_channels, in_channels * kernel_h * kernel_w
            )
        state_dict[weight_name] = value.cpu()
    return state_dict


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

    # Convert from PyTorch's NCHW to MAX's NHWC format
    pixel_values = pixel_values.permute(0, 2, 3, 1).contiguous()

    # Extract shape information
    batch_size = pixel_values.shape[0]

    # Convert to numpy for processing
    # Convert to float32 first since numpy doesn't support bfloat16
    pixel_values_np = pixel_values.cpu().float().numpy()

    # Split each image in the batch into patches.
    all_patches = []
    for i in range(batch_size):
        img = pixel_values_np[i]
        patches = extract_patches_from_image(
            img, patch_size=vision_config.patch_size
        )
        all_patches.append(patches)

    # Stack all patches - shape: (batch_size, height_patches, width_patches, channels, patch_size, patch_size)
    pixel_values_np = np.stack(all_patches)

    # Convert back to torch tensor and move to device
    pixel_values = (
        torch.from_numpy(pixel_values_np).to(dtype=torch.bfloat16).cuda()
    ).contiguous()

    # Create a minimal config object for InternVisionEmbeddings
    class MinimalConfig:
        def __init__(self, vision_config: VisionConfig) -> None:
            self.vision_config = vision_config
            self.devices = [device_ref]  # Add devices attribute

    config = MinimalConfig(vision_config)

    state_dict = prepare_embeddings_weights_for_max(embeddings_weights)

    # Create the embeddings layer.
    embeddings = InternVisionEmbeddings(config, device_ref)  # type: ignore[arg-type]
    embeddings.load_state_dict(
        state_dict=state_dict,
        override_quantization_encoding=True,
        weight_alignment=1,
    )

    # Build the graph with symbolic batch dim.
    # pixel_values now has shape: (batch_size, height_patches, width_patches, channels, patch_size, patch_size)
    input_type = TensorType(dtype, shape=pixel_values.shape, device=device_ref)

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
    "target_size",
    [
        224,  # Downscale from 448 (16 patches)
        448,  # Same size (32 patches)
        896,  # 2x upscale from 448 (64 patches)
        2688,  # Large size (192 patches)
    ],
)
def test_vision_embeddings(
    vision_config: VisionConfig,
    embeddings_weights: dict[str, torch.Tensor],
    target_size: int,
) -> None:
    """Test position embedding interpolation for different resolutions.

    Note: Image dimensions must be divisible by patch_size (14) for the
    reshape operations to work correctly.
    """
    # Create test inputs with target size
    batch_size = 1
    pixel_values = torch.randn(
        batch_size, 3, target_size, target_size, dtype=torch.bfloat16
    ).to("cuda")

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
    expected_num_patches = (target_size // vision_config.patch_size) ** 2
    expected_shape = (
        batch_size,
        expected_num_patches + 1,
        vision_config.hidden_size,
    )
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


@pytest.mark.parametrize(
    "height,width",
    [
        (224, 224),  # Small square (16x16 patches)
        (448, 672),  # Non-square, moderate size (32x48 patches)
        (
            2688,
            2044,
        ),  # Large size, adjusted to be divisible by 14 (192x146 patches)
        (
            1918,
            1078,
        ),  # HD-like resolution, adjusted to be divisible by 14 (137x77 patches)
    ],
)
def test_vision_embeddings_non_square(
    vision_config: VisionConfig,
    embeddings_weights: dict[str, torch.Tensor],
    height: int,
    width: int,
) -> None:
    """Test position embedding interpolation for non-square images."""
    # Create test inputs with specific height and width
    batch_size = 1
    pixel_values = torch.randn(
        batch_size, 3, height, width, dtype=torch.bfloat16
    ).to("cuda")

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
    expected_num_patches = (height // vision_config.patch_size) * (
        width // vision_config.patch_size
    )
    expected_shape = (
        batch_size,
        expected_num_patches + 1,
        vision_config.hidden_size,
    )
    assert max_output.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {max_output.shape}"
    )

    # Compare outputs
    torch.testing.assert_close(
        torch_output.to(torch.bfloat16),
        max_output.to(torch.bfloat16),
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.parametrize("n_devices", [2, 4])
def test_vision_embeddings_multi_gpu(
    vision_config: VisionConfig,
    embeddings_weights: dict[str, torch.Tensor],
    n_devices: int,
) -> None:
    """Test InternVisionEmbeddings sharding with multiple GPUs."""
    if n_devices > accelerator_count():
        pytest.skip(f"Not enough GPUs to run test with {n_devices} GPUs.")

    # Test with multiple devices configuration
    class MultiGPUConfig:
        def __init__(
            self, vision_config: VisionConfig, devices: list[DeviceRef]
        ) -> None:
            self.vision_config = vision_config
            self.devices = devices

    devices = [DeviceRef.GPU(i) for i in range(n_devices)]
    config = MultiGPUConfig(vision_config, devices)

    state_dict = prepare_embeddings_weights_for_max(embeddings_weights)

    # Build graph to test sharding
    # Calculate expected patch dimensions
    patch_size = vision_config.patch_size
    height_patches = vision_config.image_size // patch_size
    width_patches = vision_config.image_size // patch_size
    channels = 3
    batch_size = 2  # Simulating 2 images

    input_type = TensorType(
        vision_config.dtype,
        shape=(
            batch_size,
            height_patches,
            width_patches,
            channels,
            patch_size,
            patch_size,
        ),
        device=DeviceRef.GPU(0),
    )

    with Graph(
        "test_sharding",
        input_types=(input_type,),
    ):
        # Create embeddings and set sharding strategy
        embeddings = InternVisionEmbeddings(config, DeviceRef.GPU(0))  # type: ignore[arg-type]
        embeddings.load_state_dict(
            state_dict=state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
        )

        # Set sharding strategy for replication across devices
        from max.graph import ShardingStrategy

        embeddings.sharding_strategy = ShardingStrategy.replicate(
            num_devices=n_devices
        )

        # Test that we can shard to each device
        for shard_idx in range(n_devices):
            device = DeviceRef.GPU(shard_idx)
            sharded = embeddings.shard(shard_idx, device)

            # Verify sharded embeddings maintain configuration
            assert isinstance(sharded, InternVisionEmbeddings)
            assert sharded.embed_dim == vision_config.hidden_size
            assert sharded.image_size == vision_config.image_size
            assert sharded.patch_size == vision_config.patch_size

    # Test passes if sharding completes without errors
    print(f"Successfully tested sharding with {n_devices} devices")


def test_vision_embeddings_multi_gpu_execution(
    vision_config: VisionConfig,
    embeddings_weights: dict[str, torch.Tensor],
) -> None:
    """Test InternVisionEmbeddings execution on 2 GPUs with large image."""
    n_devices = 2
    if n_devices > accelerator_count():
        pytest.skip(f"Not enough GPUs to run test with {n_devices} GPUs.")

    # Use large image dimensions to stress test
    # Dimensions must be divisible by patch_size (14)
    height, width = 2688, 2044
    batch_size = 1  # Define batch_size explicitly

    # Create test inputs
    pixel_values = torch.randn(
        batch_size, 3, height, width, dtype=torch.float32
    ).to("cuda")

    # Convert to patches for the test
    pixel_values_np = (
        pixel_values.permute(0, 2, 3, 1).cpu().numpy()
    )  # NCHW -> NHWC

    # Split each image in the batch into patches.
    all_patches = []
    for i in range(batch_size):
        patches = extract_patches_from_image(
            pixel_values_np[i], patch_size=vision_config.patch_size
        )
        all_patches.append(patches)
    patches = np.stack(all_patches).astype(np.float32)

    # Shape is now (batch_size, height_patches, width_patches, channels, patch_size, patch_size)
    input_shape = patches.shape

    # Create multi-GPU config
    class MultiGPUConfig:
        def __init__(
            self, vision_config: VisionConfig, devices: list[DeviceRef]
        ) -> None:
            self.vision_config = vision_config
            self.devices = devices

    devices = [DeviceRef.GPU(0), DeviceRef.GPU(1)]
    config = MultiGPUConfig(vision_config, devices)

    state_dict = prepare_embeddings_weights_for_max(embeddings_weights)

    # Build graph with f32 input using patch dimensions
    input_type_gpu0 = TensorType(
        DType.float32,  # f32 input
        shape=input_shape,
        device=DeviceRef.GPU(0),
    )

    # Create embeddings for each GPU
    embeddings_gpu0 = InternVisionEmbeddings(config, DeviceRef.GPU(0))  # type: ignore[arg-type]
    embeddings_gpu0.load_state_dict(
        state_dict=state_dict,
        override_quantization_encoding=True,
        weight_alignment=1,
    )

    embeddings_gpu1 = InternVisionEmbeddings(config, DeviceRef.GPU(1))  # type: ignore[arg-type]
    embeddings_gpu1.load_state_dict(
        state_dict=state_dict,
        override_quantization_encoding=True,
        weight_alignment=1,
    )

    # Build separate graphs for each GPU
    graph0 = Graph(
        "InternVisionEmbeddings_GPU0",
        forward=embeddings_gpu0,
        input_types=(input_type_gpu0,),
    )

    input_type_gpu1 = TensorType(
        DType.float32,
        shape=input_shape,
        device=DeviceRef.GPU(1),
    )

    graph1 = Graph(
        "InternVisionEmbeddings_GPU1",
        forward=embeddings_gpu1,
        input_types=(input_type_gpu1,),
    )

    # Create sessions and compile
    session0 = InferenceSession(devices=[Accelerator(0)])
    session1 = InferenceSession(devices=[Accelerator(1)])

    compiled0 = session0.load(graph0, weights_registry=state_dict)
    compiled1 = session1.load(graph1, weights_registry=state_dict)

    # Convert patches to torch tensor for execution.
    patches_tensor = torch.from_numpy(patches).to(torch.float32).contiguous()

    # Execute on GPU 0
    patches_gpu0 = patches_tensor.to("cuda:0")
    input_tensor0 = Tensor.from_dlpack(patches_gpu0).to(Accelerator(0))
    result0 = compiled0.execute(input_tensor0)[0]
    assert isinstance(result0, Tensor)
    result0 = result0.to(CPU())

    # Copy input to GPU 1 and execute
    patches_gpu1 = patches_tensor.to("cuda:1")
    input_tensor1 = Tensor.from_dlpack(patches_gpu1).to(Accelerator(1))
    result1 = compiled1.execute(input_tensor1)[0]
    assert isinstance(result1, Tensor)
    result1 = result1.to(CPU())

    output0 = from_dlpack(result0)
    output1 = from_dlpack(result1)

    # Verify output shape
    expected_num_patches = (height // vision_config.patch_size) * (
        width // vision_config.patch_size
    )
    expected_shape = (
        batch_size,
        expected_num_patches + 1,
        vision_config.hidden_size,
    )

    assert output0.shape == expected_shape
    assert output1.shape == expected_shape

    # Compare outputs from both GPUs.
    torch.testing.assert_close(output0, output1, rtol=1e-2, atol=1e-2)

    print(
        f"Successfully tested execution on 2 GPUs with {height}x{width} image"
    )
