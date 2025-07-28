# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Tests for InternVL vision embeddings."""

import os

import numpy as np
import pytest
import torch
from internvl_impl.configuration_intern_vit import (
    InternVisionConfig as HFInternVisionConfig,
)
from internvl_impl.modeling_intern_vit import (
    InternVisionEmbeddings as HFInternVisionEmbeddings,
)
from max.driver import CPU, Accelerator, Device, Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.internvl.internvl import (
    InternVisionEmbeddings,
)
from max.pipelines.architectures.internvl.model_config import (
    InternVLConfig,
)
from max.pipelines.architectures.internvl.tokenizer import (
    extract_patches_from_image,
)
from torch.utils.dlpack import from_dlpack
from utils.assert_tensors import assert_tensors_close
from utils.config_loader import ConfigNames, get_config_loader
from utils.weight_converter import convert_hf_to_max_weights
from utils.weight_generator import get_weight_generator


@torch.no_grad()
def generate_torch_outputs(
    pixel_values: torch.Tensor,
    hf_config: HFInternVisionConfig,
    embeddings_weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Generate reference outputs using the vLLM implementation."""

    # Create the reference model
    ref_model = (
        HFInternVisionEmbeddings(
            hf_config,
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
    internvl_config: InternVLConfig,
    embeddings_weights: dict[str, torch.Tensor],
    dtype: DType,
    device: Device,
) -> torch.Tensor:
    """Generate outputs using MAX InternVisionEmbeddings implementation."""
    is_gpu = isinstance(device, Accelerator)
    device_ref = DeviceRef.GPU() if is_gpu else DeviceRef.CPU()
    pixel_values = pixel_values.cuda() if is_gpu else pixel_values.cpu()

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
            img, patch_size=internvl_config.vision_config.patch_size
        )
        all_patches.append(patches)

    # Stack all patches - shape: (batch_size, height_patches, width_patches, channels, patch_size, patch_size)
    pixel_values_np = np.stack(all_patches)

    # Convert back to torch tensor and move to device
    pixel_values = (
        torch.from_numpy(pixel_values_np).to(dtype=torch.bfloat16).cuda()
    ).contiguous()

    max_weights = convert_hf_to_max_weights(embeddings_weights)

    # Create MAX embeddings
    embeddings = InternVisionEmbeddings(internvl_config, device=device_ref)

    # Load weights
    embeddings.load_state_dict(
        state_dict=max_weights,
        strict=True,
    )

    session = InferenceSession(devices=[device])

    input_type = TensorType(dtype, shape=pixel_values.shape, device=device_ref)

    with Graph("InternVisionEmbeddings", input_types=(input_type,)) as graph:
        x = graph.inputs[0]
        output = embeddings(x.tensor)
        graph.output(output)

    compiled = session.load(graph, weights_registry=embeddings.state_dict())

    # Execute the model and get the first result
    result = compiled.execute(Tensor.from_dlpack(pixel_values).to(device))
    max_tensor = result[0]
    return from_dlpack(max_tensor)


@pytest.mark.parametrize(
    "config_name",
    [
        pytest.param(ConfigNames.INTERNVL_2B),
        pytest.param(
            ConfigNames.INTERNVL_8B,
            marks=[
                pytest.mark.skipif(
                    not os.environ.get("INTERNVL_8B_TESTS"),
                    reason="8B tests disabled (set INTERNVL_8B_TESTS env var to enable)",
                ),
            ],
        ),
        pytest.param(
            ConfigNames.INTERNVL_38B,
            marks=[
                pytest.mark.skipif(
                    not os.environ.get("INTERNVL_38B_TESTS"),
                    reason="38B tests disabled (set INTERNVL_38B_TESTS env var to enable)",
                ),
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    "target_size",
    [
        224,  # Downscale from 448 (16 patches)
        448,  # Same size (32 patches)
        672,  # 1.5x upscale from 448 (48 patches)
    ],
)
def test_vision_embeddings(config_name: ConfigNames, target_size: int) -> None:
    """Test position embedding interpolation for different resolutions.
    Note: Image dimensions must be divisible by patch_size (14) for the
    reshape operations to work correctly.
    """
    torch.manual_seed(42)

    # Load HuggingFace config and generate weights
    hf_config = get_config_loader().load_hf_vision_config(config_name)
    internvl_config = get_config_loader().create_internvl_config(config_name)
    embeddings_weights = get_weight_generator(
        config_name
    ).generate_vision_embeddings_weights()

    # Create test inputs with target size
    batch_size = 1
    pixel_values = torch.randn(
        batch_size, 3, target_size, target_size, dtype=torch.bfloat16
    ).to("cuda")

    # Generate reference output
    torch_output = generate_torch_outputs(
        pixel_values, hf_config, embeddings_weights
    )

    # Generate MAX output
    max_output = generate_max_outputs(
        pixel_values,
        internvl_config,
        embeddings_weights,
        DType.bfloat16,
        Accelerator(),
    )

    # Verify output shape
    patch_size = internvl_config.vision_config.patch_size
    expected_num_patches = (target_size // patch_size) ** 2
    expected_shape = (
        batch_size,
        expected_num_patches + 1,
        internvl_config.vision_config.hidden_size,
    )
    assert max_output.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {max_output.shape}"
    )

    # Compare outputs
    assert_tensors_close(
        torch_output,
        max_output,
        rtol=2 * torch.finfo(torch.bfloat16).eps,
        atol=8 * torch.finfo(torch.bfloat16).eps,
        message="Vision embeddings outputs do not match",
    )


@pytest.mark.parametrize(
    "config_name",
    [
        pytest.param(ConfigNames.INTERNVL_2B),
        pytest.param(
            ConfigNames.INTERNVL_8B,
            marks=[
                pytest.mark.skipif(
                    not os.environ.get("INTERNVL_8B_TESTS"),
                    reason="8B tests disabled (set INTERNVL_8B_TESTS env var to enable)",
                ),
            ],
        ),
        pytest.param(
            ConfigNames.INTERNVL_38B,
            marks=[
                pytest.mark.skipif(
                    not os.environ.get("INTERNVL_38B_TESTS"),
                    reason="38B tests disabled (set INTERNVL_38B_TESTS env var to enable)",
                ),
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    "height,width",
    [
        (224, 224),  # Small square (16x16 patches)
        (448, 672),  # Non-square, moderate size (32x48 patches)
    ],
)
def test_vision_embeddings_non_square(
    config_name: ConfigNames, height: int, width: int
) -> None:
    """Test position embedding interpolation for non-square images."""
    # Set seed for deterministic results
    torch.manual_seed(42)

    # Create test instance
    hf_config = get_config_loader().load_hf_vision_config(config_name)
    internvl_config = get_config_loader().create_internvl_config(config_name)
    embeddings_weights = get_weight_generator(
        config_name
    ).generate_vision_embeddings_weights()

    # Create test inputs with specific height and width
    batch_size = 1
    pixel_values = torch.randn(
        batch_size, 3, height, width, dtype=torch.bfloat16
    ).to("cuda")

    # Generate reference output
    torch_output = generate_torch_outputs(
        pixel_values, hf_config, embeddings_weights
    )

    # Generate MAX output
    max_output = generate_max_outputs(
        pixel_values,
        internvl_config,
        embeddings_weights,
        DType.bfloat16,
        Accelerator(),
    )

    # Verify output shape
    patch_size = internvl_config.vision_config.patch_size
    expected_num_patches = (height // patch_size) * (width // patch_size)
    expected_shape = (
        batch_size,
        expected_num_patches + 1,
        internvl_config.vision_config.hidden_size,
    )
    assert max_output.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {max_output.shape}"
    )

    # Compare outputs
    assert_tensors_close(
        torch_output,
        max_output,
        rtol=2 * torch.finfo(torch.bfloat16).eps,
        atol=8 * torch.finfo(torch.bfloat16).eps,
        message="Vision embeddings non-square outputs do not match",
    )


@pytest.mark.parametrize(
    "config_name",
    [
        pytest.param(ConfigNames.INTERNVL_2B),
        pytest.param(
            ConfigNames.INTERNVL_8B,
            marks=[
                pytest.mark.skipif(
                    not os.environ.get("INTERNVL_8B_TESTS"),
                    reason="8B tests disabled (set INTERNVL_8B_TESTS env var to enable)",
                ),
            ],
        ),
        pytest.param(
            ConfigNames.INTERNVL_38B,
            marks=[
                pytest.mark.skipif(
                    not os.environ.get("INTERNVL_38B_TESTS"),
                    reason="38B tests disabled (set INTERNVL_38B_TESTS env var to enable)",
                ),
            ],
        ),
    ],
)
@pytest.mark.parametrize("n_devices", [2, 4])
def test_vision_embeddings_multi_gpu(
    config_name: ConfigNames, n_devices: int
) -> None:
    """Test InternVisionEmbeddings sharding with multiple GPUs."""
    # Set seed for deterministic results
    torch.manual_seed(42)

    if n_devices > accelerator_count():
        pytest.skip(f"Not enough GPUs to run test with {n_devices} GPUs.")

    # Create test instance
    internvl_config = get_config_loader().create_internvl_config(config_name)
    embeddings_weights = get_weight_generator(
        config_name
    ).generate_vision_embeddings_weights()

    # Convert HuggingFace weights to MAX format using weight converter
    max_weights = convert_hf_to_max_weights(embeddings_weights)

    # Build graph to test sharding
    # Calculate expected patch dimensions
    patch_size = internvl_config.vision_config.patch_size
    height_patches = internvl_config.vision_config.image_size // patch_size
    width_patches = internvl_config.vision_config.image_size // patch_size
    channels = 3
    batch_size = 2  # Simulating 2 images

    input_type = TensorType(
        internvl_config.vision_config.dtype,
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
        embeddings = InternVisionEmbeddings(internvl_config, DeviceRef.GPU(0))
        embeddings.load_state_dict(
            state_dict=max_weights,
        )

        # Set sharding strategy for replication across devices
        from max.graph import ShardingStrategy

        embeddings.sharding_strategy = ShardingStrategy.replicate(
            num_devices=n_devices
        )

        # Test that we can shard to each device
        devices = [DeviceRef.GPU(i) for i in range(n_devices)]
        shards = embeddings.shard(devices)
        for sharded in shards:
            # Verify sharded embeddings maintain configuration
            assert isinstance(sharded, InternVisionEmbeddings)
            assert (
                sharded.embed_dim == internvl_config.vision_config.hidden_size
            )
            assert (
                sharded.image_size == internvl_config.vision_config.image_size
            )
            assert (
                sharded.patch_size == internvl_config.vision_config.patch_size
            )

    # Test passes if sharding completes without errors
    print(f"Successfully tested sharding with {n_devices} devices")


@pytest.mark.parametrize(
    "config_name",
    [
        pytest.param(ConfigNames.INTERNVL_2B),
        pytest.param(
            ConfigNames.INTERNVL_8B,
            marks=[
                pytest.mark.skipif(
                    not os.environ.get("INTERNVL_8B_TESTS"),
                    reason="8B tests disabled (set INTERNVL_8B_TESTS env var to enable)",
                ),
            ],
        ),
        pytest.param(
            ConfigNames.INTERNVL_38B,
            marks=[
                pytest.mark.skipif(
                    not os.environ.get("INTERNVL_38B_TESTS"),
                    reason="38B tests disabled (set INTERNVL_38B_TESTS env var to enable)",
                ),
            ],
        ),
    ],
)
def test_vision_embeddings_multi_gpu_execution(
    config_name: ConfigNames,
) -> None:
    """Test InternVisionEmbeddings execution on 2 GPUs with large image."""
    # Set seed for deterministic results
    torch.manual_seed(42)

    n_devices = 2
    if n_devices > accelerator_count():
        pytest.skip(f"Not enough GPUs to run test with {n_devices} GPUs.")

    # Load HuggingFace config and generate weights
    internvl_config = get_config_loader().create_internvl_config(config_name)
    embeddings_weights = get_weight_generator(
        config_name
    ).generate_vision_embeddings_weights()

    height, width = 896, 672
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
            pixel_values_np[i],
            patch_size=internvl_config.vision_config.patch_size,
        )
        all_patches.append(patches)
    patches = np.stack(all_patches).astype(np.float32)

    # Shape is now (batch_size, height_patches, width_patches, channels, patch_size, patch_size)
    input_shape = patches.shape

    devices = [DeviceRef.GPU(0), DeviceRef.GPU(1)]

    # Convert HuggingFace weights to MAX format using weight converter
    max_weights = convert_hf_to_max_weights(embeddings_weights)

    # Build graph with f32 input
    input_type_gpu0 = TensorType(
        DType.float32,  # f32 input
        shape=input_shape,
        device=DeviceRef.GPU(0),
    )

    # Create embeddings for each GPU
    embeddings_gpu0 = InternVisionEmbeddings(internvl_config, DeviceRef.GPU(0))
    embeddings_gpu0.load_state_dict(
        state_dict=max_weights,
    )

    embeddings_gpu1 = InternVisionEmbeddings(internvl_config, DeviceRef.GPU(1))
    embeddings_gpu1.load_state_dict(
        state_dict=max_weights,
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

    compiled0 = session0.load(graph0, weights_registry=max_weights)
    compiled1 = session1.load(graph1, weights_registry=max_weights)

    # Convert patches to torch tensor for execution.
    patches_tensor = torch.from_numpy(patches).to(torch.float32)

    # Execute on GPU 0
    patches_gpu0 = patches_tensor.to("cuda:0")
    input_tensor0 = Tensor.from_dlpack(patches_gpu0.contiguous()).to(
        Accelerator(0)
    )
    result0 = compiled0.execute(input_tensor0)[0]
    assert isinstance(result0, Tensor)
    result0 = result0.to(CPU())

    # Copy input to GPU 1 and execute
    patches_gpu1 = patches_tensor.to("cuda:1")
    input_tensor1 = Tensor.from_dlpack(patches_gpu1.contiguous()).to(
        Accelerator(1)
    )
    result1 = compiled1.execute(input_tensor1)[0]
    assert isinstance(result1, Tensor)
    result1 = result1.to(CPU())

    output0 = from_dlpack(result0)
    output1 = from_dlpack(result1)

    # Verify output shape
    expected_num_patches = (
        height // internvl_config.vision_config.patch_size
    ) * (width // internvl_config.vision_config.patch_size)
    expected_shape = (
        batch_size,
        expected_num_patches + 1,
        internvl_config.vision_config.hidden_size,
    )

    assert output0.shape == expected_shape
    assert output1.shape == expected_shape

    # Compare outputs from both GPUs.
    assert_tensors_close(
        output0,
        output1,
        rtol=2 * torch.finfo(torch.bfloat16).eps,
        atol=8 * torch.finfo(torch.bfloat16).eps,
        message="Vision embeddings multi-GPU outputs do not match",
    )

    print(
        f"Successfully tested execution on 2 GPUs with {height}x{width} image"
    )
