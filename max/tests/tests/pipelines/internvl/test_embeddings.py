# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Unit tests for InternVisionEmbeddings."""

from __future__ import annotations

import pytest
from max.driver import accelerator_count
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.architectures.internvl.internvl import InternVisionEmbeddings
from max.pipelines.architectures.internvl.model_config import (
    VisionConfig,
)


def create_test_vision_config(
    hidden_size: int = 1024,
    image_size: int = 448,
    patch_size: int = 14,
    dtype: DType = DType.bfloat16,
) -> VisionConfig:
    """Create a test VisionConfig."""
    return VisionConfig(
        dtype=dtype,
        hidden_size=hidden_size,
        intermediate_size=4096,
        norm_type="layer_norm",
        image_size=image_size,
        patch_size=patch_size,
        num_attention_heads=16,
        head_dim=hidden_size // 16,
        layer_norm_eps=1e-6,
        qk_normalization=True,
    )


def test_intern_vision_embeddings_init():
    """Test InternVisionEmbeddings initialization."""
    # For unit testing, we'll create a mock config object
    from dataclasses import dataclass, field

    @dataclass
    class MockVisionConfig:
        dtype: DType = DType.bfloat16
        hidden_size: int = 1024
        image_size: int = 448
        patch_size: int = 14

    @dataclass
    class MockConfig:
        vision_config: MockVisionConfig = field(
            default_factory=lambda: MockVisionConfig()
        )
        devices: list[DeviceRef] = field(
            default_factory=lambda: [DeviceRef.CPU()]
        )

    config = MockConfig()
    embeddings = InternVisionEmbeddings(config, DeviceRef.CPU())  # type: ignore[arg-type]

    # Check attributes are set correctly
    assert embeddings.embed_dim == config.vision_config.hidden_size
    assert embeddings.image_size == config.vision_config.image_size
    assert embeddings.patch_size == config.vision_config.patch_size
    assert (
        embeddings.num_patches
        == (config.vision_config.image_size // config.vision_config.patch_size)
        ** 2
    )
    assert (
        embeddings.num_positions == embeddings.num_patches + 1
    )  # +1 for class token

    # Check components are initialized
    assert embeddings.patch_embedding is not None
    assert hasattr(embeddings, "class_embedding")
    assert hasattr(embeddings, "position_embedding")


@pytest.mark.parametrize(
    "image_size,patch_size,expected_patches",
    [
        (224, 16, 196),  # 14x14 patches
        (448, 14, 1024),  # 32x32 patches
        (224, 32, 49),  # 7x7 patches
    ],
)
def test_position_embedding_shape(image_size, patch_size, expected_patches):
    """Test position embedding shape calculations."""
    from dataclasses import dataclass, field

    @dataclass
    class MockVisionConfig:
        dtype: DType = DType.bfloat16
        hidden_size: int = 768
        image_size: int = 224
        patch_size: int = 16

    @dataclass
    class MockConfig:
        vision_config: MockVisionConfig = field(
            default_factory=MockVisionConfig
        )
        devices: list[DeviceRef] = field(
            default_factory=lambda: [DeviceRef.CPU()]
        )

    config = MockConfig()
    config.vision_config.image_size = image_size
    config.vision_config.patch_size = patch_size

    embeddings = InternVisionEmbeddings(config, DeviceRef.CPU())  # type: ignore[arg-type]

    assert embeddings.num_patches == expected_patches
    assert (
        embeddings.num_positions == expected_patches + 1
    )  # +1 for class token


@pytest.mark.parametrize("n_devices", [2, 4])
def test_intern_vision_embeddings_shard(n_devices):
    """Test InternVisionEmbeddings sharding across multiple GPUs."""
    if n_devices > accelerator_count():
        pytest.skip(f"Not enough GPUs to run test with {n_devices} GPUs.")

    from dataclasses import dataclass, field

    from max.graph import Graph, TensorType

    @dataclass
    class MockVisionConfig:
        dtype: DType = DType.bfloat16
        hidden_size: int = 1024
        image_size: int = 448
        patch_size: int = 14

    @dataclass
    class MockConfig:
        vision_config: MockVisionConfig = field(
            default_factory=lambda: MockVisionConfig()
        )
        devices: list[DeviceRef] = field(default_factory=list)

    # Create config with multiple GPU devices
    config = MockConfig()
    config.devices = [DeviceRef.GPU(i) for i in range(n_devices)]

    # Need Graph context to access weight shapes after sharding
    with Graph(
        "test_shard",
        input_types=[
            TensorType(
                config.vision_config.dtype,
                (
                    1,
                    config.vision_config.image_size,
                    config.vision_config.image_size,
                    3,
                ),
                device=DeviceRef.GPU(0),
            )
        ],
    ):
        # Create embeddings on first device
        embeddings = InternVisionEmbeddings(config, DeviceRef.GPU(0))  # type: ignore[arg-type]

        # Set sharding strategy before sharding
        from max.graph import ShardingStrategy

        embeddings.sharding_strategy = ShardingStrategy.replicate(
            num_devices=n_devices
        )

        # Test sharding for each device
        for shard_idx in range(n_devices):
            device = DeviceRef.GPU(shard_idx)
            sharded_embeddings = embeddings.shard(shard_idx, device)

            # Verify the shard method returns a valid InternVisionEmbeddings instance
            assert isinstance(sharded_embeddings, InternVisionEmbeddings)

            # The sharded version should maintain the same configuration
            assert (
                sharded_embeddings.embed_dim == config.vision_config.hidden_size
            )
            assert (
                sharded_embeddings.image_size == config.vision_config.image_size
            )
            assert (
                sharded_embeddings.patch_size == config.vision_config.patch_size
            )
            assert sharded_embeddings.num_patches == embeddings.num_patches
            assert sharded_embeddings.num_positions == embeddings.num_positions
