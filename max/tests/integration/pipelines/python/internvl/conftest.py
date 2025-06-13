# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest
import torch
from max.dtype import DType
from max.pipelines.architectures.internvl.model_config import VisionConfig

"""
Fixtures for InternVL tests, including config and dummy weights.
"""


@pytest.fixture
def vision_config() -> VisionConfig:
    """Create a default vision configuration for testing."""
    return VisionConfig(
        dtype=DType.bfloat16,
        hidden_size=768,
        intermediate_size=3072,
        norm_type="layer_norm",
        image_size=448,
        patch_size=14,
        num_attention_heads=12,
        head_dim=64,
        layer_norm_eps=1e-6,
        qk_normalization=True,
    )


@pytest.fixture
def embeddings_weights(vision_config: VisionConfig) -> dict[str, torch.Tensor]:
    """
    Create dummy weights for InternVisionEmbeddings based on real model statistics.

    Weight initialization based on real InternVL model statistics:
    ┌─────────────────────┬──────────┬───────────────┬───────────────┐
    │ Weight              │ Std Dev  │ Range Min     │ Range Max     │
    ├─────────────────────┼──────────┼───────────────┼───────────────┤
    │ patch_embedding     │ 0.0134   │ -0.1328       │ 0.1162        │
    │ patch_embedding.bias│ 0.0078   │ -0.0317       │ 0.0264        │
    │ class_embedding     │ 0.1494   │ -1.8594       │ 0.8320        │
    │ position_embedding  │ 0.0177   │ -0.1318       │ 0.1387        │
    └─────────────────────┴──────────┴───────────────┴───────────────┘
    """
    torch.manual_seed(42)  # For reproducibility

    num_patches = (vision_config.image_size // vision_config.patch_size) ** 2
    num_positions = num_patches + 1

    return {
        "patch_embedding.weight": torch.randn(
            vision_config.hidden_size,
            3,
            vision_config.patch_size,
            vision_config.patch_size,
            dtype=torch.bfloat16,
        )
        * 0.0134,  # Real model std: 0.0134
        "patch_embedding.bias": torch.randn(
            vision_config.hidden_size, dtype=torch.bfloat16
        )
        * 0.0078,  # Real model std: 0.0078
        "class_embedding": torch.randn(
            1, 1, vision_config.hidden_size, dtype=torch.bfloat16
        )
        * 0.1494,  # Real model std: 0.1494
        "position_embedding": torch.randn(
            1, num_positions, vision_config.hidden_size, dtype=torch.bfloat16
        )
        * 0.0177,  # Real model std: 0.0177
    }


@pytest.fixture
def pixel_values(
    batch_size: int = 1, image_size: int = 448, seed: int = 42
) -> torch.Tensor:
    """
    Create dummy pixel values for testing.

    Pixel values after ImageNet normalization typically have std ~1.04
    """
    torch.manual_seed(seed)
    return torch.randn(
        batch_size, 3, image_size, image_size, dtype=torch.bfloat16
    ).to("cuda")
