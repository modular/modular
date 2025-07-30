# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Weight generation utilities for Qwen2.5VL tests."""

import torch

from .config_loader import ConfigNames, get_config_loader

# ==================================================
# QWEN2.5VL VISION MODEL WEIGHT STANDARD DEVIATIONS (default to 0.02 for now)
# ==================================================
# Patch Embedding
PATCH_CONV_STD = 0.0144

# Attention
QKV_PROJ_STD = 0.02
O_PROJ_STD = 0.02

# Patch Merger
PATCH_MERGER_STD = 0.02

# ==================================================


class WeightGenerator:
    """Generates test weights for Qwen2.5VL components."""

    def __init__(self, config_name: ConfigNames):
        """Initialize with config name."""
        # Set deterministic seed for reproducible weights
        torch.manual_seed(42)

        self.config_name = config_name
        self.loader = get_config_loader()
        self.config = self.loader.load_config(config_name)
        self.vision_config = self.config["vision_config"]

    def generate_vision_patch_embed_weights(self) -> dict[str, torch.Tensor]:
        """Generate vision patch embedding weights with Conv3D shape."""
        hidden_size = self.vision_config["hidden_size"]
        in_channels = self.vision_config.get("in_chans", 3)
        patch_size = self.vision_config["patch_size"]
        temporal_patch_size = self.vision_config.get("temporal_patch_size", 2)

        # Conv3D weight shape: (out_channels, in_channels, depth, height, width)
        conv_weight = (
            torch.randn(
                hidden_size,
                in_channels,
                temporal_patch_size,
                patch_size,
                patch_size,
                dtype=torch.bfloat16,
            )
            * PATCH_CONV_STD
        )

        return {
            "proj.weight": conv_weight,
            # No bias for Qwen2.5VL patch embedding
        }

    def generate_vision_attention_weights(self) -> dict[str, torch.Tensor]:
        """Generate vision attention weights for Qwen2.5VL."""
        hidden_size = self.vision_config["hidden_size"]

        weights = {
            # QKV projection (combined) with bias
            "qkv.weight": (
                torch.randn(3 * hidden_size, hidden_size) * QKV_PROJ_STD
            ).to(torch.bfloat16),
            "qkv.bias": torch.zeros(3 * hidden_size, dtype=torch.bfloat16),
            # Output projection with bias
            "proj.weight": (
                torch.randn(hidden_size, hidden_size) * O_PROJ_STD
            ).to(torch.bfloat16),
            "proj.bias": torch.zeros(hidden_size, dtype=torch.bfloat16),
        }

        return weights

    def generate_vision_patch_merger_weights(self) -> dict[str, torch.Tensor]:
        """Generate vision patch merger weights for Qwen2.5VL."""
        hidden_size = self.vision_config["hidden_size"]
        out_hidden_size = self.vision_config["out_hidden_size"]
        spatial_merge_size = self.vision_config["spatial_merge_size"]

        input_dim = hidden_size * (spatial_merge_size**2)

        weights = {}

        # RMSNorm layer weight
        weights["norm.weight"] = torch.ones(hidden_size).to(torch.bfloat16)

        # First linear layer weights (input_dim -> input_dim)
        weights["mlp.0.weight"] = PATCH_MERGER_STD * torch.randn(
            input_dim,
            input_dim,
        ).to(torch.bfloat16)

        # Second linear layer weights (input_dim -> out_hidden_size)
        weights["mlp.2.weight"] = PATCH_MERGER_STD * torch.randn(
            out_hidden_size,
            input_dim,
        ).to(torch.bfloat16)

        weights["mlp.0.bias"] = PATCH_MERGER_STD * torch.randn(
            input_dim,
        ).to(torch.bfloat16)
        weights["mlp.2.bias"] = PATCH_MERGER_STD * torch.randn(
            out_hidden_size,
        ).to(torch.bfloat16)

        return weights


def get_weight_generator(config_name: ConfigNames) -> WeightGenerator:
    """Get a weight generator instance."""
    return WeightGenerator(config_name)
