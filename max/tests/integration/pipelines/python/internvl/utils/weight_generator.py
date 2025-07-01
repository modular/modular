# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Weight generation utilities for InternVL tests."""

import torch

from .config_loader import ConfigNames, get_config_loader

# ==================================================
# INTERNVL VISION MODEL WEIGHT STANDARD DEVIATIONS (from 2B variant)
# ==================================================
# Embeddings
CLS_TOKEN_STD = 0.02
PATCH_CONV_STD = 0.02
POS_EMBED_STD = 0.02
QKV_PROJ_STD = 0.0160
O_PROJ_STD = 0.0138
Q_NORM_STD = 0.2299
K_NORM_STD = 0.2299
GATE_PROJ_STD = 0.0150
DOWN_PROJ_STD = 0.0140
NORM_STD = 0.2299
LAYER_SCALE_STD = 0.2
# ==================================================


class WeightGenerator:
    """Generates test weights for InternVL components."""

    def __init__(self, config_name: ConfigNames):
        """Initialize with config name."""
        # Set deterministic seed for reproducible weights
        torch.manual_seed(42)

        self.config_name = config_name
        self.loader = get_config_loader()
        self.hf_config = self.loader.load_hf_vision_config(config_name)
        self.dimensions = self.loader.get_model_dimensions(config_name)

    def generate_vision_attention_weights(
        self, in_larger_module: bool = False
    ) -> dict[str, torch.Tensor]:
        """Generate vision attention weights with realistic values."""
        hidden_size = self.dimensions["vision_hidden_size"]

        # Load config to check bias and normalization settings
        qkv_bias = self.hf_config.qkv_bias
        qk_normalization = self.hf_config.qk_normalization
        o_proj_bias = True  # InternVL always has an output projection bias

        prefix = "attn." if in_larger_module else ""

        weights = {
            # Attention weights (stacked QKV)
            f"{prefix}qkv.weight": (
                torch.randn(3 * hidden_size, hidden_size) * QKV_PROJ_STD
            ).to(torch.bfloat16),
            f"{prefix}proj.weight": (
                torch.randn(hidden_size, hidden_size) * O_PROJ_STD
            ).to(torch.bfloat16),
        }

        # Only add normalization weights if qk_normalization is enabled
        if qk_normalization:
            weights.update(
                {
                    f"{prefix}q_norm.weight": (
                        torch.randn(hidden_size) * Q_NORM_STD
                    ).to(torch.bfloat16),
                    f"{prefix}k_norm.weight": (
                        torch.randn(hidden_size) * K_NORM_STD
                    ).to(torch.bfloat16),
                }
            )

        # Only add bias weights if the config specifies they should exist
        if qkv_bias:
            weights[f"{prefix}qkv.bias"] = torch.zeros(
                3 * hidden_size, dtype=torch.bfloat16
            )
        if o_proj_bias:
            weights[f"{prefix}proj.bias"] = torch.zeros(
                hidden_size, dtype=torch.bfloat16
            )

        return weights

    def generate_vision_encoder_layer_weights(self) -> dict[str, torch.Tensor]:
        """Generate encoder layer weights with realistic values."""
        hidden_size = self.dimensions["vision_hidden_size"]
        intermediate_size = self.dimensions["vision_intermediate_size"]
        norm_type = self.hf_config.norm_type

        # Use composition: get attention weights from dedicated method
        weights = self.generate_vision_attention_weights(in_larger_module=True)

        weights.update(
            {
                # MLP weights
                "mlp.fc1.weight": (
                    torch.randn(intermediate_size, hidden_size) * GATE_PROJ_STD
                ).to(torch.bfloat16),
                "mlp.fc1.bias": torch.zeros(
                    intermediate_size, dtype=torch.bfloat16
                ),
                "mlp.fc2.weight": (
                    torch.randn(hidden_size, intermediate_size) * DOWN_PROJ_STD
                ).to(torch.bfloat16),
                "mlp.fc2.bias": torch.zeros(hidden_size, dtype=torch.bfloat16),
                # Layer scale parameters (correct shape)
                "ls1": torch.randn(hidden_size, dtype=torch.bfloat16)
                * LAYER_SCALE_STD,
                "ls2": torch.randn(hidden_size, dtype=torch.bfloat16)
                * LAYER_SCALE_STD,
            }
        )

        if norm_type == "layer_norm":
            weights.update(
                {
                    "norm1.weight": (torch.randn(hidden_size) * NORM_STD).to(
                        torch.bfloat16
                    ),
                    "norm1.bias": torch.zeros(
                        hidden_size, dtype=torch.bfloat16
                    ),
                    "norm2.weight": (torch.randn(hidden_size) * NORM_STD).to(
                        torch.bfloat16
                    ),
                    "norm2.bias": torch.zeros(
                        hidden_size, dtype=torch.bfloat16
                    ),
                }
            )
        elif norm_type == "rms_norm":
            weights.update(
                {
                    "norm1.weight": (torch.randn(hidden_size) * NORM_STD).to(
                        torch.bfloat16
                    ),
                    "norm2.weight": (torch.randn(hidden_size) * NORM_STD).to(
                        torch.bfloat16
                    ),
                }
            )
        else:
            raise ValueError(f"Unsupported norm type: {norm_type}")

        return weights

    def generate_vision_embeddings_weights(self) -> dict[str, torch.Tensor]:
        """Generate vision embeddings weights with correct shapes."""
        hidden_size = self.dimensions["vision_hidden_size"]
        image_size = self.dimensions["vision_image_size"]
        patch_size = self.dimensions["vision_patch_size"]
        num_patches = (image_size // patch_size) ** 2

        # For HuggingFace, use Conv2D shape
        # Conv2D: (out_channels, in_channels, kernel_h, kernel_w)
        conv_weight = (
            torch.randn(
                hidden_size, 3, patch_size, patch_size, dtype=torch.bfloat16
            )
            * PATCH_CONV_STD
        )

        return {
            "class_embedding": (
                torch.randn(1, 1, hidden_size) * CLS_TOKEN_STD
            ).to(torch.bfloat16),
            "patch_embedding.weight": conv_weight,
            "patch_embedding.bias": torch.zeros(
                hidden_size, dtype=torch.bfloat16
            ),
            "position_embedding": (
                torch.randn(1, num_patches + 1, hidden_size) * POS_EMBED_STD
            ).to(torch.bfloat16),
        }

    def generate_vision_model_weights(self) -> dict[str, torch.Tensor]:
        """Generate complete vision model weights."""
        hidden_size = self.dimensions["vision_hidden_size"]
        num_layers = self.dimensions["vision_num_layers"]
        image_size = self.dimensions["vision_image_size"]
        patch_size = self.dimensions["vision_patch_size"]
        llm_hidden_size = self.dimensions["llm_hidden_size"]

        weights = {}

        # Embeddings weights
        num_patches = (image_size // patch_size) ** 2
        weights["vision_model.embeddings.class_embedding"] = (
            torch.randn(1, 1, hidden_size) * CLS_TOKEN_STD
        ).to(torch.bfloat16)

        # For HuggingFace reference, we use Conv2D weights
        weights["vision_model.embeddings.patch_embedding.weight"] = (
            torch.randn(hidden_size, 3, patch_size, patch_size) * PATCH_CONV_STD
        ).to(torch.bfloat16)
        weights["vision_model.embeddings.patch_embedding.bias"] = torch.zeros(
            hidden_size, dtype=torch.bfloat16
        )
        weights["vision_model.embeddings.position_embedding"] = (
            torch.randn(1, num_patches + 1, hidden_size) * POS_EMBED_STD
        ).to(torch.bfloat16)

        # Encoder layer weights - use composition with existing method
        encoder_layer_weights = self.generate_vision_encoder_layer_weights()
        for i in range(num_layers):
            # Add layer index prefix to each weight name
            for weight_name, weight_tensor in encoder_layer_weights.items():
                weights[f"vision_model.encoder.layers.{i}.{weight_name}"] = (
                    weight_tensor
                )

        # Multimodal projector weights
        # For InternVL, the projector input size is vision_hidden_size * (1/downsample_ratio)^2
        # With downsample_ratio=0.5, this gives us hidden_size * 4 = 1024 * 4 = 4096
        projector_input_size = hidden_size * 4  # vision_hidden_size * (1/0.5)^2

        # Generate weights with numbered format that HuggingFace model expects
        weights["mlp1.0.weight"] = (
            torch.randn(projector_input_size) * LAYER_SCALE_STD
        ).to(torch.bfloat16)
        weights["mlp1.0.bias"] = (
            torch.randn(projector_input_size) * LAYER_SCALE_STD
        ).to(torch.bfloat16)
        weights["mlp1.1.weight"] = (
            torch.randn(llm_hidden_size, projector_input_size) * LAYER_SCALE_STD
        ).to(torch.bfloat16)
        weights["mlp1.1.bias"] = (
            torch.randn(llm_hidden_size) * LAYER_SCALE_STD
        ).to(torch.bfloat16)
        weights["mlp1.3.weight"] = (
            torch.randn(llm_hidden_size, llm_hidden_size) * LAYER_SCALE_STD
        ).to(torch.bfloat16)
        weights["mlp1.3.bias"] = (
            torch.randn(llm_hidden_size) * LAYER_SCALE_STD
        ).to(torch.bfloat16)

        return weights


def get_weight_generator(config_name: ConfigNames) -> WeightGenerator:
    """Get a weight generator instance."""
    return WeightGenerator(config_name)
