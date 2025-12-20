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

"""Weight generation utilities for Qwen3VL tests."""

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
PATCH_MERGER_STD = 0.012
PATCH_EMBED_STD = 0.02

# Layer Norm
LAYER_NORM_MEAN = 0.95
LAYER_NORM_STD = 0.02

# ==================================================


class WeightGenerator:
    """Generates test weights for Qwen3VL components."""

    def __init__(self, config_name: ConfigNames):
        """Initialize with config name."""
        # Set deterministic seed for reproducible weights
        torch.manual_seed(42)

        self.config_name = config_name
        self.loader = get_config_loader()
        self.config = self.loader.load_config(config_name)
        self.vision_config = self.config["vision_config"]

    def generate_position_embedding_weights(self) -> dict[str, torch.Tensor]:
        hidden_size = self.vision_config["hidden_size"]
        num_position_embeddings = self.vision_config["num_position_embeddings"]
        weights = {}
        weights["pos_embed.weight"] = PATCH_EMBED_STD * torch.randn(
            num_position_embeddings,
            hidden_size,
        ).to(torch.bfloat16)
        return weights

    def generate_vision_patch_merger_weights(
        self, use_postshuffle_norm: bool = False
    ) -> dict[str, torch.Tensor]:
        """Generate vision patch merger weights for Qwen3VL."""
        hidden_size = self.vision_config["hidden_size"]
        out_hidden_size = self.vision_config["out_hidden_size"]
        spatial_merge_size = self.vision_config["spatial_merge_size"]
        input_dim = hidden_size * (spatial_merge_size**2)

        weights = {}

        # Generate norm weights
        if use_postshuffle_norm:
            weights["norm.weight"] = (
                LAYER_NORM_STD * torch.randn(input_dim).to(torch.bfloat16)
                + LAYER_NORM_MEAN
            )
            weights["norm.bias"] = (
                LAYER_NORM_STD * torch.randn(input_dim).to(torch.bfloat16)
                + LAYER_NORM_MEAN
            )
        else:
            weights["norm.weight"] = (
                LAYER_NORM_STD * torch.randn(hidden_size).to(torch.bfloat16)
                + LAYER_NORM_MEAN
            )
            weights["norm.bias"] = (
                LAYER_NORM_STD * torch.randn(hidden_size).to(torch.bfloat16)
                + LAYER_NORM_MEAN
            )

        # Generate MLP weights
        weights["linear_fc1.weight"] = PATCH_MERGER_STD * torch.randn(
            input_dim, input_dim
        ).to(torch.bfloat16)
        weights["linear_fc1.bias"] = PATCH_MERGER_STD * torch.randn(
            input_dim
        ).to(torch.bfloat16)
        weights["linear_fc2.weight"] = PATCH_MERGER_STD * torch.randn(
            out_hidden_size, input_dim
        ).to(torch.bfloat16)
        weights["linear_fc2.bias"] = PATCH_MERGER_STD * torch.randn(
            out_hidden_size
        ).to(torch.bfloat16)

        return weights

    def generate_moe_weights(self) -> dict[str, torch.Tensor]:
        """Generate MoE weights for Qwen3VL-MoE."""
        text_config = self.config["text_config"]
        hidden_size = text_config["hidden_size"]
        num_experts = text_config["num_experts"]
        moe_intermediate_size = text_config["moe_intermediate_size"]

        # Standard deviation for weight initialization
        GATE_UP_STD = 0.02
        DOWN_STD = 0.02
        ROUTER_STD = 0.02

        weights = {
            # Router weights: [num_experts, hidden_size]
            "gate.gate_score.weight": (
                torch.randn(num_experts, hidden_size, dtype=torch.bfloat16)
                * ROUTER_STD
            ),
            # Expert weights: [num_experts, hidden_size, 2 * moe_intermediate_size]
            # Format: gate and up projections concatenated
            "experts.gate_up_proj": (
                torch.randn(
                    num_experts,
                    hidden_size,
                    2 * moe_intermediate_size,
                    dtype=torch.bfloat16,
                )
                * GATE_UP_STD
            ),
            # Down projection: [num_experts, moe_intermediate_size, hidden_size]
            "experts.down_proj": (
                torch.randn(
                    num_experts,
                    moe_intermediate_size,
                    hidden_size,
                    dtype=torch.bfloat16,
                )
                * DOWN_STD
            ),
        }
        return weights


def get_weight_generator(config_name: ConfigNames) -> WeightGenerator:
    """Get a weight generator instance."""
    return WeightGenerator(config_name)
