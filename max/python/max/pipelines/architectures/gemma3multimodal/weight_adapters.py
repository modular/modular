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

from __future__ import annotations

from max.graph.weights import WeightData, Weights

GEMMA3_SAFETENSOR_MAP: dict[str, str] = {
    "language_model.model.": "language_model.",
    "vision_tower.vision_model.": "",  # Strip this prefix for vision model
}


def _apply_name_mappings(name: str) -> str:
    """Apply all name mappings to a given name."""
    for before, after in GEMMA3_SAFETENSOR_MAP.items():
        name = name.replace(before, after)
    return name


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    **unused_kwargs,
) -> dict[str, WeightData]:
    new_state_dict: dict[str, WeightData] = {}

    for weight_name, value in state_dict.items():
        max_name = _apply_name_mappings(weight_name)
        new_state_dict[max_name] = value.data()

    return new_state_dict


def convert_gemma3_multimodal_vision_state_dict(
    state_dict: dict[str, Weights], **unused_kwargs
) -> dict[str, WeightData]:
    """Convert Gemma3 multimodal vision model weights for SigLipVisionModel.

    Gemma3 checkpoints have vision model weights prefixed with
    `vision_tower.vision_model.`, but SigLipVisionModel expects that prefix dropped.

    This adapter:
    1. Filters to only include vision model weights (those with
       `vision_tower.vision_model.` prefix).
    2. Strips the `vision_tower.vision_model.` prefix to match SigLipVisionModel
       expectations.
    3. Handles Conv2d weight transposition for patch embedding.

    Args:
        state_dict: The raw Gemma3 checkpoint weights.

    Returns:
        The filtered and mapped weights for SigLipVisionModel.
    """
    vision_model_state_dict: dict[str, WeightData] = {}

    for checkpoint_name, weight in state_dict.items():
        if not checkpoint_name.startswith("vision_tower.vision_model."):
            continue

        vision_model_name = checkpoint_name.replace(
            "vision_tower.vision_model.", ""
        )

        weight_data = weight.data()
        vision_model_state_dict[vision_model_name] = weight_data

    return vision_model_state_dict
