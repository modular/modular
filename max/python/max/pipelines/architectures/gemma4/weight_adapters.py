# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

GEMMA4_LANGUAGE_SAFETENSOR_MAP: dict[str, str] = {
    "model.language_model.": "",
    "language_model.model.": "",
}

# For the vision model
GEMMA4_MULTIMODAL_SAFETENSOR_MAP: dict[str, str] = {
    "model.vision_tower.": "",
    "model.embed_vision": "embed_vision",
    ".linear.": ".",
}


def convert_safetensor_language_state_dict(
    state_dict: dict[str, Weights],
) -> dict[str, WeightData]:
    """Convert safetensor state dict to MAX format for the language model."""
    new_state_dict: dict[str, WeightData] = {}

    for weight_name, value in state_dict.items():
        if weight_name.startswith("language_model.") or weight_name.startswith(
            "model.language_model."
        ):
            max_name = weight_name
            for before, after in GEMMA4_LANGUAGE_SAFETENSOR_MAP.items():
                max_name = max_name.replace(before, after)
            new_state_dict[max_name] = value.data()

    return new_state_dict


def convert_safetensor_vision_state_dict(
    state_dict: dict[str, Weights],
) -> dict[str, WeightData]:
    """Convert safetensor state dict to MAX format for the vision model."""
    new_state_dict: dict[str, WeightData] = {}

    # include vision tower weights AND multi modal weights
    for weight_name, value in state_dict.items():
        if not (
            weight_name.startswith("model.vision_tower.")
            or weight_name.startswith("model.embed_vision.")
        ):
            continue

        max_name = weight_name

        for before, after in GEMMA4_MULTIMODAL_SAFETENSOR_MAP.items():
            max_name = max_name.replace(before, after)

        weight_data = value.data()

        new_state_dict[max_name] = weight_data

    return new_state_dict
