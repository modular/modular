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

from max.driver import Tensor
from max.dtype import DType
from max.graph.shape import Shape
from max.graph.weights import WeightData, Weights

GEMMA3_LANGUAGE_SAFETENSOR_MAP: dict[str, str] = {
    "language_model.model.": "",
}

# For the vision model
GEMMA3_VISION_SAFETENSOR_MAP: dict[str, str] = {
    "vision_tower.vision_model.": "",
    "multi_modal_": "",
}

# NOTE: Huggingface implementation seems to have quite different checkpoint name conversions:
# "^language_model.model": "model.language_model",
# "^vision_tower": "model.vision_tower",
# "^multi_modal_projector": "model.multi_modal_projector",
# "^language_model.lm_head": "lm_head",


def convert_safetensor_language_state_dict(
    state_dict: dict[str, Weights], **unused_kwargs
) -> dict[str, WeightData]:
    """Convert safetensor state dict to MAX format for the language model."""
    new_state_dict: dict[str, WeightData] = {}

    for weight_name, value in state_dict.items():
        if weight_name.startswith("language_model."):
            max_name = weight_name
            for before, after in GEMMA3_LANGUAGE_SAFETENSOR_MAP.items():
                max_name = max_name.replace(before, after)
            new_state_dict[max_name] = value.data()

    return new_state_dict


def convert_safetensor_vision_state_dict(
    state_dict: dict[str, Weights],
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Convert safetensor state dict to MAX format for the vision model.

    This function only processes weights that start with 'vision_tower.vision_model.'
    and strips that prefix to match the expected MAX naming convention.
    """
    new_state_dict: dict[str, WeightData] = {}

    for weight_name, value in state_dict.items():
        if not weight_name.startswith("vision_tower.vision_model."):
            if not weight_name.startswith("multi_modal_"):
                continue

        max_name = weight_name

        for before, after in GEMMA3_VISION_SAFETENSOR_MAP.items():
            max_name = max_name.replace(before, after)

        weight_data = value.data()

        # the patch embedding weight is wonky
        if weight_name.endswith("embeddings.patch_embedding.weight"):
            assert isinstance(weight_data.data, Tensor)
            if weight_data.dtype == DType.bfloat16:
                data = weight_data.data.view(DType.float16).to_numpy()
            else:
                data = weight_data.data.to_numpy()
            transposed_data = data.transpose(2, 3, 1, 0)
            # Ensure the array is contiguous in memory
            transposed_data = transposed_data.copy()
            weight_data = WeightData(
                data=transposed_data,
                name=weight_data.name,
                dtype=weight_data.dtype,
                shape=Shape(transposed_data.shape),
                quantization_encoding=weight_data.quantization_encoding,
            )

        new_state_dict[max_name] = weight_data

    return new_state_dict
