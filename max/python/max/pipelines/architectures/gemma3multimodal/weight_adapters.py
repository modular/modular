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
from max.graph.weights import WeightData, Weights

# Maps from Safetensor to MAX weight names.
# For the language model, we keep the original structure without the language_model prefix
# because the Gemma3TextModel expects weights like "embed_tokens.weight", not "language_model.embed_tokens.weight"
GEMMA3_LANGUAGE_SAFETENSOR_MAP: dict[str, str] = {
    "language_model.model.": "",
}

# For the vision model
GEMMA3_VISION_SAFETENSOR_MAP: dict[str, str] = {
    "vision_tower.vision_model.": "",
    # Map attention output projection: HF uses "out_proj", MAX uses "o_proj"
    ".self_attn.out_proj.": ".self_attn.o_proj.",
    # Position embedding is a Weight, not a parameter with .weight suffix
    "embeddings.position_embedding.weight": "embeddings.position_embedding",
    # Note: encoder layer MLPs use fc1/fc2 (simple 2-layer MLP), no mapping needed
}

# NOTE: Huggingface implementation seems to have quite different checkpoint name conversions:
# "^language_model.model": "model.language_model",
# "^vision_tower": "model.vision_tower",
# "^multi_modal_projector": "model.multi_modal_projector",
# "^language_model.lm_head": "lm_head",

def convert_safetensor_language_state_dict(
    state_dict: dict[str, Weights],
    **unused_kwargs
) -> dict[str, WeightData]:
    """Convert safetensor state dict to MAX format for the language model."""
    new_state_dict: dict[str, WeightData] = {}

    # Remap HuggingFace -> MAX-style names for language model
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

    Handles mapping from HuggingFace SigLIP/Gemma3 vision model weight names
    to MAX engine expected names.
    """
    new_state_dict: dict[str, WeightData] = {}

    # Remap HuggingFace -> MAX-style names
    for weight_name, value in state_dict.items():
        if weight_name.startswith("vision_tower.vision_model."):
            max_name = weight_name

            # Apply all mappings from the map
            for before, after in GEMMA3_VISION_SAFETENSOR_MAP.items():
                max_name = max_name.replace(before, after)

            weight_data = value.data()

            # Special handling for patch embedding: Conv2d -> Linear conversion
            # Conv2d weights have shape [out_channels, in_channels, kernel_h, kernel_w]
            # Linear weights need shape [out_channels, in_channels * kernel_h * kernel_w]
            if max_name == "embeddings.patch_embedding.weight":
                # Convert WeightData to Tensor for reshaping
                weight_tensor = Tensor.from_dlpack(weight_data.data)

                # Get original Conv2d shape: [out_ch, in_ch, k_h, k_w]
                out_ch, in_ch, k_h, k_w = weight_tensor.shape

                # Reshape to Linear format: [out_ch, in_ch * k_h * k_w]
                weight_tensor = weight_tensor.view(
                    weight_tensor.dtype,
                    (out_ch, in_ch * k_h * k_w)
                )

                # Recreate WeightData with the new shape
                weight_data = WeightData(
                    data=weight_tensor,
                    name=weight_data.name,
                    dtype=weight_data.dtype,
                    shape=weight_data.shape.__class__(weight_tensor.shape),
                    quantization_encoding=weight_data.quantization_encoding,
                )

            # Special handling for position embedding: add batch dimension
            # HF checkpoint has shape [num_patches, hidden_size]
            # MAX expects shape [1, num_patches, hidden_size]
            elif max_name == "embeddings.position_embedding":
                # Use Tensor.view to add batch dimension without going through numpy
                weight_tensor = Tensor.from_dlpack(weight_data.data)

                # Get the original shape: [num_patches, hidden_size]
                orig_shape = weight_tensor.shape

                # Create new shape with batch dimension: [1, num_patches, hidden_size]
                new_shape = (1,) + tuple(orig_shape)

                # Use view to reshape (no data copy needed)
                weight_tensor = weight_tensor.view(weight_tensor.dtype, new_shape)

                # Recreate WeightData with the new shape
                weight_data = WeightData(
                    data=weight_tensor,
                    name=weight_data.name,
                    dtype=weight_data.dtype,
                    shape=weight_data.shape.__class__(weight_tensor.shape),
                    quantization_encoding=weight_data.quantization_encoding,
                )

            new_state_dict[max_name] = weight_data

    return new_state_dict