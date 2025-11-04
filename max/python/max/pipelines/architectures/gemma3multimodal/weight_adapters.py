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
    ".self_attn.out_proj.": ".self_attn.o_proj.",
    # Map attention weight names: checkpoint has "qkv" but model expects "qkv_proj"
    ".attn.qkv.": ".attn.qkv_proj.",
    ".attn.qkv_bias.": ".attn.qkv_proj_bias.",
    ".attn.proj.": ".attn.o_proj.",
    # Map mlp1 numbered layers to descriptive names
    "mlp1.0.": "mlp1.layer_norm.",  # Layer normalization
    "mlp1.1.": "mlp1.fc1.",  # First linear layer
    "mlp1.3.": "mlp1.fc2.",  # Second linear layer (note: it's 3, not 2)
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


# ⚠️ almost there, but some of the projection checkpoint weights aren't fitting
# ⚠️ potential confusing with the naming of some weights (eg encoder.layers, or just layers?)
def convert_safetensor_vision_state_dict(
    state_dict: dict[str, Weights],
    **unused_kwargs,
) -> dict[str, WeightData]:
    new_state_dict: dict[str, WeightData] = {}

    for weight_name, value in state_dict.items():
        if weight_name.startswith("vision_tower.vision_model."):
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

            # the *_proj self attn weights seem wonky
            # expecting 4096, 1152
            # but getting 1152, 1152
            # the former seems to only match the position_embedding weight
            # import re

            # match = re.search(
            #     r"encoder\.layers\.[0-9*]\.self_attn\..*_proj\.weight",
            #     weight_name,
            # )
            # if match:
            #     max_name = max_name.replace("encoder.", "")

            new_state_dict[max_name] = weight_data

    return new_state_dict
