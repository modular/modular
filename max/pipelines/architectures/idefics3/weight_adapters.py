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

# Maps from Idefics3 checkpoint names to Idefics3LanguageModel weight names.
IDEFICS3_LANGUAGE_MODEL_MAPPING = {
    "model.text_model.": "",
}

# Maps from Idefics3 checkpoint names to Idefics3VisionModel weight names.
IDEFICS3_VISION_MODEL_MAPPING = {
    "model.vision_model.": "",
    "model.connector.": "connector.",
}


def convert_idefics3_language_model_state_dict(
    state_dict: dict[str, Weights], **unused_kwargs
) -> dict[str, WeightData]:
    """Convert Idefics3 language model weights.

    Idefics3 checkpoints have language model weights prefixed with
    `language_model.`, but Idefics3LanguageModel expects the standard Llama3
    naming without this prefix.

    This adapter:
    1. Filters to only include language model weights (those with
       `language_model.` prefix).
    2. Strips the `language_model.model.` prefix to match DistributedLlama3
       expectations.
    3. Excludes vision model and multimodal projection weights.

    Args:
        state_dict: The raw Idefics3 checkpoint weights.
        huggingface_config: The Idefics3 HuggingFace configuration.
        pipeline_config: The pipeline configuration.

    Returns:
        The filtered and mapped weights for DistributedLlama3.
    """
    llm_state_dict: dict[str, WeightData] = {}

    for checkpoint_name, weight in state_dict.items():
        # Only process language model weights
        if checkpoint_name.startswith("lm_head."):
            llm_state_dict[checkpoint_name] = weight.data()

        elif checkpoint_name.startswith("model.text_model."):
            # Apply mapping to strip "language_model." prefixes
            llm_name = checkpoint_name
            for before, after in IDEFICS3_LANGUAGE_MODEL_MAPPING.items():
                llm_name = llm_name.replace(before, after)

            llm_state_dict[llm_name] = weight.data()

    return llm_state_dict


def convert_idefics3_vision_model_state_dict(
    state_dict: dict[str, Weights], **unused_kwargs
) -> dict[str, WeightData]:
    """Convert Idefics3 vision model weights for Idefics3VisionModel.

    Args:
        state_dict: The raw Idefics3 checkpoint weights.
        huggingface_config: The Idefics3 HuggingFace configuration.
        pipeline_config: The pipeline configuration.

    Returns:
        The filtered and mapped weights for Idefics3VisionModel.
    """
    vision_model_state_dict: dict[str, WeightData] = {}

    for checkpoint_name, weight in state_dict.items():
        if checkpoint_name.startswith("model.connector."):
            vision_model_name = checkpoint_name
            for before, after in IDEFICS3_VISION_MODEL_MAPPING.items():
                vision_model_name = vision_model_name.replace(before, after)
            vision_model_state_dict[vision_model_name] = weight.data()

        elif checkpoint_name.startswith("model.vision_model."):
            vision_model_name = checkpoint_name
            for before, after in IDEFICS3_VISION_MODEL_MAPPING.items():
                vision_model_name = vision_model_name.replace(before, after)

            weight_data = weight.data()

            # Handle patch embedding weight shape conversion
            # PyTorch: [out_channels, in_channels, height, width] -> [height, width, in_channels, out_channels]
            if vision_model_name.endswith("embeddings.patch_embedding.weight"):
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

            vision_model_state_dict[vision_model_name] = weight_data

    return vision_model_state_dict
