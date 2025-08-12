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

# Maps from Idefics3 checkpoint names to Idefics3LanguageModel weight names.
QWEN2_5_VL_MODEL_MAPPING = {
    "model.": "language_model.",
    "visual.": "vision_encoder.",
}


def convert_qwen2_5vl_model_state_dict(
    state_dict: dict[str, Weights], **unused_kwargs
) -> dict[str, WeightData]:
    """Convert Qwen2.5VL model weights.

    Qwen2.5VL checkpoints have language model weights prefixed with
    `language_model.`, but Qwen2.5VLLanguageModel expects the standard Llama3
    naming without this prefix.

    This adapter:
    1. Filters to only include language model weights (those with
       `language_model.` prefix).
    2. Strips the `language_model.model.` prefix to match Qwen2.5VLLanguageModel
       expectations.
    3. Excludes vision model and multimodal projection weights.

    Args:
        state_dict: The raw Qwen2.5VL checkpoint weights.
        huggingface_config: The Qwen2.5VL HuggingFace configuration.
        pipeline_config: The pipeline configuration.

    Returns:
        The filtered and mapped weights for Qwen2.5VLLanguageModel.
    """
    llm_state_dict: dict[str, WeightData] = {}

    for checkpoint_name, weight in state_dict.items():
        # Only process language model weights
        if checkpoint_name.startswith("lm_head."):
            llm_state_dict["language_model.lm_head.weight"] = weight.data()

        else:
            # Apply mapping to strip "language_model." prefixes
            llm_name = checkpoint_name
            for before, after in QWEN2_5_VL_MODEL_MAPPING.items():
                llm_name = llm_name.replace(before, after)

            llm_state_dict[llm_name] = weight.data()

    return llm_state_dict
