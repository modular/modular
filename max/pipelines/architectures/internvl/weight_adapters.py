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

# Maps from InternVL checkpoint names to DistributedLlama3 weight names.
INTERNVL_LANGUAGE_MODEL_MAPPING = {
    # Strip the "language_model.model." prefix from most weights
    # InternVL checkpoint: "language_model.model.layers.0.input_layernorm.weight"
    # DistributedLlama3 expects: "layers.0.input_layernorm.weight"
    "language_model.model.": "",
    # Handle the output layer separately
    # InternVL checkpoint: "language_model.lm_head.weight"
    # DistributedLlama3 expects: "lm_head.weight"
    "language_model.lm_head.": "lm_head.",
}


def convert_internvl_language_model_state_dict(
    state_dict: dict[str, Weights], **unused_kwargs
) -> dict[str, WeightData]:
    """Convert InternVL language model weights for DistributedLlama3.

    InternVL checkpoints have language model weights prefixed with
    `language_model.`, but DistributedLlama3 expects the standard Llama3
    naming without this prefix.

    This adapter:
    1. Filters to only include language model weights (those with
       `language_model.` prefix).
    2. Strips the `language_model.model.` prefix to match DistributedLlama3
       expectations.
    3. Excludes vision model and multimodal projection weights.

    Args:
        state_dict: The raw InternVL checkpoint weights.
        huggingface_config: The InternVL HuggingFace configuration.
        pipeline_config: The pipeline configuration.

    Returns:
        The filtered and mapped weights for DistributedLlama3.
    """
    llm_state_dict: dict[str, WeightData] = {}

    for checkpoint_name, weight in state_dict.items():
        # Only process language model weights (skip vision model and mlp1)
        if checkpoint_name.startswith("language_model."):
            # Apply mapping to strip "language_model." prefixes
            llm_name = checkpoint_name
            for before, after in INTERNVL_LANGUAGE_MODEL_MAPPING.items():
                llm_name = llm_name.replace(before, after)

            llm_state_dict[llm_name] = weight.data()

    return llm_state_dict
