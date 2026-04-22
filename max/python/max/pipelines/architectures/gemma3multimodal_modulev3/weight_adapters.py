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

# V3 language model mapping: compiled from root Gemma3LanguageModel,
# which has self.language_model = Gemma3TextModel(...).
# HF "language_model.model.X" → V3 "language_model.X"
# HF "language_model.lm_head.X" → V3 "language_model.lm_head.X" (unchanged)
GEMMA3_LANGUAGE_SAFETENSOR_MAP: dict[str, str] = {
    "language_model.model.": "language_model.",
    # HuggingFace uses separate q/k/v_proj; StackedLinear expects qkv_proj.{q,k,v}.
    "self_attn.q_proj.": "self_attn.qkv_proj.q.",
    "self_attn.k_proj.": "self_attn.qkv_proj.k.",
    "self_attn.v_proj.": "self_attn.qkv_proj.v.",
}

# V3 vision model mapping: compiled from root Gemma3VisionModel,
# which has self.embeddings, self.encoder, self.post_layernorm, self.projector.
GEMMA3_VISION_SAFETENSOR_MAP: dict[str, str] = {
    "vision_tower.vision_model.": "",
    "multi_modal_": "",
}


def convert_safetensor_language_state_dict(
    state_dict: dict[str, Weights],
) -> dict[str, WeightData]:
    """Convert safetensor state dict to MAX format for the V3 language model."""
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
) -> dict[str, WeightData]:
    """Convert safetensor state dict to MAX format for the V3 vision model."""
    new_state_dict: dict[str, WeightData] = {}

    for weight_name, value in state_dict.items():
        if not weight_name.startswith(
            "vision_tower.vision_model."
        ) and not weight_name.startswith("multi_modal_"):
            continue

        max_name = weight_name
        for before, after in GEMMA3_VISION_SAFETENSOR_MAP.items():
            max_name = max_name.replace(before, after)

        new_state_dict[max_name] = value.data()

    return new_state_dict
