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

# Maps from Safetensor to MAX weight names.
GPT_OSS_SAFETENSOR_MAP: dict[str, str] = {
    "model.embed_tokens.": "language_model.embed_tokens.",
    "model.norm.": "language_model.norm.",
    "lm_head.": "language_model.lm_head.",
    "model.layers.": "language_model.layers.",
    # MoE weight mappings
    ".mlp.router": ".mlp.gate.gate_score",
    ".mlp.experts.gate_up_proj": ".mlp.experts.gate_up_proj",
    ".mlp.experts.down_proj": ".mlp.experts.down_proj",
    ".mlp.experts.gate_up_proj_bias": ".mlp.experts.gate_up_proj_bias",
    ".mlp.experts.down_proj_bias": ".mlp.experts.down_proj_bias",
    ".self_attn.sinks": ".self_attn.sinks",
}


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights], **kwargs
) -> dict[str, WeightData]:
    """Convert safetensor state dict to MAX format.

    Args:
        state_dict: Dictionary of weight tensors

    Returns:
        Dictionary of converted weight data
    """

    # Now remap all weight names from HuggingFace to MAX format
    new_state_dict: dict[str, WeightData] = {}

    for weight_name, value in state_dict.items():
        max_name: str = weight_name
        for before, after in GPT_OSS_SAFETENSOR_MAP.items():
            max_name = max_name.replace(before, after)
        new_state_dict[max_name] = value.data()

    return new_state_dict
