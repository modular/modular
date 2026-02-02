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
"""Weight adapters for DeepseekV3 NextN model.
The format for the safetensor file is not canonicalized. We use
https://huggingface.co/SGLang/DeepSeek-V3-NextN as a reference.
"""

from __future__ import annotations

from max.graph.weights import WeightData, Weights
from transformers.configuration_utils import PretrainedConfig

DEEPSEEK_NEXTN_SAFETENSOR_MAP = {
    "model.": "",  # Removes the "model" prefix
    "layers.0.shared_head.norm.": "shared_head_norm.",  # Shared head norm
    "layers.0.shared_head.head.": "lm_head.",  # LM head
    "layers.0.enorm.": "enorm.",  # Embedding normalization
    "layers.0.hnorm.": "hnorm.",  # Hidden state normalization
    "layers.0.eh_proj.": "eh_proj.",  # Embedding-hidden projection
    "layers.0.embed_tokens.": "embed_tokens.",  # Embeddings
    "layers.0.": "decoder_layer.",  # All other layer 0 weights go to decoder_layer
    "gate.weight": "gate.gate_score.weight",  # MoE gate mapping
    "weight_scale_inv": "weight_scale",  # FP8 weight scale mapping
}


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: PretrainedConfig,
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Convert safetensor weights to MAX format for NextN model.

    The NextN model uses only the NextN-specific layer at index 0 in the safetensor.
    This includes: enorm, hnorm, eh_proj, embed_tokens, decoder layer components,
    shared_head_norm, and lm_head.

    Args:
        state_dict: Dictionary of weights from safetensors
        huggingface_config: HuggingFace model configuration

    Returns:
        Dictionary of weights in MAX format
    """
    new_state_dict: dict[str, WeightData] = {}

    # Map the weight names using ordered replacements
    # Order matters: more specific patterns must come before general ones
    for name, value in state_dict.items():
        max_name = name
        for before, after in DEEPSEEK_NEXTN_SAFETENSOR_MAP.items():
            max_name = max_name.replace(before, after)
        new_state_dict[max_name] = value.data()

    return new_state_dict
