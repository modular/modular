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
from transformers import AutoConfig

# Maps from Safetensor to MAX weight names.
GEMMA3_SAFETENSOR_MAP: dict[str, str] = {
    "language_model.model.": "language_model.",
}


def _apply_name_mappings(name: str) -> str:
    """Apply all name mappings to a given name."""
    for before, after in GEMMA3_SAFETENSOR_MAP.items():
        name = name.replace(before, after)
    return name


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig,
    **unused_kwargs,
) -> dict[str, WeightData]:
    new_state_dict: dict[str, WeightData] = {}

    # Remap HuggingFace -> MAX-style names
    for weight_name, value in state_dict.items():
        max_name = _apply_name_mappings(weight_name)
        new_state_dict[max_name] = value.data()

    # For quantized model, we apply the same name re-mapping to the `ignore` list
    hf_quant_config = getattr(huggingface_config, "quantization_config", None)
    if hf_quant_config and "ignore" in hf_quant_config:
        hf_quant_config["ignore"] = [
            _apply_name_mappings(module_name)
            for module_name in hf_quant_config["ignore"]
        ]

    return new_state_dict
