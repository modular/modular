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

# Maps from Safetensor to MAX weight names.
# The new ModuleV3 model uses "language_model." prefix (matching GPT-OSS pattern).
PHI3_SAFETENSOR_MAPPING = {
    "model.": "language_model.",
}


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    **kwargs,
) -> dict[str, WeightData]:
    """Convert safetensor state dict to MAX format.

    Remaps HuggingFace weight names to the MAX naming convention
    (adds ``language_model.`` prefix).
    """
    new_state_dict: dict[str, WeightData] = {}
    for weight_name, value in state_dict.items():
        max_name: str = weight_name
        for before, after in PHI3_SAFETENSOR_MAPPING.items():
            max_name = max_name.replace(before, after)
        # HF safetensors may store lm_head.weight without a "model." prefix
        if not max_name.startswith("language_model."):
            max_name = "language_model." + max_name
        new_state_dict[max_name] = value.data()
    return new_state_dict
