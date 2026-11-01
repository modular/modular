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
"""Maps Qwen2 checkpoint names onto the ModuleV3 tree."""

from __future__ import annotations

from max.graph.weights import WeightData, Weights

QWEN2_SAFETENSOR_MAPPING = {
    "model.": "language_model.",
    "lm_head.": "language_model.lm_head.",
}


def max_weight_name(hf_name: str) -> str:
    """Returns the decoder's name for a Hugging Face weight name."""
    for before, after in QWEN2_SAFETENSOR_MAPPING.items():
        if hf_name.startswith(before):
            return after + hf_name.removeprefix(before)
    return hf_name


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights], **unused_kwargs
) -> dict[str, WeightData]:
    """Renames Hugging Face weights to the names the decoder reads."""
    return {
        max_weight_name(name): value.data()
        for name, value in state_dict.items()
    }
