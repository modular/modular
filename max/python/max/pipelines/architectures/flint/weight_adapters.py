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

FLINT_SAFETENSOR_MAP: dict[str, str] = {
    # Checkpoint root to module root.
    "model.embed_tokens.": "language_model.embed_tokens.",
    "model.norm.": "language_model.norm.",
    "model.layers.": "language_model.layers.",
    "lm_head.": "language_model.lm_head.",
    # Leaves the module tree names differently.
    "self_attn.softmax_offset": "self_attn.sinks",
    "self_attn.q_norm.weight": "self_attn.q_norm_weight",
    "self_attn.k_norm.weight": "self_attn.k_norm_weight",
    "mlp.gate.weight": "mlp.gate.gate_score.weight",
    "mlp.shared_expert.": "mlp.shared_experts.",
}


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    **unused_kwargs: object,
) -> dict[str, WeightData]:
    converted: dict[str, WeightData] = {}
    for weight_name, value in state_dict.items():
        max_name = weight_name
        for before, after in FLINT_SAFETENSOR_MAP.items():
            max_name = max_name.replace(before, after)
        converted[max_name] = value.data()
    return converted
