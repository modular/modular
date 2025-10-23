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

from collections.abc import Mapping

from max.graph.weights import WeightData, Weights

# Maps from Safetensor to MAX weight names.
WHISPER_ENCODER_SAFETENSOR_MAP: dict[str, str | None] = {
    "encoder.layer_norm.": "encoder.norm.",
    "encoder.": "",
    "decoder.": None,
    ".q_proj": ".wq",
    ".k_proj": ".wk",
    ".v_proj": ".wv",
    ".out_proj": ".wo",
    ".fc.": ".mlp.",
    ".self_attn.": ".attention.",
    ".self_attn_layer_norm.": ".attention_norm.",
    ".final_layer_norm.": ".mlp_norm.",
}


def convert_safetensor_state_dict_encoder(
    state_dict: Mapping[str, Weights],
) -> dict[str, WeightData]:
    new_state_dict: dict[str, WeightData] = {}
    value: Weights | None
    for weight_name, value in state_dict.items():
        max_name = weight_name
        for before, after in WHISPER_ENCODER_SAFETENSOR_MAP.items():
            if after is None:
                if before in max_name:
                    value = None
                    break
            else:
                max_name = max_name.replace(before, after)
        if value is not None:
            new_state_dict[max_name] = value.data()
    return new_state_dict
