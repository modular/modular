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

import re

from max.graph.weights import WeightData


def convert_safetensor_state_dict(
    state_dict: dict[str, WeightData],
) -> dict[str, WeightData]:
    keys = list(state_dict.keys())
    for key in keys:
        # Remap net.2 to net.1: Diffusers uses [GELU, Dropout, Linear], while MAX uses [GELU, Linear].
        if re.match(
            r"transformer_blocks\.\d+\.(ff|ff_context)\.net\.2\.(weight|bias)",
            key,
        ):
            state_dict[key.replace("net.2.", "net.1.")] = state_dict.pop(key)

        # Remap attention output projection:
        # Diffusers commonly represents `to_out` as a small Sequential/ModuleList like:
        #   to_out = [Linear(...), Dropout(...)]
        # producing weight names `to_out.0.weight` / `to_out.0.bias`.
        # In this MAX port, `to_out` is a single `Linear`, producing `to_out.weight` / `to_out.bias`.
        if re.match(
            r"transformer_blocks\.\d+\.attn\.to_out\.0\.(weight|bias)",
            key,
        ):
            state_dict[key.replace("to_out.0.", "to_out.")] = state_dict.pop(
                key
            )

        # Same pattern for the added/context stream output.
        if re.match(
            r"transformer_blocks\.\d+\.attn\.to_add_out\.0\.(weight|bias)",
            key,
        ):
            state_dict[key.replace("to_add_out.0.", "to_add_out.")] = (
                state_dict.pop(key)
            )
    return state_dict
