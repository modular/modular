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
"""The Muse Spark DSpark drafter's weight names."""

from __future__ import annotations

from collections.abc import Mapping

from max.graph.weights import WeightData, Weights


def convert_draft_state_dict(
    state_dict: Mapping[str, Weights],
) -> dict[str, WeightData]:
    """Maps the drafter's checkpoint onto the drafter's module tree.

    The drafter scores the target's whole vocabulary, so its ``d2t`` maps
    every id to itself and is dropped once that is checked.

    Raises:
        ValueError: If ``d2t`` maps any id elsewhere.
    """
    out: dict[str, WeightData] = {}
    for name, weight in state_dict.items():
        data = weight.data()
        if name == "d2t":
            if data.to_buffer().to_numpy().any():
                raise ValueError(
                    "The DSpark draft's d2t maps draft ids to other target"
                    " ids, which a full-vocabulary draft never does."
                )
            continue
        name = name.replace(".self_attn.o_gate_proj.", ".self_attn.gate_proj.")
        out[name] = data
    return out
