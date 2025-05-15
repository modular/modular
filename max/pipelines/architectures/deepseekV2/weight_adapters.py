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

import numpy as np
from max.dtype import DType
from max.graph import Shape
from max.graph.weights import WeightData, Weights

NUM_LAYERS = 27
NUM_EXPERTS = 64
MOE_INTERMEDIATE_SIZE = 1408
HIDDEN_SIZE = 2048


# Maps from Safetensor to MAX weight names.
DEEPSEEK_SAFETENSOR_MAP = {
    "model.": "",  # Removes the "model" prefix.
    "shared_experts.": "shared_expert_",
    "gate.": "gate.gate_score.",
}


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights], **unused_kwargs
) -> dict[str, WeightData]:
    new_state_dict: dict[str, WeightData] = {}

    # TODO(GEX-2124): This concatenation of expert weights is a workaround for a
    # compilation bug and should be removed once the bug is fixed.
    for i in range(NUM_LAYERS):
        if (
            f"model.layers.{i}.mlp.experts.0.down_proj.weight"
            in state_dict.keys()
        ):  # if current layer uses MoE
            concat_down_proj = []
            concat_up_proj = []
            concat_gate_proj = []
            for j in range(NUM_EXPERTS):
                concat_down_proj.append(
                    state_dict[
                        f"model.layers.{i}.mlp.experts.{j}.down_proj.weight"
                    ].raw_tensor()
                )
                concat_up_proj.append(
                    state_dict[
                        f"model.layers.{i}.mlp.experts.{j}.up_proj.weight"
                    ].raw_tensor()
                )
                concat_gate_proj.append(
                    state_dict[
                        f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight"
                    ].raw_tensor()
                )

            new_state_dict[f"layers.{i}.mlp.experts.down_proj.weight"] = (
                WeightData(
                    np.stack(concat_down_proj, axis=0),
                    name=f"model.layers.{i}.mlp.experts.down_proj.weight",
                    dtype=DType.bfloat16,
                    shape=Shape(
                        [NUM_EXPERTS, HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE]
                    ),
                )
            )
            new_state_dict[f"layers.{i}.mlp.experts.up_proj.weight"] = (
                WeightData(
                    np.stack(concat_up_proj, axis=0),
                    name=f"model.layers.{i}.mlp.experts.up_proj.weight",
                    dtype=DType.bfloat16,
                    shape=Shape(
                        [NUM_EXPERTS, MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE]
                    ),
                )
            )
            new_state_dict[f"layers.{i}.mlp.experts.gate_proj.weight"] = (
                WeightData(
                    np.stack(concat_gate_proj, axis=0),
                    name=f"model.layers.{i}.mlp.experts.gate_proj.weight",
                    dtype=DType.bfloat16,
                    shape=Shape(
                        [NUM_EXPERTS, MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE]
                    ),
                )
            )

    # Map the weight names.
    for name, value in state_dict.items():
        if ".experts." not in name:
            max_name = name
            for before, after in DEEPSEEK_SAFETENSOR_MAP.items():
                max_name = max_name.replace(before, after)
            new_state_dict[max_name] = value.data()

    return new_state_dict
