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

import dataclasses

from max.driver import Buffer
from max.dtype import DType
from max.graph.type import Shape
from max.graph.weights import WeightData, Weights

GEMMA4_LANGUAGE_SAFETENSOR_MAP: dict[str, str] = {
    "model.language_model.": "",
    "language_model.model.": "",
    "router.proj.weight": "moe_block.gate.gate_score.weight",
    "router.scale": "moe_block.gate.scale",
    "router.per_expert_scale": "moe_block.gate.per_expert_scale",
    "pre_feedforward_layernorm_2.weight": "moe_block.pre_expert_norm.weight",
    "experts.": "moe_block.experts.",
}

GEMMA4_VISION_SAFETENSOR_MAP: dict[str, str] = {
    "model.vision_tower.": "",
    "model.embed_vision": "embed_vision",
    ".linear.": ".",
}


def convert_safetensor_language_state_dict(
    state_dict: dict[str, Weights],
) -> dict[str, WeightData]:
    """Convert safetensor state dict to MAX format for the language model."""
    new_state_dict: dict[str, WeightData] = {}

    for weight_name, value in state_dict.items():
        if not (
            weight_name.startswith("language_model.")
            or weight_name.startswith("model.language_model.")
        ):
            continue

        max_name = weight_name
        for before, after in GEMMA4_LANGUAGE_SAFETENSOR_MAP.items():
            max_name = max_name.replace(before, after)

        data = value.data()

        if max_name.endswith(".weight_scale") and data.dtype == DType.uint8:
            data = dataclasses.replace(data, dtype=DType.float8_e8m0fnu)

        # Stacked MoE expert weights: split into individual per-expert weights.
        # HF stores gate_up_proj [num_experts, 2*moe_dim, hidden_dim]
        # and down_proj [num_experts, hidden_dim, moe_dim] as single tensors.
        if "moe_block.experts.gate_up_proj" in max_name:
            prefix = max_name.split("moe_block.experts.")[0]
            buf = Buffer.from_dlpack(data.data)
            num_experts = buf.shape[0]
            half = buf.shape[1] // 2
            expert_shape = [half, buf.shape[2]]
            for j in range(num_experts):
                for proj, s in [
                    ("gate_proj", slice(None, half)),
                    ("up_proj", slice(half, None)),
                ]:
                    name = f"{prefix}moe_block.experts.{j}.{proj}.weight"
                    proj_buf = buf[j : j + 1, s, :].view(
                        data.dtype, expert_shape
                    )
                    new_state_dict[name] = WeightData(
                        proj_buf, name, data.dtype, Shape(expert_shape)
                    )
            continue

        if "moe_block.experts.down_proj" in max_name:
            prefix = max_name.split("moe_block.experts.")[0]
            buf = Buffer.from_dlpack(data.data)
            num_experts = buf.shape[0]
            expert_shape = list(buf.shape[1:])
            for j in range(num_experts):
                name = f"{prefix}moe_block.experts.{j}.down_proj.weight"
                expert_buf = buf[j : j + 1, :, :].view(data.dtype, expert_shape)
                new_state_dict[name] = WeightData(
                    expert_buf, name, data.dtype, Shape(expert_shape)
                )
            continue

        new_state_dict[max_name] = data

    return new_state_dict


def convert_safetensor_vision_state_dict(
    state_dict: dict[str, Weights],
) -> dict[str, WeightData]:
    """Convert safetensor state dict to MAX format for the vision model."""
    new_state_dict: dict[str, WeightData] = {}

    for weight_name, value in state_dict.items():
        if not (
            weight_name.startswith("model.vision_tower.")
            or weight_name.startswith("model.embed_vision.")
        ):
            continue

        max_name = weight_name
        for before, after in GEMMA4_VISION_SAFETENSOR_MAP.items():
            max_name = max_name.replace(before, after)

        new_state_dict[max_name] = value.data()

    return new_state_dict
