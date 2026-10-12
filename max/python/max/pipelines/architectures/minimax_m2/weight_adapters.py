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
"""Weight adapters for MiniMax-M2 models.

Maps HuggingFace checkpoint weight names to MAX nn.Module expected names.
MiniMax-M2 uses w1/w2/w3 naming for expert MLPs and block_sparse_moe
for the MoE container, with sigmoid routing and e_score_correction_bias.
"""

from __future__ import annotations

import dataclasses
import re

import numpy as np
from max.dtype import DType
from max.graph.weights import WeightData, Weights
from max.pipelines.lib import PipelineConfig
from transformers import AutoConfig

_EXPERT_WEIGHT_RE = re.compile(
    r"(model\.layers\.\d+\.block_sparse_moe\.experts\.\d+\.w\d+)\.weight$"
)

# Checkpoint -> MAX weight name mapping.
#
# HuggingFace checkpoint structure:
#   model.layers.{i}.block_sparse_moe.gate.weight
#   model.layers.{i}.block_sparse_moe.e_score_correction_bias
#   model.layers.{i}.block_sparse_moe.experts.{j}.w1.weight
#   model.layers.{i}.block_sparse_moe.experts.{j}.w2.weight
#   model.layers.{i}.block_sparse_moe.experts.{j}.w3.weight
#   model.layers.{i}.block_sparse_moe.experts.{j}.w1.weight_scale_inv
#   model.layers.{i}.self_attn.q_proj.weight
#   model.layers.{i}.self_attn.q_norm.weight
#   ...
#
# MAX nn.Module structure:
#   layers.{i}.mlp.gate.gate_score.weight
#   layers.{i}.mlp.gate.e_score_correction_bias
#   layers.{i}.mlp.experts.{j}.gate_proj.weight
#   layers.{i}.mlp.experts.{j}.down_proj.weight
#   layers.{i}.mlp.experts.{j}.up_proj.weight
#   layers.{i}.mlp.experts.{j}.gate_proj.weight_scale
#   layers.{i}.self_attn.q_proj.weight
#   layers.{i}.self_attn.q_norm.weight
#   ...
MINIMAX_M2_SAFETENSOR_MAP: dict[str, str] = {
    "model.": "",  # Strip model. prefix
    "block_sparse_moe.gate.weight": "mlp.gate.gate_score.weight",
    "block_sparse_moe.e_score_correction_bias": "mlp.gate.e_score_correction_bias",
    "block_sparse_moe.experts.": "mlp.experts.",
    # MiniMax-M2 expert MLP naming -> MAX MLP naming
    ".w1.": ".gate_proj.",  # gate projection
    ".w3.": ".up_proj.",  # up projection
    ".w2.": ".down_proj.",  # down projection
    "weight_scale_inv": "weight_scale",  # FP8 scale naming
}


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig,
    pipeline_config: PipelineConfig,
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Convert MiniMax-M2 safetensor weights to MAX format.

    Performs name mapping from HuggingFace conventions to MAX conventions
    and casts FP8 scale tensors to float32.

    Args:
        state_dict: The raw checkpoint weights.
        huggingface_config: HuggingFace model configuration.
        pipeline_config: Pipeline configuration.

    Returns:
        The transformed weights for the MAX MiniMax-M2 model.
    """
    new_state_dict: dict[str, WeightData] = {}

    num_hidden_layers = huggingface_config.num_hidden_layers
    skip_prefixes = tuple(
        f"model.layers.{i}."
        for i in range(num_hidden_layers, num_hidden_layers + 10)
    )
    skip_suffixes = (
        "input_quantizer",
        "weight_quantizer",
        "k_bmm_quantizer",
        "v_bmm_quantizer",
    )

    # Some NVFP4 experts may be missing an input_scale; default to 1.0.
    has_any_input_scale = any(k.endswith(".input_scale") for k in state_dict)
    if has_any_input_scale:
        for safetensor_name in state_dict:
            m = _EXPERT_WEIGHT_RE.match(safetensor_name)
            if m and f"{m.group(1)}.input_scale" not in state_dict:
                max_name = f"{m.group(1)}.input_scale"
                for before, after in MINIMAX_M2_SAFETENSOR_MAP.items():
                    max_name = max_name.replace(before, after)
                new_state_dict[max_name] = WeightData.from_numpy(
                    np.array(1.0, dtype=np.float32), max_name
                )

    for safetensor_name, value in state_dict.items():
        if safetensor_name.startswith(skip_prefixes):
            continue
        if safetensor_name.endswith(skip_suffixes):
            continue

        max_name = safetensor_name
        for before, after in MINIMAX_M2_SAFETENSOR_MAP.items():
            max_name = max_name.replace(before, after)

        data = value.data()

        # Safetensors stores E8M0 scales as uint8; reinterpret.
        if max_name.endswith(".weight_scale") and data.dtype == DType.uint8:
            data = dataclasses.replace(data, dtype=DType.float8_e8m0fnu)

        new_state_dict[max_name] = data

    return new_state_dict
