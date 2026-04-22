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

from max.dtype import DType
from max.graph.weights import WeightData, Weights
from max.pipelines.lib import PipelineConfig
from transformers import AutoConfig

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

    # Skip MTP (multi-token prediction) layer weights
    num_hidden_layers = huggingface_config.num_hidden_layers
    skip_prefixes = tuple(
        f"model.layers.{i}."
        for i in range(num_hidden_layers, num_hidden_layers + 10)
    )

    for safetensor_name, value in state_dict.items():
        # Skip MTP layers
        if safetensor_name.startswith(skip_prefixes):
            continue

        max_name = safetensor_name
        for before, after in MINIMAX_M2_SAFETENSOR_MAP.items():
            max_name = max_name.replace(before, after)

        weight_data = value.data()

        # Cast FP8 scale tensors to float32
        if (
            ("weight_scale" in max_name or "input_scale" in max_name)
            and weight_data.dtype.is_float()
            and weight_data.dtype != DType.float32
        ):
            weight_data = weight_data.astype(DType.float32)

        new_state_dict[max_name] = weight_data

    return new_state_dict
