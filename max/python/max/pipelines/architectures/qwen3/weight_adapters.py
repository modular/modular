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
"""Weight adapters for Qwen3 and Qwen3-MoE models."""

from __future__ import annotations

from max.graph.weights import WeightData, Weights
from max.pipelines.lib import PipelineConfig
from transformers import AutoConfig

# Maps from Safetensor to MAX weight names for Qwen3-MoE.
# Individual expert weights pass through with name mapping only; no stacking.
# Legacy Linear/MLP use ".weight" directly (no ".linear"); MoEGate's router is
# gate_score (a Linear), so HF's mlp.gate.weight -> mlp.gate.gate_score.weight.
QWEN3_MOE_SAFETENSOR_MAPPING = {
    "model.": "",  # Removes the "model" prefix.
    "mlp.gate.weight": "mlp.gate.gate_score.weight",  # Router gate
}


def convert_qwen3_moe_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig,
    pipeline_config: PipelineConfig,
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Convert Qwen3-MoE weights to MAX format.

    Weights are passed through directly from the checkpoint with only name
    mapping: no stacking or numpy conversion. The base MoE layer uses
    individual expert MLPs, so HuggingFace's per-expert weight names map
    one-to-one to MAX's per-expert parameter names.

    HuggingFace Qwen3-MoE format (unchanged layout in MAX):
        - model.layers.{i}.mlp.experts.{j}.gate_proj.weight
        - model.layers.{i}.mlp.experts.{j}.up_proj.weight
        - model.layers.{i}.mlp.experts.{j}.down_proj.weight
        - model.layers.{i}.mlp.gate.weight

    MAX Qwen3-MoE (base MoE with individual experts; legacy Linear uses
    ".weight" directly; router is gate.gate_score):
        - layers.{i}.mlp.experts.{j}.gate_proj.weight
        - layers.{i}.mlp.experts.{j}.up_proj.weight
        - layers.{i}.mlp.experts.{j}.down_proj.weight
        - layers.{i}.mlp.gate.gate_score.weight

    Args:
        state_dict: The raw Qwen3-MoE checkpoint weights.
        huggingface_config: HuggingFace model configuration.
        pipeline_config: Pipeline configuration.

    Returns:
        The transformed weights for MAX Qwen3-MoE model.
    """
    new_state_dict: dict[str, WeightData] = {}

    for safetensor_name, value in state_dict.items():
        max_name = safetensor_name
        for before, after in QWEN3_MOE_SAFETENSOR_MAPPING.items():
            max_name = max_name.replace(before, after)
        new_state_dict[max_name] = value.data()

    return new_state_dict
