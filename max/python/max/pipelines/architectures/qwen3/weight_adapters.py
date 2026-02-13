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

import re
from collections import defaultdict

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.graph.type import Shape
from max.graph.weights import WeightData, Weights
from max.pipelines.lib import PipelineConfig
from transformers import AutoConfig

# Maps from Safetensor to MAX weight names for Qwen3-MoE.
QWEN3_MOE_SAFETENSOR_MAPPING = {
    "model.": "",  # Removes the "model" prefix.
    "mlp.gate.weight": "mlp.gate.gate_score.weight",  # Rename router gate
}


def _weight_data_to_numpy(weight_data: WeightData) -> np.ndarray:
    """Convert WeightData to numpy array, handling bfloat16 specially.

    numpy's from_dlpack doesn't support bfloat16, so we use MAX's Buffer
    to handle the conversion by viewing the data as uint16 (same bit width).

    Args:
        weight_data: The weight data to convert.

    Returns:
        Numpy array with the weight data. For bfloat16, returns a uint16 view.
    """
    if weight_data.dtype == DType.bfloat16:
        # bfloat16 is not supported by numpy's dlpack, so we view as uint16
        # (same 16-bit layout) for stacking operations
        buffer = Buffer.from_dlpack(weight_data.data)
        # View the buffer as uint16 (same bit pattern)
        buffer_uint16 = buffer.view(dtype=DType.uint16, shape=buffer.shape)
        return buffer_uint16.to_numpy()
    else:
        # Match llama3/weight_adapters.py by suppressing this type error.
        return np.from_dlpack(weight_data.data)  # type: ignore


def _numpy_to_weight_data(
    arr: np.ndarray, name: str, original_dtype: DType
) -> WeightData:
    """Convert numpy array back to WeightData, handling bfloat16 specially.

    Args:
        arr: Numpy array (may be uint16 view for bfloat16).
        name: Name for the weight.
        original_dtype: Original dtype to restore.

    Returns:
        WeightData with the correct dtype.
    """
    if original_dtype == DType.bfloat16:
        # The array is uint16 view of bfloat16 data
        # Create a buffer and view it back as bfloat16
        buffer = Buffer.from_numpy(arr)
        buffer_bf16 = buffer.view(dtype=DType.bfloat16, shape=buffer.shape)
        return WeightData(
            data=buffer_bf16,
            name=name,
            dtype=original_dtype,
            shape=Shape(buffer_bf16.shape),
        )
    else:
        return WeightData.from_numpy(arr.copy(), name)


def convert_qwen3_moe_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig,
    pipeline_config: PipelineConfig,
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Convert Qwen3-MoE weights to MAX format.

    This adapter handles:
    1. Standard weight name mapping (removes model. prefix)
    2. Stacking individual expert weights into [num_experts, ...] tensors
    3. Combining gate_proj and up_proj into gate_up_proj tensors
    4. Renaming router gate weight

    HuggingFace Qwen3-MoE format:
        - model.layers.{i}.mlp.experts.{j}.gate_proj.weight: [moe_dim, hidden_dim]
        - model.layers.{i}.mlp.experts.{j}.up_proj.weight: [moe_dim, hidden_dim]
        - model.layers.{i}.mlp.experts.{j}.down_proj.weight: [hidden_dim, moe_dim]
        - model.layers.{i}.mlp.gate.weight: [num_experts, hidden_dim]

    MAX Qwen3MoE expected format:
        - layers.{i}.mlp.experts.gate_up_proj: [num_experts, hidden_dim, 2*moe_dim]
        - layers.{i}.mlp.experts.down_proj: [num_experts, moe_dim, hidden_dim]
        - layers.{i}.mlp.gate.gate_score.weight: [num_experts, hidden_dim]

    Args:
        state_dict: The raw Qwen3-MoE checkpoint weights.
        huggingface_config: HuggingFace model configuration.
        pipeline_config: Pipeline configuration.

    Returns:
        The transformed weights for MAX Qwen3MoE model.
    """
    new_state_dict: dict[str, WeightData] = {}

    # Pattern to match expert weights
    expert_pattern = re.compile(
        r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight"
    )

    # Collect expert weights by layer for stacking
    # Structure: {layer_idx: {expert_idx: {proj_type: weight_data}}}
    expert_weights: dict[int, dict[int, dict[str, WeightData]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    # First pass: separate expert weights from other weights
    for safetensor_name, value in state_dict.items():
        match = expert_pattern.match(safetensor_name)

        if match:
            # This is an expert weight - collect it for stacking
            layer_idx = int(match.group(1))
            expert_idx = int(match.group(2))
            proj_type = match.group(3)  # gate_proj, up_proj, or down_proj
            expert_weights[layer_idx][expert_idx][proj_type] = value.data()
        else:
            # Standard weight - apply name mapping
            max_name = safetensor_name
            for before, after in QWEN3_MOE_SAFETENSOR_MAPPING.items():
                max_name = max_name.replace(before, after)
            new_state_dict[max_name] = value.data()

    # Second pass: stack expert weights
    for layer_idx in sorted(expert_weights.keys()):
        experts = expert_weights[layer_idx]
        num_experts = len(experts)

        # Collect and stack gate_up projections
        gate_projs = []
        up_projs = []
        down_projs = []

        # Get the dtype from first expert's weight for later restoration
        first_expert = experts[0]
        original_dtype = first_expert["gate_proj"].dtype

        for expert_idx in range(num_experts):
            expert_data = experts[expert_idx]
            gate_projs.append(_weight_data_to_numpy(expert_data["gate_proj"]))
            up_projs.append(_weight_data_to_numpy(expert_data["up_proj"]))
            down_projs.append(_weight_data_to_numpy(expert_data["down_proj"]))

        # Stack gate and up projections
        # HF format: [moe_dim, hidden_dim] per expert
        # Target format: [num_experts, hidden_dim, 2*moe_dim]
        # We need to transpose and concatenate gate + up along last dim

        # Stack all experts: [num_experts, moe_dim, hidden_dim]
        stacked_gate = np.stack(gate_projs, axis=0)
        stacked_up = np.stack(up_projs, axis=0)

        # Transpose to [num_experts, hidden_dim, moe_dim]
        stacked_gate = np.transpose(stacked_gate, (0, 2, 1))
        stacked_up = np.transpose(stacked_up, (0, 2, 1))

        # Concatenate gate and up along last dimension: [num_experts, hidden_dim, 2*moe_dim]
        gate_up_proj = np.concatenate([stacked_gate, stacked_up], axis=2)

        # Ensure contiguous memory layout
        gate_up_proj = np.ascontiguousarray(gate_up_proj)

        # Store gate_up_proj
        gate_up_name = f"layers.{layer_idx}.mlp.experts.gate_up_proj"
        new_state_dict[gate_up_name] = _numpy_to_weight_data(
            gate_up_proj, gate_up_name, original_dtype
        )

        # Stack down projections
        # HF format: [hidden_dim, moe_dim] per expert
        # Target format: [num_experts, moe_dim, hidden_dim]
        stacked_down = np.stack(down_projs, axis=0)
        # Transpose to [num_experts, moe_dim, hidden_dim]
        stacked_down = np.transpose(stacked_down, (0, 2, 1))

        # Ensure contiguous memory layout
        stacked_down = np.ascontiguousarray(stacked_down)

        down_name = f"layers.{layer_idx}.mlp.experts.down_proj"
        new_state_dict[down_name] = _numpy_to_weight_data(
            stacked_down, down_name, original_dtype
        )

    return new_state_dict
