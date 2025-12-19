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

"""Weight conversion utilities for Qwen3VL tests."""

import torch
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeTextSparseMoeBlock,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeVisionPatchMerger as HFQwen3VLMoeVisionPatchMerger,
)


def load_weights_to_hf_merger(
    hf_merger: HFQwen3VLMoeVisionPatchMerger,
    weights: dict[str, torch.Tensor],
) -> None:
    """Load weights into HuggingFace merger."""
    device = weights["norm.weight"].device
    # Load norm weights
    hf_merger.norm.weight.data = weights["norm.weight"].cuda()
    hf_merger.norm.bias.data = weights["norm.bias"].cuda()

    # Load MLP weights
    hf_merger.linear_fc1.weight.data = weights["linear_fc1.weight"].cuda()
    hf_merger.linear_fc1.bias.data = weights["linear_fc1.bias"].cuda()
    hf_merger.linear_fc2.weight.data = weights["linear_fc2.weight"].cuda()
    hf_merger.linear_fc2.bias.data = weights["linear_fc2.bias"].cuda()


def load_weights_to_hf_moe_block(
    sparse_moe_block: Qwen3VLMoeTextSparseMoeBlock,
    moe_weights: dict[str, torch.Tensor],
    device: torch.device,
) -> None:
    """Load weights from MAX Qwen3VLMoE format into HuggingFace MoE block.

    This function converts weights from MAX's Qwen3VLMoE layer format to HF's
    Qwen3VLMoeTextSparseMoeBlock format. The weight formats happen to match, so
    no transposition is needed.

    Args:
        sparse_moe_block: The HF Qwen3VLMoeTextSparseMoeBlock instance to load weights into.
        moe_weights: Dictionary containing MoE weights in MAX Qwen3VLMoE format:
            - "gate.gate_score.weight": [num_experts, hidden_dim]
            - "experts.gate_up_proj": [num_experts, hidden_dim, 2*moe_dim] (concatenated)
              Example: [128, 2048, 1536] where hidden_dim=2048, moe_dim=768
            - "experts.down_proj": [num_experts, moe_dim, hidden_dim]
              Example: [128, 768, 2048] where moe_dim=768, hidden_dim=2048
        device: Device to load weights onto.

    Note:
        MAX's Qwen3VLMoE stores weights as:
        - gate_up_proj: [num_experts, hidden_dim, 2*moe_dim]
        - down_proj: [num_experts, moe_dim, hidden_dim]

        HF's Qwen3VLMoeTextExperts expects:
        - gate_up_proj: [num_experts, hidden_size, 2*expert_dim]
        - down_proj: [num_experts, expert_dim, hidden_size]
    """
    # Load router weights (gate)
    sparse_moe_block.gate.weight.data = moe_weights[
        "gate.gate_score.weight"
    ].to(device)

    # Load expert weights
    # - gate_up_proj: [num_experts, hidden_size, 2*expert_dim]
    # - down_proj: [num_experts, expert_dim, hidden_size]

    gate_up_proj = moe_weights["experts.gate_up_proj"].to(device)
    sparse_moe_block.experts.gate_up_proj.data = gate_up_proj

    down_proj = moe_weights["experts.down_proj"].to(device)
    sparse_moe_block.experts.down_proj.data = down_proj
