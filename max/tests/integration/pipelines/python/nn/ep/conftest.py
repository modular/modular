# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest
import torch

"""
Fixtures for EP tests, including dummy weights.
"""

MOE_DIM = 2048
HIDDEN_DIM = 7168
NUM_EXPERTS = 64
WEIGHTS_STDDEV = 0.01


@pytest.fixture
def moe_weights() -> dict[str, torch.Tensor]:
    torch.manual_seed(42)

    moe_weights = {}

    # Gate weights for router
    moe_weights["gate.gate_score.weight"] = (
        torch.randn(NUM_EXPERTS, HIDDEN_DIM, dtype=torch.bfloat16)
        * WEIGHTS_STDDEV
    )

    # Individual expert weights
    for expert_idx in range(NUM_EXPERTS):
        moe_weights[f"experts.{expert_idx}.gate_proj.weight"] = (
            torch.randn(MOE_DIM, HIDDEN_DIM, dtype=torch.bfloat16)
            * WEIGHTS_STDDEV
        )

        moe_weights[f"experts.{expert_idx}.up_proj.weight"] = (
            torch.randn(MOE_DIM, HIDDEN_DIM, dtype=torch.bfloat16)
            * WEIGHTS_STDDEV
        )

        moe_weights[f"experts.{expert_idx}.down_proj.weight"] = (
            torch.randn(HIDDEN_DIM, MOE_DIM, dtype=torch.bfloat16)
            * WEIGHTS_STDDEV
        )

    # Shared experts weights
    moe_weights["shared_experts.down_proj.weight"] = (
        torch.randn(HIDDEN_DIM, MOE_DIM, dtype=torch.bfloat16) * WEIGHTS_STDDEV
    )

    moe_weights["shared_experts.gate_proj.weight"] = (
        torch.randn(MOE_DIM, HIDDEN_DIM, dtype=torch.bfloat16) * WEIGHTS_STDDEV
    )

    moe_weights["shared_experts.up_proj.weight"] = (
        torch.randn(MOE_DIM, HIDDEN_DIM, dtype=torch.bfloat16) * WEIGHTS_STDDEV
    )

    return moe_weights
