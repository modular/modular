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
def moe_weights_fp8() -> dict[str, torch.Tensor]:
    torch.manual_seed(42)

    fp8_dtype = torch.float8_e4m3fn
    scale_dtype = torch.float32
    fp8_max = torch.finfo(fp8_dtype).max
    fp8_min = torch.finfo(fp8_dtype).min

    moe_weights = {}

    # Gate weights for router
    moe_weights["gate.gate_score.weight"] = (
        torch.randn(NUM_EXPERTS, HIDDEN_DIM, dtype=torch.bfloat16)
        * WEIGHTS_STDDEV
    )

    # Individual expert weights
    FP8_WEIGHTS_MULTIPLIER = 100
    SCALE_MIN = 0.5 * (WEIGHTS_STDDEV / FP8_WEIGHTS_MULTIPLIER)
    SCALE_MAX = WEIGHTS_STDDEV / FP8_WEIGHTS_MULTIPLIER
    for expert_idx in range(NUM_EXPERTS):
        moe_weights[f"experts.{expert_idx}.gate_proj.weight"] = (
            (
                torch.randn(MOE_DIM, HIDDEN_DIM, dtype=torch.bfloat16)
                * FP8_WEIGHTS_MULTIPLIER
            )
            .clamp(fp8_min, fp8_max)
            .to(fp8_dtype)
        )
        moe_weights[f"experts.{expert_idx}.gate_proj.weight_scale"] = (
            torch.rand(MOE_DIM // 128, HIDDEN_DIM // 128, dtype=scale_dtype)
            * (SCALE_MAX - SCALE_MIN)
            + SCALE_MIN
        )

        moe_weights[f"experts.{expert_idx}.up_proj.weight"] = (
            (
                torch.randn(MOE_DIM, HIDDEN_DIM, dtype=torch.bfloat16)
                * FP8_WEIGHTS_MULTIPLIER
            )
            .clamp(fp8_min, fp8_max)
            .to(fp8_dtype)
        )
        moe_weights[f"experts.{expert_idx}.up_proj.weight_scale"] = (
            torch.rand(MOE_DIM // 128, HIDDEN_DIM // 128, dtype=scale_dtype)
            * (SCALE_MAX - SCALE_MIN)
            + SCALE_MIN
        )

        moe_weights[f"experts.{expert_idx}.down_proj.weight"] = (
            (
                torch.randn(HIDDEN_DIM, MOE_DIM, dtype=torch.bfloat16)
                * FP8_WEIGHTS_MULTIPLIER
            )
            .clamp(fp8_min, fp8_max)
            .to(fp8_dtype)
        )
        moe_weights[f"experts.{expert_idx}.down_proj.weight_scale"] = (
            torch.rand(HIDDEN_DIM // 128, MOE_DIM // 128, dtype=scale_dtype)
            * (SCALE_MAX - SCALE_MIN)
            + SCALE_MIN
        )

    # Shared experts weights
    moe_weights["shared_experts.down_proj.weight"] = (
        (
            torch.randn(HIDDEN_DIM, MOE_DIM, dtype=torch.bfloat16)
            * FP8_WEIGHTS_MULTIPLIER
        )
        .clamp(fp8_min, fp8_max)
        .to(fp8_dtype)
    )
    moe_weights["shared_experts.down_proj.weight_scale"] = (
        torch.rand(HIDDEN_DIM // 128, MOE_DIM // 128, dtype=scale_dtype)
        * (SCALE_MAX - SCALE_MIN)
        + SCALE_MIN
    )
    moe_weights["shared_experts.gate_proj.weight"] = (
        (
            torch.randn(MOE_DIM, HIDDEN_DIM, dtype=torch.bfloat16)
            * FP8_WEIGHTS_MULTIPLIER
        )
        .clamp(fp8_min, fp8_max)
        .to(fp8_dtype)
    )
    moe_weights["shared_experts.gate_proj.weight_scale"] = (
        torch.rand(MOE_DIM // 128, HIDDEN_DIM // 128, dtype=scale_dtype)
        * (SCALE_MAX - SCALE_MIN)
        + SCALE_MIN
    )
    moe_weights["shared_experts.up_proj.weight"] = (
        (
            torch.randn(MOE_DIM, HIDDEN_DIM, dtype=torch.bfloat16)
            * FP8_WEIGHTS_MULTIPLIER
        )
        .clamp(fp8_min, fp8_max)
        .to(fp8_dtype)
    )
    moe_weights["shared_experts.up_proj.weight_scale"] = (
        torch.rand(MOE_DIM // 128, HIDDEN_DIM // 128, dtype=scale_dtype)
        * (SCALE_MAX - SCALE_MIN)
        + SCALE_MIN
    )

    return moe_weights


def _dequantize_weight_blockwise_fp8(
    value: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    # expand the scales from (gm, gn) to (gm * 128, gn * 128)
    expanded_scale = scale.repeat_interleave(128, dim=0).repeat_interleave(
        128, dim=1
    )

    # dequantize the value
    return (value.to(scale.dtype) * expanded_scale).to(torch.bfloat16)


@pytest.fixture
def moe_weights(
    moe_weights_fp8: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    moe_weights = {}

    # first copy the gate weight as is
    moe_weights["gate.gate_score.weight"] = moe_weights_fp8[
        "gate.gate_score.weight"
    ]

    # then dequantize the expert weights
    for key, value in moe_weights_fp8.items():
        if key.endswith(".weight_scale"):
            weights_key = key.replace(".weight_scale", ".weight")

            moe_weights[weights_key] = _dequantize_weight_blockwise_fp8(
                moe_weights_fp8[weights_key], value
            )

    return moe_weights
