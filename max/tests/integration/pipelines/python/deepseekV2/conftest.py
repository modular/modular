# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import json
import os
from pathlib import Path

import pytest
import torch
from torch_reference.configuration_deepseek import (
    DeepseekV2Config,
)

"""
Fixtures for DeepseekV2 tests, including config, generated input tensors, and dummy weights.
"""

WEIGHT_STDDEV = 0.001


@pytest.fixture
def config() -> DeepseekV2Config:
    config = DeepseekV2Config()
    path = os.getenv("PIPELINES_TESTDATA")
    config_path = Path(path) / "config.json"  # type: ignore
    with open(config_path) as file:
        data = json.load(file)
    config.update(data)
    return config


@pytest.fixture
def input_tensor(
    config: DeepseekV2Config,
    seq_len: int = 7,
    batch_size: int = 1,
    seed: int = 42,
) -> torch.Tensor:
    torch.manual_seed(seed)  # Set fixed seed for reproducibility
    return torch.randn(
        batch_size,
        seq_len,
        config.hidden_size,
        dtype=torch.bfloat16,
    )


@pytest.fixture
def input_tensor_rope(
    config: DeepseekV2Config,
    seq_len: int = 7,
    batch_size: int = 1,
    seed: int = 1234,
) -> torch.Tensor:
    torch.manual_seed(seed)  # Set fixed seed for reproducibility

    # x: [bs, num_attention_heads, seq_len, head_size]

    return torch.randn(
        batch_size,
        config.num_attention_heads,
        seq_len,
        config.qk_rope_head_dim,
        dtype=torch.bfloat16,
    )


@pytest.fixture
def attention_mask(
    seq_len: int = 7,
    batch_size: int = 1,
) -> torch.Tensor:
    # Create causal mask where future tokens can't attend to past tokens
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    causal_mask = torch.zeros(
        1, batch_size, seq_len, seq_len, dtype=torch.bfloat16
    )
    causal_mask.masked_fill_(mask, float("-inf")).to(torch.bfloat16)
    return causal_mask


@pytest.fixture
def dummy_moe_weight(
    config: DeepseekV2Config, seed: int = 1234
) -> torch.Tensor:
    """
    Fixture to create dummy weights for an MLP layer.
    Returns tensors in bfloat16 format.
    """
    torch.manual_seed(seed)  # Set fixed seed for reproducibility
    n_experts = (
        config.n_routed_experts if config.n_routed_experts is not None else 64
    )
    return (
        torch.randn(n_experts, config.hidden_size, dtype=torch.float32)
        * WEIGHT_STDDEV
    )


@pytest.fixture
def shared_expert_weights(config: DeepseekV2Config) -> dict[str, torch.Tensor]:
    """Create dummy weights for shared experts"""
    torch.manual_seed(42)  # For reproducibility
    assert isinstance(config.moe_intermediate_size, int)
    assert isinstance(config.n_shared_experts, int)
    shared_experts_intermediate_size = (
        config.moe_intermediate_size * config.n_shared_experts
    )
    expert = {
        "down_proj.weight": torch.randn(
            config.hidden_size,
            shared_experts_intermediate_size,
            dtype=torch.bfloat16,
        )
        * WEIGHT_STDDEV,
        "gate_proj.weight": torch.randn(
            shared_experts_intermediate_size,
            config.hidden_size,
            dtype=torch.bfloat16,
        )
        * WEIGHT_STDDEV,
        "up_proj.weight": torch.randn(
            shared_experts_intermediate_size,
            config.hidden_size,
            dtype=torch.bfloat16,
        )
        * WEIGHT_STDDEV,
    }
    return expert


@pytest.fixture
def expert_weights(config: DeepseekV2Config) -> list[dict[str, torch.Tensor]]:
    """Create dummy weights for individual experts"""
    experts = []
    n_experts = (
        config.n_routed_experts if config.n_routed_experts is not None else 64
    )
    for i in range(n_experts):
        torch.manual_seed(i)  # For reproducibility
        expert = {
            "down_proj.weight": torch.randn(
                config.hidden_size,
                config.moe_intermediate_size,
                dtype=torch.bfloat16,
            )
            * WEIGHT_STDDEV,
            "gate_proj.weight": torch.randn(
                config.moe_intermediate_size,
                config.hidden_size,
                dtype=torch.bfloat16,
            )
            * WEIGHT_STDDEV,
            "up_proj.weight": torch.randn(
                config.moe_intermediate_size,
                config.hidden_size,
                dtype=torch.bfloat16,
            )
            * WEIGHT_STDDEV,
        }
        experts.append(expert)
    return experts


@pytest.fixture
def attention_weights(config: DeepseekV2Config) -> dict[str, torch.Tensor]:
    """Create dummy weights for DeepseekV2Attention module"""
    torch.manual_seed(42)  # For reproducibility

    weight_scale = 192.0  # so that we won't get overflow in the attention layer

    weights = {}

    # Query projection weights
    if config.q_lora_rank is not None:
        weights["q_a_proj.weight"] = (
            torch.randn(
                config.q_lora_rank,
                config.hidden_size,
                dtype=torch.bfloat16,
            )
            / weight_scale
        )
        weights["q_a_layernorm.weight"] = torch.ones(
            config.q_lora_rank, dtype=torch.bfloat16
        )
        weights["q_b_proj.weight"] = (
            torch.randn(
                config.num_attention_heads
                * (config.qk_nope_head_dim + config.qk_rope_head_dim),
                config.q_lora_rank,
                dtype=torch.bfloat16,
            )
            / weight_scale
        )
    else:
        weights["q_proj.weight"] = (
            torch.randn(
                config.num_attention_heads
                * (config.qk_nope_head_dim + config.qk_rope_head_dim),
                config.hidden_size,
                dtype=torch.bfloat16,
            )
            / weight_scale
        )

    # Key-value projection weights
    weights["kv_a_proj_with_mqa.weight"] = (
        torch.randn(
            config.kv_lora_rank + config.qk_rope_head_dim,
            config.hidden_size,
            dtype=torch.bfloat16,
        )
        / weight_scale
    )
    weights["kv_a_layernorm.weight"] = torch.ones(
        config.kv_lora_rank, dtype=torch.bfloat16
    )
    weights["kv_b_proj.weight"] = (
        torch.randn(
            config.num_attention_heads
            * (config.qk_nope_head_dim + config.v_head_dim),
            config.kv_lora_rank,
            dtype=torch.bfloat16,
        )
        / weight_scale
    )

    # Output projection weights
    weights["o_proj.weight"] = (
        torch.randn(
            config.hidden_size,
            config.num_attention_heads * config.v_head_dim,
            dtype=torch.bfloat16,
        )
        / weight_scale
    )

    return weights
