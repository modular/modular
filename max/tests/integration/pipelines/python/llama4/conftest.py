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
from transformers.models.llama4.configuration_llama4 import (
    Llama4Config,
    Llama4TextConfig,
)

"""
Fixtures for Llama4 tests, including config, generated input tensors, and dummy weights.
"""

WEIGHTS_STDDEV = 0.001


@pytest.fixture
def text_config() -> Llama4TextConfig:
    config = Llama4TextConfig()
    path = os.getenv("PIPELINES_TESTDATA")
    config_path = Path(path) / "config.json"  # type: ignore
    with open(config_path) as file:
        data = json.load(file)
    config.update(data)
    return config


@pytest.fixture
def config(text_config: Llama4TextConfig) -> Llama4Config:
    config = Llama4Config()
    config.text_config = text_config
    return config


@pytest.fixture
def input_tensor(
    config: Llama4TextConfig,
    seq_len: int = 7,
    batch_size: int = 1,
    seed: int = 42,
) -> torch.Tensor:
    torch.manual_seed(seed)  # Set fixed seed for reproducibility
    return torch.randn(
        batch_size,
        seq_len,
        5120,
        dtype=torch.bfloat16,
    )


@pytest.fixture
def dummy_router_weight(
    text_config: Llama4TextConfig, seed: int = 1234
) -> torch.Tensor:
    """
    Fixture to create dummy weights for an MLP layer.
    Returns tensors in bfloat16 format.
    """
    torch.manual_seed(seed)  # Set fixed seed for reproducibility
    return (
        torch.randn(
            (text_config.num_local_experts, text_config.hidden_size),
            dtype=torch.bfloat16,
        )
        * WEIGHTS_STDDEV
    )


@pytest.fixture
def shared_expert_weights(
    text_config: Llama4TextConfig,
) -> dict[str, torch.Tensor]:
    """Create dummy weights for shared experts"""
    torch.manual_seed(42)  # For reproducibility

    expert = {
        "down_proj.weight": torch.randn(
            text_config.hidden_size,
            text_config.intermediate_size,
            dtype=torch.bfloat16,
        )
        * WEIGHTS_STDDEV,
        "gate_proj.weight": torch.randn(
            text_config.intermediate_size,
            text_config.hidden_size,
            dtype=torch.bfloat16,
        )
        * WEIGHTS_STDDEV,
        "up_proj.weight": torch.randn(
            text_config.intermediate_size,
            text_config.hidden_size,
            dtype=torch.bfloat16,
        )
        * WEIGHTS_STDDEV,
    }
    return expert


@pytest.fixture
def expert_weights(text_config: Llama4TextConfig) -> dict[str, torch.Tensor]:
    """Create dummy weights for experts"""
    torch.manual_seed(42)  # For reproducibility
    expert = {
        "down_proj": torch.randn(
            text_config.num_local_experts,
            text_config.intermediate_size,
            text_config.hidden_size,
            dtype=torch.bfloat16,
        )
        * WEIGHTS_STDDEV,
        "gate_up_proj": torch.randn(
            text_config.num_local_experts,
            text_config.hidden_size,
            text_config.intermediate_size_mlp,
            dtype=torch.bfloat16,
        )
        * WEIGHTS_STDDEV,
    }
    return expert
