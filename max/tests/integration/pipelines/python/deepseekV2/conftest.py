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


@pytest.fixture
def config() -> DeepseekV2Config:
    config = DeepseekV2Config()
    path = os.getenv("PIPELINES_TESTDATA")
    config_path = Path(path) / "config.json"  # type: ignore
    with open(config_path, "r") as file:
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
        config.qk_nope_head_dim,
        dtype=torch.bfloat16,
    )


@pytest.fixture
def attention_mask(
    config: DeepseekV2Config,
    seq_len: int = 40,
    batch_size: int = 1,
    seed: int = 1234,
) -> torch.Tensor:
    # TODO: This likely needs to be generated differently to produce a valid attention mask (MODELS-369).
    torch.manual_seed(seed)  # Set fixed seed for reproducibility
    return torch.randn(
        1,
        batch_size,
        seq_len,
        seq_len,
        dtype=torch.bfloat16,
    )


@pytest.fixture
def dummy_moe_weight(
    config: DeepseekV2Config, seed: int = 1234
) -> torch.Tensor:
    """
    Fixture to create dummy weights for an MLP layer.
    Returns tensors in bfloat16 format.
    """
    torch.manual_seed(seed)  # Set fixed seed for reproducibility
    return torch.randn(
        (config.n_routed_experts, config.hidden_size), dtype=torch.float32
    )
