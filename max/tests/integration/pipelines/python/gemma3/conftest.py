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
from transformers.models.gemma3.configuration_gemma3 import (
    Gemma3TextConfig,
)

"""
Fixtures for gemma3 tests, including config, generated input tensors, and dummy
weights.
"""


@pytest.fixture
def text_config() -> Gemma3TextConfig:
    config = Gemma3TextConfig()
    path = os.getenv("PIPELINES_TESTDATA")
    config_path = Path(path) / "config.json"  # type: ignore
    with open(config_path) as file:
        data = json.load(file)
    config.update(data)
    return config


@pytest.fixture
def input_indices(
    text_config: Gemma3TextConfig,
    batch_size: int = 1,
    seq_len: int = 7,
    seed: int = 0,
) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randint(
        0,
        text_config.vocab_size,
        (batch_size, seq_len),
        dtype=torch.long,
    )


@pytest.fixture
def input_tensor(
    text_config: Gemma3TextConfig,
    batch_size: int = 1,
    seq_len: int = 7,
    seed: int = 42,
) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(
        batch_size,
        seq_len,
        text_config.hidden_size,
        dtype=torch.bfloat16,
    )


@pytest.fixture
def embedding_weights(text_config: Gemma3TextConfig) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(
        text_config.vocab_size,
        text_config.hidden_size,
        dtype=torch.bfloat16,
    )


@pytest.fixture
def rms_weight(text_config: Gemma3TextConfig) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(
        text_config.hidden_size,
        dtype=torch.float32,
    )
