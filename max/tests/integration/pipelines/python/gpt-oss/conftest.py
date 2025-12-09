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
from transformers.models.gpt_oss.configuration_gpt_oss import (
    GptOssConfig,
)

"""
Fixtures for GPT-OSS tests, including config, generated input tensors, and dummy
weights.
"""

WEIGHTS_STDDEV = 0.01


@pytest.fixture
def config() -> GptOssConfig:
    """Create a simple GPT-OSS config for testing."""
    path = os.environ["PIPELINES_TESTDATA"]
    config_path = Path(path) / "config.json"
    with open(config_path) as file:
        data = json.load(file)

    return GptOssConfig(**data, attn_implementation="eager")


@pytest.fixture
def input_tensor(config: GptOssConfig) -> torch.Tensor:
    """Generate a random input tensor for testing."""
    torch.manual_seed(42)
    return torch.randn(1, 76, config.hidden_size)
