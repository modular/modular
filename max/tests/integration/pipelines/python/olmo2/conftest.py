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
from transformers.models.olmo2.modular_olmo2 import Olmo2Config

"""
Fixtures for olmo2 tests, including config, generated input tensors, and dummy
weights.
"""


@pytest.fixture
def text_config() -> Olmo2Config:
    config = Olmo2Config()
    path = os.getenv("PIPELINES_TESTDATA")
    config_path = Path(path) / "config.json"  # type: ignore
    with open(config_path) as file:
        data = json.load(file)
    config.update(data)
    return config


@pytest.fixture
def attention_weights(text_config: Olmo2Config) -> dict[str, torch.Tensor]:
    torch.manual_seed(39)

    # Using actual standard deviations and means calculated from OLMo2 model weights
    # These values are based on the actual "allenai/OLMo-2-1124-7B" checkpoint
    O_PROJ_STD = 0.01850747
    O_PROJ_MEAN = 0.00000021
    K_PROJ_STD = 0.01497349
    K_PROJ_MEAN = -0.00000022
    Q_PROJ_STD = 0.01545093
    Q_PROJ_MEAN = -0.00000049
    V_PROJ_STD = 0.01876341
    V_PROJ_MEAN = 0.00000109
    # OLMo2 normalizes full projection dimension (different from Qwen3)
    Q_NORM_STD = 0.39965060
    Q_NORM_MEAN = 0.67815530
    K_NORM_STD = 0.40187800
    K_NORM_MEAN = 0.66121697

    text_config.head_dim = (
        text_config.hidden_size // text_config.num_attention_heads
    )

    q_dim = text_config.head_dim * text_config.num_attention_heads
    kv_dim = text_config.head_dim * text_config.num_key_value_heads

    assert q_dim == 128 * 32
    assert kv_dim == 128 * 32

    hidden_size = text_config.hidden_size

    return {
        # OLMo2: RMSNorm on full projection dimension (not just head_dim)
        "q_norm.weight": torch.randn(
            q_dim,  # num_attention_heads * head_dim
            dtype=torch.float32,
        )
        * Q_NORM_STD
        + Q_NORM_MEAN,
        "k_norm.weight": torch.randn(
            kv_dim,  # num_key_value_heads * head_dim
            dtype=torch.float32,
        )
        * K_NORM_STD
        + K_NORM_MEAN,
        "k_proj.weight": torch.randn(
            kv_dim,
            hidden_size,
            dtype=torch.float32,
        )
        * K_PROJ_STD
        + K_PROJ_MEAN,
        "o_proj.weight": torch.randn(
            hidden_size,
            q_dim,
            dtype=torch.float32,
        )
        * O_PROJ_STD
        + O_PROJ_MEAN,
        "q_proj.weight": torch.randn(
            q_dim,
            hidden_size,
            dtype=torch.float32,
        )
        * Q_PROJ_STD
        + Q_PROJ_MEAN,
        "v_proj.weight": torch.randn(
            kv_dim,
            hidden_size,
            dtype=torch.float32,
        )
        * V_PROJ_STD
        + V_PROJ_MEAN,
    }
