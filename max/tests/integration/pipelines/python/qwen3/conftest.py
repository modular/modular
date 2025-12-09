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
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

"""
Fixtures for qwen3 tests, including config, generated input tensors, and dummy
weights.
"""


@pytest.fixture
def text_config() -> Qwen3Config:
    config = Qwen3Config(attn_implementation="eager")
    path = os.environ["PIPELINES_TESTDATA"]
    config_path = Path(path) / "config.json"
    with open(config_path) as file:
        data = json.load(file)
    config.update(data)
    return config


@pytest.fixture
def input_indices(
    text_config: Qwen3Config,
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
    text_config: Qwen3Config,
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
def embedding_weights(text_config: Qwen3Config) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(
        text_config.vocab_size,
        text_config.hidden_size,
        dtype=torch.bfloat16,
    )


@pytest.fixture
def rms_weight(text_config: Qwen3Config) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(
        text_config.hidden_size,
        dtype=torch.float32,
    )


@pytest.fixture
def attention_weights(text_config: Qwen3Config) -> dict[str, torch.Tensor]:
    torch.manual_seed(42)

    # calculated from Qwen/Qwen3-1.7B checkpoint
    # ==================================================
    # ATTENTION LAYER WEIGHT STANDARD DEVIATIONS
    # ==================================================
    # Q_PROJ_STD: 0.035518
    # K_PROJ_STD: 0.031698
    # V_PROJ_STD: 0.034377
    # O_PROJ_STD: 0.031565
    # Q_NORM_STD: 0.788619
    # K_NORM_STD: 1.858506
    # ==================================================

    # Qwen/Qwen3-32B checkpoint
    # ==================================================
    # Q_PROJ_STD: 0.018800
    # K_PROJ_STD: 0.023036
    # V_PROJ_STD: 0.024043
    # O_PROJ_STD: 0.022142
    # Q_NORM_STD: 0.474626
    # K_NORM_STD: 0.832018
    # ==================================================

    O_PROJ_STD = 0.035518
    K_PROJ_STD = 0.031698
    Q_PROJ_STD = 0.034377
    V_PROJ_STD = 0.031565
    K_NORM_STD = 0.788619
    Q_NORM_STD = 1.858506

    q_dim = text_config.head_dim * text_config.num_attention_heads
    kv_dim = text_config.head_dim * text_config.num_key_value_heads
    hidden_size = text_config.hidden_size

    return {
        "k_norm.weight": torch.randn(
            text_config.head_dim,
            dtype=torch.bfloat16,
        )
        * K_NORM_STD,
        "k_proj.weight": torch.randn(
            kv_dim,
            hidden_size,
            dtype=torch.bfloat16,
        )
        * K_PROJ_STD,
        "o_proj.weight": torch.randn(
            hidden_size,
            q_dim,
            dtype=torch.bfloat16,
        )
        * O_PROJ_STD,
        "q_norm.weight": torch.randn(
            text_config.head_dim,
            dtype=torch.bfloat16,
        )
        * Q_NORM_STD,
        "q_proj.weight": torch.randn(
            q_dim,
            hidden_size,
            dtype=torch.bfloat16,
        )
        * Q_PROJ_STD,
        "v_proj.weight": torch.randn(
            kv_dim,
            hidden_size,
            dtype=torch.bfloat16,
        )
        * V_PROJ_STD,
    }
