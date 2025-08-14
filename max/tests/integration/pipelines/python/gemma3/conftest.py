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
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig

"""
Fixtures for gemma3 tests, including config, generated input tensors, and dummy
weights.
"""


@pytest.fixture
def text_config() -> Gemma3TextConfig:
    path = os.environ["PIPELINES_TESTDATA"]
    config_path = Path(path) / "config.json"
    with open(config_path) as file:
        data = json.load(file)
    # Use "text_config" for the multimodal variants
    if "text_config" in data:
        return Gemma3TextConfig(
            **data["text_config"], attn_implementation="eager"
        )
    else:
        return Gemma3TextConfig(**data, attn_implementation="eager")


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


@pytest.fixture
def attention_weights(text_config: Gemma3TextConfig) -> dict[str, torch.Tensor]:
    torch.manual_seed(42)

    # calculated from google/gemma-3-1b-it checkpoint
    O_PROJ_STD = 0.0237
    K_PROJ_STD = 0.0309
    Q_PROJ_STD = 0.0284
    V_PROJ_STD = 0.0309
    K_NORM_STD = 0.793
    Q_NORM_STD = 0.68

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
