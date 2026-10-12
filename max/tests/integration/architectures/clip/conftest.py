# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch

"""
Fixtures for CLIP tests, including config, generated input tensors, attention
masks, and dummy weights.
"""


@pytest.fixture
def clip_config() -> dict[str, Any]:
    """Load CLIP configuration from testdata."""
    path = os.environ["PIPELINES_TESTDATA"]
    config_path = Path(path) / "config.json"
    with open(config_path) as file:
        return json.load(file)


@pytest.fixture
def input_tensor(clip_config: dict[str, Any]) -> torch.Tensor:
    """Generate random input tensor for CLIP attention testing.

    Shape: (batch_size, seq_length, hidden_size)
    """
    torch.manual_seed(42)
    batch_size = 1
    seq_length = 77  # Standard CLIP text sequence length
    hidden_size = clip_config["hidden_size"]
    return (
        torch.randn(batch_size, seq_length, hidden_size)
        .to(torch.bfloat16)
        .to("cuda")
    )


@pytest.fixture
def attention_mask(clip_config: dict[str, Any]) -> torch.Tensor | None:
    """Generate attention mask for padding tokens.

    Returns None to test without padding mask (all tokens are valid).
    For testing with mask, return a tensor of shape (batch_size, seq_length).
    """
    # Return None for now - testing without padding mask
    return None


@pytest.fixture
def causal_attention_mask(clip_config: dict[str, Any]) -> torch.Tensor | None:
    """Generate causal attention mask for autoregressive attention.

    Shape: (batch_size, 1, seq_length, seq_length) or (1, 1, seq_length, seq_length)
    Returns None to test without causal mask.
    """
    # For CLIP text encoder, causal mask is typically used
    # Return None for basic test - can be extended later
    return None


@pytest.fixture
def attention_weights(clip_config: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Generate random weights for CLIP attention layer.

    Weight structure matches both transformers.CLIPAttention and MAX CLIPAttention:
    - q_proj, k_proj, v_proj: projection layers with bias
    - out_proj: output projection with bias
    """
    torch.manual_seed(100)

    embed_dim = clip_config["hidden_size"]

    WEIGHT_SCALE = 0.02
    BIAS_SCALE = 0.1

    weights = {
        # Query projection
        "q_proj.weight": torch.randn(
            embed_dim,
            embed_dim,
            dtype=torch.bfloat16,
        )
        * WEIGHT_SCALE,
        "q_proj.bias": torch.randn(
            embed_dim,
            dtype=torch.bfloat16,
        )
        * BIAS_SCALE,
        # Key projection
        "k_proj.weight": torch.randn(
            embed_dim,
            embed_dim,
            dtype=torch.bfloat16,
        )
        * WEIGHT_SCALE,
        "k_proj.bias": torch.randn(
            embed_dim,
            dtype=torch.bfloat16,
        )
        * BIAS_SCALE,
        # Value projection
        "v_proj.weight": torch.randn(
            embed_dim,
            embed_dim,
            dtype=torch.bfloat16,
        )
        * WEIGHT_SCALE,
        "v_proj.bias": torch.randn(
            embed_dim,
            dtype=torch.bfloat16,
        )
        * BIAS_SCALE,
        # Output projection
        "out_proj.weight": torch.randn(
            embed_dim,
            embed_dim,
            dtype=torch.bfloat16,
        )
        * WEIGHT_SCALE,
        "out_proj.bias": torch.randn(
            embed_dim,
            dtype=torch.bfloat16,
        )
        * BIAS_SCALE,
    }

    return weights
