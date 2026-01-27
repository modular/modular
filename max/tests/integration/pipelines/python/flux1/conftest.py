# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from typing import Any
import json
import os
from pathlib import Path

import pytest
import torch

"""
Fixtures for flux1 tests, including config, generated input tensors, and dummy
weights.
"""

# Flux1 dev 1K×1K T2I generation input dimensions:
# hidden_states : torch.Size([1, 4096, 3072]), torch.bfloat16
# encoder_hidden_states : torch.Size([1, 512, 3072]), torch.bfloat16
# attention_mask : None
# image_rotary_emb [0]: torch.Size([4608, 128]), torch.float32
# image_rotary_emb [1]: torch.Size([4608, 128]), torch.float32


@pytest.fixture
def flux_config() -> dict[str, Any]:
    """Load Flux configuration from testdata."""
    path = os.environ["PIPELINES_TESTDATA"]
    config_path = Path(path) / "config.json"
    with open(config_path) as file:
        return json.load(file)


@pytest.fixture
def input_tensor(flux_config: dict[str, Any]) -> torch.Tensor:
    """Generate random input tensor (image latents) for testing.

    Shape: (batch_size, seq_len, hidden_dim)
    where hidden_dim = num_attention_heads * attention_head_dim
    """
    torch.manual_seed(42)
    hidden_dim = (
        flux_config["num_attention_heads"] * flux_config["attention_head_dim"]
    )
    # 1K×1K generation: 4096 image tokens
    return torch.randn(1, 4096, hidden_dim).to(torch.bfloat16).to("cuda")


@pytest.fixture
def encoder_hidden_states(flux_config: dict[str, Any]) -> torch.Tensor:
    """Generate random encoder hidden states (text embeddings) for testing.

    Shape: (batch_size, text_seq_len, hidden_dim)
    """
    torch.manual_seed(43)
    hidden_dim = (
        flux_config["num_attention_heads"] * flux_config["attention_head_dim"]
    )
    # T5 encoder: 512 text tokens
    return torch.randn(1, 512, hidden_dim).to(torch.bfloat16).to("cuda")


@pytest.fixture
def attention_weights(flux_config: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Generate random weights for FluxAttention layer"""

    # Weight statistics from the attention layer of the 3rd transformer block in FLUX.1-dev
    WEIGHT_STATS: dict[str, tuple[float, float]] = {
        "norm_q.weight": (0.2969, 0.8555),
        "norm_k.weight": (0.2061, 0.8047),
        "norm_added_q.weight": (0.0762, 0.6875),
        "norm_added_k.weight": (0.1719, 0.7383),
        "to_q.weight": (0.0320, 0),
        "to_q.bias": (0.0530, 0),
        "to_k.weight": (0.0311, 0),
        "to_k.bias": (0.0654, 0),
        "to_v.weight": (0.0226, 0),
        "to_v.bias": (0.0039, 0),
        "to_out.0.weight": (0.0300, 0),
        "to_out.0.bias": (0.0195, 0),
        "add_q_proj.weight": (0.0361, 0),
        "add_q_proj.bias": (0.0405, 0),
        "add_k_proj.weight": (0.0364, 0),
        "add_k_proj.bias": (0.0605, 0),
        "add_v_proj.weight": (0.0269, 0),
        "add_v_proj.bias": (0.0280, 0),
        "to_add_out.weight": (0.0349, 0),
        "to_add_out.bias": (0.0201, 0),
    }

    hidden_dim = (
        flux_config["num_attention_heads"] * flux_config["attention_head_dim"]
    )
    inner_dim = hidden_dim
    head_dim = flux_config["attention_head_dim"]

    weights = {
        # Main image stream projections (with bias)
        "to_q.weight": torch.randn(
            inner_dim, hidden_dim,
            dtype=torch.bfloat16,
        ) 
        * WEIGHT_STATS["to_q.weight"][0] + WEIGHT_STATS["to_q.weight"][1],
        "to_q.bias": torch.randn(
            inner_dim,
            dtype=torch.bfloat16,
        ) 
        * WEIGHT_STATS["to_q.bias"][0] + WEIGHT_STATS["to_q.bias"][1],
        "to_k.weight": torch.randn(
            inner_dim, hidden_dim,
            dtype=torch.bfloat16,
        ) 
        * WEIGHT_STATS["to_k.weight"][0] + WEIGHT_STATS["to_k.weight"][1],
        "to_k.bias": torch.randn(
            inner_dim,
            dtype=torch.bfloat16,
        ) 
        * WEIGHT_STATS["to_k.bias"][0] + WEIGHT_STATS["to_k.bias"][1],
        "to_v.weight": torch.randn(
            inner_dim, hidden_dim,
            dtype=torch.bfloat16,
        ) 
        * WEIGHT_STATS["to_v.weight"][0] + WEIGHT_STATS["to_v.weight"][1],
        "to_v.bias": torch.randn(
            inner_dim,
            dtype=torch.bfloat16,
        ) 
        * WEIGHT_STATS["to_v.bias"][0] + WEIGHT_STATS["to_v.bias"][1],

        # Q/K normalization
        "norm_q.weight": torch.randn(
            head_dim,
            dtype=torch.bfloat16,
        ) 
        * WEIGHT_STATS["norm_q.weight"][0] + WEIGHT_STATS["norm_q.weight"][1],
        "norm_k.weight": torch.randn(
            head_dim,
            dtype=torch.bfloat16,
        ) 
        * WEIGHT_STATS["norm_k.weight"][0] + WEIGHT_STATS["norm_k.weight"][1],

        # Output projection (with bias)
        "to_out.0.weight": torch.randn(
            hidden_dim, inner_dim,
            dtype=torch.bfloat16,
        ) 
        * WEIGHT_STATS["to_out.0.weight"][0] + WEIGHT_STATS["to_out.0.weight"][1],
        "to_out.0.bias": torch.randn(
            hidden_dim,
            dtype=torch.bfloat16,
        ) 
        * WEIGHT_STATS["to_out.0.bias"][0] + WEIGHT_STATS["to_out.0.bias"][1],

        # Encoder (text) stream projections for dual-stream attention (with bias)
        "add_q_proj.weight": torch.randn(
            inner_dim, hidden_dim,
            dtype=torch.bfloat16,
        ) 
        * WEIGHT_STATS["add_q_proj.weight"][0] + WEIGHT_STATS["add_q_proj.weight"][1],
        "add_q_proj.bias": torch.randn(
            inner_dim,
            dtype=torch.bfloat16,
        ) 
        * WEIGHT_STATS["add_q_proj.bias"][0] + WEIGHT_STATS["add_q_proj.bias"][1],
        "add_k_proj.weight": torch.randn(
            inner_dim, hidden_dim,
            dtype=torch.bfloat16,
        ) 
        * WEIGHT_STATS["add_k_proj.weight"][0] + WEIGHT_STATS["add_k_proj.weight"][1],
        "add_k_proj.bias": torch.randn(
            inner_dim,
            dtype=torch.bfloat16,
        ) 
        * WEIGHT_STATS["add_k_proj.bias"][0] + WEIGHT_STATS["add_k_proj.bias"][1],
        "add_v_proj.weight": torch.randn(
            inner_dim, hidden_dim,
            dtype=torch.bfloat16,
        ) 
        * WEIGHT_STATS["add_v_proj.weight"][0] + WEIGHT_STATS["add_v_proj.weight"][1],
        "add_v_proj.bias": torch.randn(
            inner_dim,
            dtype=torch.bfloat16,
        ) 
        * WEIGHT_STATS["add_v_proj.bias"][0] + WEIGHT_STATS["add_v_proj.bias"][1],

        # Encoder Q/K normalization
        "norm_added_q.weight": torch.randn(
            head_dim,
            dtype=torch.bfloat16,
        ) 
        * WEIGHT_STATS["norm_added_q.weight"][0] + WEIGHT_STATS["norm_added_q.weight"][1],
        "norm_added_k.weight": torch.randn(
            head_dim,
            dtype=torch.bfloat16,
        ) 
        * WEIGHT_STATS["norm_added_k.weight"][0] + WEIGHT_STATS["norm_added_k.weight"][1],

        # Encoder output projection (with bias)
        "to_add_out.weight": torch.randn(
            hidden_dim, inner_dim,
            dtype=torch.bfloat16,
        ) 
        * WEIGHT_STATS["to_add_out.weight"][0] + WEIGHT_STATS["to_add_out.weight"][1],
        "to_add_out.bias": torch.randn(
            hidden_dim,
            dtype=torch.bfloat16,
        ) 
        * WEIGHT_STATS["to_add_out.bias"][0] + WEIGHT_STATS["to_add_out.bias"][1],
    }

    return weights


@pytest.fixture
def image_rotary_emb(
    flux_config: dict[str, Any],
    input_tensor: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate rotary position embeddings (cos, sin tensors)"""
    torch.manual_seed(44)
    head_dim = flux_config["attention_head_dim"]

    # Total sequence length = encoder tokens + image tokens
    total_seq_len = encoder_hidden_states.shape[1] + input_tensor.shape[1]
    # For 1K×1K: 512 + 4096 = 4608

    # Flux uses full head dimension for rotary embeddings
    # Generated in float32 for numerical precision
    cos = torch.randn(total_seq_len, head_dim).to(torch.float32).to("cuda")
    sin = torch.randn(total_seq_len, head_dim).to(torch.float32).to("cuda")

    return (cos, sin)
