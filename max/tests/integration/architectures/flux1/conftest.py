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
Fixtures for flux1 tests, including config, generated input tensors, and dummy
weights.
"""


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
    # 1024x1024 generation: 4096 image tokens
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
    hidden_dim = (
        flux_config["num_attention_heads"] * flux_config["attention_head_dim"]
    )
    inner_dim = hidden_dim
    head_dim = flux_config["attention_head_dim"]

    # Weight statistics from the attention layer of the 3rd transformer block in FLUX.1-dev
    # Format: {weight_name: (shape_tuple, std, mean)}
    WEIGHT_STATS: dict[str, tuple[tuple[int, ...], float, float]] = {
        "norm_q.weight": ((head_dim,), 0.2969, 0.8555),
        "norm_k.weight": ((head_dim,), 0.2061, 0.8047),
        "norm_added_q.weight": ((head_dim,), 0.0762, 0.6875),
        "norm_added_k.weight": ((head_dim,), 0.1719, 0.7383),
        "to_q.weight": ((inner_dim, hidden_dim), 0.0320, 0),
        "to_q.bias": ((inner_dim,), 0.0530, 0),
        "to_k.weight": ((inner_dim, hidden_dim), 0.0311, 0),
        "to_k.bias": ((inner_dim,), 0.0654, 0),
        "to_v.weight": ((inner_dim, hidden_dim), 0.0226, 0),
        "to_v.bias": ((inner_dim,), 0.0039, 0),
        "to_out.0.weight": ((hidden_dim, inner_dim), 0.0300, 0),
        "to_out.0.bias": ((hidden_dim,), 0.0195, 0),
        "add_q_proj.weight": ((inner_dim, hidden_dim), 0.0361, 0),
        "add_q_proj.bias": ((inner_dim,), 0.0405, 0),
        "add_k_proj.weight": ((inner_dim, hidden_dim), 0.0364, 0),
        "add_k_proj.bias": ((inner_dim,), 0.0605, 0),
        "add_v_proj.weight": ((inner_dim, hidden_dim), 0.0269, 0),
        "add_v_proj.bias": ((inner_dim,), 0.0280, 0),
        "to_add_out.weight": ((hidden_dim, inner_dim), 0.0349, 0),
        "to_add_out.bias": ((hidden_dim,), 0.0201, 0),
    }

    weights = {}
    for key, (shape, std, mean) in WEIGHT_STATS.items():
        weights[key] = (
            torch.randn(shape, dtype=torch.bfloat16).to("cuda") * std + mean
        )

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

    # Flux uses full head dimension for rotary embeddings
    # Generated in float32 for numerical precision
    cos = torch.randn(total_seq_len, head_dim).to(torch.float32).to("cuda")
    sin = torch.randn(total_seq_len, head_dim).to(torch.float32).to("cuda")

    return (cos, sin)
