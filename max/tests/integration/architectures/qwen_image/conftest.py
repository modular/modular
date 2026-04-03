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

"""Fixtures for QwenImage parity tests: config, input tensors, dummy weights."""

import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch


@pytest.fixture
def qwen_config() -> dict[str, Any]:
    """Load QwenImage configuration from testdata."""
    path = os.environ["PIPELINES_TESTDATA"]
    config_path = Path(path) / "config.json"
    with open(config_path) as file:
        return json.load(file)


@pytest.fixture
def hidden_states(qwen_config: dict[str, Any]) -> torch.Tensor:
    """Random image latent hidden states.

    Shape: (batch_size, img_seq_len, inner_dim)
    inner_dim = num_attention_heads * attention_head_dim = 24 * 128 = 3072
    """
    torch.manual_seed(42)
    inner_dim = (
        qwen_config["num_attention_heads"] * qwen_config["attention_head_dim"]
    )
    # Use 256 image tokens (small 128x128 latent, 64x64 patches)
    return torch.randn(1, 256, inner_dim).to(torch.bfloat16).to("cuda")


@pytest.fixture
def encoder_hidden_states(qwen_config: dict[str, Any]) -> torch.Tensor:
    """Random text encoder hidden states.

    Shape: (batch_size, txt_seq_len, inner_dim)
    Note: in the block, text is already projected to inner_dim (3072),
    not joint_attention_dim (3584).
    """
    torch.manual_seed(43)
    inner_dim = (
        qwen_config["num_attention_heads"] * qwen_config["attention_head_dim"]
    )
    return torch.randn(1, 64, inner_dim).to(torch.bfloat16).to("cuda")


@pytest.fixture
def temb(qwen_config: dict[str, Any]) -> torch.Tensor:
    """Random timestep embedding.

    Shape: (batch_size, inner_dim)
    """
    torch.manual_seed(44)
    inner_dim = (
        qwen_config["num_attention_heads"] * qwen_config["attention_head_dim"]
    )
    return torch.randn(1, inner_dim).to(torch.bfloat16).to("cuda")


def _compute_rope_freqs(
    seq_len: int, head_dim: int, theta: int = 10000
) -> torch.Tensor:
    """Compute complex-valued RoPE frequencies (same math as diffusers).

    Returns: complex tensor of shape [seq_len, head_dim // 2]
    """
    pos_index = torch.arange(seq_len, dtype=torch.float32)
    half_dim = head_dim // 2
    freq_base = 1.0 / torch.pow(
        theta,
        torch.arange(0, half_dim * 2, 2, dtype=torch.float32) / (half_dim * 2),
    )
    freqs = torch.outer(pos_index, freq_base)
    return torch.polar(torch.ones_like(freqs), freqs)


@pytest.fixture
def image_rotary_emb_diffusers(
    qwen_config: dict[str, Any],
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE in diffusers format: (img_freqs, txt_freqs) as complex tensors.

    img_freqs: [img_seq_len, head_dim // 2] complex
    txt_freqs: [txt_seq_len, head_dim // 2] complex
    """
    head_dim = qwen_config["attention_head_dim"]
    img_seq_len = hidden_states.shape[1]
    txt_seq_len = encoder_hidden_states.shape[1]

    img_freqs = _compute_rope_freqs(img_seq_len, head_dim).to("cuda")
    txt_freqs = _compute_rope_freqs(txt_seq_len, head_dim).to("cuda")
    return (img_freqs, txt_freqs)


@pytest.fixture
def image_rotary_emb_max(
    image_rotary_emb_diffusers: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """RoPE in MAX format: interleaved [cos, sin] freqs_cis tensor.

    Concatenated as [txt, img] to match MAX's concat order.
    freqs_cis: [txt_seq_len + img_seq_len, head_dim] float32
    Layout: [cos0, sin0, cos1, sin1, ...] per position.
    """
    img_freqs, txt_freqs = image_rotary_emb_diffusers
    # MAX concatenates text first, then image
    full_freqs = torch.cat([txt_freqs, img_freqs], dim=0)
    # Convert complex → interleaved real: [S, D//2] → [S, D]
    cos = full_freqs.real.float()  # [S, D//2]
    sin = full_freqs.imag.float()  # [S, D//2]
    freqs_cis = torch.stack([cos, sin], dim=-1).reshape(full_freqs.shape[0], -1)
    return freqs_cis.to("cuda")


@pytest.fixture
def block_weights(qwen_config: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Random weights for a single QwenImageTransformerBlock.

    Uses realistic weight statistics (std, mean) from actual model weights.
    """
    inner_dim = (
        qwen_config["num_attention_heads"] * qwen_config["attention_head_dim"]
    )
    head_dim = qwen_config["attention_head_dim"]
    mlp_hidden_dim = int(inner_dim * 4.0)

    # Format: {weight_name: (shape, std, mean)}
    WEIGHT_STATS: dict[str, tuple[tuple[int, ...], float, float]] = {
        # Per-block modulation
        "img_mod.1.weight": ((6 * inner_dim, inner_dim), 0.02, 0.0),
        "img_mod.1.bias": ((6 * inner_dim,), 0.01, 0.0),
        "txt_mod.1.weight": ((6 * inner_dim, inner_dim), 0.02, 0.0),
        "txt_mod.1.bias": ((6 * inner_dim,), 0.01, 0.0),
        # Attention - main stream
        "attn.to_q.weight": ((inner_dim, inner_dim), 0.032, 0.0),
        "attn.to_q.bias": ((inner_dim,), 0.053, 0.0),
        "attn.to_k.weight": ((inner_dim, inner_dim), 0.031, 0.0),
        "attn.to_k.bias": ((inner_dim,), 0.065, 0.0),
        "attn.to_v.weight": ((inner_dim, inner_dim), 0.023, 0.0),
        "attn.to_v.bias": ((inner_dim,), 0.004, 0.0),
        "attn.to_out.0.weight": ((inner_dim, inner_dim), 0.030, 0.0),
        "attn.to_out.0.bias": ((inner_dim,), 0.020, 0.0),
        "attn.norm_q.weight": ((head_dim,), 0.30, 0.86),
        "attn.norm_k.weight": ((head_dim,), 0.21, 0.80),
        # Attention - encoder stream
        "attn.add_q_proj.weight": ((inner_dim, inner_dim), 0.036, 0.0),
        "attn.add_q_proj.bias": ((inner_dim,), 0.041, 0.0),
        "attn.add_k_proj.weight": ((inner_dim, inner_dim), 0.036, 0.0),
        "attn.add_k_proj.bias": ((inner_dim,), 0.061, 0.0),
        "attn.add_v_proj.weight": ((inner_dim, inner_dim), 0.027, 0.0),
        "attn.add_v_proj.bias": ((inner_dim,), 0.028, 0.0),
        "attn.to_add_out.weight": ((inner_dim, inner_dim), 0.035, 0.0),
        "attn.to_add_out.bias": ((inner_dim,), 0.020, 0.0),
        "attn.norm_added_q.weight": ((head_dim,), 0.076, 0.69),
        "attn.norm_added_k.weight": ((head_dim,), 0.17, 0.74),
        # Image MLP
        "img_mlp.net.0.proj.weight": ((mlp_hidden_dim, inner_dim), 0.02, 0.0),
        "img_mlp.net.0.proj.bias": ((mlp_hidden_dim,), 0.01, 0.0),
        "img_mlp.net.2.weight": ((inner_dim, mlp_hidden_dim), 0.02, 0.0),
        "img_mlp.net.2.bias": ((inner_dim,), 0.01, 0.0),
        # Text MLP
        "txt_mlp.net.0.proj.weight": ((mlp_hidden_dim, inner_dim), 0.02, 0.0),
        "txt_mlp.net.0.proj.bias": ((mlp_hidden_dim,), 0.01, 0.0),
        "txt_mlp.net.2.weight": ((inner_dim, mlp_hidden_dim), 0.02, 0.0),
        "txt_mlp.net.2.bias": ((inner_dim,), 0.01, 0.0),
    }

    torch.manual_seed(100)
    weights = {}
    for key, (shape, std, mean) in WEIGHT_STATS.items():
        weights[key] = (
            torch.randn(shape, dtype=torch.bfloat16).to("cuda") * std + mean
        )
    return weights
