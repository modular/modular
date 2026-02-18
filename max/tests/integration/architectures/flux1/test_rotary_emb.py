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

from typing import Any

import pytest
import torch
from diffusers.models.embeddings import (
    apply_rotary_emb as diffusers_apply_rotary_emb,
)
from max.pipelines.architectures.flux1.layers.embeddings import (
    apply_rotary_emb as max_apply_rotary_emb,
)
from max.tensor import Tensor
from torch.utils.dlpack import from_dlpack

# query : torch.Size([1, 4608, 24, 128]), torch.bfloat16
# key : torch.Size([1, 4608, 24, 128]), torch.bfloat16
# image_rotary_emb[0] : torch.Size([4608, 128]), torch.float32
# image_rotary_emb[1] : torch.Size([4608, 128]), torch.float32


@pytest.fixture
def query_tensor(flux_config: dict[str, Any]) -> torch.Tensor:
    """Generate random query tensor for rotary embedding test.

    Shape: [B, S, H, D] (batch, sequence, heads, head_dim)
    Both diffusers and max use this shape format.
    For Flux1 dev:
    - batch_size: 1
    - sequence_len: 4608 (512 text + 4096 image tokens)
    - num_heads: 24
    - head_dim: 128
    """
    torch.manual_seed(45)
    batch_size = 1
    num_heads = flux_config["num_attention_heads"]
    head_dim = flux_config["attention_head_dim"]
    # Total sequence length for dual-stream attention
    seq_len = 4608  # 512 text + 4096 image

    return (
        torch.randn(batch_size, seq_len, num_heads, head_dim)
        .to(torch.bfloat16)
        .to("cuda")
    )


@pytest.fixture
def rotary_emb(
    flux_config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate rotary position embeddings (cos, sin tensors).

    Shape: [S, D] where S is sequence length and D is head_dim
    """
    torch.manual_seed(46)
    head_dim = flux_config["attention_head_dim"]
    seq_len = 4608  # Total sequence length

    cos = torch.randn(seq_len, head_dim).to(torch.float32).to("cuda")
    sin = torch.randn(seq_len, head_dim).to(torch.float32).to("cuda")

    return (cos, sin)


@torch.no_grad()
def generate_torch_outputs(
    query_tensor: torch.Tensor,
    rotary_emb: tuple[torch.Tensor, torch.Tensor],
    sequence_dim: int = 2,
) -> torch.Tensor:
    """Generates the outputs of the PyTorch (diffusers) apply_rotary_emb function.

    Args:
        query_tensor: Input query tensor [B, S, H, D].
        rotary_emb: Tuple of (cos, sin) rotary embeddings [S, D].
        sequence_dim: Dimension representing the sequence (1 or 2).

    Returns:
        Output tensor from the PyTorch apply_rotary_emb function.
    """
    output = diffusers_apply_rotary_emb(
        x=query_tensor,
        freqs_cis=rotary_emb,
        use_real=True,
        use_real_unbind_dim=-1,  # Used for flux
        sequence_dim=sequence_dim,
    )

    return output


def generate_max_outputs(
    query_tensor: torch.Tensor,
    rotary_emb: tuple[torch.Tensor, torch.Tensor],
    sequence_dim: int = 2,
) -> torch.Tensor:
    """Generates the outputs of the MAX apply_rotary_emb function.

    Args:
        query_tensor: Input query tensor [B, S, H, D] (same shape as diffusers).
        rotary_emb: Tuple of (cos, sin) rotary embeddings [S, D].
        sequence_dim: Dimension representing the sequence (1 or 2).

    Returns:
        Output buffer from the MAX apply_rotary_emb function.
    """
    # Create MAX tensors - both diffusers and max use [B, S, H, D] shape
    query_max = Tensor.from_dlpack(query_tensor.cuda())
    cos_max = Tensor.from_dlpack(rotary_emb[0].cuda())
    sin_max = Tensor.from_dlpack(rotary_emb[1].cuda())

    # Apply rotary embedding
    output_max = max_apply_rotary_emb(
        x=query_max,
        freqs_cis=(cos_max, sin_max),
        sequence_dim=sequence_dim,
    )

    # Convert back to torch
    output_torch = from_dlpack(output_max).to(torch.bfloat16)

    return output_torch


def test_apply_rotary_emb_sequence_dim_1(
    query_tensor: torch.Tensor,
    rotary_emb: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Test that MAX apply_rotary_emb matches PyTorch (diffusers) apply_rotary_emb output
    with sequence_dim=1 (used in Flux attention).

    Args:
        query_tensor: Input query tensor [B, S, H, D].
        rotary_emb: Tuple of (cos, sin) rotary embeddings [S, D].
    """
    torch_output = generate_torch_outputs(
        query_tensor, rotary_emb, sequence_dim=1
    )

    max_output = generate_max_outputs(query_tensor, rotary_emb, sequence_dim=1)

    # Compare outputs
    torch.testing.assert_close(
        torch_output.to(torch.bfloat16),
        max_output.to(torch.bfloat16),
        rtol=2 * torch.finfo(torch.bfloat16).eps,
        atol=8 * torch.finfo(torch.bfloat16).eps,
    )
