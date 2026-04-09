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
"""Tests for Gemma4 multi-dimensional RoPE (vision encoder path)."""

from __future__ import annotations

import torch
from conftest import (  # type: ignore[import-not-found]
    VISION_HEAD_DIM,
    torch_apply_multidimensional_rope,
)
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, TensorValue
from max.pipelines.architectures.gemma4.layers.rotary_embedding import (
    apply_multidimensional_rope,
)


def _freqs_cis_to_per_dim_cos_sin(
    freqs_cis: torch.Tensor, ndim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Converts MAX-style ``freqs_cis`` ``[..., head_dim // 2, 2]`` to
    HF-style ``cos`` / ``sin`` ``[..., head_dim]`` that are compatible with
    per-dim splitting by ``torch_apply_multidimensional_rope``.

    Each spatial dim gets ``channels_per_dim`` activation channels and
    ``channels_per_dim // 2`` frequency components.  The HF rotate-half
    convention duplicates the cos/sin for each dim so that the full tensors
    have shape ``[..., head_dim]``.
    """
    # freqs_cis: [..., head_dim // 2, 2]
    half_dim = freqs_cis.shape[-2]
    freqs_per_dim = half_dim // ndim

    cos_parts = []
    sin_parts = []
    for k in range(ndim):
        start = k * freqs_per_dim
        end = start + freqs_per_dim
        c = freqs_cis[..., start:end, 0]  # [..., freqs_per_dim]
        s = freqs_cis[..., start:end, 1]  # [..., freqs_per_dim]
        # HF rotate-half expects duplicated halves per dim chunk
        cos_parts.append(torch.cat([c, c], dim=-1))  # [..., channels_per_dim]
        sin_parts.append(torch.cat([s, s], dim=-1))  # [..., channels_per_dim]

    return torch.cat(cos_parts, dim=-1), torch.cat(sin_parts, dim=-1)


def test_matches_torch_reference() -> None:
    """MAX apply_multidimensional_rope must match the torch reference."""
    batch = 2
    seq_len = 16
    n_heads = 8
    head_dim = VISION_HEAD_DIM
    ndim = 2

    torch.manual_seed(42)
    x_torch = torch.randn(batch, seq_len, n_heads, head_dim)

    # Generate freqs_cis as ground truth: [batch, seq_len, head_dim // 2, 2]
    freqs_cis_torch = torch.randn(batch, seq_len, head_dim // 2, 2)

    # Derive HF-compatible per-dim cos/sin from freqs_cis
    cos_hf, sin_hf = _freqs_cis_to_per_dim_cos_sin(freqs_cis_torch, ndim)

    # Unsqueeze for broadcast over heads: [batch, seq_len, 1, head_dim]
    cos_hf_bcast = cos_hf.unsqueeze(2)
    sin_hf_bcast = sin_hf.unsqueeze(2)

    # Torch reference result
    ref = torch_apply_multidimensional_rope(
        x_torch, cos_hf_bcast, sin_hf_bcast, ndim
    )

    # Unsqueeze for broadcast over heads: [batch, seq_len, 1, head_dim // 2, 2]
    freqs_cis_bcast = freqs_cis_torch.unsqueeze(2)

    # Run through MAX graph
    device = CPU()
    session = InferenceSession(devices=[device])
    with Graph(
        "test_md_rope",
        input_types=[
            TensorType(DType.float32, shape=list(x_torch.shape), device=device),
            TensorType(
                DType.float32,
                shape=list(freqs_cis_bcast.shape),
                device=device,
            ),
        ],
    ) as graph:
        x_in, fc_in = graph.inputs
        assert isinstance(x_in, TensorValue)
        assert isinstance(fc_in, TensorValue)
        result = apply_multidimensional_rope(x_in, fc_in, ndim=ndim)
        graph.output(result)

    compiled = session.load(graph, weights_registry={})
    (buf,) = compiled.execute(x_torch.numpy(), freqs_cis_bcast.numpy())
    max_result = torch.from_dlpack(buf).cpu().float()

    torch.testing.assert_close(max_result, ref, rtol=1e-5, atol=1e-5)
    assert max_result.shape == x_torch.shape


def test_ndim_1_matches_standard_rope() -> None:
    """With ndim=1, multidimensional RoPE should equal standard 1D RoPE."""
    torch.manual_seed(42)
    batch, seq_len, n_heads, head_dim = 1, 8, 4, 64

    x_torch = torch.randn(batch, seq_len, n_heads, head_dim)

    # Generate freqs_cis as ground truth: [batch, seq_len, head_dim // 2, 2]
    freqs_cis_torch = torch.randn(batch, seq_len, head_dim // 2, 2)

    # Derive HF-compatible cos/sin (ndim=1 gives standard duplicated halves)
    cos_hf, sin_hf = _freqs_cis_to_per_dim_cos_sin(freqs_cis_torch, ndim=1)

    # Standard 1D reference (rotate_half convention)
    cos_bcast = cos_hf.unsqueeze(2)
    sin_bcast = sin_hf.unsqueeze(2)
    x1 = x_torch[..., : head_dim // 2]
    x2 = x_torch[..., head_dim // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    ref = (x_torch * cos_bcast) + (rotated * sin_bcast)

    # MAX multidimensional with ndim=1
    freqs_cis_bcast = freqs_cis_torch.unsqueeze(2)

    device = CPU()
    session = InferenceSession(devices=[device])
    with Graph(
        "test_md_rope_ndim1",
        input_types=[
            TensorType(DType.float32, shape=list(x_torch.shape), device=device),
            TensorType(
                DType.float32,
                shape=list(freqs_cis_bcast.shape),
                device=device,
            ),
        ],
    ) as graph:
        x_in, fc_in = graph.inputs
        assert isinstance(x_in, TensorValue)
        assert isinstance(fc_in, TensorValue)
        result = apply_multidimensional_rope(x_in, fc_in, ndim=1)
        graph.output(result)

    compiled = session.load(graph, weights_registry={})
    (buf,) = compiled.execute(x_torch.numpy(), freqs_cis_bcast.numpy())
    max_result = torch.from_dlpack(buf).cpu().float()

    torch.testing.assert_close(max_result, ref, rtol=1e-5, atol=1e-5)
