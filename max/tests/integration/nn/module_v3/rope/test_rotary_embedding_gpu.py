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
"""Accelerator tests for the ModuleV3 common-layers ``RotaryEmbedding``."""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU, Accelerator, accelerator_count
from max.experimental.nn.common_layers.rotary_embedding import (
    RotaryEmbedding,
)
from max.experimental.tensor import Tensor
from max.graph import Dim

_N_HEADS = 2
_HEAD_DIM = 8
_THETA = 10000.0


def _reference_rope(x: np.ndarray, start_pos: int) -> np.ndarray:
    """Applies interleaved RoPE to ``x`` of shape (batch, seq, heads, dim)."""
    seq_len = x.shape[1]
    inv_freq = 1.0 / (_THETA ** (np.arange(0, _HEAD_DIM, 2) / _HEAD_DIM))
    positions = np.arange(start_pos, start_pos + seq_len)
    angles = np.outer(positions, inv_freq)[None, :, None, :]
    cos, sin = np.cos(angles), np.sin(angles)
    x_re, x_im = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x_re * cos - x_im * sin
    out[..., 1::2] = x_re * sin + x_im * cos
    return out


@pytest.mark.skipif(not accelerator_count(), reason="requires an accelerator")
def test_forward_runs_on_accelerator_input() -> None:
    """The CPU-built frequency table follows the input onto the accelerator."""
    rope = RotaryEmbedding(
        dim=_N_HEADS * _HEAD_DIM,
        n_heads=_N_HEADS,
        theta=_THETA,
        max_seq_len=16,
        device=Accelerator(),
    )
    start_pos = 3
    x_np = (
        np.random.default_rng(0)
        .standard_normal((1, 5, _N_HEADS, _HEAD_DIM))
        .astype(np.float32)
    )

    result = rope(
        Tensor.from_dlpack(x_np).to(Accelerator()), start_pos=Dim(start_pos)
    )

    assert result.device == Accelerator()
    np.testing.assert_allclose(
        np.from_dlpack(result.to(CPU())),
        _reference_rope(x_np, start_pos),
        rtol=1e-5,
        atol=1e-5,
    )
