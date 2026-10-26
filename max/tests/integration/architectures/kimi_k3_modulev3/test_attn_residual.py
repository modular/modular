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
"""Numerical verification for the attention-residual mixture: Kimi K3's only
residual stream.

Verifies the ``attn_res_mix`` custom kernel's output against an independent
numpy softmax-mixture reference.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Accelerator
from max.dtype import DType
from max.experimental.tensor import Tensor, default_device
from max.pipelines.architectures.kimi_k3_modulev3.layers.attn_residual import (
    kimi_k3_attn_res_mix,
)


def _numpy_attn_res_mix(
    candidates: np.ndarray,
    proj_weight: np.ndarray,
    norm_weight: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Independent reference: RMS-normalize each candidate, score against
    norm_weight*proj_weight, softmax over the candidate axis, weighted sum
    over the RAW candidates (not the normalized ones -- this is the specific
    thing V2's own docstring warns a port gets backwards)."""
    c64 = candidates.astype(np.float64)  # [T, num_candidates, H]
    rms = np.sqrt(np.mean(c64**2, axis=-1, keepdims=True) + eps)
    normed = (c64 / rms) * norm_weight.astype(np.float64)
    scores = normed @ proj_weight.astype(np.float64).reshape(-1)  # [T, N]
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return np.sum(c64 * weights[..., None], axis=1)


@pytest.mark.compile_on_gpu("attn_res_mix has no custom.declare declaration")
def test_attn_res_mix_matches_independent_reference() -> None:
    rng = np.random.default_rng(42)
    eps = 1e-5
    seq_len, hidden, num_candidates = 4, 16, 3
    candidates_np = rng.standard_normal(
        (seq_len, num_candidates, hidden)
    ).astype(np.float32)
    proj_np = rng.standard_normal((1, hidden)).astype(np.float32)
    norm_np = rng.standard_normal((hidden,)).astype(np.float32)
    expected = _numpy_attn_res_mix(candidates_np, proj_np, norm_np, eps)

    with default_device(Accelerator()):
        prefix_sum = Tensor(candidates_np[:, -1, :], dtype=DType.float32).cast(
            DType.bfloat16
        )
        stack = [
            Tensor(candidates_np[:, i, :], dtype=DType.float32).cast(
                DType.bfloat16
            )
            for i in range(num_candidates - 1)
        ]
        proj = Tensor(proj_np, dtype=DType.float32).cast(DType.bfloat16)
        norm = Tensor(norm_np, dtype=DType.float32).cast(DType.bfloat16)
        out = kimi_k3_attn_res_mix(prefix_sum, stack, proj, norm, eps)
        out_np = out.cast(DType.float32).to_numpy().astype(np.float64)

    rel_error = np.max(np.abs(out_np - expected)) / max(
        np.max(np.abs(expected)), 1e-12
    )
    assert rel_error < 1e-2, f"attn_res_mix diverges: rel_error={rel_error}"
