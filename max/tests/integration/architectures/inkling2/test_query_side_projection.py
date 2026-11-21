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
"""Tests for the Inkling 2 fused q/r/gate/sink projection."""

from __future__ import annotations

import numpy as np
from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.pipelines.architectures.inkling2.layers.attention import (
    _QuerySideProjection,
)
from max.tensor import Tensor, default_dtype
from numpy.typing import NDArray

HIDDEN = 16
HEADS = 8
HEAD_DIM = 4
D_REL = 2
TOKENS = 3

Weights = dict[str, NDArray[np.float32]]


def _random_weights(rng: np.random.Generator) -> Weights:
    """Per-projection weights at the checkpoint's shapes."""

    def normal(*shape: int) -> NDArray[np.float32]:
        return rng.standard_normal(shape).astype(np.float32)

    return {
        "wq_du": normal(HEADS * HEAD_DIM, HIDDEN),
        "wr_du": normal(HEADS * D_REL, HIDDEN),
        "gate_attn": normal(HEADS * HEAD_DIM, HIDDEN),
        "attn_sink_proj": normal(HEADS, HIDDEN),
    }


def _build(weights: Weights, num_heads: int) -> _QuerySideProjection:
    """Builds the projection holding ``num_heads`` heads' worth of weights."""
    with default_dtype(DType.float32):
        proj = _QuerySideProjection(
            hidden=HIDDEN,
            num_heads=num_heads,
            head_dim=HEAD_DIM,
            d_rel=D_REL,
        )
    for name, weight in weights.items():
        getattr(proj, name).weight = Tensor(weight, device=CPU())
    return proj


def test_fused_projection_matches_per_projection() -> None:
    """One GEMM over the fused weight reproduces the four projections.

    The forward runs through the head-major, padded fused weight, so a wrong
    row layout fails here too. The module is built and run under one
    ``F.lazy()`` block so the whole check is a single graph: an eager compile
    costs minutes on CI's CPU executors.
    """
    rng = np.random.default_rng(1)
    weights = _random_weights(rng)
    x_np = rng.standard_normal((TOKENS, HIDDEN)).astype(np.float32)
    with F.lazy():
        proj = _build(weights, HEADS)
        projected = proj(Tensor(x_np, device=CPU()))
    q, r, gate, sink = (out.to_numpy() for out in projected)

    np.testing.assert_allclose(
        q,
        (x_np @ weights["wq_du"].T).reshape(TOKENS, HEADS, HEAD_DIM),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        r,
        (x_np @ weights["wr_du"].T).reshape(TOKENS, HEADS, D_REL),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        gate,
        (x_np @ weights["gate_attn"].T).reshape(TOKENS, HEADS, HEAD_DIM),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        sink, x_np @ weights["attn_sink_proj"].T, rtol=1e-6, atol=1e-6
    )
