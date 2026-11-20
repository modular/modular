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
import pytest
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
WIDTH = 2 * HEAD_DIM + D_REL + 1
# Each head's rows round up to a multiple of 8, zero-filled.
PADDED_WIDTH = 16
TOKENS = 3
# The deployed degree: every rank's block has one shape, so it compiles once.
TP_DEGREES = [4]

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


def _head_major(weights: Weights, heads: int) -> NDArray[np.float32]:
    """The expected head-major interleave of one set of projections."""
    parts = [
        weights["wq_du"].reshape(heads, HEAD_DIM, HIDDEN),
        weights["wr_du"].reshape(heads, D_REL, HIDDEN),
        weights["gate_attn"].reshape(heads, HEAD_DIM, HIDDEN),
        weights["attn_sink_proj"].reshape(heads, 1, HIDDEN),
        np.zeros((heads, PADDED_WIDTH - WIDTH, HIDDEN), np.float32),
    ]
    return np.concatenate(parts, axis=1).reshape(heads * PADDED_WIDTH, HIDDEN)


def test_projection_names_stay_native() -> None:
    """The fused layer loads the checkpoint's four projection names."""
    proj = _build(_random_weights(np.random.default_rng(0)), HEADS)
    names = {name for name, _ in proj.parameters}
    assert names == {
        "wq_du.weight",
        "wr_du.weight",
        "gate_attn.weight",
        "attn_sink_proj.weight",
    }


def test_fused_weight_is_head_major() -> None:
    weights = _random_weights(np.random.default_rng(0))
    proj = _build(weights, HEADS)
    np.testing.assert_array_equal(
        proj.fused_weight.to_numpy(), _head_major(weights, HEADS)
    )


def test_inkling2_heads_pad_to_aligned_rows() -> None:
    """Inkling 2's 273 rows per head pad to 280, so 12 heads per rank align."""
    proj = _QuerySideProjection(
        hidden=HIDDEN, num_heads=12, head_dim=128, d_rel=16
    )
    assert proj.width == 273
    assert proj.padded_width == 280
    # A rank's bf16 output row is 16-byte aligned, as the SM100 matmul needs.
    assert proj.num_heads * proj.padded_width * 2 % 16 == 0


def test_an_aligned_width_takes_no_padding() -> None:
    """A width already a multiple of 8 projects with no pad segment."""
    rng = np.random.default_rng(3)
    d_rel = 7  # 2 * 4 + 7 + 1 = 16 rows per head
    with default_dtype(DType.float32):
        proj = _QuerySideProjection(
            hidden=HIDDEN, num_heads=HEADS, head_dim=HEAD_DIM, d_rel=d_rel
        )
    weights = {
        "wq_du": rng.standard_normal((HEADS * HEAD_DIM, HIDDEN)),
        "wr_du": rng.standard_normal((HEADS * d_rel, HIDDEN)),
        "gate_attn": rng.standard_normal((HEADS * HEAD_DIM, HIDDEN)),
        "attn_sink_proj": rng.standard_normal((HEADS, HIDDEN)),
    }
    for name, weight in weights.items():
        getattr(proj, name).weight = Tensor(
            weight.astype(np.float32), device=CPU()
        )
    assert proj.padded_width == proj.width == 16
    x = rng.standard_normal((TOKENS, HIDDEN)).astype(np.float32)
    q, r, gate, sink = proj(Tensor(x, device=CPU()))
    np.testing.assert_allclose(
        q.to_numpy().reshape(TOKENS, -1),
        x @ weights["wq_du"].T,
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        r.to_numpy().reshape(TOKENS, -1),
        x @ weights["wr_du"].T,
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        gate.to_numpy().reshape(TOKENS, -1),
        x @ weights["gate_attn"].T,
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        sink.to_numpy(), x @ weights["attn_sink_proj"].T, rtol=1e-5, atol=1e-5
    )


def test_fused_projection_matches_per_projection() -> None:
    """One GEMM over the fused weight reproduces the four projections."""
    rng = np.random.default_rng(1)
    weights = _random_weights(rng)
    x_np = rng.standard_normal((TOKENS, HIDDEN)).astype(np.float32)
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


@pytest.mark.parametrize("num_devices", TP_DEGREES)
def test_rank_rows_are_that_ranks_own_projection_shards(
    num_devices: int,
) -> None:
    """Each rank's contiguous rows are its own shards, not a slice of a part.

    A plain ``[q; r; gate; sink]`` row concat would give a rank
    ``heads * width / num_devices`` rows of ``q`` alone. Head-major rows make
    a rank's block exactly the interleave of the shards it already holds, so
    the fused weight can be built from column-parallel sharded weights.
    """
    weights = _random_weights(np.random.default_rng(2))
    fused = _head_major(weights, HEADS)
    heads_per_rank = HEADS // num_devices
    rows_per_rank = heads_per_rank * PADDED_WIDTH

    for rank in range(num_devices):
        first_head = rank * heads_per_rank
        shard = {
            name: weight.reshape(HEADS, -1, HIDDEN)[
                first_head : first_head + heads_per_rank
            ].reshape(-1, HIDDEN)
            for name, weight in weights.items()
        }
        rank_proj = _build(shard, heads_per_rank)
        np.testing.assert_array_equal(
            rank_proj.fused_weight.to_numpy(),
            fused[rank * rows_per_rank : (rank + 1) * rows_per_rank],
        )
