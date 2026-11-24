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
"""Sharding and no-padding tests for the Inkling 2 fused q-side projection.

Split from test_query_side_projection so each target stays under the CI
time cap: every case compiles its own host graphs.
"""

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
WIDTH = 2 * HEAD_DIM + D_REL + 1
# Each head's rows round up to a multiple of 8, zero-filled.
PADDED_WIDTH = 16
TOKENS = 3
# The deployed degree.
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


def test_rank_shards_and_unpadded_width_in_one_graph() -> None:
    """Rank shards keep their own rows, and an aligned width needs no pad.

    A plain ``[q; r; gate; sink]`` row concat would give a rank
    ``heads * width / num_devices`` rows of ``q`` alone. Head-major rows make
    a rank's block exactly the interleave of the shards it already holds, so
    the fused weight can be built from column-parallel sharded weights. The
    first and last ranks are checked, where an off-by-one in the split shows,
    beside a width that is already a multiple of 8 and takes no pad segment.

    Every check is built under one ``F.lazy()`` block so they realize in a
    single graph: an eager compile costs minutes on CI's CPU executors.
    """
    num_devices = TP_DEGREES[0]
    weights = _random_weights(np.random.default_rng(2))
    fused = _head_major(weights, HEADS)
    heads_per_rank = HEADS // num_devices
    rows_per_rank = heads_per_rank * PADDED_WIDTH
    ranks = sorted({0, num_devices - 1})

    rng = np.random.default_rng(3)
    d_rel = 7  # 2 * 4 + 7 + 1 = 16 rows per head
    aligned_weights = {
        "wq_du": rng.standard_normal((HEADS * HEAD_DIM, HIDDEN)),
        "wr_du": rng.standard_normal((HEADS * d_rel, HIDDEN)),
        "gate_attn": rng.standard_normal((HEADS * HEAD_DIM, HIDDEN)),
        "attn_sink_proj": rng.standard_normal((HEADS, HIDDEN)),
    }
    x = rng.standard_normal((TOKENS, HIDDEN)).astype(np.float32)

    with F.lazy():
        rank_weights = []
        for rank in ranks:
            first_head = rank * heads_per_rank
            shard = {
                name: weight.reshape(HEADS, -1, HIDDEN)[
                    first_head : first_head + heads_per_rank
                ].reshape(-1, HIDDEN)
                for name, weight in weights.items()
            }
            rank_weights.append(_build(shard, heads_per_rank).fused_weight)
        with default_dtype(DType.float32):
            aligned = _QuerySideProjection(
                hidden=HIDDEN, num_heads=HEADS, head_dim=HEAD_DIM, d_rel=d_rel
            )
        for name, weight in aligned_weights.items():
            getattr(aligned, name).weight = Tensor(
                weight.astype(np.float32), device=CPU()
            )
        q, r, gate, sink = aligned(Tensor(x, device=CPU()))

    for rank, rank_weight in zip(ranks, rank_weights, strict=True):
        np.testing.assert_array_equal(
            rank_weight.to_numpy(),
            fused[rank * rows_per_rank : (rank + 1) * rows_per_rank],
        )
    assert aligned.padded_width == aligned.width == 16
    for got, name in (
        (q.to_numpy().reshape(TOKENS, -1), "wq_du"),
        (r.to_numpy().reshape(TOKENS, -1), "wr_du"),
        (gate.to_numpy().reshape(TOKENS, -1), "gate_attn"),
        (sink.to_numpy(), "attn_sink_proj"),
    ):
        np.testing.assert_allclose(
            got, x @ aligned_weights[name].T, rtol=1e-5, atol=1e-5
        )
