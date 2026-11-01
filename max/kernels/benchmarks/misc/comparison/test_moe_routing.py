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

import numpy as np
import pytest
from moe_routing import (
    SF_MN_GROUP_SIZE,
    Routing,
    dispatch_layout,
    pick_rank,
    rank_counts,
    routed_counts,
    routed_topk,
    scale_rows_capacity,
)


@pytest.mark.parametrize("routing", list(Routing))
@pytest.mark.parametrize("num_tokens", [1, 7, 61])
def test_each_token_picks_distinct_experts(
    routing: Routing, num_tokens: int
) -> None:
    topk = routed_topk(num_tokens, 8, 32, routing, ep_size=4)

    assert topk.shape == (num_tokens, 8)
    assert topk.dtype == np.int32
    assert ((topk >= 0) & (topk < 32)).all()
    for row in topk:
        assert len(set(row.tolist())) == 8
    assert routed_counts(topk, 32).sum() == num_tokens * 8


@pytest.mark.parametrize("num_tokens", [1, 5, 13, 48])
def test_balanced_spreads_experts_and_ranks(num_tokens: int) -> None:
    # 13 tokens * 3 picks = 39 rows over 12 experts and 4 ranks: three
    # leftover rows, which a plain round-robin would give to rank 0.
    counts = routed_counts(
        routed_topk(num_tokens, 3, 12, Routing.BALANCED, ep_size=4), 12
    )
    rank_totals = rank_counts(counts, 4).sum(axis=1)

    assert counts.max() - counts.min() <= 1
    assert rank_totals.max() - rank_totals.min() <= 1


def test_random_routings_are_seeded() -> None:
    for routing in (Routing.UNIFORM, Routing.SKEWED):
        a = routed_topk(64, 8, 256, routing, seed=3)
        assert (a == routed_topk(64, 8, 256, routing, seed=3)).all()
        assert (a != routed_topk(64, 8, 256, routing, seed=4)).any()


def test_skewed_routing_has_hot_experts() -> None:
    uniform = routed_counts(routed_topk(192, 8, 256, Routing.UNIFORM), 256)
    skewed = routed_counts(routed_topk(192, 8, 256, Routing.SKEWED), 256)
    mean = 192 * 8 / 256

    assert skewed.max() > 10 * mean
    assert skewed.max() > 2 * uniform.max()


def test_top_k_above_num_experts_raises() -> None:
    with pytest.raises(ValueError):
        routed_topk(4, 9, 8, Routing.UNIFORM)


def test_rank_counts_needs_an_even_split() -> None:
    with pytest.raises(ValueError):
        rank_counts(np.zeros(10, dtype=np.int64), 4)


def test_pick_rank() -> None:
    # Rank totals: 5, 9, 1, 3.
    counts = np.array([2, 3, 4, 5, 0, 1, 1, 2])

    assert pick_rank(counts, 4, "busiest") == 1
    assert pick_rank(counts, 4, "median") == 3
    assert pick_rank(counts, 4, "2") == 2


def test_dispatch_layout_starts_each_expert_on_a_fresh_block() -> None:
    counts = np.array([0, 1, SF_MN_GROUP_SIZE, SF_MN_GROUP_SIZE + 1, 300])
    layout = dispatch_layout(counts)

    assert layout.expert_start.tolist() == [0, 0, 1, 129, 258, 558]
    assert layout.expert_start.dtype == np.uint32
    assert layout.a_scale_offsets.dtype == np.uint32
    assert layout.expert_ids.tolist() == [0, 1, 2, 3, 4]
    assert layout.num_rows == 558
    first_block = (
        layout.expert_start[:-1].astype(np.int64) // SF_MN_GROUP_SIZE
        + layout.a_scale_offsets
    )
    # An empty expert takes no block; the others take ceil(count / 128).
    assert first_block.tolist() == [0, 0, 1, 2, 4]
    assert layout.num_scale_blocks == 7


def test_scale_rows_capacity_pads_one_block_per_expert() -> None:
    assert scale_rows_capacity(65536, 32) == 65536 // SF_MN_GROUP_SIZE + 32
