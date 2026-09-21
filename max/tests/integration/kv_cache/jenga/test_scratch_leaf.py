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
"""Tests for the scratch cache group: one never-published block per request."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheGroupId,
    KVLeafRegion,
    PagedKVLeafRegion,
    RecurrentKVLeafRegion,
    RecurrentStateParams,
    RecurrentStateRegion,
    ScratchKVLeafRegion,
)
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.kv_cache import InsufficientBlocksError
from max.pipelines.kv_cache.paged_kv_cache.jenga_block_manager import (
    JengaBlockManager,
    KVLeafInfo,
    create_groups,
    create_pools,
)
from max.pipelines.kv_cache.paged_kv_cache.recurrent_coordinator import (
    RecurrentKVGroupCoordinator,
)
from max.pipelines.kv_cache.paged_kv_cache.scratch_coordinator import (
    ScratchKVGroupCoordinator,
)
from max.pipelines.request.base import RequestID

BLOCK_SIZE = 4

FULL = "full"
REC = "state/rec"
RING = "state/ring"

STATE_PARAMS = RecurrentStateParams(
    devices=[DeviceRef.CPU()],
    regions=(
        RecurrentStateRegion(
            leaf_id=REC, num_layers=2, row_shape=(4,), dtype=DType.bfloat16
        ),
        RecurrentStateRegion(
            leaf_id=RING,
            num_layers=2,
            row_shape=(8, 4),
            dtype=DType.bfloat16,
            scratch=True,
        ),
    ),
)
"""A published state, and beside it a ring of 8 captured inputs per layer."""

LEAVES = {
    FULL: KVLeafInfo(1, KVCacheGroupId.full()),
    REC: KVLeafInfo(1, KVCacheGroupId.recurrent()),
    RING: KVLeafInfo(1, KVCacheGroupId.scratch()),
}


def leaf_regions() -> dict[str, KVLeafRegion]:
    """The paged leaf, plus the two the state tree declares."""
    return {
        FULL: PagedKVLeafRegion(
            leaf_id=FULL,
            group_id=KVCacheGroupId.full(),
            bytes_per_page=1,
            page_size=BLOCK_SIZE,
        ),
        **STATE_PARAMS.leaves(),
    }


def make_manager(
    num_huge_blocks: int = 999, *, enable_prefix_caching: bool = True
) -> JengaBlockManager:
    pools = create_pools(LEAVES, num_huge_blocks)
    return JengaBlockManager(
        pools=pools,
        block_size=BLOCK_SIZE,
        enable_prefix_caching=enable_prefix_caching,
        groups=create_groups(LEAVES, pools, BLOCK_SIZE),
        leaves=leaf_regions(),
    )


def make_ctx(num_tokens: int, *, offset: int = 0) -> TextContext:
    return TextContext(
        request_id=RequestID(),
        max_length=4096,
        tokens=TokenBuffer(
            np.arange(offset, offset + num_tokens, dtype=np.int64)
        ),
    )


def scratch_group(bm: JengaBlockManager) -> ScratchKVGroupCoordinator:
    group = bm.groups[RING]
    assert isinstance(group, ScratchKVGroupCoordinator)
    return group


def state_group(bm: JengaBlockManager) -> RecurrentKVGroupCoordinator:
    group = bm.groups[REC]
    assert isinstance(group, RecurrentKVGroupCoordinator)
    return group


def ring_block(bm: JengaBlockManager, ctx: TextContext) -> int | None:
    """The block id the request's ring lives in, or None before it grows."""
    blocks = scratch_group(bm).rows[ctx.request_id][RING]
    return blocks[0].bid if blocks else None


def state_block(bm: JengaBlockManager, ctx: TextContext) -> int | None:
    runs_in = state_group(bm).live_blocks(ctx.request_id)
    return None if runs_in is None else runs_in[REC]


def forward(bm: JengaBlockManager, ctx: TextContext, token: int = 42) -> None:
    """Runs one step the way the pipeline does."""
    bm.alloc(ctx)
    state_group(bm).resume(ctx, 0)
    ctx.update(token)
    state_group(bm).checkpoint(ctx, 0)
    bm.step(ctx)


# ===--------------------------------------------------------------------=== #
# One block, held for the request's life
# ===--------------------------------------------------------------------=== #


def test_a_request_draws_exactly_one_ring_block() -> None:
    bm = make_manager()
    ctx = make_ctx(8)
    bm.claim(ctx)
    bm.alloc(ctx)

    assert len(scratch_group(bm).rows[ctx.request_id][RING]) == 1


def test_the_ring_block_survives_the_boundaries_the_state_rotates_across() -> (
    None
):
    # The point of the group: a checkpoint moves the state onto a successor,
    # and the ring is the one thing that must not move with it.
    bm = make_manager()
    ctx = make_ctx(16)
    bm.claim(ctx)
    bm.alloc(ctx)
    first_ring = ring_block(bm, ctx)
    # Seeded with the pre-rotation block: only the prompt-consuming forward
    # lands on a boundary, so sampling after each forward alone would record
    # one block eight times and pass an empty test.
    state_blocks: set[int | None] = {state_block(bm, ctx)}

    for _ in range(8):
        forward(bm, ctx)
        state_blocks.add(state_block(bm, ctx))
        assert ring_block(bm, ctx) == first_ring

    assert len(state_blocks) > 1, "the state has to rotate for this to mean it"


def test_a_ring_block_is_not_the_state_block_it_rides_beside() -> None:
    """Each leaf draws from its own free list, so the two rows are unrelated.

    Load-bearing for a kernel's arguments: a scratch pool takes its own slot
    index, because indexing it by the state's row lands on the page the pool
    handed to whatever cache owns that huge block.
    """
    bm = make_manager()
    ctx = make_ctx(8)
    bm.claim(ctx)
    bm.alloc(ctx)

    assert ring_block(bm, ctx) != state_block(bm, ctx)


def test_growing_again_draws_no_second_ring_block() -> None:
    bm = make_manager()
    ctx = make_ctx(16)
    bm.claim(ctx)
    bm.alloc(ctx)
    first = ring_block(bm, ctx)

    bm.alloc(ctx)

    row = scratch_group(bm).rows[ctx.request_id][RING]
    assert len(row) == 1
    assert row[0].bid == first


def test_release_gives_the_ring_block_back() -> None:
    # More requests in sequence than the pool could hold at once, so a ring
    # block that was never freed would run it dry.
    bm = make_manager(num_huge_blocks=16)

    for _ in range(100):
        ctx = make_ctx(8)
        bm.claim(ctx)
        forward(bm, ctx)
        assert ring_block(bm, ctx) is not None
        bm.release(ctx)
        assert ctx.request_id not in scratch_group(bm).rows


# ===--------------------------------------------------------------------=== #
# Invisible to everything keyed on a hash
# ===--------------------------------------------------------------------=== #


def test_the_ring_is_never_published() -> None:
    bm = make_manager()
    ctx = make_ctx(16)
    bm.claim(ctx)
    for _ in range(4):
        forward(bm, ctx)

    assert bm.pools[0].prefix_caches[RING] == {}
    assert bm.pools[0].prefix_caches[FULL], (
        "the paged leaf has to publish for this to mean anything"
    )


def test_a_scratch_group_does_not_shorten_a_hit() -> None:
    # A group that answered 0 would take the settling loop to nothing and
    # silently disable prefix caching for the whole model.
    bm = make_manager()
    first = make_ctx(16)
    bm.claim(first)
    for _ in range(4):
        forward(bm, first)
    bm.release(first)

    second = make_ctx(24)
    bm.claim(second)
    bm.alloc(second)

    assert second.cached_prefix_length == 16


def test_the_scratch_group_claims_nothing_from_a_hit() -> None:
    bm = make_manager()
    keys: Sequence[bytes] = [b"k0", b"k1"]

    assert scratch_group(bm).claimable_hashes(keys) == ()
    assert scratch_group(bm).claim_hit_blocks(keys, 0) == {RING: []}
    assert scratch_group(bm).longest_cache_hit(keys, 0) == len(keys)
    assert scratch_group(bm).blocks_held_of_connector_hit(len(keys)) == 0


def test_the_connector_is_not_told_about_a_scratch_leaf() -> None:
    bm = make_manager()

    assert RING in bm._leaf_ids
    assert RING not in bm._cacheable_leaf_ids
    assert all(
        not group.group_id.is_scratch()
        for group in bm._cacheable_groups.values()
    )


# ===--------------------------------------------------------------------=== #
# Allocation and admission still cover it
# ===--------------------------------------------------------------------=== #


def test_the_ring_block_is_counted_before_a_request_is_admitted() -> None:
    # Admission prices every leaf, so the ring competes for the budget rather
    # than being a carve-out beside it.
    bm = make_manager()

    assert leaf_regions()[RING].blocks_to_reserve(99) == 1
    assert bm._blocks_to_reserve(16)[RING] == 1


def test_a_pool_with_nothing_to_spare_refuses_the_request() -> None:
    bm = make_manager(num_huge_blocks=3)
    ctx = make_ctx(16)
    bm.claim(ctx)

    with pytest.raises(InsufficientBlocksError):
        bm.alloc(ctx)


def test_the_forward_is_handed_the_one_block_the_ring_lives_in() -> None:
    bm = make_manager()
    ctx = make_ctx(8)
    bm.claim(ctx)
    bm.alloc(ctx)

    plans = scratch_group(bm).forward_blocks([ctx], [2])

    assert plans == {RING: [[ring_block(bm, ctx)]]}


def test_a_forward_before_alloc_is_a_wiring_bug_not_a_silent_zero() -> None:
    bm = make_manager()
    ctx = make_ctx(8)
    bm.claim(ctx)

    with pytest.raises(ValueError, match="alloc must run"):
        scratch_group(bm).forward_blocks([ctx], [1])


# ===--------------------------------------------------------------------=== #
# The group id
# ===--------------------------------------------------------------------=== #


def test_a_scratch_group_has_no_window() -> None:
    with pytest.raises(ValueError, match="no window"):
        KVCacheGroupId.scratch().blocks_in_window(BLOCK_SIZE)


def test_a_scratch_group_rejects_a_window_size() -> None:
    with pytest.raises(ValueError, match="must be -1 for scratch groups"):
        KVCacheGroupId(type="scratch", window_size=8)


def test_a_scratch_leaf_is_not_cacheable_and_the_others_are() -> None:
    regions = leaf_regions()

    assert not regions[RING].cacheable
    assert regions[FULL].cacheable
    assert regions[REC].cacheable


def test_the_region_flag_is_what_puts_the_leaf_in_the_scratch_group() -> None:
    # The tree decides, so an arch declares a ring by marking a region rather
    # than reaching for a leaf class at the call site.
    leaves = STATE_PARAMS.leaves()

    assert isinstance(leaves[RING], ScratchKVLeafRegion)
    assert leaves[RING].group_id == KVCacheGroupId.scratch()
    assert leaves[RING].blocks_to_reserve(99) == 1
    assert isinstance(leaves[REC], RecurrentKVLeafRegion)
    assert leaves[REC].group_id == KVCacheGroupId.recurrent()
    assert leaves[REC].blocks_to_reserve(99) == 2


def test_a_scratch_region_still_declares_its_pool_and_row_table() -> None:
    # The ring rides the row-addressed pipeline unchanged: the kernel is given
    # a pool to index and one row per layer, as a published state is.
    (per_device,) = STATE_PARAMS.get_symbolic_inputs()
    ring = per_device.leaves[1]

    assert list(ring.pool.shape)[1:] == [8, 4]
    assert list(ring.live_row_ids.shape)[0] == 2
