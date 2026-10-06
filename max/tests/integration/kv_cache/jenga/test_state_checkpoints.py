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
"""Tests for recurrent-state checkpoints in the Jenga block manager."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

import numpy as np
import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheGroupId,
    KVLeafRegion,
    PagedKVLeafRegion,
    RecurrentStateParams,
    RecurrentStateRegion,
)
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.kv_cache import InsufficientBlocksError
from max.pipelines.kv_cache.paged_kv_cache.block_utils import (
    LittleKVCacheBlock,
)
from max.pipelines.kv_cache.paged_kv_cache.jenga_block_manager import (
    JengaBlockManager,
    KVLeafInfo,
    create_groups,
    create_pools,
)
from max.pipelines.kv_cache.paged_kv_cache.recurrent_coordinator import (
    RecurrentKVGroupCoordinator,
)
from max.pipelines.request.base import RequestID

BLOCK_SIZE = 4

FULL = "full"
CONV = "state/conv"
REC = "state/rec"


STATE_PARAMS_REGIONS = (
    RecurrentStateRegion(
        leaf_id=CONV, num_layers=2, row_shape=(4,), dtype=DType.bfloat16
    ),
    RecurrentStateRegion(
        leaf_id=REC, num_layers=3, row_shape=(2,), dtype=DType.bfloat16
    ),
)
"""Two state leaves with *different* layer counts."""

STATE_PARAMS = RecurrentStateParams(
    devices=[DeviceRef.CPU()], regions=STATE_PARAMS_REGIONS
)
"""The tree the group factory reads its geometry out of."""


STATE_LEAVES = {
    FULL: KVLeafInfo(1, KVCacheGroupId.full()),
    CONV: KVLeafInfo(1, KVCacheGroupId.recurrent()),
    REC: KVLeafInfo(1, KVCacheGroupId.recurrent()),
}


def state_groups(bm: JengaBlockManager) -> list[RecurrentKVGroupCoordinator]:
    """The coordinators under test, one per state leaf.

    The state tree has two recurrent leaves (conv and rec), and each is its
    own coordinator now, so what used to be one group's answer is the union
    of theirs.
    """
    groups = [
        group
        for group in bm.groups.values()
        if isinstance(group, RecurrentKVGroupCoordinator)
    ]
    assert groups, "no recurrent leaf in this manager"
    return groups


def live(bm: JengaBlockManager, ctx: TextContext) -> dict[str, int] | None:
    """The blocks this request's recurrence runs in, drawing them if new."""
    merged: dict[str, int] = {}
    for group in state_groups(bm):
        blocks = group.live_blocks(ctx.request_id)
        if blocks is None:
            # One leaf without a live block means the state is not drawn,
            # which is what a single group reported for the whole tree.
            return None
        merged.update(blocks)
    return merged


def resume(
    bm: JengaBlockManager, ctx: TextContext
) -> Mapping[str, tuple[int | None, int]]:
    """The block each leaf's next forward resumes its state from."""
    merged: dict[str, tuple[int | None, int]] = {}
    for group in state_groups(bm):
        merged.update(group.resume(ctx, 0))
    return merged


def checkpoint(
    bm: JengaBlockManager, ctx: TextContext
) -> Mapping[str, tuple[int | None, int]]:
    """The blocks the forward that just ran should be copied between."""
    merged: dict[str, tuple[int | None, int]] = {}
    for group in state_groups(bm):
        merged.update(group.checkpoint(ctx, 0))
    return merged


def make_manager(
    num_huge_blocks: int = 999,
    *,
    enable_prefix_caching: bool = True,
) -> JengaBlockManager:
    pools = create_pools(STATE_LEAVES, num_huge_blocks)
    return JengaBlockManager(
        pools=pools,
        block_size=BLOCK_SIZE,
        enable_prefix_caching=enable_prefix_caching,
        groups=create_groups(
            STATE_LEAVES,
            pools,
            BLOCK_SIZE,
            enable_prefix_caching=enable_prefix_caching,
        ),
        leaves=STATE_PARAMS.leaves(),
    )


def make_ctx(num_tokens: int, *, offset: int = 0) -> TextContext:
    """A request whose prompt is ``num_tokens`` tokens from ``offset`` up."""
    return TextContext(
        request_id=RequestID(),
        max_length=4096,
        tokens=TokenBuffer(
            np.arange(offset, offset + num_tokens, dtype=np.int64)
        ),
    )


def forward(bm: JengaBlockManager, ctx: TextContext, token: int = 42) -> None:
    """Runs one step the way the pipeline does."""
    bm.alloc(ctx)
    resume(bm, ctx)
    ctx.update(token)
    # The checkpoint runs before the commit, so the copy reads a block nothing
    # has published or freed.
    checkpoint(bm, ctx)
    bm.step(ctx)


def published(bm: JengaBlockManager) -> list[bytes]:
    return list(bm.pools[0].prefix_caches[CONV])


# ===--------------------------------------------------------------------=== #
# The blocks a forward runs in
# ===--------------------------------------------------------------------=== #


def test_a_claimed_request_runs_in_blocks_of_every_state_leaf() -> None:
    bm = make_manager()
    ctx = make_ctx(8)
    bm.claim(ctx)
    bm.alloc(ctx)

    blocks = live(bm, ctx)
    assert blocks is not None
    assert set(blocks) == {CONV, REC}


def test_a_fresh_request_resumes_from_a_wiped_block() -> None:
    # A drawn block holds whatever its last request wrote, and the kernels
    # read the incoming state unconditionally, so it has to be zeroed.
    bm = make_manager()
    ctx = make_ctx(8)
    bm.claim(ctx)
    bm.alloc(ctx)

    runs_in = live(bm, ctx)
    assert runs_in is not None
    assert resume(bm, ctx) == {
        leaf_id: (None, runs_in[leaf_id]) for leaf_id in runs_in
    }, "no source block means the rows are wiped"


def test_a_request_that_has_run_carries_its_own_state_forward() -> None:
    bm = make_manager()
    ctx = make_ctx(8)
    bm.claim(ctx)
    forward(bm, ctx)

    assert resume(bm, ctx) == {}, (
        "a second forward continues in the block the first one wrote"
    )


def test_the_blocks_are_kept_across_a_forward_that_publishes_nothing() -> None:
    # Drawn once and written in place; only publishing moves a request on.
    bm = make_manager()
    ctx = make_ctx(8)
    bm.claim(ctx)
    bm.alloc(ctx)
    first = live(bm, ctx)

    bm.alloc(ctx)

    assert live(bm, ctx) == first


def test_a_pool_with_nothing_to_spare_refuses_the_request() -> None:
    # A state block holds the only copy of something no recomputation can
    # rebuild, so there is nothing to degrade to.
    bm = make_manager(num_huge_blocks=3)
    ctx = make_ctx(8)
    bm.claim(ctx)

    with pytest.raises(InsufficientBlocksError):
        bm.alloc(ctx)


# ===--------------------------------------------------------------------=== #
# Publishing and reuse
# ===--------------------------------------------------------------------=== #


def test_cached_kv_without_a_state_is_not_reusable() -> None:
    # One cache_length serves every layer, so KV the recurrence has no state
    # for cannot be resumed from.
    bm = make_manager()
    first = make_ctx(16)
    bm.claim(first)
    forward(bm, first)
    for block in list(bm.pools[0].prefix_caches[CONV].values()):
        bm.pools[0].uncommit_block(block)
    bm.release(first)

    second = make_ctx(16)
    bm.claim(second)
    bm.alloc(second)

    assert second.cached_prefix_length == 0


def test_a_published_state_licenses_a_hit_at_its_block() -> None:
    bm = make_manager()
    first = make_ctx(16)
    bm.claim(first)
    for _ in range(4):
        forward(bm, first)
    bm.release(first)
    assert published(bm)

    second = make_ctx(24)
    bm.claim(second)
    bm.alloc(second)

    # The first request ran on past its prompt, so the boundary at 16 was
    # committed by a later step and the repeat resumes at it.
    assert second.cached_prefix_length == 16


def test_resuming_reads_the_published_state_where_it_lies() -> None:
    # The matched blocks may be matched again, so the request reads them and
    # writes its own rather than writing theirs.
    bm = make_manager()
    first = make_ctx(16)
    bm.claim(first)
    for _ in range(4):
        forward(bm, first)
    bm.release(first)

    second = make_ctx(24)
    bm.claim(second)
    bm.alloc(second)

    resumed = resume(bm, second)
    assert resumed, "a hit must give the forward a block to copy in"
    runs_in = live(bm, second)
    assert runs_in is not None
    for leaf_id, (src, dst) in resumed.items():
        assert src is not None, "the hit named a block to read"
        assert dst == runs_in[leaf_id]
        assert src != dst, (
            "the matched blocks may be matched again, so the request copies"
            " them into its own rather than writing theirs"
        )


def test_a_forward_publishes_kv_but_not_state() -> None:
    # A state is published at a boundary chosen for it, not by every forward.
    bm = make_manager()
    ctx = make_ctx(16)
    bm.claim(ctx)
    bm.alloc(ctx)
    ctx.update(42)
    bm.step(ctx)

    assert bm.pools[0].prefix_caches[FULL], "KV should commit every block"
    assert not published(bm), "a forward should publish no state"


def test_a_checkpoint_moves_the_request_onto_a_successor() -> None:
    # The block that ran holds the state its boundary hash names, so it is
    # published where it lies and the request continues elsewhere.
    bm = make_manager()
    ctx = make_ctx(16)
    bm.claim(ctx)
    bm.alloc(ctx)
    before = live(bm, ctx)
    ctx.update(42)
    checkpoint(bm, ctx)

    assert live(bm, ctx) != before


# ===--------------------------------------------------------------------=== #
# Checkpoints
# ===--------------------------------------------------------------------=== #


def test_a_forward_ending_on_a_boundary_checkpoints() -> None:
    # The block that ran holds the state at the boundary the forward ended
    # on, which is the one boundary anyone can name.
    bm = make_manager()
    ctx = make_ctx(16)
    bm.claim(ctx)
    bm.alloc(ctx)
    ctx.update(42)

    copies = checkpoint(bm, ctx)
    assert set(copies) == {region.leaf_id for region in STATE_PARAMS_REGIONS}
    leaves = STATE_PARAMS.leaves()
    for leaf_id, (src, dst) in copies.items():
        assert src is not None, "a checkpoint always names a block to copy"
        assert src != dst, "a successor is a block of its own"
        rows = leaves[leaf_id].bound_row_copies(src, dst)
        for src_rows, dst_rows in rows.values():
            assert len(src_rows) == len(dst_rows), (
                "a block folds to the same rows either way"
            )


def test_a_forward_ending_mid_block_rotates_nothing() -> None:
    # Shorter than one block: there is no boundary behind it to name.
    bm = make_manager()
    ctx = make_ctx(3)
    bm.claim(ctx)
    bm.alloc(ctx)
    ctx.update(42)

    assert checkpoint(bm, ctx) == {}


def test_a_forward_ending_past_a_boundary_checkpoints_nothing() -> None:
    # Past a boundary and off it; the case above is num_blocks zero.
    bm = make_manager()
    ctx = make_ctx(18)
    bm.claim(ctx)
    bm.alloc(ctx)
    ctx.update(42)

    assert ctx.tokens.processed_length % BLOCK_SIZE != 0, "fixture must be off"
    assert ctx.tokens.processed_length // BLOCK_SIZE > 0, "and past a boundary"
    assert checkpoint(bm, ctx) == {}


def test_one_checkpoint_is_outstanding_at_a_time() -> None:
    # The row holds a published block until its boundary is committed.
    bm = make_manager()
    ctx = make_ctx(16)
    bm.claim(ctx)
    bm.alloc(ctx)
    ctx.update(42)

    assert checkpoint(bm, ctx) != {}
    assert checkpoint(bm, ctx) == {}


def test_the_block_published_is_the_one_the_forward_ran_in() -> None:
    # Nothing is handed to the cache that the request is still writing: the
    # successor it moves to is untouched, and the block it left is not.
    bm = make_manager()
    ctx = make_ctx(16)
    bm.claim(ctx)
    bm.alloc(ctx)
    ran_in = live(bm, ctx)
    assert ran_in is not None
    ctx.update(42)

    copies = checkpoint(bm, ctx)

    for leaf_id, (src, _) in copies.items():
        assert src == ran_in[leaf_id]


def test_a_checkpoint_is_published_by_the_step_that_reaches_it() -> None:
    # The checkpoint moves the block out of the slot the recurrence runs in
    # before the commit scans the row, so the boundary it names is
    # committable straight away rather than a step later.
    bm = make_manager()
    ctx = make_ctx(16)
    bm.claim(ctx)
    forward(bm, ctx)

    assert published(bm), "the step that reaches the boundary names it"


def test_a_boundary_reached_while_generating_is_checkpointed() -> None:
    # A chat turn's prompt is the last turn's plus its answer, so what is
    # generated here is the prefix the next turn extends.
    bm = make_manager()
    ctx = make_ctx(6)
    bm.claim(ctx)
    forward(bm, ctx)

    kept = 0
    for _ in range(10):
        forward(bm, ctx)
        kept = len(published(bm))

    assert kept >= 2, "generation keeps advancing the boundary it publishes"


def test_nothing_is_published_when_prefix_caching_is_off() -> None:
    # The blocks a recurrence runs in are still drawn: it cannot run without
    # them.
    bm = make_manager(enable_prefix_caching=False)
    ctx = make_ctx(16)
    bm.claim(ctx)
    for _ in range(4):
        forward(bm, ctx)

    assert live(bm, ctx) is not None
    assert not published(bm)


def test_admission_prices_the_successor_a_boundary_draws() -> None:
    # The pool holds two requests at one state block each. A checkpoint takes
    # a second block, so a batch admitted without it finds the pool empty at
    # the boundary, in the step, where nothing can requeue it.
    bm = make_manager(num_huge_blocks=7)
    admitted: list[TextContext] = []
    for idx in range(2):
        ctx = make_ctx(BLOCK_SIZE, offset=idx * BLOCK_SIZE)
        bm.claim(ctx)
        try:
            bm.alloc(ctx)
        except InsufficientBlocksError:
            continue  # the scheduler requeues what admission refuses
        admitted.append(ctx)
    assert admitted, "the pool must hold at least one request"

    for ctx in admitted:
        resume(bm, ctx)
        ctx.update(42)
        assert checkpoint(bm, ctx), "the forward must end on a boundary"
        bm.step(ctx)


def held_state_blocks(bm: JengaBlockManager, ctx: TextContext) -> int:
    """Returns the state blocks the request holds, its successors included."""
    held = 0
    for group in state_groups(bm):
        row = group.rows.get(ctx.request_id)
        if row is not None:
            held += sum(not block.is_null for block in row[group.leaf_id])
        held += ctx.request_id in group.successors
    return held


@pytest.mark.parametrize("caching", [True, False])
def test_a_state_holds_its_successor_from_admission_only_with_caching(
    caching: bool,
) -> None:
    # With caching on each state also holds the block its next checkpoint
    # continues in, whether or not this forward reaches a boundary.
    bm = make_manager(enable_prefix_caching=caching)
    ctx = make_ctx(BLOCK_SIZE - 1)
    bm.claim(ctx)

    bm.alloc(ctx)

    per_state = 2 if caching else 1
    assert held_state_blocks(bm, ctx) == per_state * len(state_groups(bm))


def test_a_state_never_holds_more_than_its_two_budgeted_blocks() -> None:
    # A checkpoint takes the successor, and admission does not replace it
    # until the checkpoint is published, so the budget's two blocks hold
    # across a boundary. Release gives every one back.
    bm = make_manager()
    free_before = bm.huge_block_count().free
    ctx = make_ctx(BLOCK_SIZE)
    bm.claim(ctx)
    budget = 2 * len(state_groups(bm))
    for _ in range(3 * BLOCK_SIZE):
        bm.alloc(ctx)
        resume(bm, ctx)
        assert held_state_blocks(bm, ctx) <= budget
        ctx.update(42)
        checkpoint(bm, ctx)
        bm.step(ctx)

    bm.release(ctx)
    assert bm.huge_block_count().free == free_before


def test_release_commits_nothing() -> None:
    # A checkpoint is committed by a later step, so one still outstanding at
    # release is never published.
    bm = make_manager()
    ctx = make_ctx(16)
    bm.claim(ctx)
    forward(bm, ctx)
    before = len(published(bm))

    bm.release(ctx)

    assert len(published(bm)) == before


# ===--------------------------------------------------------------------=== #
# Checkpointing while the newest token is a placeholder
# ===--------------------------------------------------------------------=== #

PROMPT_TOKENS = 16
"""Long enough that generated tokens reach a published block."""


def overlap_forward(
    bm: JengaBlockManager, ctx: TextContext, token: int = 42
) -> None:
    """Runs one step the way the overlap pipeline does.

    The token the forward is producing is not sampled yet, so the context
    carries a placeholder until it is realized. Everything the cache does
    happens while that placeholder is the newest token.
    """
    bm.alloc(ctx)
    resume(bm, ctx)
    ctx.update_with_future_token()
    checkpoint(bm, ctx)
    bm.step(ctx)
    ctx.realize_future_token(token)


def test_a_placeholder_newest_token_publishes_the_hashes_a_real_one_does() -> (
    None
):
    # A placeholder stands in for a token nobody has sampled, so if its value
    # reached a block hash the two pipelines would file the same prefix under
    # different keys and neither would ever reuse the other's state.
    realized = make_manager()
    first = make_ctx(PROMPT_TOKENS)
    realized.claim(first)
    for _ in range(20):
        forward(realized, first)

    placeholder = make_manager()
    second = make_ctx(PROMPT_TOKENS)
    placeholder.claim(second)
    for _ in range(20):
        overlap_forward(placeholder, second)

    # The first block published is the prompt's last, so the comparison only
    # covers a *generated* token once more than that many have landed. Stop
    # short and it compares prompt hashes, which agree however badly the
    # newest token is handled.
    assert len(published(realized)) > PROMPT_TOKENS // BLOCK_SIZE
    assert published(placeholder) == published(realized)


def test_a_state_published_under_a_placeholder_licenses_a_hit() -> None:
    # The checkpoint is what makes a state reusable, and it runs while the
    # placeholder is newest, so this is the reuse the overlap guard forbids.
    bm = make_manager()
    first = make_ctx(16)
    bm.claim(first)
    for _ in range(4):
        overlap_forward(bm, first)
    bm.release(first)
    assert published(bm)

    second = make_ctx(24)
    bm.claim(second)
    bm.alloc(second)

    assert second.cached_prefix_length == 16
    resumed = resume(bm, second)
    assert resumed, "a hit must give the forward a block to copy in"
    for src, _ in resumed.values():
        assert src is not None, "the hit named a block to read"


# ============================================================================
# The run every group accepts at once
# ============================================================================

HYBRID_SLIDING = "sliding"

HYBRID_LEAVES = {
    FULL: KVLeafInfo(1, KVCacheGroupId.full()),
    HYBRID_SLIDING: KVLeafInfo(
        1, KVCacheGroupId("sliding_window", 3 * BLOCK_SIZE)
    ),
    CONV: KVLeafInfo(1, KVCacheGroupId.recurrent()),
    REC: KVLeafInfo(1, KVCacheGroupId.recurrent()),
}
"""All three kinds of group in one manager."""


def hybrid_leaves() -> dict[str, KVLeafRegion]:
    """The paged leaves alongside the state ones the params object owns."""
    paged = {
        leaf_id: PagedKVLeafRegion(
            leaf_id=leaf_id,
            group_id=info.group_id,
            bytes_per_page=1,
            page_size=BLOCK_SIZE,
        )
        for leaf_id, info in HYBRID_LEAVES.items()
        if not info.group_id.is_recurrent()
    }
    return {**paged, **STATE_PARAMS.leaves()}


def make_hybrid_manager(num_huge_blocks: int = 999) -> JengaBlockManager:
    pools = create_pools(HYBRID_LEAVES, num_huge_blocks)
    return JengaBlockManager(
        pools=pools,
        block_size=BLOCK_SIZE,
        groups=create_groups(
            HYBRID_LEAVES, pools, BLOCK_SIZE, enable_prefix_caching=True
        ),
        leaves=hybrid_leaves(),
    )


def publish(bm: JengaBlockManager, leaf_id: str, keys: Sequence[bytes]) -> None:
    """Commits one block per key in a single leaf, then drops the references."""
    pool = bm.pools[0]
    blocks = [pool.alloc_block(leaf_id) for _ in keys]
    for block, key in zip(blocks, keys, strict=True):
        pool.commit_into_prefix_cache(key, block)
    for block in reversed(blocks):
        pool.free_block(block)


@pytest.mark.parametrize("seed", range(16))
def test_a_hybrid_hit_settles_on_the_deepest_run_every_group_accepts(
    seed: int,
) -> None:
    """The settling loop finds the deepest run no group shortens.

    The expected run is the deepest prefix every group accepts at once,
    computed by brute force since the loop's own answer is under test.
    """
    rng = random.Random(seed)
    for _ in range(200):
        bm = make_hybrid_manager()
        keys = [f"k{idx}".encode() for idx in range(rng.randint(0, 10))]
        # Each leaf publishes its own subset, so the groups disagree about
        # how deep a hit they can serve.
        for leaf_id in HYBRID_LEAVES:
            publish(
                bm,
                leaf_id,
                [key for key in keys if rng.random() < 0.7],
            )

        groups = list(bm.groups.values())
        expected = max(
            n
            for n in range(len(keys), -1, -1)
            if all(g.longest_cache_hit(keys[:n], 0) == n for g in groups)
        )
        assert (
            bm._find_longest_device_prefix_cache_hit(keys, 0, False) == expected
        ), f"seed={seed} keys={len(keys)}"


# ===--------------------------------------------------------------------=== #
# The checkpoint an external tier is copying in
# ===--------------------------------------------------------------------=== #


def splice_landing_state(
    bm: JengaBlockManager, ctx: TextContext, depth: int
) -> dict[str, LittleKVCacheBlock]:
    """Hands each state leaf a checkpoint a connector is still loading."""
    pool = bm.pools[0]
    landing: dict[str, LittleKVCacheBlock] = {}
    for group in state_groups(bm):
        leaf_id = group.leaf_id
        block = pool.alloc_block(leaf_id)
        row = [pool.null_little_blocks[leaf_id]] * (depth - 1) + [block]
        group.extend(ctx.request_id, {leaf_id: []}, {leaf_id: row}, 0)
        landing[leaf_id] = block
    ctx.tokens.skip_processing(depth * BLOCK_SIZE)
    return landing


def test_a_state_still_landing_is_not_the_block_the_recurrence_runs_in() -> (
    None
):
    """A loading checkpoint is resumed from, never run in."""
    bm = make_manager()
    ctx = make_ctx(3 * BLOCK_SIZE)
    bm.claim(ctx)
    landing = splice_landing_state(bm, ctx, depth=2)

    bm.alloc(ctx)

    runs_in = live(bm, ctx)
    assert runs_in is not None
    for leaf_id, block in landing.items():
        assert runs_in[leaf_id] != block.bid, (
            f"{leaf_id} would run in the block its hit is still copying into"
        )
    # The scheduler holds the request out of the batch until the copy lands.
    assert resume(bm, ctx) == {
        leaf_id: (block.bid, runs_in[leaf_id])
        for leaf_id, block in landing.items()
    }


def test_a_landing_state_holds_up_no_successor() -> None:
    """A loading checkpoint does not stop admission drawing the successor."""
    bm = make_manager()
    ctx = make_ctx(3 * BLOCK_SIZE)
    bm.claim(ctx)
    landing = splice_landing_state(bm, ctx, depth=2)
    bm.alloc(ctx)
    # Landing block, live block and the successor, as a device hit holds.
    assert held_state_blocks(bm, ctx) == 3 * len(state_groups(bm))

    # The forward runs to the next boundary, so it checkpoints.
    resume(bm, ctx)
    ctx.update(42)
    fills = checkpoint(bm, ctx)
    bm.step(ctx)

    assert set(fills) == set(landing), "the boundary found no successor"
    for group in state_groups(bm):
        assert landing[group.leaf_id].block_hash is not None
        assert ctx.request_id not in group.loading


def test_a_row_cut_back_forgets_the_state_that_was_landing() -> None:
    """A rolled-back hit frees the landing block; its copy keeps its own pin."""
    bm = make_manager()
    ctx = make_ctx(3 * BLOCK_SIZE)
    bm.claim(ctx)
    landing = splice_landing_state(bm, ctx, depth=2)

    for group in state_groups(bm):
        group.shrink_to_fit(ctx.request_id, 0, 0)

    for group in state_groups(bm):
        assert ctx.request_id not in group.loading
    assert all(block.ref_cnt == 0 for block in landing.values())
    # The request starts over as a plain miss.
    ctx.tokens.rewind_processing(2 * BLOCK_SIZE)
    bm.alloc(ctx)
    assert live(bm, ctx) is not None


def test_an_external_tier_is_asked_for_the_state_at_the_hits_depth() -> None:
    """A hit loads one state block and claims only the deepest hash."""
    bm = make_manager()
    keys = [f"k{idx}".encode() for idx in range(5)]
    for group in state_groups(bm):
        assert group.blocks_held_of_connector_hit(5) == 1
        assert group.blocks_held_of_connector_hit(1) == 1
        assert group.blocks_held_of_connector_hit(0) == 0
        assert list(group.claimable_hashes(keys)) == keys[-1:]
        assert list(group.claimable_hashes([])) == []


# A speculative step commits 1 to K+1 tokens, so with prefix caching on a
# recurrent group keeps publishing checkpoints at the lagging length the
# overlap pipeline hands it.

_SPEC_STRIDES = (1, 2, 3, 4) * 10
"""Tokens committed per step, cycling the counts a K=3 verify produces."""


def forward_n(bm: JengaBlockManager, ctx: TextContext, num_tokens: int) -> bool:
    """Runs one forward of ``num_tokens`` and returns whether it checkpointed."""
    bm.alloc(ctx)
    resume(bm, ctx)
    for _ in range(num_tokens):
        ctx.update(42)
    checkpointed = checkpoint(bm, ctx) != {}
    bm.step(ctx)
    return checkpointed


def count_checkpoints(strides: Sequence[int]) -> int:
    """Returns how many of ``strides`` forwards published a checkpoint."""
    bm = make_manager()
    ctx = make_ctx(BLOCK_SIZE)
    bm.claim(ctx)
    fired = 0
    for num_tokens in strides:
        fired += forward_n(bm, ctx, num_tokens)
    return fired


def test_a_speculative_stride_keeps_checkpointing_while_caching_is_on() -> None:
    assert count_checkpoints(_SPEC_STRIDES) > 1
