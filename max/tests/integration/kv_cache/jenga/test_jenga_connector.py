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

"""Jenga's KVConnector hookup, driven by a fake connector.

The real connectors need a GPU, so these use a stand-in that records what it
was asked for and completes only when told to -- which is what lets the
deferred publish be tested: an onloaded page must stay invisible to other
requests until its copy has actually landed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheGroupId,
    PagedKVLeafRegion,
    RecurrentStateParams,
    RecurrentStateRegion,
)
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.kv_cache import InsufficientBlocksError
from max.pipelines.kv_cache.kv_connector import (
    ByteCount,
    KVConnector,
    KVConnectorTransfer,
    KVLoadFailed,
    KVLoadRefused,
    KVTransfer,
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

FULL = "full"
VALUES = "values"
SCALES = "scales"
SLIDING = "sliding"
STATE = "state"


class FakeTransfer:
    """A transfer that reports complete only once ``synchronize`` is called.

    Setting ``fails`` makes every poll raise ``error``, as a terminal transfer
    failure does: the destination blocks hold no valid KV, and the failure is
    sticky so a second poller cannot see completion instead.
    """

    def __init__(self, g0: Mapping[str, Sequence[int]] | None = None) -> None:
        self.g0_blocks_per_leaf = {} if g0 is None else g0
        self.done = False
        self.fails = False
        self.error: Exception = KVLoadFailed("transfer failed")

    def is_complete(self) -> bool:
        if self.fails:
            raise self.error
        return self.done

    def synchronize(self) -> None:
        self.done = True


class FakeConnector:
    """Serves whatever hashes each leaf holds, and records calls."""

    name = "FakeConnector"

    def __init__(
        self,
        leaves: Mapping[str, KVCacheGroupId] | Sequence[str],
        asynchronous: bool = True,
    ) -> None:
        self.leaves = (
            dict(leaves)
            if isinstance(leaves, Mapping)
            else {leaf: KVCacheGroupId.full() for leaf in leaves}
        )
        self.held: dict[str, set[bytes]] = {leaf: set() for leaf in self.leaves}
        self.evict_before_load: set[bytes] = set()
        # Raised by the next `load` in place of posting anything.
        self.load_error: Exception | None = None
        # Whether the next load's transfer fails on its very first poll.
        self.fail_on_first_poll = False
        self.asynchronous = asynchronous
        self.lookups: list[tuple[list[bytes], int, bytes | None]] = []
        self.loads: list[
            tuple[dict[str, list[int]], dict[str, list[bytes]], int]
        ] = []
        self.offloads: list[
            tuple[dict[str, list[int]], dict[str, list[bytes]]]
        ] = []
        self.touches: list[tuple[list[bytes], int]] = []
        self.transfers: list[FakeTransfer] = []
        self.polls = 0

    def lookup(
        self,
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
        hint: bytes | None = None,
    ) -> Mapping[str, Sequence[bool]]:
        """States residency and nothing else.

        What a pattern of residency is WORTH is the manager's to work out from
        each leaf's shape, which is the point of the split.
        """
        self.lookups.append((list(block_hashes), replica_idx, hint))
        return {
            leaf_id: [h in self.held[leaf_id] for h in block_hashes]
            for leaf_id in self.leaves
        }

    def load(
        self,
        block_ids: Mapping[str, Sequence[int]],
        block_hashes: Mapping[str, Sequence[bytes]],
        replica_idx: int = 0,
        hint: bytes | None = None,
    ) -> KVTransfer:
        self.loads.append(
            (
                {leaf: list(ids) for leaf, ids in block_ids.items()},
                {leaf: list(hs) for leaf, hs in block_hashes.items()},
                replica_idx,
            )
        )
        for leaf_id, hashes in block_hashes.items():
            assert len(hashes) == len(block_ids[leaf_id]), (
                f"leaf {leaf_id!r} got {len(block_ids[leaf_id])} blocks for "
                f"{len(hashes)} hashes"
            )
            for block_hash in hashes:
                if block_hash in self.evict_before_load:
                    raise KVLoadRefused(
                        f"evicted between lookup and load: {block_hash!r}",
                        leaf_id=leaf_id,
                        block_hash=block_hash,
                    )
                assert block_hash in self.held[leaf_id], (
                    f"asked for {block_hash!r}, which lookup never reported"
                )
        if self.load_error is not None:
            raise self.load_error
        transfer = FakeTransfer()
        transfer.fails = self.fail_on_first_poll
        if not self.asynchronous:
            transfer.done = True
        self.transfers.append(transfer)
        return transfer

    def offload(
        self,
        block_ids: Mapping[str, Sequence[int]],
        block_hashes: Mapping[str, Sequence[bytes]],
        replica_idx: int = 0,
    ) -> KVConnectorTransfer:
        for leaf_id, hashes in block_hashes.items():
            assert len(hashes) == len(block_ids[leaf_id]), (
                f"leaf {leaf_id!r} got {len(block_ids[leaf_id])} blocks for "
                f"{len(hashes)} hashes"
            )
        self.offloads.append(
            (
                {leaf: list(ids) for leaf, ids in block_ids.items()},
                {leaf: list(hs) for leaf, hs in block_hashes.items()},
            )
        )
        for leaf_id, hashes in block_hashes.items():
            self.held[leaf_id].update(hashes)
        transfer = FakeTransfer(
            {leaf: list(ids) for leaf, ids in block_ids.items()},
        )
        if not self.asynchronous:
            transfer.done = True
        self.transfers.append(transfer)
        return transfer

    def touch(
        self, block_hashes: Sequence[bytes], replica_idx: int = 0
    ) -> None:
        self.touches.append((list(block_hashes), replica_idx))

    def poll_transfers(self) -> None:
        self.polls += 1

    def reset_prefix_cache(self) -> None:
        for held in self.held.values():
            held.clear()

    def hold(self, block_hashes: Iterable[bytes]) -> None:
        """Marks the hashes held by every leaf."""
        hashes = list(block_hashes)
        for held in self.held.values():
            held.update(hashes)

    def shutdown(self) -> None:
        return None

    @property
    def metrics(self) -> KVCacheMetrics:
        return KVCacheMetrics()

    def take_metrics(self) -> KVCacheMetrics:
        return self.metrics

    @property
    def host_byte_count(self) -> ByteCount:
        return ByteCount(free=1, total=2)

    @property
    def disk_byte_count(self) -> ByteCount:
        return ByteCount(free=3, total=4)


def make_ctx(tokens: Sequence[int]) -> TextContext:
    return TextContext(
        request_id=RequestID(),
        max_length=4096,
        tokens=TokenBuffer(np.array(tokens, dtype=np.int64)),
    )


def make_manager(
    connector: KVConnector | None,
    leaves: Sequence[str] = (FULL,),
    num_huge_blocks: int = 16,
    leaf_infos: Mapping[str, KVLeafInfo] | None = None,
) -> JengaBlockManager:
    leaf_infos = leaf_infos or {
        leaf: KVLeafInfo(1, KVCacheGroupId.full()) for leaf in leaves
    }
    pools = create_pools(leaf_infos, num_huge_blocks)
    return JengaBlockManager(
        pools=pools,
        groups=create_groups(leaf_infos, pools, 1, enable_prefix_caching=True),
        leaves={
            leaf_id: PagedKVLeafRegion(
                leaf_id=leaf_id,
                group_id=info.group_id,
                bytes_per_page=1,
                page_size=1,
            )
            for leaf_id, info in leaf_infos.items()
            if not info.group_id.is_recurrent()
        },
        block_size=1,
        enable_prefix_caching=True,
        max_num_input_tokens=None,
        num_draft_tokens=0,
        num_draft_tokens_per_step=0,
        connector=connector,
    )


PAGE = 4
"""Tokens per page, more than one so checkpoints are sparser than the run."""


def make_hybrid_manager(
    connector: KVConnector, num_huge_blocks: int = 64
) -> JengaBlockManager:
    """A full-attention leaf beside a recurrent state."""
    state_params = RecurrentStateParams(
        devices=[DeviceRef.CPU()],
        regions=(
            RecurrentStateRegion(
                leaf_id=STATE,
                num_layers=1,
                row_shape=(4,),
                dtype=DType.float32,
            ),
        ),
    )
    leaf_infos = {
        FULL: KVLeafInfo(1, KVCacheGroupId.full()),
        STATE: KVLeafInfo(1, KVCacheGroupId.recurrent()),
    }
    pools = create_pools(leaf_infos, num_huge_blocks)
    return JengaBlockManager(
        pools=pools,
        groups=create_groups(
            leaf_infos, pools, PAGE, enable_prefix_caching=True
        ),
        leaves={
            FULL: PagedKVLeafRegion(
                leaf_id=FULL,
                group_id=KVCacheGroupId.full(),
                bytes_per_page=1,
                page_size=PAGE,
            ),
            **state_params.leaves(),
        },
        block_size=PAGE,
        enable_prefix_caching=True,
        max_num_input_tokens=None,
        num_draft_tokens=0,
        num_draft_tokens_per_step=0,
        connector=connector,
    )


def hybrid_connector() -> FakeConnector:
    return FakeConnector(
        {FULL: KVCacheGroupId.full(), STATE: KVCacheGroupId.recurrent()}
    )


def forward_to_boundary(manager: JengaBlockManager, ctx: TextContext) -> None:
    """Runs one forward cut at a page boundary, checkpointing the state.

    The cache manager checkpoints in its ``step``; the block manager does not.
    """
    end = ctx.tokens.processed_length + ctx.tokens.active_length
    cut = end - end % PAGE - ctx.tokens.processed_length
    if 0 < cut < ctx.tokens.active_length:
        ctx.tokens.chunk(cut)
    manager.alloc(ctx)
    ctx.update(9)
    for group in manager.groups.values():
        group.checkpoint(ctx, 0)
    manager.step(ctx)


def test_offload_then_onload_defers_publish_until_the_copy_lands() -> None:
    connector = FakeConnector([FULL])
    manager = make_manager(connector)

    # A forward's committed blocks are saved out, and an async offload pins
    # its sources until the write lands.
    first = make_ctx([1, 2, 3, 4])
    manager.claim(first)
    manager.alloc(first)
    first.update(9)
    manager.step(first)
    manager.offload(0)
    assert connector.offloads, "no offload issued"
    offloaded_hashes = connector.offloads[-1][1][FULL]
    assert manager.pending_transfers_exist(0), "async offload did not pin"
    manager.release(first)

    # Land the offload: a pinned commit survives reset_prefix_cache, so the
    # device tier is only truly empty once those pins are gone.
    for pending in connector.transfers:
        pending.synchronize()
    manager.poll_transfers()
    assert not manager.pending_transfers_exist(0), "offload pins never dropped"
    manager.reset_prefix_cache()
    assert not manager.pools[0].prefix_caches[FULL], "device tier not empty"

    # With the device tier empty, the same prompt must be served by the
    # connector instead.
    connector.hold(offloaded_hashes)
    connector.loads.clear()
    second = make_ctx([1, 2, 3, 4])
    manager.claim(second)
    transfer = manager.alloc(second)
    assert connector.loads, "no load issued"
    asked = connector.loads[-1][1][FULL]
    served = len(asked)
    assert served > 0
    assert second.tokens.processed_length == served
    assert not transfer.is_complete(), "async load should still be in flight"

    # The heart of it: nothing may read these pages before the copy lands.
    prefix_cache = manager.pools[0].prefix_caches[FULL]
    assert all(h not in prefix_cache for h in asked[:served]), (
        "onloaded pages were published before their copy landed"
    )
    transfer.synchronize()
    manager.poll_transfers()
    assert any(h in prefix_cache for h in asked[:served]), (
        "onloaded pages were never published after landing"
    )
    assert not manager.pending_transfers_exist(0), "onload pins never dropped"


def test_multi_leaf_sends_distinct_per_leaf_block_ids() -> None:
    """Each leaf is tiled separately, so both must reach the connector.

    A single id list broadcast across leaves -- what a non-Jenga manager can
    get away with -- would address one leaf with the other's page index.
    """
    connector = FakeConnector([VALUES, SCALES])
    manager = make_manager(connector, leaves=(VALUES, SCALES))
    ctx = make_ctx([1, 2, 3, 4])
    manager.claim(ctx)
    manager.alloc(ctx)
    ctx.update(9)
    manager.step(ctx)
    manager.offload(0)

    assert connector.offloads, "no offload issued"
    block_ids, hashes = connector.offloads[-1]
    assert set(block_ids) == set(hashes) == {VALUES, SCALES}
    assert hashes[VALUES] == hashes[SCALES]
    assert (
        len(block_ids[VALUES]) == len(block_ids[SCALES]) == len(hashes[VALUES])
    )
    assert block_ids[VALUES] != block_ids[SCALES], (
        "leaves shared a page index; they have separate bid spaces"
    )


def test_device_hit_and_onload_splice_into_one_run() -> None:
    """A partial device hit is extended by the connector, not replaced."""
    connector = FakeConnector([FULL], asynchronous=False)
    manager = make_manager(connector)

    first = make_ctx([1, 2, 3, 4, 5])
    manager.claim(first)
    manager.alloc(first)
    first.update(9)
    manager.step(first)
    manager.offload(0)
    offloaded = connector.offloads[-1][1][FULL]
    manager.release(first)

    # Keep only the leading block on device; the connector still holds the
    # rest, so the two hits have to meet in the middle.
    manager.reset_prefix_cache()
    connector.hold(offloaded)
    pool = manager.pools[0]
    assert not pool.prefix_caches[FULL]
    replay = make_ctx([1])
    manager.claim(replay)
    manager.alloc(replay)
    replay.update(9)
    manager.step(replay)
    manager.release(replay)
    assert len(pool.prefix_caches[FULL]) == 1

    connector.loads.clear()
    second = make_ctx([1, 2, 3, 4, 5])
    manager.claim(second)
    manager.alloc(second)

    # The replay's own last token is never hashed, so one block stays uncached.
    reusable = offloaded[:-1]
    asked = connector.loads[-1][1][FULL]
    assert manager.metrics.device_blocks_served == 1, (
        "device hit should stop at the one cached block"
    )
    assert asked == list(reusable[1:]), (
        "onload must resume after the device hit"
    )
    assert second.cached_prefix_length == len(reusable)
    assert second.cached_prefix_external_length == len(reusable) - 1
    assert second.tokens.processed_length == len(reusable)
    # Recency covers the whole reused run, device hit AND connector onload,
    # which is what BlockManager touches on the legacy path. Gating the touch
    # on the device hit alone would skip it entirely under
    # MODULAR_ONLY_USE_KV_CONNECTOR_LAST_LEVEL_CACHE, where there is never one.
    # `release` also re-ranks a finished request's sequence (CLIN-1893), so
    # earlier releases in this test contribute touches of their own. The
    # admission touch is the one under test, and it is the most recent.
    assert connector.touches[-1] == (list(reusable), 0)
    # One contiguous row: the device page first, then the onloaded ones,
    # then whatever the forward still has to fill.
    row = manager.get_req_blocks_per_leaf(second)[FULL]
    assert row[: len(reusable)] == [
        pool.prefix_caches[FULL][h].bid for h in reusable
    ]


def test_onload_is_skipped_when_the_run_does_not_fit() -> None:
    """Onloading is all-or-nothing: a run the pool cannot hold is dropped."""
    connector = FakeConnector([VALUES, SCALES], asynchronous=False)
    filler = make_manager(
        connector, leaves=(VALUES, SCALES), num_huge_blocks=16
    )
    ctx = make_ctx([1, 2, 3, 4, 5])
    filler.claim(ctx)
    filler.alloc(ctx)
    ctx.update(9)
    filler.step(ctx)
    filler.offload(0)
    assert any(connector.held.values())

    # Two huge blocks between them cannot carve a 4-page run for both leaves.
    tight = make_manager(connector, leaves=(VALUES, SCALES), num_huge_blocks=3)
    pool = tight.pools[0]
    free_before = pool.num_free_huge_blocks
    connector.loads.clear()
    replay = make_ctx([1, 2, 3, 4, 5])
    tight.claim(replay)
    with pytest.raises(InsufficientBlocksError):
        tight.alloc(replay)

    assert not connector.loads, "asked for a run the pool cannot hold"
    assert replay.tokens.processed_length == 0
    # Nothing was drawn on the way out.
    assert pool.num_free_huge_blocks == free_before
    assert not pool.prefix_caches[VALUES]


def test_a_lookup_that_finds_nothing_posts_no_load() -> None:
    """A miss asks and then stops. Nothing is owed back to the connector.

    The connector holds nothing, so the reconcile agrees on no prefix. What a
    leasing connector does with the lookup it took is its own business -- the
    manager has no release to call.
    """
    connector = FakeConnector([VALUES, SCALES], asynchronous=False)
    manager = make_manager(
        connector, leaves=(VALUES, SCALES), num_huge_blocks=16
    )
    ctx = make_ctx([1, 2, 3, 4, 5])
    manager.claim(ctx)
    manager.alloc(ctx)

    assert connector.lookups, "the connector was never asked"
    assert not connector.loads


def test_last_level_cache_only_forces_every_hit_through_the_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The benchmarking flag disables device hits, not device commits."""
    monkeypatch.setenv("MODULAR_ONLY_USE_KV_CONNECTOR_LAST_LEVEL_CACHE", "1")
    connector = FakeConnector([FULL], asynchronous=False)
    manager = make_manager(connector)

    first = make_ctx([1, 2, 3, 4, 5])
    manager.claim(first)
    manager.alloc(first)
    first.update(9)
    manager.step(first)
    manager.offload(0)
    offloaded = connector.offloads[-1][1][FULL]
    manager.release(first)

    # The device prefix cache still holds the run; the lookup must ignore it.
    pool = manager.pools[0]
    assert all(h in pool.prefix_caches[FULL] for h in offloaded)

    connector.hold(offloaded)
    connector.loads.clear()
    second = make_ctx([1, 2, 3, 4, 5])
    manager.claim(second)
    manager.alloc(second)

    assert manager.metrics.device_blocks_served == 0
    assert second.cached_prefix_external_length == second.cached_prefix_length
    assert second.cached_prefix_length > 0
    assert connector.loads[-1][1][FULL] == list(offloaded[:-1])
    # The recency touch must still fire with NO device hit. This is the
    # regression guard for the call site's own rationale: gating it on
    # `num_hit_blocks` instead of the reused run skips it entirely under
    # MODULAR_ONLY_USE_KV_CONNECTOR_LAST_LEVEL_CACHE, which is exactly the
    # configuration that leans on the connector hardest. No other test can see
    # that mutation, because every other one has a device hit and the touch
    # argument is byte-identical either way.
    assert connector.touches[-1] == (list(offloaded[:-1]), 0)


def test_alloc_drains_landed_transfers_so_pins_do_not_accumulate() -> None:
    """Offload pins are returned by the next alloc, as in the paged manager.

    Nothing in the serving loop calls ``poll_transfers`` on its own, so an
    alloc that skips the drain leaks every offload source for the whole run.
    """
    connector = FakeConnector([FULL])
    manager = make_manager(connector)
    pool = manager.pools[0]
    free_huge = pool.num_free_huge_blocks

    for _ in range(4):
        ctx = make_ctx([1, 2, 3, 4])
        manager.claim(ctx)
        manager.alloc(ctx)
        ctx.update(9)
        manager.step(ctx)
        manager.offload(0)
        manager.release(ctx)
        # The copy engine finishes between iterations.
        for pending in connector.transfers:
            pending.synchronize()

    assert manager.pending_transfers_exist(0), "offload never pinned"
    drain = make_ctx([7, 8])
    manager.claim(drain)
    manager.alloc(drain)
    manager.release(drain)

    assert not manager.pending_transfers_exist(0), "alloc did not drain pins"
    assert pool.num_free_huge_blocks == free_huge, (
        "huge blocks never came back: pinned pages kept them out of the pool"
    )


def test_swa_connector_onload_null_pads_and_skips_prefix_cache_commit() -> None:
    groups = {
        FULL: KVCacheGroupId.full(),
        SLIDING: KVCacheGroupId("sliding_window", 4),
    }
    connector = FakeConnector(groups, asynchronous=False)
    manager = make_manager(
        connector,
        leaf_infos={
            FULL: KVLeafInfo(1, groups[FULL]),
            SLIDING: KVLeafInfo(1, groups[SLIDING]),
        },
    )

    first = make_ctx([1, 2, 3, 4, 5, 6])
    manager.claim(first)
    manager.alloc(first)
    first.update(9)
    manager.step(first)
    manager.offload(0)
    offloaded = connector.offloads[-1][1][FULL]
    manager.release(first)
    manager.reset_prefix_cache()
    connector.hold(offloaded)

    replay = make_ctx([1, 2, 3, 4, 5, 6])
    manager.claim(replay)
    manager.alloc(replay)

    block_ids, hashes, _ = connector.loads[-1]
    asked = hashes[FULL]
    assert asked == list(offloaded[:-1])
    # The windowed leaf is asked only for the tail its attention reads, and
    # given exactly that many blocks.
    window = groups[SLIDING].blocks_in_window(1)
    assert hashes[SLIDING] == asked[-window:]
    assert len(block_ids[SLIDING]) == window

    pool = manager.pools[0]
    sliding_blocks = manager.get_req_blocks_per_leaf(replay)[SLIDING]
    null_count = len(asked) - groups[SLIDING].blocks_in_window(1)
    assert sliding_blocks[:null_count] == [0] * null_count
    assert len(pool.prefix_caches[SLIDING]) == len(asked) - null_count
    assert all(
        not block.is_null for block in pool.prefix_caches[SLIDING].values()
    )
    assert len(pool.prefix_caches[FULL]) == len(asked)


def test_the_drain_settles_offloads_rather_than_step() -> None:
    """A posted offload is settled by the manager's drain, not by ``step``.

    dKV's ``offload`` only acquires its slots and posts the writes; the blocks
    stay ``Filling`` -- counted under ``g1_blocks`` but unreadable -- until
    their transfer settles. Nothing settling them is a 0% hit rate while dKV
    appears to fill up. ``step`` used to run a post-forward barrier for that;
    the transfer's own completion is what does it now, observed by
    ``poll_transfers``, which also hands the connector its own drain.
    """
    connector = FakeConnector([FULL])
    manager = make_manager(connector)

    ctx = make_ctx([1, 2, 3, 4])
    manager.claim(ctx)
    manager.alloc(ctx)
    ctx.update(9)
    manager.step(ctx)
    manager.offload(0)

    assert connector.offloads, "step should leave a committed run to offload"
    posted = connector.transfers[-1]
    assert not posted.done, "an in-flight offload is not settled by step"

    polls_before = connector.polls
    manager.poll_transfers()
    assert connector.polls > polls_before, (
        "the drain has to reach the connector, whose own transfers settle "
        "resources no poll of the manager's covers"
    )

    pool = manager.pools[0]
    sources = [pool.block(FULL, bid) for bid in posted.g0_blocks_per_leaf[FULL]]
    # The request holds these pages too, so the write's own pin shows up as one
    # ref on top of that rather than as an absolute count.
    pinned = [block.ref_cnt for block in sources]

    posted.synchronize()
    manager.poll_transfers()

    assert not manager.pending_transfers_exist(0)
    assert [block.ref_cnt for block in sources] == [n - 1 for n in pinned], (
        "a settled write has to hand its source pages back"
    )


def test_a_failed_poll_unpins_without_publishing() -> None:
    """A poll that raises unpins the onload's pages and commits nothing.

    The transfer settled itself on the way to failing, and the pages it was
    filling hold no valid KV, so publishing them would let any later request
    hit garbage. The pin still has to come off, or those pages are lost to the
    pool for the rest of the process. An error that is not ``KVLoadFailed`` is
    a bug rather than a copy that failed, so it still reaches the caller.
    """
    connector = FakeConnector([FULL])
    manager = make_manager(connector)

    # Fill the connector from a forward, then empty the device tier so the
    # same prompt has to come back over an onload.
    first = make_ctx([1, 2, 3, 4])
    manager.claim(first)
    manager.alloc(first)
    first.update(9)
    manager.step(first)
    manager.offload(0)
    offloaded_hashes = connector.offloads[-1][1][FULL]
    for pending in connector.transfers:
        pending.synchronize()
    manager.poll_transfers()
    manager.release(first)
    manager.reset_prefix_cache()

    connector.hold(offloaded_hashes)
    second = make_ctx([1, 2, 3, 4])
    manager.claim(second)
    transfer = manager.alloc(second)
    asked = connector.loads[-1][1][FULL]
    assert asked, "no onload issued"
    assert isinstance(transfer, FakeTransfer)
    transfer.fails = True
    transfer.error = RuntimeError("transfer failed")

    pool = manager.pools[0]
    onloaded = [
        pool.block(FULL, bid)
        for bid in manager.get_req_blocks_per_leaf(second)[FULL]
    ]

    with pytest.raises(RuntimeError, match="transfer failed"):
        manager.poll_transfers()

    prefix_cache = pool.prefix_caches[FULL]
    assert all(h not in prefix_cache for h in asked), (
        "a failed onload's pages must not be published"
    )
    assert not manager.pending_transfers_exist(0)
    assert all(block.ref_cnt == 1 for block in onloaded), (
        "the failed transfer's pin is released, leaving the allocation's own"
    )


def _hold_only_in_the_connector(
    manager: JengaBlockManager,
    connector: FakeConnector,
    tokens: Sequence[int],
) -> list[bytes]:
    """Leaves ``tokens``' KV in the connector and nowhere on the device.

    Runs one forward over them, lands its offload, then empties the device
    tier, so the same prompt can only come back over an onload. Returns the
    run of hashes the full-attention leaf holds.
    """
    first = make_ctx(tokens)
    manager.claim(first)
    manager.alloc(first)
    first.update(9)
    manager.step(first)
    manager.offload(0)
    _, offloaded = connector.offloads[-1]
    for pending in connector.transfers:
        pending.synchronize()
    manager.poll_transfers()
    manager.release(first)
    manager.reset_prefix_cache()
    for leaf_id, hashes in offloaded.items():
        connector.held[leaf_id].update(hashes)
    return offloaded[FULL]


@pytest.mark.parametrize("fails_at", ["post", "first_poll"])
def test_a_load_that_fails_to_post_is_served_as_a_miss(fails_at: str) -> None:
    """A transport fault before anything is spliced costs a hit, not the worker.

    The load can raise while posting, or the manager's own poll right after it
    can see the copy fail. Nothing is in flight into the rows either way, so
    they go straight back, and the request skips the connector for the rest of
    its claim rather than failing the same way on every admission.
    """
    connector = FakeConnector([FULL])
    manager = make_manager(connector)
    _hold_only_in_the_connector(manager, connector, [1, 2, 3, 4])
    pool = manager.pools[0]
    free_before = pool.num_free_blocks(FULL)
    if fails_at == "post":
        connector.load_error = KVLoadFailed("memory transfer failed")
    else:
        connector.fail_on_first_poll = True

    ctx = make_ctx([1, 2, 3, 4])
    manager.claim(ctx)
    transfer = manager.alloc(ctx)

    assert transfer.is_complete()
    assert ctx.tokens.processed_length == 0
    assert ctx.cached_prefix_length == 0
    assert not manager.pending_transfers_exist(0)
    assert not pool.prefix_caches[FULL]
    assert manager.take_metrics().connector_load_failures == 1

    lookups = len(connector.lookups)
    manager.alloc(ctx)
    assert len(connector.lookups) == lookups, (
        "the request skips the connector now"
    )

    manager.release(ctx)
    assert pool.num_free_blocks(FULL) == free_before, "the rows went back"


def _hold_in_connector_with_a_device_hit(
    manager: JengaBlockManager,
    connector: FakeConnector,
    tokens: Sequence[int],
    num_device_blocks: int,
) -> list[bytes]:
    """Puts ``tokens``' KV in the connector and its first blocks on device.

    Returns the hashes the connector holds, in prefix order.
    """
    held = _hold_only_in_the_connector(manager, connector, tokens)
    by_leaf = {
        leaf_id: set(hashes) for leaf_id, hashes in connector.held.items()
    }
    connector.reset_prefix_cache()
    device = make_ctx(tokens[:num_device_blocks])
    manager.claim(device)
    manager.alloc(device)
    device.update(9)
    manager.step(device)
    manager.release(device)
    for leaf_id, hashes in by_leaf.items():
        connector.held[leaf_id].update(hashes)
    return held


def test_a_failed_onload_recomputes_from_the_device_hit() -> None:
    """The failed copy costs the onload and nothing else.

    The scheduler's cordon sweep sees the failure first here. The request's
    next ``alloc`` still undoes the splice and takes the device hit back
    without the connector: the onloaded pages go back unpublished, and the
    request recomputes from where its device hit ends.
    """
    connector = FakeConnector([FULL])
    manager = make_manager(connector, num_huge_blocks=32)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    held = _hold_in_connector_with_a_device_hit(manager, connector, tokens, 3)
    pool = manager.pools[0]
    free_before = pool.num_free_blocks(FULL)
    device_bids = [pool.prefix_caches[FULL][h].bid for h in held[:3]]

    ctx = make_ctx(tokens)
    manager.claim(ctx)
    transfer = manager.alloc(ctx)
    assert ctx.cached_prefix_external_length == 4
    assert isinstance(transfer, FakeTransfer)
    transfer.fails = True
    with pytest.raises(KVLoadFailed):
        transfer.is_complete()

    lookups = len(connector.lookups)
    resumed = manager.alloc(ctx)

    assert resumed.is_complete()
    assert len(connector.lookups) == lookups, (
        "the recompute skips the connector"
    )
    assert ctx.tokens.processed_length == 3
    assert ctx.cached_prefix_length == 3
    assert ctx.cached_prefix_external_length == 0
    assert manager.get_req_blocks_per_leaf(ctx)[FULL][:3] == device_bids
    assert all(h not in pool.prefix_caches[FULL] for h in held[3:7]), (
        "a failed copy's pages must not be published"
    )
    assert not manager.pending_transfers_exist(0)
    assert manager.take_metrics().connector_load_failures == 1

    manager.release(ctx)
    assert pool.num_free_blocks(FULL) == free_before, "every page went back"


def test_a_failed_windowed_onload_takes_back_the_window_it_freed() -> None:
    """A sliding leaf cannot just be trimmed back to the device hit.

    Splicing an onload that reaches past the window frees the device hit's
    windowed pages and nulls their slots, so the window ending at the device
    hit is gone from the row. Rolling back means taking it from the device
    tier again.
    """
    groups = {
        FULL: KVCacheGroupId.full(),
        SLIDING: KVCacheGroupId("sliding_window", 4),
    }
    connector = FakeConnector(groups)
    manager = make_manager(
        connector,
        num_huge_blocks=64,
        leaf_infos={
            FULL: KVLeafInfo(1, groups[FULL]),
            SLIDING: KVLeafInfo(1, groups[SLIDING]),
        },
    )
    tokens = list(range(1, 11))
    held = _hold_in_connector_with_a_device_hit(manager, connector, tokens, 3)
    pool = manager.pools[0]
    window_bids = [pool.prefix_caches[SLIDING][h].bid for h in held[:3]]

    ctx = make_ctx(tokens)
    manager.claim(ctx)
    transfer = manager.alloc(ctx)
    sliding_row = manager.get_req_blocks_per_leaf(ctx)[SLIDING]
    assert sliding_row[:3] == [0, 0, 0], "the device window was nulled"
    assert isinstance(transfer, FakeTransfer)
    transfer.fails = True
    manager.poll_transfers()

    manager.alloc(ctx)

    assert ctx.tokens.processed_length == 3
    assert manager.get_req_blocks_per_leaf(ctx)[SLIDING][:3] == window_bids
    assert all(h not in pool.prefix_caches[SLIDING] for h in held[3:9])
    assert manager.take_metrics().connector_load_failures == 1


def test_a_request_released_before_its_onload_fails_leaves_nothing() -> None:
    """A request cancelled mid-onload has nothing left to roll back."""
    connector = FakeConnector([FULL])
    manager = make_manager(connector)
    _hold_only_in_the_connector(manager, connector, [1, 2, 3, 4])
    pool = manager.pools[0]
    free_before = pool.num_free_blocks(FULL)

    ctx = make_ctx([1, 2, 3, 4])
    manager.claim(ctx)
    transfer = manager.alloc(ctx)
    manager.release(ctx)
    assert isinstance(transfer, FakeTransfer)
    transfer.fails = True
    manager.poll_transfers()

    assert not manager.pending_transfers_exist(0)
    assert pool.num_free_blocks(FULL) == free_before
    assert manager.take_metrics().connector_load_failures == 1


def test_a_failed_onload_leaves_the_state_row_a_fresh_admission_holds() -> None:
    """A recurrent leaf cannot be trimmed back to its device hit either.

    Trimming keeps its live block, so taking the device hit again would land
    the device checkpoint behind a live block that never ran. The rollback
    hands the request a fresh claim instead: it resumes from the device
    checkpoint and runs in one live block.
    """
    connector = hybrid_connector()
    manager = make_hybrid_manager(connector)
    pool = manager.pools[0]

    # The connector holds two pages, and the state at the second boundary.
    first = make_ctx(list(range(1, 2 * PAGE + 1)))
    manager.claim(first)
    forward_to_boundary(manager, first)
    manager.offload(0)
    _, offloaded = connector.offloads[-1]
    manager.release(first)
    for pending in connector.transfers:
        pending.synchronize()
    manager.poll_transfers()
    manager.reset_prefix_cache()

    # The device holds the first page, and the state at its boundary.
    device = make_ctx(list(range(1, PAGE + 2)))
    manager.claim(device)
    forward_to_boundary(manager, device)
    (device_hash,) = pool.prefix_caches[STATE]
    device_state = pool.prefix_caches[STATE][device_hash]
    for leaf_id, hashes in offloaded.items():
        connector.held[leaf_id].update(hashes)
    free_before = pool.num_free_blocks(STATE)

    ctx = make_ctx(list(range(1, 3 * PAGE + 1)))
    manager.claim(ctx)
    transfer = manager.alloc(ctx)
    assert ctx.tokens.processed_length == 2 * PAGE, "device page plus onload"
    assert isinstance(transfer, FakeTransfer)
    transfer.fails = True
    manager.poll_transfers()

    manager.alloc(ctx)

    state_group = manager.groups[STATE]
    assert isinstance(state_group, RecurrentKVGroupCoordinator)
    assert ctx.tokens.processed_length == PAGE
    runs_in = state_group.live_blocks(ctx.request_id)
    assert runs_in is not None
    assert state_group.resume(ctx, 0) == {
        STATE: (device_state.bid, runs_in[STATE])
    }
    null_bid = pool.null_little_blocks[STATE].bid
    row = manager.get_req_blocks_per_leaf(ctx)[STATE]
    assert [bid for bid in row if bid != null_bid] == [
        device_state.bid,
        runs_in[STATE],
    ], "one device checkpoint, then one live block"

    manager.release(ctx)
    assert pool.num_free_blocks(STATE) == free_before


def test_a_request_that_ran_past_a_failed_onload_is_not_recomputed() -> None:
    """A caller that never held the request back has already read the pages.

    There is no clean prefix to go back to by then, so the next ``alloc``
    fails loudly instead of recomputing over KV it cannot trust.
    """
    connector = FakeConnector([FULL])
    manager = make_manager(connector)
    _hold_only_in_the_connector(manager, connector, [1, 2, 3, 4])

    ctx = make_ctx([1, 2, 3, 4])
    manager.claim(ctx)
    transfer = manager.alloc(ctx)
    ctx.update(9)
    manager.step(ctx)
    assert isinstance(transfer, FakeTransfer)
    transfer.fails = True

    with pytest.raises(RuntimeError, match="ran past") as excinfo:
        manager.alloc(ctx)
    assert isinstance(excinfo.value.__cause__, KVLoadFailed)


def test_step_without_a_connector_still_commits() -> None:
    """A manager with no connector must not reach for one."""
    manager = make_manager(None)

    ctx = make_ctx([1, 2, 3, 4])
    manager.claim(ctx)
    manager.alloc(ctx)
    ctx.update(9)
    manager.step(ctx)

    manager.release(ctx)


def test_a_state_offloads_only_where_a_checkpoint_landed() -> None:
    """The attention run offloads whole, the state only at its checkpoint."""
    connector = hybrid_connector()
    manager = make_hybrid_manager(connector)
    ctx = make_ctx(list(range(1, 2 * PAGE + 1)))
    manager.claim(ctx)
    forward_to_boundary(manager, ctx)
    manager.offload(0)

    assert connector.offloads, "no offload issued"
    block_ids, hashes = connector.offloads[-1]
    assert len(hashes[FULL]) == 2, "the attention run was cut short"
    assert hashes[STATE] == hashes[FULL][1:], (
        "the state checkpoints at the boundary the forward ended on"
    )
    assert len(block_ids[FULL]) == 2
    assert len(block_ids[STATE]) == 1
    assert set(hashes[STATE]) == set(manager.pools[0].prefix_caches[STATE])


def test_a_state_reloads_from_the_connector_at_the_deepest_checkpoint() -> None:
    """With the device tier empty, the connector serves the KV and the state."""
    connector = hybrid_connector()
    manager = make_hybrid_manager(connector)
    pool = manager.pools[0]

    first = make_ctx(list(range(1, 2 * PAGE + 1)))
    manager.claim(first)
    forward_to_boundary(manager, first)
    manager.offload(0)
    _, offloaded = connector.offloads[-1]
    manager.release(first)
    for pending in connector.transfers:
        pending.synchronize()
    manager.poll_transfers()
    manager.reset_prefix_cache()
    assert not pool.prefix_caches[FULL] and not pool.prefix_caches[STATE]
    for leaf_id, hashes in offloaded.items():
        connector.held[leaf_id].update(hashes)

    # A page longer, so its chain reaches the checkpoint.
    second = make_ctx(list(range(1, 3 * PAGE + 1)))
    manager.claim(second)
    transfer = manager.alloc(second)

    block_ids, asked, _ = connector.loads[-1]
    assert asked[FULL] == offloaded[FULL]
    assert asked[STATE] == offloaded[STATE]
    assert len(block_ids[STATE]) == 1
    assert second.tokens.processed_length == 2 * PAGE
    assert second.cached_prefix_external_length == 2 * PAGE
    assert manager.metrics.cache_tokens == 2 * PAGE

    # Until the copy lands the loaded state is neither published nor run in.
    state_group = manager.groups[STATE]
    assert isinstance(state_group, RecurrentKVGroupCoordinator)
    (loaded_bid,) = block_ids[STATE]
    runs_in = state_group.live_blocks(second.request_id)
    assert runs_in is not None
    assert runs_in[STATE] != loaded_bid, (
        "the recurrence would run in the block the copy is filling"
    )
    assert not transfer.is_complete()
    assert not pool.prefix_caches[STATE]
    transfer.synchronize()
    manager.poll_transfers()
    (checkpoint_hash,) = offloaded[STATE]
    assert pool.prefix_caches[STATE][checkpoint_hash].bid == loaded_bid
    assert state_group.resume(second, 0) == {
        STATE: (loaded_bid, runs_in[STATE])
    }

    # The forward ends on the next boundary, and the successor is there.
    second.update(9)
    assert state_group.checkpoint(second, 0), "the boundary found no successor"
    manager.step(second)
