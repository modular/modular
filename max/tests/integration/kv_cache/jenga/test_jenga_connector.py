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

import logging
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheGroupId,
    PagedKVLeafRegion,
    RecurrentStateParams,
    RecurrentStateRegion,
)
from max.nn.kv_cache.cache_params import KVCacheMemory
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.kv_cache import InsufficientBlocksError
from max.pipelines.kv_cache.connectors.null_connector import NullConnector
from max.pipelines.kv_cache.kv_connector import (
    ByteCount,
    CompletedLookup,
    KVConnector,
    KVConnectorTransfer,
    KVLoadFailed,
    KVLoadRefused,
    KVLookup,
    KVTransfer,
)
from max.pipelines.kv_cache.paged_kv_cache import jenga_block_manager
from max.pipelines.kv_cache.paged_kv_cache._common import PrefixCacheHits
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


class FakeLookup:
    """A started lookup that answers only once ``answer`` is called.

    Setting ``error`` makes every poll raise it, as a broken handle would.
    """

    def __init__(self, masks: Mapping[str, Sequence[bool]]) -> None:
        self._masks = masks
        self.answered = False
        self.error: Exception | None = None
        self.polls = 0

    def is_complete(self) -> bool:
        self.polls += 1
        if self.error is not None:
            raise self.error
        return self.answered

    def masks(self) -> Mapping[str, Sequence[bool]]:
        if not self.answered:
            raise RuntimeError("the lookup has not answered yet")
        return self._masks

    def answer(self) -> None:
        self.answered = True


class FakeConnector:
    """Serves whatever hashes each leaf holds, and records calls.

    ``deferred_lookups`` makes ``start_lookup`` hand back a :class:`FakeLookup`
    that answers only when told to, the way a connector whose lookup is a
    round trip does. ``probe_answers`` makes ``probe`` answer from what the
    connector holds rather than sending the manager back to ``lookup``, and
    ``probe_leaves`` narrows that answer to the leaves it names.
    """

    name = "FakeConnector"

    def __init__(
        self,
        leaves: Mapping[str, KVCacheGroupId] | Sequence[str],
        asynchronous: bool = True,
        skips_held: bool = False,
        deferred_lookups: bool = False,
        probe_answers: bool = False,
    ) -> None:
        self.deferred_lookups = deferred_lookups
        self.probe_answers = probe_answers
        self.probe_leaves: Sequence[str] | None = None
        self.started: list[FakeLookup] = []
        self.loaded_from: list[KVLookup] = []
        self.abandoned: list[KVLookup] = []
        self.probes: list[tuple[list[bytes], list[int], bytes | None]] = []
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
        # Whether an offload skips the hashes a leaf already holds, as the
        # real connectors do, and so reports only the blocks it took.
        self.skips_held = skips_held
        self.lookups: list[tuple[list[bytes], int, bytes | None]] = []
        # Only the calls to the blocking `lookup`, which a deferred alloc
        # must never make because it runs on the scheduler thread.
        self.blocking_lookups = 0
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
        self.blocking_lookups += 1
        self.lookups.append((list(block_hashes), replica_idx, hint))
        return self._resident(block_hashes)

    def _resident(
        self, block_hashes: Sequence[bytes]
    ) -> Mapping[str, Sequence[bool]]:
        return {
            leaf_id: [h in self.held[leaf_id] for h in block_hashes]
            for leaf_id in self.leaves
        }

    def start_lookup(
        self,
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
        hint: bytes | None = None,
    ) -> KVLookup:
        self.lookups.append((list(block_hashes), replica_idx, hint))
        masks = self._resident(block_hashes)
        if not self.deferred_lookups:
            return CompletedLookup(masks)
        lookup = FakeLookup(masks)
        self.started.append(lookup)
        return lookup

    def defers_lookup(self, replica_idx: int = 0) -> bool:
        return self.deferred_lookups

    def load_lookup(
        self,
        lookup: KVLookup,
        block_ids: Mapping[str, Sequence[int]],
        block_hashes: Mapping[str, Sequence[bytes]],
        replica_idx: int = 0,
        hint: bytes | None = None,
    ) -> KVTransfer:
        self.loaded_from.append(lookup)
        return self.load(
            block_ids, block_hashes, replica_idx=replica_idx, hint=hint
        )

    def abandon_lookup(self, lookup: KVLookup) -> None:
        self.abandoned.append(lookup)

    def probe(
        self,
        block_hashes: Sequence[bytes],
        replica_idxs: Sequence[int],
        hint: bytes | None = None,
    ) -> Sequence[Mapping[str, Sequence[bool]]] | None:
        self.probes.append((list(block_hashes), list(replica_idxs), hint))
        if not self.probe_answers:
            return None
        resident = {
            leaf_id: [h in self.held[leaf_id] for h in block_hashes]
            for leaf_id in (
                self.leaves if self.probe_leaves is None else self.probe_leaves
            )
        }
        return [resident for _ in replica_idxs]

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
        taken = {
            leaf_id: [
                bid
                for bid, block_hash in zip(
                    ids, block_hashes[leaf_id], strict=True
                )
                if not (self.skips_held and block_hash in self.held[leaf_id])
            ]
            for leaf_id, ids in block_ids.items()
        }
        for leaf_id, hashes in block_hashes.items():
            self.held[leaf_id].update(hashes)
        transfer = FakeTransfer(taken)
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


def test_an_offload_pins_only_the_blocks_the_connector_took() -> None:
    """A hash the connector already holds is skipped, so its page is not pinned.

    The connector reports what it took in ``g0_blocks_per_leaf``; pinning
    everything handed over would hold pages no copy reads until the transfer
    lands.
    """
    connector = FakeConnector([FULL], skips_held=True)
    manager = make_manager(connector)

    tokens = [1, 2, 3, 4]
    ctx = make_ctx(tokens)
    manager.claim(ctx)
    manager.alloc(ctx)
    ctx.update(9)
    manager.step(ctx)
    assert not connector.offloads, "step leaves the run for the offload call"

    pool = manager.pools[0]
    rows = manager.get_req_blocks_per_leaf(ctx)[FULL]
    committed = [b for b in rows if pool.block(FULL, b).block_hash is not None]
    assert len(committed) > 1
    held_hash = pool.block(FULL, committed[0]).block_hash
    assert held_hash is not None
    connector.held[FULL].add(held_hash)
    before = {bid: pool.block(FULL, bid).ref_cnt for bid in committed}

    manager.offload(0)

    posted = connector.transfers[-1]
    assert posted.g0_blocks_per_leaf[FULL] == committed[1:]
    after = {bid: pool.block(FULL, bid).ref_cnt for bid in committed}
    assert after[committed[0]] == before[committed[0]], (
        "a block the connector skipped must not be pinned"
    )
    assert all(after[b] == before[b] + 1 for b in committed[1:]), (
        "every block the connector took is pinned until its copy lands"
    )

    posted.synchronize()
    manager.poll_transfers()
    assert {bid: pool.block(FULL, bid).ref_cnt for bid in committed} == before


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


def test_a_connector_without_an_async_lookup_answers_at_once() -> None:
    """The protocol defaults wrap the synchronous calls, so nothing changes.

    A connector that answers from local memory gets a handle that is already
    complete, a load that is a plain ``load``, an abandon that does nothing
    and no pricing probe, which sends the manager down the paths it ran
    before any of this existed.
    """
    connector = NullConnector()
    hashes = [bytes([1]) * 8, bytes([2]) * 8]

    lookup = connector.start_lookup(hashes)

    assert lookup.is_complete()
    assert lookup.masks() == connector.lookup(hashes)
    transfer = connector.load_lookup(lookup, {"full": [1, 2]}, {"full": hashes})
    assert transfer.is_complete()
    connector.abandon_lookup(lookup)
    assert connector.probe(hashes, [0, 1]) is None


def test_a_deferred_lookup_holds_the_request_out_until_it_answers() -> None:
    """The connector's answer is waited for off the batch, then loaded once.

    While the lookup is outstanding the request holds nothing and has
    skipped nothing, so the scheduler can readmit it unchanged. The alloc
    after the answer loads from that same lookup rather than asking again.
    """
    connector = FakeConnector([FULL], deferred_lookups=True)
    manager = make_manager(connector)
    held = _hold_only_in_the_connector(manager, connector, [1, 2, 3, 4])
    # The request's own last token is never hashed, so one block stays unasked.
    asked = held[:-1]
    pool = manager.pools[0]
    free_before = pool.num_free_blocks(FULL)
    connector.lookups.clear()

    ctx = make_ctx([1, 2, 3, 4])
    manager.claim(ctx)
    pending = manager.alloc(ctx, defer_lookup=True)

    assert not pending.is_complete()
    assert not connector.loads
    assert ctx.tokens.processed_length == 0
    assert manager.num_req_blocks(ctx) == 0
    assert pool.num_free_blocks(FULL) == free_before
    (lookup,) = connector.started
    assert connector.lookups == [(asked, 0, None)]

    lookup.answer()
    assert pending.is_complete()
    transfer = manager.alloc(ctx, defer_lookup=True)

    assert len(connector.lookups) == 1, "the answer is reused, not asked again"
    assert connector.loaded_from == [lookup]
    assert not connector.abandoned
    assert connector.loads[-1][1][FULL] == asked
    assert ctx.tokens.processed_length == len(asked)
    assert ctx.cached_prefix_external_length == len(asked)
    assert not transfer.is_complete(), "the onload itself is still in flight"


def test_a_deferred_answer_is_read_from_where_the_device_hit_ends() -> None:
    """A device hit that grew while the lookup was out shortens the load.

    The lookup was asked past the device hit as counted when it started. By
    the time it answers another request may have put more of the prefix on
    device, so the answer is read from the new hit's end and only the rest
    is loaded, still out of the same lookup.
    """
    connector = FakeConnector([FULL], asynchronous=False, deferred_lookups=True)
    manager = make_manager(connector, num_huge_blocks=32)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    held = _hold_in_connector_with_a_device_hit(manager, connector, tokens, 3)
    asked = held[:-1]

    ctx = make_ctx(tokens)
    manager.claim(ctx)
    manager.alloc(ctx, defer_lookup=True)
    assert connector.lookups[-1][0] == asked[3:]

    # Another request lands the fourth block on device meanwhile.
    other = make_ctx(tokens[:5])
    manager.claim(other)
    manager.alloc(other)
    manager.release(other)
    assert asked[3] in manager.pools[0].prefix_caches[FULL]

    (lookup,) = connector.started
    lookup.answer()
    manager.alloc(ctx, defer_lookup=True)

    assert connector.loaded_from == [lookup]
    assert connector.loads[-1][1][FULL] == asked[4:]
    assert ctx.tokens.processed_length == len(asked)
    assert ctx.cached_prefix_length == len(asked)
    assert ctx.cached_prefix_external_length == len(asked) - 4


def test_a_deferred_answer_is_handed_back_when_the_device_hit_covers_it() -> (
    None
):
    """A device hit that grew over everything asked leaves nothing to load.

    Another request put the whole prefix on device while the lookup was out,
    so the request is served from the device tier and the answer goes back
    unread.
    """
    connector = FakeConnector([FULL], asynchronous=False, deferred_lookups=True)
    manager = make_manager(connector, num_huge_blocks=32)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    held = _hold_only_in_the_connector(manager, connector, tokens)
    asked = held[:-1]

    ctx = make_ctx(tokens)
    manager.claim(ctx)
    manager.alloc(ctx, defer_lookup=True)
    assert connector.lookups[-1][0] == asked

    other = make_ctx(tokens)
    manager.claim(other)
    manager.alloc(other)
    manager.release(other)
    assert all(h in manager.pools[0].prefix_caches[FULL] for h in asked)
    loads = len(connector.loads)

    (lookup,) = connector.started
    lookup.answer()
    transfer = manager.alloc(ctx, defer_lookup=True)

    assert transfer.is_complete()
    assert connector.abandoned == [lookup]
    assert connector.loaded_from == []
    assert len(connector.loads) == loads
    assert ctx.tokens.processed_length == len(asked)
    assert ctx.cached_prefix_length == len(asked)
    assert ctx.cached_prefix_external_length == 0


def test_starts_lookup_reads_what_the_deferred_alloc_then_does() -> None:
    """The check the scheduler asks before allocating matches the alloc.

    The scheduler bounds the lookups it has out by asking first, so a
    request reads as starting one exactly when the deferred alloc that
    follows starts one. A lookup already out is consumed rather than
    started, and a prompt the device tier covers asks nothing.
    """
    connector = FakeConnector([FULL], asynchronous=False, deferred_lookups=True)
    manager = make_manager(connector, num_huge_blocks=32)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    _hold_only_in_the_connector(manager, connector, tokens)

    fresh = make_ctx(tokens)
    manager.claim(fresh)
    assert manager.starts_lookup(fresh)
    manager.alloc(fresh, defer_lookup=True)
    assert len(connector.started) == 1
    assert not manager.starts_lookup(fresh)

    other = make_ctx(tokens)
    manager.claim(other)
    manager.alloc(other)
    manager.release(other)
    covered = make_ctx(tokens)
    manager.claim(covered)
    assert not manager.starts_lookup(covered)
    manager.alloc(covered, defer_lookup=True)
    assert len(connector.started) == 1


def test_a_lookup_answered_in_place_neither_starts_nor_awaits() -> None:
    """A connector that answers in place never holds a request out.

    So neither the check the scheduler asks before allocating nor the one
    it asks after reads a lookup, and the deferred alloc onloads at once.
    """
    connector = FakeConnector([FULL], asynchronous=True)
    manager = make_manager(connector, num_huge_blocks=32)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    held = _hold_only_in_the_connector(manager, connector, tokens)

    ctx = make_ctx(tokens)
    manager.claim(ctx)
    assert not manager.starts_lookup(ctx)
    transfer = manager.alloc(ctx, defer_lookup=True)

    assert not transfer.is_complete()
    assert not manager.awaits_lookup(ctx)
    assert connector.loads[-1][1][FULL] == held[:-1]


def test_a_lookup_handed_back_is_started_afresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A handed back answer is abandoned, and the next alloc asks again.

    The hand back counts against the request's hand backs and leaves its
    re-asks alone.
    """
    connector = FakeConnector([FULL], asynchronous=False, deferred_lookups=True)
    manager = make_manager(connector, num_huge_blocks=32)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    held = _hold_only_in_the_connector(manager, connector, tokens)

    ctx = make_ctx(tokens)
    manager.claim(ctx)
    manager.alloc(ctx, defer_lookup=True)
    (first,) = connector.started
    first.answer()
    age = jenga_block_manager._LOOKUP_HAND_BACK_AGE_S
    monkeypatch.setattr(jenga_block_manager, "_LOOKUP_HAND_BACK_AGE_S", 0.0)
    assert manager.hand_back_lookup(ctx)
    monkeypatch.setattr(jenga_block_manager, "_LOOKUP_HAND_BACK_AGE_S", age)

    assert connector.abandoned == [first]
    assert not manager.awaits_lookup(ctx)
    assert manager.starts_lookup(ctx)
    assert manager._state_of(ctx).connector_lookup_hand_backs == 1
    assert manager._state_of(ctx).connector_lookup_reasks == 0

    manager.alloc(ctx, defer_lookup=True)
    (_, again) = connector.started
    again.answer()
    transfer = manager.alloc(ctx, defer_lookup=True)

    assert transfer.is_complete()
    assert connector.loaded_from == [again]
    assert connector.loads[-1][1][FULL] == held[:-1]


def test_a_deferred_answer_is_asked_again_when_the_device_hit_shrinks() -> None:
    """A device hit evicted while the lookup was out leaves a gap it cannot fill.

    The answer starts past where the device hit used to end, so the run
    cannot continue from the shorter hit. It is handed back unused and the
    connector is asked again over everything past the shorter hit, off the
    scheduler thread like the first lookup. The check the scheduler asks
    before allocating reads that as a lookup started, so the bound on
    lookups out covers it. The request holds nothing until that answer is
    in, and then loads the whole prefix out of it.
    """
    connector = FakeConnector([FULL], asynchronous=False, deferred_lookups=True)
    manager = make_manager(connector, num_huge_blocks=32)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    held = _hold_in_connector_with_a_device_hit(manager, connector, tokens, 3)
    connector.lookups.clear()
    connector.loads.clear()
    connector.blocking_lookups = 0

    ctx = make_ctx(tokens)
    manager.claim(ctx)
    manager.alloc(ctx, defer_lookup=True)
    assert connector.lookups[-1][0] == held[3:-1]
    manager.pools[0].reset_prefix_cache()

    (first,) = connector.started
    first.answer()
    assert manager.starts_lookup(ctx)
    pending = manager.alloc(ctx, defer_lookup=True)

    assert not pending.is_complete()
    assert manager.awaits_lookup(ctx), "the re-ask holds the request out"
    assert connector.abandoned == [first]
    assert connector.blocking_lookups == 0
    assert connector.lookups[-1] == (held[:-1], 0, None)
    assert not connector.loads
    assert manager.num_req_blocks(ctx) == 0
    assert ctx.tokens.processed_length == 0

    (_, again) = connector.started
    again.answer()
    assert pending.is_complete()
    transfer = manager.alloc(ctx, defer_lookup=True)

    assert transfer.is_complete()
    assert connector.loaded_from == [again]
    assert connector.abandoned == [first]
    assert connector.blocking_lookups == 0
    assert connector.loads[-1][1][FULL] == held[:-1]
    assert ctx.tokens.processed_length == len(held) - 1
    assert ctx.cached_prefix_external_length == len(held) - 1


def test_a_lookup_asked_again_is_served_as_a_miss_at_its_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lookup asked after a shrink is bounded the way the first one is."""
    connector = FakeConnector([FULL], asynchronous=False, deferred_lookups=True)
    manager = make_manager(connector, num_huge_blocks=32)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    _hold_in_connector_with_a_device_hit(manager, connector, tokens, 3)
    connector.loads.clear()
    connector.blocking_lookups = 0

    ctx = make_ctx(tokens)
    manager.claim(ctx)
    manager.alloc(ctx, defer_lookup=True)
    manager.pools[0].reset_prefix_cache()
    (first,) = connector.started
    first.answer()
    pending = manager.alloc(ctx, defer_lookup=True)
    (_, again) = connector.started
    assert not pending.is_complete()

    monkeypatch.setattr(jenga_block_manager, "_CONNECTOR_LOOKUP_TIMEOUT_S", 0.0)
    assert pending.is_complete()
    assert connector.abandoned == [first, again]
    transfer = manager.alloc(ctx, defer_lookup=True)

    assert transfer.is_complete()
    assert connector.blocking_lookups == 0
    assert not connector.loads
    assert ctx.tokens.processed_length == 0
    assert manager.num_req_blocks(ctx) == len(tokens)


def test_a_request_whose_hit_keeps_shrinking_computes_the_rest() -> None:
    """Constant eviction cannot hold a request out of the batch for good.

    Each shrink of the device hit under an answered lookup asks the
    connector again, but only twice for one request. The third shrink
    serves what is still on device and computes the rest, without asking
    again and without the blocking lookup. A request held out again keeps
    none of the device hit it found, so it pins nothing while it waits.
    """
    connector = FakeConnector([FULL], asynchronous=False, deferred_lookups=True)
    manager = make_manager(connector, num_huge_blocks=32)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    held = _hold_in_connector_with_a_device_hit(manager, connector, tokens, 6)
    connector.loads.clear()
    connector.blocking_lookups = 0
    pool = manager.pools[0]
    cache = pool.prefix_caches[FULL]
    free_before = pool.num_free_blocks(FULL)

    ctx = make_ctx(tokens)
    manager.claim(ctx)
    manager.alloc(ctx, defer_lookup=True)
    for depth in (5, 4):
        connector.started[-1].answer()
        pool.uncommit_block(cache[held[depth]])
        assert manager.starts_lookup(ctx)
        pending = manager.alloc(ctx, defer_lookup=True)

        assert not pending.is_complete()
        assert connector.lookups[-1][0] == held[depth:-1]
        assert manager.num_req_blocks(ctx) == 0
        assert pool.num_free_blocks(FULL) == free_before

    connector.started[-1].answer()
    pool.uncommit_block(cache[held[3]])
    assert not manager.starts_lookup(ctx)
    transfer = manager.alloc(ctx, defer_lookup=True)

    assert transfer.is_complete()
    assert len(connector.started) == 3
    assert connector.abandoned == connector.started
    assert connector.blocking_lookups == 0
    assert not connector.loads
    assert ctx.tokens.processed_length == 3
    assert ctx.cached_prefix_length == 3
    assert ctx.cached_prefix_external_length == 0


def test_a_lookup_answering_right_after_alloc_still_reads_pending_once() -> (
    None
):
    """The caller's first poll cannot see an answer alloc did not act on.

    The batch constructor polls the transfer straight after alloc to decide
    whether to cordon the request. A lookup answered in between would read
    complete there and send a request holding no pages into the batch, so
    that first poll repeats what alloc saw and the next one moves on.
    """
    connector = FakeConnector([FULL], deferred_lookups=True)
    manager = make_manager(connector)
    _hold_only_in_the_connector(manager, connector, [1, 2, 3, 4])

    ctx = make_ctx([1, 2, 3, 4])
    manager.claim(ctx)
    pending = manager.alloc(ctx, defer_lookup=True)
    (lookup,) = connector.started
    lookup.answer()

    assert not pending.is_complete()
    assert manager.num_req_blocks(ctx) == 0
    assert pending.is_complete()


def test_release_hands_back_an_unanswered_lookup() -> None:
    """A request cancelled while its lookup is out owes the handle back."""
    connector = FakeConnector([FULL], deferred_lookups=True)
    manager = make_manager(connector)
    _hold_only_in_the_connector(manager, connector, [1, 2, 3, 4])

    ctx = make_ctx([1, 2, 3, 4])
    manager.claim(ctx)
    manager.alloc(ctx, defer_lookup=True)
    (lookup,) = connector.started
    manager.release(ctx)

    assert connector.abandoned == [lookup]
    assert not manager.contains(ctx)


class _FakePanic(BaseException):
    """Stands in for pyo3's ``PanicException``, which is not an ``Exception``."""


def test_release_forgets_the_request_when_its_hand_back_panics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pages are freed before the hand back, so the request must go too.

    Otherwise the manager still reads it as claimed over pages it no longer
    holds, and releasing it again frees those pages a second time.
    """
    connector = FakeConnector([FULL], deferred_lookups=True)
    manager = make_manager(connector)
    _hold_only_in_the_connector(manager, connector, [1, 2, 3, 4])

    ctx = make_ctx([1, 2, 3, 4])
    manager.claim(ctx)
    manager.alloc(ctx, defer_lookup=True)

    def panic(lookup: KVLookup) -> None:
        raise _FakePanic

    monkeypatch.setattr(connector, "abandon_lookup", panic)

    with pytest.raises(_FakePanic):
        manager.release(ctx)

    assert not manager.contains(ctx)


def test_a_lookup_that_answers_a_miss_is_handed_back() -> None:
    """A miss owes the handle back and then allocates for the compute."""
    connector = FakeConnector([FULL], deferred_lookups=True)
    manager = make_manager(connector)

    ctx = make_ctx([1, 2, 3, 4])
    manager.claim(ctx)
    pending = manager.alloc(ctx, defer_lookup=True)
    assert not pending.is_complete()
    (lookup,) = connector.started
    lookup.answer()
    transfer = manager.alloc(ctx, defer_lookup=True)

    assert transfer.is_complete()
    assert connector.abandoned == [lookup]
    assert not connector.loads
    assert ctx.tokens.processed_length == 0
    assert manager.num_req_blocks(ctx) == 4


@pytest.mark.parametrize("fails_at", ["post", "first_poll"])
def test_a_load_that_fails_hands_back_the_lookup(fails_at: str) -> None:
    """A failed load out of a lookup leaves no lease outstanding."""
    connector = FakeConnector([FULL], deferred_lookups=True)
    manager = make_manager(connector)
    _hold_only_in_the_connector(manager, connector, [1, 2, 3, 4])
    if fails_at == "post":
        connector.load_error = KVLoadFailed("memory transfer failed")
    else:
        connector.fail_on_first_poll = True

    ctx = make_ctx([1, 2, 3, 4])
    manager.claim(ctx)
    manager.alloc(ctx, defer_lookup=True)
    (lookup,) = connector.started
    lookup.answer()
    transfer = manager.alloc(ctx, defer_lookup=True)

    assert transfer.is_complete()
    assert connector.loaded_from == [lookup]
    assert connector.abandoned == [lookup]
    assert ctx.tokens.processed_length == 0
    assert manager.take_metrics().connector_load_failures == 1


@pytest.mark.parametrize("breaks_at", ["masks", "load"])
def test_a_lookup_consumed_by_a_raising_alloc_is_handed_back(
    breaks_at: str,
) -> None:
    """A bug surfacing while the answer is consumed strands no lease.

    The lookup is taken off the request before it is read, so whatever
    raises between reading it and loading out of it has to hand it back
    on the way out.
    """
    connector = FakeConnector([FULL], deferred_lookups=True)
    manager = make_manager(connector)
    _hold_only_in_the_connector(manager, connector, [1, 2, 3, 4])

    ctx = make_ctx([1, 2, 3, 4])
    manager.claim(ctx)
    manager.alloc(ctx, defer_lookup=True)
    (lookup,) = connector.started
    lookup.answer()
    if breaks_at == "masks":
        lookup._masks = {}
        expected: type[Exception] = KeyError
    else:
        connector.load_error = ValueError("rows do not match the tree")
        expected = ValueError

    with pytest.raises(expected):
        manager.alloc(ctx, defer_lookup=True)

    assert connector.abandoned == [lookup]


def test_a_run_the_pool_cannot_hold_hands_back_the_lookup() -> None:
    """Dropping a run that does not fit still owes the handle back."""
    connector = FakeConnector(
        [VALUES, SCALES], asynchronous=False, deferred_lookups=True
    )
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

    tight = make_manager(connector, leaves=(VALUES, SCALES), num_huge_blocks=3)
    replay = make_ctx([1, 2, 3, 4, 5])
    tight.claim(replay)
    tight.alloc(replay, defer_lookup=True)
    (lookup,) = connector.started
    lookup.answer()
    with pytest.raises(InsufficientBlocksError):
        tight.alloc(replay, defer_lookup=True)

    assert connector.abandoned == [lookup]
    assert not connector.loads
    tight.release(replay)
    assert connector.abandoned == [lookup], "nothing left to hand back"


def test_the_synchronous_alloc_never_waits_on_a_deferred_lookup() -> None:
    """A caller that reads the blocks right after ``alloc`` is served at once.

    The DI decode node hands the request's block ids to prefill as soon as
    ``alloc`` returns, so it cannot be told to come back later.
    """
    connector = FakeConnector([FULL], asynchronous=False, deferred_lookups=True)
    manager = make_manager(connector)
    held = _hold_only_in_the_connector(manager, connector, [1, 2, 3, 4])

    ctx = make_ctx([1, 2, 3, 4])
    manager.claim(ctx)
    transfer = manager.alloc(ctx)

    assert transfer.is_complete()
    assert not connector.started
    assert connector.loads[-1][1][FULL] == held[:-1]
    assert ctx.tokens.processed_length == len(held) - 1


def test_a_lookup_whose_poll_raises_is_served_as_a_miss(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A handle that raises reads as an answered miss rather than an error.

    The scheduler polls it on every iteration, so raising out of the poll
    would take the scheduler down. It is handed back once, logged once and
    never polled again, and the next alloc computes the prompt.
    """
    connector = FakeConnector([FULL], deferred_lookups=True)
    manager = make_manager(connector)
    _hold_only_in_the_connector(manager, connector, [1, 2, 3, 4])

    ctx = make_ctx([1, 2, 3, 4])
    manager.claim(ctx)
    pending = manager.alloc(ctx, defer_lookup=True)
    (lookup,) = connector.started
    assert not pending.is_complete()
    lookup.error = RuntimeError("the extension lost the handle")

    with caplog.at_level(logging.ERROR, logger="max.pipelines"):
        assert pending.is_complete()
        assert pending.is_complete()
    polls = lookup.polls
    transfer = manager.alloc(ctx, defer_lookup=True)

    assert lookup.polls == polls
    assert transfer.is_complete()
    assert connector.abandoned == [lookup]
    assert not connector.loads
    assert ctx.tokens.processed_length == 0
    assert manager.num_req_blocks(ctx) == 4
    assert len(caplog.records) == 1


def test_a_lookup_that_never_answers_is_served_as_a_miss_at_its_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request is not held out of the batch forever on a stuck lookup."""
    connector = FakeConnector([FULL], deferred_lookups=True)
    manager = make_manager(connector)
    _hold_only_in_the_connector(manager, connector, [1, 2, 3, 4])

    ctx = make_ctx([1, 2, 3, 4])
    manager.claim(ctx)
    pending = manager.alloc(ctx, defer_lookup=True)
    (lookup,) = connector.started
    assert not pending.is_complete()
    assert not connector.abandoned

    monkeypatch.setattr(jenga_block_manager, "_CONNECTOR_LOOKUP_TIMEOUT_S", 0.0)
    assert pending.is_complete()
    assert connector.abandoned == [lookup]
    transfer = manager.alloc(ctx, defer_lookup=True)

    assert transfer.is_complete()
    assert connector.abandoned == [lookup]
    assert not connector.loads
    assert ctx.tokens.processed_length == 0
    assert manager.num_req_blocks(ctx) == 4


def test_a_request_handed_back_too_often_keeps_its_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Past its hand backs a request keeps its last answer and loads it.

    A request passed over on every pass would otherwise ask for good.
    """
    monkeypatch.setattr(jenga_block_manager, "_LOOKUP_HAND_BACK_AGE_S", 0.0)
    connector = FakeConnector([FULL], asynchronous=False, deferred_lookups=True)
    manager = make_manager(connector, num_huge_blocks=32)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    held = _hold_only_in_the_connector(manager, connector, tokens)
    ctx = make_ctx(tokens)
    manager.claim(ctx)

    for _ in range(jenga_block_manager._MAX_CONNECTOR_LOOKUP_HAND_BACKS):
        manager.alloc(ctx, defer_lookup=True)
        connector.started[-1].answer()
        manager.hand_back_lookup(ctx)
    manager.alloc(ctx, defer_lookup=True)
    last = connector.started[-1]
    last.answer()
    assert manager.hand_back_lookup(ctx)

    assert last not in connector.abandoned
    assert manager.awaits_lookup(ctx)
    transfer = manager.alloc(ctx, defer_lookup=True)
    assert transfer.is_complete()
    assert connector.loaded_from == [last]
    assert connector.loads[-1][1][FULL] == held[:-1]


def test_a_fresh_answer_is_kept_when_handed_back() -> None:
    """An answer younger than the hand back age stays with its request.

    Its lease still has room to load, and asking again would cost another
    leased read, so the hand back keeps it, reports that it did, and the
    next alloc loads it.
    """
    connector = FakeConnector([FULL], asynchronous=False, deferred_lookups=True)
    manager = make_manager(connector, num_huge_blocks=32)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    held = _hold_only_in_the_connector(manager, connector, tokens)
    ctx = make_ctx(tokens)
    manager.claim(ctx)
    manager.alloc(ctx, defer_lookup=True)
    (first,) = connector.started
    first.answer()

    assert not manager.hand_back_lookup(ctx)

    assert connector.abandoned == []
    transfer = manager.alloc(ctx, defer_lookup=True)
    assert transfer.is_complete()
    assert connector.started == [first]
    assert connector.loaded_from == [first]
    assert connector.loads[-1][1][FULL] == held[:-1]


class _Clock:
    """Stands in for the block manager's ``time``, moved only by the test."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now


def test_a_lookups_age_counts_the_time_its_start_took(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lookup is aged from before it was started, not after.

    The connector may stamp the lease before its start returns, so an age
    taken after the start could read young by however long it took, and
    an answer kept as fresh could outlive its lease.
    """
    clock = _Clock()
    monkeypatch.setattr(jenga_block_manager, "time", clock)
    connector = FakeConnector([FULL], asynchronous=False, deferred_lookups=True)
    manager = make_manager(connector, num_huge_blocks=32)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    _hold_only_in_the_connector(manager, connector, tokens)
    start_lookup = connector.start_lookup

    def slow_start_lookup(
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
        hint: bytes | None = None,
    ) -> KVLookup:
        clock.now += 4.0
        return start_lookup(block_hashes, replica_idx=replica_idx, hint=hint)

    monkeypatch.setattr(connector, "start_lookup", slow_start_lookup)
    ctx = make_ctx(tokens)
    manager.claim(ctx)
    manager.alloc(ctx, defer_lookup=True)
    (first,) = connector.started
    first.answer()
    clock.now += jenga_block_manager._LOOKUP_HAND_BACK_AGE_S - 4.0

    assert manager.hand_back_lookup(ctx)
    assert connector.abandoned == [first]


def test_an_answer_too_old_to_load_is_asked_again_when_consumed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An answer a request waited on past the hand back age is not loaded.

    Its lease may lapse before the load reaches the store, and a refused
    load serves the request as a miss. So the alloc that would consume it
    hands it back and asks again with the request still held out, and the
    check the scheduler asks before allocating reads that as a lookup
    started, so the bound on lookups out covers it.
    """
    connector = FakeConnector([FULL], asynchronous=False, deferred_lookups=True)
    manager = make_manager(connector, num_huge_blocks=32)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    held = _hold_only_in_the_connector(manager, connector, tokens)
    connector.blocking_lookups = 0
    ctx = make_ctx(tokens)
    manager.claim(ctx)
    first_pending = manager.alloc(ctx, defer_lookup=True)
    assert not first_pending.is_complete()
    (first,) = connector.started
    first.answer()
    assert first_pending.is_complete(), "the scheduler readmits on this poll"
    age = jenga_block_manager._LOOKUP_HAND_BACK_AGE_S
    monkeypatch.setattr(jenga_block_manager, "_LOOKUP_HAND_BACK_AGE_S", 0.0)

    assert manager.starts_lookup(ctx)
    pending = manager.alloc(ctx, defer_lookup=True)

    assert not pending.is_complete()
    assert manager.awaits_lookup(ctx)
    assert connector.abandoned == [first]
    assert connector.loaded_from == []
    assert connector.lookups[-1] == (held[:-1], 0, None)
    assert connector.blocking_lookups == 0
    assert manager.num_req_blocks(ctx) == 0
    assert ctx.tokens.processed_length == 0
    assert manager._state_of(ctx).connector_lookup_hand_backs == 1

    monkeypatch.setattr(jenga_block_manager, "_LOOKUP_HAND_BACK_AGE_S", age)
    (_, second) = connector.started
    second.answer()
    transfer = manager.alloc(ctx, defer_lookup=True)

    assert transfer.is_complete()
    assert connector.loaded_from == [second]
    assert connector.loads[-1][1][FULL] == held[:-1]


def test_an_answer_too_old_to_load_is_loaded_past_the_hand_backs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request whose answers keep going stale loads its last one.

    Asking again when consuming spends the hand backs, so a request whose
    queue keeps it waiting cannot ask for good. Its last answer is loaded
    however old, and a lease that lapsed is refused and served as a miss.
    """
    monkeypatch.setattr(jenga_block_manager, "_LOOKUP_HAND_BACK_AGE_S", 0.0)
    connector = FakeConnector([FULL], asynchronous=False, deferred_lookups=True)
    manager = make_manager(connector, num_huge_blocks=32)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    held = _hold_only_in_the_connector(manager, connector, tokens)
    connector.loads.clear()
    ctx = make_ctx(tokens)
    manager.claim(ctx)
    manager.alloc(ctx, defer_lookup=True)
    for _ in range(jenga_block_manager._MAX_CONNECTOR_LOOKUP_HAND_BACKS):
        connector.started[-1].answer()
        assert not manager.alloc(ctx, defer_lookup=True).is_complete()
    last = connector.started[-1]
    last.answer()
    connector.evict_before_load = set(held)

    assert not manager.starts_lookup(ctx)
    transfer = manager.alloc(ctx, defer_lookup=True)

    assert transfer.is_complete()
    assert (
        len(connector.started)
        == 1 + jenga_block_manager._MAX_CONNECTOR_LOOKUP_HAND_BACKS
    )
    assert connector.loaded_from == [last]
    assert connector.abandoned == connector.started
    assert ctx.tokens.processed_length == 0
    assert manager.num_req_blocks(ctx) == len(tokens)


def test_hand_backs_leave_the_re_asks_a_shrunken_hit_needs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request handed back as often as allowed still re-asks past a shrink.

    Hand backs have their own budget, so a request a queue passed over can
    still start its lookup over from a device hit an eviction shortened,
    rather than compute everything past it.
    """
    monkeypatch.setattr(jenga_block_manager, "_LOOKUP_HAND_BACK_AGE_S", 0.0)
    connector = FakeConnector([FULL], asynchronous=False, deferred_lookups=True)
    manager = make_manager(connector, num_huge_blocks=32)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    held = _hold_in_connector_with_a_device_hit(manager, connector, tokens, 3)
    connector.loads.clear()
    ctx = make_ctx(tokens)
    manager.claim(ctx)
    for _ in range(jenga_block_manager._MAX_CONNECTOR_LOOKUP_HAND_BACKS):
        manager.alloc(ctx, defer_lookup=True)
        connector.started[-1].answer()
        assert manager.hand_back_lookup(ctx)
    manager.alloc(ctx, defer_lookup=True)
    manager.pools[0].reset_prefix_cache()
    connector.started[-1].answer()

    pending = manager.alloc(ctx, defer_lookup=True)

    assert not pending.is_complete()
    assert connector.lookups[-1] == (held[:-1], 0, None)
    connector.started[-1].answer()
    transfer = manager.alloc(ctx, defer_lookup=True)
    assert transfer.is_complete()
    assert connector.loads[-1][1][FULL] == held[:-1]
    assert ctx.tokens.processed_length == len(held) - 1


def test_a_lookup_that_gave_up_is_not_asked_again_after_a_hand_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Handing back a lookup past its deadline keeps the request a miss.

    The store did not answer in time, so asking it again for the same
    request would only add to its load.
    """
    connector = FakeConnector([FULL], deferred_lookups=True)
    manager = make_manager(connector)
    _hold_only_in_the_connector(manager, connector, [1, 2, 3, 4])
    ctx = make_ctx([1, 2, 3, 4])
    manager.claim(ctx)
    pending = manager.alloc(ctx, defer_lookup=True)
    assert not pending.is_complete()
    monkeypatch.setattr(jenga_block_manager, "_CONNECTOR_LOOKUP_TIMEOUT_S", 0.0)
    assert pending.is_complete()

    manager.hand_back_lookup(ctx)

    assert not manager.starts_lookup(ctx)
    transfer = manager.alloc(ctx, defer_lookup=True)
    assert transfer.is_complete()
    assert len(connector.started) == 1
    assert ctx.tokens.processed_length == 0


def test_a_synchronous_alloc_hands_back_an_outstanding_lookup() -> None:
    """An alloc that cannot wait drops the lookup and asks synchronously.

    Its caller reads the request's blocks as soon as alloc returns, so an
    answer still outstanding from an earlier deferred alloc is handed back
    and the synchronous lookup serves the request at once.
    """
    connector = FakeConnector([FULL], asynchronous=False, deferred_lookups=True)
    manager = make_manager(connector)
    held = _hold_only_in_the_connector(manager, connector, [1, 2, 3, 4])
    connector.lookups.clear()

    ctx = make_ctx([1, 2, 3, 4])
    manager.claim(ctx)
    manager.alloc(ctx, defer_lookup=True)
    (lookup,) = connector.started
    transfer = manager.alloc(ctx)

    assert transfer.is_complete()
    assert connector.abandoned == [lookup]
    assert connector.loaded_from == []
    assert len(connector.lookups) == 2
    assert connector.loads[-1][1][FULL] == held[:-1]
    assert ctx.tokens.processed_length == len(held) - 1


def make_dp_manager(
    connector: FakeConnector, copies_across_replicas: bool = False
) -> JengaBlockManager:
    """Two replicas tiling the same huge blocks.

    ``copies_across_replicas`` backs each replica with host memory so a
    prefix one replica holds can be copied to the other.
    """
    leaf_infos = {FULL: KVLeafInfo(1, KVCacheGroupId.full())}
    pools = create_pools(leaf_infos, 16, num_replicas=2)
    return JengaBlockManager(
        pools=pools,
        groups=create_groups(leaf_infos, pools, 1, enable_prefix_caching=True),
        leaves={
            FULL: PagedKVLeafRegion(
                leaf_id=FULL,
                group_id=KVCacheGroupId.full(),
                bytes_per_page=1,
                page_size=1,
            )
        },
        block_size=1,
        enable_prefix_caching=True,
        connector=connector,
        replica_kv_memory=[
            {
                FULL: KVCacheMemory(
                    replicated=False,
                    buffers=[Buffer.from_numpy(np.zeros((16, 1), np.uint8))],
                )
            }
            for _ in range(2)
        ],
        enable_dp_cross_replica_prefix_copy=copies_across_replicas,
    )


def test_a_peer_hit_that_no_longer_fits_is_served_as_a_miss() -> None:
    """A device hit counted on another replica can come up short when taken.

    The count saw the whole prompt on the other replica, so no lookup was
    started. The copy then finds no room, and the gap is served as a miss
    rather than asked about, neither by the blocking lookup nor off the
    thread, so the request is refused for want of pages like any other
    miss. Once the copy fits, the device hit serves it.
    """
    connector = FakeConnector([FULL], asynchronous=False, deferred_lookups=True)
    manager = make_dp_manager(connector, copies_across_replicas=True)
    tokens = [1, 2, 3, 4, 5, 6]
    held = _hold_only_in_the_connector(manager, connector, tokens)
    peer = make_ctx(tokens)
    manager.claim(peer, replica_idx=1)
    manager.alloc(peer)
    manager.release(peer)
    assert all(h in manager.pools[1].prefix_caches[FULL] for h in held[:-1])
    pool = manager.pools[0]
    filler = [pool.alloc_block(FULL) for _ in range(pool.num_free_blocks(FULL))]
    connector.blocking_lookups = 0
    lookups = len(connector.lookups)

    ctx = make_ctx(tokens)
    manager.claim(ctx)
    with pytest.raises(InsufficientBlocksError):
        manager.alloc(ctx, defer_lookup=True)

    assert connector.blocking_lookups == 0
    assert not connector.started
    assert len(connector.lookups) == lookups
    assert manager.num_req_blocks(ctx) == 0

    for block in filler:
        pool.free_block(block)
    transfer = manager.alloc(ctx, defer_lookup=True)

    assert transfer.is_complete()
    assert connector.blocking_lookups == 0
    assert not connector.started
    assert ctx.tokens.processed_length == len(held) - 1
    assert ctx.cached_prefix_external_length == 0


def test_a_gap_after_the_take_is_asked_in_place_when_the_connector_can(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A connector that answers in place is asked about a short take's gap.

    The count is made to see the whole prompt, as it does when a peer
    replica holds it, so no lookup is started, and the take then finds
    nothing, as a copy that found no room does. Asking a connector that
    answers in place costs no round trip, so the gap is loaded rather than
    computed, as the synchronous alloc does.
    """
    connector = FakeConnector([FULL], asynchronous=False)
    manager = make_manager(connector, num_huge_blocks=32)
    held = _hold_only_in_the_connector(manager, connector, [1, 2, 3, 4, 5])
    connector.blocking_lookups = 0
    monkeypatch.setattr(
        manager,
        "_count_device_prefix_blocks",
        lambda desired_hashes, replica_idx: len(desired_hashes),
    )

    ctx = make_ctx([1, 2, 3, 4, 5])
    manager.claim(ctx)
    transfer = manager.alloc(ctx, defer_lookup=True)

    assert transfer.is_complete()
    assert connector.blocking_lookups == 1
    assert connector.loads[-1][1][FULL] == held[:-1]
    assert ctx.cached_prefix_external_length == len(held) - 1


def test_a_shrunken_peer_hit_is_asked_again_before_anything_is_copied() -> None:
    """A request is held out again before any page is copied into it.

    The lookup was asked past two blocks the count saw on the other
    replica, and one of them is evicted there before the answer is read.
    The shrink is caught before the device hit is taken, so the request is
    held out on a new lookup having copied nothing. A page a copy was
    enqueued into stays held by the request, because the copy can land
    after the page is reused if it goes back to the pool, and then
    overwrites what the new owner put there.
    """
    connector = FakeConnector([FULL], asynchronous=False, deferred_lookups=True)
    manager = make_dp_manager(connector, copies_across_replicas=True)
    tokens = [1, 2, 3, 4, 5, 6, 7]
    held = _hold_only_in_the_connector(manager, connector, tokens)
    asked = held[:-1]
    peer = make_ctx(tokens[:3])
    manager.claim(peer, replica_idx=1)
    manager.alloc(peer)
    manager.release(peer)
    peer_cache = manager.pools[1].prefix_caches[FULL]
    assert all(h in peer_cache for h in asked[:2])
    connector.blocking_lookups = 0
    manager.take_metrics()
    ctx = make_ctx(tokens)
    manager.claim(ctx)

    def alloc_holding_every_copy() -> tuple[KVTransfer, int]:
        transfer = manager.alloc(ctx, defer_lookup=True)
        copied = manager.take_metrics().cross_replica_blocks_copied
        if copied:
            local = manager.pools[0].prefix_caches[FULL][asked[0]]
            assert local.ref_cnt > 0, "the copied page went back to the pool"
            assert manager.num_req_blocks(ctx) > 0
        return transfer, copied

    pending, copied = alloc_holding_every_copy()

    assert not pending.is_complete()
    assert copied == 0
    (first,) = connector.started
    assert connector.lookups[-1] == (asked[2:], 0, None)

    manager.pools[1].uncommit_block(peer_cache[asked[1]])
    first.answer()
    pending, copied = alloc_holding_every_copy()

    assert not pending.is_complete()
    assert copied == 0, "nothing is copied before the request is held out"
    assert connector.abandoned == [first]
    assert connector.lookups[-1] == (asked[1:], 0, None)
    assert manager.num_req_blocks(ctx) == 0

    (_, again) = connector.started
    again.answer()
    transfer, copied = alloc_holding_every_copy()

    assert transfer.is_complete()
    assert copied == 1
    assert connector.loaded_from == [again]
    assert connector.loads[-1][1][FULL] == asked[1:]
    assert connector.blocking_lookups == 0
    assert ctx.tokens.processed_length == len(asked)
    assert ctx.cached_prefix_external_length == len(asked) - 1


def _price_setup(
    connector: FakeConnector,
) -> tuple[JengaBlockManager, TextContext, list[bytes]]:
    """The connector holds a whole run, and replica 1 holds its first two blocks.

    Returns the hashes a fresh request over the run asks about.
    """
    manager = make_dp_manager(connector)
    tokens = [1, 2, 3, 4, 5, 6]
    held = _hold_only_in_the_connector(manager, connector, tokens)
    device = make_ctx(tokens[:3])
    manager.claim(device, replica_idx=1)
    manager.alloc(device)
    manager.release(device)
    assert all(h in manager.pools[1].prefix_caches[FULL] for h in held[:2])
    connector.lookups.clear()
    connector.probes.clear()
    return manager, make_ctx(tokens), held[:-1]


def test_pricing_probes_the_connector_once_for_every_replica() -> None:
    """One probe over the whole chain prices every replica.

    The connector's tiers are shared, so one answer serves them all, and
    each replica reads it from where its own device hit ends.
    """
    connector = FakeConnector([FULL], asynchronous=False, probe_answers=True)
    manager, ctx, asked = _price_setup(connector)

    hits = manager.get_prefix_cache_hit_counts(ctx)

    assert connector.probes == [(asked, [0, 1], None)]
    assert not connector.lookups
    assert hits == [
        PrefixCacheHits(device_blocks=0, external_blocks=len(asked)),
        PrefixCacheHits(device_blocks=2, external_blocks=len(asked) - 2),
    ]


def test_pricing_reconciles_only_the_leaves_a_probe_covers() -> None:
    """A probe may answer for one representative leaf.

    The dKV connector prices from its first full-attention leaf when its
    server predates presence, so the leaves the answer leaves out neither
    decide the hit nor fail the pricing.
    """
    connector = FakeConnector(
        [FULL, SCALES], asynchronous=False, probe_answers=True
    )
    connector.probe_leaves = [FULL]
    manager = make_manager(connector, leaves=(FULL, SCALES))
    tokens = [1, 2, 3, 4, 5, 6]
    held = _hold_only_in_the_connector(manager, connector, tokens)
    connector.held[SCALES].clear()
    connector.lookups.clear()

    (hits,) = manager.get_prefix_cache_hit_counts(make_ctx(tokens))

    assert hits == PrefixCacheHits(
        device_blocks=0, external_blocks=len(held) - 1
    )
    assert len(connector.probes) == 1
    assert not connector.lookups


def test_pricing_falls_back_to_a_lookup_per_replica_without_a_probe() -> None:
    """A connector with no probe is priced the way it always was."""
    connector = FakeConnector([FULL], asynchronous=False)
    manager, ctx, asked = _price_setup(connector)

    hits = manager.get_prefix_cache_hit_counts(ctx)

    assert len(connector.probes) == 1
    assert connector.lookups == [(asked, 0, None), (asked[2:], 1, None)]
    assert hits == [
        PrefixCacheHits(device_blocks=0, external_blocks=len(asked)),
        PrefixCacheHits(device_blocks=2, external_blocks=len(asked) - 2),
    ]
