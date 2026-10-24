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
"""KVCache management based on the Jenga paper.

This module implements the JengaBlockManager on top of the JengaBlockPool.
It is used to manage the allocation and release of blocks to requests.
It also manages the prefix cache hits for the requests.

We use a two level huge-little block hierarchy to allocate the blocks among the
different caches. This allows the memory to be fungible between the caches.
"""

from __future__ import annotations

import logging
import time
from bisect import bisect_left
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import chain

from max._kv_core import longest_joint_prefix_hit
from max.driver import Buffer, batch_inplace_copy
from max.nn.kv_cache import KVCacheGroupId, KVLeafRegion
from max.nn.kv_cache.cache_params import KVCacheMemory
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.pipelines.context import TextContext
from max.pipelines.kv_cache.kv_connector import (
    BlockCount,
    CompletedTransfer,
    KVConnector,
    KVLoadFailed,
    KVLoadRefused,
    KVLookup,
    KVTransfer,
)
from max.pipelines.modeling.types import RequestID
from max.profiler import traced
from max.support.math import ceildiv

from ._common import (
    PrefixCacheHits,
    _compute_seq_len,
    _OnloadSplice,
    _resolve_only_use_kv_connector_last_level_cache,
    compute_block_hashes,
)
from .block_utils import InsufficientBlocksError, KVHashAlgo, LittleKVCacheBlock
from .jenga_block_pool import JengaBlockPool, _pristine_pool_can_satisfy
from .kv_group_coordinator import (
    FullKVGroupCoordinator,
    KVGroupCoordinatorInterface,
    SlidingWindowKVGroupCoordinator,
)
from .recurrent_coordinator import (
    RecurrentKVGroupCoordinator,
)
from .scratch_coordinator import (
    ScratchKVGroupCoordinator,
)

logger = logging.getLogger("max.pipelines")

# A request waits this long on one lookup before it is served as a miss. Half
# the dKV client's 10 s plan TTL, so an answer still leaves time to load it.
# Each re-ask gets its own deadline.
_CONNECTOR_LOOKUP_TIMEOUT_S = 5.0

# How often one request's connector lookup is asked again because the device
# hit it was asked past shrank before the answer was read. Past that a shrink
# is computed, so constant eviction cannot hold the request out of the batch
# for good.
_MAX_CONNECTOR_LOOKUP_REASKS = 2

# How often the scheduler may hand back one request's answered lookup. Its own
# budget, so a request passed over in its queue keeps the re-asks a later
# shrink needs. Past that an answer is kept, so a queue that keeps passing the
# request over cannot hold it out of the batch for good.
_MAX_CONNECTOR_LOOKUP_HAND_BACKS = 2

# An answered lookup younger than this is kept for a request still waiting in
# its queue rather than handed back, because asking again costs another
# leased read with its pins and G2 stages. One older is asked again when it is
# consumed, counted as a hand back, because its lease may lapse before the
# load reaches the store and a refused load serves the request as a miss.
# Counted from the lookup's start, which precedes its lease, so at least 3 s
# of the 10 s lease remain, several forward passes for a request popped just
# under it to load. The lease is the shorter of dkv-connector's
# DEFAULT_PLAN_TTL and the server's DKV_READER_LEASE_DEADLINE, and neither
# reaches Python, so this has to follow them by hand.
_LOOKUP_HAND_BACK_AGE_S = 7.0


@dataclass
class _PendingTransfer:
    """An in-flight async connector transfer and the pages it pins.

    ``blocks`` are pinned per leaf until the copy lands, so nothing evicts or
    reuses them mid-copy. ``commit_hashes`` is set only for onloads, whose
    prefix-cache publish is deferred until the H2D has actually landed.
    """

    event: KVTransfer
    blocks: dict[str, list[LittleKVCacheBlock]]
    commit_hashes: list[bytes] | None = None
    splice: _OnloadSplice | None = None

    def pinned_blocks(self) -> Iterator[LittleKVCacheBlock]:
        """Every block this transfer pins, across all its leaves."""
        return chain.from_iterable(self.blocks.values())


class _PendingLookup:
    """A connector lookup one ``alloc`` started for a later one to consume.

    Asked over the request's hashes from ``first_block`` on, counted from its
    committed index, because the device tier was expected to serve the blocks
    before that. The ``alloc`` that consumes it takes the device hit for real
    and reads the answer from wherever that hit ends.

    It reads answered once the connector answers, once polling the handle
    raises, or once it has been out for ``_CONNECTOR_LOOKUP_TIMEOUT_S``. The
    last two hand the handle back at once and read as a miss, so a request
    is never held out of the batch for good and an idle engine does not
    keep polling a lookup that will never answer.
    """

    def __init__(
        self,
        connector: KVConnector,
        handle: KVLookup,
        first_block: int,
        started_at: float,
    ) -> None:
        self.handle = handle
        self.first_block = first_block
        self.missed = False
        self._connector = connector
        self._started_at = started_at
        self._answered = False
        self._returned = False

    def is_answered(self) -> bool:
        """Returns whether the lookup can be consumed.

        Never blocks and never raises.
        """
        if self._answered:
            return True

        try:
            complete = self.handle.is_complete()
        except Exception:
            logger.exception(
                "polling a %s lookup failed, so its request is served as a "
                "cache miss",
                self._connector.name,
            )
            self._give_up()
            return True

        if complete:
            self._answered = True
        elif time.monotonic() - self._started_at >= _CONNECTOR_LOOKUP_TIMEOUT_S:
            logger.warning(
                "a %s lookup did not answer within %.1f s, so its request is "
                "served as a cache miss",
                self._connector.name,
                _CONNECTOR_LOOKUP_TIMEOUT_S,
            )
            self._give_up()
        return self._answered

    def age_s(self) -> float:
        """Returns how long ago the lookup started, in seconds."""
        return time.monotonic() - self._started_at

    def is_stale(self) -> bool:
        """Returns whether the answer is too old to load.

        Never polls the handle, so asking before an alloc takes nothing.
        """
        return (
            self._answered
            and not self.missed
            and self.age_s() >= _LOOKUP_HAND_BACK_AGE_S
        )

    def abandon(self) -> None:
        """Hands the handle back, once however often this is called."""
        if self._returned:
            return
        self._returned = True
        self._connector.abandon_lookup(self.handle)

    def _give_up(self) -> None:
        """Hands the handle back and reads as a miss from now on."""
        self.missed = True
        self._answered = True
        self.abandon()


class _PendingLookupTransfer:
    """What ``alloc`` returns while the connector is still answering a lookup.

    Nothing is in flight into device memory, so there is nothing to pin and
    nothing to drain. It reads complete once the lookup can be consumed,
    which is the scheduler's cue to run ``alloc`` again.
    """

    def __init__(self, lookup: _PendingLookup) -> None:
        self._lookup = lookup
        self._polled = False

    def is_complete(self) -> bool:
        """Returns whether the lookup can be consumed.

        Never blocks and never raises. The first poll repeats the unanswered
        reading ``alloc`` took rather than asking again. A lookup can answer
        on another thread between the two, and the caller polls straight
        after ``alloc`` to decide whether to cordon a request that holds no
        pages yet, so reading it answered there would run the request with
        nothing allocated.
        """
        if not self._polled:
            self._polled = True
            return False
        return self._lookup.is_answered()

    def synchronize(self) -> None:
        """Returns at once, since no copy is in flight to wait for."""
        return


@dataclass(frozen=True)
class _PageCopy:
    """One leaf's page, fetched from another replica's device memory.

    ``src_replica`` is per page so leaves of the same hit may come from
    different replicas.
    """

    leaf_id: str
    dst_bid: int
    src_bid: int
    src_replica: int


def create_pools(
    leaf_infos: Mapping[str, KVLeafInfo],
    num_huge_blocks: int,
    num_replicas: int = 1,
) -> list[JengaBlockPool]:
    """Returns one pool per replica, each tiling the same huge blocks."""
    ratios = {leaf_id: leaf.ratio for leaf_id, leaf in leaf_infos.items()}
    return [
        JengaBlockPool(num_huge_blocks, ratios) for _ in range(num_replicas)
    ]


def _max_seq_len_fitting_in_geometry(
    leaves: Mapping[str, KVLeafRegion],
    block_size: int,
    allocatable_huge_blocks: int,
    cache_ratios: Mapping[str, int],
    *,
    enable_prefix_caching: bool,
) -> int | None:
    """Returns the longest single request an empty pool of this geometry serves.

    Binary search over the fit, which is monotonic because no leaf's demand
    decreases with length. ``None`` when no length exhausts the pool, which
    happens when every leaf's demand plateaus (sliding windows, states).

    Args:
        leaves: The leaves the pool serves.
        block_size: Tokens per page.
        allocatable_huge_blocks: Huge blocks excluding the null block.
        cache_ratios: Little pages of each leaf per huge block.
        enable_prefix_caching: Whether states hold a checkpoint block.
    """

    def fits(seq_len: int) -> bool:
        num_blocks = ceildiv(seq_len, block_size)
        demand = {
            leaf_id: leaf.blocks_to_reserve(
                num_blocks, enable_prefix_caching=enable_prefix_caching
            )
            for leaf_id, leaf in leaves.items()
        }
        return _pristine_pool_can_satisfy(
            allocatable_huge_blocks, cache_ratios, demand
        )

    # No request outruns one leaf given the whole budget at the largest ratio,
    # so that bounds the search; fitting at the bound means no finite bound.
    upper_bound = (
        allocatable_huge_blocks * block_size * max(cache_ratios.values())
    )
    search_space = upper_bound + 2
    idx = bisect_left(
        range(search_space), True, key=lambda seq_len: not fits(seq_len)
    )
    return (idx - 1) if idx < search_space else None


def create_groups(
    leaf_infos: Mapping[str, KVLeafInfo],
    pools: Sequence[JengaBlockPool],
    page_size: int,
    *,
    enable_prefix_caching: bool,
) -> dict[str, KVGroupCoordinatorInterface]:
    """Returns one coordinator per leaf, keyed by leaf id.

    One per LEAF, not one per attention group. A coordinator holding several
    leaves had to reconcile them itself -- a hash only some of them held was
    unusable, so every residency question was an AND over its leaves -- and
    that is the same reconcile the manager already runs across coordinators.
    Splitting them means the rule is written once: the joint prefix hit over
    one-leaf coordinators agrees with what a multi-leaf coordinator computed
    internally, because ANDing inside a group then across groups is ANDing.

    TODO: "group" now means "leaf" everywhere in this module. Rename the
    concept -- ``create_groups``, ``KVGroupCoordinatorInterface``,
    ``self._groups`` -- in a follow-up. Kept here so this change stays
    structural and reviewable.
    """
    return {
        leaf_id: create_kv_group_coordinator(
            pools,
            leaf_id,
            leaf.group_id,
            page_size,
            enable_prefix_caching=enable_prefix_caching,
        )
        for leaf_id, leaf in leaf_infos.items()
    }


def create_kv_group_coordinator(
    pools: Sequence[JengaBlockPool],
    leaf_id: str,
    group_id: KVCacheGroupId,
    page_size: int,
    *,
    enable_prefix_caching: bool = False,
) -> KVGroupCoordinatorInterface:
    """Returns the group implementation matching the leaves' access pattern."""
    if group_id.is_sliding_window():
        return SlidingWindowKVGroupCoordinator(
            pools=pools,
            leaf_id=leaf_id,
            group_id=group_id,
            page_size=page_size,
            window_size=group_id.window_size,
        )
    if group_id.is_full():
        return FullKVGroupCoordinator(
            pools=pools,
            leaf_id=leaf_id,
            group_id=group_id,
            page_size=page_size,
        )
    if group_id.is_recurrent():
        return RecurrentKVGroupCoordinator(
            pools=pools,
            leaf_id=leaf_id,
            group_id=group_id,
            page_size=page_size,
            enable_prefix_caching=enable_prefix_caching,
        )
    if group_id.is_scratch():
        return ScratchKVGroupCoordinator(
            pools=pools,
            leaf_id=leaf_id,
            group_id=group_id,
            page_size=page_size,
        )
    raise ValueError(f"no coordinator holds a {group_id} group")


@dataclass
class RequestCacheState:
    """Everything the manager tracks for one live request."""

    replica_idx: int

    hashes: list[bytes] = field(default_factory=list)
    """The chained key of each full block of the request's tokens."""

    committed_idx: int = 0
    """How far the published prefix reaches, in tokens."""

    skips_connector: bool = False
    """Whether a connector load failed for this request.

    It then reads from the device tier alone until released, so a transport
    that keeps failing costs it one attempt rather than one per admission.
    """

    failed_onload: tuple[_OnloadSplice, KVLoadFailed] | None = None
    """An onload whose copy failed after it was spliced in, for the next
    ``alloc`` to roll back."""

    pending_lookup: _PendingLookup | None = None
    """A connector lookup started for this request and not yet consumed."""

    connector_lookup_reasks: int = 0
    """How often the connector was asked again because the device hit
    shrank under a lookup, which ``_MAX_CONNECTOR_LOOKUP_REASKS`` bounds."""

    connector_lookup_hand_backs: int = 0
    """How often the scheduler handed back an answered lookup, which
    ``_MAX_CONNECTOR_LOOKUP_HAND_BACKS`` bounds."""

    def lookup_is_stale(self) -> bool:
        """Returns whether the answer is too old to load and is asked again.

        Only while the request has hand backs left, so one whose queue keeps
        passing it over loads its last answer rather than asking for good.
        """
        return (
            self.pending_lookup is not None
            and self.pending_lookup.is_stale()
            and self.connector_lookup_hand_backs
            < _MAX_CONNECTOR_LOOKUP_HAND_BACKS
        )


@dataclass(frozen=True)
class KVLeafInfo:
    """How one cache tiles a huge block, and which group it belongs to.

    ``ratio`` is the number of little blocks per huge block.
    """

    ratio: int
    group_id: KVCacheGroupId


@dataclass(frozen=True, slots=True)
class _LeafOccupancy:
    """What one leaf holds of a replica's pool, in its own small pages."""

    used: int
    """Pages a request holds."""

    evictable: int
    """Unreferenced pages still in the prefix cache, which allocation evicts."""

    empty: int
    """Pages the leaf can allocate without evicting one of its own commits."""

    total: int
    """Pages the leaf would have if it held every allocatable huge block."""

    used_huge_blocks: int
    """Huge blocks typed to the leaf that a request holds."""

    evictable_huge_blocks: int
    """Huge blocks no request holds that are still typed to the leaf."""

    @property
    def pressure_pct(self) -> float:
        """Returns the share of the leaf's pages neither evictable nor empty."""
        return BlockCount(
            free=self.evictable + self.empty, total=self.total
        ).used_pct


@dataclass(frozen=True, slots=True)
class _PoolOccupancy:
    """A snapshot of one replica's pool, per leaf."""

    leaves: dict[str, _LeafOccupancy]

    empty_huge_blocks: int
    """Huge blocks no leaf has ever claimed."""


class JengaBlockManager:
    """Assigns blocks to requests and manages prefix cache hits."""

    def __init__(
        self,
        pools: Sequence[JengaBlockPool],
        block_size: int,
        enable_prefix_caching: bool = True,
        kv_hash_algo: KVHashAlgo = "ahash64",
        kv_hash_seed: bytes | None = None,
        max_num_input_tokens: int | None = None,
        num_draft_tokens: int = 0,
        num_draft_tokens_per_step: int = 0,
        connector: KVConnector | None = None,
        replica_kv_memory: Sequence[Mapping[str, KVCacheMemory]] | None = None,
        enable_dp_cross_replica_prefix_copy: bool = True,
        *,
        groups: Mapping[str, KVGroupCoordinatorInterface],
        leaves: Mapping[str, KVLeafRegion],
    ) -> None:
        """Assigns blocks out of ``pools``, one per replica.

        ``groups`` decides which blocks a request holds; ``leaves`` says what
        each one costs and how the graph reaches it. ``block_size`` is in
        tokens.
        """
        self._block_size = block_size
        self._enable_prefix_caching = enable_prefix_caching
        self._only_use_kv_connector_last_level_cache = (
            _resolve_only_use_kv_connector_last_level_cache()
        )
        self._kv_hash_algo = kv_hash_algo
        self._kv_hash_seed = kv_hash_seed
        self._max_num_input_tokens = max_num_input_tokens
        self._num_draft_tokens = num_draft_tokens
        self._num_draft_tokens_per_step = num_draft_tokens_per_step
        self._metrics = KVCacheMetrics()

        self.pools = list(pools)
        self._num_replicas = len(self.pools)

        # Per-replica device memory, keyed by leaf, used to copy committed
        # prefix blocks between replicas. None when the caller has no buffers.
        self._replica_kv_memory = replica_kv_memory
        self._cross_replica_copy_enabled = (
            enable_dp_cross_replica_prefix_copy
            and self._num_replicas > 1
            and replica_kv_memory is not None
        )

        for leaf_id, group in groups.items():
            assert list(group.pools) == self.pools, (
                f"leaf {leaf_id} draws from other pools than this"
                " manager's; both come from the same slabs or neither does"
            )
        # One coordinator per leaf, keyed by leaf id -- see `create_groups`.
        self._groups: dict[str, KVGroupCoordinatorInterface] = dict(groups)
        self._leaves = dict(leaves)
        self._leaf_ids = list(self._groups)
        # A leaf carrying no hash cannot be looked up, committed, or handed to
        # an external tier, so every content-addressed path iterates these
        # instead. Allocation and admission still cover all of them.
        self._cacheable_groups = {
            leaf_id: group
            for leaf_id, group in self._groups.items()
            if not group.group_id.is_scratch()
        }
        self._cacheable_leaf_ids = list(self._cacheable_groups)
        self._requests: dict[RequestID, RequestCacheState] = {}

        # State for the KVConnector.
        self._connector = connector

        self._pending_transfers: list[list[_PendingTransfer]] = [
            [] for _ in range(self._num_replicas)
        ]
        # Runs of newly committed hashes awaiting an `offload` call.
        self._pending_offloads: list[list[list[bytes]]] = [
            [] for _ in range(self._num_replicas)
        ]

    # ============================================================================
    # Request Lifecycle APIs
    # ============================================================================

    @traced
    def claim(self, ctx: TextContext, replica_idx: int = 0) -> None:
        """Pins a request to one replica, which owns it until it is released."""
        req_id = ctx.request_id
        existing = self._requests.get(req_id)
        if existing is not None:
            raise ValueError(
                f"Request is already claimed, on replica "
                f"{existing.replica_idx}: {req_id}"
            )
        self._requests[req_id] = RequestCacheState(replica_idx=replica_idx)
        for group in self._groups.values():
            group.claim(req_id)

    @property
    def groups(self) -> Mapping[str, KVGroupCoordinatorInterface]:
        """The per-leaf coordinators this manager owns, keyed by leaf id."""
        return self._groups

    def contains(self, ctx: TextContext) -> bool:
        """Returns whether the request is registered with the block manager."""
        return ctx.request_id in self._requests

    @traced
    def release(self, ctx: TextContext) -> None:
        """Frees every page the request holds, in every cache."""
        ctx.kv_cache_write_ranges.clear()
        req_id = ctx.request_id
        replica_idx = self._replica_of(ctx)

        for group in self._groups.values():
            group.release(req_id, replica_idx)

        self._touch_committed_sequence(ctx)

        # The pages are already free, so the request goes before the hand
        # back and one that raises cannot leave it claimed over them.
        state = self._requests.pop(req_id)
        self._abandon_pending_lookup(state)

    def _abandon_pending_lookup(self, state: RequestCacheState) -> None:
        """Hands back a lookup the request started and will never load from."""
        pending, state.pending_lookup = state.pending_lookup, None
        if pending is not None:
            pending.abandon()

    def _touch_committed_sequence(self, ctx: TextContext) -> None:
        """Re-ranks the request's whole committed sequence in the external tier.

        See ``BlockManager._touch_committed_sequence``: the connector orders a
        group by the keys of one call, and a sequence is committed over several,
        so each later commit outranks the earlier ones until this runs
        (CLIN-1893). Best-effort; ``touch`` never raises into the caller.
        """
        if self._connector is None or not self._enable_prefix_caching:
            return
        state = self._state_of(ctx)
        num_committed_blocks = state.committed_idx // self._block_size
        if not num_committed_blocks:
            return
        self._connector.touch(
            state.hashes[:num_committed_blocks],
            replica_idx=self._replica_of(ctx),
        )

    # ============================================================================
    # Allocation & Reuse APIs
    # ============================================================================

    @traced
    def alloc(
        self, ctx: TextContext, *, defer_lookup: bool = False
    ) -> KVTransfer:
        """Gives every cache the pages the next forward needs.

        Args:
            ctx: The request, already claimed on a replica.
            defer_lookup: Whether the caller holds the request out of the
                batch while the returned transfer is incomplete and calls
                ``alloc`` again once it completes. Only then does a
                connector lookup run off this thread, because the request
                holds no pages until that next call. A caller that reads
                the request's blocks right after this one leaves it off and
                gets the synchronous lookup, and a lookup an earlier call
                left outstanding is then handed back rather than waited on.

        Returns:
            The transfer the caller polls before the request runs, already
            complete when nothing was onloaded. With ``defer_lookup`` it may
            instead track a lookup the connector has not answered, in which
            case the request holds no pages and the next ``alloc`` finishes
            the reuse.

        Raises:
            InsufficientBlocksError: If the pool cannot serve all of the
                request's caches at once, in which case it draws nothing.
            RuntimeError: If the request was run past an onload whose copy
                failed, rather than held out of the batch until it landed.
        """
        replica_idx = self._replica_of(ctx)

        # Drain landed transfers first: their pinned pages are only returned
        # here, so skipping this leaks every offload source for the run.
        self.poll_transfers()
        self._rollback_failed_onload(ctx, replica_idx)

        committed_before = self._state_of(ctx).committed_idx
        transfer = self._reuse_blocks_from_prefix_cache(
            ctx, replica_idx, defer_lookup=defer_lookup
        )
        # The connector has not answered yet. Nothing is spliced or drawn, so
        # there is no demand to size and nothing to roll back, and the caller
        # holds the request out until the transfer completes.
        if self._state_of(ctx).pending_lookup is not None:
            return transfer

        self._metrics.input_tokens += ctx.tokens.active_length

        # Check if we have enough blocks available to satisfy the demand.
        pool = self.pools[replica_idx]
        num_required_blocks = self._num_required_blocks(ctx)
        demand: dict[str, int] = {}
        for group in self._groups.values():
            demand.update(
                group.blocks_to_allocate(ctx.request_id, num_required_blocks)
            )
        if not pool.can_satisfy_demand(demand):
            self._rollback_prefix_reuse(ctx, replica_idx, committed_before)
            raise InsufficientBlocksError(
                f"Serving {demand} needs more huge blocks than are available"
            )

        for group in self._groups.values():
            group.grow(ctx.request_id, num_required_blocks, replica_idx)

        return transfer

    @traced
    def starts_lookup(self, ctx: TextContext) -> bool:
        """Returns whether a deferred ``alloc`` would start a connector lookup.

        Reads the same state the reuse would and takes nothing, so a caller
        can decide whether it can afford the lookup before one is started.
        A request whose outstanding lookup is consumed rather than started
        reads ``False``, and so does one whose connector answers in place,
        because that lookup never holds the request out. An answer too old
        to load, or one the device hit it was asked past shrank under, is
        asked again rather than consumed, so it reads ``True``, until the
        request's re-asks after a shrink run out and the rest is computed.
        """
        state = self._state_of(ctx)
        if (
            self._connector is None
            or not self._enable_prefix_caching
            or ctx.tokens.processed_length != 0
            or state.skips_connector
            or not self._connector.defers_lookup(self._replica_of(ctx))
        ):
            return False

        self._compute_hashes_for_request(ctx)
        desired_hashes = state.hashes[state.committed_idx // self._block_size :]
        num_device_blocks = self._count_device_prefix_blocks(
            desired_hashes, self._replica_of(ctx)
        )
        pending = state.pending_lookup
        if pending is None or state.lookup_is_stale():
            return num_device_blocks < len(desired_hashes)
        return (
            pending.is_answered()
            and not pending.missed
            and num_device_blocks < pending.first_block
            and state.connector_lookup_reasks < _MAX_CONNECTOR_LOOKUP_REASKS
        )

    def start_lookup(self, ctx: TextContext) -> KVTransfer:
        """Starts the lookup a deferred ``alloc`` would start, taking no pages.

        For a request that has to wait its turn anyway, so the connector
        answers while it waits. An answer that comes back at once is kept
        for the ``alloc`` that admits the request rather than consumed here.

        Returns:
            The transfer to poll, which reads complete once the lookup can
            be consumed, or a complete one when there was nothing to ask.
        """
        state = self._state_of(ctx)
        self._compute_hashes_for_request(ctx)
        self._connector_lookup_is_ready(
            ctx,
            state.hashes[state.committed_idx // self._block_size :],
            self._replica_of(ctx),
            start=True,
        )
        if state.pending_lookup is None:
            return CompletedTransfer()
        return _PendingLookupTransfer(state.pending_lookup)

    def awaits_lookup(self, ctx: TextContext) -> bool:
        """Returns whether the request's last ``alloc`` left a lookup out.

        Such a request holds no pages, and the next ``alloc`` after the
        lookup answers takes them. A lookup that answered inside the
        ``alloc`` was consumed there and reads ``False``.
        """
        return self._state_of(ctx).pending_lookup is not None

    def hand_back_lookup(self, ctx: TextContext) -> bool:
        """Hands back the request's lookup without loading from it.

        Only once its answer is ``_LOOKUP_HAND_BACK_AGE_S`` old, because a
        fresher one can still be loaded once the request reaches the head of
        its queue, and asking again costs another leased read. The request
        then reads as starting a fresh lookup, and the hand back is counted
        against ``_MAX_CONNECTOR_LOOKUP_HAND_BACKS``, so a request passed over
        again and again keeps its last answer and loads it, or is served as a
        miss, rather than asking for good. A no-op when it has none out, and
        when its lookup already gave up, so a store that did not answer is
        not asked again for that request.

        Returns ``False`` when it kept a fresh answer, which a later call may
        still hand back, and ``True`` when there is nothing left to hand
        back.
        """
        state = self._state_of(ctx)
        pending = state.pending_lookup
        if (
            pending is None
            or pending.missed
            or state.connector_lookup_hand_backs
            >= _MAX_CONNECTOR_LOOKUP_HAND_BACKS
        ):
            return True
        if pending.age_s() < _LOOKUP_HAND_BACK_AGE_S:
            return False

        state.connector_lookup_hand_backs += 1
        self._abandon_pending_lookup(state)
        return True

    def alloc_dummy(self, ctx: TextContext, replica_idx: int = 0) -> None:
        """Claims a dummy request and points it at the replica's null page."""
        self.claim(ctx, replica_idx)
        seq_len = _compute_seq_len(
            ctx,
            num_draft_tokens=self._num_draft_tokens,
            num_draft_tokens_per_step=self._num_draft_tokens_per_step,
        )
        num_required_blocks = ceildiv(seq_len, self._block_size)
        for group in self._groups.values():
            group.grow_with_padding(
                ctx.request_id, num_required_blocks, replica_idx
            )

    @traced
    def step(self, ctx: TextContext) -> None:
        """Records what the forward just wrote and slides every window."""
        replica_idx = self._replica_of(ctx)
        write_ranges = ctx.kv_cache_write_ranges
        ctx.kv_cache_write_ranges = {}
        for leaf_id, (start, end) in write_ranges.items():
            group = self._groups[leaf_id]
            if not isinstance(group, SlidingWindowKVGroupCoordinator):
                raise ValueError("Partial writes require a sliding-window leaf")
            if not 0 <= start <= end <= ctx.tokens.processed_length:
                raise ValueError("Write range exceeds processed tokens")
        for leaf_id, (start, end) in write_ranges.items():
            group = self._groups[leaf_id]
            assert isinstance(group, SlidingWindowKVGroupCoordinator)
            group.record_write_range(ctx.request_id, start, end)
        for leaf_id, group in self._groups.items():
            if (
                isinstance(group, SlidingWindowKVGroupCoordinator)
                and leaf_id not in write_ranges
            ):
                group.record_full_write(
                    ctx.request_id, ctx.tokens.processed_length
                )
        if self._enable_prefix_caching:
            self._commit_blocks_into_prefix_cache(ctx, replica_idx)

        num_filled_blocks = self._num_filled_blocks(ctx)
        for group in self._groups.values():
            group.advance(ctx.request_id, num_filled_blocks, replica_idx)

    def get_prefix_cache_hit_counts(
        self, ctx: TextContext
    ) -> list[PrefixCacheHits]:
        """Counts the number of prefix cache hits for a request per replica.

        Read-only. With cross-replica copies on, a block held by any replica
        counts as a hit, without checking there is room to copy it in.

        Both halves run the same reconcile the reuse path runs, the device
        one over the block pools and the external one over the connector's
        masks, so this cannot disagree with what a real admit would serve
        on the leaves the masks cover. The external half is one ``probe``
        over the whole chain, answered for every replica at once since the
        connector's tiers are shared, and read from wherever each replica's
        device hit ends. A probe may answer for one representative leaf, as
        the dKV connector does against a server without presence, so only
        the leaves it covers are reconciled, and the count can then exceed
        what an admit serves when another leaf holds less. A connector with
        no probe is asked with a ``lookup`` per replica instead, which is
        advisory, and this never goes on to load. The dKV connector declines
        the probe only for a tree without a full-attention leaf. Its one
        caller, the prefix-aware DP prefill balancer, fails open.

        Returns:
            A list of PrefixCacheHits for each replica.
        """
        desired_hashes = self._compute_block_hashes(ctx, [])
        device_hits = [
            self._find_longest_device_prefix_cache_hit(
                desired_hashes, replica_idx, self._cross_replica_copy_enabled
            )
            for replica_idx in range(self._num_replicas)
        ]
        probed = None
        if self._connector is not None and any(
            num_hit_blocks < len(desired_hashes)
            for num_hit_blocks in device_hits
        ):
            # TODO(CLIN-2105): The probe still runs on the scheduler thread,
            # a few ms per presence call against a healthy dKV but up to its
            # RPC timeout against a slow one, and a request requeued under KV
            # pressure is priced again on every pass. Move it behind
            # start_presence once presence time shows up in batch creation.
            probed = self._connector.probe(
                desired_hashes,
                range(self._num_replicas),
                hint=ctx.dkv_cache_hint,
            )

        hit_counts: list[PrefixCacheHits] = []
        for replica_idx, num_hit_blocks in enumerate(device_hits):
            remaining = desired_hashes[num_hit_blocks:]
            if self._connector is None or not remaining:
                num_external_blocks = 0
            elif probed is not None:
                num_external_blocks = self._longest_connector_hit(
                    {
                        leaf_id: mask[num_hit_blocks:]
                        for leaf_id, mask in probed[replica_idx].items()
                    },
                    covered_only=True,
                )
            else:
                num_external_blocks = (
                    self._find_longest_connector_prefix_cache_hit(
                        remaining, replica_idx, ctx.dkv_cache_hint
                    )
                )
            hit_counts.append(
                PrefixCacheHits(
                    device_blocks=num_hit_blocks,
                    external_blocks=num_external_blocks,
                )
            )
        return hit_counts

    def reset_prefix_cache(self) -> None:
        """Drops every commit no request is holding, in every cache."""
        for pool in self.pools:
            pool.reset_prefix_cache()
        if self._connector is not None:
            self._connector.reset_prefix_cache()

    # ============================================================================
    # KVConnector
    # ============================================================================

    @traced
    def offload(self, replica_idx: int = 0) -> None:
        """Offloads the recently committed blocks to the connector.

        A run stops at the first hash an attention leaf has evicted. A
        recurrent leaf holds only checkpoint boundaries, so its missing hashes
        are skipped.
        """
        connector = self._connector
        if connector is None:
            return
        pool = self.pools[replica_idx]
        for hashes in self._pending_offloads[replica_idx]:
            src: dict[str, list[LittleKVCacheBlock]] = {
                leaf_id: [] for leaf_id in self._cacheable_leaf_ids
            }
            block_hashes: dict[str, list[bytes]] = {
                leaf_id: [] for leaf_id in self._cacheable_leaf_ids
            }
            for block_hash in hashes:
                held = {
                    leaf_id: pool.prefix_caches[leaf_id].get(block_hash)
                    for leaf_id in self._cacheable_leaf_ids
                }
                if any(
                    block is None
                    for leaf_id, block in held.items()
                    if not self._groups[leaf_id].group_id.is_recurrent()
                ):
                    break
                for leaf_id, block in held.items():
                    if block is not None:
                        src[leaf_id].append(block)
                        block_hashes[leaf_id].append(block_hash)
            if not any(block_hashes.values()):
                continue
            bids = {
                leaf_id: [b.bid for b in bids] for leaf_id, bids in src.items()
            }
            event = connector.offload(
                bids,
                block_hashes,
                replica_idx=replica_idx,
            )
            # Use the event to determine which pages the connector needs to read
            # and track only those in the transfer.
            if not event.is_complete():
                by_bid = {
                    leaf_id: {block.bid: block for block in blocks}
                    for leaf_id, blocks in src.items()
                }
                self._track_transfer(
                    event,
                    {
                        leaf_id: [by_bid[leaf_id][bid] for bid in taken]
                        for leaf_id, taken in event.g0_blocks_per_leaf.items()
                    },
                    replica_idx,
                )
        self._pending_offloads[replica_idx].clear()

    def poll_transfers(self) -> None:
        """Drains completed async transfers on the scheduler thread.

        For each pending async transfer, check if it has completed. If so, we
        may commit the hashes into the prefix cache and then unpin the blocks.
        A failed onload is unpinned without a commit and left for its request's
        next ``alloc`` to roll back.
        """
        if self._connector is not None:
            self._connector.poll_transfers()
        for replica_idx, pending_list in enumerate(self._pending_transfers):
            if not pending_list:
                continue
            pool = self.pools[replica_idx]
            still_pending: list[_PendingTransfer] = []
            unreached = iter(pending_list)
            for pending in unreached:
                try:
                    complete = pending.event.is_complete()
                except KVLoadFailed as failure:
                    # Settled on the way to failing, so its pages can go back,
                    # unpublished: they hold no valid KV.
                    for block in pending.pinned_blocks():
                        pool.free_block(block)
                    if pending.splice is not None:
                        self._fail_onload(pending.splice, failure)
                    continue
                except BaseException:
                    # A poll that raises has settled its own transfer, and the
                    # blocks it was filling hold no valid KV. Unpin them
                    # without committing, and put back the entries this pass
                    # never reached -- leaving them on a list it already drained
                    # would free their blocks a second time.
                    for block in pending.pinned_blocks():
                        pool.free_block(block)
                    self._pending_transfers[replica_idx] = still_pending + list(
                        unreached
                    )
                    raise
                if not complete:
                    still_pending.append(pending)
                    continue
                if pending.commit_hashes is not None:
                    self._commit_onloaded_blocks(
                        pool, pending.blocks, pending.commit_hashes
                    )
                for block in pending.pinned_blocks():
                    pool.free_block(block)
            self._pending_transfers[replica_idx] = still_pending

    def pending_transfers_exist(self, replica_idx: int = 0) -> bool:
        """Returns whether any async transfer is in flight on the replica."""
        return bool(self._pending_transfers[replica_idx])

    def _find_longest_connector_prefix_cache_hit(
        self, desired: Sequence[bytes], replica_idx: int, hint: bytes | None
    ) -> int:
        """Returns how many of ``desired`` the connector can serve at once.

        One lookup over the whole tree, reconciled by the same rules the
        device tier runs over its own block pools.
        """
        assert self._connector is not None
        resident = self._connector.lookup(
            desired, replica_idx=replica_idx, hint=hint
        )
        return self._longest_connector_hit(resident)

    def _longest_connector_hit(
        self,
        resident: Mapping[str, Sequence[bool]],
        covered_only: bool = False,
    ) -> int:
        """Returns how deep a run the connector's ``resident`` masks serve.

        ``covered_only`` reconciles only the leaves ``resident`` answers
        for, which a pricing probe may narrow to one representative leaf.
        A lookup the manager loads from has to answer for every leaf.
        """
        # Restricted to the cacheable leaves rather than passed straight
        # through: a connector answering for a leaf this manager does not
        # reconcile would otherwise decide part of the hit.
        return longest_joint_prefix_hit(
            [
                (
                    group.group_id.prefix_hit_shape(self._block_size),
                    resident[leaf_id],
                )
                for leaf_id, group in self._cacheable_groups.items()
                if not covered_only or leaf_id in resident
            ]
        )

    @traced
    def _lookup_connector_prefix_cache_hit(
        self,
        request_id: RequestID,
        desired: Sequence[bytes],
        replica_idx: int,
        hint: bytes | None,
        num_device_blocks: int,
        defer_lookup: bool,
    ) -> tuple[int, dict[str, list[LittleKVCacheBlock]], KVTransfer]:
        """Loads the prefix of ``desired`` the connector's tiers can serve.

        Two phases: ask what the connector holds and settle how deep a prefix
        that is worth, then hand it the pages and the hashes to put in them.
        Knowing the depth first means each leaf is allocated exactly what it
        reads -- a windowed leaf gets its window, not the whole run -- and
        the slots it has slid past get the null block here, the same row
        :meth:`SlidingWindowKVGroupCoordinator.claim_hit_blocks` builds for
        the device tier. Both phases get the same ``hint``.
        ``num_device_blocks`` is the device hit spliced in ahead of the load,
        which a failed copy rolls back along with it. With ``defer_lookup``
        a connector whose lookup is a round trip is never asked on this
        thread.

        A lookup an earlier deferred ``alloc`` started is consumed here
        rather than asked again, and handed back however this exits, raising
        included.

        Returns:
            How many blocks were loaded, the blocks they are landing in per
            leaf, and the transfer tracking the copy.
        """
        state = self._requests[request_id]
        pending, state.pending_lookup = state.pending_lookup, None
        try:
            return self._load_connector_prefix_cache_hit(
                request_id,
                desired,
                replica_idx,
                hint,
                num_device_blocks,
                pending,
                defer_lookup,
            )
        except BaseException:
            if pending is not None:
                pending.abandon()
            raise

    def _load_connector_prefix_cache_hit(
        self,
        request_id: RequestID,
        desired: Sequence[bytes],
        replica_idx: int,
        hint: bytes | None,
        num_device_blocks: int,
        pending: _PendingLookup | None,
        defer_lookup: bool,
    ) -> tuple[int, dict[str, list[LittleKVCacheBlock]], KVTransfer]:
        """Runs :meth:`_lookup_connector_prefix_cache_hit` over ``pending``.

        Hands ``pending`` back on every path that returns without loading
        from it, and leaves a raise to the caller to hand it back.
        """
        connector = self._connector
        empty: dict[str, list[LittleKVCacheBlock]] = {
            leaf_id: [] for leaf_id in self._cacheable_leaf_ids
        }
        miss = (0, empty, CompletedTransfer())
        state = self._requests[request_id]
        if connector is None or not desired or state.skips_connector:
            if pending is not None:
                pending.abandon()
            return miss
        # Already handed back, because polling it raised or it ran past its
        # deadline.
        if pending is not None and pending.missed:
            return miss

        # A device hit deeper than the one the lookup was asked past leaves
        # a suffix of the answer to read, and a shallower one leaves a gap no
        # answer covers. A deferred alloc asks about that gap on this thread
        # only when the connector answers in place, and otherwise serves it
        # as a miss with the hit kept.
        if pending is not None and num_device_blocks < pending.first_block:
            pending.abandon()
            pending = None

        lookup: KVLookup | None = None
        if pending is None:
            if defer_lookup and connector.defers_lookup(replica_idx):
                return miss
            resident = connector.lookup(
                desired, replica_idx=replica_idx, hint=hint
            )
        else:
            offset = num_device_blocks - pending.first_block
            lookup = pending.handle
            resident = {
                leaf_id: mask[offset:]
                for leaf_id, mask in lookup.masks().items()
            }
        aligned = self._longest_connector_hit(resident)

        pool = self.pools[replica_idx]
        # What each leaf READS of a hit that deep, which for a windowed leaf is
        # its window rather than the whole prefix.
        held = {
            leaf_id: group.blocks_held_of_connector_hit(aligned)
            for leaf_id, group in self._cacheable_groups.items()
        }
        # Too few blocks to schedule this request at all. Report no hit and let
        # the caller raise InsufficientBlocksError once it has released what
        # the request owns.
        if aligned == 0 or not pool.can_satisfy_demand(held):
            if pending is not None:
                pending.abandon()
            return miss

        rows = {
            leaf_id: [pool.alloc_block(leaf_id) for _ in range(num_blocks)]
            for leaf_id, num_blocks in held.items()
        }
        row_bids = {
            leaf_id: [block.bid for block in blocks]
            for leaf_id, blocks in rows.items()
        }
        row_hashes = {
            leaf_id: list(desired[aligned - held[leaf_id] : aligned])
            for leaf_id in rows
        }
        try:
            if lookup is None:
                event = connector.load(
                    row_bids, row_hashes, replica_idx=replica_idx, hint=hint
                )
            else:
                event = connector.load_lookup(
                    lookup,
                    row_bids,
                    row_hashes,
                    replica_idx=replica_idx,
                    hint=hint,
                )
            landed = event.is_complete()
        except (KVLoadRefused, KVLoadFailed) as err:
            # Evicted between the two calls, no room to stage the copy, or a
            # transport fault while posting or on this first poll; nothing is
            # in flight into the rows either way, and nothing is spliced yet.
            # There is no shorter hit to fall back to: a windowed leaf's
            # remaining run no longer ends where the prefix does.
            if isinstance(err, KVLoadFailed):
                self._skip_connector_after_failure(request_id, err)
            else:
                logger.warning("serving a request as a cache miss: %s", err)
            if pending is not None:
                pending.abandon()
            for blocks in rows.values():
                for block in blocks:
                    pool.free_block(block)
            return miss

        # Every leaf's row has to be `aligned` long: the groups splice them onto
        # rows that stay in lockstep.
        loaded_blocks = {
            leaf_id: [pool.null_little_blocks[leaf_id]]
            * (aligned - held[leaf_id])
            + rows[leaf_id]
            for leaf_id in rows
        }
        loaded_hashes = list(desired[:aligned])
        if landed:
            self._commit_onloaded_blocks(pool, loaded_blocks, loaded_hashes)
        else:
            start_idx = state.committed_idx
            self._track_transfer(
                event,
                loaded_blocks,
                replica_idx,
                commit_hashes=loaded_hashes,
                splice=_OnloadSplice(
                    request_id=request_id,
                    start_idx=start_idx,
                    end_idx=start_idx
                    + (num_device_blocks + aligned) * self._block_size,
                ),
            )
        return aligned, loaded_blocks, event

    def _commit_onloaded_blocks(
        self,
        pool: JengaBlockPool,
        blocks: Mapping[str, list[LittleKVCacheBlock]],
        hashes: Sequence[bytes],
    ) -> None:
        """Publishes landed onload pages, skipping any hash already in cache."""
        for leaf_id, leaf_blocks in blocks.items():
            prefix_cache = pool.prefix_caches[leaf_id]
            for block, block_hash in zip(leaf_blocks, hashes, strict=True):
                if (
                    block.block_hash is None
                    and block_hash not in prefix_cache
                    and not block.is_null
                ):
                    pool.commit_into_prefix_cache(block_hash, block)

    def _skip_connector_after_failure(
        self, request_id: RequestID, failure: KVLoadFailed
    ) -> None:
        """Counts a failed load and serves its request without the connector."""
        assert self._connector is not None
        logger.warning(
            "%s load for request %s failed; recomputing its prefix without "
            "the connector: %s",
            self._connector.name,
            request_id,
            failure,
        )
        self._metrics.connector_load_failures += 1
        # A request released while its copy was in flight has no claim left
        # to skip the connector for.
        state = self._requests.get(request_id)
        if state is not None:
            state.skips_connector = True

    def _fail_onload(
        self, splice: _OnloadSplice, failure: KVLoadFailed
    ) -> None:
        """Leaves a failed onload for its request's next alloc to roll back."""
        self._skip_connector_after_failure(splice.request_id, failure)
        state = self._requests.get(splice.request_id)
        if state is not None:
            state.failed_onload = (splice, failure)

    def _rollback_failed_onload(
        self, ctx: TextContext, replica_idx: int
    ) -> None:
        """Undoes the prefix an onload spliced in, if its copy failed.

        The request goes back to a fresh claim, device hit included, and the
        reuse that follows in :meth:`alloc` takes the device hit again without
        the connector. A reuse only splices into a request that has processed
        nothing, so a fresh claim is exactly where it started. Trimming the
        rows instead would not be: splicing past a windowed leaf's window
        frees the device hit's pages there, and a recurrent leaf keeps its
        live block, so the device hit would land behind it.

        Raises:
            RuntimeError: If the request moved past the splice before its copy
                failed, which only a caller that never held it back can let
                happen. It has read the failed pages by then, so there is no
                clean prefix to recompute from.
        """
        state = self._state_of(ctx)
        if state.failed_onload is None:
            return
        splice, failure = state.failed_onload
        state.failed_onload = None
        if (
            state.committed_idx != splice.end_idx
            or ctx.tokens.processed_length != splice.end_idx
        ):
            raise RuntimeError(
                f"request {ctx.request_id} ran past an onload whose copy "
                "failed, so its KV cannot be trusted"
            ) from failure
        for group in self._groups.values():
            group.release(ctx.request_id, replica_idx)
            group.claim(ctx.request_id)
        state.committed_idx = 0
        ctx.tokens.rewind_processing(ctx.tokens.processed_length)
        self._metrics.cache_tokens -= splice.end_idx - splice.start_idx
        ctx.cached_prefix_length = 0
        ctx.cached_prefix_external_length = 0

    def _track_transfer(
        self,
        event: KVTransfer,
        blocks: Mapping[str, list[LittleKVCacheBlock]],
        replica_idx: int,
        commit_hashes: list[bytes] | None = None,
        splice: _OnloadSplice | None = None,
    ) -> None:
        """Tracks an async connector transfer and the pages it pins.

        The pin (a ``touch``) keeps the pages out of the eviction and free
        paths while the copy engine is still reading or writing them.

        The KVCache will poll the transfer via ``poll_transfers``. When the
        transfer completes, the pages will be unpinned.
        """
        if not any(blocks.values()):
            return
        pool = self.pools[replica_idx]
        for leaf_blocks in blocks.values():
            for block in leaf_blocks:
                pool.touch(block)
        self._pending_transfers[replica_idx].append(
            _PendingTransfer(
                event=event,
                blocks={
                    leaf_id: list(leaf_blocks)
                    for leaf_id, leaf_blocks in blocks.items()
                },
                commit_hashes=commit_hashes,
                splice=splice,
            )
        )

    # ============================================================================
    # Misc
    # ============================================================================

    @property
    def metrics(self) -> KVCacheMetrics:
        """Returns the block manager's metrics."""
        metrics = self._metrics
        if self._connector is not None:
            metrics += self._connector.metrics
        return metrics

    def take_metrics(self) -> KVCacheMetrics:
        """Reads and clears this manager's and its connector's metrics."""
        metrics = self._metrics
        self._metrics = KVCacheMetrics()
        if self._connector is not None:
            metrics += self._connector.take_metrics()
        return metrics

    @property
    def effective_max_seq_length(self) -> int | None:
        """Returns the longest single request an empty pool serves."""
        pool = self.pools[0]
        return _max_seq_len_fitting_in_geometry(
            self._leaves,
            self._block_size,
            self.huge_block_count().total,
            pool.cache_ratios,
            enable_prefix_caching=self._enable_prefix_caching,
        )

    def _blocks_to_reserve(self, seq_len: int) -> dict[str, int]:
        """Returns the blocks each leaf draws for a ``seq_len``-token request."""
        num_blocks = ceildiv(seq_len, self._block_size)
        return {
            leaf_id: leaf.blocks_to_reserve(
                num_blocks, enable_prefix_caching=self._enable_prefix_caching
            )
            for leaf_id, leaf in self._leaves.items()
        }

    def _fits_in_cache(self, seq_len: int) -> bool:
        """Whether an empty pool could serve one ``seq_len``-token request."""
        return self.pools[0].can_satisfy_demand(
            self._blocks_to_reserve(seq_len), at_capacity=True
        )

    def get_req_blocks_per_leaf(self, ctx: TextContext) -> dict[str, list[int]]:
        """Returns the pages the request holds, per leaf.

        Each leaf tiles huge blocks at its own ratio, so one leaf's block ids
        say nothing about another's.
        """
        self._replica_of(ctx)
        return {
            leaf_id: [block.bid for block in blocks]
            for group in self._groups.values()
            for leaf_id, blocks in group.blocks_of(ctx.request_id).items()
        }

    def num_req_blocks(self, ctx: TextContext) -> int:
        """Returns the request's length in pages, its longest leaf row."""
        return max(
            len(blocks) for blocks in self.get_req_blocks_per_leaf(ctx).values()
        )

    def huge_block_count(self, replica_idx: int = 0) -> BlockCount:
        """Returns the huge-block occupancy for the given replica.

        ``total`` excludes the null block (huge block 0), which every cache
        shares and which is never allocable.
        """
        pool = self.pools[replica_idx]
        return BlockCount(
            free=pool.num_free_huge_blocks,
            total=pool.num_allocatable_huge_blocks,
        )

    def little_block_count(self, replica_idx: int = 0) -> dict[str, BlockCount]:
        """Returns each leaf's little-block occupancy for the given replica."""
        pool = self.pools[replica_idx]
        total_huge_blocks = pool.num_allocatable_huge_blocks
        return {
            leaf_id: BlockCount(
                free=pool.num_free_blocks(leaf_id),
                total=total_huge_blocks * pool.cache_ratios[leaf_id],
            )
            for leaf_id in self._leaf_ids
        }

    def occupancy(self, replica_idx: int = 0) -> _PoolOccupancy:
        """Returns each leaf's pages and huge blocks on the given replica.

        Scans the whole pool, so call it for periodic logging, not per batch.
        """
        pool = self.pools[replica_idx]
        total_huge_blocks = pool.num_allocatable_huge_blocks
        leaves = {}
        for leaf_id in self._leaf_ids:
            used, evictable, empty, used_huge, evictable_huge = pool.occupancy(
                leaf_id
            )
            leaves[leaf_id] = _LeafOccupancy(
                used=used,
                evictable=evictable,
                empty=empty,
                total=total_huge_blocks * pool.cache_ratios[leaf_id],
                used_huge_blocks=used_huge,
                evictable_huge_blocks=evictable_huge,
            )
        return _PoolOccupancy(
            leaves=leaves, empty_huge_blocks=pool.num_pristine_huge_blocks
        )

    # ============================================================================
    # Internal
    # ============================================================================

    def _state_of(self, ctx: TextContext) -> RequestCacheState:
        """Returns the request's tracked state, or raises if it is unclaimed."""
        state = self._requests.get(ctx.request_id)
        if state is None:
            raise ValueError(
                f"Request is not claimed, so it holds no pages to work with: "
                f"{ctx.request_id}"
            )
        return state

    def _replica_of(self, ctx: TextContext) -> int:
        """Returns the replica the request was claimed on."""
        return self._state_of(ctx).replica_idx

    def _num_required_blocks(self, ctx: TextContext) -> int:
        """Returns how far into a request's row the next forward reaches."""
        seq_len = _compute_seq_len(
            ctx,
            num_draft_tokens=self._num_draft_tokens,
            num_draft_tokens_per_step=self._num_draft_tokens_per_step,
            max_num_input_tokens=self._max_num_input_tokens,
        )
        return ceildiv(seq_len, self._block_size)

    def _num_filled_blocks(self, ctx: TextContext) -> int:
        """Returns how many of the request's blocks a forward has filled."""
        return ctx.tokens.processed_length // self._block_size

    @traced
    def _compute_block_hashes(
        self, ctx: TextContext, existing_hashes: Sequence[bytes]
    ) -> list[bytes]:
        return compute_block_hashes(
            ctx,
            existing_hashes,
            self._block_size,
            self._kv_hash_algo,
            self._kv_hash_seed,
        )

    @traced
    def _compute_hashes_for_request(self, ctx: TextContext) -> list[bytes]:
        """Extends the request's hash chain to cover its newest full blocks."""
        hashes = self._state_of(ctx).hashes
        hashes.extend(
            compute_block_hashes(
                ctx,
                hashes,
                self._block_size,
                self._kv_hash_algo,
                self._kv_hash_seed,
            )
        )
        return hashes

    def _find_longest_device_prefix_cache_hit(
        self,
        desired_hashes: Sequence[bytes],
        replica_idx: int,
        allow_cross_replica: bool,
    ) -> int:
        """Returns how many blocks every leaf can serve at once."""
        # One residency pass per leaf, read repeatedly by the settling loop
        # below. It used to re-read the prefix caches on every turn, because
        # narrowing meant re-slicing the hashes rather than moving a bound.
        return longest_joint_prefix_hit(
            [
                (
                    group.group_id.prefix_hit_shape(self._block_size),
                    group.residency(
                        desired_hashes, replica_idx, allow_cross_replica
                    ),
                )
                for group in self._cacheable_groups.values()
            ]
        )

    def _lookup_device_prefix_cache_hit(
        self,
        desired_hashes: Sequence[bytes],
        replica_idx: int = 0,
    ) -> tuple[dict[str, list[LittleKVCacheBlock]], int]:
        """Finds the longest run of ``desired_hashes`` the device cache holds.

        Returns:
            The hit pages of each leaf, and how many blocks long the run is.
            The caller splices the pages onto the request.
        """
        if self._only_use_kv_connector_last_level_cache:
            return {leaf_id: [] for leaf_id in self._groups}, 0

        num_hit_blocks = self._find_longest_device_prefix_cache_hit(
            desired_hashes, replica_idx, self._cross_replica_copy_enabled
        )

        if self._cross_replica_copy_enabled:
            self._copy_prefix_from_peers(
                desired_hashes[:num_hit_blocks], replica_idx
            )
            # The copy is best effort, so re-read what actually landed. Every
            # page that fit is local now, and one that did not shortens the
            # hit instead of breaking it.
            num_hit_blocks = self._find_longest_device_prefix_cache_hit(
                desired_hashes, replica_idx, False
            )

        hit_hashes = desired_hashes[:num_hit_blocks]
        hit_blocks: dict[str, list[LittleKVCacheBlock]] = {}
        for group in self._groups.values():
            hit_blocks.update(group.claim_hit_blocks(hit_hashes, replica_idx))
        return hit_blocks, num_hit_blocks

    def _copy_prefix_from_peers(
        self, hit_hashes: Sequence[bytes], replica_idx: int
    ) -> None:
        """Copies the pages of ``hit_hashes`` that only peer replicas hold.

        Best effort. The caller re-reads the prefix cache afterwards, so a
        page that does not fit shortens the hit rather than corrupting it.
        Every page taken is held until the end, so allocating for a later one
        cannot evict an earlier one.
        """
        pool = self.pools[replica_idx]
        held: list[LittleKVCacheBlock] = []
        copies: list[_PageCopy] = []
        staged: list[tuple[bytes, LittleKVCacheBlock]] = []
        try:
            for group in self._groups.values():
                for block_hash in group.claimable_hashes(hit_hashes):
                    src = group.find_replica_with_hash(
                        block_hash, replica_idx, allow_cross_replica=True
                    )
                    if src is None:
                        break
                    leaf_id = group.leaf_id
                    try:
                        local = pool.prefix_caches[leaf_id].get(block_hash)
                        if local is not None:
                            pool.touch(local)
                            held.append(local)
                        else:
                            dst = pool.alloc_block(leaf_id)
                            held.append(dst)
                            copies.append(
                                _PageCopy(
                                    leaf_id=leaf_id,
                                    dst_bid=dst.bid,
                                    src_bid=self.pools[src]
                                    .prefix_caches[leaf_id][block_hash]
                                    .bid,
                                    src_replica=src,
                                )
                            )
                            staged.append((block_hash, dst))
                    except InsufficientBlocksError:
                        # No room left; this group copies no further. A hash
                        # left half-committed is harmless: every page that
                        # landed holds the right bytes for its leaf, and a hit
                        # needs every leaf, so the re-read just will not count
                        # it.
                        break

            num_bytes = self._submit_page_copies(replica_idx, copies)

            # Publish only once the copies are enqueued: a committed hash is
            # visible to every request, so its page must already be filled.
            for block_hash, dst in staged:
                pool.commit_into_prefix_cache(block_hash, dst)
            self._metrics.cross_replica_blocks_copied += len(copies)
            self._metrics.cross_replica_bytes_copied += num_bytes
        finally:
            for block in held:
                pool.free_block(block)

    def _submit_page_copies(
        self, dst_replica: int, copies: Sequence[_PageCopy]
    ) -> int:
        """Batch-copies pages from other replicas into ``dst_replica``.

        Every page view is built before the call so host-side getitem work
        does not sit between the peer copies. Destinations all live on
        ``dst_replica``, so mixed sources still cost one submission.

        Returns:
            How many bytes the copy moves. Each leaf is measured at its own
            page size, which is the only comparable unit here: one block of
            the prefix is a page in every leaf of its group, and those pages
            are not the same size.
        """
        if not copies:
            return 0
        assert self._replica_kv_memory is not None

        num_bytes = 0
        dst_pages: list[Buffer] = []
        src_pages: list[Buffer] = []
        for page in copies:
            src_unit = self._replica_kv_memory[page.src_replica][page.leaf_id]
            dst_unit = self._replica_kv_memory[dst_replica][page.leaf_id]
            # Every TP shard is fanned out with its own point-to-point copy.
            for src_buf, dst_buf in zip(
                src_unit.buffers, dst_unit.buffers, strict=True
            ):
                dst_pages.append(dst_buf[page.dst_bid, :])
                src_pages.append(src_buf[page.src_bid, :])
            num_bytes += src_unit.bytes_per_page * len(src_unit.buffers)

        batch_inplace_copy(dst_pages, src_pages)
        return num_bytes

    def _reuse_blocks_from_prefix_cache(
        self, ctx: TextContext, replica_idx: int = 0, *, defer_lookup: bool
    ) -> KVTransfer:
        """Splices the longest prefix-cache hit into the request.

        The device hit is extended by whatever the connector's external tiers
        still hold; both runs are spliced on the same way, so they skip the
        same tokens and are committed at the same index.

        Returns:
            The transfer tracking the onload's copy, already complete when
            nothing was onloaded. With ``defer_lookup`` it may instead track
            a connector lookup that has not answered, in which case nothing
            was spliced and the request's ``pending_lookup`` is set.
        """
        # Only try to reuse blocks if the ctx is fresh (ie: no tokens are processed)
        if not self._enable_prefix_caching or ctx.tokens.processed_length != 0:
            return CompletedTransfer()

        self._compute_hashes_for_request(ctx)

        state = self._state_of(ctx)
        committed_blocks = state.committed_idx // self._block_size
        desired_hashes = state.hashes[committed_blocks:]

        if not self._connector_lookup_is_ready(
            ctx, desired_hashes, replica_idx, start=defer_lookup
        ):
            assert state.pending_lookup is not None
            return _PendingLookupTransfer(state.pending_lookup)

        hit_blocks, num_hit_blocks = self._lookup_device_prefix_cache_hit(
            desired_hashes, replica_idx
        )
        # Ask the connector to load the hashes that are remaining.
        num_loaded, loaded_blocks, transfer = (
            self._lookup_connector_prefix_cache_hit(
                ctx.request_id,
                desired_hashes[num_hit_blocks:],
                replica_idx,
                hint=ctx.dkv_cache_hint,
                num_device_blocks=num_hit_blocks,
                defer_lookup=defer_lookup,
            )
        )
        num_reused = num_hit_blocks + num_loaded

        # Refresh the connector's recency over everything this request reuses,
        # committed + device + connector-loaded, which is the run BlockManager
        # touches on the legacy path. Gating on the device hit alone would skip
        # the touch entirely under
        # MODULAR_ONLY_USE_KV_CONNECTOR_LAST_LEVEL_CACHE, where the device tier
        # never reports one -- so the path would go unexercised in exactly the
        # configuration that leans on the connector hardest.
        if self._connector is not None and committed_blocks + num_reused:
            self._connector.touch(
                self._state_of(ctx).hashes[: committed_blocks + num_reused],
                replica_idx=replica_idx,
            )

        self._metrics.device_blocks_served += num_hit_blocks
        self._metrics.cache_tokens += num_reused * self._block_size
        ctx.cached_prefix_length = num_reused * self._block_size
        ctx.cached_prefix_external_length = num_loaded * self._block_size

        if num_reused == 0:
            return transfer

        # The hit resumes at the committed index, so whatever the request holds
        # beyond it belongs to a chunk that is about to be re-planned.
        self._release_uncommitted_blocks(ctx, replica_idx)

        for group in self._groups.values():
            group.extend(ctx.request_id, hit_blocks, loaded_blocks, replica_idx)

        committed_idx = (
            self._state_of(ctx).committed_idx + num_reused * self._block_size
        )
        self._state_of(ctx).committed_idx = committed_idx

        skip_amount = committed_idx - ctx.tokens.processed_length
        ctx.tokens.skip_processing(skip_amount)
        assert ctx.tokens.active_length >= 1, (
            "No active tokens after prefix caching! A 100% prefix cache hit "
            "leaves nothing to compute logits from, so compute_block_hashes "
            "must never hash the last token."
        )
        return transfer

    def _connector_lookup_is_ready(
        self,
        ctx: TextContext,
        desired_hashes: Sequence[bytes],
        replica_idx: int,
        *,
        start: bool,
    ) -> bool:
        """Returns whether this request's reuse can go ahead without waiting.

        Starts a lookup when ``start`` is set and none is outstanding, and
        starts it over when the device hit it was asked past shrank while it
        was out or its answer grew too old to load. ``True`` when there is
        nothing to ask, when the outstanding lookup can be consumed, or when
        the request's reuse runs the synchronous lookup instead. A lookup
        still outstanding when ``start`` is not set is handed back so that
        the synchronous lookup runs, because that caller reads the request's
        blocks as soon as ``alloc`` returns.
        """
        state = self._state_of(ctx)
        connector = self._connector
        # An answer the request waited on in its queue past the hand back age
        # is asked again from the current device hit, with the request held
        # out again, rather than loaded under a lease that may lapse first.
        pending = state.pending_lookup
        if (
            start
            and pending is not None
            and pending.is_answered()
            and state.lookup_is_stale()
        ):
            self._abandon_pending_lookup(state)
            state.connector_lookup_hand_backs += 1
            pending = None
        if pending is None:
            if not start or connector is None or state.skips_connector:
                return True
            # Counted rather than claimed, so nothing is held while the
            # connector answers. The alloc that consumes the answer takes the
            # device hit for real and reads the answer from where it ends.
            num_device_blocks = self._count_device_prefix_blocks(
                desired_hashes, replica_idx
            )
            remaining = desired_hashes[num_device_blocks:]
            if not remaining:
                return True
            pending = self._start_connector_lookup(
                ctx, remaining, replica_idx, first_block=num_device_blocks
            )
            return pending.is_answered()

        if not pending.is_answered():
            if start:
                return False
            state.pending_lookup = None
            pending.abandon()
            return True
        if not start or pending.missed:
            return True

        # The answer starts where the device hit ended when it was asked. A
        # hit that shrank meanwhile, typically an eviction, leaves a gap no
        # mask covers, so the lookup is started over from the shorter count
        # with the request still held out. This has to be settled before the
        # hit is taken, because taking it can enqueue a cross-replica copy
        # into the pages it claims, and those must stay held until the copy
        # lands. A hit that keeps shrinking stops being chased after
        # ``_MAX_CONNECTOR_LOOKUP_REASKS`` of these and the rest is computed.
        num_device_blocks = self._count_device_prefix_blocks(
            desired_hashes, replica_idx
        )
        if num_device_blocks >= pending.first_block:
            return True
        self._abandon_pending_lookup(state)
        if state.connector_lookup_reasks >= _MAX_CONNECTOR_LOOKUP_REASKS:
            return True

        state.connector_lookup_reasks += 1
        replacement = self._start_connector_lookup(
            ctx,
            desired_hashes[num_device_blocks:],
            replica_idx,
            first_block=num_device_blocks,
        )
        return replacement.is_answered()

    def _count_device_prefix_blocks(
        self, desired_hashes: Sequence[bytes], replica_idx: int
    ) -> int:
        """Returns how many blocks the device hit would cover if taken now.

        Counts without claiming or copying anything, and counts blocks a
        peer replica holds when cross-replica copies are on, the same way
        :meth:`_lookup_device_prefix_cache_hit` takes them.
        """
        if self._only_use_kv_connector_last_level_cache:
            return 0
        return self._find_longest_device_prefix_cache_hit(
            desired_hashes, replica_idx, self._cross_replica_copy_enabled
        )

    def _start_connector_lookup(
        self,
        ctx: TextContext,
        remaining: Sequence[bytes],
        replica_idx: int,
        first_block: int,
    ) -> _PendingLookup:
        """Starts a lookup over ``remaining`` for a later ``alloc`` to read.

        ``remaining`` begins ``first_block`` blocks past the committed index,
        and the lookup becomes the request's ``pending_lookup``.
        """
        assert self._connector is not None
        # Taken before the store is asked, so the age never trails the lease.
        started_at = time.monotonic()
        pending = _PendingLookup(
            self._connector,
            self._connector.start_lookup(
                remaining, replica_idx=replica_idx, hint=ctx.dkv_cache_hint
            ),
            first_block=first_block,
            started_at=started_at,
        )
        self._state_of(ctx).pending_lookup = pending
        return pending

    def _rollback_prefix_reuse(
        self, ctx: TextContext, replica_idx: int, committed_idx: int
    ) -> None:
        """Undoes the prefix-cache splice made past ``committed_idx``.

        Keeps a failed ``alloc`` from leaving the request half-served: the
        spliced blocks go back (an in-flight onload keeps its own pin until it
        lands), the token window is rewound over them, and the cached-prefix
        attribution is cleared.
        """
        state = self._state_of(ctx)
        reused_tokens = state.committed_idx - committed_idx
        if reused_tokens == 0:
            return
        state.committed_idx = committed_idx
        self._release_uncommitted_blocks(ctx, replica_idx)
        self._metrics.cache_tokens -= reused_tokens
        ctx.cached_prefix_length = 0
        ctx.cached_prefix_external_length = 0

    def _release_uncommitted_blocks(
        self, ctx: TextContext, replica_idx: int
    ) -> None:
        """Drops the blocks past the committed index, in every cache."""
        committed_idx = self._state_of(ctx).committed_idx
        num_committed_blocks = committed_idx // self._block_size

        for group in self._groups.values():
            group.shrink_to_fit(
                ctx.request_id, num_committed_blocks, replica_idx
            )

        delta = ctx.tokens.processed_length - committed_idx
        if delta > 0:
            ctx.tokens.rewind_processing(delta)
        elif delta < 0:
            ctx.tokens.skip_processing(-delta)

    def _commit_blocks_into_prefix_cache(
        self, ctx: TextContext, replica_idx: int
    ) -> None:
        """Publishes the request's newly filled blocks into the prefix caches."""
        req_hashes = self._compute_hashes_for_request(ctx)
        first_block = self._state_of(ctx).committed_idx // self._block_size

        last_block = min(self._num_filled_blocks(ctx), len(req_hashes))

        for group in self._groups.values():
            group.commit(ctx.request_id, req_hashes, last_block, replica_idx)

        self._state_of(ctx).committed_idx = last_block * self._block_size

        # Queue the newly committed run for the next `offload`. Every leaf
        # commits the same hashes in lockstep, so one run covers them all.
        if self._connector is not None and last_block > first_block:
            self._pending_offloads[self._replica_of(ctx)].append(
                list(req_hashes[first_block:last_block])
            )
