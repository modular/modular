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
from bisect import bisect_left
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from max.driver import Buffer, batch_inplace_copy
from max.nn.kv_cache import KVCacheGroupId, KVLeafRegion
from max.nn.kv_cache.cache_params import KVCacheMemory
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.pipelines.context import TextContext
from max.pipelines.kv_cache.kv_connector import BlockCount, KVConnector
from max.pipelines.modeling.types import RequestID
from max.profiler import traced
from max.support.math import ceildiv

from ..prefix_hit import longest_joint_prefix_hit
from .block_manager import (
    CompletedTransfer,
    KVLoadRefused,
    KVTransfer,
    PrefixCacheHits,
    _compute_seq_len,
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
    """

    def fits(seq_len: int) -> bool:
        num_blocks = ceildiv(seq_len, block_size)
        demand = {
            leaf_id: leaf.blocks_to_reserve(num_blocks)
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


def _leaf_ids_by_group_id(
    leaf_infos: Mapping[str, KVLeafInfo],
) -> dict[KVCacheGroupId, list[str]]:
    """Returns each group's leaf ids, in first-appearance order.

    The order must not vary with the run's hash seed, hence the dict rather
    than a set.
    """
    group_ids = dict.fromkeys(leaf.group_id for leaf in leaf_infos.values())
    return {
        group_id: [
            leaf_id
            for leaf_id, leaf in leaf_infos.items()
            if leaf.group_id == group_id
        ]
        for group_id in group_ids
    }


def create_groups(
    leaf_infos: Mapping[str, KVLeafInfo],
    pools: Sequence[JengaBlockPool],
    page_size: int,
) -> dict[KVCacheGroupId, KVGroupCoordinatorInterface]:
    """Returns a coordinator per group the leaves fall into."""
    return {
        group_id: create_kv_group_coordinator(
            pools, group_leaves, group_id, page_size
        )
        for group_id, group_leaves in _leaf_ids_by_group_id(leaf_infos).items()
    }


def create_kv_group_coordinator(
    pools: Sequence[JengaBlockPool],
    leaf_ids: Sequence[str],
    group_id: KVCacheGroupId,
    page_size: int,
) -> KVGroupCoordinatorInterface:
    """Returns the group implementation matching the leaves' access pattern."""
    if group_id.is_sliding_window():
        return SlidingWindowKVGroupCoordinator(
            pools=pools,
            leaf_ids=leaf_ids,
            group_id=group_id,
            page_size=page_size,
            window_size=group_id.window_size,
        )
    if group_id.is_full():
        return FullKVGroupCoordinator(
            pools=pools, leaf_ids=leaf_ids, group_id=group_id
        )
    if group_id.is_recurrent():
        return RecurrentKVGroupCoordinator(
            pools=pools,
            leaf_ids=leaf_ids,
            group_id=group_id,
            page_size=page_size,
        )
    if group_id.is_scratch():
        return ScratchKVGroupCoordinator(
            pools=pools, leaf_ids=leaf_ids, group_id=group_id
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


@dataclass(frozen=True)
class KVLeafInfo:
    """How one cache tiles a huge block, and which group it belongs to.

    ``ratio`` is the number of little blocks per huge block.
    """

    ratio: int
    group_id: KVCacheGroupId


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
        groups: Mapping[KVCacheGroupId, KVGroupCoordinatorInterface],
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

        for group_id, group in groups.items():
            assert list(group.pools) == self.pools, (
                f"group {group_id} draws from other pools than this"
                " manager's; both come from the same slabs or neither does"
            )
        self._groups: dict[KVCacheGroupId, KVGroupCoordinatorInterface] = dict(
            groups
        )
        self._leaves = dict(leaves)
        self._leaf_ids = [
            leaf_id
            for group in self._groups.values()
            for leaf_id in group.leaf_ids
        ]
        # A leaf carrying no hash cannot be looked up, committed, or handed to
        # an external tier, so every content-addressed path iterates these
        # instead. Allocation and admission still cover all of them.
        self._cacheable_groups = [
            group
            for group in self._groups.values()
            if not group.group_id.is_scratch()
        ]
        self._cacheable_leaf_ids = [
            leaf_id
            for group in self._cacheable_groups
            for leaf_id in group.leaf_ids
        ]
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
    def groups(self) -> Mapping[KVCacheGroupId, KVGroupCoordinatorInterface]:
        """The cache groups this manager owns."""
        return self._groups

    def contains(self, ctx: TextContext) -> bool:
        """Returns whether the request is registered with the block manager."""
        return ctx.request_id in self._requests

    @traced
    def release(self, ctx: TextContext) -> None:
        """Frees every page the request holds, in every cache."""
        req_id = ctx.request_id
        replica_idx = self._replica_of(ctx)

        for group in self._groups.values():
            group.release(req_id, replica_idx)

        self._touch_committed_sequence(ctx)

        del self._requests[req_id]

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
    def alloc(self, ctx: TextContext) -> KVTransfer:
        """Gives every cache the pages the next forward needs.

        Raises:
            InsufficientBlocksError: If the pool cannot serve all of the
                request's caches at once, in which case it draws nothing.
        """
        replica_idx = self._replica_of(ctx)

        # Drain landed transfers first: their pinned pages are only returned
        # here, so skipping this leaks every offload source for the run.
        self.poll_transfers()

        committed_before = self._state_of(ctx).committed_idx
        transfer = self._reuse_blocks_from_prefix_cache(ctx, replica_idx)

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
        """Settles the last forward's offloads before recording this one.

        Records what the forward just wrote and slides every window.

        A synchronous connector publishes an offload's blocks in
        ``wait_for_offloads``, so the barrier has to run or those blocks stay
        unreadable and no later load can hit them. An asynchronous connector
        settles through ``poll_transfers`` instead, so this is a no-op for it.
        """
        if self._connector is not None:
            self._connector.wait_for_offloads()
        replica_idx = self._replica_of(ctx)
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

        Both halves run the same reconcile the reuse path runs -- the device
        one over the block pools, the external one over the connector's
        ``lookup`` masks -- so this cannot disagree with what a real admit
        would serve. That costs a lookup per replica for a connector whose
        tiers are not local, and nothing else: a lookup is advisory, and this
        never goes on to load. Its one caller (prefix-aware DP prefill
        balancing) fails open.

        Returns:
            A list of PrefixCacheHits for each replica.
        """
        desired_hashes = self._compute_block_hashes(ctx, [])
        hit_counts: list[PrefixCacheHits] = []
        for replica_idx in range(self._num_replicas):
            num_hit_blocks = self._find_longest_device_prefix_cache_hit(
                desired_hashes, replica_idx, self._cross_replica_copy_enabled
            )
            remaining = desired_hashes[num_hit_blocks:]
            num_external_blocks = (
                self._find_longest_connector_prefix_cache_hit(
                    remaining, replica_idx, ctx.dkv_cache_hint
                )
                if self._connector is not None and remaining
                else 0
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
        """Offloads the recently produced KV states to the connector."""
        connector = self._connector
        if connector is None:
            return
        pool = self.pools[replica_idx]
        for hashes in self._pending_offloads[replica_idx]:
            src: dict[str, list[LittleKVCacheBlock]] = {
                leaf_id: [] for leaf_id in self._cacheable_leaf_ids
            }
            block_hashes: list[bytes] = []
            for block_hash in hashes:
                if any(
                    block_hash not in pool.prefix_caches[leaf_id]
                    for leaf_id in self._cacheable_leaf_ids
                ):
                    # Evicted from at least one leaf since it was committed, so
                    # the row is no longer whole: truncate the run here.
                    break
                for leaf_id in self._cacheable_leaf_ids:
                    block = pool.prefix_caches[leaf_id][block_hash]
                    src[leaf_id].append(block)
                block_hashes.append(block_hash)
            if not block_hashes:
                continue
            bids = {
                leaf_id: [b.bid for b in bids] for leaf_id, bids in src.items()
            }
            event = connector.offload(
                bids,
                block_hashes,
                replica_idx=replica_idx,
            )
            # An asynchronous connector reads these pages on its own engine, so
            # pin them until the D2H lands. A synchronous one is already done.
            if not event.is_complete():
                self._track_transfer(event, src, replica_idx)
        self._pending_offloads[replica_idx].clear()

    def poll_transfers(self) -> None:
        """Drains completed async transfers on the scheduler thread.

        For each pending async transfer, check if it has completed. If so, we
        may commit the hashes into the prefix cache and then unpin the blocks.
        """
        for replica_idx, pending_list in enumerate(self._pending_transfers):
            if not pending_list:
                continue
            pool = self.pools[replica_idx]
            still_pending: list[_PendingTransfer] = []
            for pending in pending_list:
                if not pending.event.is_complete():
                    still_pending.append(pending)
                    continue
                if pending.commit_hashes is not None:
                    self._commit_onloaded_blocks(
                        pool, pending.blocks, pending.commit_hashes
                    )
                for leaf_blocks in pending.blocks.values():
                    for block in leaf_blocks:
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

        def rule(group: KVGroupCoordinatorInterface) -> Callable[[int], int]:
            # A group's leaves are written in lockstep, so a hash only some
            # of them hold is unusable -- what `_holds_every_leaf` does for
            # the device pools. A factory, not a lambda over the loop
            # variable, which would late-bind every rule to the last group.
            return lambda candidate: group.longest_hit(
                candidate,
                lambda idx: all(
                    resident[leaf_id][idx] for leaf_id in group.leaf_ids
                ),
            )

        return longest_joint_prefix_hit(
            len(desired), [rule(group) for group in self._cacheable_groups]
        )

    @traced
    def _lookup_connector_prefix_cache_hit(
        self,
        desired: Sequence[bytes],
        replica_idx: int,
        hint: bytes | None,
    ) -> tuple[int, dict[str, list[LittleKVCacheBlock]], KVTransfer]:
        """Loads the prefix of ``desired`` the connector's tiers can serve.

        Two phases: ask what the connector holds and settle how deep a prefix
        that is worth, then hand it the pages and the hashes to put in them.
        Knowing the depth first means each leaf is allocated exactly what it
        reads -- a windowed leaf gets its window, not the whole run -- and
        the slots it has slid past get the null block here, the same row
        :meth:`SlidingWindowKVGroupCoordinator.claim_hit_blocks` builds for
        the device tier. Both phases get the same ``hint``.

        Returns:
            How many blocks were loaded, the blocks they are landing in per
            leaf, and the transfer tracking the copy.
        """
        connector = self._connector
        empty: dict[str, list[LittleKVCacheBlock]] = {
            leaf_id: [] for leaf_id in self._cacheable_leaf_ids
        }
        miss = (0, empty, CompletedTransfer())
        if connector is None or not desired:
            return miss
        aligned = self._find_longest_connector_prefix_cache_hit(
            desired, replica_idx, hint
        )
        if aligned == 0:
            return miss

        pool = self.pools[replica_idx]
        # What each leaf READS of a hit that deep, which for a windowed leaf is
        # its window rather than the whole prefix.
        held = {
            leaf_id: group.blocks_held_of_connector_hit(aligned)
            for group in self._cacheable_groups
            for leaf_id in group.leaf_ids
        }
        # Too few blocks to schedule this request at all. Report no hit and let
        # the caller raise InsufficientBlocksError once it has released what
        # the request owns.
        if not pool.can_satisfy_demand(held):
            return miss

        rows = {
            leaf_id: [pool.alloc_block(leaf_id) for _ in range(num_blocks)]
            for leaf_id, num_blocks in held.items()
        }
        try:
            event = connector.load(
                {
                    leaf_id: [block.bid for block in blocks]
                    for leaf_id, blocks in rows.items()
                },
                {
                    leaf_id: list(desired[aligned - held[leaf_id] : aligned])
                    for leaf_id in rows
                },
                replica_idx=replica_idx,
                hint=hint,
            )
        except KVLoadRefused as refused:
            # Evicted between the two calls, or no room to stage the copy.
            # There is no shorter hit to fall back to: a windowed leaf's
            # remaining run no longer ends where the prefix does.
            logger.warning("serving a request as a cache miss: %s", refused)
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
        if event.is_complete():
            self._commit_onloaded_blocks(pool, loaded_blocks, loaded_hashes)
        else:
            self._track_transfer(
                event, loaded_blocks, replica_idx, commit_hashes=loaded_hashes
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

    def _track_transfer(
        self,
        event: KVTransfer,
        blocks: Mapping[str, list[LittleKVCacheBlock]],
        replica_idx: int,
        commit_hashes: list[bytes] | None = None,
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
        )

    def _blocks_to_reserve(self, seq_len: int) -> dict[str, int]:
        """Returns the blocks each leaf draws for a ``seq_len``-token request."""
        num_blocks = ceildiv(seq_len, self._block_size)
        return {
            leaf_id: leaf.blocks_to_reserve(num_blocks)
            for leaf_id, leaf in self._leaves.items()
        }

    def _fits_in_cache(self, seq_len: int) -> bool:
        """Whether an empty pool could serve one ``seq_len``-token request."""
        return self.pools[0].can_satisfy_demand(
            self._blocks_to_reserve(seq_len), at_capacity=True
        )

    def get_req_blocks_per_leaf(self, ctx: TextContext) -> dict[str, list[int]]:
        """Returns the pages the request holds, per leaf.

        Distinct from :meth:`PagedKVCacheManagerInterface.get_req_blocks`
        (a single flat ``list[int]``, sized for one leaf): Jenga's caches
        aren't interchangeable, so this returns one list per leaf instead.
        """
        self._replica_of(ctx)
        return {
            leaf_id: [block.bid for block in blocks]
            for group in self._groups.values()
            for leaf_id, blocks in group.blocks_of(ctx.request_id).items()
        }

    def get_req_blocks(self, ctx: TextContext) -> list[int]:
        """Returns block IDs the request holds for the first leaf.

        TODO: Delete this method after refactoring downstream callers.
        """
        return next(iter(self.get_req_blocks_per_leaf(ctx).values()))

    def huge_block_count(self, replica_idx: int = 0) -> BlockCount:
        """Returns the huge-block occupancy for the given replica.

        ``total`` excludes the null block (huge block 0), which every cache
        shares and which is never allocable.
        """
        pool = self.pools[replica_idx]
        return BlockCount(
            free=len(pool.free_huge_blocks), total=len(pool.huge_blocks)
        )

    def little_block_count(self, replica_idx: int = 0) -> dict[str, BlockCount]:
        """Returns each leaf's little-block occupancy for the given replica."""
        pool = self.pools[replica_idx]
        total_huge_blocks = len(pool.huge_blocks)
        return {
            leaf_id: BlockCount(
                free=pool.num_free_blocks(leaf_id),
                total=total_huge_blocks * pool.cache_ratios[leaf_id],
            )
            for leaf_id in self._leaf_ids
        }

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
        """Returns how many blocks every group can serve at once."""

        def rule(
            group: KVGroupCoordinatorInterface,
        ) -> Callable[[int], int]:
            # `candidate` is a length, and each group answers under it, so the
            # slice is what narrowing means here. A factory rather than a
            # lambda closing over the loop variable, which would late-bind
            # every rule to the last group.
            return lambda candidate: group.longest_cache_hit(
                desired_hashes[:candidate], replica_idx, allow_cross_replica
            )

        return longest_joint_prefix_hit(
            len(desired_hashes),
            [rule(group) for group in self._cacheable_groups],
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
            return {
                leaf_id: []
                for group in self._groups.values()
                for leaf_id in group.leaf_ids
            }, 0

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
                    try:
                        for leaf_id in group.leaf_ids:
                            local = pool.prefix_caches[leaf_id].get(block_hash)
                            if local is not None:
                                pool.touch(local)
                                held.append(local)
                                continue
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
        self, ctx: TextContext, replica_idx: int = 0
    ) -> KVTransfer:
        """Splices the longest prefix-cache hit into the request.

        The device hit is extended by whatever the connector's external tiers
        still hold; both runs are spliced on the same way, so they skip the
        same tokens and are committed at the same index.

        Returns:
            The transfer tracking the onload's copy, already complete when
            nothing was onloaded.
        """
        # Only try to reuse blocks if the ctx is fresh (ie: no tokens are processed)
        if not self._enable_prefix_caching or ctx.tokens.processed_length != 0:
            return CompletedTransfer()

        self._compute_hashes_for_request(ctx)

        committed_blocks = self._state_of(ctx).committed_idx // self._block_size
        desired_hashes = self._state_of(ctx).hashes[committed_blocks:]

        hit_blocks, num_hit_blocks = self._lookup_device_prefix_cache_hit(
            desired_hashes, replica_idx
        )
        # Ask the connector to load the hashes that are remaining.
        num_loaded, loaded_blocks, transfer = (
            self._lookup_connector_prefix_cache_hit(
                desired_hashes[num_hit_blocks:],
                replica_idx,
                hint=ctx.dkv_cache_hint,
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
