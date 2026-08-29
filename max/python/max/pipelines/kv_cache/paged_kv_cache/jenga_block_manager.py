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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from max.nn.kv_cache import KVCacheGroupId
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.pipelines.context import TextContext
from max.pipelines.kv_cache.kv_connector import BlockCount
from max.pipelines.modeling.types import RequestID
from max.profiler import traced
from max.support.math import ceildiv

from .block_manager import (
    CompletedTransfer,
    KVConnectorTransfer,
    _compute_seq_len,
    compute_block_hashes,
)
from .block_utils import InsufficientBlocksError, KVHashAlgo, LittleKVCacheBlock
from .jenga_block_pool import JengaBlockPool

logger = logging.getLogger("max.pipelines")


@dataclass(frozen=True)
class KVGroupCoordinatorInterface:
    """Finds and claims the prefix-cache hit one group of caches can serve.

    The leaves of a group are written in lockstep, so a hash is only reusable
    when every one of them holds it, and how deep the group can resume depends
    on how far back its attention reads.
    """

    pools: Sequence[JengaBlockPool]
    leaf_ids: Sequence[str]
    group_id: KVCacheGroupId

    def is_in_prefix_cache(self, block_hash: bytes, replica_idx: int) -> bool:
        """Whether every cache of the group has committed ``block_hash``."""
        return all(
            block_hash in self.pools[replica_idx].prefix_caches[leaf_id]
            for leaf_id in self.leaf_ids
        )

    def longest_cache_hit(
        self,
        desired_hashes: Sequence[bytes],
        replica_idx: int,
    ) -> int:
        """Returns how many of ``desired_hashes`` this group could resume from.

        Args:
            desired_hashes: The blocks the request wants, from its committed
                index up.
            replica_idx: Which pool to read.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def claim_hit_blocks(
        self,
        desired_hashes: Sequence[bytes],
        replica_idx: int,
    ) -> dict[str, list[LittleKVCacheBlock]]:
        """Claims the blocks for the given hashes."""
        raise NotImplementedError("Subclasses must implement this method.")

    def null_pad_blocks(
        self,
        rows: Mapping[str, list[LittleKVCacheBlock]],
        num_committed_blocks: int,
        replica_idx: int,
    ) -> None:
        """Returns the pages the group's attention just slid past.

        Args:
            rows: The request's blocks, per leaf of this group, mutated in
                place: a released slot is overwritten with the null block so
                the row stays as long as the request's block count.
            num_committed_blocks: How far the request's committed prefix
                reaches, which is what the window is measured back from.
            replica_idx: Which pool the pages return to.
        """
        raise NotImplementedError("Subclasses must implement this method.")


@dataclass(frozen=True)
class FullKVGroupCoordinator(KVGroupCoordinatorInterface):
    """A group whose caches read their whole history."""

    def longest_cache_hit(
        self,
        desired_hashes: Sequence[bytes],
        replica_idx: int,
    ) -> int:
        """Returns the run of committed hashes from the root."""
        for num_hit_blocks, block_hash in enumerate(desired_hashes):
            if not self.is_in_prefix_cache(block_hash, replica_idx):
                return num_hit_blocks
        return len(desired_hashes)

    def claim_hit_blocks(
        self,
        desired_hashes: Sequence[bytes],
        replica_idx: int,
    ) -> dict[str, list[LittleKVCacheBlock]]:
        """Adopts every block of the hit: the group reads its whole history."""
        pool = self.pools[replica_idx]
        rows: dict[str, list[LittleKVCacheBlock]] = {
            leaf_id: [] for leaf_id in self.leaf_ids
        }
        for block_hash in desired_hashes:
            for leaf_id in self.leaf_ids:
                block = pool.prefix_caches[leaf_id][block_hash]
                pool.touch(block)
                rows[leaf_id].append(block)
        return rows

    def null_pad_blocks(
        self,
        rows: Mapping[str, list[LittleKVCacheBlock]],
        num_committed_blocks: int,
        replica_idx: int,
    ) -> None:
        """Keeps every page: this group reads its whole history."""
        return


@dataclass(frozen=True)
class SlidingWindowKVGroupCoordinator(KVGroupCoordinatorInterface):
    """This group needs ``blocks_in_window`` sized run to serve a cache hit."""

    window_size: int
    page_size: int

    @property
    def _blocks_in_window(self) -> int:
        return ceildiv(self.group_id.window_size - 1, self.page_size)

    def longest_cache_hit(
        self,
        desired_hashes: Sequence[bytes],
        replica_idx: int,
    ) -> int:
        """Returns the longest windowed cache hit we can serve.

        Computing eligible Prefix Cache hits for sliding window differs greatly
        from full attn. Recall that the window size includes the query token.
        Say the query token is idx=42 and the window size is 10. This means
        that the query token will attend to tokens from idx=32 to idx=41.

        For a concrete example:

        [X]: Token is in Prefix Cache
         . : Token is not in Prefix Cache
         ^ : Eligible Prefix Cache hit

          Tokens [A]  [B]   .   [D]  [E]  [F]   .    .   [I]  [J]  [K]  [L]  [M]
        w_size=1  ^    ^    ^    ^    ^    ^    ^    ^    ^    ^    ^    ^    ^
        w_size=2  ^    ^         ^    ^    ^              ^    ^    ^    ^    ^
        w_size=3  ^    ^              ^    ^                   ^    ^    ^    ^
        w_size=4  ^    ^                   ^                        ^    ^    ^
        w_size=5  ^    ^                                                 ^    ^
        w_size=6  ^    ^                                                      ^
        w_size=7  ^    ^

        Notice that as window_size increases, the number of indices eligible for
        a cache hit decreases. Additionally, we can count consecutive runs of
        window_size-1 tokens to determine eligibility. For example, [DEF] is a
        run of 3 tokens so token F is a valid cache hit for w_size=4 and below.

        Additionally, partial window cache hits is possible if the run starts from
        the start of sequence. For example, [A] and [AB] are valid cache hits for
        any window size.

        Also window_size=1 is a degenerate case where we always get 100% cache
        hit rate since the query token does not attend to any historical tokens.
        """
        # This is a degenerate case. When window_size=1, we always get 100%
        # cache hit rate.
        if self._blocks_in_window == 0:
            return len(desired_hashes)

        run = 0
        for idx in range(len(desired_hashes) - 1, -1, -1):
            if not self.is_in_prefix_cache(desired_hashes[idx], replica_idx):
                # The run is broken. Reset the run counter.
                run = 0
                continue
            run += 1
            # If the run is at least than the window size, we have a complete window.
            if run >= self._blocks_in_window:
                return idx + run
        # No complete window. The surviving run, if any, ends at index 0.
        # We can skip the blocks_in_window check in this case.
        return run

    def claim_hit_blocks(
        self,
        desired_hashes: Sequence[bytes],
        replica_idx: int,
    ) -> dict[str, list[LittleKVCacheBlock]]:
        """Adopts the window ending at the hit and nulls every slot below it."""
        pool = self.pools[replica_idx]
        low = max(0, len(desired_hashes) - self._blocks_in_window)
        if not all(
            self.is_in_prefix_cache(block_hash, replica_idx)
            for block_hash in desired_hashes[low:]
        ):
            low = len(desired_hashes)

        rows: dict[str, list[LittleKVCacheBlock]] = {
            leaf_id: [pool.null_little_blocks[leaf_id]] * low
            for leaf_id in self.leaf_ids
        }
        for block_hash in desired_hashes[low:]:
            for leaf_id in self.leaf_ids:
                block = pool.prefix_caches[leaf_id][block_hash]
                pool.touch(block)
                rows[leaf_id].append(block)
        return rows

    def null_pad_blocks(
        self,
        rows: Mapping[str, list[LittleKVCacheBlock]],
        num_committed_blocks: int,
        replica_idx: int,
    ) -> None:
        """Frees the pages below the window, nulling their slots."""
        pool = self.pools[replica_idx]
        first_needed = max(0, num_committed_blocks - self._blocks_in_window)
        for leaf_id in self.leaf_ids:
            req_blocks = rows[leaf_id]
            null_block = pool.null_little_blocks[leaf_id]
            for idx in range(first_needed - 1, -1, -1):
                if req_blocks[idx].is_null:
                    break
                pool.free_block(req_blocks[idx])
                req_blocks[idx] = null_block


def create_kv_group_coordinator(
    pools: Sequence[JengaBlockPool],
    leaf_ids: Sequence[str],
    group_id: KVCacheGroupId,
    page_size: int,
) -> KVGroupCoordinatorInterface:
    """Returns the coordinator matching the group's attention pattern."""
    if group_id.is_sliding_window():
        return SlidingWindowKVGroupCoordinator(
            pools=pools,
            leaf_ids=leaf_ids,
            group_id=group_id,
            page_size=page_size,
            window_size=group_id.window_size,
        )
    return FullKVGroupCoordinator(
        pools=pools, leaf_ids=leaf_ids, group_id=group_id
    )


@dataclass(frozen=True)
class KVLeaf:
    """One cache's share of the pool, and the pages each request holds in it.

    Every leaf of a group is written in lockstep, so a request's row is the
    same length in all of them.
    """

    leaf_id: str
    group_id: KVCacheGroupId
    req_to_blocks: dict[RequestID, list[LittleKVCacheBlock]] = field(
        default_factory=dict
    )


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
        leaf_infos: Mapping[str, KVLeafInfo],
        num_huge_blocks: int,
        block_size: int,
        enable_prefix_caching: bool = True,
        num_replicas: int = 1,
        kv_hash_algo: KVHashAlgo = "ahash64",
        kv_hash_seed: bytes | None = None,
        max_num_input_tokens: int | None = None,
        num_draft_tokens: int = 0,
        num_draft_tokens_per_step: int = 0,
    ) -> None:
        self._block_size = block_size
        self._enable_prefix_caching = enable_prefix_caching
        self._kv_hash_algo = kv_hash_algo
        self._kv_hash_seed = kv_hash_seed
        self._max_num_input_tokens = max_num_input_tokens
        self._num_draft_tokens = num_draft_tokens
        self._num_draft_tokens_per_step = num_draft_tokens_per_step
        self._metrics = KVCacheMetrics()

        ratios = {leaf_id: leaf.ratio for leaf_id, leaf in leaf_infos.items()}
        self.pools = [
            JengaBlockPool(num_huge_blocks, ratios) for _ in range(num_replicas)
        ]

        self._leaves = {
            leaf_id: KVLeaf(
                leaf_id,
                group_id=leaf_info.group_id,
            )
            for leaf_id, leaf_info in leaf_infos.items()
        }

        # Deduplicate in first-appearance order rather than through a set:
        # groups are claimed and scanned in this order, so a set would make
        # which cache gets which page depend on the run's hash seed.
        group_ids = dict.fromkeys(
            leaf.group_id for leaf in self._leaves.values()
        )
        self._groups: dict[KVCacheGroupId, KVGroupCoordinatorInterface] = {
            group_id: create_kv_group_coordinator(
                self.pools,
                [
                    leaf.leaf_id
                    for leaf in self._leaves.values()
                    if leaf.group_id == group_id
                ],
                group_id,
                self._block_size,
            )
            for group_id in group_ids
        }

        self._req_to_hashes: dict[RequestID, list[bytes]] = {}
        self._req_to_committed_idx: dict[RequestID, int] = {}
        self._req_to_replica: dict[RequestID, int] = {}

    # ============================================================================
    # Request Lifecycle APIs
    # ============================================================================

    @traced
    def claim(self, ctx: TextContext, replica_idx: int = 0) -> None:
        """Pins a request to one replica, which owns it until it is released."""
        req_id = ctx.request_id
        existing = self._req_to_replica.get(req_id)
        if existing is not None:
            raise ValueError(
                f"Request is already claimed, on replica {existing}: {req_id}"
            )
        self._req_to_replica[req_id] = replica_idx
        self._req_to_hashes[req_id] = []
        self._req_to_committed_idx[req_id] = 0
        for leaf in self._leaves.values():
            leaf.req_to_blocks[req_id] = []

    def contains(self, ctx: TextContext) -> bool:
        """Returns whether the request is registered with the block manager."""
        return ctx.request_id in self._req_to_replica

    @traced
    def release(self, ctx: TextContext) -> None:
        """Frees every page the request holds, in every cache."""
        req_id = ctx.request_id
        pool = self.pools[self._replica_of(ctx)]

        for leaf in self._leaves.values():
            # Free in reverse so the tail blocks become eviction candidates
            # first: a later request sharing this prefix wants the head.
            for block in reversed(leaf.req_to_blocks.pop(req_id)):
                pool.free_block(block)

        del self._req_to_replica[req_id]
        del self._req_to_hashes[req_id]
        del self._req_to_committed_idx[req_id]

    # ============================================================================
    # Allocation & Reuse APIs
    # ============================================================================

    @traced
    def alloc(self, ctx: TextContext) -> KVConnectorTransfer:
        """Gives every cache the pages the next forward needs.

        Raises:
            InsufficientBlocksError: If the pool cannot serve all of the
                request's caches at once, in which case it draws nothing.
        """
        replica_idx = self._replica_of(ctx)

        # If the request is fresh, try to reuse blocks from the prefix cache.
        if ctx.tokens.processed_length == 0:
            self._reuse_blocks_from_prefix_cache(ctx, replica_idx)

        self._metrics.input_tokens += ctx.tokens.active_length

        # Check if we have enough blocks available to satisfy the demand.
        pool = self.pools[replica_idx]
        demand = {
            leaf_id: self._num_blocks_to_allocate(ctx, leaf_id)
            for leaf_id in self._leaves
        }
        self._check_admission(pool, demand)

        # Allocate the new blocks for the request.
        for leaf_id, num_new_blocks in demand.items():
            req_blocks = self._leaves[leaf_id].req_to_blocks[ctx.request_id]
            for _ in range(num_new_blocks):
                req_blocks.append(pool.alloc_block(leaf_id))

        return CompletedTransfer.load()

    @traced
    def alloc_dummy(self, ctx: TextContext, replica_idx: int = 0) -> None:
        """Claims a dummy request and points it at the replica's null page."""
        self.claim(ctx, replica_idx)
        pool = self.pools[replica_idx]
        seq_len = _compute_seq_len(
            ctx,
            num_draft_tokens=self._num_draft_tokens,
            num_draft_tokens_per_step=self._num_draft_tokens_per_step,
        )
        num_required_blocks = ceildiv(seq_len, self._block_size)
        for leaf_id, leaf in self._leaves.items():
            null_block = pool.null_little_blocks[leaf_id]
            leaf.req_to_blocks[ctx.request_id] = [
                null_block
            ] * num_required_blocks

    @traced
    def step(self, ctx: TextContext) -> None:
        """Records what the forward just wrote, and slides every window."""
        replica_idx = self._replica_of(ctx)
        pool = self.pools[replica_idx]
        if self._enable_prefix_caching:
            self._commit_blocks_into_prefix_cache(ctx, pool)

        rows = {
            leaf_id: leaf.req_to_blocks[ctx.request_id]
            for leaf_id, leaf in self._leaves.items()
        }
        num_filled_blocks = self._num_filled_blocks(ctx)
        for group in self._groups.values():
            group.null_pad_blocks(rows, num_filled_blocks, replica_idx)

    def reset_prefix_cache(self) -> None:
        """Drops every commit no request is holding, in every cache."""
        for pool in self.pools:
            pool.reset_prefix_cache()

    # ============================================================================
    # Misc
    # ============================================================================

    @property
    def metrics(self) -> KVCacheMetrics:
        """Returns the block manager's metrics."""
        return self._metrics

    def reset_metrics(self) -> None:
        """Resets the block manager's metrics to zero."""
        self._metrics = KVCacheMetrics()

    @property
    def effective_max_seq_length(self) -> int | None:
        """Returns the longest single-request sequence every leaf could serve simultaneously.

        ``None`` if there is no finite bound (every leaf is sliding-window
        and each window fits the budget). Binary search over
        :meth:`_fits_in_cache`, which is monotonic in ``seq_len``: a leaf's
        block requirement never decreases -- it grows for full attention,
        and plateaus once a sliding window is fully covered. That makes this
        equivalent to the largest ``seq_len`` for which every leaf still
        fits, without needing to simulate how the shared huge-block budget
        gets partitioned across leaves.
        """
        # Nothing can outrun a single leaf handed the entire budget at the
        # most generous ratio, so that is a safe ceiling to search up to.
        # Still fitting AT the ceiling means there is no finite bound: every
        # leaf must be sliding-window, since only their demand plateaus.
        upper_bound = (
            self.huge_block_count().total
            * self._block_size
            * max(self.pools[0].cache_ratios.values())
        )
        search_space = upper_bound + 2
        idx = bisect_left(
            range(search_space),
            True,
            key=lambda seq_len: not self._fits_in_cache(seq_len),
        )
        return (idx - 1) if idx < search_space else None

    def _blocks_demanded(self, seq_len: int) -> dict[str, int]:
        """Returns the pages each leaf holds for a ``seq_len``-token request."""
        demand = {}
        for leaf_id, leaf in self._leaves.items():
            num_blocks = ceildiv(seq_len, self._block_size)
            if leaf.group_id.is_sliding_window():
                # A window only keeps its most recent tokens resident, so its
                # demand stops growing once the window itself is covered.
                num_blocks = min(
                    num_blocks,
                    ceildiv(leaf.group_id.window_size, self._block_size),
                )
            demand[leaf_id] = num_blocks
        return demand

    def _fits_in_cache(self, seq_len: int) -> bool:
        """Whether an empty pool could serve one ``seq_len``-token request.

        Measures capacity, not current occupancy, so it counts against every
        huge block rather than the free ones and credits nothing a live
        request already holds.
        """
        pool = self.pools[0]
        needed = self._huge_blocks_for_demand(
            pool, self._blocks_demanded(seq_len)
        )
        return needed <= self.huge_block_count().total

    def get_req_blocks_per_leaf(self, ctx: TextContext) -> dict[str, list[int]]:
        """Returns the pages the request holds, per leaf.

        Distinct from :meth:`PagedKVCacheManagerInterface.get_req_blocks`
        (a single flat ``list[int]``, sized for one leaf): Jenga's caches
        aren't interchangeable, so this returns one list per leaf instead.
        """
        self._replica_of(ctx)
        return {
            leaf_id: [block.bid for block in leaf.req_to_blocks[ctx.request_id]]
            for leaf_id, leaf in self._leaves.items()
        }

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
            for leaf_id in self._leaves
        }

    # ============================================================================
    # Internal
    # ============================================================================

    def _replica_of(self, ctx: TextContext) -> int:
        """Returns the replica the request was claimed on."""
        replica_idx = self._req_to_replica.get(ctx.request_id)
        if replica_idx is None:
            raise ValueError(
                f"Request is not claimed, so it holds no pages to work with: "
                f"{ctx.request_id}"
            )
        return replica_idx

    def _num_blocks_to_allocate(self, ctx: TextContext, leaf_id: str) -> int:
        """Returns how many pages of ``leaf_id`` the next forward still needs."""
        num_current_blocks = len(
            self._leaves[leaf_id].req_to_blocks[ctx.request_id]
        )
        seq_len = _compute_seq_len(
            ctx,
            num_draft_tokens=self._num_draft_tokens,
            num_draft_tokens_per_step=self._num_draft_tokens_per_step,
            max_num_input_tokens=self._max_num_input_tokens,
        )
        num_required_blocks = ceildiv(seq_len, self._block_size)
        return max(num_required_blocks - num_current_blocks, 0)

    def _num_filled_blocks(self, ctx: TextContext) -> int:
        """Returns how many of the request's blocks a forward has filled.

        Trailing future-token placeholders count as processed positions once a
        later forward is enqueued behind them, but nothing has written their
        KV yet, so they do not fill a block.
        """
        num_realized_tokens = len(ctx.tokens) - ctx.pending_future_count
        return (
            min(ctx.tokens.processed_length, num_realized_tokens)
            // self._block_size
        )

    @traced
    def _compute_hashes_for_request(self, ctx: TextContext) -> list[bytes]:
        """Extends the request's hash chain to cover its newest full blocks."""
        hashes = self._req_to_hashes[ctx.request_id]
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

    def _huge_blocks_for_demand(
        self,
        pool: JengaBlockPool,
        demand: Mapping[str, int],
        free_little_blocks: Mapping[str, int] | None = None,
    ) -> int:
        """Converts a per-leaf page demand into the huge blocks it must claim.

        A huge block is carved for exactly one leaf, so leaves never share
        one: each cache's demand is converted at its own ratio and the
        totals are summed. Counting each cache's free pages on their own
        would let every one of them believe it has room while together they
        overrun the pool.

        ``free_little_blocks`` credits pages already carved for a leaf and
        still free; omit it to size a pool that has carved nothing yet.
        """
        num_huge_blocks = 0
        for leaf_id, num_pages in demand.items():
            already_free = (
                free_little_blocks.get(leaf_id, 0) if free_little_blocks else 0
            )
            shortfall = num_pages - already_free
            if shortfall > 0:
                num_huge_blocks += ceildiv(
                    shortfall, pool.cache_ratios[leaf_id]
                )
        return num_huge_blocks

    def _check_admission(
        self, pool: JengaBlockPool, demand: dict[str, int]
    ) -> None:
        """Rejects a request the pool cannot serve, before it draws anything."""
        num_huge_blocks_needed = self._huge_blocks_for_demand(
            pool,
            demand,
            {
                leaf_id: len(blocks)
                for leaf_id, blocks in pool.free_little_blocks.items()
            },
        )
        num_free = len(pool.free_huge_blocks)
        if num_huge_blocks_needed > num_free:
            raise InsufficientBlocksError(
                f"Serving {demand} needs {num_huge_blocks_needed} huge blocks "
                f"but only {num_free} are free"
            )

    def _find_longest_prefix_cache_hit(
        self, ctx: TextContext, replica_idx: int = 0
    ) -> tuple[dict[str, list[LittleKVCacheBlock]], int]:
        """Finds the longest prefix cache hit for the request."""
        num_committed_blocks = (
            self._req_to_committed_idx[ctx.request_id] // self._block_size
        )
        desired_hashes = self._req_to_hashes[ctx.request_id][
            num_committed_blocks:
        ]

        # Global caches first: they read their whole history, so their run from
        # the root is the tightest bound available and it costs the cheapest
        # scan to find.
        if KVCacheGroupId.full() in self._groups:
            num_hit_blocks = self._groups[
                KVCacheGroupId.full()
            ].longest_cache_hit(desired_hashes, replica_idx)
            desired_hashes = desired_hashes[:num_hit_blocks]

        windowed = [
            group
            for group in self._groups.values()
            if group.group_id.is_sliding_window()
        ]
        while windowed and desired_hashes:
            old_num_hit_blocks = len(desired_hashes)
            for window_group in windowed:
                num_hit_blocks = window_group.longest_cache_hit(
                    desired_hashes, replica_idx
                )
                desired_hashes = desired_hashes[:num_hit_blocks]

            shrank = len(desired_hashes) < old_num_hit_blocks
            # A lone window group is its own fixed point -- re-asking it under
            # its own answer returns that answer -- so only a model with two
            # different windows can need another pass.
            if not shrank or len(windowed) == 1:
                break

        hit_blocks: dict[str, list[LittleKVCacheBlock]] = {}
        for group in self._groups.values():
            hit_blocks.update(
                group.claim_hit_blocks(
                    desired_hashes,
                    replica_idx,
                )
            )

        return hit_blocks, len(desired_hashes)

    def _reuse_blocks_from_prefix_cache(
        self, ctx: TextContext, replica_idx: int = 0
    ) -> int:
        """Splices the longest prefix-cache hit into the request.

        Returns:
            How many tokens the request may skip because the hit already holds
            their KV. Zero when nothing was reused.
        """
        if not self._enable_prefix_caching:
            return 0

        if ctx.tokens.processed_length != 0:
            raise ValueError(
                "Cannot reuse blocks from the prefix cache for a request that "
                "has already processed tokens."
            )

        self._compute_hashes_for_request(ctx)

        hit_blocks, num_hit_blocks = self._find_longest_prefix_cache_hit(
            ctx, replica_idx
        )

        # Updates cache hit metrics. This manager has no KV connector, so every
        # reused token came from the device prefix cache and none is external.
        self._metrics.device_blocks_served += num_hit_blocks
        self._metrics.cache_tokens += num_hit_blocks * self._block_size
        ctx.cached_prefix_length = num_hit_blocks * self._block_size
        ctx.cached_prefix_external_length = 0

        if num_hit_blocks == 0:
            return 0

        # The hit resumes at the committed index, so whatever the request holds
        # beyond it belongs to a chunk that is about to be re-planned.
        self._release_uncommitted_blocks(ctx, replica_idx)

        # The claim already nulls the slots a windowed group slid past, so the
        # rows splice on at the same length in every leaf.
        for leaf_id, blocks in hit_blocks.items():
            self._leaves[leaf_id].req_to_blocks[ctx.request_id].extend(blocks)

        committed_idx = (
            self._req_to_committed_idx[ctx.request_id]
            + num_hit_blocks * self._block_size
        )
        self._req_to_committed_idx[ctx.request_id] = committed_idx

        skip_amount = committed_idx - ctx.tokens.processed_length
        ctx.tokens.skip_processing(skip_amount)
        assert ctx.tokens.active_length >= 1, (
            "No active tokens after prefix caching! A 100% prefix cache hit "
            "leaves nothing to compute logits from, so compute_block_hashes "
            "must never hash the last token."
        )
        return skip_amount

    def _release_uncommitted_blocks(
        self, ctx: TextContext, replica_idx: int
    ) -> None:
        """Drops the blocks past the committed index, in every cache."""
        pool = self.pools[replica_idx]
        committed_idx = self._req_to_committed_idx[ctx.request_id]
        num_committed_blocks = committed_idx // self._block_size

        for leaf in self._leaves.values():
            req_blocks = leaf.req_to_blocks[ctx.request_id]
            assert len(req_blocks) >= num_committed_blocks
            for _ in range(len(req_blocks) - num_committed_blocks):
                pool.free_block(req_blocks.pop())

        delta = ctx.tokens.processed_length - committed_idx
        if delta > 0:
            ctx.tokens.rewind_processing(delta)
        elif delta < 0:
            ctx.tokens.skip_processing(-delta)

    def _commit_blocks_into_prefix_cache(
        self, ctx: TextContext, pool: JengaBlockPool
    ) -> None:
        """Publishes the blocks the forward filled to the prefix caches."""
        req_hashes = self._compute_hashes_for_request(ctx)
        first_block = (
            self._req_to_committed_idx[ctx.request_id] // self._block_size
        )

        last_block = min(self._num_filled_blocks(ctx), len(req_hashes))

        for leaf in self._leaves.values():
            req_blocks = leaf.req_to_blocks[ctx.request_id]
            for block_idx in range(first_block, last_block):
                block = req_blocks[block_idx]
                # A twin block already serving this hash means the bytes are
                # already published, so the request adopts it and drops its own.
                twin = pool.get_or_commit_into_prefix_cache(
                    req_hashes[block_idx], block
                )
                if twin is not None:
                    req_blocks[block_idx] = twin

        self._req_to_committed_idx[ctx.request_id] = (
            last_block * self._block_size
        )
