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

"""Block manager for PagedAttention KVCache.

Handles allocating new blocks for requests as well as prefix caching/reuse.
This is done very efficiently and largely avoids Python memory allocations.

This logic is largely borrowed from vLLM v1:
- https://docs.vllm.ai/en/latest/design/v1/prefix_caching.html
- https://github.com/vllm-project/vllm/blob/f53a0586b9c88a78167157296555b7664c398055/vllm/v1/core/kv_cache_manager.py#L1
- https://github.com/vllm-project/vllm/blob/f53a0586b9c88a78167157296555b7664c398055/vllm/v1/core/kv_cache_utils.py#L1
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

from max.driver import Buffer, batch_inplace_copy
from max.nn.kv_cache.cache_params import KVCacheMemory
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.pipelines.context import TextContext
from max.pipelines.modeling.types import RequestID
from max.profiler import traced
from max.support.math import ceildiv

from ._common import (
    PrefixCacheHits,
    _compute_seq_len,
    compute_block_hashes,
)
from .block_pool import BlockPool
from .block_utils import (
    InsufficientBlocksError,
    KVCacheBlock,
    KVHashAlgo,
)

logger = logging.getLogger("max.pipelines")


class BlockManager:
    """Manages allocation and deallocation of paged KV cache blocks.

    A single ``BlockManager`` is responsible for every data-parallel (DP)
    replica. Device (GPU) memory is physically partitioned per replica, so the
    manager owns one :class:`BlockPool` per replica (``device_block_pools``).
    A request lives on exactly one replica for its lifetime; replica-scoped
    methods take a ``replica_idx`` (defaulting to ``0`` for the common
    single-replica case).

    The device prefix cache is shared across replicas in the sense that a
    lookup for a request on replica ``B`` can hit a block physically resident
    on replica ``A``: matching pages are copied device-to-device onto ``B``
    (a single ``batch_inplace_copy``). The
    ``enable_dp_cross_replica_prefix_copy`` config flag turns these copies
    off, in which case cross-replica reuse is a miss.
    """

    @traced
    def __init__(
        self,
        total_num_blocks: int,
        block_size: int,
        enable_prefix_caching: bool,
        enable_runtime_checks: bool = False,
        *,
        num_replicas: int = 1,
        kv_hash_algo: KVHashAlgo = "ahash64",
        kv_hash_seed: bytes | None = None,
        replica_kv_memory: Sequence[Mapping[str, KVCacheMemory]] | None = None,
        enable_dp_cross_replica_prefix_copy: bool = True,
    ) -> None:
        if num_replicas < 1:
            raise ValueError("BlockManager requires at least one replica")

        self.total_num_blocks = total_num_blocks
        self.block_size = block_size
        self.num_replicas = num_replicas

        self.kv_hash_algo: KVHashAlgo = kv_hash_algo
        self.kv_hash_seed: bytes | None = kv_hash_seed

        # Whether to enable prefix caching.
        self.enable_prefix_caching = enable_prefix_caching

        # Per-replica offload-ready device memory units, keyed by leaf id,
        # used to copy committed prefix blocks device-to-device between
        # replicas. Required (non-None) when ``num_replicas > 1`` and prefix
        # caching is on.
        self._replica_kv_memory: list[Mapping[str, KVCacheMemory]] | None = (
            list(replica_kv_memory) if replica_kv_memory is not None else None
        )

        # Whether a cross-replica device prefix-cache hit may be served by a
        # device-to-device copy. Requires per-replica device memory handles;
        # the enable_dp_cross_replica_prefix_copy config flag turns it off so
        # that cross-replica reuse is a miss.
        self._cross_replica_copy_enabled = (
            self._replica_kv_memory is not None
            and enable_dp_cross_replica_prefix_copy
        )

        # One pool of device blocks per replica.
        self.device_block_pools: list[BlockPool] = [
            BlockPool(
                total_num_blocks,
                enable_runtime_checks=enable_runtime_checks,
            )
            for _ in range(self.num_replicas)
        ]

        # Mapping from request ID to the replica it is assigned to. A request
        # lives on a single replica for its whole lifetime.
        self.req_to_replica: dict[RequestID, int] = {}

        # Mapping from request ID to blocks to track the blocks allocated
        # for each request, so that we can free the blocks when the request
        # is finished.
        self.req_to_blocks: dict[RequestID, list[KVCacheBlock]] = defaultdict(
            list
        )

        # Mapping from request ID to kv block hashes.
        # This is to avoid recomputing the block hashes for each call of
        # `get_computed_blocks` or `allocate_slots`.
        self.req_to_hashes: dict[RequestID, list[bytes]] = defaultdict(list)

        # Mapping from request ID to committed index (number of tokens
        # committed into the prefix cache). This replaces reliance on
        # the context's committed_idx.
        self.req_to_committed_idx: dict[RequestID, int] = defaultdict(int)

        # Metrics for the KV cache.
        self._metrics = KVCacheMetrics()

        # Whether to enable runtime checks.
        self.enable_runtime_checks = enable_runtime_checks

    @property
    def device_block_pool(self) -> BlockPool:
        """The replica-0 device block pool (single-replica convenience)."""
        return self.device_block_pools[0]

    @traced
    def claim(self, ctx: TextContext, replica_idx: int = 0) -> None:
        """Pins a request to one replica, which owns it until it is released."""
        request_id = ctx.request_id
        existing = self.req_to_replica.get(request_id)
        if existing is not None:
            raise ValueError(
                f"Request is already claimed, on replica {existing}: "
                f"{request_id}"
            )
        self.req_to_replica[request_id] = replica_idx

    def _replica_of(self, ctx: TextContext) -> int:
        """Returns the replica the request was claimed on."""
        replica_idx = self.req_to_replica.get(ctx.request_id)
        if replica_idx is None:
            raise ValueError(
                f"Request is not claimed, so it holds no pages to work with: "
                f"{ctx.request_id}"
            )
        return replica_idx

    @traced
    def step(self, ctx: TextContext) -> None:
        """Step the block manager by committing blocks into prefix cache."""
        self.assert_runtime_invariants(ctx)

        if not self.enable_prefix_caching:
            return

        # Compute block hashes. These hashes are used by the subsequent methods.
        self.compute_hashes_for_request(ctx)

        # Now that we generated new tokens, we can possibly commit additional
        # blocks into prefix cache.
        self.commit_to_prefix_cache(ctx)

        self.assert_runtime_invariants(ctx)

    @traced
    def compute_hashes_for_request(
        self,
        ctx: TextContext,
    ) -> None:
        """Computes the block hashes for the request."""
        hashes = self.req_to_hashes[ctx.request_id]
        new_hashes = compute_block_hashes(
            ctx,
            hashes,
            self.block_size,
            self.kv_hash_algo,
            self.kv_hash_seed,
        )
        hashes.extend(new_hashes)

    @traced
    def reuse_blocks_from_prefix_cache(
        self,
        ctx: TextContext,
    ) -> int:
        """Reuses blocks from prefix cache.

        Full blocks are directly reused and appended to the request's blocks.
        Partial blocks can be reused via COW.

        Args:
            ctx: The request context.

        Returns:
            The number of tokens skipped due to prefix-cache reuse, 0 when no
            blocks were reused.
        """
        # Reject an unclaimed request here rather than deeper in: the early
        # returns below skip every other call that would resolve its replica.
        self._replica_of(ctx)
        self.assert_runtime_invariants(ctx)

        if not self.enable_prefix_caching or ctx.tokens.active_length == 1:
            return 0

        # Identify a request's first admission so we record one cache-hit
        # observation per request, not one per chunked-prefill chunk.
        is_first_admission = ctx.tokens.processed_length == 0

        req_blocks = self.req_to_blocks[ctx.request_id]

        # Compute block hashes. These hashes are used by the subsequent methods.
        self.compute_hashes_for_request(ctx)

        # Query prefix cache for full blocks.
        prefix_cache_blocks = self.get_full_blocks_from_prefix_cache(ctx)

        if len(prefix_cache_blocks) > 0:
            # Update metrics.
            self._metrics.cache_tokens += (
                len(prefix_cache_blocks) * self.block_size
            )

            # Since we got cache hits, clear out existing uncommitted blocks
            self.release_uncommitted_blocks(ctx)

            # Append them to the request's blocks.
            req_blocks.extend(prefix_cache_blocks)
            prev_committed_idx = self.req_to_committed_idx[ctx.request_id]
            new_committed_idx = (
                prev_committed_idx + len(prefix_cache_blocks) * self.block_size
            )
            self.req_to_committed_idx[ctx.request_id] = new_committed_idx

            skip_amount = new_committed_idx - ctx.tokens.processed_length
            ctx.tokens.skip_processing(skip_amount)
            assert ctx.tokens.active_length >= 1, (
                "No active tokens after prefix caching! "
                "We should never get 100% prefix cache hit rate. "
                "Something went wrong!"
            )
            if is_first_admission:
                ctx.cached_prefix_length = skip_amount
                ctx.cached_prefix_external_length = 0
            return skip_amount

        if is_first_admission:
            ctx.cached_prefix_length = 0
            ctx.cached_prefix_external_length = 0
        return 0

    @traced
    def _count_full_blocks_from_prefix_cache(
        self,
        desired_hashes: Sequence[bytes],
        replica_idx: int = 0,
    ) -> int:
        """Returns the count of device blocks with the desired hashes.

        A hash counts as a device hit if it is resident in *any* replica's
        device prefix cache, because a cross-replica hit is served by a
        device-to-device copy onto ``replica_idx`` rather than a recompute.
        When cross-replica copies are unavailable or disabled (see
        ``enable_dp_cross_replica_prefix_copy``), only ``replica_idx``'s
        own cache counts, matching what the reuse path can actually serve.
        """
        local_cache = self.device_block_pools[replica_idx].prefix_cache
        device_prefix_cache_hits = []
        desired_host_hashes = []
        for hash_value in desired_hashes:
            if self._cross_replica_copy_enabled:
                _, block = self._find_block_in_any_replica(
                    hash_value, replica_idx
                )
            else:
                block = local_cache.get(hash_value)
            if block is not None:
                # Device hashes with prefix cache hit (local or cross-replica)
                device_prefix_cache_hits.append(hash_value)
            else:
                # Record potential host hash
                desired_host_hashes.append(hash_value)

        # Ignoring host cache hits in this calculation as it may be expensive
        # to compute. Eg: due to querying an external process.
        device_prefix_cache_hit_count = len(device_prefix_cache_hits)

        return device_prefix_cache_hit_count

    def _find_block_in_any_replica(
        self, block_hash: bytes, preferred_replica: int
    ) -> tuple[int, KVCacheBlock | None]:
        """Finds a committed block for ``block_hash`` on any replica.

        The preferred (local) replica is checked first so that local hits never
        incur a cross-replica copy. Returns ``(replica_idx, block)`` for a hit,
        or ``(preferred_replica, None)`` when no replica has the block.
        """
        local_cache = self.device_block_pools[preferred_replica].prefix_cache
        block = local_cache.get(block_hash)
        if block is not None:
            return preferred_replica, block
        for replica_idx in range(self.num_replicas):
            if replica_idx == preferred_replica:
                continue
            block = self.device_block_pools[replica_idx].prefix_cache.get(
                block_hash
            )
            if block is not None:
                return replica_idx, block
        return preferred_replica, None

    @traced
    def _get_full_blocks_from_device_prefix_cache(
        self,
        desired_hashes: Sequence[bytes],
        replica_idx: int = 0,
    ) -> list[KVCacheBlock]:
        """Returns device blocks on ``replica_idx`` with the desired hashes.

        Blocks resident in ``replica_idx``'s own prefix cache are reused
        directly. Blocks committed on a *different* replica are materialized
        onto ``replica_idx`` via a device-to-device copy into a freshly
        allocated block, which is then committed into the local prefix cache so
        subsequent requests on this replica hit locally (SERVOPT-1500). When
        cross-replica copies are disabled via
        ``enable_dp_cross_replica_prefix_copy``, the chain stops at the
        first local miss so the external tier (host/disk) can serve the rest.
        """
        local_pool = self.device_block_pools[replica_idx]
        local_cache = local_pool.prefix_cache

        blocks: list[KVCacheBlock] = []
        # Cross-replica copies grouped by source replica: one batched
        # device-to-device transfer per group rather than one round trip per
        # page. Commit into the local prefix cache is deferred until after
        # enqueue so a concurrent lookup cannot hit an empty destination page.
        pending_copies: dict[int, list[tuple[bytes, KVCacheBlock, int]]] = {}
        for block_hash in desired_hashes:
            local_block = local_cache.get(block_hash)
            if local_block is not None:
                local_pool.touch(local_block)
                blocks.append(local_block)
                self._metrics.device_blocks_served += 1
                continue

            # Local miss: a cross-replica hit can only be served when
            # device-to-device copies are enabled; otherwise stop the prefix
            # chain here and let the external tier serve the rest.
            if not self._cross_replica_copy_enabled:
                break

            # Look for the block on another replica.
            src_replica, src_block = self._find_block_in_any_replica(
                block_hash, replica_idx
            )
            if src_block is None:
                break

            # A cross-replica hit needs a free local block to copy into. If
            # none is available, stop the prefix chain here (it must remain
            # contiguous).
            if local_pool.num_free_blocks == 0:
                break

            # Allocate and queue the copy; commit only after enqueue (below)
            # so others cannot hit an empty page.
            dst_block = self.allocate_device_block(replica_idx)
            pending_copies.setdefault(src_replica, []).append(
                (block_hash, dst_block, src_block.bid)
            )
            blocks.append(dst_block)
            self._metrics.cross_replica_blocks_copied += 1

        # Enqueue batched D2D copies, then publish destinations into the local
        # prefix cache so subsequent requests on this replica hit locally.
        for src_replica, pending in pending_copies.items():
            self._copy_blocks_across_replicas(
                dst_replica=replica_idx,
                src_replica=src_replica,
                block_id_pairs=[
                    (dst_block.bid, src_block_id)
                    for _, dst_block, src_block_id in pending
                ],
            )
        for pending in pending_copies.values():
            for block_hash, dst_block, _ in pending:
                local_pool.commit_into_prefix_cache(block_hash, dst_block)

        return blocks

    def _copy_blocks_across_replicas(
        self,
        dst_replica: int,
        src_replica: int,
        block_id_pairs: Sequence[tuple[int, int]],
    ) -> None:
        """Batch-copies pages from ``src_replica`` to ``dst_replica``.

        Pages across every unit go into one ``batch_inplace_copy``; the driver
        groups them by destination device and submits one batch per device.

        All page views are built before the call so getitem host work does not
        sit between the peer copies.

        ``block_id_pairs`` is a sequence of ``(dst_block_id, src_block_id)``.
        """
        assert self._replica_kv_memory is not None
        src_units = self._replica_kv_memory[src_replica]
        dst_units = self._replica_kv_memory[dst_replica]
        if not block_id_pairs:
            return

        dst_pages: list[Buffer] = []
        src_pages: list[Buffer] = []
        for leaf_id, src_unit in src_units.items():
            dst_unit = dst_units[leaf_id]
            # Every shard is fanned out with an independent point-to-point copy
            # (no broadcast collective).
            for src_buf, dst_buf in zip(
                src_unit.buffers, dst_unit.buffers, strict=True
            ):
                for dst_id, src_id in block_id_pairs:
                    dst_pages.append(dst_buf[dst_id, :])
                    src_pages.append(src_buf[src_id, :])
            self._metrics.cross_replica_bytes_copied += (
                src_unit.bytes_per_page
                * len(src_unit.buffers)
                * len(block_id_pairs)
            )

        batch_inplace_copy(dst_pages, src_pages)

    @traced
    def count_cached_prefix_blocks_per_replica(
        self, block_hashes: Sequence[bytes]
    ) -> list[PrefixCacheHits]:
        """Counts each replica's contiguous cached prefix of ``block_hashes``.

        Walks ``block_hashes`` in prefix order through each replica's device
        prefix cache, mirroring the reuse order of
        :meth:`get_full_blocks_from_prefix_cache`, and stops at the first block
        that is not resident.

        Unlike the reuse path this allocates and onboards nothing, touches no
        LRU state, and creates no per-request state, so it is safe to call for
        requests that are not (and may never be) claimed on any replica, such
        as for prefix-aware data-parallel routing. Counts reflect index
        presence only and ignore transient constraints the reuse path
        enforces (such as free device blocks to copy into).

        Args:
            block_hashes: The request's block hash chain, in prefix order.

        Returns:
            One :class:`PrefixCacheHits` per replica, indexed by replica.
        """
        if not self.enable_prefix_caching:
            return [PrefixCacheHits() for _ in range(self.num_replicas)]
        return [
            PrefixCacheHits(
                device_blocks=self._count_device_prefix_blocks(
                    block_hashes, replica_idx
                )
            )
            for replica_idx in range(self.num_replicas)
        ]

    def _count_device_prefix_blocks(
        self, block_hashes: Sequence[bytes], replica_idx: int
    ) -> int:
        """Counts the leading hashes the device reuse path would serve."""
        device_prefix_cache = self.device_block_pools[replica_idx].prefix_cache
        num_device_hits = 0
        for block_hash in block_hashes:
            if self._cross_replica_copy_enabled:
                _, blk = self._find_block_in_any_replica(
                    block_hash, replica_idx
                )
                present = blk is not None
            else:
                present = block_hash in device_prefix_cache
            if not present:
                break
            num_device_hits += 1
        return num_device_hits

    @traced
    def get_full_blocks_from_prefix_cache(
        self, ctx: TextContext
    ) -> list[KVCacheBlock]:
        """Gets the computed (cached) blocks for the request.

        Note that the computed blocks must be full.

        Returns:
            The device blocks, whether resident on the request's replica or
            copied from another one, that continue the request's committed
            prefix.
        """
        assert self.enable_prefix_caching

        replica_idx = self._replica_of(ctx)
        req_hashes = self.req_to_hashes[ctx.request_id]
        num_committed_blocks = (
            self.req_to_committed_idx[ctx.request_id] // self.block_size
        )
        uncommitted_hashes = req_hashes[num_committed_blocks:]

        return self._get_full_blocks_from_device_prefix_cache(
            uncommitted_hashes, replica_idx
        )

    @traced
    def commit_to_prefix_cache(
        self,
        ctx: TextContext,
    ) -> None:
        """Commits all blocks whose hashes are known for prefix caching.

        This increments the committed_idx.

        Args:
            ctx: TextContext.
        """
        replica_idx = self._replica_of(ctx)
        pool = self.device_block_pools[replica_idx]
        req_blocks = self.req_to_blocks[ctx.request_id]
        req_hashes = self.req_to_hashes[ctx.request_id]
        num_committed_blocks = (
            self.req_to_committed_idx[ctx.request_id] // self.block_size
        )

        # Count the number of tokens for which we know the values of and align
        # to the block size.
        num_computed_blocks = ctx.tokens.processed_length // self.block_size

        # Commit blocks into the prefix cache.
        for block_idx in range(num_committed_blocks, num_computed_blocks):
            block = req_blocks[block_idx]
            block_hash = req_hashes[block_idx]

            new_block = pool.get_or_commit_into_prefix_cache(block_hash, block)
            # If the block is already int the prefix cache, we skip the commit.
            # Then we overwrite the req blocks with the existing block that contains
            # the same contents.
            if new_block is not None:
                req_blocks[block_idx] = new_block

        # Bump the committed index.
        self.req_to_committed_idx[ctx.request_id] = (
            num_computed_blocks * self.block_size
        )

    def release(self, ctx: TextContext) -> None:
        """Release the blocks for the request.

        Raises:
            ValueError: If the request is not claimed, which includes a second
                release: the claim is what names the pool the pages return to,
                so there is no such thing as releasing without one.
        """
        request_id = ctx.request_id
        pool = self.device_block_pools[self._replica_of(ctx)]
        blocks = self.req_to_blocks[request_id]
        ordered_blocks: Iterable[KVCacheBlock] = blocks
        if self.enable_prefix_caching:
            # Free blocks in reverse order so that the tail blocks are
            # freed first.
            ordered_blocks = reversed(blocks)

        for block in ordered_blocks:
            pool.free_block(block)

        self.req_to_blocks.pop(request_id, None)
        self.req_to_hashes.pop(request_id, None)
        self.req_to_replica.pop(request_id, None)

        # Committed idx is only used with the prefix cache
        # therefore this may not always be in the dict.
        if request_id in self.req_to_committed_idx:
            del self.req_to_committed_idx[request_id]

    @traced
    def allocate_new_blocks(
        self,
        ctx: TextContext,
        num_draft_tokens: int = 0,
        num_draft_tokens_per_step: int = 0,
    ) -> None:
        """Allocate new blocks for a request to accommodate additional tokens.

        Calculates the number of additional blocks needed based on the current sequence
        length, then allocates them from the device block pool.
        Validates that there are sufficient free blocks available and that the current
        blocks can accommodate the completed tokens.

        Args:
            ctx: The request context containing sequence information and token indices.
            num_draft_tokens: Total draft tokens generated per speculative
                iteration. Zero for non-speculative decode.
            num_draft_tokens_per_step: Number of draft KV positions written
                per draft forward. Zero when speculative decoding is
                disabled; one for autoregressive drafts (``eagle``, ``mtp``);
                equal to ``num_draft_tokens`` for block drafts (``dflash``, ``dspark``).
                Used by ``_compute_seq_len`` to size the cache for block
                drafts, whose ``forward_block`` writes one extra position
                past the bonus token.

        Raises:
            InsufficientBlocksError: If there are insufficient free blocks to
            satisfy the allocation.
        """
        replica_idx = self._replica_of(ctx)
        pool = self.device_block_pools[replica_idx]

        # It is impossible to schedule this request, even if it was the only req
        # and could use the entire KV cache.
        # This should literally never happen unless the user sets an absurdly
        # large max seq len or the KV cache is very small.
        total_kv_slots = self.total_num_blocks * self.block_size
        seq_len = (
            len(ctx.tokens)
            + len(ctx.spec_decoding_state.draft_tokens_to_verify)
            + len(ctx.spec_decoding_state.maybe_accepted_draft_tokens)
        )
        if seq_len > total_kv_slots:
            raise InsufficientBlocksError(
                f"Insufficient KV pages for a single request with {seq_len} tokens.\n"
                f"The KVCache has {self.total_num_blocks} pages with page size {self.block_size}. This is only enough to support {total_kv_slots} tokens.\n"
                "You must restart your process and set a lower max seq len to prevent a single request from using the entire KV cache."
            )

        # Update metrics.
        self._metrics.input_tokens += ctx.tokens.active_length

        # Determine number of new blocks to allocate.
        num_new_blocks = self.num_blocks_to_allocate(
            ctx,
            num_draft_tokens,
            num_draft_tokens_per_step,
        )

        # Verify that processed tokens fit within the currently allocated blocks.
        current_blocks = self.req_to_blocks[ctx.request_id]
        num_current_blocks = len(current_blocks)
        processed_length = ctx.tokens.processed_length
        assert ctx.tokens.processed_length <= (
            num_current_blocks * self.block_size
        ), (
            f"Expected at least {ceildiv(processed_length, self.block_size)} "
            f"blocks to store KV for {processed_length} committed tokens, but "
            f"only {num_current_blocks} are assigned."
        )

        # Check that we have enough free blocks to allocate the new blocks.
        if num_new_blocks > pool.num_free_blocks:
            free = pool.num_free_blocks
            in_use = self.total_num_blocks - free
            raise InsufficientBlocksError(
                f"Cannot get {num_new_blocks} free blocks from the free block queue"
                f" (only {free} available; {in_use}/{self.total_num_blocks} blocks"
                f" currently in use)"
            )

        # Allocate new blocks.
        for _ in range(num_new_blocks):
            new_block = self.allocate_device_block(replica_idx)
            current_blocks.append(new_block)

    @traced
    def num_blocks_to_allocate(
        self,
        ctx: TextContext,
        num_draft_tokens: int = 0,
        num_draft_tokens_per_step: int = 0,
    ) -> int:
        """Calculates the number of new blocks to allocate for a request.

        Args:
            ctx: The request context containing sequence information and token indices.
            num_draft_tokens: Total draft tokens generated per speculative
                iteration. Zero for non-speculative decode.
            num_draft_tokens_per_step: Number of draft KV positions written
                per draft forward. Zero when speculative decoding is
                disabled; one for autoregressive drafts (``eagle``, ``mtp``);
                equal to ``num_draft_tokens`` for block drafts (``dflash``, ``dspark``).
                Used by ``_compute_seq_len`` to size the cache for block
                drafts, whose ``forward_block`` writes one extra position
                past the bonus token.

        Returns:
            The number of new blocks to allocate.
        """
        current_blocks = self.req_to_blocks[ctx.request_id]
        num_current_blocks = len(current_blocks)
        current_seq_len = _compute_seq_len(
            ctx,
            num_draft_tokens,
            num_draft_tokens_per_step,
        )
        num_required_blocks = ceildiv(current_seq_len, self.block_size)
        num_new_blocks = num_required_blocks - num_current_blocks

        return max(num_new_blocks, 0)

    @traced
    def allocate_device_block(self, replica_idx: int = 0) -> KVCacheBlock:
        """Allocates a single block from the replica's device block pool."""
        new_block, _ = self.device_block_pools[replica_idx].alloc_block()
        return new_block

    def release_uncommitted_blocks(
        self,
        ctx: TextContext,
    ) -> None:
        """Release the uncommitted blocks for the request."""
        pool = self.device_block_pools[self._replica_of(ctx)]
        req_blocks = self.req_to_blocks[ctx.request_id]
        num_committed_blocks = (
            self.req_to_committed_idx[ctx.request_id] // self.block_size
        )
        assert len(req_blocks) >= num_committed_blocks
        num_uncommitted_blocks = len(req_blocks) - num_committed_blocks
        for _ in range(num_uncommitted_blocks):
            block = req_blocks.pop()
            pool.free_block(block)
        delta = (
            ctx.tokens.processed_length
            - self.req_to_committed_idx[ctx.request_id]
        )
        if delta > 0:
            ctx.tokens.rewind_processing(delta)
        elif delta < 0:
            ctx.tokens.skip_processing(-delta)

    def rollback_prefix_reuse(self, ctx: TextContext, num_tokens: int) -> None:
        """Undoes a ``reuse_blocks_from_prefix_cache`` splice of ``num_tokens``.

        For an allocation that fails after the splice, so the request is left
        as consistent as it was found: the spliced blocks go back to the pool
        (an in-flight onload keeps its own pin until it lands), the token
        window is rewound over them, and a first admission's cached-prefix
        attribution is cleared.
        """
        if num_tokens == 0:
            return
        self.req_to_committed_idx[ctx.request_id] -= num_tokens
        self.release_uncommitted_blocks(ctx)
        self._metrics.cache_tokens -= num_tokens
        if ctx.tokens.processed_length == 0:
            ctx.cached_prefix_length = 0
            ctx.cached_prefix_external_length = 0

    def register_dummy_request(self, ctx: TextContext) -> None:
        """Maps a dummy request to the replica pool's reserved null block."""
        request_id = ctx.request_id
        assert self.req_to_blocks[request_id] == []
        self.req_to_blocks[request_id] = [
            self.device_block_pools[self._replica_of(ctx)].null_block
        ]

    @traced
    def get_req_blocks(self, ctx: TextContext) -> list[int]:
        """Get the block ids for a request."""
        self._replica_of(ctx)
        return [block.bid for block in self.req_to_blocks[ctx.request_id]]

    @traced
    def reset_prefix_cache(self) -> None:
        """Resets the device prefix caches for all replicas."""
        for pool in self.device_block_pools:
            pool.reset_prefix_cache()

    @property
    def metrics(self) -> KVCacheMetrics:
        """Returns this manager's metrics."""
        return self._metrics

    def take_metrics(self) -> KVCacheMetrics:
        """Reads and clears this manager's metrics."""
        own = self._metrics
        self._metrics = KVCacheMetrics()
        return own

    @traced
    def assert_runtime_invariants(self, ctx: TextContext) -> None:
        """Asserts runtime invariants when runtime checks are enabled."""
        if not self.enable_runtime_checks:
            return

        # Get the active block ids, partitioned by the replica that owns each
        # request's blocks.
        active_block_ids_by_replica: list[list[int]] = [
            [] for _ in range(self.num_replicas)
        ]
        for request_id, blocks in self.req_to_blocks.items():
            req_replica = self.req_to_replica.get(request_id, 0)
            for block in blocks:
                active_block_ids_by_replica[req_replica].append(block.bid)
                # Check that all active blocks have a ref_cnt > 0
                assert block.ref_cnt > 0

        # Check that each block pool is consistent
        for pool, active_block_ids in zip(
            self.device_block_pools,
            active_block_ids_by_replica,
            strict=True,
        ):
            pool.assert_runtime_invariants(active_block_ids)

        # Get the request hashes and blocks
        req_hashes = self.req_to_hashes[ctx.request_id]
        req_blocks = self.req_to_blocks[ctx.request_id]

        # Check that the number of committed blocks for request is correct
        num_committed_blocks = (
            self.req_to_committed_idx[ctx.request_id] // self.block_size
        )
        num_committed = 0
        for block in req_blocks:
            if block.block_hash is None:
                break
            num_committed += 1
        assert num_committed == num_committed_blocks

        # Check that the req block hashes are consistent with req blocks
        for hash_value, block in zip(req_hashes, req_blocks, strict=False):
            assert block.block_hash is None or block.block_hash == hash_value
