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
import os
from collections import defaultdict
from collections.abc import Iterable, Sequence

from max.nn.kv_cache.cache_params import KVCacheMemory
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.pipelines.context import (
    TextAndVisionContext,
    TextContext,
    TokenHashOverride,
)
from max.pipelines.kv_cache.kv_connector import KVConnector, to_block_hash_bytes
from max.pipelines.kv_cache.memory_tier import MemoryTier
from max.pipelines.modeling.types import RequestID
from max.profiler import traced
from max.support.math import ceildiv

from .block_pool import BlockPool
from .block_utils import (
    InsufficientBlocksError,
    KVCacheBlock,
    KVHashAlgo,
    hash_request_tokens,
)

logger = logging.getLogger("max.pipelines")


def _compute_seq_len(
    ctx: TextContext,
    num_draft_tokens: int,
    num_draft_tokens_per_step: int = 1,
) -> int:
    # Each term accounts for one category of tokens that need a KV slot:
    #
    #   ctx.tokens                    : prompt + tokens generated so far
    #   maybe_accepted_draft_tokens   : draft tokens being verified in the
    #                                   *previous* batch (overlap scheduler);
    #                                   conservative: assume all are accepted
    #   2 * num_draft_tokens          : drafts to verify *next* batch
    #                                   + drafts written *during* that batch
    #   1                             : one regular decode step
    #   -1                            : the last generated token has no KV entry
    #
    # Block-draft correction (DFlash): the draft model's ``forward_block``
    # writes ``num_draft_tokens_per_step + 1`` positions in a single batched
    # call, starting at ``bumped_cache_length = pre_cache_length +
    # commit_lengths``. Compared to the autoregressive-draft accounting above
    # (which assumes one draft KV per step), that's an extra position past the
    # bonus token — exactly the slot the ``- 1`` here was reclaiming under the
    # "last generated token has no KV entry" optimization. For block drafts
    # that bonus position *does* get a KV entry (forward_block writes it as
    # part of the speculative tail), so we add it back.
    block_draft_extra = (
        1 if num_draft_tokens_per_step == num_draft_tokens else 0
    )
    seq_len = (
        len(ctx.tokens)
        + len(ctx.spec_decoding_state.maybe_accepted_draft_tokens)
        + 2 * num_draft_tokens
        + 1
        + block_draft_extra
        - 1
    )
    return seq_len


def _resolve_only_use_kv_connector_last_level_cache() -> bool:
    """Resolve whether to only use the KVConnector last level cache.

    When this is set, the device prefix cache will be disabled. All KVCache hits
    will strictly be served from the KVConnector. This is primarily used for
    testing and benchmarking the performance of the KVConnector. Do NOT use this
    flag in production.

    With the local connector, the last level cache is the host memory. With the
    tiered connector, the last level cache is the disk.
    """
    enabled = os.getenv(
        "MODULAR_ONLY_USE_KV_CONNECTOR_LAST_LEVEL_CACHE", "0"
    ).lower() in (
        "1",
        "true",
        "yes",
        "y",
    )
    if enabled:
        logger.info(
            "Detected MODULAR_ONLY_USE_KV_CONNECTOR_LAST_LEVEL_CACHE flag, only using KVConnector prefix cache."
        )
    return enabled


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
    on replica ``A``: the block's pages are copied device-to-device onto ``B``
    via :meth:`KVCacheMemory.copy_block_to` (SERVOPT-1500). External tiers
    (host/disk) are reached through a single ``KVConnector`` shared by every
    replica; each ``load``/``offload`` passes the ``replica_idx`` so the
    connector can select that replica's device buffers (SERVOPT-1501).
    """

    @traced
    def __init__(
        self,
        device_memory_tier: MemoryTier,
        total_num_blocks: int,
        block_size: int,
        connector: KVConnector,
        enable_prefix_caching: bool,
        enable_runtime_checks: bool = False,
        *,
        num_replicas: int = 1,
        kv_hash_algo: KVHashAlgo = "ahash64",
        kv_hash_seed: bytes | None = None,
        replica_kv_memory: Sequence[Sequence[KVCacheMemory]] | None = None,
    ) -> None:
        if num_replicas < 1:
            raise ValueError("BlockManager requires at least one replica")

        self.total_num_blocks = total_num_blocks
        self.block_size = block_size
        self.num_replicas = num_replicas

        self.kv_hash_algo: KVHashAlgo = kv_hash_algo
        self.kv_hash_seed: bytes | None = kv_hash_seed
        self._salt_dropped_warned: bool = False

        if kv_hash_algo not in connector.supported_hash_algos:
            raise ValueError(
                f"kv_cache_hash_algo={kv_hash_algo!r} is not supported by "
                f"connector={connector.name!r} (supports="
                f"{sorted(connector.supported_hash_algos)})"
            )

        # Whether to enable prefix caching.
        self.enable_prefix_caching = enable_prefix_caching

        # A single connector for external cache tiers (host memory, disk, dKV),
        # shared across every replica. It owns host memory / a host block pool
        # and the H2D/D2H transfers; ``load``/``offload`` take a ``replica_idx``
        # to select the device endpoint.
        self.connector = connector

        # Per-replica offload-ready device memory units, used to copy committed
        # prefix blocks device-to-device between replicas. Required (non-None)
        # when ``num_replicas > 1`` and prefix caching is on.
        self._replica_kv_memory: list[list[KVCacheMemory]] | None = (
            [list(units) for units in replica_kv_memory]
            if replica_kv_memory is not None
            else None
        )

        # Ordered offload sequences pending delivery to each replica's
        # connector. Each entry is (parent_seq_hash, ordered block hashes): one
        # contiguous run of newly-committed blocks, in prefix order, chaining
        # onto parent_seq_hash (None = root). Ordering and parentage are
        # preserved so connectors that chain sequences (dKV) can reconstruct the
        # prefix; hash-keyed connectors (host/disk) ignore the parent.
        self._pending_offloads: list[
            list[tuple[int | bytes | None, list[int] | list[bytes]]]
        ] = [[] for _ in range(self.num_replicas)]

        # One pool of device blocks per replica.
        self.device_block_pools: list[BlockPool] = [
            BlockPool(
                device_memory_tier,
                total_num_blocks,
                enable_prefix_caching,
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
        self.req_to_hashes: dict[RequestID, list[int] | list[bytes]] = (
            defaultdict(list)
        )

        # Mapping from request ID to committed index (number of tokens
        # committed into the prefix cache). This replaces reliance on
        # the context's committed_idx.
        self.req_to_committed_idx: dict[RequestID, int] = defaultdict(int)

        # Metrics for the KV cache.
        self._metrics = KVCacheMetrics()

        # Whether to enable runtime checks.
        self.enable_runtime_checks = enable_runtime_checks

        # Whether to only use the KVConnector last level cache.
        # When this is set, the device prefix cache will be disabled. This is
        # primarily used for testing and benchmarking the performance of the
        # KVConnector.
        self._only_use_kv_connector_last_level_cache = (
            _resolve_only_use_kv_connector_last_level_cache()
        )

    @property
    def device_block_pool(self) -> BlockPool:
        """The replica-0 device block pool (single-replica convenience)."""
        return self.device_block_pools[0]

    def _register_replica(
        self, request_id: RequestID, replica_idx: int
    ) -> None:
        """Records the replica a request is assigned to (idempotent)."""
        existing = self.req_to_replica.get(request_id)
        if existing is not None and existing != replica_idx:
            raise ValueError(
                f"Request {request_id} is already assigned to replica "
                f"{existing}, cannot reassign to {replica_idx}"
            )
        self.req_to_replica[request_id] = replica_idx

    @traced
    def step(self, ctx: TextContext, replica_idx: int = 0) -> None:
        """Step the block manager by committing blocks into prefix cache."""
        self.assert_runtime_invariants(ctx, replica_idx)

        if not self.enable_prefix_caching:
            return

        # Compute block hashes. These hashes are used by the subsequent methods.
        self.compute_hashes_for_request(ctx)

        # Now that we generated new tokens, we can possibly commit additional
        # blocks into prefix cache.
        self.commit_to_prefix_cache(ctx, replica_idx)

        self.assert_runtime_invariants(ctx, replica_idx)

    @traced
    def compute_hashes_for_request(
        self,
        ctx: TextContext,
    ) -> None:
        """Computes the block hashes for the request."""
        hashes = self.req_to_hashes[ctx.request_id]

        num_hashed_tokens = len(hashes) * self.block_size
        # We do not compute the hash for the last token because it is ineligible
        # for prefix caching. This is because 100% prefix cache hit is illegal
        # and will result in a 0 input tokens for the request. Hence the minus 1.
        num_hashable_tokens = len(ctx.tokens) - 1
        num_unhashed_tokens = num_hashable_tokens - num_hashed_tokens
        if num_unhashed_tokens < self.block_size:
            return

        parent_hash_value: int | bytes | None = None
        if len(hashes) > 0:
            parent_hash_value = hashes[-1]

        unhashed_tokens = ctx.tokens[num_hashed_tokens:num_hashable_tokens]

        token_hash_overrides: list[TokenHashOverride] = []
        if isinstance(ctx, TextAndVisionContext):
            for img in ctx.images:
                if img.image_hash is None:
                    raise ValueError(
                        "hash_request_tokens requires `image_hash` to be present. Found None."
                    )
                token_hash_overrides.append(
                    TokenHashOverride(
                        token_idx=img.start_idx,
                        token_hash=img.image_hash,
                        source="image",
                    )
                )
            token_hash_overrides.extend(ctx.token_hash_overrides)

        cache_salt = ctx.cache_salt
        if cache_salt is not None and self.kv_hash_algo == "ahash64":
            if not self._salt_dropped_warned:
                logger.warning(
                    "cache_salt was supplied on a request but "
                    "kv_cache_hash_algo=ahash64; salt is being dropped. Set "
                    "kv_cache_hash_algo=sha256 to enable per-request "
                    "prefix-cache isolation."
                )
                self._salt_dropped_warned = True
            cache_salt = None

        new_hashes = hash_request_tokens(
            token_ids=unhashed_tokens,
            block_size=self.block_size,
            parent_hash=parent_hash_value,
            prefix_length=num_hashed_tokens,
            token_hash_overrides=token_hash_overrides,
            algo=self.kv_hash_algo,
            seed=self.kv_hash_seed,
            salt=cache_salt,
        )
        hashes.extend(new_hashes)  # type: ignore[arg-type]

    @traced
    def reuse_blocks_from_prefix_cache(
        self,
        ctx: TextContext,
        replica_idx: int = 0,
        skip_tokens: bool = True,
    ) -> int:
        """Reuses blocks from prefix cache.

        Full blocks are directly reused and appended to the request's blocks.
        Partial blocks can be reused via COW.

        Args:
            ctx: The request context.
            replica_idx: Index of the replica the request is assigned to.
            skip_tokens: When True (default), advances the context's active
                token window via ``ctx.tokens.skip_processing`` to reflect
                the reused prefix-cache blocks.  Set to False when multiple
                cache managers share a context and the caller will apply the
                skip separately.

        Returns:
            The number of tokens that were (or should be) skipped due to
            prefix-cache reuse.  Returns 0 when no blocks were reused.
        """
        self._register_replica(ctx.request_id, replica_idx)
        self.assert_runtime_invariants(ctx, replica_idx)

        if not self.enable_prefix_caching or ctx.tokens.active_length == 1:
            return 0

        # Identify a request's first admission so we record one cache-hit
        # observation per request, not one per chunked-prefill chunk.
        is_first_admission = ctx.tokens.processed_length == 0

        req_blocks = self.req_to_blocks[ctx.request_id]

        # Compute block hashes. These hashes are used by the subsequent methods.
        self.compute_hashes_for_request(ctx)

        # Query prefix cache for full blocks.
        prefix_cache_blocks = self.get_full_blocks_from_prefix_cache(
            ctx, replica_idx
        )

        if len(prefix_cache_blocks) > 0:
            # Update metrics.
            self._metrics.cache_tokens += (
                len(prefix_cache_blocks) * self.block_size
            )

            # Since we got cache hits, clear out existing uncommitted blocks
            self.release_uncommitted_blocks(
                ctx, replica_idx, skip_tokens=skip_tokens
            )

            # Append them to the request's blocks.
            req_blocks.extend(prefix_cache_blocks)
            prev_committed_idx = self.req_to_committed_idx[ctx.request_id]
            new_committed_idx = (
                prev_committed_idx + len(prefix_cache_blocks) * self.block_size
            )
            self.req_to_committed_idx[ctx.request_id] = new_committed_idx

            skip_amount = new_committed_idx - ctx.tokens.processed_length
            if skip_tokens:
                ctx.tokens.skip_processing(skip_amount)
                assert ctx.tokens.active_length >= 1, (
                    "No active tokens after prefix caching! "
                    "We should never get 100% prefix cache hit rate. "
                    "Something went wrong!"
                )
            if is_first_admission:
                ctx.cached_prefix_length = skip_amount
            return skip_amount

        if is_first_admission:
            ctx.cached_prefix_length = 0
        return 0

    @traced
    def _count_full_blocks_from_prefix_cache(
        self,
        desired_hashes: Sequence[int | bytes],
        replica_idx: int = 0,
    ) -> int:
        """Returns the count of device blocks with the desired hashes.

        A hash counts as a device hit if it is resident in *any* replica's
        device prefix cache, because a cross-replica hit is served by a
        device-to-device copy onto ``replica_idx`` rather than a recompute.
        """
        device_prefix_cache_hits = []
        desired_host_hashes = []
        for hash_value in desired_hashes:
            _, block = self._find_block_in_any_replica(hash_value, replica_idx)
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
        self, block_hash: int | bytes, preferred_replica: int
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
        desired_hashes: Sequence[int | bytes],
        replica_idx: int = 0,
    ) -> list[KVCacheBlock]:
        """Returns device blocks on ``replica_idx`` with the desired hashes.

        Blocks resident in ``replica_idx``'s own prefix cache are reused
        directly. Blocks committed on a *different* replica are materialized
        onto ``replica_idx`` via a device-to-device copy into a freshly
        allocated block, which is then committed into the local prefix cache so
        subsequent requests on this replica hit locally (SERVOPT-1500).
        """
        if self._only_use_kv_connector_last_level_cache:
            return []

        local_pool = self.device_block_pools[replica_idx]
        local_cache = local_pool.prefix_cache

        blocks: list[KVCacheBlock] = []
        for block_hash in desired_hashes:
            local_block = local_cache.get(block_hash)
            if local_block is not None:
                local_pool.touch(local_block)
                blocks.append(local_block)
                continue

            # Local miss: look for the block on another replica.
            src_replica, src_block = self._find_block_in_any_replica(
                block_hash, replica_idx
            )
            if src_block is None:
                break

            # A cross-replica hit needs a free local block to copy into, and
            # device memory handles to copy with. If either is missing, stop the
            # prefix chain here (it must remain contiguous).
            if (
                self._replica_kv_memory is None
                or local_pool.num_free_blocks == 0
            ):
                break

            # Materialize the block on this replica via a device-to-device copy
            # and commit it locally so future requests here hit directly. The
            # copy is enqueued on the destination device's default stream -- the
            # same stream the forward pass runs on -- so it is ordered before
            # the block is read, and the source block (on another replica's
            # pool, which this method never allocates from) cannot be recycled
            # before the copy is enqueued. No pinning or synchronization needed.
            dst_block = self.allocate_device_block(replica_idx)
            self._copy_block_across_replicas(
                dst_replica=replica_idx,
                src_replica=src_replica,
                dst_block_id=dst_block.bid,
                src_block_id=src_block.bid,
            )
            local_pool.commit_into_prefix_cache(block_hash, dst_block)
            blocks.append(dst_block)
            self._metrics.cross_replica_blocks_copied += 1

        return blocks

    def _copy_block_across_replicas(
        self,
        dst_replica: int,
        src_replica: int,
        dst_block_id: int,
        src_block_id: int,
    ) -> None:
        """Copies one page from ``src_replica`` to ``dst_replica`` (per shard)."""
        assert self._replica_kv_memory is not None
        src_units = self._replica_kv_memory[src_replica]
        dst_units = self._replica_kv_memory[dst_replica]
        for src_unit, dst_unit in zip(src_units, dst_units, strict=True):
            src_unit.copy_block_to(dst_unit, dst_block_id, src_block_id)

    @traced
    def _get_full_blocks_from_host_prefix_cache(
        self,
        desired_hashes: Sequence[int | bytes],
        replica_idx: int = 0,
    ) -> list[KVCacheBlock]:
        """Returns a list of device blocks with the desired hashes.

        These device blocks are newly allocated and initialized with the
        contents of the host blocks via the connector.
        """
        connector = self.connector
        pool = self.device_block_pools[replica_idx]
        if connector.num_host_blocks == 0 or not desired_hashes:
            return []

        # Limit by available device blocks.
        num_hashes_to_load = min(len(desired_hashes), pool.num_free_blocks)
        desired_hashes = desired_hashes[:num_hashes_to_load]
        blocks = [
            self.allocate_device_block(replica_idx)
            for _ in range(num_hashes_to_load)
        ]

        # Query connector for available blocks from host cache.
        block_ids = [b.bid for b in blocks]
        num_loaded = connector.load(
            block_ids,
            [to_block_hash_bytes(h) for h in desired_hashes],
            replica_idx=replica_idx,
        )

        # The connector may return fewer hashes than requested.
        for surplus_block in blocks[num_loaded:]:
            pool.free_block(surplus_block)
        loaded_blocks = blocks[:num_loaded]
        loaded_hashes = desired_hashes[:num_loaded]

        # Commit the device blocks into the device prefix cache.
        for block, block_hash in zip(loaded_blocks, loaded_hashes, strict=True):
            if block_hash in pool.prefix_cache:
                # When this env var is set, we may perform host/disk -> device
                # transfers of blocks already resident in the device prefix cache.
                # If the block is already in the device prefix cache, we skip the
                # commit.
                assert self._only_use_kv_connector_last_level_cache
                continue
            pool.commit_into_prefix_cache(block_hash, block)

        return loaded_blocks

    @traced
    def count_full_blocks_from_prefix_caches(
        self, ctx: TextContext, replica_idx: int = 0
    ) -> int:
        """Returns the number of computed (cached) blocks related to this request.

        Note that only full blocks are counted.
        """
        if not self.enable_prefix_caching or ctx.tokens.active_length == 1:
            return 0

        self.compute_hashes_for_request(ctx)
        num_committed_blocks = (
            self.req_to_committed_idx[ctx.request_id] // self.block_size
        )
        req_hashes = self.req_to_hashes[ctx.request_id]
        uncommitted_hashes = req_hashes[num_committed_blocks:]

        return self._count_full_blocks_from_prefix_cache(
            uncommitted_hashes, replica_idx
        )

    @traced
    def get_full_blocks_from_prefix_cache(
        self, ctx: TextContext, replica_idx: int = 0
    ) -> list[KVCacheBlock]:
        """Gets the computed (cached) blocks for the request.

        Note that the computed blocks must be full.
        """
        assert self.enable_prefix_caching

        req_hashes = self.req_to_hashes[ctx.request_id]
        num_committed_blocks = (
            self.req_to_committed_idx[ctx.request_id] // self.block_size
        )
        uncommitted_hashes = req_hashes[num_committed_blocks:]

        # query the device prefix cache for full blocks
        device_blocks = self._get_full_blocks_from_device_prefix_cache(
            uncommitted_hashes, replica_idx
        )

        if self.connector.num_host_blocks == 0:
            return device_blocks

        # remove the hashes that were found in the device prefix cache
        if len(device_blocks) > 0:
            uncommitted_hashes = uncommitted_hashes[len(device_blocks) :]

        # query the host prefix cache for full blocks via connector
        host_blocks = self._get_full_blocks_from_host_prefix_cache(
            uncommitted_hashes, replica_idx
        )
        return device_blocks + host_blocks

    @traced
    def commit_to_prefix_cache(
        self,
        ctx: TextContext,
        replica_idx: int = 0,
    ) -> None:
        """Commits all blocks whose hashes are known for prefix caching.

        This increments the committed_idx.

        Args:
            ctx: TextContext.
            replica_idx: Index of the replica the request is assigned to.
        """
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

        # Queue the newly-committed blocks as one ordered offload sequence. Its
        # parent is the block immediately before this run in the prefix
        # (None = root); that block was committed and offloaded in a previous
        # step.
        if num_computed_blocks > num_committed_blocks:
            parent_seq_hash = (
                req_hashes[num_committed_blocks - 1]
                if num_committed_blocks > 0
                else None
            )
            new_block_hashes = req_hashes[
                num_committed_blocks:num_computed_blocks
            ]
            self._pending_offloads[replica_idx].append(
                (parent_seq_hash, new_block_hashes)
            )

        # Bump the committed index.
        self.req_to_committed_idx[ctx.request_id] = (
            num_computed_blocks * self.block_size
        )

    def offload(self, replica_idx: int = 0) -> None:
        """Offload the pending sequences to the replica's connector.

        Each pending sequence is delivered as one ordered ``offload`` call so
        connectors can chain it onto ``parent_seq_hash``. Hashes are re-resolved
        to their current device blocks here; if a block was evicted since it was
        committed, the run is truncated at that point (the remaining blocks'
        parent would be absent), so the connector never sees a gap-chain.
        """
        prefix_cache = self.device_block_pools[replica_idx].prefix_cache
        connector = self.connector
        for parent_seq_hash, hashes in self._pending_offloads[replica_idx]:
            block_ids = []
            block_hashes = []
            for block_hash in hashes:
                if block_hash not in prefix_cache:
                    # Block evicted since commit; truncate the run here.
                    break
                block_ids.append(prefix_cache[block_hash].bid)
                block_hashes.append(block_hash)
            if block_hashes:
                connector.offload(
                    block_ids,
                    [to_block_hash_bytes(h) for h in block_hashes],
                    None
                    if parent_seq_hash is None
                    else to_block_hash_bytes(parent_seq_hash),
                    replica_idx=replica_idx,
                )
        self._pending_offloads[replica_idx].clear()

    def release(self, request_id: RequestID, replica_idx: int = 0) -> None:
        """Release the blocks for the request."""
        pool = self.device_block_pools[replica_idx]
        blocks = self.req_to_blocks[request_id]
        ordered_blocks: Iterable[KVCacheBlock] = blocks
        if self.enable_prefix_caching:
            # Free blocks in reverse order so that the tail blocks are
            # freed first.
            ordered_blocks = reversed(blocks)

        for block in ordered_blocks:
            pool.free_block(block)

        self.req_to_blocks[request_id] = []
        self.req_to_hashes[request_id] = []
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
        num_draft_tokens_per_step: int = 1,
        replica_idx: int = 0,
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
                per draft forward. One for autoregressive drafts
                (``eagle``, ``mtp``); equal to
                ``num_draft_tokens`` for block drafts (``dflash``). Used by
                ``_compute_seq_len`` to size the cache for block drafts,
                whose ``forward_block`` writes one extra position past the
                bonus token.
            replica_idx: Index of the replica the request is assigned to.

        Raises:
            InsufficientBlocksError: If there are insufficient free blocks to
            satisfy the allocation.
        """
        self._register_replica(ctx.request_id, replica_idx)
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

        # Verify that committed tokens fit within the currently allocated
        # blocks.  We check against committed_idx (block-manager-internal
        # state) rather than ctx.tokens.processed_length, because the latter
        # is shared across multiple cache managers and may not reflect this
        # cache's state when skip_tokens=False is used.
        current_blocks = self.req_to_blocks[ctx.request_id]
        num_current_blocks = len(current_blocks)
        committed_idx = self.req_to_committed_idx[ctx.request_id]
        assert committed_idx <= (num_current_blocks * self.block_size), (
            f"Expected at least {ceildiv(committed_idx, self.block_size)} "
            f"blocks to store KV for {committed_idx} committed tokens, but "
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
        num_draft_tokens_per_step: int = 1,
    ) -> int:
        """Calculates the number of new blocks to allocate for a request.

        Args:
            ctx: The request context containing sequence information and token indices.
            num_draft_tokens: Total draft tokens generated per speculative
                iteration. Zero for non-speculative decode.
            num_draft_tokens_per_step: Number of draft KV positions written
                per draft forward. One for autoregressive drafts
                (``eagle``, ``mtp``); equal to
                ``num_draft_tokens`` for block drafts (``dflash``). Used by
                ``_compute_seq_len`` to size the cache for block drafts,
                whose ``forward_block`` writes one extra position past the
                bonus token.

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
        replica_idx: int = 0,
        skip_tokens: bool = True,
    ) -> None:
        """Release the uncommitted blocks for the request."""
        pool = self.device_block_pools[replica_idx]
        req_blocks = self.req_to_blocks[ctx.request_id]
        num_committed_blocks = (
            self.req_to_committed_idx[ctx.request_id] // self.block_size
        )
        assert len(req_blocks) >= num_committed_blocks
        num_uncommitted_blocks = len(req_blocks) - num_committed_blocks
        for _ in range(num_uncommitted_blocks):
            block = req_blocks.pop()
            pool.free_block(block)
        if skip_tokens:
            delta = (
                ctx.tokens.processed_length
                - self.req_to_committed_idx[ctx.request_id]
            )
            if delta > 0:
                ctx.tokens.rewind_processing(delta)
            elif delta < 0:
                ctx.tokens.skip_processing(-delta)

    def register_dummy_request(
        self, request_id: RequestID, replica_idx: int = 0
    ) -> None:
        """Maps a dummy request to the replica pool's reserved null block."""
        assert self.req_to_blocks[request_id] == []
        self._register_replica(request_id, replica_idx)
        self.req_to_blocks[request_id] = [
            self.device_block_pools[replica_idx].null_block
        ]

    @traced
    def get_req_blocks(
        self, request_id: RequestID, replica_idx: int = 0
    ) -> list[int]:
        """Get the block ids for a request."""
        return [block.bid for block in self.req_to_blocks[request_id]]

    @traced
    def reset_prefix_cache(self) -> None:
        """Resets the device prefix caches for all replicas.

        Note: Host prefix cache reset is handled by the connector.
        """
        for pool in self.device_block_pools:
            pool.reset_prefix_cache()

    @property
    def metrics(self) -> KVCacheMetrics:
        """Returns combined metrics for this manager and its connector."""
        return self._metrics + self.connector.metrics

    def reset_metrics(self) -> None:
        """Resets local metrics to zero."""
        self._metrics = KVCacheMetrics()

    @traced
    def assert_runtime_invariants(
        self, ctx: TextContext, replica_idx: int = 0
    ) -> None:
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
