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
from collections.abc import Iterable

from max.interfaces import (
    RequestID,
    TextGenerationContext,
    VLMTextGenerationContext,
)
from max.kv_cache.kv_connector import KVConnector
from max.kv_cache.memory_tier import MemoryTier
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.profiler import traced
from max.support.math import ceildiv

from .block_pool import BlockPool
from .block_utils import (
    DEFAULT_PARENT_HASH,
    InsufficientBlocksError,
    KVCacheBlock,
    hash_request_tokens,
)

logger = logging.getLogger("max.pipelines")


def _compute_seq_len(
    ctx: TextGenerationContext, num_steps: int, num_speculative_steps: int
) -> int:
    # Each term accounts for one category of tokens that need a KV slot:
    #
    #   ctx.tokens                    : prompt + tokens generated so far
    #   maybe_accepted_draft_tokens   : draft tokens being verified in the
    #                                   *previous* batch (overlap scheduler);
    #                                   conservative: assume all are accepted
    #   num_speculative_steps (x1)    : draft tokens to verify in the *next* batch
    #   num_speculative_steps (x1)    : draft tokens generated *during* that batch
    #   num_steps                     : regular decode steps
    #   -1                            : the last generated token has no KV entry
    #
    # NOTE: `draft_tokens_to_verify` is intentionally excluded. Using
    # `2 * num_speculative_steps` unconditionally is always correct and
    # avoids an under-allocation when `draft_tokens_to_verify` is empty but
    # the pipeline populates it with dummy draft tokens (_MAGIC_DRAFT_TOKEN_ID).
    seq_len = (
        len(ctx.tokens)
        + len(ctx.spec_decoding_state.maybe_accepted_draft_tokens)
        + 2 * num_speculative_steps
        + num_steps
        - 1
    )
    return seq_len


def _resolve_only_use_kv_connector_last_level_cache() -> bool:
    """Resolve whether to only use the KVConnector last level cache.

    When this is set, the device prefix cache will be disabled. All KVCache hits
    will stricly be served from the KVConnector. This is primarily used for
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
    """Manages allocation and deallocation of paged KV cache blocks."""

    @traced
    def __init__(
        self,
        device_memory_tier: MemoryTier,
        total_num_blocks: int,
        block_size: int,
        connector: KVConnector,
        enable_prefix_caching: bool,
        enable_runtime_checks: bool = False,
    ) -> None:
        self.total_num_blocks = total_num_blocks
        self.block_size = block_size

        # Whether to enable prefix caching.
        self.enable_prefix_caching = enable_prefix_caching

        # Connector for external cache tiers (host memory, etc.)
        # The connector owns host memory, host block pool, and H2D/D2H transfers.
        self.connector = connector

        # A pool of device blocks.
        self.device_block_pool = BlockPool(
            device_memory_tier,
            total_num_blocks,
            enable_prefix_caching,
            enable_runtime_checks=enable_runtime_checks,
        )

        # Mapping from request ID to blocks to track the blocks allocated
        # for each request, so that we can free the blocks when the request
        # is finished.
        self.req_to_blocks: dict[RequestID, list[KVCacheBlock]] = defaultdict(
            list
        )

        # Mapping from request ID to kv block hashes.
        # This is to avoid recomputing the block hashes for each call of
        # `get_computed_blocks` or `allocate_slots`.
        self.req_to_hashes: dict[RequestID, list[int]] = defaultdict(list)

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

    @traced
    def step(self, ctx: TextGenerationContext) -> None:
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
        ctx: TextGenerationContext,
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

        parent_hash_value = None
        if len(hashes) > 0:
            parent_hash_value = hashes[-1]

        unhashed_tokens = ctx.tokens[num_hashed_tokens:num_hashable_tokens]

        images = ctx.images if isinstance(ctx, VLMTextGenerationContext) else []
        new_hashes = hash_request_tokens(
            token_ids=unhashed_tokens,
            block_size=self.block_size,
            parent_hash=parent_hash_value,
            prefix_length=num_hashed_tokens,
            images=images,
        )
        hashes.extend(new_hashes)

    @traced
    def reuse_blocks_from_prefix_cache(
        self,
        ctx: TextGenerationContext,
        skip_tokens: bool = True,
    ) -> int:
        """Reuses blocks from prefix cache.

        Full blocks are directly reused and appended to the request's blocks.
        Partial blocks can be reused via COW.

        Args:
            ctx: The request context.
            skip_tokens: When True (default), advances the context's active
                token window via ``ctx.tokens.skip_processing`` to reflect
                the reused prefix-cache blocks.  Set to False when multiple
                cache managers share a context and the caller will apply the
                skip separately.

        Returns:
            The number of tokens that were (or should be) skipped due to
            prefix-cache reuse.  Returns 0 when no blocks were reused.
        """
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
            self.release_uncommitted_blocks(ctx, skip_tokens=skip_tokens)

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
        self, ctx: TextGenerationContext, desired_hashes: list[int]
    ) -> int:
        """Returns the count of device and host blocks with the desired hashes."""
        # Count the number of device block hashes that are in the device prefix cache.
        device_prefix_cache = self.device_block_pool.hash_to_committed_block

        device_prefix_cache_hits = []
        desired_host_hashes = []
        for hash_value in desired_hashes:
            if hash_value in device_prefix_cache:
                # Device hashes with prefix cache hit
                device_prefix_cache_hits.append(hash_value)
            else:
                # Record potential host hash
                desired_host_hashes.append(hash_value)

        # Ignoring host cache hits in this calculation as it may be expensive
        # to compute. Eg: due to querying an external process.
        device_prefix_cache_hit_count = len(device_prefix_cache_hits)

        return device_prefix_cache_hit_count

    @traced
    def _get_full_blocks_from_device_prefix_cache(
        self,
        desired_hashes: list[int],
    ) -> list[KVCacheBlock]:
        """Returns a list of device blocks with the desired hashes."""
        if self._only_use_kv_connector_last_level_cache:
            return []

        device_prefix_cache = self.device_block_pool.hash_to_committed_block

        blocks = []
        for block_hash in desired_hashes:
            hash_value = block_hash
            if hash_value not in device_prefix_cache:
                break
            block = device_prefix_cache[hash_value]
            blocks.append(block)
            self.device_block_pool.touch(block)

        return blocks

    @traced
    def _get_full_blocks_from_host_prefix_cache(
        self, desired_hashes: list[int]
    ) -> list[KVCacheBlock]:
        """Returns a list of device blocks with the desired hashes.

        These device blocks are newly allocated and initialized with the
        contents of the host blocks via the connector.
        """
        if self.connector.num_host_blocks == 0 or not desired_hashes:
            return []

        # Limit by available device blocks.
        num_hashes_to_load = min(
            len(desired_hashes), len(self.device_block_pool.free_block_queue)
        )
        desired_hashes = desired_hashes[:num_hashes_to_load]
        blocks = [
            self.allocate_device_block() for _ in range(num_hashes_to_load)
        ]

        # Query connector for available blocks from host cache.
        block_ids = [b.bid for b in blocks]
        num_loaded = self.connector.load(block_ids, desired_hashes)

        # The connector may return fewer hashes than requested.
        for surplus_block in blocks[num_loaded:]:
            self.device_block_pool.free_block(surplus_block)
        loaded_blocks = blocks[:num_loaded]
        loaded_hashes = desired_hashes[:num_loaded]

        # Commit the device blocks into the device prefix cache.
        for block, block_hash in zip(loaded_blocks, loaded_hashes, strict=True):
            if block_hash in self.device_block_pool.hash_to_committed_block:
                # When this env var is set, we may perform host/disk -> device
                # transfers of blocks already resident in the device prefix cache.
                # If the block is already in the device prefix cache, we skip the
                # commit.
                assert self._only_use_kv_connector_last_level_cache
                continue
            self.device_block_pool.commit_into_prefix_cache(block_hash, block)

        return loaded_blocks

    @traced
    def count_full_blocks_from_prefix_caches(
        self, ctx: TextGenerationContext
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
            ctx, uncommitted_hashes
        )

    @traced
    def get_full_blocks_from_prefix_cache(
        self, ctx: TextGenerationContext
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
            uncommitted_hashes
        )

        if self.connector.num_host_blocks == 0:
            return device_blocks

        # remove the hashes that were found in the device prefix cache
        if len(device_blocks) > 0:
            uncommitted_hashes = uncommitted_hashes[len(device_blocks) :]

        # query the host prefix cache for full blocks via connector
        host_blocks = self._get_full_blocks_from_host_prefix_cache(
            uncommitted_hashes
        )
        return device_blocks + host_blocks

    @traced
    def commit_to_prefix_cache(
        self,
        ctx: TextGenerationContext,
    ) -> None:
        """Commits all blocks whose hashes are known for prefix caching.

        This increments the committed_idx.

        Args:
            ctx: TextGenerationContext.
        """
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

            new_block = self.device_block_pool.get_or_commit_into_prefix_cache(
                block_hash, block
            )
            # If the block is already int the prefix cache, we skip the commit.
            # Then we overwrite the req blocks with the existing block that contains
            # the same contents.
            if new_block is not None:
                req_blocks[block_idx] = new_block

        # Bump the committed index.
        self.req_to_committed_idx[ctx.request_id] = (
            num_computed_blocks * self.block_size
        )

        # Save the blocks to the connector.
        current_parent = (
            req_hashes[num_committed_blocks - 1]
            if num_committed_blocks > 0
            else DEFAULT_PARENT_HASH
        )
        block_ids = [
            block.bid
            for block in req_blocks[num_committed_blocks:num_computed_blocks]
        ]
        block_hashes = req_hashes[num_committed_blocks:num_computed_blocks]
        self.connector.save(
            block_ids, block_hashes, parent_seq_hash=current_parent
        )

    def release(self, request_id: RequestID) -> None:
        """Release the blocks for the request."""
        blocks = self.req_to_blocks[request_id]
        ordered_blocks: Iterable[KVCacheBlock] = blocks
        if self.enable_prefix_caching:
            # Free blocks in reverse order so that the tail blocks are
            # freed first.
            ordered_blocks = reversed(blocks)

        for block in ordered_blocks:
            self.device_block_pool.free_block(block)

        self.req_to_blocks[request_id] = []
        self.req_to_hashes[request_id] = []

        # Committed idx is only used with the prefix cache
        # therefore this may not always be in the dict.
        if request_id in self.req_to_committed_idx:
            del self.req_to_committed_idx[request_id]

    @traced
    def allocate_new_blocks(
        self,
        ctx: TextGenerationContext,
        num_steps: int = 1,
        num_speculative_steps: int = 0,
    ) -> None:
        """Allocate new blocks for a request to accommodate additional tokens.

        Calculates the number of additional blocks needed based on the current sequence
        length and number of steps, then allocates them from the device block pool.
        Validates that there are sufficient free blocks available and that the current
        blocks can accommodate the completed tokens.

        Args:
            ctx: The request context containing sequence information and token indices.
            num_steps: Number of additional steps to allocate blocks for. Defaults to 1.
            num_speculative_steps: Number of speculative steps to allocate blocks for. Defaults to 0.

        Raises:
            InsufficientBlocksError: If there are insufficient free blocks to
            satisfy the allocation.
        """
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
            ctx, num_steps, num_speculative_steps
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
        if num_new_blocks > len(self.device_block_pool.free_block_queue):
            raise InsufficientBlocksError(
                f"Cannot get {num_new_blocks} free blocks from the free block queue (only {len(self.device_block_pool.free_block_queue)} available)"
            )

        # Allocate new blocks.
        for _ in range(num_new_blocks):
            new_block = self.allocate_device_block()
            current_blocks.append(new_block)

    @traced
    def num_blocks_to_allocate(
        self,
        ctx: TextGenerationContext,
        num_steps: int = 1,
        num_speculative_steps: int = 0,
    ) -> int:
        """Calculates the number of new blocks to allocate for a request.

        Args:
            ctx: The request context containing sequence information and token indices.
            num_steps: Number of additional steps to allocate blocks for. Defaults to 1.
            num_speculative_steps: Number of speculative steps to allocate blocks for. Defaults to 0.

        Returns:
            The number of new blocks to allocate.
        """
        current_blocks = self.req_to_blocks[ctx.request_id]
        num_current_blocks = len(current_blocks)
        current_seq_len = _compute_seq_len(
            ctx, num_steps, num_speculative_steps
        )
        num_required_blocks = ceildiv(current_seq_len, self.block_size)
        num_new_blocks = num_required_blocks - num_current_blocks

        return max(num_new_blocks, 0)

    @traced
    def allocate_device_block(self) -> KVCacheBlock:
        """Allocates a single block from the device block pool."""
        new_block, _ = self.device_block_pool.alloc_block()
        return new_block

    def release_uncommitted_blocks(
        self,
        ctx: TextGenerationContext,
        skip_tokens: bool = True,
    ) -> None:
        """Release the uncommitted blocks for the request."""
        req_blocks = self.req_to_blocks[ctx.request_id]
        num_committed_blocks = (
            self.req_to_committed_idx[ctx.request_id] // self.block_size
        )
        assert len(req_blocks) >= num_committed_blocks
        num_uncommitted_blocks = len(req_blocks) - num_committed_blocks
        for _ in range(num_uncommitted_blocks):
            block = req_blocks.pop()
            self.device_block_pool.free_block(block)
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
        self, request_id: RequestID, sentinel_request_id: RequestID
    ) -> None:
        """Maps a dummy request to the sentinel's block via ref-count sharing."""
        sentinel_blocks = self.req_to_blocks[sentinel_request_id]
        assert len(sentinel_blocks) == 1
        sentinel_block = sentinel_blocks[0]
        self.device_block_pool.touch(sentinel_block)
        self.req_to_blocks[request_id] = [sentinel_block]

    @traced
    def get_req_blocks(self, request_id: RequestID) -> list[int]:
        """Get the block ids for a request."""
        return [block.bid for block in self.req_to_blocks[request_id]]

    @traced
    def reset_prefix_cache(self) -> None:
        """Resets the device prefix cache.

        Note: Host prefix cache reset is handled by the connector.
        """
        self.device_block_pool.reset_prefix_cache()

    @property
    def metrics(self) -> KVCacheMetrics:
        """Returns combined metrics for this manager and its connector."""
        return self._metrics + self.connector.metrics

    def reset_metrics(self) -> None:
        """Resets local metrics to zero."""
        self._metrics = KVCacheMetrics()

    @traced
    def assert_runtime_invariants(self, ctx: TextGenerationContext) -> None:
        """Asserts runtime invariants when runtime checks are enabled."""
        if not self.enable_runtime_checks:
            return

        # Get the active block ids
        active_block_ids = []
        for blocks in self.req_to_blocks.values():
            for block in blocks:
                active_block_ids.append(block.bid)
                # Check that all active blocks have a ref_cnt > 0
                assert block.ref_cnt > 0

        # Check that the block pool is consistent
        self.device_block_pool.assert_runtime_invariants(active_block_ids)

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
