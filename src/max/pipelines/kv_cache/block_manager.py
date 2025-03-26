# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

TODO E2EOPT-116: Port block_manager.py and block_utils.py to Mojo
"""

from __future__ import annotations

import logging
import multiprocessing
from collections import defaultdict
from typing import Iterable, Optional

import numpy as np
from max.profiler import traced
from max.serve.kvcache_agent.kvcache_agent import KVCacheChangeMessage
from max.serve.kvcache_agent.kvcache_agent_service_v1_pb2 import (  # type: ignore
    MemoryTier,
    UpdateType,
)
from max.support.math import ceildiv

from .block_utils import (
    ROOT_BLOCK_HASH,
    BlockHashType,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    hash_block_tokens,
    hash_request_tokens,
)
from .paged_cache_metadata import PagedCacheMetadata
from .simple_trie import SimpleTrie

logger = logging.getLogger("max.pipelines")


class BlockManager:
    @traced
    def __init__(
        self,
        total_num_blocks: int,
        block_size: int,
        enable_prefix_caching: bool,
        enable_runtime_checks: bool = False,
    ):
        # The number of tokens in a single page.
        self.block_size = block_size

        # Whether to enable prefix caching.
        self.enable_prefix_caching = enable_prefix_caching

        # The total number of blocks we'll have per-device.
        self.total_num_blocks = total_num_blocks

        # A Block pool of all kv-cache blocks.
        self.block_pool: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(self.total_num_blocks)
        ]

        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        self.free_block_queue = FreeKVCacheBlockQueue(self.block_pool)

        # {block_hash: block}. A committed block is a full block with a block
        # hash that can be used for prefix caching.
        # The cached block may be used by running requests or in the free_block_queue
        # that could potentially be evicted.
        self.committed_block_hash_to_block: dict[
            BlockHashType, KVCacheBlock
        ] = {}

        # Mapping from parent block hash to child block hash.
        self.parent_to_child_hash: dict[int, SimpleTrie] = defaultdict(
            SimpleTrie
        )

        # Mapping from request ID to blocks to track the blocks allocated
        # for each request, so that we can free the blocks when the request
        # is finished.
        self.req_to_blocks: dict[int, list[KVCacheBlock]] = defaultdict(list)

        # Mapping from request ID to kv block hashes.
        # This is to avoid recomputing the block hashes for each call of
        # `get_computed_blocks` or `allocate_slots`.
        self.req_to_block_hashes: dict[int, list[BlockHashType]] = defaultdict(
            list
        )

        # Cache hit rate metrics.
        self.prompt_tokens = 0
        self.cached_prompt_tokens = 0

        # Whether to enable runtime checks.
        self.enable_runtime_checks = enable_runtime_checks

        # Queue for the KV Cache Agent updates
        self.kv_cache_agent_queue: Optional[multiprocessing.Queue] = None

    @traced
    def fetch(
        self, seq_id: int, data: PagedCacheMetadata
    ) -> tuple[list[int], Optional[tuple[int, int, int]]]:
        """Fetch the block ids used by a request.

        Args:
            seq_id: The ID of the request.
            data: The data of the request.

        Returns:
            A list of block IDs.
        """
        self.assert_runtime_invariants(seq_id, data)

        cow_args = self.reuse_from_prefix_cache(seq_id, data)

        # TODO E2EOPT-111:
        # Commit the blocks whose hashes are known for prefix caching. This lets
        # one request from a batch to write to a kv entry and other requests
        # from the same batch can read from that same kv entry.
        # if self.enable_prefix_caching:
        #     self.commit_full_blocks(seq_id, data)

        self.allocate_new_blocks(seq_id, data)

        blocks = self.get_req_blocks(seq_id)

        self.assert_runtime_invariants(seq_id, data)
        return blocks, cow_args

    @traced
    def step(self, seq_id: int, data: PagedCacheMetadata) -> None:
        """Step the block manager by committing blocks into prefix cache."""
        self.assert_runtime_invariants(seq_id, data)

        if not self.enable_prefix_caching:
            return

        # Compute block hashes. These hashes are used by the subsequent methods.
        self.compute_block_hashes_for_request(seq_id, data)

        # Now that we generated new tokens, we can possibly commit additional
        # blocks into prefix cache.
        self.commit_to_prefix_cache(seq_id, data)

        self.assert_runtime_invariants(seq_id, data)

    @traced
    def query_fetch_stats(
        self, seq_id: int, data: PagedCacheMetadata
    ) -> tuple[set[int], int, int]:
        """Query about the stats about running the fetch operation for a given
        sequence. It is OK if some seq_id are not in the cache.

        This method does not modify the state of the paged cache.

        Returns:
            - prefix_cache_blocks: Prefix cache blocks that would be reused for this seq.
            - tokens_to_encode: Number of tokens in prompt we need to encode when running the fetch.
            - new_pages_needed: Number of new pages we need to allocate when running the fetch.
        """
        prefix_cache_block_ids = []
        if self.enable_prefix_caching:
            # Compute block hashes. These hashes are used by the subsequent methods.
            self.compute_block_hashes_for_request(seq_id, data)

            # Query prefix cache for full blocks.
            prefix_cache_blocks = self.get_full_blocks_from_prefix_cache(
                seq_id, data
            )
            for block in prefix_cache_blocks:
                prefix_cache_block_ids.append(block.block_id)

        # Determine number of new blocks to allocate.
        num_required_blocks = ceildiv(data.seq_len, self.block_size)
        req_blocks = self.req_to_blocks[seq_id]
        num_new_blocks = (
            num_required_blocks - len(req_blocks) - len(prefix_cache_block_ids)
        )
        assert num_new_blocks >= 0

        # Determine the number of tokens to encode.
        new_cached_idx = max(
            data.cached_idx,
            data.committed_idx + len(prefix_cache_block_ids) * self.block_size,
        )
        tokens_to_encode = data.inflight_idx - new_cached_idx

        # Empty out the request info if this is not a real request.
        if seq_id < 0:
            if seq_id in self.req_to_blocks:
                assert len(self.req_to_blocks[seq_id]) == 0
                del self.req_to_blocks[seq_id]
            if seq_id in self.req_to_block_hashes:
                del self.req_to_block_hashes[seq_id]

        return set(prefix_cache_block_ids), tokens_to_encode, num_new_blocks

    @traced
    def compute_block_hashes_for_request(
        self,
        seq_id: int,
        data: PagedCacheMetadata,
    ):
        """Compute the block hashes for the request."""

        block_hashes = self.req_to_block_hashes[seq_id]
        parent_block_hash_value = None
        if len(block_hashes) > 0:
            parent_block_hash_value = block_hashes[-1].hash_value

        unhashed_tokens = data.tokens[
            len(block_hashes) * self.block_size : data.inflight_idx
        ]
        new_block_hashes = hash_request_tokens(
            self.block_size, unhashed_tokens, parent_block_hash_value
        )
        block_hashes.extend(new_block_hashes)

    @traced
    def reuse_from_prefix_cache(
        self, seq_id: int, data: PagedCacheMetadata
    ) -> Optional[tuple[int, int, int]]:
        """Reuse blocks from prefix cache.

        Full blocks are directly reused and appended to the request's blocks.
        Partial blocks can be reused via COW. The blocks/tokens to copy to and
        from are returned as a tuple.

        This also updates the cache hit rate metrics.

        Returns:
            (fresh_block_id, partial_block_id, tokens_matched) if a partial block is reused.
            None if no partial block is reused.
        """
        req_blocks = self.req_to_blocks[seq_id]

        # Update cache hit rate metrics.
        orig_prompt_len = data.num_prompt_tokens
        self.prompt_tokens += orig_prompt_len - 1

        if not self.enable_prefix_caching:
            return None

        # Compute block hashes. These hashes are used by the subsequent methods.
        self.compute_block_hashes_for_request(seq_id, data)

        # Query prefix cache for full blocks.
        prefix_cache_blocks = self.get_full_blocks_from_prefix_cache(
            seq_id, data
        )

        orig_cached_idx = data.cached_idx

        if len(prefix_cache_blocks) > 0:
            # Touch the computed blocks to make sure they won't be evicted.
            self.touch(prefix_cache_blocks)

            # Since we got cache hits, clear out existing uncommitted blocks
            self.release_uncommitted_blocks(seq_id, data)

            # Append them to the request's blocks.
            req_blocks.extend(prefix_cache_blocks)
            data.committed_idx += len(prefix_cache_blocks) * self.block_size
            data.cached_idx = data.committed_idx

            # Check that the cached_idx has increased.
            assert data.cached_idx > orig_cached_idx
            orig_cached_idx = data.cached_idx

        # Query prefix cache for partial blocks
        partial_block, tokens_matched = (
            self.get_partial_block_from_prefix_cache(seq_id, data)
        )
        cow_args = None
        if partial_block is not None:
            # Since we got cache hits, clear out existing uncommitted blocks
            self.release_uncommitted_blocks(seq_id, data)

            # Append them to the request's blocks.
            fresh_block = self.alloc_block()
            req_blocks.append(fresh_block)
            data.cached_idx += tokens_matched

            # Update COW arguments
            cow_args = (
                fresh_block.block_id,
                partial_block.block_id,
                tokens_matched,
            )

            # Check that the cached_idx has increased.
            assert data.cached_idx > orig_cached_idx
            orig_cached_idx = data.cached_idx

        # Update cache hit rate metrics.
        new_prompt_len = data.num_prompt_tokens
        self.cached_prompt_tokens += orig_prompt_len - new_prompt_len

        return cow_args

    @traced
    def get_full_blocks_from_prefix_cache(
        self,
        seq_id: int,
        data: PagedCacheMetadata,
    ) -> list[KVCacheBlock]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A list of blocks that are computed for the request.
        """

        assert self.enable_prefix_caching

        req_block_hashes = self.req_to_block_hashes[seq_id]
        num_committed_blocks = data.committed_idx // self.block_size
        # we exclude the last inflight token to ensure that there is at least
        # one prompt token to be encoded.
        num_inflight_blocks = (data.inflight_idx - 1) // self.block_size
        uncommitted_block_hashes = req_block_hashes[
            num_committed_blocks:num_inflight_blocks
        ]

        prefix_cache_blocks = []
        for block_hash in uncommitted_block_hashes:
            if block_hash not in self.committed_block_hash_to_block:
                break
            prefix_cache_block = self.committed_block_hash_to_block[block_hash]
            prefix_cache_blocks.append(prefix_cache_block)

        return prefix_cache_blocks

    @traced
    def get_partial_block_from_prefix_cache(
        self,
        seq_id: int,
        data: PagedCacheMetadata,
    ) -> tuple[Optional[KVCacheBlock], int]:
        """Get the computed (cached) blocks for the request."""
        assert self.enable_prefix_caching

        if self.block_size == 1:
            return None, 0

        req_block_hashes = self.req_to_block_hashes[seq_id]
        num_committed_blocks = data.committed_idx // self.block_size

        parent_hash = ROOT_BLOCK_HASH
        if num_committed_blocks > 0:
            parent_hash = req_block_hashes[num_committed_blocks - 1]
        parent_tokens = data.tokens[
            num_committed_blocks * self.block_size : (data.inflight_idx - 1)
        ]
        if len(parent_tokens) == 0:
            return None, 0

        # Find the longest prefix match in the prefix cache.
        children = self.parent_to_child_hash[parent_hash.hash_value]

        parent_tokens = parent_tokens[: self.block_size]
        res = children.find_string_with_largest_common_prefix(
            tuple(parent_tokens)
        )
        if res is None:
            return None, 0
        best_child_tokens, best_tokens_matched = res

        # It is not profitable to do COW if this request's partial block has
        # at least as many tokens as the best match in the prefix cache.
        current_tokens_in_partial_block = data.cached_idx % self.block_size
        if current_tokens_in_partial_block >= best_tokens_matched:
            return None, 0

        child_hash = hash_block_tokens(
            parent_hash.hash_value,
            np.array(best_child_tokens),
        )
        child_block = self.committed_block_hash_to_block[child_hash]
        return child_block, best_tokens_matched

    @traced
    def commit_to_prefix_cache(
        self,
        seq_id: int,
        data: PagedCacheMetadata,
    ) -> None:
        """Commits all blocks whose hashes are known for prefix caching.

        This increments the committed_idx.

        Args:
            seq_id: The ID of the request.
            data: The data of the request.
        """

        req_blocks = self.req_to_blocks[seq_id]
        req_block_hashes = self.req_to_block_hashes[seq_id]
        num_committed_blocks = data.committed_idx // self.block_size

        # Count the number of tokens for which we know the values of and align
        # to the block size.
        num_computed_blocks = data.cached_idx // self.block_size

        # Commit these blocks into the prefix cache.
        for block_idx in range(num_committed_blocks, num_computed_blocks):
            block = req_blocks[block_idx]

            # Get the block hash.
            block_hash = req_block_hashes[block_idx]

            if block_hash in self.committed_block_hash_to_block:
                # Check if a block with the same hash is already committed.
                # If so, we reuse the already committed block.
                prefix_cache_block = self.committed_block_hash_to_block[
                    block_hash
                ]
                if block.block_id == prefix_cache_block.block_id:
                    continue

                self.touch([prefix_cache_block])
                req_blocks[block_idx] = prefix_cache_block

                # Free the block we currently have.
                assert block.block_hash is None
                block.ref_cnt -= 1
                if block.ref_cnt == 0:
                    self.free_block_queue.append(block)
            else:
                # Update and added the full block to the cache.
                assert block.block_hash is None
                block.block_hash = block_hash
                self.committed_block_hash_to_block[block_hash] = block
                if self.kv_cache_agent_queue is not None:
                    logger.debug(
                        f"Updating KV Cache Agent with block {block_hash.hash_value}, memory tier {MemoryTier.MEMORY_TIER_GPU}, update type {UpdateType.UPDATE_TYPE_ADDED}"
                    )
                    self.kv_cache_agent_queue.put(
                        KVCacheChangeMessage(
                            cache_id=str(block_hash.hash_value),
                            memory_tier=MemoryTier.MEMORY_TIER_GPU,
                            update_type=UpdateType.UPDATE_TYPE_ADDED,
                        )
                    )

                parent_block_hash = ROOT_BLOCK_HASH
                if block_idx > 0:
                    parent_block_hash = req_block_hashes[block_idx - 1]
                tokens = data.tokens[
                    block_idx * self.block_size : (block_idx + 1)
                    * self.block_size
                ]
                self.parent_to_child_hash[parent_block_hash.hash_value].insert(
                    tuple(tokens)
                )

        data.committed_idx = num_computed_blocks * self.block_size

    def release(self, seq_id: int) -> None:
        """Release the blocks for the request."""

        blocks = self.req_to_blocks[seq_id]
        ordered_blocks: Iterable[KVCacheBlock] = blocks
        if self.enable_prefix_caching:
            # Free blocks in reverse order so that the tail blocks are
            # freed first.
            ordered_blocks = reversed(blocks)

        for block in ordered_blocks:
            self.free_block(block)

        self.req_to_blocks[seq_id] = []
        self.req_to_block_hashes[seq_id] = []

    @traced
    def allocate_new_blocks(
        self, seq_id: int, data: PagedCacheMetadata
    ) -> None:
        # Determine number of new blocks to allocate.
        req_blocks = self.req_to_blocks[seq_id]
        num_required_blocks = ceildiv(data.seq_len, self.block_size)
        num_new_blocks = num_required_blocks - len(req_blocks)
        assert num_new_blocks >= 0

        # Allocate new blocks.
        if num_new_blocks > self.free_block_queue.num_free_blocks:
            raise RuntimeError(
                f"Cannot get {num_new_blocks} free blocks from the free block queue"
            )
        for _ in range(num_new_blocks):
            new_block = self.alloc_block()
            req_blocks.append(new_block)

    @traced
    def alloc_block(self) -> KVCacheBlock:
        """Allocate a block from the free block queue."""

        # First allocate block
        curr_block = self.free_block_queue.popleft()
        assert curr_block.ref_cnt == 0

        # If the block is committed into prefix cache, evict it.
        block_hash = curr_block.block_hash
        assert self.enable_prefix_caching or block_hash is None
        if block_hash is not None:
            if block_hash in self.committed_block_hash_to_block:
                self.committed_block_hash_to_block[block_hash]
                del self.committed_block_hash_to_block[block_hash]
                parent_block_hash = block_hash.parent_hash_value
                del self.parent_to_child_hash[parent_block_hash][
                    block_hash.token_ids
                ]
                if self.kv_cache_agent_queue is not None:
                    logger.debug(
                        f"Updating KV Cache Agent with block {block_hash.hash_value}, memory tier {MemoryTier.MEMORY_TIER_GPU}, update type {UpdateType.UPDATE_TYPE_ADDED}"
                    )
                    self.kv_cache_agent_queue.put(
                        KVCacheChangeMessage(
                            cache_id=str(block_hash.hash_value),
                            memory_tier=MemoryTier.MEMORY_TIER_GPU,
                            update_type=UpdateType.UPDATE_TYPE_REMOVED,
                        )
                    )

            curr_block.block_hash = None

        curr_block.ref_cnt += 1
        assert curr_block.block_hash is None
        return curr_block

    @traced
    def free_block(self, block: KVCacheBlock) -> None:
        """Free a block by decreasing its reference count.

        If the reference count is 0, the block is added to the free block queue.

        Note that a block can be in both the prefix cache and the free block
        queue at the same time.
        """
        block.ref_cnt -= 1
        if block.ref_cnt == 0:
            self.free_block_queue.append(block)

    @traced
    def touch(self, blocks: list[KVCacheBlock]) -> None:
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix.

        Args:
            blocks: A list of blocks to touch.
        """
        for block in blocks:
            # ref_cnt=0 means this block is in the free list (i.e. eviction
            # candidate), so remove it.
            if block.ref_cnt == 0:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1

    @property
    def free_blocks(self) -> set[int]:
        """Get the number of free blocks."""
        return self.free_block_queue.free_blocks

    @property
    def cache_hit_rate(self) -> float:
        """Get the percentage of prompt tokens that were retrieved from the cache."""
        if self.prompt_tokens == 0:
            return 0
        return self.cached_prompt_tokens / self.prompt_tokens

    def release_uncommitted_blocks(
        self, seq_id: int, data: PagedCacheMetadata
    ) -> None:
        """Release the uncommitted blocks for the request."""
        req_blocks = self.req_to_blocks[seq_id]
        num_committed_blocks = data.committed_idx // self.block_size
        assert len(req_blocks) >= num_committed_blocks
        num_uncommitted_blocks = len(req_blocks) - num_committed_blocks
        for _ in range(num_uncommitted_blocks):
            block = req_blocks.pop()
            self.free_block(block)
        data.cached_idx = data.committed_idx

    def get_req_blocks(self, seq_id: int) -> list[int]:
        """Get the block ids for a request."""
        return [block.block_id for block in self.req_to_blocks[seq_id]]

    @traced
    def assert_runtime_invariants(
        self, seq_id: int, data: PagedCacheMetadata
    ) -> None:
        """If runtime checks are enabled, assert that the runtime checks are
        correct.
        """
        if not self.enable_runtime_checks:
            return

        # Check that the total number of blocks is correct.
        free_blocks = len(self.free_block_queue)
        active_block_ids = []
        for blocks in self.req_to_blocks.values():
            for block in blocks:
                active_block_ids.append(block.block_id)
                # Check that all active blocks have a ref_cnt > 0
                assert block.ref_cnt > 0
        assert free_blocks + len(set(active_block_ids)) == self.total_num_blocks

        # Check that all blocks in the prefix cache are committed.
        for block_hash, block in self.committed_block_hash_to_block.items():
            assert block_hash is not None
            assert block.block_hash == block_hash

        # Check that the number of committed blocks for request is correct
        num_committed_blocks = data.committed_idx // self.block_size
        num_committed = 0
        for block in self.req_to_blocks[seq_id]:
            if block.block_hash is None:
                break
            num_committed += 1
        assert num_committed == num_committed_blocks

        # Check that the tokens in the request line up with the contents of the hashes
        req_hashes = self.req_to_block_hashes[seq_id]
        req_blocks = self.req_to_blocks[seq_id]
        for hash_idx, req_hash in enumerate(req_hashes):
            tokens = data.tokens[
                hash_idx * self.block_size : (hash_idx + 1) * self.block_size
            ]
            assert req_hash.token_ids == tuple(tokens)

        # Check that the req block hashes are consistent with req blocks
        for hash_value, block in zip(req_hashes, req_blocks):
            assert block.block_hash is None or block.block_hash == hash_value

        # Check that the req block hashes are consistent with parents
        for hash_idx in range(1, len(req_hashes)):
            # check that hashing parent with token ids of current block
            # yields the same hash as the parent block hash
            curr_block_hash = req_hashes[hash_idx]
            prev_block_hash = req_hashes[hash_idx - 1]
            assert (
                curr_block_hash.parent_hash_value == prev_block_hash.hash_value
            )
            assert curr_block_hash == hash_block_tokens(
                prev_block_hash.hash_value,
                np.array(curr_block_hash.token_ids),
            )
