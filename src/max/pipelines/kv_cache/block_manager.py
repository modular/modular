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
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np
from max.pipelines.context import InputContext
from max.profiler import traced
from max.serve.kvcache_agent.kvcache_agent_service_v1_pb2 import (  # type: ignore
    MemoryTier,
)
from max.support.math import ceildiv

from .block_pool import BlockPool
from .block_utils import (
    ROOT_BLOCK_HASH,
    BlockCopyOp,
    BlockCopyType,
    BlockHashType,
    KVCacheBlock,
    hash_block_tokens,
    hash_request_tokens,
)


class BlockManager:
    @traced
    def __init__(
        self,
        device_memory_tier: MemoryTier,
        total_num_blocks: int,
        total_num_host_blocks: int,
        block_size: int,
        enable_prefix_caching: bool,
        enable_runtime_checks: bool = False,
    ):
        # The number of tokens in a single page.
        self.block_size = block_size

        # Whether to enable prefix caching.
        self.enable_prefix_caching = enable_prefix_caching

        # A pool of device blocks.
        self.device_block_pool = BlockPool(
            device_memory_tier,
            total_num_blocks,
            enable_prefix_caching,
            enable_runtime_checks,
        )

        # A pool of host blocks.
        self.host_block_pool: BlockPool | None = None
        if total_num_host_blocks > 0:
            self.host_block_pool = BlockPool(
                MemoryTier.MEMORY_TIER_CPU,
                total_num_host_blocks,
                enable_prefix_caching,
                enable_runtime_checks,
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

        # Mapping from request ID to block copy operations.
        # Note that the BlockCopyOp owns an instance of a Host block.
        # We should increment it when creating a BlockCopyOp and decrement it
        # when the block copy operation is deleted.
        self.req_to_block_copy_ops: dict[int, list[BlockCopyOp]] = defaultdict(
            list
        )
        self.d2h_eviction_copy_ops: list[BlockCopyOp] = []

        # Cache hit rate metrics.
        self.prompt_tokens = 0
        self.cached_prompt_tokens = 0

        # Whether to enable runtime checks.
        self.enable_runtime_checks = enable_runtime_checks

    @traced
    def step(self, ctx: InputContext) -> None:
        """Step the block manager by committing blocks into prefix cache."""
        self.assert_runtime_invariants(ctx)

        if not self.enable_prefix_caching:
            return

        # Compute block hashes. These hashes are used by the subsequent methods.
        self.compute_block_hashes_for_request(ctx)

        # Now that we generated new tokens, we can possibly commit additional
        # blocks into prefix cache.
        self.commit_to_prefix_cache(ctx)

        self.assert_runtime_invariants(ctx)

    @traced
    def rollback(self, ctx: InputContext) -> None:
        """Rollback the block manager by discarding all blocks after ctx.start_idx.

        This may delete block hashes and blocks assigned to a request."""

        new_num_hashes = ctx.start_idx // self.block_size
        req_hashes = self.req_to_block_hashes[ctx.cache_seq_id]

        # Delete all hashes after ctx.start_idx
        assert len(req_hashes) >= new_num_hashes
        while len(req_hashes) > new_num_hashes:
            req_hashes.pop()

        new_num_blocks = ceildiv(ctx.start_idx, self.block_size)
        new_num_committed_blocks = ctx.start_idx // self.block_size

        # Evict blocks from prefix cache
        req_blocks = self.req_to_blocks[ctx.cache_seq_id]
        for block in req_blocks[new_num_committed_blocks:]:
            # If the block is committed into prefix cache, uncommit it.
            block_hash = block.block_hash
            assert self.enable_prefix_caching or block_hash is None
            assert block.ref_cnt == 1  # should only be one ref to the block
            if block_hash is not None:
                self.device_block_pool.uncommit_block(block)

        # Unassign blocks from request
        assert len(req_blocks) >= new_num_blocks
        while len(req_blocks) > new_num_blocks:
            self.device_block_pool.free_block(req_blocks.pop())

        ctx.set_token_indices(
            committed_idx=new_num_committed_blocks * self.block_size
        )

    @traced
    def compute_block_hashes_for_request(
        self,
        ctx: InputContext,
    ):
        """Compute the block hashes for the request."""

        seq_id = ctx.cache_seq_id
        block_hashes = self.req_to_block_hashes[seq_id]
        parent_block_hash_value = None
        if len(block_hashes) > 0:
            parent_block_hash_value = block_hashes[-1].value

        unhashed_tokens = ctx.tokens[
            len(block_hashes) * self.block_size : ctx.current_length
        ]
        new_block_hashes = hash_request_tokens(
            self.block_size, unhashed_tokens, parent_block_hash_value
        )
        block_hashes.extend(new_block_hashes)

    @traced
    def reuse_blocks_from_prefix_cache(self, ctx: InputContext) -> None:
        """Reuse blocks from prefix cache.

        Full blocks are directly reused and appended to the request's blocks.
        Partial blocks can be reused via COW. The blocks/tokens to copy to and
        from are returned as a tuple.

        This also updates the cache hit rate metrics.
        """
        self.assert_runtime_invariants(ctx)

        seq_id = ctx.cache_seq_id
        req_blocks = self.req_to_blocks[seq_id]

        # Update cache hit rate metrics.
        orig_prompt_len = ctx.active_length
        self.prompt_tokens += orig_prompt_len - 1

        if not self.enable_prefix_caching:
            return

        # Compute block hashes. These hashes are used by the subsequent methods.
        self.compute_block_hashes_for_request(ctx)

        # Query prefix cache for full blocks.
        prefix_cache_blocks = self.get_full_blocks_from_prefix_cache(ctx)
        orig_start_idx = ctx.start_idx

        if len(prefix_cache_blocks) > 0:
            # Since we got cache hits, clear out existing uncommitted blocks
            self.release_uncommitted_blocks(ctx)

            # Append them to the request's blocks.
            req_blocks.extend(prefix_cache_blocks)
            new_committed_idx = (
                ctx.committed_idx + len(prefix_cache_blocks) * self.block_size
            )
            ctx.set_token_indices(
                committed_idx=new_committed_idx,
                start_idx=new_committed_idx,
            )
            assert ctx.committed_idx == ctx.start_idx

            # Check that the cached_idx has increased.
            assert ctx.start_idx > orig_start_idx
            orig_start_idx = ctx.start_idx

        # Query prefix cache for partial blocks
        partial_block, tokens_matched = (
            self.get_partial_block_from_prefix_cache(ctx)
        )

        if partial_block is not None:
            # Since we got cache hits, clear out existing uncommitted blocks
            self.release_uncommitted_blocks(ctx)

            # Touch and free block to move it to end of the free list.
            self.device_block_pool.touch(partial_block)
            self.device_block_pool.free_block(partial_block)

            # We can only perform COW if we can allocate a new block to copy into
            if self.device_block_pool.free_block_queue:
                # Append them to the request's blocks.
                block_hash = partial_block.block_hash
                assert block_hash is not None

                fresh_block = self.allocate_device_block()
                req_blocks.append(fresh_block)
                ctx.bump_token_indices(
                    start_idx=tokens_matched,
                )

                # Record a COW operation.
                cow_op = BlockCopyOp(
                    block_copy_type=BlockCopyType.D2D_COW,
                    dst=fresh_block,
                    src=partial_block,
                    num_tokens=tokens_matched,
                    block_hash=block_hash,
                )
                self.req_to_block_copy_ops[seq_id].append(cow_op)

                # Check that the cached_idx has increased.
                assert ctx.start_idx > orig_start_idx
                orig_start_idx = ctx.start_idx

        # Update cache hit rate metrics.
        new_prompt_len = ctx.active_length
        self.cached_prompt_tokens += orig_prompt_len - new_prompt_len

    @traced
    def get_full_blocks_from_prefix_cache(
        self,
        ctx: InputContext,
    ) -> list[KVCacheBlock]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.
        """

        assert self.enable_prefix_caching

        seq_id = ctx.cache_seq_id
        req_block_hashes = self.req_to_block_hashes[seq_id]
        num_committed_blocks = ctx.committed_idx // self.block_size
        # we exclude the last inflight token to ensure that there is at least
        # one prompt token to be encoded.
        num_inflight_blocks = (ctx.current_length - 1) // self.block_size
        uncommitted_block_hashes = req_block_hashes[
            num_committed_blocks:num_inflight_blocks
        ]

        blocks = []
        device_prefix_cache = (
            self.device_block_pool.block_hash_to_committed_block
        )
        host_prefix_cache = (
            self.host_block_pool.block_hash_to_committed_block
            if self.host_block_pool is not None
            else None
        )
        for block_hash in uncommitted_block_hashes:
            hash_value = block_hash.value
            if hash_value in device_prefix_cache:
                block = device_prefix_cache[hash_value]
                blocks.append(block)
                self.device_block_pool.touch(block)
            elif (
                host_prefix_cache is not None
                and hash_value in host_prefix_cache
                and len(self.device_block_pool.free_block_queue) > 0
            ):
                assert self.host_block_pool is not None

                host_block = host_prefix_cache[hash_value]
                assert host_block.block_hash is not None
                self.host_block_pool.touch(host_block)

                # Allocate a new device block.
                device_block = self.allocate_device_block()
                blocks.append(device_block)

                # Record a H2D block copy operation.
                h2d_op = BlockCopyOp(
                    block_copy_type=BlockCopyType.H2D_MEMCPY,
                    src=host_block,
                    dst=device_block,
                    num_tokens=self.block_size,
                    block_hash=host_block.block_hash,
                )
                self.req_to_block_copy_ops[seq_id].append(h2d_op)
            else:
                break

        return blocks

    @traced
    def get_partial_block_from_prefix_cache(
        self,
        ctx: InputContext,
    ) -> tuple[KVCacheBlock | None, int]:
        """Get the computed (cached) blocks for the request."""
        assert self.enable_prefix_caching

        if self.block_size == 1:
            return None, 0

        seq_id = ctx.cache_seq_id
        req_block_hashes = self.req_to_block_hashes[seq_id]
        num_committed_blocks = ctx.committed_idx // self.block_size

        parent_hash = ROOT_BLOCK_HASH
        if num_committed_blocks > 0:
            parent_hash = req_block_hashes[num_committed_blocks - 1]
        parent_tokens = ctx.tokens[
            num_committed_blocks * self.block_size : ctx.current_length - 1
        ]
        if len(parent_tokens) == 0:
            return None, 0

        # Find the longest prefix match in the prefix cache.
        children = self.device_block_pool.parent_hash_to_child_token_ids[
            parent_hash.value
        ]

        parent_tokens = parent_tokens[: self.block_size]
        res = children.find_string_with_largest_common_prefix(
            tuple(parent_tokens)
        )
        if res is None:
            return None, 0
        best_child_tokens, best_tokens_matched = res
        assert best_tokens_matched < self.block_size

        # It is not profitable to do COW if this request's partial block has
        # at least as many tokens as the best match in the prefix cache.
        current_tokens_in_partial_block = ctx.start_idx % self.block_size
        if current_tokens_in_partial_block >= best_tokens_matched:
            return None, 0

        child_hash = hash_block_tokens(
            parent_hash.value,
            np.array(best_child_tokens),
        )
        child_block = self.device_block_pool.block_hash_to_committed_block[
            child_hash.value
        ]
        return child_block, best_tokens_matched

    @traced
    def commit_to_prefix_cache(
        self,
        ctx: InputContext,
    ) -> None:
        """Commits all blocks whose hashes are known for prefix caching.

        This increments the committed_idx.

        Args:
            seq_id: The ID of the request.
            data: The data of the request.
        """

        seq_id = ctx.cache_seq_id
        req_blocks = self.req_to_blocks[seq_id]
        req_block_hashes = self.req_to_block_hashes[seq_id]
        num_committed_blocks = ctx.committed_idx // self.block_size

        # Count the number of tokens for which we know the values of and align
        # to the block size.
        num_computed_blocks = ctx.start_idx // self.block_size

        # Commit these blocks into the prefix cache.
        for block_idx in range(num_committed_blocks, num_computed_blocks):
            block = req_blocks[block_idx]

            # Get the block hash.
            block_hash = req_block_hashes[block_idx]

            # Get the parent block hash.
            new_block = self.device_block_pool.get_or_commit_into_prefix_cache(
                block_hash, block
            )
            if new_block is not None:
                req_blocks[block_idx] = new_block

        ctx.set_token_indices(
            committed_idx=num_computed_blocks * self.block_size,
        )

    def release(self, seq_id: int) -> None:
        """Release the blocks for the request."""

        blocks = self.req_to_blocks[seq_id]
        ordered_blocks: Iterable[KVCacheBlock] = blocks
        if self.enable_prefix_caching:
            # Free blocks in reverse order so that the tail blocks are
            # freed first.
            ordered_blocks = reversed(blocks)

        for block in ordered_blocks:
            self.device_block_pool.free_block(block)

        self.req_to_blocks[seq_id] = []
        self.req_to_block_hashes[seq_id] = []

    @traced
    def allocate_new_blocks(
        self, ctx: InputContext, num_steps: int = 1
    ) -> None:
        # Determine number of new blocks to allocate.
        seq_id = ctx.cache_seq_id
        req_blocks = self.req_to_blocks[seq_id]
        seq_len = ctx.current_length + num_steps - 1
        num_required_blocks = ceildiv(seq_len, self.block_size)
        num_new_blocks = num_required_blocks - len(req_blocks)
        num_new_blocks = max(num_new_blocks, 0)

        # Allocate new blocks.
        if num_new_blocks > len(self.device_block_pool.free_block_queue):
            raise RuntimeError(
                f"Cannot get {num_new_blocks} free blocks from the free block queue"
            )
        for _ in range(num_new_blocks):
            new_block = self.allocate_device_block()
            req_blocks.append(new_block)

    @traced
    def maybe_offload_gpu_block_to_host(
        self, gpu_block: KVCacheBlock, old_hash: BlockHashType | None
    ) -> None:
        # Can't swap if there is no host block pool.
        if self.host_block_pool is None:
            return

        # Can't swap if the block was not previously committed.
        if old_hash is None:
            return

        # Can't swap if there are no free host blocks.
        if len(self.host_block_pool.free_block_queue) == 0:
            return

        # Should not swap if another block with the same hash is present.
        if old_hash.value in self.host_block_pool.block_hash_to_committed_block:
            return

        # Allocate a host block
        host_block, _ = self.host_block_pool.alloc_block()

        # Create a D2H block copy operation.
        d2h_op = BlockCopyOp(
            block_copy_type=BlockCopyType.D2H_MEMCPY,
            src=gpu_block,
            dst=host_block,
            num_tokens=self.block_size,
            block_hash=old_hash,
        )
        self.d2h_eviction_copy_ops.append(d2h_op)

        # Commit the host block into the host prefix cache.
        self.host_block_pool.commit_into_prefix_cache(old_hash, host_block)

    @traced
    def allocate_device_block(self) -> KVCacheBlock:
        new_block, block_hash = self.device_block_pool.alloc_block()
        self.maybe_offload_gpu_block_to_host(new_block, block_hash)
        return new_block

    @property
    def cache_hit_rate(self) -> float:
        """Get the percentage of prompt tokens that were retrieved from the cache."""
        if self.prompt_tokens == 0:
            return 0
        return self.cached_prompt_tokens / self.prompt_tokens

    def release_uncommitted_blocks(self, ctx: InputContext) -> None:
        """Release the uncommitted blocks for the request."""
        seq_id = ctx.cache_seq_id
        req_blocks = self.req_to_blocks[seq_id]
        num_committed_blocks = ctx.committed_idx // self.block_size
        assert len(req_blocks) >= num_committed_blocks
        num_uncommitted_blocks = len(req_blocks) - num_committed_blocks
        for _ in range(num_uncommitted_blocks):
            block = req_blocks.pop()
            self.device_block_pool.free_block(block)
        ctx.set_token_indices(
            start_idx=ctx.committed_idx,
        )

    def get_req_blocks(self, seq_id: int) -> list[int]:
        """Get the block ids for a request."""
        return [block.block_id for block in self.req_to_blocks[seq_id]]

    def reset_d2h_eviction_copy_ops(self) -> None:
        """Reset the D2H eviction operations."""
        self.d2h_eviction_copy_ops.clear()

    def reset_req_copy_ops(self, seq_id: int) -> None:
        """Reset the block copy operations for a request."""
        self.req_to_block_copy_ops[seq_id].clear()

    def get_req_copy_ops(self, seq_id: int) -> list[BlockCopyOp]:
        """Get the block copy operations for a request."""
        return self.req_to_block_copy_ops[seq_id]

    @traced
    def assert_runtime_invariants(self, ctx: InputContext) -> None:
        """If runtime checks are enabled, assert that the runtime checks are
        correct.
        """
        if not self.enable_runtime_checks:
            return

        # Get the active block ids
        active_block_ids = []
        for blocks in self.req_to_blocks.values():
            for block in blocks:
                active_block_ids.append(block.block_id)
                # Check that all active blocks have a ref_cnt > 0
                assert block.ref_cnt > 0

        # Check that the block pool is consistent
        self.device_block_pool.assert_runtime_invariants(active_block_ids)

        # Get the request hashes and blocks
        seq_id = ctx.cache_seq_id
        req_hashes = self.req_to_block_hashes[seq_id]
        req_blocks = self.req_to_blocks[seq_id]

        # Check that the number of committed blocks for request is correct
        num_committed_blocks = ctx.committed_idx // self.block_size
        num_committed = 0
        for block in req_blocks:
            if block.block_hash is None:
                break
            num_committed += 1
        assert num_committed == num_committed_blocks

        # Check that the tokens in the request line up with the contents of the hashes
        for hash_idx, req_hash in enumerate(req_hashes):
            tokens = ctx.tokens[
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
            assert curr_block_hash.parent_hash_value == prev_block_hash.value
            assert curr_block_hash == hash_block_tokens(
                prev_block_hash.value,
                np.array(curr_block_hash.token_ids),
            )
