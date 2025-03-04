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

"""Prefix cache to enable reuse of KV projections during context encoding with PagedAttention."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, SymbolicDim, TensorType, ops
from max.profiler import traced

from .paged_cache_metadata import PagedCacheMetadata
from .radix_trie import RadixTrie, TrieNode


@traced
def construct_cow_strided_memcpy_graph(
    block_shape: list[int | str], dtype: DType, devices: list[Device]
) -> Graph:
    """
    Returns a graph for performing COW operations on the KV cache.
    """
    assert len(block_shape) == 6
    ds = [DeviceRef(device.label, device.id) for device in devices]
    batch_dim = SymbolicDim("batch")
    blocks_ty = [BufferType(dtype, shape=block_shape, device=d) for d in ds]
    block_src_idx_ty = TensorType(DType.uint32, shape=[batch_dim])
    block_dst_idx_ty = TensorType(DType.uint32, shape=[batch_dim])
    num_tokens_ty = TensorType(DType.uint32, shape=[batch_dim])
    max_num_tokens_ty = TensorType(DType.uint32, shape=[])

    with Graph(
        "mo.kv_collection_cow_strided_memcpy.paged",
        input_types=[
            block_dst_idx_ty,
            block_src_idx_ty,
            num_tokens_ty,
            max_num_tokens_ty,
            *blocks_ty,
        ],
        output_types=[],
    ) as graph:
        (
            block_dst_idx_tensor,
            block_src_idx_tensor,
            num_tokens_tensor,
            max_num_tokens_tensor,
            *all_blocks,
        ) = graph.inputs
        for blocks in all_blocks:
            ops.inplace_custom(
                "mo.kv_collection_cow_strided_memcpy.paged",
                values=[
                    blocks,
                    block_dst_idx_tensor,
                    block_src_idx_tensor,
                    num_tokens_tensor,
                    max_num_tokens_tensor,
                ],
                out_types=[],
            )
        graph.output()

    return graph


class PrefixCache:
    def __init__(
        self,
        session: InferenceSession,
        page_size: int,
        block_shape: list[int | str],
        dtype: DType,
        devices: list[Device],
        tensors: list[Tensor],
        enable_cow: bool = True,
    ):
        self.page_size = page_size
        self.enable_cow = enable_cow
        self.radix_trie = RadixTrie(page_size=self.page_size)
        self.tensors = tensors

        self.cow_count = 0
        # List of (block_dst, block_src, num_tokens)
        self.cow_enqueued_args: list[tuple[int, int, int]] = []
        if self.enable_cow and self.page_size > 1:
            # Load single op graph for performing memory transfers needed for COW
            self.cow_strided_memcpy_graph = session.load(
                construct_cow_strided_memcpy_graph(
                    block_shape,
                    dtype,
                    devices,
                ),
            )
        self.all_tokens = 0
        self.cache_hit_tokens = 0

        # This is a pointer into the radix trie indicating the prefix of the sequence
        # that has been committed into the radix trie.
        self.active_requests: dict[int, TrieNode] = {}

    def __contains__(self, block: int) -> bool:
        """Check if a block is owned by the prefix cache."""
        return block in self.radix_trie.get_all_blocks()

    def external_claim(self, seq_id: int) -> None:
        """Claim a sequence for use by the prefix cache.

        This initializes the cursor in the trie for the given sequence at the
        root, indicating that no blocks are committed for this sequence yet.
        """
        assert seq_id not in self.active_requests
        self.active_requests[seq_id] = self.radix_trie.root

    def release(self, seq_id: int) -> None:
        """Release a sequence from the prefix cache.

        This decrements the ref count of committed blocks used by the sequence.
        """
        assert seq_id in self.active_requests
        node = self.active_requests[seq_id]
        self.radix_trie.mark_not_in_use_by(node, seq_id)
        del self.active_requests[seq_id]

    @property
    def blocks(self) -> set[int]:
        """Returns all blocks owned by the prefix cache."""
        return self.radix_trie.get_all_blocks()

    @property
    def stale_blocks(self) -> set[int]:
        """Returns all blocks that are evictable/stale.

        Stale blocks are those that are not in use by any sequence (refcount == 0)
        """
        return self.radix_trie.get_evictable_blocks()

    @property
    def cache_hit_rate(self) -> float:
        """Returns the prefix cache hit rate."""
        if self.all_tokens == 0:
            return 0.0
        assert self.cache_hit_tokens <= self.all_tokens
        return self.cache_hit_tokens / self.all_tokens

    def validate_req_state_valid(
        self,
        seq_id: int,
        committed_tokens: np.ndarray,
        committed_blocks: list[int],
    ):
        """Check that the committed tokens and blocks match what was actually
        committed into the radix trie."""
        assert seq_id in self.active_requests
        node = self.active_requests[seq_id]
        # Climb up the trie from the given node, accumulating all the
        # prefix tokens and blocks.
        tokens, blocks = node.get_prefix_tokens_and_blocks()
        assert (tokens == committed_tokens).all()
        assert blocks == committed_blocks

    def evict_blocks(self, blocks_to_evict: Optional[int] = None) -> list[int]:
        """Evict a percentage of all blocks according to a LRU policy on the trie leaves."""
        if blocks_to_evict is None:
            blocks_to_evict = len(self.blocks)
        return self.radix_trie.evict_blocks(desired_num_evicted=blocks_to_evict)

    def _release_partial_block(
        self,
        data: PagedCacheMetadata,
        free_block_fn: Callable[[int], None],
    ) -> None:
        """Release the partially cached and uncommitted block.

        There may be a partially cached block if the seq len was not a multiple
        of page size after the last `step` operation. We may want to release the
        partial block if we can retrieve KV projections for additional tokens
        in the block from the cache:

        e.g:
            - partial_block b0 = ["I", "love", "to", "dance"] (cached = 2 tokens)
            - we have block b1 = ["I", "love", "to", "sing"] (cached = 4 tokens)
              in the prefix cache
            - we can delete b0 and reuse b1 for the first three tokens for COW
        """
        assert data.committed_idx < data.cached_idx
        partial_blocks = data.committable_blocks
        assert len(partial_blocks) == 1
        free_block_fn(partial_blocks[0])
        data.blocks.pop()
        partial_tokens = data.cached_idx - data.committed_idx
        assert 0 < partial_tokens < self.page_size
        data.cached_idx -= partial_tokens
        assert data.committed_idx == data.cached_idx

    def _fetch_query_cache(
        self,
        seq_id: int,
        data: PagedCacheMetadata,
    ) -> tuple[TrieNode, list[int]]:
        """A helper method used by `fetch` which queries the prefix cache for
        full blocks which the request should reuse.

        This method does not modify the state of the prefix cache.

        Returns:
            - node: the new position of the trie node for a given sequence once
                    it adds the prefix blocks to its assigned blocks
            - prefix_blocks: blocks the sequence should reuse from the cache
        """
        node = self.active_requests.get(seq_id, self.radix_trie.root)

        # If there is only one committable token, that means that the prompt
        # is one token. We cannot reduce the prompt length any further since
        # the model expects a prompt of length at least 1.
        committable_tokens = data.committable_tokens[:-1]
        if len(committable_tokens) == 0:
            return node, []

        # Query trie for all but last token.
        node, prefix_blocks = self.radix_trie.match_prefix(
            committable_tokens, node=node
        )

        return node, prefix_blocks

    def fetch(
        self,
        seq_ids_and_data: dict[int, PagedCacheMetadata],
        free_block_fn: Callable[[int], None],
        alloc_block_fn: Callable[[], int],
    ) -> dict[int, list[int]]:
        """Extend the kv cache for given batch of requests with any cached prefixes.

        This will increment the committed_idx and cached_idx if there is a cache
        hit. The prompt will be trimmed in the event that cached_idx is bumped.
        """
        seq_ids_and_prefix_blocks = {}
        for seq_id, data in seq_ids_and_data.items():
            # may enqueue COW ops within prefix cache
            seq_ids_and_prefix_blocks[seq_id] = self._fetch_request(
                seq_id, data, free_block_fn, alloc_block_fn
            )

        # Batch execute all of the COW memcpy enqueued during _fetch_request
        if self.enable_cow:
            self.batch_execute_enqueued_cow()

        return seq_ids_and_prefix_blocks

    @traced
    def _fetch_request(
        self,
        seq_id: int,
        data: PagedCacheMetadata,
        free_block_fn: Callable[[int], None],
        alloc_block_fn: Callable[[], int],
    ) -> list[int]:
        """Extend the kv cache for given request with any cached prefixes."""
        node, prefix_blocks = self._fetch_query_cache(seq_id, data)

        # Update the cache hit rate metrics.
        num_cache_hit_tokens = len(prefix_blocks) * self.page_size
        self.cache_hit_tokens += num_cache_hit_tokens
        self.all_tokens += len(data.committable_tokens) - 1

        # Exit early if there are no cache hits.
        if len(prefix_blocks) == 0:
            if self.enable_cow:
                self._fetch_cow(seq_id, data, free_block_fn, alloc_block_fn)
            return []

        self.active_requests[seq_id] = node

        # Mark the prefix blocks we retrieved from the radix trie cache as
        # in use by this sequence so they don't get evicted prematurely.
        self.radix_trie.mark_in_use_by(node, seq_id)

        # If there is a block with partially cached tokens, we should release it
        # if the cache hit blocks already contain these tokens and more
        if data.committed_idx < data.cached_idx and num_cache_hit_tokens > 0:
            assert data.committed_idx + num_cache_hit_tokens > data.cached_idx
            self._release_partial_block(data, free_block_fn)

        data.blocks.extend(prefix_blocks)
        # Bump the committed_idx since we got cache hits
        data.committed_idx += num_cache_hit_tokens
        data.cached_idx += num_cache_hit_tokens

        if self.enable_cow:
            self._fetch_cow(seq_id, data, free_block_fn, alloc_block_fn)

        return prefix_blocks

    def _fetch_cow_query_cache(
        self,
        node: TrieNode,
        committable_tokens: np.ndarray,
        partial_tokens: int,
    ) -> tuple[Optional[int], int]:
        """A helper method used by `_fetch_cow` which queries the prefix cache
        for a block resident in prefix cache to copy tokens from for COW.

        This method does not modify the state of the prefix cache.

        Returns:
            - partial_match_block: block to copy tokens from, this is None in
                                   the event that performing COW is not beneficial
            - num_cache_hit_tokens: tokens to copy from the block
        """
        assert self.enable_cow
        assert self.page_size > 1
        assert self.cow_strided_memcpy_graph is not None
        assert len(committable_tokens) > 0
        assert 0 <= partial_tokens < self.page_size

        # Match page_size tokens in the radix trie
        committable_tokens = committable_tokens[:-1]
        if len(committable_tokens) == 0:
            return None, 0
        committable_tokens_cropped = list(committable_tokens[: self.page_size])
        res = node.find_block_with_largest_common_prefix(
            committable_tokens_cropped
        )
        if res is None:
            return None, 0
        partial_match_block, num_cache_hit_tokens = res
        assert 0 < num_cache_hit_tokens < self.page_size

        # No point in performing COW if we have more cached but uncommitted tokens
        # in the existing partial block than the matched prefix length.
        if num_cache_hit_tokens <= partial_tokens:
            return None, 0
        return partial_match_block, num_cache_hit_tokens

    @property
    def cow_blocks_copied(self) -> int:
        """Get the number of cow operations performed."""
        return self.cow_count

    def reset_cow_blocks_copied(self) -> None:
        """Reset the number of cow operations performed."""
        self.cow_count = 0

    def _fetch_cow(
        self,
        seq_id: int,
        data: PagedCacheMetadata,
        free_block_fn: Callable[[int], None],
        alloc_block_fn: Callable[[], int],
    ) -> None:
        """Extend the kv cache for given request with any cached prefixes by
        copying a portion of the tokens in a committed block to a fresh block.

        If COW is needed by a request, the blocks to copy to/from will be enqueued
        into a list. We should call `batch_execute_enqueued_cow` to execute all
        of the COW memcpy operations at once.

        This will keep the committed_idx the same, but increment the cached_idx
        by between [1, page_size) tokens if we do perform a cow operation. The
        prompt will be trimmed in the event that cached_idx is bumped.
        """
        assert self.enable_cow

        # If page_size is 1, there is no need to perform COW
        if self.page_size == 1:
            return
        assert self.cow_strided_memcpy_graph is not None

        committable_tokens = data.committable_tokens
        node = self.active_requests[seq_id]
        partial_tokens = data.cached_idx - data.committed_idx
        partial_match_block, num_cache_hit_tokens = self._fetch_cow_query_cache(
            node, committable_tokens, partial_tokens
        )
        if partial_match_block is None:
            return
        assert num_cache_hit_tokens > 0
        assert num_cache_hit_tokens > partial_tokens

        # If we have a partially cached block, we need to release it before
        # appending additional blocks.
        if partial_tokens > 0:
            assert data.committed_idx + num_cache_hit_tokens > data.cached_idx
            self._release_partial_block(data, free_block_fn)

        # Copy prefix_len tokens from partial_match_block to new_block.
        new_block = alloc_block_fn()
        self.cow_count += 1
        # Enqueue the COW memcpy args for later execution
        self.cow_enqueued_args.append(
            (new_block, partial_match_block, num_cache_hit_tokens)
        )
        data.blocks.append(new_block)
        data.cached_idx += num_cache_hit_tokens
        assert len(data.prompt_tokens) > 0
        assert data.cached_idx < data.inflight_idx

    @traced
    def batch_execute_enqueued_cow(self):
        """Execute all of the COW memcpy operations enqueued during `fetch`.

        This launches 1 kernel even if we need N strided memcpys.
        """

        if not (self.enable_cow and self.page_size > 1):
            return
        assert self.cow_strided_memcpy_graph is not None

        if len(self.cow_enqueued_args) == 0:
            return

        # Convert the list of (block_dst, block_src, num_tokens) to tensors
        args = np.array(self.cow_enqueued_args, dtype=np.uint32)
        # copy is needed to make the tensors contiguous
        block_dst_idx_tensor = np.ascontiguousarray(args[:, 0])
        block_src_idx_tensor = np.ascontiguousarray(args[:, 1])
        num_tokens_tensor = np.ascontiguousarray(args[:, 2])
        max_num_tokens_scalar = np.max(num_tokens_tensor)

        # Execute the COW operation
        self.cow_strided_memcpy_graph.execute(
            block_dst_idx_tensor,
            block_src_idx_tensor,
            num_tokens_tensor,
            max_num_tokens_scalar,
            *self.tensors,
        )
        self.cow_enqueued_args = []

    def query_fetch_stats(
        self, seq_id: int, data: PagedCacheMetadata
    ) -> tuple[set[int], int]:
        """Query the prefix trie for the cache hit stats for the given sequence.

        This method does not modify the state of the prefix cache.

        Returns:
            - prefix_blocks: Prefix cache blocks that would be reused for this seq.
            - num_cache_hit_tokens: Number of new cached tokens retrieved from prefix cache.
        """
        if len(data.prompt_tokens) == 1:
            return set(), 0

        node, prefix_blocks = self._fetch_query_cache(seq_id, data)
        num_cache_hit_tokens = len(prefix_blocks) * self.page_size

        # If COW is enabled, we should consider the number of additional cache
        # hits that would be incurred by performing a COW operation.
        if self.enable_cow and self.page_size > 1:
            committable_tokens = data.committable_tokens
            committable_tokens = committable_tokens[num_cache_hit_tokens:]
            partial_tokens = data.cached_idx - data.committed_idx
            if len(prefix_blocks) > 0:
                partial_tokens = 0
            partial_match_block, num_cow_cache_hit_tokens = (
                self._fetch_cow_query_cache(
                    node, committable_tokens, partial_tokens
                )
            )
            if partial_match_block is not None:
                num_cache_hit_tokens += num_cow_cache_hit_tokens

        # Determine the number of cache hit tokens, excluding any tokens that
        # were already cached in a partial block.
        new_cached_idx = max(
            data.committed_idx + num_cache_hit_tokens, data.cached_idx
        )
        num_cache_hit_tokens = new_cached_idx - data.cached_idx
        assert num_cache_hit_tokens >= 0
        return set(prefix_blocks), num_cache_hit_tokens

    @traced
    def step(
        self,
        seq_id: int,
        data: PagedCacheMetadata,
        free_block_fn: Callable[[int], None],
    ) -> None:
        """Now that we have written to the inflight blocks, we will try to commit
        them to the radix trie.

        This increments the committed_idx. We guarantee that the number of committed
        tokens will be a multiple of the page size. There may be some uncommitted
        tokens left over due to there being a partial page at the end. Thus the
        number of uncommitted tokens will always be less than the page size.
        """
        committable_tokens = data.committable_tokens_aligned
        node = self.active_requests[seq_id]
        node, existing_blocks = self.radix_trie.match_prefix(
            committable_tokens, node=node
        )
        self.active_requests[seq_id] = node

        # If we computed a kv entry for a token that was already cached,
        # we will just release that block we just computed.
        for b0, b1 in zip(existing_blocks, data.committable_blocks_aligned):
            if b0 != b1:
                free_block_fn(b1)

        # Replace the inflight blocks with the existing prefix blocks.
        committed_block_idx = data.committed_idx // self.page_size
        data.blocks[
            committed_block_idx : committed_block_idx + len(existing_blocks)
        ] = existing_blocks
        data.committed_idx += len(existing_blocks) * self.page_size

        committable_tokens = data.committable_tokens_aligned
        committable_blocks = data.committable_blocks_aligned
        assert len(committable_tokens) % self.page_size == 0
        assert (
            len(committable_tokens) == len(committable_blocks) * self.page_size
        )

        # If there are any tokens to commit, insert them into the prefix cache.
        node = self.radix_trie.insert(
            committable_tokens,
            committable_blocks,
            node=node,
        )
        self.active_requests[seq_id] = node
        data.committed_idx += len(committable_tokens)

        # Mark the recently committed blocks as in use by this sequence
        # so they don't get evicted prematurely.
        self.radix_trie.mark_in_use_by(node, seq_id)
