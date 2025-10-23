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

"""Utilities for PagedAttention KVCache block manager."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from max._core_mojo import block_hasher
from max.interfaces import ImageMetadata
from max.profiler import traced

# Note that we use 'None' as a string here instead of None because
# as of Python 3.12, hash(None) returns a constant predictable value.
# This could possibly make it easier to find and exploit hash
# collisions. 'None' as a string will be hashed differently per process,
# but consistently within the same process. This is the same as the
# behavior of None prior to Python 3.12.
DEFAULT_PARENT_HASH = hash("None")


@traced
def hash_request_tokens(
    token_ids: npt.NDArray[np.integer[Any]],
    block_size: int,
    parent_hash: int | None = None,
    prefix_length: int = -1,
    images: list[ImageMetadata] | None = None,
) -> list[int]:
    """Hash the tokens of a request using the Mojo implementation.

    If images are provided, we will set the first vision_token_id to the value
    of the image hash.

    This method should leave the contents of the array unchanged on return.
    """
    if parent_hash is None:
        parent_hash = DEFAULT_PARENT_HASH

    # If images are provided, temporarily replace the first vision_token_id with the image hash
    token_to_reset: dict[int, int] = {}
    if images:
        if prefix_length == -1:
            raise ValueError(
                "prefix_length must be set when images are provided"
            )
        for img in images:
            idx = img.start_idx - prefix_length
            if 0 <= idx < len(token_ids):
                token_to_reset[idx] = token_ids[idx]
                token_ids[idx] = img.image_hash

    # Call into the super fast Mojo block hasher
    hash_vals = block_hasher(token_ids, block_size, parent_hash)
    assert len(hash_vals) == len(token_ids) // block_size

    # Reset the contents of the array to the original values
    for idx, token in token_to_reset.items():
        token_ids[idx] = token

    return hash_vals


@dataclass
class KVCacheBlock:
    """KV-cache block metadata."""

    # Block ID, ranging from 0 to total_num_blocks - 1.
    bid: int
    # Reference count.
    ref_cnt: int = 0
    # The hash of the block composed of (block hash, tuple of token IDs).
    # It is only available when the block is full.
    block_hash: int | None = None

    # Used to construct a doubly linked list for free blocks.
    # These two attributes should only be manipulated by FreeKVCacheBlockQueue.
    prev_free_block: KVCacheBlock | None = None
    next_free_block: KVCacheBlock | None = None

    def __repr__(self) -> str:
        return f"KVCacheBlock(bid={self.bid}, ref_cnt={self.ref_cnt}, block_hash={self.block_hash})"


class FreeKVCacheBlockQueue:
    """This class organizes a list of KVCacheBlock objects to a doubly linked
    list of free blocks. We implement this class instead of using Python
    builtin deque to support removing a block in the middle of the queue
    in O(1) time. To close the performance gap to the builtin deque which is
    implemented in C++, this class does not allocate any Python objects when
    manipulating the linked list. Instead, this class manipulates the
    prev_free_block and next_free_block attributes of the given blocks.

    The queue is ordered by block ID in the beginning. When a block is allocated
    and then freed, it will be appended back with the eviction order:
    1. The least recent used block is at the front (LRU).
    2. If two blocks have the same last accessed time (allocated by the
       same sequence), the one with more hash tokens (the tail of a block
       chain) is at the front.
    Note that we maintain this order by reversing the block order when free
    blocks of a request. This operation is outside of this class.

    Args:
        blocks: A list of KVCacheBlock objects.
    """

    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        self.num_free_blocks = len(blocks)
        self.free_blocks = set(block.bid for block in blocks)

        # Initialize the doubly linked list of free blocks.
        self.free_list_head: KVCacheBlock | None = blocks[0]
        self.free_list_tail: KVCacheBlock | None = blocks[-1]
        for i in range(self.num_free_blocks):
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < self.num_free_blocks - 1:
                blocks[i].next_free_block = blocks[i + 1]

    def __len__(self) -> int:
        return self.num_free_blocks

    @traced
    def popleft(self) -> KVCacheBlock:
        """Pop the first free block and reduce num_free_blocks by 1.

        Returns:
            The first free block.
        """
        if not self.free_list_head:
            raise ValueError("No free blocks available")

        block = self.free_list_head
        self.remove(block)
        return block

    @traced
    def remove(self, block: KVCacheBlock) -> None:
        """Remove a block in the free list and reduce num_free_blocks by 1.

        Args:
            block: The block to remove.
        """
        if block.prev_free_block is not None:
            # Link the previous block to the next block.
            block.prev_free_block.next_free_block = block.next_free_block
        if block.next_free_block is not None:
            # Link the next block to the previous block.
            block.next_free_block.prev_free_block = block.prev_free_block

        if block == self.free_list_head:
            # Update the head if the block is the head.
            self.free_list_head = block.next_free_block
        if block == self.free_list_tail:
            # Update the tail if the block is the tail.
            self.free_list_tail = block.prev_free_block

        # Remove the block from the linked list.
        block.prev_free_block = block.next_free_block = None
        self.num_free_blocks -= 1
        self.free_blocks.remove(block.bid)

    @traced
    def append(self, block: KVCacheBlock) -> None:
        """Put a block back into the free list and increase
        num_free_blocks by 1.

        Args:
            block: The block to append.
        """
        if self.free_list_tail is not None:
            # Link the last block to the new block.
            self.free_list_tail.next_free_block = block
            block.prev_free_block = self.free_list_tail
            self.free_list_tail = block
        else:
            # The free list is empty.
            assert self.free_list_head is None
            self.free_list_head = self.free_list_tail = block

        block.next_free_block = None
        self.num_free_blocks += 1
        self.free_blocks.add(block.bid)
