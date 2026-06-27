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

"""Utilities for PagedAttention KVCache block manager."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal, overload

import numpy as np
import numpy.typing as npt
from max._core_mojo import block_hasher, block_hasher_sha256
from max.nn.kv_cache.cache_params import KVHashAlgo
from max.pipelines.context import TokenHashOverride
from max.profiler import traced

__all__ = ["KVHashAlgo"]


class InsufficientBlocksError(Exception):
    """Exception raised when there are insufficient free blocks to satisfy an allocation."""


DEFAULT_PARENT_HASH = 0
_ZERO_SEED: bytes = b"\x00" * 32
"""The zero seed for the SHA-256 algorithm.
Deterministic behaviour across restarts for benchmarking.
"""


def _make_root_parent_hash(seed: bytes | None, salt: str | None) -> bytes:
    """Combine cluster-level `seed` and per-request `salt` into a 32-byte root parent hash for the SHA-256 chain.

    `effective = (seed or _ZERO_SEED) XOR sha256(salt or b"")`

    Both factors are 32 bytes; the XOR is byte-wise. When neither is
    supplied, returns 32 zero bytes (preserves cross-restart cache reuse
    for benchmark workloads).
    """
    base = seed if seed is not None else _ZERO_SEED
    if len(base) != 32:
        raise ValueError(f"seed must be exactly 32 bytes, got {len(base)}")
    if salt is None:
        return bytes(base)
    # Hash the salt to get a 32-byte digest
    salt_digest = hashlib.sha256(salt.encode("utf-8")).digest()

    # XOR the base with the salt digest
    return bytes(b ^ s for b, s in zip(base, salt_digest, strict=False))


def _truncate_to_signed64(digest: bytes) -> int:
    """Reduce a 32-byte SHA-256 digest to a signed 64-bit Python int.

    Takes the first 8 bytes interpreted big-endian. Converts to a signed
    int (high bit becomes negative), matching the existing
    `mojo_block_hasher` behaviour.
    """
    n = int.from_bytes(digest[:8], "big", signed=False)
    if n >= 1 << 63:
        n -= 1 << 64
    return n


# ahash64 overload returning ints
@overload
def hash_request_tokens(
    token_ids: npt.NDArray[np.integer[Any]],
    block_size: int,
    parent_hash: int | None = ...,
    prefix_length: int = ...,
    token_hash_overrides: list[TokenHashOverride] | None = ...,
    *,
    algo: Literal["ahash64"] = ...,
    seed: bytes | None = ...,
    salt: str | None = ...,
) -> list[int]: ...


# sha256_64 overload returning ints
@overload
def hash_request_tokens(
    token_ids: npt.NDArray[np.integer[Any]],
    block_size: int,
    parent_hash: int | bytes | None = ...,
    prefix_length: int = ...,
    token_hash_overrides: list[TokenHashOverride] | None = ...,
    *,
    algo: Literal["sha256_64"],
    seed: bytes | None = ...,
    salt: str | None = ...,
) -> list[int]: ...


# sha256 overload returning bytes
@overload
def hash_request_tokens(
    token_ids: npt.NDArray[np.integer[Any]],
    block_size: int,
    parent_hash: int | bytes | None = ...,
    prefix_length: int = ...,
    token_hash_overrides: list[TokenHashOverride] | None = ...,
    *,
    algo: Literal["sha256"],
    seed: bytes | None = ...,
    salt: str | None = ...,
) -> list[bytes]: ...


@overload
def hash_request_tokens(
    token_ids: npt.NDArray[np.integer[Any]],
    block_size: int,
    parent_hash: int | bytes | None = ...,
    prefix_length: int = ...,
    token_hash_overrides: list[TokenHashOverride] | None = ...,
    *,
    algo: KVHashAlgo,
    seed: bytes | None = ...,
    salt: str | None = ...,
) -> list[int] | list[bytes]: ...


@traced
def hash_request_tokens(
    token_ids: npt.NDArray[np.integer[Any]],
    block_size: int,
    parent_hash: int | bytes | None = None,
    prefix_length: int = -1,
    token_hash_overrides: list[TokenHashOverride] | None = None,
    *,
    algo: KVHashAlgo = "ahash64",
    seed: bytes | None = None,
    salt: str | None = None,
) -> list[int] | list[bytes] | None:
    """Hash the tokens of a request using the Mojo implementation.

    Token hash overrides let callers replace one placeholder token per media
    item with a content hash while computing prefix-cache keys.

    This method should leave the contents of the array unchanged on return.
    """
    if algo == "ahash64" and (seed is not None or salt is not None):
        raise ValueError(
            "seed/salt are only valid with algo=sha256 or "
            "algo=sha256_64; pass algo to enable"
        )

    overrides_in_slice: dict[int, int] = {}
    if token_hash_overrides:
        if prefix_length == -1:
            raise ValueError(
                "prefix_length must be set when token hash overrides are provided"
            )
        for override in token_hash_overrides:
            idx = override.token_idx - prefix_length
            if 0 <= idx < len(token_ids):
                if idx in overrides_in_slice:
                    raise ValueError(
                        "Multiple token hash overrides target the same token index."
                    )
                overrides_in_slice[idx] = override.token_hash

    # Temporarily replace the selected placeholder tokens with content hashes.
    # All validation above happens before mutation so errors cannot leak
    # modified tokens.
    token_to_reset: dict[int, int] = {}
    try:
        for idx, token_hash in overrides_in_slice.items():
            token_to_reset[idx] = token_ids[idx]
            token_ids[idx] = token_hash

        hash_vals: list[int] | list[bytes]
        if algo == "ahash64":
            ph_int = DEFAULT_PARENT_HASH if parent_hash is None else parent_hash
            assert isinstance(ph_int, int), (
                f"ahash64 algo requires int parent_hash, got{type(parent_hash)}"
            )
            hash_vals = block_hasher(token_ids, block_size, ph_int)

        elif algo in ("sha256", "sha256_64"):
            if parent_hash is None:
                ph_bytes = _make_root_parent_hash(seed, salt)
            elif isinstance(parent_hash, bytes):
                if len(parent_hash) != 32:
                    raise ValueError(
                        f"algo={algo} requires 32-byte parent_hash, got"
                        f"{len(parent_hash)}"
                    )
                ph_bytes = parent_hash
            else:
                raise TypeError(
                    "algo=sha256/sha256_64 requires bytes parent_hash, got"
                    f"{type(parent_hash).__name__}"
                )

            full_digests: list[bytes] = block_hasher_sha256(
                token_ids, block_size, ph_bytes
            )
            if algo == "sha256":
                hash_vals = full_digests
            else:
                hash_vals = [_truncate_to_signed64(d) for d in full_digests]
        else:
            raise ValueError(f"unknown algo={algo}")

        assert len(hash_vals) == len(token_ids) // block_size
        return hash_vals
    finally:
        # Restore any mutated media tokens, even on error.
        for idx, token in token_to_reset.items():
            token_ids[idx] = token


@dataclass
class KVCacheBlock:
    """KV-cache block metadata."""

    # Block ID, ranging from 0 to total_num_blocks - 1.
    bid: int
    # Reference count.
    ref_cnt: int = 0
    # The hash of the block composed of (block hash, tuple of token IDs).
    # It is only available when the block is full.
    block_hash: int | bytes | None = None
    # Whether the block is the null block.
    is_null: bool = False

    # Used to construct a doubly linked list for free blocks.
    # These two attributes should only be manipulated by FreeKVCacheBlockQueue.
    prev_free_block: KVCacheBlock | None = None
    next_free_block: KVCacheBlock | None = None

    def __repr__(self) -> str:
        return f"KVCacheBlock(bid={self.bid}, ref_cnt={self.ref_cnt}, block_hash={self.block_hash!r})"


class FreeKVCacheBlockQueue:
    """Organizes KVCacheBlock objects as a doubly linked list of free blocks.

    We implement this class instead of using Python
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
        """Removes a block from the free list and reduces num_free_blocks by 1.

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
        """Puts a block back into the free list and increases num_free_blocks by 1.

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
