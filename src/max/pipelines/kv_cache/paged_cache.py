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

"""PagedAttention-enabled KV cache for the Transformer leveraging the mo.opaque pattern."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any, Dict, List, Optional, cast

import numpy as np
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    DeviceRef,
    TensorType,
    TensorValue,
    _OpaqueType,
    _OpaqueValue,
    ops,
)

from ._utils import build_max_lengths_tensor
from .cache_params import KVCacheParams
from .manager import (
    KVCacheInputs,
    KVCacheInputSymbols,
    KVCacheManager,
    RaggedKVCacheInputs,
)
from .paged_cache_metadata import PagedCacheMetadata, ceildiv
from .prefix_cache import PrefixCache

logger = logging.getLogger("max.pipelines")

PERCENTAGE_BLOCKS_TO_EVICT = 0.05


@dataclass
class PagedCacheInputSymbols(KVCacheInputSymbols):
    kv_blocks: TensorType
    cache_lengths: TensorType
    lookup_table: TensorType
    max_lengths: TensorType


class PagedKVCacheType(_OpaqueType):
    """PagedAttention Mojo KV Cache graph type."""

    def __init__(self) -> None:
        """Creates an opaque type containing a paged KV Cache."""
        super().__init__("PagedKVCache")


class PagedKVCacheCollectionType(_OpaqueType):
    """The graph type for a "view" of the cache for the given sequences in the
    batch.

    This object does not own the underlying buffers in k_cache and v_cache,
    it's borrowing them from the BlockWrappers in our ContinuousKVCacheManager.
    It does own the Pointer[NDBuffer[type, 3]] and valid_lengths buffer
    """

    def __init__(self) -> None:
        """Creates an opaque type containing a paged KV cache collection."""
        super().__init__("PagedKVCacheCollection")


class PagedKVCache(_OpaqueValue):
    """PagedAttention Mojo KV cache graph value."""


class PagedKVCacheCollection(_OpaqueValue):
    """The graph value for a view of the KV cache."""


class FetchPagedKVCacheCollection:
    def __init__(self, kv_params: KVCacheParams) -> None:
        self.kv_params = kv_params

    def __call__(
        self,
        blocks: TensorValue,  # NDBuffer[type, 6, Self.blocks_shape]
        cache_lengths: TensorValue,  # NDBuffer[DType.uint32, 1],
        lookup_table: TensorValue,  # NDBuffer[DType.uint32, 2],
        is_cache_empty: TensorValue,
    ) -> PagedKVCacheCollection:
        """Constructs a PagedKVCacheCollection for use downstream."""

        # Explicit validation.
        if blocks.dtype != self.kv_params.dtype:
            msg = (
                f"expected blocks to be dtype: {self.kv_params.dtype}, got"
                f" {blocks.dtype}"
            )
            raise ValueError(msg)

        if blocks.rank != 6:
            msg = f"expected blocks to be of rank 6, got {blocks.rank}"
            raise ValueError(msg)

        # For all tensors other than the blocks tensor, the length should be equivalent
        # to batch size, which is unknown within the graph at this stage.
        if cache_lengths.dtype != DType.uint32:
            msg = f"expected cache lengths to be dtype: uint32, got {cache_lengths.dtype}"
            raise ValueError(msg)

        if cache_lengths.rank != 1:
            msg = f"expected cache lengths to be of rank 1, got {cache_lengths.rank}"
            raise ValueError(msg)

        if lookup_table.dtype != DType.uint32:
            msg = f"expected lookup_table to be dtype: uint32, got {lookup_table.dtype}"
            raise ValueError(msg)

        if lookup_table.rank != 2:
            msg = f"expected lookup_table to be of rank 2, got {lookup_table.rank}"
            raise ValueError(msg)

        return PagedKVCacheCollection(
            ops.custom(
                "mo.kv_collection_ctor.paged",
                values=[blocks, cache_lengths, lookup_table, is_cache_empty],
                out_types=[PagedKVCacheCollectionType()],
                parameters={
                    "num_heads": self.kv_params.n_kv_heads_per_device,
                    "head_dim": self.kv_params.head_dim,
                    "page_size": int(blocks.shape[3]),
                },
            )[0].opaque
        )


class PagedKVCacheManager(KVCacheManager):
    def __init__(
        self,
        params: KVCacheParams,
        max_batch_size: int,
        max_seq_len: int,
        num_layers: int,
        devices: list[Device],
        session: InferenceSession,
        cache_memory: int,
        page_size: int = 128,
        enable_runtime_checks: bool = False,
    ):
        """
        Args:
            params: The KVCacheParams for the given pipeline.
            max_batch_size: The maximum number of active
                requests that the manager should support.
            max_seq_len: The maximum sequence length we will generate.
            num_layers: The number of layers in the model.
            devices: The devices on which the manager will allocate memory.
            session: The inference session to load ops from.
            cache_memory: The total amount of memory available for caching.
                This is aggregated across all devices.
            page_size: The number of tokens that will be stored in a single page.
            enable_runtime_checks: Whether to enable runtime correctness checks.
        """
        # The number of tokens in a single page.
        self.page_size = page_size

        # The number of bytes that a single page will occupy.
        single_page_size_bytes = (
            2
            * num_layers
            * params.n_kv_heads_per_device
            * params.head_dim
            * page_size
            * params.dtype.size_in_bytes
        )

        # Normalize cache_memory across all devices.
        cache_memory_per_device = cache_memory // len(devices)

        # The total number of pages we'll have per-device.
        self.total_num_pages = int(
            cache_memory_per_device // single_page_size_bytes
        )

        if self.total_num_pages == 0:
            raise RuntimeError(
                f"Insufficient cache memory to allocate even a single page.\n"
                f"One page requires {single_page_size_bytes} bytes but only {cache_memory_per_device} bytes are available."
            )

        if max_batch_size > self.total_num_pages:
            logger.warning(
                f"Insufficient cache memory to support a batch containing {max_batch_size} requests with one token per request. "
                f"Need to allocate at least {max_batch_size} blocks, but only have enough memory for {self.total_num_pages} blocks. "
                f"One page requires {single_page_size_bytes} bytes but only {cache_memory_per_device} bytes are available."
            )

        blocks_needed_for_max_seq_len = ceildiv(max_seq_len, page_size)
        if blocks_needed_for_max_seq_len > self.total_num_pages:
            logger.warning(
                f"Insufficient cache memory to support a batch containing one request at the max sequence length of {max_seq_len} tokens. "
                f"Need to allocate at least {blocks_needed_for_max_seq_len} blocks, but only have enough memory for {self.total_num_pages} blocks. "
                f"One page requires {single_page_size_bytes} bytes but only {cache_memory_per_device} bytes are available."
            )

        # call our base class constructor
        super().__init__(
            params=params,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            devices=devices,
            session=session,
            is_ragged=True,
        )

        # Initialize the set of available blocks.
        self.available_blocks = set(range(self.total_num_pages))

        # Initialize the blocks for each device.
        self.blocks: list[Tensor] = []
        for device in self.devices:
            self.blocks.append(
                Tensor.zeros(
                    self.block_shape(),  # type: ignore
                    self.params.dtype,
                    device=device,
                )
            )

        self.active_requests: Dict[int, PagedCacheMetadata] = {}

        self.prefix_cache: Optional[PrefixCache] = None
        if params.enable_prefix_caching:
            self.prefix_cache = PrefixCache(
                session=session,
                page_size=self.page_size,
                block_shape=self.block_shape(is_parameterized=True),
                dtype=self.params.dtype,
                devices=devices,
                tensors=self.blocks,
            )

        # Whether to enable runtime correctness checks. These correctness checks
        # are expensive and should only be used in tests.
        self.enable_runtime_checks = enable_runtime_checks

    def _runtime_check(self) -> None:
        if not self.enable_runtime_checks:
            return
        assert self._count_all_pages() == self.total_num_pages
        if self.prefix_cache is None:
            return
        for seq_id, data in self.active_requests.items():
            self.prefix_cache.validate_req_state_valid(
                seq_id,
                data.committed_tokens,
                data.committed_blocks,
            )

    @property
    def cache_hit_rate(self) -> float:
        if self.prefix_cache is None:
            return 0.0
        return self.prefix_cache.cache_hit_rate

    def alloc_block(self) -> int:
        if len(self.available_blocks) == 0 and self.prefix_cache is not None:
            blocks_to_evict = self.total_num_pages * PERCENTAGE_BLOCKS_TO_EVICT
            blocks_to_evict = int(max(1, blocks_to_evict))
            evicted = self.prefix_cache.evict_blocks(blocks_to_evict)
            for block in evicted:
                self.available_blocks.add(block)

        if len(self.available_blocks) == 0:
            raise RuntimeError(
                f"All {self.total_num_pages} KVCache pages have been exhausted! "
                "You must restart your process and set a smaller batch size or max seq len."
            )

        block = self.available_blocks.pop()
        return block

    def release_block(self, block: int) -> None:
        """We can release a block if prefix caching is disabled or if it is not committed.

        If it is committed, it may be in the radix tree and in use by other sequences.
        This means it can't be safely released without further checks.
        """
        if self.prefix_cache is None:
            self.available_blocks.add(block)
            return

        # We can only add the block to the available set if it is not committed
        # to the prefix cache.
        if block not in self.prefix_cache:
            self.available_blocks.add(block)

    @classmethod
    def _block_size_per_token(
        cls, params: KVCacheParams, num_layers: int
    ) -> int:
        return (
            reduce(mul, cls._block_shape(params, 1, 1, num_layers), 1)
            * params.dtype.size_in_bytes
        )

    @classmethod
    def estimated_memory_size(
        cls,
        params: KVCacheParams,
        max_batch_size: int,
        max_seq_len: int,
        num_layers: int,
        available_cache_memory: int,
        devices: list[Device],
        **kwargs: Any,
    ) -> int:
        # Determine how much size is necessary to store the full cache based on max_batch_size and max_seq_len.
        # If that's less than available_cache_memory, return that.
        # Otherwise, return available_cache_memory.
        # This is to prevent over-allocation on devices with a large amount of free memory (e.g. CPUs).
        assert params.page_size is not None
        block_size_per_token = cls._block_size_per_token(
            params, num_layers
        ) * len(devices)

        # round up our max_seq_len to the nearest page_size
        max_seq_len_round_up = (
            math.ceil(max_seq_len / params.page_size) * params.page_size
        )
        size_to_support_full_cache = (
            block_size_per_token * max_batch_size * max_seq_len_round_up
        )

        # return the minimum of the two
        return min(available_cache_memory, size_to_support_full_cache)

    @classmethod
    def infer_optimal_batch_size(
        cls,
        params: KVCacheParams,
        max_seq_len: int,
        num_layers: int,
        available_cache_memory: int,
        devices: list[Device],
        **kwargs: Any,
    ) -> int:
        # We just hard-code a default of 512 for paged attention.
        # The worst case scenario if this is too high is that we'll evict
        # requests at an elevated rate. We print warnings in that case so users
        # are aware of what needs to be tweaked/changed.
        return 512

    def block_shape(
        self,
        is_parameterized: bool = False,
    ) -> list[int | str]:
        return self._block_shape(
            self.params,
            self.total_num_pages,
            self.page_size,
            self.num_layers,
            is_parameterized,
        )

    @classmethod
    def _block_shape(
        cls,
        params: KVCacheParams,
        total_num_pages: int,
        page_size: int,
        num_layers: int,
        is_parameterized: bool = False,
    ) -> list[int | str]:
        # split k and v caches across a single dim
        # 0 = key
        # 1 = value
        kv_dim = 2
        return [
            num_layers,
            kv_dim,
            "total_num_pages" if is_parameterized else total_num_pages,
            page_size,
            params.n_kv_heads_per_device,
            params.head_dim,
        ]

    def get_num_free_blocks(self) -> int:
        if self.prefix_cache is None:
            return len(self.available_blocks)
        return len(self.available_blocks) + len(self.prefix_cache.stale_blocks)

    def get_num_used_blocks(self) -> int:
        return self.total_num_pages - self.get_num_free_blocks()

    def can_fetch(
        self, seq_ids_and_prompts: dict[int, np.ndarray], num_steps: int = 1
    ) -> bool:
        """Checks if there are sufficient KV pages to run `fetch` on given batch.

        It is OK if some seq_id are not in the cache. We assume the cache lengths
        are zero in those cases.
        """

        total_blocks_to_allocate = 0
        all_cache_hit_blocks: set[int] = set()

        for seq_id, prompt in seq_ids_and_prompts.items():
            data = self.active_requests.get(
                seq_id, PagedCacheMetadata(self.page_size, self.max_seq_len)
            )

            # Extend the kv cache for given request with any cached prefixes.
            cached_blocks: list[int] = []
            if self.prefix_cache is not None:
                cached_blocks = self.prefix_cache.get_cached_blocks(
                    seq_id, prompt
                )

            # Compute the total sequence length and the number of pages required to store it.
            total_sequence_length = (
                data.cached_idx + len(prompt) + num_steps - 1
            )
            num_pages_required = ceildiv(total_sequence_length, self.page_size)

            # Compute the number of *new* pages we need to allocate.
            blocks_to_allocate = (
                num_pages_required - len(data.blocks) - len(cached_blocks)
            )

            total_blocks_to_allocate += blocks_to_allocate
            all_cache_hit_blocks.update(cached_blocks)

        num_evictable_blocks = 0
        if self.prefix_cache is not None:
            # the blocks in the prefix cache that will be used by sequences in
            # this batch are no longer eligible for eviction / allocation.
            num_evictable_blocks = len(
                self.prefix_cache.stale_blocks - all_cache_hit_blocks
            )

        num_free_blocks = len(self.available_blocks) + num_evictable_blocks

        return total_blocks_to_allocate <= num_free_blocks

    def get_num_cached_tokens(self, prompt: np.ndarray) -> int:
        """Returns the number of tokens in the CE prompt that are found in the
        prefix cache.
        """
        if self.prefix_cache is None:
            return 0
        return self.prefix_cache.get_num_cached_tokens(prompt)

    def _fetch(
        self, seq_ids_and_prompts: dict[int, np.ndarray], num_steps: int = 1
    ) -> List[KVCacheInputs]:
        """This method identifies available blocks to service the given requests and marks them as inflight.
        They're assigned to the request as "in-flight" until step is called.

        Generally the prompt length is n for prefill, and 1 for decode step. Additionally, there is not a
        kv entry associated with each token in the prompt.

        When prefix caching is enabled, and KV entries can be retrieved for some tokens in the prompt, the
        input `seq_ids_and_prompts` will be modified. Each prompt will be shortened to only include the tokens
        for which we do not have a cached KV entry. Note that we will never return a empty prompt.
        """
        self._runtime_check()

        max_seq_len_in_batch = -1
        # before we start making any changes, validate that we won't over-write the cache
        for batch_idx, (seq_id, prompt) in enumerate(
            seq_ids_and_prompts.items()
        ):
            # Validate there aren't other inflight requests for this sequence.
            assert seq_id not in self.fetch_metadata

            # Add prompt and inflight tokens to the token array
            data = self.active_requests[seq_id]
            data.fetch(prompt, num_steps)

            # Compute the total sequence length
            if data.seq_len > max_seq_len_in_batch:
                max_seq_len_in_batch = data.seq_len

            assert data.seq_len <= self.max_seq_len, (
                f"seq_id: {seq_id} would overrun the max cache length of {self.max_seq_len} "
                f"with {len(prompt)} new tokens. Existing length: {data.cached_idx}"
            )

        max_num_pages = ceildiv(max_seq_len_in_batch, self.page_size)

        # Allocate the buffers containing metadata about the batch.
        # [0, total_num_pages) are the valid block ids and total_num_pages
        # denotes an unassigned block.
        batch_size = len(seq_ids_and_prompts)
        lut_table_np = np.full(
            (batch_size, max_num_pages), self.total_num_pages, dtype=np.uint32
        )
        cache_lengths_np = np.zeros((batch_size,), dtype=np.uint32)

        # Iterate over requests and query prefix cache
        all_cache_hit_blocks: set[int] = set()
        for batch_idx, (seq_id, prompt) in enumerate(
            seq_ids_and_prompts.items()
        ):
            # Ensure we've called claim for this sequence id.
            if seq_id not in self.active_requests:
                raise ValueError(f"seq_id: {seq_id} not in active requests.")

            if self.prefix_cache is not None:
                data = self.active_requests[seq_id]
                # bump the committed_idx, and possibly the cached_idx
                prefix_blocks = self.prefix_cache.fetch(
                    seq_id,
                    data,
                    free_block_fn=self.release_block,
                    alloc_block_fn=self.alloc_block,
                )
                all_cache_hit_blocks.update(prefix_blocks)
                # Possibly trim the input prompt.
                seq_ids_and_prompts[seq_id] = data.prompt_tokens

        # Determine the number of pages required for each sequence.
        max_seq_length = 0
        max_cache_length = 0
        total_sequence_length = 0
        total_blocks_to_allocate = 0
        blocks_to_allocate_by_seq = {}
        for batch_idx, (seq_id, prompt) in enumerate(
            seq_ids_and_prompts.items()
        ):
            data = self.active_requests[seq_id]

            # Get the existing cache length for this sequence.
            cache_length = data.cached_idx
            cache_lengths_np[batch_idx] = cache_length

            # Update the maximum lengths seen so far.
            max_seq_length = max(max_seq_length, len(prompt))
            max_cache_length = max(max_cache_length, cache_length)

            # Compute the total sequence length and the number of pages required to store it.
            total_sequence_length += data.seq_len
            num_pages_required = ceildiv(data.seq_len, self.page_size)

            # Compute the number of *new* pages we need to allocate.
            num_new_pages = num_pages_required - len(data.blocks)
            assert num_new_pages >= 0
            blocks_to_allocate_by_seq[seq_id] = num_new_pages
            total_blocks_to_allocate += num_new_pages

        # Check if we have enough free blocks to service all requests.
        num_evictable_blocks = 0
        if self.prefix_cache is not None:
            # the blocks in the prefix cache that will be used by sequences in
            # this batch are no longer eligible for eviction / allocation.
            num_evictable_blocks = len(
                self.prefix_cache.stale_blocks - all_cache_hit_blocks
            )
        num_free_blocks = len(self.available_blocks) + num_evictable_blocks
        if total_blocks_to_allocate > num_free_blocks:
            raise RuntimeError(
                f"Not enough free blocks to service all {len(seq_ids_and_prompts)} requests.\n"
                f"Need an additional {total_blocks_to_allocate} blocks to store KV projections for all {total_sequence_length} tokens.\n"
                f"But only {num_free_blocks} out of {self.total_num_pages} cache blocks are available to be allocated.\n"
                f"You must restart your process and set a smaller batch size or max sequence length.\n"
            )

        # Allocate additional pages for each request in the batch
        for batch_idx, (seq_id, num_new_pages) in enumerate(
            blocks_to_allocate_by_seq.items()
        ):
            data = self.active_requests[seq_id]

            # Assign some new pages to this request.
            for _ in range(num_new_pages):
                next_block = self.alloc_block()
                data.blocks.append(next_block)

            # Populate the lookup table with the new pages.
            for i, block_idx in enumerate(data.blocks):
                lut_table_np[batch_idx, i] = block_idx

        # Build a tensor of maximum lengths. Each step slices the first row to
        # advance to the values for the next row.
        max_lengths_host = build_max_lengths_tensor(
            num_steps, max_seq_length, max_cache_length
        )

        lut_table_host = Tensor.from_numpy(lut_table_np)
        cache_lengths_host = Tensor.from_numpy(cache_lengths_np)

        ret_list = []
        for i, device in enumerate(self.devices):
            ret_list.append(
                RaggedKVCacheInputs(
                    blocks=self.blocks[i],
                    cache_lengths=cache_lengths_host.to(device=device),
                    lookup_table=lut_table_host.to(device=device),
                    max_lengths=max_lengths_host,
                )
            )

        self._runtime_check()

        return cast(List[KVCacheInputs], ret_list)

    def input_symbols(
        self,
    ) -> list[PagedCacheInputSymbols]:
        return [
            PagedCacheInputSymbols(
                kv_blocks=TensorType(
                    self.params.dtype,
                    shape=self.block_shape(is_parameterized=True),
                    device=DeviceRef(self.devices[i].label, self.devices[i].id),
                ),
                cache_lengths=TensorType(
                    DType.uint32,
                    shape=["batch_size"],
                    device=DeviceRef(self.devices[i].label, self.devices[i].id),
                ),
                lookup_table=TensorType(
                    DType.uint32,
                    shape=["batch_size", "max_num_pages"],
                    device=DeviceRef(self.devices[i].label, self.devices[i].id),
                ),
                max_lengths=TensorType(
                    DType.uint32, shape=["steps_remaining", 2]
                ),
            )
            for i in range(len(self.devices))
        ]

    def claim(self, n: int) -> list[int]:
        """Claims `n` blocks of memory in the cache for incoming requests.

        This returns a list of sequence ids, which identify a sequence's
        location within the cache. This sequence id can then be passed
        in the fetch function to return the ContinuousBatchingKVCacheCollection
        for those sequences.
        """
        seq_ids = super().claim(n)
        for seq_id in seq_ids:
            self.active_requests[seq_id] = PagedCacheMetadata(
                self.page_size, self.max_seq_len
            )
        if self.prefix_cache is not None:
            for seq_id in seq_ids:
                self.prefix_cache.external_claim(seq_id)
        return seq_ids

    def external_claim(self, seq_ids: list[int]) -> None:
        """Variant of the above where sequence ids are reserved externally."""
        super().external_claim(seq_ids)
        for seq_id in seq_ids:
            self.active_requests[seq_id] = PagedCacheMetadata(
                self.page_size, self.max_seq_len
            )
        if self.prefix_cache is not None:
            for seq_id in seq_ids:
                self.prefix_cache.external_claim(seq_id)

    def _count_all_pages(self) -> int:
        available_blocks = self.available_blocks
        prefix_cache_blocks = set()
        if self.prefix_cache is not None:
            prefix_cache_blocks = self.prefix_cache.blocks
        uncommitted_blocks = set()
        for seq_id in self.active_requests:
            uncommitted_blocks.update(
                self.active_requests[seq_id].uncommitted_blocks
            )
        return len(available_blocks | prefix_cache_blocks | uncommitted_blocks)

    def purge_prefix_cache(self) -> None:
        if self.prefix_cache is None:
            return
        evicted = self.prefix_cache.evict_blocks()
        for block in evicted:
            self.available_blocks.add(block)

    def release(self, seq_id: int) -> None:
        """Release `seq_id` provided, marking this sequence as complete.
        This returns the seq_id back to the available pool of cache memory,
        allowing it to be reused when a new sequence is claimed.
        """
        super().release(seq_id)
        data = self.active_requests[seq_id]

        if self.prefix_cache is not None:
            self.prefix_cache.release(seq_id)

        for block in data.blocks:
            self.release_block(block)
        del self.active_requests[seq_id]

    def _step(
        self,
        seq_ids_and_new_tokens: dict[int, np.ndarray],
    ) -> None:
        """Update the `cache_lengths` objects to not that a new
        kv projection step has occurred, and that the underlying memory
        has been written to. This `cache_lengths` value is then used
        downstream in `fetch` to track what section of memory should
        be used in the kernels.
        """
        self._runtime_check()

        for seq_id, new_tokens in seq_ids_and_new_tokens.items():
            if seq_id not in self.active_requests:
                raise ValueError(f"seq_id: {seq_id} not in active requests.")

            # Write the new tokens into the token array and bump the cached_idx
            data = self.active_requests[seq_id]
            data.step(new_tokens)

            if self.prefix_cache is not None:
                # Bump the committed_idx
                self.prefix_cache.step(
                    seq_id,
                    data,
                    free_block_fn=self.release_block,
                )

            expected_num_pages = ceildiv(data.seq_len, self.page_size)
            actual_num_pages = len(data.blocks)
            if expected_num_pages != actual_num_pages:
                raise ValueError(
                    f"Mismatch between expected and actual number of pages for seq_id: {seq_id}. Expected: {expected_num_pages}, Actual: {actual_num_pages}"
                )

        self._runtime_check()
