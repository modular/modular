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
from typing import Any, List, cast

import numpy as np
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    DeviceRef,
    Dim,
    TensorType,
    TensorValue,
    _OpaqueType,
    _OpaqueValue,
    ops,
)
from max.profiler import traced
from max.support.human_readable_formatter import to_human_readable_bytes
from max.support.math import ceildiv

from ._utils import build_max_lengths_tensor
from .block_manager import BlockManager
from .cache_params import KVCacheParams
from .cow import CowExecutor
from .manager import (
    KVCacheInputs,
    KVCacheInputSymbols,
    KVCacheManager,
    RaggedKVCacheInputs,
)
from .paged_cache_metadata import PagedCacheMetadata

logger = logging.getLogger("max.pipelines")


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


class PagedKVCacheCollectionFA3Fallback(_OpaqueValue):
    """The graph value for a view of the KV cache."""


class PagedKVCacheCollection(_OpaqueValue):
    """The graph value for a view of the KV cache."""


class FetchPagedKVCacheCollectionFA3Fallback:
    def __init__(self, kv_params: KVCacheParams, num_layers: int) -> None:
        self.kv_params = kv_params
        self.num_layers = num_layers

    def __call__(
        self,
        blocks: TensorValue,
        cache_lengths: TensorValue,
        lookup_table: TensorValue,
        is_cache_empty: TensorValue,
    ) -> PagedKVCacheCollectionFA3Fallback:
        """Constructs a PagedKVCacheCollection for use downstream.

        This constructs a KVCache for use with the DaoLabds FA3 backend.
        """

        # Explicit validation.
        if blocks.dtype != self.kv_params.dtype:
            msg = (
                f"expected blocks to be dtype: {self.kv_params.dtype}, got"
                f" {blocks.dtype}"
            )
            raise ValueError(msg)

        if blocks.rank != 5:
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

        # expand our lookup table to fit to num_blocks
        # we need a different lookup table for each layer
        # TODO(austin) move this to a unified location, right now it's split across the codebase.
        num_layers = ops.constant(self.num_layers, DType.uint32)
        start_constant = ops.constant(0, DType.uint32)
        step_constant = ops.constant(1, DType.uint32)
        layers_arange = ops.range(
            start_constant,
            num_layers,
            step_constant,
            out_dim=Dim(self.num_layers),
        )
        if blocks.device is not None:
            layers_arange = layers_arange.to(blocks.device)
        layers_arange = ops.reshape(layers_arange, shape=[-1, 1, 1])
        lookup_table = ops.reshape(
            lookup_table,
            shape=[1, lookup_table.shape[0], lookup_table.shape[1]],
        )

        lookup_table = ops.tile(lookup_table, repeats=[self.num_layers, 1, 1])
        lookup_table = lookup_table * self.num_layers + layers_arange
        cache_lengths_cast = cache_lengths.cast(DType.int32)
        lookup_table_cast = lookup_table.cast(DType.int32)

        return PagedKVCacheCollectionFA3Fallback(
            ops.custom(
                "mo.kv_collection_ctor.paged_fa3_fallback",
                values=[
                    blocks,
                    cache_lengths_cast,
                    lookup_table_cast,
                    is_cache_empty,
                ],
                out_types=[PagedKVCacheCollectionType()],
                parameters={
                    "num_heads": self.kv_params.n_kv_heads_per_device,
                    "head_dim": self.kv_params.head_dim,
                    "page_size": int(blocks.shape[2]),
                },
            )[0].opaque
        )


class FetchPagedKVCacheCollection:
    def __init__(self, kv_params: KVCacheParams, **kwargs: Any) -> None:
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
    @traced
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
            page_size: The number of tokens that will be stored in a single block.
            enable_runtime_checks: Whether to enable runtime correctness checks.
        """
        # The number of tokens in a single block.
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

        # The total number of blocks we'll have per-device.
        self.total_num_pages = int(
            cache_memory_per_device // single_page_size_bytes
        )

        # Validate that we are allocating enough blocks.
        single_page_size_bytes_str = to_human_readable_bytes(
            single_page_size_bytes
        )
        cache_memory_per_device_str = to_human_readable_bytes(
            cache_memory_per_device
        )
        if self.total_num_pages == 0:
            raise RuntimeError(
                f"Insufficient cache memory to allocate even a single page.\n"
                f"One page requires {single_page_size_bytes_str} but only {cache_memory_per_device_str} are available."
            )

        if max_batch_size > self.total_num_pages:
            logger.warning(
                f"Insufficient cache memory to support a batch containing {max_batch_size} requests with one token per request. "
                f"Need to allocate at least {max_batch_size} pages, but only have enough memory for {self.total_num_pages} pages. "
                f"One page requires {single_page_size_bytes_str} but only {cache_memory_per_device_str} are available."
            )

        blocks_needed_for_max_seq_len = ceildiv(max_seq_len, page_size)
        if blocks_needed_for_max_seq_len > self.total_num_pages:
            logger.warning(
                f"Insufficient cache memory to support a batch containing one request at the max sequence length of {max_seq_len} tokens. "
                f"Need to allocate at least {blocks_needed_for_max_seq_len} pages, but only have enough memory for {self.total_num_pages} pages. "
                f"One page requires {single_page_size_bytes_str} but only {cache_memory_per_device_str} are available."
            )

        logger.info(
            f"Paged KVCache Manager allocated {self.total_num_pages} pages using {single_page_size_bytes_str} per page"
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

        # Initialize the block buffers for each device.
        self.tensors: list[Tensor] = []
        for device in self.devices:
            self.tensors.append(
                Tensor.zeros(
                    self.block_shape(),  # type: ignore
                    self.params.dtype,
                    device=device,
                )
            )

        # Initialize block manager
        self.block_manager = BlockManager(
            total_num_blocks=self.total_num_pages,
            block_size=self.page_size,
            enable_prefix_caching=self.params.enable_prefix_caching,
            enable_runtime_checks=enable_runtime_checks,
        )

        # Whether prefix caching is enabled.
        self.enable_prefix_caching = self.params.enable_prefix_caching

        # Create cow executor
        self.cow_executor = CowExecutor(
            session=self.session,
            block_shape=self.block_shape(),
            dtype=self.params.dtype,
            devices=self.devices,
            tensors=self.tensors,
            page_size=self.page_size,
            enable_prefix_caching=self.enable_prefix_caching,
        )

        # Mapping from seq ID to blocks to track request state.
        self.active_requests: dict[int, PagedCacheMetadata] = {}

        # Preallocate a PagedCacheMetadata to use for sequences not in the cache.
        # This is to reduce the number of allocations. This is NOT thread safe.
        self.tmp_data = PagedCacheMetadata(self.page_size, self.max_seq_len)

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
            "total_num_pages" if is_parameterized else total_num_pages,
            kv_dim,
            num_layers,
            page_size,
            params.n_kv_heads_per_device,
            params.head_dim,
        ]

    @traced
    def can_fetch(
        self, seq_ids_and_prompts: dict[int, np.ndarray], num_steps: int = 1
    ) -> bool:
        """Checks if there are sufficient KV pages to run `fetch` on given batch.

        Sequences which have not been previously added to the cache can be handled
        by this method. We assume the cache lengths are zero in those cases.
        """

        tot_new_pages_needed = 0
        all_cache_hit_blocks: set[int] = set()

        for seq_id, prompt in seq_ids_and_prompts.items():
            prefix_blocks, _, new_pages_needed = self.query_fetch_stats(
                seq_id, prompt, num_steps
            )
            tot_new_pages_needed += new_pages_needed
            all_cache_hit_blocks.update(prefix_blocks)

        num_free_blocks = len(self.free_blocks - all_cache_hit_blocks)

        return tot_new_pages_needed <= num_free_blocks

    @traced
    def query_fetch_stats(
        self, seq_id: int, prompt: np.ndarray, num_steps: int = 1
    ) -> tuple[set[int], int, int]:
        """Query about the stats about running the fetch operation for a given
        sequence. It is OK if some seq_id are not in the cache.

        This method does not modify the state of the paged cache.

        Returns:
            - prefix_cache_blocks: Prefix cache blocks that would be reused for this seq.
            - tokens_to_encode: Number of tokens in prompt we need to encode when running the fetch.
            - new_pages_needed: Number of new pages we need to allocate when running the fetch.
        """
        reusing_tmp_data = False
        if seq_id in self.active_requests:
            data = self.active_requests[seq_id]
        else:
            reusing_tmp_data = True
            data = self.tmp_data

        # write the prompt into the token array
        data.fetch(prompt, num_steps)

        prefix_cache_blocks, tokens_to_encode, new_pages_needed = (
            self.block_manager.query_fetch_stats(seq_id, data)
        )

        # reverse the fetch operation so that this method does not mutate state
        data.undo_fetch(prompt, num_steps)

        if reusing_tmp_data:
            self.tmp_data.clear()

        return prefix_cache_blocks, tokens_to_encode, new_pages_needed

    @traced
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

        max_seq_len = -1
        for batch_idx, (seq_id, prompt) in enumerate(
            seq_ids_and_prompts.items()
        ):
            # Validate there aren't other inflight requests for this sequence.
            assert seq_id not in self.fetch_metadata

            data = self.active_requests[seq_id]
            data.fetch(prompt, num_steps)

            # Compute the total sequence length
            assert data.seq_len <= self.max_seq_len
            max_seq_len = max(max_seq_len, data.seq_len)

        # Allocate the buffers containing metadata about the batch.
        # [0, total_num_pages) are the valid block ids and total_num_pages
        # denotes an unassigned block.
        max_num_pages = ceildiv(max_seq_len, self.page_size)
        batch_size = len(seq_ids_and_prompts)
        lut_table_np = np.full(
            (batch_size, max_num_pages), self.total_num_pages, dtype=np.uint32
        )
        cache_lengths_np = np.zeros((batch_size,), dtype=np.uint32)

        # Update cache_lengths and max_lengths.
        max_prompt_len = 0
        max_cached_len = 0
        for batch_idx, (seq_id, prompt) in enumerate(
            seq_ids_and_prompts.items()
        ):
            # Query prefix cache and allocate new blocks.
            data = self.active_requests[seq_id]
            blocks, cow_args = self.block_manager.fetch(seq_id, data)
            if cow_args is not None:
                self.cow_executor.enqueue_cow(*cow_args)

            # We trim the prompt in place in the event of cache hits.
            prompt = data.prompt_tokens
            seq_ids_and_prompts[seq_id] = prompt

            # Populate the lookup table with the new pages.
            for i, block_idx in enumerate(blocks):
                lut_table_np[batch_idx, i] = block_idx

            # Get the existing cache length for this sequence.
            cache_length = data.cached_idx
            cache_lengths_np[batch_idx] = cache_length

            # Update the maximum lengths seen so far.
            max_prompt_len = max(max_prompt_len, len(prompt))
            max_cached_len = max(max_cached_len, cache_length + len(prompt))

        # Execute all COW memcpy operations.
        self.cow_executor.batch_async_execute()

        # Build a tensor of maximum lengths. Each step slices the first row to
        # advance to the values for the next row.
        max_lengths_host = build_max_lengths_tensor(
            num_steps, max_prompt_len, max_cached_len
        )

        # Convert from numpy to host tensors.
        lut_table_host = Tensor.from_numpy(lut_table_np)
        cache_lengths_host = Tensor.from_numpy(cache_lengths_np)

        ret_list = []
        for i, device in enumerate(self.devices):
            ret_list.append(
                RaggedKVCacheInputs(
                    blocks=self.tensors[i],
                    cache_lengths=cache_lengths_host.to(device=device),
                    lookup_table=lut_table_host.to(device=device),
                    max_lengths=max_lengths_host,
                )
            )

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
        return seq_ids

    def external_claim(self, seq_ids: list[int]) -> None:
        """Variant of the above where sequence ids are reserved externally."""
        super().external_claim(seq_ids)
        for seq_id in seq_ids:
            self.active_requests[seq_id] = PagedCacheMetadata(
                self.page_size, self.max_seq_len
            )

    def release(self, seq_id: int) -> None:
        """Release `seq_id` provided, marking this sequence as complete.
        This returns the seq_id back to the available pool of cache memory,
        allowing it to be reused when a new sequence is claimed.
        """
        super().release(seq_id)
        self.block_manager.release(seq_id)
        del self.active_requests[seq_id]

    @traced
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
        for seq_id, new_tokens in seq_ids_and_new_tokens.items():
            if seq_id not in self.active_requests:
                raise ValueError(f"seq_id: {seq_id} not in active requests.")

            # Write the new tokens into the token array and bump the cached_idx
            data = self.active_requests[seq_id]
            data.step(new_tokens)

            # We possible commit new blocks into the prefix cache.
            self.block_manager.step(seq_id, data)

    @property
    def free_blocks(self) -> set[int]:
        """Get the set of free blocks."""
        return self.block_manager.free_blocks

    @property
    def used_blocks_pct(self) -> float:
        """Get the percentage of blocks that are in usee."""
        pct = (
            self.total_num_pages - len(self.free_blocks)
        ) / self.total_num_pages
        assert 0 <= pct <= 1
        return pct

    @property
    def free_blocks_pct(self) -> float:
        """Get the percentage of blocks that are free."""
        pct = len(self.free_blocks) / self.total_num_pages
        assert 0 <= pct <= 1
        return pct

    @property
    def cache_hit_rate(self) -> float:
        """Get the percentage of prompt tokens that were retrieved from the cache."""
        pct = self.block_manager.cache_hit_rate
        assert 0 <= pct <= 1
        return pct

    @property
    def cow_blocks_copied(self) -> int:
        """The number of blocks that have been copied due to COW."""
        # TODO E2EOPT-115: Re-enable COW for paged_cache v2
        if self.cow_executor is None:
            return 0
        return self.cow_executor.cow_blocks_copied

    def reset_cow_blocks_copied(self) -> None:
        """Reset the number of cow operations performed."""
        self.cow_executor.reset_cow_blocks_copied()

    def get_req_blocks(self, seq_id: int) -> list[int]:
        """Get the block ids for a request."""
        return self.block_manager.get_req_blocks(seq_id)


class PagedKVCacheManagerFA3Fallback(PagedKVCacheManager):
    def input_symbols(self) -> list[PagedCacheInputSymbols]:
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
            kv_dim,
            "total_num_pages"
            if is_parameterized
            else (total_num_pages * num_layers),
            page_size,
            params.n_kv_heads_per_device,
            params.head_dim,
        ]
