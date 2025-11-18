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
import queue
from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any

import numpy as np
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, TensorValue
from max.interfaces import RequestID, TextGenerationContext, get_blocking
from max.nn.kv_cache.cache_params import KVCacheParams
from max.nn.kv_cache.manager import RaggedKVCacheInputs
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.nn.kv_cache.nested_iterable import NestedIterableDataclass
from max.nn.kv_cache.utils import build_max_lengths_tensor
from max.profiler import traced
from max.serve.kvcache_agent.kvcache_agent_service_v1_pb2 import (  # type: ignore
    MemoryTier,
)
from max.serve.queue.zmq_queue import ZmqPullSocket, ZmqPushSocket
from max.support.human_readable_formatter import to_human_readable_bytes
from max.support.math import ceildiv

from .block_copy_engine import BlockCopyEngine
from .block_manager import BlockManager

logger = logging.getLogger("max.pipelines")


@dataclass
class PagedCacheInputSymbols(NestedIterableDataclass):
    kv_blocks: BufferType
    cache_lengths: TensorType
    lookup_table: TensorType
    max_lengths: TensorType


class _TPPagedKVCacheManager:
    """Internal class used for managing KVCache blocks that supports tensor parallelism.

    This class should not be used directly by scheduler/pipelines. Instead, we
    should use the PagedKVCacheManager class instead.

    This class does NOT support data parallelism.
    """

    page_size: int
    """Number of tokens stored per block."""

    total_num_pages: int
    """Total number of logical pages (complete token slots) available.

    In tensor parallelism, each page's KV data is sharded across all devices,
    but this count represents complete logical pages (where all shards together
    form one complete page of `page_size` tokens).
    """

    device_tensors: list[Tensor]
    """List of tensors holding the KV cache blocks, one per device."""

    host_tensors: list[Tensor] | None
    """Tensor holding the KV cache blocks on the host for swapping (if enabled)."""

    total_num_host_pages: int
    """Total number of blocks allocated on the host for swapping (if enabled)."""

    block_manager: BlockManager
    """Manages allocation, eviction, and reuse of KV cache blocks."""

    enable_prefix_caching: bool
    """Flag indicating if prefix caching (block reuse) is enabled."""

    enable_kvcache_swapping_to_host: bool
    """Flag indicating if swapping blocks to host memory is enabled."""

    @traced
    def __init__(
        self,
        params: KVCacheParams,
        max_batch_size: int,
        max_seq_len: int,
        num_layers: int,
        devices: Sequence[Device],
        session: InferenceSession,
        available_cache_memory: int,
        zmq_endpoint_base: str | None = None,
        page_size: int = 128,
        enable_runtime_checks: bool = False,
    ) -> None:
        """Initialize the tensor-parallel paged KV cache manager.

        Args:
            params: The KVCacheParams for the given pipeline.
            max_batch_size: The maximum number of active requests that the
                manager should support.
            max_seq_len: The maximum sequence length we will generate.
            num_layers: The number of layers in the model.
            devices: The devices on which the manager will allocate memory.
                For tensor parallelism, KV cache data is sharded across these devices.
            session: The inference session to load ops from.
            available_cache_memory: The total amount of memory available for caching,
                summed across all devices. For example, if each of 4 devices has 1GB
                available, pass 4GB here. The memory calculation accounts for the fact
                that each page's data is sharded across devices in tensor parallelism.
            page_size: The number of tokens that will be stored in a single logical page.
            enable_runtime_checks: Whether to enable runtime correctness checks.
        """
        self.params = params
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers

        # Validate devices aligns with the n_devices in params
        if len(devices) != params.n_devices:
            raise ValueError(
                "n_devices provided in KVCacheParams, does not match number of devices initialized in the _TPPagedKVCacheManager"
            )

        self.devices = devices
        self.session = session

        if params.data_parallel_degree > 1:
            raise ValueError(
                "_TPPagedKVCacheManager does not support data parallelism."
            )

        # Attributes for managing available slots.
        self._available = set(range(self.max_batch_size))

        # Mappings between request IDs and sequence IDs.
        self._request_to_seq_id: dict[RequestID, int] = {}

        # The number of tokens in a single block.
        self.page_size = page_size

        # The number of bytes that a single page will occupy across all devices.
        # In tensor parallelism, each token's KV data is sharded across devices,
        # so bytes_required_per_token includes storage on all devices.
        single_page_size_bytes = (
            self.bytes_required_per_token(params, num_layers) * page_size
        )

        # Calculate the total number of logical pages available.
        # Since both available_cache_memory and single_page_size_bytes account
        # for all devices (total memory across devices / bytes per page across devices),
        # this gives us the number of complete logical pages we can store.
        self.total_num_pages = int(
            available_cache_memory // single_page_size_bytes
        )

        increment_cache_lengths_graph = (
            self._create_increment_cache_lengths_graph()
        )
        self.increment_cache_lengths_model = session.load(
            increment_cache_lengths_graph
        )

        # Validate that we are allocating enough blocks.
        single_page_size_bytes_str = to_human_readable_bytes(
            single_page_size_bytes
        )
        cache_memory_str = to_human_readable_bytes(available_cache_memory)
        across_x_devices_str = (
            f" across {len(devices)} devices" if len(devices) > 1 else ""
        )
        if self.total_num_pages == 0:
            raise RuntimeError(
                f"Insufficient cache memory to allocate even a single page.\n"
                f"One page requires {single_page_size_bytes_str} but only "
                f"{cache_memory_str} are available{across_x_devices_str}."
            )

        if max_batch_size > self.total_num_pages:
            memory_needed_str = to_human_readable_bytes(
                max_batch_size * single_page_size_bytes
            )
            logger.warning(
                f"Insufficient cache memory to support a batch containing {max_batch_size} "
                f"requests with one token per request. Need to allocate at least {max_batch_size} "
                f"pages ({memory_needed_str}), but only have enough memory for {self.total_num_pages} "
                f"pages ({cache_memory_str}{across_x_devices_str})."
            )

        blocks_needed_for_max_seq_len = ceildiv(max_seq_len, page_size)
        if blocks_needed_for_max_seq_len > self.total_num_pages:
            memory_needed_str = to_human_readable_bytes(
                blocks_needed_for_max_seq_len * single_page_size_bytes
            )
            logger.warning(
                f"Insufficient cache memory to support a batch containing one request "
                f"at the max sequence length of {max_seq_len} tokens. "
                f"Need to allocate at least {blocks_needed_for_max_seq_len} "
                f"pages ({memory_needed_str}), but only have enough memory for "
                f"{self.total_num_pages} pages ({cache_memory_str}{across_x_devices_str})."
            )

        # Whether prefix caching is enabled.
        self.enable_prefix_caching = self.params.enable_prefix_caching

        # Watches for requests to reset the prefix cache.
        self.reset_prefix_cache_backend: ResetPrefixCacheBackend | None = None
        if zmq_endpoint_base is not None and self.enable_prefix_caching:
            self.reset_prefix_cache_backend = ResetPrefixCacheBackend(
                zmq_endpoint_base
            )

        # Whether kvcache swapping to host is enabled
        self.enable_kvcache_swapping_to_host = (
            self.params.enable_kvcache_swapping_to_host
        )

        if (
            self.enable_kvcache_swapping_to_host
            and not self.enable_prefix_caching
        ):
            raise ValueError(
                "KVCache swapping to host is only supported when prefix caching is enabled"
            )

        # Initialize the block buffers for each device.
        self.device_tensors = []
        for device in self.devices:
            self.device_tensors.append(
                Tensor(
                    shape=self.block_shape(),  # type: ignore
                    dtype=self.params.dtype,
                    device=device,
                )
            )

        logger.debug(
            f"Paged KVCache Manager allocated {self.total_num_pages} device pages using "
            f"{single_page_size_bytes_str} per page{across_x_devices_str}."
        )

        self.host_tensors = None
        self.total_num_host_pages = 0
        if params.enable_kvcache_swapping_to_host:
            GiB = 1024 * 1024 * 1024
            host_kvcache_swap_space_gb = params.host_kvcache_swap_space_gb
            assert host_kvcache_swap_space_gb is not None
            host_kvcache_swap_space_bytes = int(
                host_kvcache_swap_space_gb * GiB
            )

            # The number of bytes that a single page will occupy on the host.
            # Note that this considers n_kv_heads, not n_kv_heads_per_device.
            self.total_num_host_pages = (
                host_kvcache_swap_space_bytes // single_page_size_bytes
            )

            if self.total_num_host_pages == 0:
                human_readable_host_swap_space_gb = to_human_readable_bytes(
                    host_kvcache_swap_space_bytes
                )
                raise RuntimeError(
                    f"Insufficient host swap space to allocate even a single page. "
                    f"One page requires {single_page_size_bytes_str} but only "
                    f"{human_readable_host_swap_space_gb} are available on host."
                )

            logger.debug(
                f"Paged KVCache Manager allocated {self.total_num_host_pages} host pages using "
                f"{single_page_size_bytes_str} per page."
            )
            self.host_tensors = []
            # Construct host tensors for each device.
            for dev in devices:
                if dev.is_host:
                    raise ValueError(
                        "Host device detected. Paging to host is not supported when executing on CPU."
                    )
                self.host_tensors.append(
                    Tensor(
                        shape=self.block_shape(self.total_num_host_pages),  # type: ignore
                        dtype=self.params.dtype,
                        device=dev,
                        pinned=True,
                    )
                )

        # Initialize block copy engine.
        self.block_copy_engine: BlockCopyEngine | None = None
        if self.enable_prefix_caching:
            self.block_copy_engine = BlockCopyEngine(
                block_size=self.page_size,
                num_device_blocks=self.total_num_pages,
                device_tensors=self.device_tensors,
                num_host_blocks=self.total_num_host_pages,
                host_tensors=self.host_tensors,
            )

        # Initialize block manager
        device_memory_tier = (
            MemoryTier.MEMORY_TIER_CPU
            if devices[0].is_host
            else MemoryTier.MEMORY_TIER_GPU
        )
        self.block_manager = BlockManager(
            device_memory_tier=device_memory_tier,
            total_num_blocks=self.total_num_pages,
            total_num_host_blocks=self.total_num_host_pages,
            block_size=self.page_size,
            block_copy_engine=self.block_copy_engine,
            enable_prefix_caching=self.params.enable_prefix_caching,
            enable_runtime_checks=enable_runtime_checks,
        )

    @classmethod
    def bytes_required_per_token(
        cls, params: KVCacheParams, num_layers: int
    ) -> int:
        """Compute total bytes required to store one token in the KV cache.

        In tensor parallelism (TP), a token's KV cache is sharded across devices.
        For example, with 32 KV heads and 4 devices, each device stores 8 heads.
        This method computes the total bytes needed across all devices to store
        one complete token.

        The calculation:
        1. Computes bytes per device: (2 * num_layers * n_kv_heads_per_device *
           head_dim * dtype_size)
        2. Multiplies by n_devices to get total bytes across all shards

        This total is used with total available memory (summed across devices) to
        determine how many logical pages can fit.

        Args:
            params: KV cache configuration (dtype, heads, devices, MLA, etc.).
                Assumes data_parallel_degree == 1 (no data parallelism).
            num_layers: Number of transformer layers contributing KV per token.

        Returns:
            Total bytes per token across all tensor-parallel devices.

        Example:
            With 4 devices, 32 KV heads, 128 head_dim, 32 layers, float16:
            - Per-device: 2 * 32 * 8 * 128 * 2 = 131,072 bytes
            - Total: 131,072 * 4 = 524,288 bytes per token across all devices
        """
        # Per-device bytes required for one token
        per_device_bytes = (
            reduce(mul, cls._block_shape(params, 1, 1, num_layers), 1)
            * params.dtype.size_in_bytes
        )
        # Total across all devices (assumes data_parallel_degree == 1)
        return per_device_bytes * params.n_devices

    @classmethod
    def page_size_adjusted_max_seq_len(
        cls, max_seq_len: int, page_size: int
    ) -> int:
        """
        Round up max_seq_len to the nearest multiple of page_size.

        The paged KV cache allocates memory in fixed-size pages, where each
        page stores ``page_size`` tokens. Even if a sequence only partially
        fills the last page, the full page must be allocated. Rounding up
        ensures our memory estimates account for whole-page allocation
        granularity, preventing underestimation and subsequent OOMs or
        insufficient page counts during scheduling.

        Args:
            max_seq_len: The original maximum sequence length.
            page_size: The page size in tokens.

        Returns:
            The smallest multiple of page_size greater than or equal to max_seq_len.
        """
        return math.ceil(max_seq_len / page_size) * page_size

    @classmethod
    def bytes_required_to_support_single_sequence(
        cls, params: KVCacheParams, num_layers: int, max_seq_len: int
    ) -> int:
        """
        Calculate the total bytes required to support a single sequence of length `max_seq_len`.

        This takes into account tensor parallelism (number of devices), number of heads,
        layers, and memory granularity due to page size.

        Args:
            params: KV cache configuration parameters.
            num_layers: Number of transformer layers.
            max_seq_len: Desired maximum sequence length.

        Returns:
            Total bytes required to store the KV cache for a single sequence of given length.
        """
        if params.page_size is None:
            raise ValueError(
                "page_size cannot be done when working with the Paged KV Cache"
            )

        return cls.bytes_required_per_token(
            params, num_layers
        ) * cls.page_size_adjusted_max_seq_len(max_seq_len, params.page_size)

    @classmethod
    def max_supported_sequence_length(
        cls, params: KVCacheParams, num_layers: int, memory_available: int
    ) -> int:
        """Return the maximum sequence length supported given available memory.

        The result is rounded down to the nearest multiple of ``params.page_size``
        because the paged KV cache allocates memory in whole pages. We compute
        how many full pages can be supported by the available memory and then
        convert back to tokens.

        Args:
            params: KV cache configuration parameters (must have a non-None page_size).
            num_layers: Number of transformer layers.
            memory_available: Total bytes available for KV cache across all devices.

        Returns:
            The maximum supported sequence length in tokens (multiple of page_size).
        """
        assert params.page_size is not None
        bytes_per_page = (
            cls.bytes_required_per_token(params, num_layers) * params.page_size
        )
        max_pages = memory_available // bytes_per_page
        return max_pages * params.page_size

    @classmethod
    def estimated_memory_size(
        cls,
        params: KVCacheParams,
        max_batch_size: int,
        max_seq_len: int,
        num_layers: int,
        available_cache_memory: int,
        devices: Sequence[Device],
        **kwargs: Any,
    ) -> int:
        """
        Estimate the usable KV cache memory given hardware limits and configuration.

        This function calculates how much memory can be used for caching by considering:
        1. The theoretical size required to accommodate a maximum work scenario, i.e., storing the KV cache
           for `max_batch_size` sequences, each up to `max_seq_len` tokens.
           - The memory per sequence is calculated via `cls.bytes_required_to_support_single_sequence`, which
             adjusts the sequence length to respect the page size allocation granularity and considers num_layers, heads, etc.
           - This value is then multiplied by the number of concurrent sequences (`max_batch_size`).
        2. The available KV cache memory on the system/hardware (`available_cache_memory`), as determined by unrelated
           components accounting for other usages (weights, activations, kernel buffers, etc.)

        The function guarantees that the actual allocated cache will not exceed the physical available memory,
        even if the configuration requests more (e.g., a large batch, large max_seq_len).

        Args:
            params: The KVCacheParams for the model and device configuration.
            max_batch_size: The maximum number of concurrent sequences supported.
            max_seq_len: The maximum sequence length to support.
            num_layers: The number of transformer layers.
            available_cache_memory: The maximum cache memory (in bytes) available, after other usages are subtracted.
            devices: List of devices participating in caching (unused here but included for API completeness).
            **kwargs: Additional parameters for extensibility.

        Returns:
            The minimum of (1) the memory required to support the configuration, and (2) memory available for caching.
            This is a safe, conservative estimate to avoid OOM and ensure configuration compliance.
        """
        size_to_support_full_cache = (
            cls.bytes_required_to_support_single_sequence(
                params, num_layers, max_seq_len
            )
            * max_batch_size
        )

        return min(available_cache_memory, size_to_support_full_cache)

    def block_shape(
        self,
        total_num_pages: int | None = None,
        num_layers: int | None = None,
        is_parameterized: bool = False,
    ) -> list[int | str]:
        if total_num_pages is None:
            total_num_pages = self.total_num_pages

        if num_layers is None:
            num_layers = self.num_layers

        return self._block_shape(
            self.params,
            total_num_pages,
            self.page_size,
            num_layers,
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
        kv_dim = 2 if not params.is_mla else 1
        return [
            "total_num_pages" if is_parameterized else total_num_pages,
            kv_dim,
            num_layers,
            page_size,
            params.n_kv_heads_per_device,
            params.head_dim,
        ]

    @traced
    def _does_req_need_more_blocks(
        self, ctx: TextGenerationContext, num_steps: int
    ) -> bool:
        """Determines if a request needs additional blocks."""
        seq_len = ctx.current_length + num_steps - 1
        num_blocks = len(self.block_manager.req_to_blocks[ctx.request_id])
        return seq_len > num_blocks * self.page_size

    @traced
    def get_pct_used_blocks_after_allocation(
        self, ctx: TextGenerationContext, num_steps: int = 1
    ) -> float:
        """Get the percentage of blocks used after allocating for a request."""
        num_used_blocks = self.total_num_pages - self.num_free_blocks
        num_needed_blocks = (
            num_used_blocks
            + self.block_manager.num_blocks_to_allocate(ctx, num_steps)
        )
        assert self.total_num_pages > 0
        return min(
            1.0,
            num_needed_blocks / self.total_num_pages,
        )

    @traced
    def alloc(self, data: TextGenerationContext, num_steps: int = 1) -> None:
        """Allocates blocks for a request to run for N steps.

        This method allocates blocks needed by a request to run for N steps.
        When prefix caching is enabled, some of the allocated blocks may be
        retrieved from the prefix cache.

        Args:
            data: The text generation context for the request. The request ID
                must already be assigned to a replica via `claim`.
            num_steps: The number of steps to reserve blocks for. Default: 1.

        Raises:
            InsufficientBlocksError: If there are insufficient free blocks to
            satisfy the allocation.
        """
        self.block_manager.reuse_blocks_from_prefix_cache(data)
        self.block_manager.allocate_new_blocks(data, num_steps)

    @traced
    def get_runtime_inputs(
        self, batch: Sequence[TextGenerationContext], num_steps: int = 1
    ) -> Sequence[RaggedKVCacheInputs]:
        """Get the graph inputs for a batch of requests.

        This method will raise a RuntimeError if any request has insufficient blocks
        already allocated to it to run for the given number of steps.

        Args:
            batch: Batch of requests
            num_steps: Number of steps to run for
        """

        if self.block_copy_engine is not None:
            self.block_copy_engine.wait_for_completion()

        if self.reset_prefix_cache_backend is not None:
            if self.reset_prefix_cache_backend.should_reset_prefix_cache():
                self.reset_prefix_cache()

        max_seq_len = -1
        for batch_idx, ctx in enumerate(batch):  # noqa: B007
            # Allocate blocks for request if we need more.
            if self._does_req_need_more_blocks(ctx, num_steps):
                raise ValueError(
                    f"Called fetch with request {ctx.request_id} but it does not have sufficient blocks. `alloc` must be called first."
                )

            # Compute the total sequence length
            seq_len = ctx.current_length + num_steps - 1
            if seq_len > self.max_seq_len:
                raise RuntimeError(
                    f"Request has current length ({ctx.current_length}) + num_steps ({num_steps}) - 1 = {seq_len} which exceeds model max_seq_len of {self.max_seq_len}"
                )
            max_seq_len = max(max_seq_len, seq_len)

        # Allocate the buffers containing metadata about the batch.
        # [0, total_num_pages) are the valid block ids and total_num_pages
        # denotes an unassigned block.
        max_num_pages = ceildiv(max_seq_len, self.page_size)
        batch_size = len(batch)
        lut_table_np = np.full(
            (batch_size, max_num_pages), self.total_num_pages, dtype=np.uint32
        )
        cache_lengths_np = np.zeros((batch_size,), dtype=np.uint32)

        # Update cache_lengths and max_lengths.
        max_prompt_len = 0
        max_cached_len = 0
        for batch_idx, ctx in enumerate(batch):
            # Get the blocks for this request.
            blocks = self.block_manager.get_req_blocks(ctx.request_id)

            # Sanity check that we have enough blocks.
            seq_len = ctx.current_length + num_steps - 1
            num_required_blocks = ceildiv(seq_len, self.page_size)
            assert len(blocks) >= num_required_blocks
            if len(blocks) > num_required_blocks:
                blocks = blocks[:num_required_blocks]

            # Vectorized assignment of block indices to lookup table
            lut_table_np[batch_idx, : len(blocks)] = np.array(
                blocks, dtype=np.uint32
            )

            # Get the existing cache length for this sequence.
            cache_length = ctx.start_idx
            cache_lengths_np[batch_idx] = cache_length

            # Update the maximum lengths seen so far.
            prompt_tokens = ctx.active_length
            max_prompt_len = max(max_prompt_len, prompt_tokens)
            max_cached_len = max(max_cached_len, cache_length + prompt_tokens)

        self.block_manager.eagerly_offload_recently_committed_blocks()

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
                    blocks=self.device_tensors[i],
                    cache_lengths=cache_lengths_host.to(device=device),
                    lookup_table=lut_table_host.to(device=device),
                    max_lengths=max_lengths_host,
                )
            )

        return ret_list

    def get_symbolic_inputs(
        self,
        devices: Sequence[Device] | None = None,
        num_layers: int | None = None,
    ) -> Sequence[NestedIterableDataclass]:
        return self._input_symbols(devices, num_layers, dynamic_dim_prefix="")

    def _input_symbols(
        self,
        devices: Sequence[Device] | None = None,
        num_layers: int | None = None,
        dynamic_dim_prefix: str = "",
    ) -> Sequence[PagedCacheInputSymbols]:
        """Returns the input symbols for the paged KV cache.

        Args:
            devices: The devices to use for the input symbols.
            num_layers: The number of layers to use for the input symbols.
            dynamic_dim_prefix: The prefix to use for the dynamic dimensions.
                This is used to differentiate between the different inputs
                between replicas.
        """
        if devices is None:
            devices = self.devices

        if num_layers is None:
            num_layers = self.num_layers

        return [
            PagedCacheInputSymbols(
                kv_blocks=BufferType(
                    self.params.dtype,
                    shape=self.block_shape(
                        num_layers=num_layers, is_parameterized=True
                    ),
                    device=DeviceRef(device.label, device.id),
                ),
                cache_lengths=TensorType(
                    DType.uint32,
                    shape=[dynamic_dim_prefix + "batch_size"],
                    device=DeviceRef(device.label, device.id),
                ),
                lookup_table=TensorType(
                    DType.uint32,
                    shape=[
                        dynamic_dim_prefix + "batch_size",
                        dynamic_dim_prefix + "max_num_pages",
                    ],
                    device=DeviceRef(device.label, device.id),
                ),
                max_lengths=TensorType(
                    DType.uint32,
                    shape=[dynamic_dim_prefix + "steps_remaining", 2],
                    device=DeviceRef.CPU(),
                ),
            )
            for device in devices
        ]

    def release(self, request_id: RequestID) -> None:
        """Release the sequence associated with :obj:`request_id`, marking this sequence as complete.
        This returns the sequence ID back to the available pool of cache memory,
        allowing it to be reused when a new sequence is claimed.
        """
        # Get the sequence ID from the request ID for internal use
        if request_id not in self._request_to_seq_id:
            raise ValueError(
                f"Attempted to release request ID {request_id} but it is not claimed"
            )

        # Call the base class release method with the request_id
        if request_id not in self._request_to_seq_id:
            raise ValueError(
                f"Attempted to release request ID {request_id} but it is not claimed"
            )

        # Look up the sequence ID
        seq_id = self._request_to_seq_id[request_id]

        # Clean up mappings
        del self._request_to_seq_id[request_id]

        self._available.add(seq_id)

        # Call the block manager release method with the request_id
        self.block_manager.release(request_id)

    @traced
    def step(self, batch: Sequence[TextGenerationContext]) -> None:
        """Commit new tokens into the prefix cache.

        This is a no-op if prefix caching is disabled.
        """
        for ctx in batch:
            # We possibly commit new blocks into the prefix cache.
            self.block_manager.step(ctx)

    @property
    def num_free_blocks(self) -> int:
        """Get the set of free blocks."""
        return len(self.block_manager.device_block_pool.free_blocks)

    @property
    def used_blocks_pct(self) -> float:
        """Get the percentage of blocks that are in usee."""
        pct = (
            self.total_num_pages - self.num_free_blocks
        ) / self.total_num_pages
        assert 0 <= pct <= 1
        return pct

    @property
    def host_committed_block_pct(self) -> float:
        """Get the percentage of host blocks that are committed."""
        if self.block_manager.host_block_pool is None:
            return 0
        host_committed_blocks = len(
            self.block_manager.host_block_pool.hash_to_committed_block
        )
        pct = host_committed_blocks / self.total_num_host_pages
        assert 0 <= pct <= 1
        return pct

    @property
    def free_blocks_pct(self) -> float:
        """Get the percentage of blocks that are free."""
        pct = self.num_free_blocks / self.total_num_pages
        assert 0 <= pct <= 1
        return pct

    def get_req_blocks(self, request_id: RequestID) -> Sequence[int]:
        """Get the block ids for a request."""
        return self.block_manager.get_req_blocks(request_id)

    def _create_increment_cache_lengths_graph(self) -> Graph:
        input_symbols = self.get_symbolic_inputs()
        cache_lengths_types = [
            input_symbols[i][1] for i in range(len(self.devices))
        ]

        input_row_offsets_type = TensorType(
            DType.uint32,
            shape=["input_row_offsets_len"],
            device=DeviceRef(self.devices[0].label, self.devices[0].id),
        )

        with Graph(
            "update_cache_lengths",
            input_types=[input_row_offsets_type, *cache_lengths_types],
        ) as graph:
            inp_row_offset, *cache_lengths = (
                inp.tensor for inp in graph.inputs
            )
            # broadcast the inp_row_offset to all devices (naive)
            # get rid of this if statement after #51465 merges
            if len(self.devices) > 1:
                input_row_offsets = [
                    inp_row_offset.to(DeviceRef(d.label, d.id))
                    for d in self.devices
                ]
            else:
                input_row_offsets = [inp_row_offset]
            outputs = []
            for i in range(len(self.devices)):
                cache_length = cache_lengths[i]
                assert isinstance(cache_length, TensorValue)
                right_slice = input_row_offsets[i][1:].rebind(
                    cache_length.shape
                )
                left_slice = input_row_offsets[i][
                    : input_row_offsets[i].shape[0] - 1
                ].rebind(cache_length.shape)
                increment_amount = right_slice - left_slice
                outputs.append(cache_length + increment_amount)
            graph.output(*outputs)

        return graph

    def increment_cache_lengths(
        self,
        kv_cache_inputs: Sequence[RaggedKVCacheInputs],
        prev_model_inputs: Any,
    ) -> Sequence[RaggedKVCacheInputs]:
        """Prepares cache inputs for the next token in multistep execution.

        Updates the cache lengths for the next inference step without requiring device
        synchronization or memory copies. This is crucial for maintaining performance
        during multi-token generation.

        Args:
            kv_cache_inputs: Current cache state tuples (blocks, lengths, lookup, max_lengths)
            prev_model_inputs: Previous model inputs including row offsets

        Returns:
            Updated cache input tuples with incremented lengths.
        """
        blocks = [kv_cache_inputs[i].blocks for i in range(len(self.devices))]
        cache_lengths = [
            kv_cache_inputs[i].cache_lengths for i in range(len(self.devices))
        ]
        lookup_table = [
            kv_cache_inputs[i].lookup_table for i in range(len(self.devices))
        ]

        # max_lengths is host allocated and the same across all devices.
        max_lengths = kv_cache_inputs[0].max_lengths

        # Update the cache_lengths of our batch by the previous sequence length.
        # Handle both single tensor and list of tensors for compatibility
        if isinstance(prev_model_inputs.input_row_offsets, list):
            # InternVL case: use the first tensor (row offsets are identical across devices)
            row_offsets = prev_model_inputs.input_row_offsets[0]
        else:
            # Standard case: single tensor
            row_offsets = prev_model_inputs.input_row_offsets
        row_offsets = row_offsets.to(self.devices[0])

        updated_cache_lengths = self.increment_cache_lengths_model.execute(
            row_offsets, *cache_lengths
        )

        # Advance to the next step of the max_lengths tensor.
        updated_max_lengths = max_lengths[1:, :]

        # Return our updated batch.
        assert isinstance(kv_cache_inputs, list)
        for i in range(len(self.devices)):
            updated_cache_length = updated_cache_lengths[i]
            assert isinstance(updated_cache_length, Tensor)
            kv_cache_inputs[i] = RaggedKVCacheInputs(
                blocks=blocks[i],
                cache_lengths=updated_cache_length,
                lookup_table=lookup_table[i],
                max_lengths=updated_max_lengths,
            )
        return kv_cache_inputs

    def claim(self, request_id: RequestID) -> None:
        """Reserve a sequence ID for the given request ID."""
        if request_id in self._request_to_seq_id:
            raise ValueError(f"Request ID {request_id} is already claimed")

        if not self._available:
            raise ValueError("No available sequence IDs to claim")

        # Get the next available sequence ID
        seq_id = self._available.pop()

        # Update mappings
        self._request_to_seq_id[request_id] = seq_id

    def contains(self, request_id: RequestID) -> bool:
        """Check if the given request ID is currently active in the cache.

        Args:
            request_id: The request ID to check for.

        Returns:
            True if the request ID is active in the cache, False otherwise.
        """
        return request_id in self._request_to_seq_id

    @property
    def metrics(self) -> KVCacheMetrics:
        return self.block_manager.metrics

    def reset_metrics(self) -> None:
        self.block_manager.reset_metrics()

    def reset_prefix_cache(self) -> None:
        self.block_manager.reset_prefix_cache()


ZMQ_RESET_PREFIX_CACHE_ENDPOINT = "reset_prefix_cache"


class ResetPrefixCacheBackend:
    def __init__(self, zmq_endpoint_base: str):
        self.socket = ZmqPullSocket[None](
            endpoint=f"{zmq_endpoint_base}-{ZMQ_RESET_PREFIX_CACHE_ENDPOINT}",
            payload_type=None,
        )

    def should_reset_prefix_cache(self, blocking: bool = False) -> bool:
        # If blocking is True, we do not return until we receive a message from
        # the frontend to reset the prefix cache. Hence, it will always return True.
        if blocking:
            get_blocking(self.socket)
            return True

        # If non-blocking, we return True if there is a message in the queue.
        try:
            self.socket.get_nowait()
            return True
        except queue.Empty:
            return False


class ResetPrefixCacheFrontend:
    def __init__(self, zmq_endpoint_base: str):
        self.socket = ZmqPushSocket[None](
            endpoint=f"{zmq_endpoint_base}-{ZMQ_RESET_PREFIX_CACHE_ENDPOINT}",
            payload_type=None,
        )

    def enqueue_reset_prefix_cache(self) -> None:
        self.socket.put_nowait(None)
