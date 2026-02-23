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

"""PagedAttention-enabled KV cache for the Transformer leveraging the mo.opaque pattern."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.interfaces import RequestID, TextGenerationContext
from max.kv_cache.kv_connector import KVConnector
from max.nn.legacy.kv_cache import KVCacheParams, RaggedKVCacheInputs
from max.nn.legacy.kv_cache.metrics import KVCacheMetrics
from max.nn.legacy.kv_cache.utils import build_max_lengths_tensor
from max.profiler import traced
from max.serve.kvcache_agent.kvcache_agent_service_v1_pb2 import (  # type: ignore
    MemoryTier,
)
from max.support.math import ceildiv

from ..connectors import create_connector
from .block_manager import BlockManager

logger = logging.getLogger("max.pipelines")


def _contiguous_prefix_2d(buffer: Buffer, rows: int, cols: int) -> Buffer:
    """Returns a contiguous 2D prefix view of ``buffer``.

    The returned buffer aliases the original storage and has shape
    ``(rows, cols)``.
    """
    if rows < 0 or cols < 0:
        raise ValueError("rows and cols must be non-negative")

    num_elements = rows * cols
    if num_elements > buffer.num_elements:
        raise ValueError(
            "Requested contiguous prefix exceeds backing buffer capacity: "
            f"{num_elements} > {buffer.num_elements}."
        )

    flat = buffer.view(buffer.dtype, (buffer.num_elements,))
    return flat[:num_elements].view(buffer.dtype, (rows, cols))


class _PersistentKVInputBuffers:
    """Persistent buffers backing runtime LUT and cache-length inputs."""

    max_batch_size: int
    """Maximum number of request rows currently allocated."""

    lut_table_host: Buffer
    """LUT host buffer."""

    cache_lengths_host: Buffer
    """Cache lengths host buffer."""

    lut_table_by_device: list[Buffer]
    """LUT on each device."""

    cache_lengths_by_device: list[Buffer]
    """Cache lengths on each device."""

    def __init__(
        self,
        max_batch_size: int,
        max_total_num_pages: int,
        devices: Sequence[Device],
    ):
        self.max_batch_size = max_batch_size
        device0 = devices[0]
        pinned = not device0.is_host

        # Runtime lookup-table shape is [max_batch_size, max_total_num_pages]:
        # rows map to request slots in the current batch and columns map to
        # per-request page slots.
        # [0, total_num_pages) are the valid block ids and total_num_pages
        # denotes an unassigned block.
        self.lut_table_host = Buffer(
            shape=(max_batch_size, max_total_num_pages),
            dtype=DType.uint32,
            device=device0,
            pinned=pinned,
        )

        if pinned:
            self.lut_table_host.disable_auto_sync()

        self.cache_lengths_host = Buffer(
            shape=(max_batch_size,),
            dtype=DType.uint32,
            device=device0,
            pinned=pinned,
        )

        if pinned:
            self.cache_lengths_host.disable_auto_sync()

        self.lut_table_by_device = []
        self.cache_lengths_by_device = []
        for device in devices:
            self.lut_table_by_device.append(
                self.lut_table_host.to(device=device)
            )
            self.cache_lengths_by_device.append(
                self.cache_lengths_host.to(device=device)
            )

    def values(self) -> tuple[Buffer, Buffer, list[Buffer], list[Buffer]]:
        return (
            self.lut_table_host,
            self.cache_lengths_host,
            self.lut_table_by_device,
            self.cache_lengths_by_device,
        )


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

    device_tensors: list[Buffer]
    """List of tensors holding the KV cache blocks, one per device."""

    device_scale_tensors: list[Buffer] | None
    """List of scales for the quantized KV cache blocks, one per device."""

    block_manager: BlockManager
    """Manages allocation, eviction, and reuse of KV cache blocks."""

    connector: KVConnector
    """Connector for external cache tiers (host memory, LMCache, etc.)."""

    enable_prefix_caching: bool
    """Flag indicating if prefix caching (block reuse) is enabled."""

    enable_kvcache_swapping_to_host: bool
    """Flag indicating if swapping blocks to host memory is enabled."""

    @traced
    def __init__(
        self,
        params: KVCacheParams,
        total_num_pages: int,
        total_num_host_pages: int,
        devices: Sequence[Device],
        session: InferenceSession,
        max_batch_size: int,
        enable_runtime_checks: bool = False,
    ) -> None:
        """Initialize the tensor-parallel paged KV cache manager.

        Args:
            params: The KVCacheParams for the given pipeline.
            total_num_pages: Total number of device pages across all TP shards.
            total_num_host_pages: Total number of host pages for swapping.
            devices: The devices on which the manager will allocate memory.
                For tensor parallelism, KV cache data is sharded across these devices.
            session: The inference session to load ops from.
            max_batch_size: Maximum runtime batch size expected for this
                replica. Runtime lookup-table and cache-length buffers are
                preallocated to this row capacity.
            enable_runtime_checks: Whether to enable runtime correctness checks.
        """
        self.params = params
        self.total_num_pages = total_num_pages
        self.total_num_host_pages = total_num_host_pages
        self.page_size = params.page_size
        self.devices = devices
        self.session = session

        # Validate devices aligns with the n_devices in params
        if len(devices) != params.n_devices:
            raise ValueError(
                "Number of devices provided in KVCacheParams does not match the number of devices initialized in the _TPPagedKVCacheManager"
            )

        if params.data_parallel_degree > 1:
            raise ValueError(
                "_TPPagedKVCacheManager does not support data parallelism."
            )

        # Track the set of requests that are currently claimed.
        self._claimed_requests: set[RequestID] = set()
        self._max_batch_size = max_batch_size
        if self._max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")

        max_total_num_pages = self.total_num_pages

        self._persistent_kv_input_buffers = _PersistentKVInputBuffers(
            max_batch_size=self._max_batch_size,
            max_total_num_pages=max_total_num_pages,
            devices=self.devices,
        )

        # Whether prefix caching is enabled.
        self.enable_prefix_caching = self.params.enable_prefix_caching

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
            # Zero-initializing GPU device tensors does not introduce significant latency.
            # Memory is initialized because OOB TMA reads of uninitialized memory on GPU can result in NaNs in downstream kernels.
            self.device_tensors.append(
                Buffer.zeros(
                    shape=[total_num_pages, *params.shape_per_block],
                    dtype=self.params.dtype,
                    device=device,
                )
            )

        self.device_scale_tensors = None
        if self.params.dtype in (DType.float8_e4m3fn, DType.float8_e4m3fnuz):
            assert params.kvcache_quant_config is not None
            self.device_scale_tensors = []
            scale_dtype = params.kvcache_quant_config.scale_dtype
            for device in self.devices:
                self.device_scale_tensors.append(
                    Buffer.zeros(
                        shape=[total_num_pages, *params.shape_per_scale_block],
                        dtype=scale_dtype,
                        device=device,
                    )
                )

        # Initialize connector for external cache tiers (host memory, LMCache, etc.)
        # The connector owns host memory, host block pool, and handles H2D/D2H transfers.
        self.connector: KVConnector = create_connector(
            params=params,
            devices=devices,
            device_tensors=self.device_tensors,
            device_scale_tensors=self.device_scale_tensors,
            total_num_host_blocks=total_num_host_pages,
            session=session,
        )

        # Initialize block manager for device-side allocation and prefix caching.
        # The connector is passed to BlockManager for host cache operations.
        device_memory_tier = (
            MemoryTier.MEMORY_TIER_CPU
            if devices[0].is_host
            else MemoryTier.MEMORY_TIER_GPU
        )
        self.block_manager = BlockManager(
            device_memory_tier=device_memory_tier,
            total_num_blocks=self.total_num_pages,
            block_size=self.page_size,
            connector=self.connector,
            enable_prefix_caching=self.params.enable_prefix_caching,
            enable_runtime_checks=enable_runtime_checks,
        )

    @traced
    def _does_req_need_more_blocks(
        self, ctx: TextGenerationContext, num_steps: int
    ) -> bool:
        """Determines if a request needs additional blocks."""
        seq_len = len(ctx.tokens) + num_steps - 1
        num_blocks = len(self.block_manager.req_to_blocks[ctx.request_id])
        return seq_len > num_blocks * self.page_size

    @traced
    def get_pct_used_blocks_after_allocation(
        self, ctx: TextGenerationContext, num_steps: int = 1
    ) -> float:
        """Gets the percentage of blocks used after allocating for a request."""
        num_needed_blocks = (
            self.num_used_pages
            + self.block_manager.num_blocks_to_allocate(ctx, num_steps)
        )
        assert self.num_pages > 0
        return min(
            1.0,
            num_needed_blocks / self.num_pages,
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
        """Gets the graph inputs for a batch of requests.

        This method will raise a RuntimeError if any request has insufficient blocks
        already allocated to it to run for the given number of steps.

        Args:
            batch: Batch of requests.
            num_steps: Number of steps to run for.
        """
        # Wait for any pending connector operations (H2D loads from host cache).
        self.connector.sync()

        max_seq_len = 0
        for ctx in batch:
            # Allocate blocks for request if we need more.
            if self._does_req_need_more_blocks(ctx, num_steps):
                raise ValueError(
                    f"Called fetch with request {ctx.request_id} but it does not have sufficient blocks. `alloc` must be called first."
                )

            # Compute the total sequence length
            seq_len = len(ctx.tokens) + num_steps - 1
            max_seq_len = max(max_seq_len, seq_len)

        max_total_num_pages = ceildiv(max_seq_len, self.page_size)
        batch_size = len(batch)
        if batch_size > self._max_batch_size:
            raise ValueError(
                "Runtime batch size exceeds preallocated KV runtime "
                f"buffer capacity: {batch_size} > {self._max_batch_size}."
            )

        runtime_inputs = self._persistent_kv_input_buffers
        # Slice row views from persistent full-capacity buffers so we only
        # update active requests while preserving stable shapes/pointers for
        # replayed graphs.
        lut_table_host = _contiguous_prefix_2d(
            runtime_inputs.lut_table_host,
            rows=batch_size,
            cols=max_total_num_pages,
        )
        cache_lengths_host = runtime_inputs.cache_lengths_host[:batch_size]
        # Take a contiguous view of the LUT buffer, which is written to below.
        lut_table_by_device = [
            _contiguous_prefix_2d(
                buffer,
                rows=batch_size,
                cols=max_total_num_pages,
            )
            for buffer in runtime_inputs.lut_table_by_device
        ]
        cache_lengths_by_device = [
            buffer[:batch_size]
            for buffer in runtime_inputs.cache_lengths_by_device
        ]

        assert lut_table_host.is_contiguous
        assert all(buffer.is_contiguous for buffer in lut_table_by_device)

        lut_table_np = lut_table_host.to_numpy()
        lut_table_np.fill(self.total_num_pages)
        cache_lengths_np = cache_lengths_host.to_numpy()
        cache_lengths_np.fill(0)

        # Update cache_lengths and max_lengths.
        max_prompt_len = 0
        max_cached_len = 0
        for batch_idx, ctx in enumerate(batch):
            # Get the blocks for this request.
            blocks = self.block_manager.get_req_blocks(ctx.request_id)

            # Sanity check that we have enough blocks.
            seq_len = len(ctx.tokens) + num_steps - 1
            num_required_blocks = ceildiv(seq_len, self.page_size)
            assert len(blocks) >= num_required_blocks
            if len(blocks) > num_required_blocks:
                blocks = blocks[:num_required_blocks]

            # Vectorized assignment of block indices to lookup table
            lut_table_np[batch_idx, : len(blocks)] = np.array(
                blocks, dtype=np.uint32
            )

            # Get the existing cache length for this sequence.
            cache_length = ctx.tokens.processed_length
            cache_lengths_np[batch_idx] = cache_length

            # Update the maximum lengths seen so far.
            prompt_tokens = ctx.tokens.active_length
            max_prompt_len = max(max_prompt_len, prompt_tokens)
            max_cached_len = max(max_cached_len, cache_length + prompt_tokens)

        # Initiate any pending async saves to external cache tiers.
        self.connector.flush()

        # Build a tensor of maximum lengths. Each step slices the first row to
        # advance to the values for the next row. This should not be allocated
        # on pinned memory since it is exclusively accessed on the CPU and never
        # copied to the GPU.
        max_lengths_host = build_max_lengths_tensor(
            num_steps, max_prompt_len, max_cached_len
        )

        ret_list: list[RaggedKVCacheInputs] = []
        for cache_lengths_device, lookup_table_device, device_blocks in zip(
            cache_lengths_by_device,
            lut_table_by_device,
            self.device_tensors,
            strict=True,
        ):
            cache_lengths_device.inplace_copy_from(cache_lengths_host)
            lookup_table_device.inplace_copy_from(lut_table_host)
            scales = None
            if self.device_scale_tensors is not None:
                assert len(self.device_tensors) == len(
                    self.device_scale_tensors
                )
                scales = self.device_scale_tensors[len(ret_list)]
            ret_list.append(
                RaggedKVCacheInputs(
                    blocks=device_blocks,
                    cache_lengths=cache_lengths_device,
                    lookup_table=lookup_table_device,
                    max_lengths=max_lengths_host,
                    kv_scales=scales,
                )
            )

        return ret_list

    def release(self, request_id: RequestID) -> None:
        """Releases the sequence associated with :obj:`request_id`, marking it complete.

        Returns the sequence ID to the pool of cache memory for reuse.
        """
        if request_id not in self._claimed_requests:
            raise ValueError(
                f"Attempted to release request ID {request_id} but it is not claimed"
            )

        self._claimed_requests.remove(request_id)

        # Get block IDs before releasing
        block_ids = self.block_manager.get_req_blocks(request_id)

        # Call the block manager release method with the request_id
        self.block_manager.release(request_id)

        # Notify connector of request completion
        self.connector.on_request_complete(request_id, block_ids)

    @traced
    def step(self, batch: Sequence[TextGenerationContext]) -> None:
        """Commit new tokens into the prefix cache.

        This is a no-op if prefix caching is disabled.
        """
        for ctx in batch:
            # We possibly commit new blocks into the prefix cache.
            self.block_manager.step(ctx)

    @property
    def num_pages(self) -> int:
        return self.total_num_pages

    @property
    def num_used_pages(self) -> int:
        """Get the set of used blocks."""
        free_blocks = self.block_manager.device_block_pool.free_blocks
        return self.total_num_pages - len(free_blocks)

    @property
    def num_host_pages(self) -> int:
        """Total number of host blocks available."""
        return self.connector.num_host_blocks

    @property
    def num_used_host_pages(self) -> int:
        """Number of host blocks currently in use."""
        return self.connector.num_used_host_blocks

    @property
    def host_tensors(self) -> list[Buffer] | None:
        """Host tensors for KV cache swapping (owned by connector)."""
        return self.connector.host_tensors

    @property
    def host_scale_tensors(self) -> list[Buffer] | None:
        """Host scale tensors for FP8 quantization (owned by connector)."""
        return self.connector.host_scale_tensors

    def get_req_blocks(self, request_id: RequestID) -> Sequence[int]:
        """Get the block ids for a request."""
        return self.block_manager.get_req_blocks(request_id)

    def claim(self, request_id: RequestID) -> None:
        """Reserve a sequence ID for the given request ID."""
        if request_id in self._claimed_requests:
            raise ValueError(f"Request ID {request_id} is already claimed")
        self._claimed_requests.add(request_id)

    def contains(self, request_id: RequestID) -> bool:
        """Check if the given request ID is currently active in the cache.

        Args:
            request_id: The request ID to check for.

        Returns:
            True if the request ID is active in the cache, False otherwise.
        """
        return request_id in self._claimed_requests

    @property
    def metrics(self) -> KVCacheMetrics:
        return self.block_manager.metrics

    def reset_metrics(self) -> None:
        self.block_manager.reset_metrics()

    def reset_prefix_cache(self) -> None:
        """Reset the prefix cache on both device and host."""
        self.block_manager.reset_prefix_cache()
        self.connector.reset_prefix_cache()
