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
import queue
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, TensorType
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
        total_num_pages: int,
        total_num_host_pages: int,
        max_batch_size: int,
        max_seq_len: int,
        devices: Sequence[Device],
        session: InferenceSession,
        zmq_endpoint_base: str | None = None,
        enable_runtime_checks: bool = False,
    ) -> None:
        """Initialize the tensor-parallel paged KV cache manager.

        Args:
            params: The KVCacheParams for the given pipeline.
            devices: The devices on which the manager will allocate memory.
                For tensor parallelism, KV cache data is sharded across these devices.
            session: The inference session to load ops from.
            enable_runtime_checks: Whether to enable runtime correctness checks.
        """
        self.params = params
        self.total_num_pages = total_num_pages
        self.total_num_host_pages = total_num_host_pages
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.page_size = params.page_size
        self.devices = devices
        self.session = session

        # Validate devices aligns with the n_devices in params
        if len(devices) != params.n_devices:
            raise ValueError(
                "n_devices provided in KVCacheParams, does not match number of devices initialized in the _TPPagedKVCacheManager"
            )

        if params.data_parallel_degree > 1:
            raise ValueError(
                "_TPPagedKVCacheManager does not support data parallelism."
            )

        # Track the set of requests that are currently claimed.
        self._claimed_requests: set[RequestID] = set()

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
                    shape=[total_num_pages, *params.shape_per_block],
                    dtype=self.params.dtype,
                    device=device,
                )
            )

        self.host_tensors = None
        if params.enable_kvcache_swapping_to_host:
            self.host_tensors = []
            # Construct host tensors for each device.
            for dev in devices:
                if dev.is_host:
                    raise ValueError(
                        "Host device detected. Paging to host is not supported when executing on CPU."
                    )
                self.host_tensors.append(
                    Tensor(
                        shape=[total_num_host_pages, *params.shape_per_block],
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
        max_total_num_pages = ceildiv(max_seq_len, self.page_size)
        batch_size = len(batch)
        lut_table_np = np.full(
            (batch_size, max_total_num_pages),
            self.total_num_pages,
            dtype=np.uint32,
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
            num_layers = self.params.num_layers

        return [
            PagedCacheInputSymbols(
                kv_blocks=BufferType(
                    self.params.dtype,
                    shape=[
                        "total_num_pages",
                        *self.params.shape_per_block,
                    ],
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
        if request_id not in self._claimed_requests:
            raise ValueError(
                f"Attempted to release request ID {request_id} but it is not claimed"
            )

        self._claimed_requests.remove(request_id)

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

    def claim(self, request_id: RequestID) -> None:
        """Reserve a sequence ID for the given request ID."""
        if request_id in self._claimed_requests:
            raise ValueError(f"Request ID {request_id} is already claimed")
        if len(self._claimed_requests) == self.max_batch_size:
            raise ValueError(
                f"Unable to claim request ID {request_id} due to batch size limit."
            )
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
