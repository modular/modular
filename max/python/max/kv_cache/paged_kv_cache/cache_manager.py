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

"""Implements the :class:`PagedKVCacheManager` for managing paged KV cache with data and tensor parallelism."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
from max.driver import CPU, Buffer, Device, DevicePinnedBuffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.interfaces import RequestID, TextGenerationContext
from max.kv_cache.kv_connector import KVConnector
from max.kv_cache.memory_tier import MemoryTier
from max.nn.kv_cache import (
    KVCacheBuffer,
    KVCacheInputs,
    KVCacheParamInterface,
    KVCacheParams,
    MultiKVCacheParams,
)
from max.nn.kv_cache import KVCacheInputsPerDevice as _KVCacheInputsPerDevice
from max.nn.kv_cache.data_parallelism_utils import split_into_groups
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.nn.kv_cache.utils import (
    AttentionDispatchResolver,
    build_max_lengths_tensor,
)
from max.profiler import traced
from max.support.math import ceildiv

from ..connectors import create_connector
from .block_manager import BlockManager, _compute_seq_len

logger = logging.getLogger("max.pipelines")

KVCacheInputsPerDevice = _KVCacheInputsPerDevice[Buffer, Buffer]


#: Padding added to every LUT inner dim (columns per batch row). The SIMD
#: ``populate`` in ``PagedKVCache`` reads up to 16 consecutive ``uint32``
#: entries past ``base_kv_row / page_size``; this buffer keeps those reads
#: in-bounds of the allocation for partial-tile tails. The value is also
#: a multiple of 8 so the inner-dim stride stays 32-byte aligned for the
#: ``ld.global.v{N}.u32`` vector loads.
_LUT_TAIL_PAD = 16


def _padded_lut_cols(cols: int) -> int:
    """Round an LUT inner dim up to a multiple of 8 plus a SIMD tail pad.

    Kept in lockstep with the invariant asserted in
    ``max/kernels/src/kv_cache/types.mojo`` (``PagedKVCache.populate``):
    ``lookup_table.dim[1]`` is a multiple of 8 and is at least
    ``logical_cols + 15`` so a 16-wide SIMD lookup load from any valid
    ``first_lut_idx`` stays in-bounds.
    """
    return ((cols + 7) // 8) * 8 + _LUT_TAIL_PAD


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


class _PersistentKVDeviceInputBuffers:
    """Persistent device buffers backing runtime LUT/cache-length inputs."""

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
        self.lut_table_by_device = []
        self.cache_lengths_by_device = []
        # Pad the inner dim so the SIMD ``populate`` in ``PagedKVCache``
        # can always load up to 16 consecutive uint32s past any valid
        # ``first_lut_idx`` without going OOB of this backing allocation.
        padded_inner = _padded_lut_cols(max_total_num_pages)
        for device in devices:
            self.lut_table_by_device.append(
                Buffer(
                    shape=(max_batch_size, padded_inner),
                    dtype=DType.uint32,
                    device=device,
                )
            )
            self.cache_lengths_by_device.append(
                Buffer(
                    shape=(max_batch_size,),
                    dtype=DType.uint32,
                    device=device,
                )
            )

    def values(self) -> tuple[list[Buffer], list[Buffer]]:
        return (
            self.lut_table_by_device,
            self.cache_lengths_by_device,
        )


@dataclass
class _ReplicaMetadata:
    block_manager: BlockManager
    """Manages allocation, eviction, and reuse of KV cache blocks."""

    connector: KVConnector
    """Connector for external cache tiers (host memory, LMCache, etc.)."""

    persistent_kv_device_input_buffers: _PersistentKVDeviceInputBuffers
    """Persistent device input buffers for the KV cache."""

    device_buffers: list[KVCacheBuffer]
    """Device buffers for each KV cache (length 1 for single-cache models)."""

    devices: Sequence[Device]
    """Devices for the replica."""

    attention_dispatch_resolver: AttentionDispatchResolver
    """Attention dispatch resolver for the replica."""

    claimed_requests: set[RequestID] = field(default_factory=set)
    """Set of request IDs claimed on this replica."""

    # Store last host buffers to ensure lifetimes outlive async copies.
    last_lut_table_host: Buffer | None = None
    last_cache_lengths_host: Buffer | None = None


class PagedKVCacheManager:
    """Paged KVCache manager with data and tensor parallelism support.

    .. code-block:: python

        # Allocate metadata for requests in batch
        kv_manager.claim(ctx1.request_id, replica_idx=0)
        kv_manager.claim(ctx2.request_id, replica_idx=1)

        # Allocate blocks for these requests
        kv_manager.alloc(ctx1, replica_idx=0, num_steps=10)
        kv_manager.alloc(ctx2, replica_idx=1, num_steps=10)

        # Get KVCache inputs to feed to graph
        kv_cache_inputs = kv_manager.runtime_inputs(
            [[ctx1, ctx2]], num_steps=10
        )

        # Run model...
        # Update requests with newly generated tokens
        ctx1.update(42)
        ctx2.update(42)

        # Commit newly written blocks to prefix cache
        kv_manager.step([[ctx1, ctx2]])

        # Release metadata and KV blocks for these requests
        kv_manager.release(ctx1.request_id, replica_idx=0)
        kv_manager.release(ctx2.request_id, replica_idx=1)
    """

    def __init__(
        self,
        params: KVCacheParamInterface,
        session: InferenceSession,
        total_num_pages: int,
        total_num_host_pages: int = 0,
        enable_runtime_checks: bool = False,
        *,
        max_batch_size: int,
        other_kv_managers_device_buffers_per_replica: list[list[Buffer]]
        | None = None,
    ) -> None:
        """Initialize the multi-device paged KV cache manager.

        Args:
            params: KV cache parameters.  Pass ``MultiKVCacheParams`` for
                models with more than one KV cache.
            session: The MAX Engine inference session.
            total_num_pages: The total number of pages to allocate.
            total_num_host_pages: The total number of host pages to allocate.
            max_batch_size: Maximum runtime batch size used to preallocate
                per-replica runtime lookup-table/cache-length row capacity.
            enable_runtime_checks: Whether to enable runtime checks.
            other_kv_managers_device_buffers_per_replica:
                A list of lists of device buffers for other KV managers that should
                be offloaded by this KV manager's KVConnectors. This is a massive
                hack due to the lack of unified KVCache that handles multiple KVs.
        """
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")

        self.params = params

        if isinstance(params, MultiKVCacheParams):
            self._cache_params: list[KVCacheParams] = list(params.params)
        else:
            assert isinstance(params, KVCacheParams)
            self._cache_params = [params]
        self._num_caches = len(self._cache_params)

        primary_params = self._cache_params[0]
        devices = [d.to_device() for d in primary_params.devices]
        self._total_num_pages = total_num_pages
        self._total_num_host_pages = total_num_host_pages
        self._max_batch_size = max_batch_size

        num_replicas = primary_params.data_parallel_degree
        assert len(devices) % num_replicas == 0, (
            "Number of devices must be divisible by number of replicas"
        )
        devices_per_replica = split_into_groups(devices, num_replicas)

        device_memory_tier = (
            MemoryTier.MEMORY_TIER_CPU
            if devices[0].is_host
            else MemoryTier.MEMORY_TIER_GPU
        )

        all_device_buffers: list[list[KVCacheBuffer]] = [
            cp.allocate_buffers(total_num_pages) for cp in self._cache_params
        ]

        self._replica: list[_ReplicaMetadata] = []
        for replica_idx in range(num_replicas):
            replica_devices = devices_per_replica[replica_idx]

            replica_device_buffers = [
                cache_bufs[replica_idx] for cache_bufs in all_device_buffers
            ]
            dispatch_resolver = AttentionDispatchResolver(
                devices=[DeviceRef.from_device(d) for d in replica_devices],
                is_mla=primary_params.is_mla,
                n_kv_heads_per_device=primary_params.n_kv_heads_per_device,
                num_q_heads_per_device=primary_params.num_q_heads_per_device,
                # TODO(SERVOPT-1094): Replace with quantized_kv_cache
                # once SnapMLA uses a valid scale_dtype.
                is_fp8_kv=primary_params.is_fp8_kv_dtype,
            )

            # TODO(SERVOPT-1254): Connector uses primary cache buffer only.
            replica_params = primary_params.copy_as_dp_1(
                replica_idx=replica_idx
            )
            device_buffers_to_offload = replica_device_buffers[0].all_buffers
            if other_kv_managers_device_buffers_per_replica is not None:
                device_buffers_to_offload.extend(
                    other_kv_managers_device_buffers_per_replica[replica_idx]
                )
            connector = create_connector(
                params=replica_params,
                devices=replica_devices,
                device_buffers=device_buffers_to_offload,
                total_num_host_blocks=total_num_host_pages,
                total_num_blocks=total_num_pages,
            )

            persistent_kv_device_input_buffers = (
                _PersistentKVDeviceInputBuffers(
                    max_batch_size=max_batch_size,
                    max_total_num_pages=total_num_pages,
                    devices=replica_devices,
                )
            )

            block_manager = BlockManager(
                device_memory_tier=device_memory_tier,
                total_num_blocks=total_num_pages,
                block_size=primary_params.page_size,
                connector=connector,
                enable_prefix_caching=primary_params.enable_prefix_caching,
                enable_runtime_checks=enable_runtime_checks,
            )

            self._replica.append(
                _ReplicaMetadata(
                    block_manager=block_manager,
                    connector=connector,
                    persistent_kv_device_input_buffers=persistent_kv_device_input_buffers,
                    device_buffers=replica_device_buffers,
                    devices=replica_devices,
                    attention_dispatch_resolver=dispatch_resolver,
                )
            )

    @property
    def num_caches(self) -> int:
        """Number of KV caches managed (1 for single-cache, N for multi)."""
        return self._num_caches

    def cache_params(self, cache_idx: int = 0) -> KVCacheParams:
        """Returns the ``KVCacheParams`` for a specific cache."""
        return self._cache_params[cache_idx]

    def dispatch_resolver(
        self, replica_idx: int = 0
    ) -> AttentionDispatchResolver:
        """Returns the attention dispatch resolver for a replica."""
        return self._replica[replica_idx].attention_dispatch_resolver

    def get_pct_used_blocks_after_allocation(
        self, ctx: TextGenerationContext, replica_idx: int, num_steps: int = 1
    ) -> float:
        """Gets the percentage of blocks used after allocating for a request.

        Args:
            ctx: The request context containing sequence information and token indices.
            replica_idx: Index of the replica to query.
            num_steps: Number of additional steps to allocate blocks for. Defaults to 1.

        Returns:
            The percentage of total blocks used after allocating for the request.
        """
        block_manager = self._replica[replica_idx].block_manager
        num_needed_blocks = (
            self.get_num_used_pages(replica_idx)
            + block_manager.num_blocks_to_allocate(ctx, num_steps)
            - block_manager.count_full_blocks_from_prefix_caches(ctx)
        )
        return min(
            1.0,
            num_needed_blocks / self._total_num_pages,
        )

    def alloc(
        self,
        data: TextGenerationContext,
        replica_idx: int,
        num_steps: int = 1,
    ) -> None:
        """Allocates blocks for a request to run for N steps.

        When prefix caching is enabled, some of the allocated blocks may be
        retrieved from the prefix cache and the context's active token window
        is advanced accordingly.

        Args:
            data: The text generation context for the request. The request ID
                must already be assigned to a replica via ``claim``.
            replica_idx: Index of the replica to allocate on.
            num_steps: The number of steps to reserve blocks for. Default: 1.

        Raises:
            InsufficientBlocksError: If there are insufficient free blocks to
            satisfy the allocation.
        """
        replica = self._replica[replica_idx]
        replica.block_manager.reuse_blocks_from_prefix_cache(data)
        replica.block_manager.allocate_new_blocks(
            data, num_steps, self.params.num_eagle_speculative_tokens
        )

    def _does_req_need_more_blocks(
        self,
        ctx: TextGenerationContext,
        num_steps: int,
        replica_idx: int,
    ) -> bool:
        """Determines if a request needs additional blocks."""
        replica = self._replica[replica_idx]
        block_manager = replica.block_manager
        seq_len = _compute_seq_len(
            ctx,
            num_steps,
            self.params.num_eagle_speculative_tokens,
        )
        num_blocks = len(block_manager.req_to_blocks[ctx.request_id])
        return seq_len > num_blocks * self.params.page_size

    @traced
    def _runtime_inputs_for_replica(
        self,
        replica_idx: int,
        batch: Sequence[TextGenerationContext],
        num_steps: int = 1,
        *,
        max_cache_length: int | None = None,
    ) -> list[KVCacheInputsPerDevice]:
        """Gets runtime inputs for a batch of requests.

        Args:
            replica_idx: Index of the replica to get runtime inputs for.
            batch: Batch of request contexts.
            num_steps: Number of decode steps for the fetch.
            max_cache_length: Optional explicit max cache length to size LUT
                views. If not provided, uses request-derived runtime length.

        Raises:
            ValueError: If a request in ``batch`` is missing allocated blocks,
                if ``batch`` exceeds preallocated runtime capacity, or if
                ``max_cache_length`` implies a LUT shape that is invalid.
        """
        # Wait for any pending connector operations (H2D loads from host cache).
        replica = self._replica[replica_idx]
        replica.connector.sync()

        max_seq_len = 0
        for ctx in batch:
            # Allocate blocks for request if we need more.
            if self._does_req_need_more_blocks(
                ctx,
                num_steps,
                replica_idx=replica_idx,
            ):
                raise ValueError(
                    f"Called runtime_inputs with request {ctx.request_id} but it does not have sufficient blocks. `alloc` must be called first."
                )

            # Compute the total sequence length
            seq_len = _compute_seq_len(
                ctx, num_steps, self.params.num_eagle_speculative_tokens
            )
            max_seq_len = max(max_seq_len, seq_len)

        required_num_pages = ceildiv(max_seq_len, self.params.page_size)
        if max_cache_length is None:
            lut_num_pages = required_num_pages
        else:
            if max_cache_length < 1:
                raise ValueError("max_cache_length must be positive")
            lut_num_pages = ceildiv(max_cache_length, self.params.page_size)
            if lut_num_pages < required_num_pages:
                raise ValueError(
                    "capture max_cache_length cannot be smaller than the "
                    "request-required runtime cache length: "
                    f"{max_cache_length} < {max_seq_len}."
                )

        batch_size = len(batch)
        if batch_size > self._max_batch_size:
            raise ValueError(
                "Runtime batch size exceeds preallocated KV runtime "
                f"buffer capacity: {batch_size} > {self._max_batch_size}."
            )
        if lut_num_pages > self._total_num_pages:
            raise ValueError(
                "Runtime LUT view exceeds allocated page capacity: "
                f"{lut_num_pages} > {self._total_num_pages}."
            )

        # Allocate pinned host staging each invocation so async H2D submissions
        # do not race with subsequent host writes to reused staging buffers.
        device0 = replica.devices[0]

        # Runtime lookup-table shape is [batch_size, padded_lut_num_pages]:
        # rows map to request slots in the current batch and columns map to
        # per-request page slots, padded so the SIMD ``populate`` in
        # ``PagedKVCache`` can safely over-read past any valid
        # ``first_lut_idx``. [0, total_num_pages) are the valid block ids
        # and total_num_pages denotes an unassigned block.
        padded_lut_num_pages = _padded_lut_cols(lut_num_pages)
        if device0.is_host:
            lut_table_host: Buffer = Buffer(
                shape=(batch_size, padded_lut_num_pages),
                dtype=DType.uint32,
                device=device0,
            )
            cache_lengths_host: Buffer = Buffer(
                shape=(batch_size,),
                dtype=DType.uint32,
                device=device0,
            )
        else:
            lut_table_host = DevicePinnedBuffer(
                shape=(batch_size, padded_lut_num_pages),
                dtype=DType.uint32,
                device=device0,
            )
            cache_lengths_host = DevicePinnedBuffer(
                shape=(batch_size,),
                dtype=DType.uint32,
                device=device0,
            )

        runtime_inputs = replica.persistent_kv_device_input_buffers
        # Take a contiguous view of the LUT buffer, which is written to below.
        lut_table_by_device = [
            _contiguous_prefix_2d(
                buffer,
                rows=batch_size,
                cols=padded_lut_num_pages,
            )
            for buffer in runtime_inputs.lut_table_by_device
        ]
        cache_lengths_by_device = [
            buffer[:batch_size]
            for buffer in runtime_inputs.cache_lengths_by_device
        ]

        assert lut_table_host.is_contiguous
        assert cache_lengths_host.is_contiguous
        assert all(buffer.is_contiguous for buffer in lut_table_by_device)

        lut_table_np = lut_table_host.to_numpy()
        lut_table_np.fill(self._total_num_pages)
        cache_lengths_np = cache_lengths_host.to_numpy()
        cache_lengths_np.fill(0)

        # Update cache_lengths and max_lengths.
        max_prompt_len = 0
        max_cached_len = 0
        for batch_idx, ctx in enumerate(batch):
            # Get the blocks for this request.
            blocks = self.get_req_blocks(ctx.request_id, replica_idx)

            # Sanity check that we have enough blocks.
            seq_len = _compute_seq_len(
                ctx, num_steps, self.params.num_eagle_speculative_tokens
            )
            num_required_blocks = ceildiv(seq_len, self.params.page_size)
            assert len(blocks) >= num_required_blocks
            if len(blocks) > num_required_blocks:
                blocks = blocks[:num_required_blocks]

            # Vectorized assignment of block indices to lookup table
            lut_table_np[batch_idx, : len(blocks)] = np.array(
                blocks, dtype=np.uint32
            )

            # Get the existing cache length for this sequence.
            cache_length = ctx.tokens.processed_length + len(
                ctx.spec_decoding_state.maybe_accepted_draft_tokens
            )
            cache_lengths_np[batch_idx] = cache_length

            # Update the maximum lengths seen so far.
            prompt_tokens = ctx.tokens.active_length + len(
                ctx.spec_decoding_state.draft_tokens_to_verify
            )
            max_prompt_len = max(max_prompt_len, prompt_tokens)
            max_cached_len = max(max_cached_len, cache_length + prompt_tokens)

        # Initiate any pending async saves to external cache tiers.
        replica.connector.flush()

        # Build a tensor of maximum lengths. Each step slices the first row to
        # advance to the values for the next row. This should not be allocated
        # on pinned memory since it is exclusively accessed on the CPU and never
        # copied to the GPU.
        absolute_max_cached_len = (
            max_cached_len + self.params.num_eagle_speculative_tokens
        )
        max_lengths_host = build_max_lengths_tensor(
            num_steps,
            max_prompt_len,
            absolute_max_cached_len,
        )
        # Copy shared LUT and cache_lengths to each TP shard's device buffer.
        num_tp_shards = len(replica.devices)
        for tp_shard in range(num_tp_shards):
            cache_lengths_by_device[tp_shard].inplace_copy_from(
                cache_lengths_host
            )
            lut_table_by_device[tp_shard].inplace_copy_from(lut_table_host)

        replica.last_lut_table_host = lut_table_host
        replica.last_cache_lengths_host = cache_lengths_host

        # Keep metadata aligned with kernel-side dispatch inputs.
        # `k.max_context_length()` in flash attention corresponds to the
        # max cached context length for this step (including active prompt
        # tokens), i.e. `max_cached_len` here.
        resolved_metadata = (
            replica.attention_dispatch_resolver.resolve_for_replica(
                batch_size,
                max_prompt_len,
                absolute_max_cached_len,
            )
        )

        if self.params.num_eagle_speculative_tokens > 0:
            draft_resolved_metadata: list[Buffer] | None = (
                replica.attention_dispatch_resolver.resolve_for_replica(
                    batch_size,
                    1,
                    absolute_max_cached_len,
                )
            )
        else:
            draft_resolved_metadata = None

        ret_list: list[KVCacheInputsPerDevice] = []
        for cache_idx in range(self._num_caches):
            device_buffer = replica.device_buffers[cache_idx]

            for tp_shard in range(num_tp_shards):
                metadata = resolved_metadata[tp_shard]
                draft_metadata = (
                    draft_resolved_metadata[tp_shard]
                    if draft_resolved_metadata is not None
                    else None
                )
                block_device = device_buffer.values[tp_shard].device
                if metadata.device not in (CPU(), block_device):
                    raise AssertionError(
                        "attention_dispatch_metadata must be host-resident "
                        "or on the shard device; got "
                        f"{metadata.device} for cache {cache_idx}, shard "
                        f"{tp_shard} on {block_device}."
                    )

                ret_list.append(
                    KVCacheInputsPerDevice(
                        kv_blocks=device_buffer.values[tp_shard],
                        cache_lengths=cache_lengths_by_device[tp_shard],
                        lookup_table=lut_table_by_device[tp_shard],
                        max_lengths=max_lengths_host,
                        kv_scales=(
                            device_buffer.scales[tp_shard]
                            if device_buffer.scales is not None
                            else None
                        ),
                        attention_dispatch_metadata=metadata,
                        draft_attention_dispatch_metadata=draft_metadata,
                    )
                )

        return ret_list

    def runtime_inputs(
        self,
        batches: Sequence[Sequence[TextGenerationContext]],
        num_steps: int = 1,
        *,
        max_cache_length: int | None = None,
    ) -> KVCacheInputs[Buffer, Buffer]:
        """Gets the graph inputs for per-replica batches of requests.

        This method will raise a RuntimeError if any request has insufficient blocks
        already allocated to it to run for the given number of steps.

        Args:
            batches: Per-replica batches of requests
            num_steps: Number of steps to run for
            max_cache_length: Optional explicit max cache length to size LUT
                views. If not provided, uses request-derived runtime length.
        """
        if len(batches) != len(self._replica):
            raise ValueError(
                f"Number of batches must match number of replicas. Expected {len(self._replica)}, got {len(batches)}"
            )
        ret_list: list[KVCacheInputsPerDevice] = []
        for replica_idx, ctxs in enumerate(batches):
            ret_list.extend(
                self._runtime_inputs_for_replica(
                    replica_idx,
                    ctxs,
                    num_steps,
                    max_cache_length=max_cache_length,
                )
            )
        return KVCacheInputs(inputs=ret_list)

    @contextmanager
    def scalar_metadata_on_host(self) -> Iterator[None]:
        """Temporarily keep scalar dispatch metadata on CPU.

        Within this context the attention dispatch resolvers return host
        buffers so that graph-capture replay can perform a single
        CPU-to-GPU ``inplace_copy_from`` instead of a redundant
        GPU-to-GPU copy.
        """
        for replica in self._replica:
            replica.attention_dispatch_resolver.host_only = True
        try:
            yield
        finally:
            for replica in self._replica:
                replica.attention_dispatch_resolver.host_only = False

    def alloc_dummy(
        self,
        request_id: RequestID,
        replica_idx: int,
        sentinel_request_id: RequestID,
    ) -> None:
        """Claims a dummy request and shares the sentinel's block on a replica."""
        self.claim(request_id, replica_idx)
        replica = self._replica[replica_idx]
        replica.block_manager.register_dummy_request(
            request_id, sentinel_request_id
        )

    def release(self, request_id: RequestID, replica_idx: int) -> None:
        """Releases blocks for the request on the given replica."""
        replica = self._replica[replica_idx]
        if request_id not in replica.claimed_requests:
            raise ValueError(
                f"Attempted to release request ID {request_id} but it is not claimed"
            )

        replica.claimed_requests.remove(request_id)

        # Get block IDs before releasing
        block_ids = replica.block_manager.get_req_blocks(request_id)

        # Call the block manager release method with the request_id
        replica.block_manager.release(request_id)

        # Notify connector of request completion
        replica.connector.on_request_complete(request_id, block_ids)

    def claim(self, request_id: RequestID, replica_idx: int) -> None:
        """Reserves a sequence ID for the given request ID."""
        replica = self._replica[replica_idx]
        if request_id in replica.claimed_requests:
            raise ValueError(f"Request ID {request_id} is already claimed")
        replica.claimed_requests.add(request_id)

    @contextmanager
    def reserve(
        self,
        replica_batches: Sequence[Sequence[TextGenerationContext]],
        *,
        num_steps: int = 1,
    ) -> Iterator[None]:
        """Claims, allocates, and releases contexts within a scope.

        This helper is for ephemeral flows (for example, warmup capture) where
        request IDs should be released when leaving the scope.

        Args:
            replica_batches: Per-replica lists of contexts to reserve.
            num_steps: Number of steps to allocate for each context.
        """
        claimed: list[tuple[RequestID, int]] = []
        try:
            for replica_idx, contexts in enumerate(replica_batches):
                for context in contexts:
                    if self.contains(
                        context.request_id, replica_idx=replica_idx
                    ):
                        raise ValueError(
                            "reserve() requires unclaimed request IDs, but "
                            f"{context.request_id!r} is already claimed on "
                            f"replica {replica_idx}."
                        )
                    self.claim(context.request_id, replica_idx=replica_idx)
                    claimed.append((context.request_id, replica_idx))
                    self.alloc(
                        context, replica_idx=replica_idx, num_steps=num_steps
                    )
            yield
        finally:
            for request_id, replica_idx in claimed:
                self.release(request_id, replica_idx=replica_idx)

    def step(self, batches: Sequence[Sequence[TextGenerationContext]]) -> None:
        """Commits new tokens into the prefix cache for per-replica batches."""
        for replica, ctxs in zip(self._replica, batches, strict=True):
            for ctx in ctxs:
                replica.block_manager.step(ctx)

    def contains(self, request_id: RequestID, replica_idx: int) -> bool:
        """Returns whether the request is present on the given replica."""
        replica = self._replica[replica_idx]
        return request_id in replica.claimed_requests

    def reset_metrics(self) -> None:
        """Resets metrics for all replica managers."""
        for replica in self._replica:
            replica.block_manager.reset_metrics()

    def reset_prefix_cache(self) -> None:
        """Resets the prefix cache for all replica managers."""
        for replica in self._replica:
            replica.block_manager.reset_prefix_cache()
            replica.connector.reset_prefix_cache()

    def get_metrics(self, replica_idx: int) -> KVCacheMetrics:
        """Returns metrics for the given replica."""
        replica = self._replica[replica_idx]
        return replica.block_manager.metrics

    def get_req_blocks(
        self, request_id: RequestID, replica_idx: int
    ) -> list[int]:
        """Returns block IDs for the request on the given replica."""
        replica = self._replica[replica_idx]
        return replica.block_manager.get_req_blocks(request_id)

    def get_num_pages(self, replica_idx: int) -> int:
        """Returns total number of pages for the replica."""
        return self._total_num_pages

    def get_num_used_pages(self, replica_idx: int) -> int:
        """Returns number of used pages for the replica."""
        replica = self._replica[replica_idx]
        block_manager = replica.block_manager
        free_blocks = block_manager.device_block_pool.free_blocks
        return self._total_num_pages - len(free_blocks)

    def get_num_host_pages(self, replica_idx: int) -> int:
        """Returns number of host pages for the replica."""
        return self._total_num_host_pages

    def get_num_used_host_pages(self, replica_idx: int) -> int:
        """Returns number of used host pages for the replica."""
        replica = self._replica[replica_idx]
        return replica.connector.num_used_host_blocks

    def get_device_buffer(
        self, replica_idx: int, cache_idx: int = 0
    ) -> KVCacheBuffer:
        """Returns device buffer for a specific cache on a replica.

        Args:
            replica_idx: Index of the replica.
            cache_idx: Index of the cache (default 0 = primary cache).
        """
        replica = self._replica[replica_idx]
        return replica.device_buffers[cache_idx]
