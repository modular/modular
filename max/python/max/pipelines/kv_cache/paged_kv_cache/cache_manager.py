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
from max.driver import Buffer, Device, DevicePinnedBuffer
from max.dtype import DType
from max.engine import InferenceSession
from max.nn.kv_cache import (
    BatchCharacteristics,
    KVCacheInputs,
    KVCacheInputsInterface,
    KVCacheParamInterface,
)
from max.nn.kv_cache import KVCacheInputsPerDevice as _KVCacheInputsPerDevice
from max.nn.kv_cache.cache_params import (
    KVCacheAssignments,
    KVCacheBufferInterface,
    KVCacheMemory,
)
from max.nn.kv_cache.data_parallelism_utils import split_into_groups
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.nn.kv_cache.utils import build_max_lengths_tensors
from max.pipelines.context import TextContext
from max.pipelines.kv_cache.kv_connector import KVConnector
from max.pipelines.kv_cache.memory_tier import MemoryTier
from max.pipelines.modeling.types import RequestID
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


def prompt_tokens_for_context(ctx: TextContext) -> int:
    """Returns the per-step query (prompt) width for ``ctx``.

    The new tokens processed this step plus any draft tokens to verify. Matches
    the decode kernel's ``q_max_seq_len`` (prefill: full prompt; decode:
    ``1 + num_draft_tokens_to_verify``).
    """
    return ctx.tokens.active_length + len(
        ctx.spec_decoding_state.draft_tokens_to_verify
    )


def cache_valid_length_for_context(
    ctx: TextContext, num_draft_tokens: int
) -> int:
    """Returns the maximum valid cache length this forward reads for ``ctx``.

    Already-cached tokens (processed + accepted draft) plus this step's query
    width (:func:`prompt_tokens_for_context`) plus the draft tokens written this
    step (``num_draft_tokens``). Shared with the graph-capture replay path so
    its upper-bound characteristics match what this manager prepares.
    """
    return (
        ctx.tokens.processed_length
        + len(ctx.spec_decoding_state.maybe_accepted_draft_tokens)
        + prompt_tokens_for_context(ctx)
        + num_draft_tokens
    )


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
    """The shared block manager (same instance for every replica).

    Stored per replica for backward-compatible access; all replica-scoped calls
    pass the replica index. Device memory is partitioned per replica inside the
    manager.
    """

    connector: KVConnector
    """Connector for external cache tiers (host memory, LMCache, etc.)."""

    persistent_kv_device_input_buffers: _PersistentKVDeviceInputBuffers
    """Persistent device input buffers for the KV cache."""

    devices: Sequence[Device]
    """Devices for the replica."""

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
        kv_manager.alloc(ctx1, replica_idx=0)
        kv_manager.alloc(ctx2, replica_idx=1)

        # Get KVCache inputs to feed to graph
        kv_cache_inputs = kv_manager.runtime_inputs([[ctx1, ctx2]])

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
        """
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")

        self.params = params

        devices = [d.to_device() for d in params.devices]
        self._total_num_pages = total_num_pages
        self._total_num_host_pages = total_num_host_pages
        self._max_batch_size = max_batch_size

        num_replicas = params.data_parallel_degree
        assert len(devices) % num_replicas == 0, (
            "Number of devices must be divisible by number of replicas"
        )
        devices_per_replica = split_into_groups(devices, num_replicas)

        device_memory_tier = (
            MemoryTier.MEMORY_TIER_CPU
            if devices[0].is_host
            else MemoryTier.MEMORY_TIER_GPU
        )

        # Allocate one extra page for the null block.
        self._kv_buffers: Sequence[KVCacheBufferInterface] = (
            params.allocate_buffers(total_num_pages + 1)
        )

        # Per-replica offload-ready KV memory (each replica's device buffers).
        replica_kv_memory = [
            self._kv_buffers[replica_idx].to_memory()
            for replica_idx in range(num_replicas)
        ]

        # A single connector serves every replica; each ``load``/``offload``
        # passes ``replica_idx`` to select the device endpoint.
        self._connector = create_connector(
            kv_connector=params.kv_connector,
            kv_connector_config=params.kv_connector_config,
            devices=devices,
            replica_kv_memory=replica_kv_memory,
            total_num_host_blocks=total_num_host_pages,
            kv_hash_algo=params.kv_hash_algo,
        )

        persistent_buffers: list[_PersistentKVDeviceInputBuffers] = [
            _PersistentKVDeviceInputBuffers(
                max_batch_size=max_batch_size,
                max_total_num_pages=total_num_pages,
                devices=devices_per_replica[replica_idx],
            )
            for replica_idx in range(num_replicas)
        ]

        # When there is more than one replica and prefix caching is enabled, a
        # request admitted on one replica can reuse a prefix block resident on
        # another replica's GPU via a device-to-device copy (SERVOPT-1500). The
        # block manager needs each replica's device memory to perform the copy.
        cross_replica_kv_memory: Sequence[Sequence[KVCacheMemory]] | None = None
        if num_replicas > 1 and params.enable_prefix_caching:
            cross_replica_kv_memory = replica_kv_memory

        # A single block manager owns every replica's device block pool and the
        # single shared connector.
        self._block_manager = BlockManager(
            device_memory_tier=device_memory_tier,
            total_num_blocks=total_num_pages,
            block_size=params.page_size,
            connector=self._connector,
            enable_prefix_caching=params.enable_prefix_caching,
            enable_runtime_checks=enable_runtime_checks,
            num_replicas=num_replicas,
            kv_hash_algo=params.kv_hash_algo,
            kv_hash_seed=params.kv_hash_seed,
            replica_kv_memory=cross_replica_kv_memory,
        )

        self._replica: list[_ReplicaMetadata] = [
            _ReplicaMetadata(
                block_manager=self._block_manager,
                connector=self._connector,
                persistent_kv_device_input_buffers=persistent_buffers[
                    replica_idx
                ],
                devices=devices_per_replica[replica_idx],
            )
            for replica_idx in range(num_replicas)
        ]

    def get_pct_used_blocks_after_allocation(
        self, ctx: TextContext, replica_idx: int
    ) -> float:
        """Gets the percentage of blocks used after allocating for a request.

        Args:
            ctx: The request context containing sequence information and token indices.
            replica_idx: Index of the replica to query.

        Returns:
            The percentage of total blocks used after allocating for the request.
        """
        block_manager = self._block_manager
        num_needed_blocks = (
            self.get_num_used_pages(replica_idx)
            + block_manager.num_blocks_to_allocate(
                ctx,
                self.params.num_draft_tokens,
                self.params.num_draft_tokens_per_step,
            )
            - block_manager.count_full_blocks_from_prefix_caches(
                ctx, replica_idx
            )
        )
        return min(
            1.0,
            num_needed_blocks / self._total_num_pages,
        )

    def alloc(
        self,
        data: TextContext,
        replica_idx: int,
    ) -> None:
        """Allocates blocks for a request.

        When prefix caching is enabled, some of the allocated blocks may be
        retrieved from the prefix cache and the context's active token window
        is advanced accordingly.

        Args:
            data: The text generation context for the request. The request ID
                must already be assigned to a replica via ``claim``.
            replica_idx: Index of the replica to allocate on.

        Raises:
            InsufficientBlocksError: If there are insufficient free blocks to
            satisfy the allocation.
        """
        self._block_manager.reuse_blocks_from_prefix_cache(
            data, replica_idx=replica_idx
        )
        self._block_manager.allocate_new_blocks(
            data,
            self.params.num_draft_tokens,
            self.params.num_draft_tokens_per_step,
            replica_idx=replica_idx,
        )

    def _does_req_need_more_blocks(
        self,
        ctx: TextContext,
        replica_idx: int,
    ) -> bool:
        """Determines if a request needs additional blocks."""
        block_manager = self._block_manager
        seq_len = _compute_seq_len(
            ctx,
            self.params.num_draft_tokens,
            self.params.num_draft_tokens_per_step,
        )
        num_blocks = len(block_manager.req_to_blocks[ctx.request_id])
        return seq_len > num_blocks * self.params.page_size

    @traced
    def _compute_kv_cache_assignments(
        self,
        replica_idx: int,
        batch: Sequence[TextContext],
        *,
        max_cache_length: int | None = None,
        batch_characteristics: BatchCharacteristics | None = None,
    ) -> KVCacheAssignments:
        """Computes KV cache assignments for a batch of requests.

        Args:
            replica_idx: Index of the replica to get runtime inputs for.
            batch: Batch of request contexts.
            max_cache_length: Optional explicit max cache length to size LUT
                views. If not provided, uses request-derived runtime length.
            batch_characteristics: Optional upper-bound batch shape used to
                prepare attention dispatch metadata. When provided, the dispatch
                metadata (and ``max_prompt_length``/``max_cache_length``) is
                resolved from these
                (e.g. graph-capture-aligned) values rather than the batch's real
                values, so the resolved key matches a captured graph. The batch's
                real values must not exceed these. When ``None``, the metadata is
                prepared from the real per-replica values.

        Raises:
            ValueError: If a request in ``batch`` is missing allocated blocks,
                if ``batch`` exceeds preallocated runtime capacity, if
                ``max_cache_length`` implies a LUT shape that is invalid, or if
                the real batch shape exceeds ``batch_characteristics``.
        """
        replica = self._replica[replica_idx]

        max_seq_len = 0
        for ctx in batch:
            # Allocate blocks for request if we need more.
            if self._does_req_need_more_blocks(
                ctx,
                replica_idx=replica_idx,
            ):
                raise ValueError(
                    f"Called runtime_inputs with request {ctx.request_id} but it does not have sufficient blocks. `alloc` must be called first."
                )

            # Compute the total sequence length
            seq_len = _compute_seq_len(
                ctx,
                self.params.num_draft_tokens,
                self.params.num_draft_tokens_per_step,
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
        shape = (batch_size, padded_lut_num_pages)
        dtype = DType.uint32
        device = device0
        buffer_cls = Buffer if device0.is_host else DevicePinnedBuffer
        lut_table_host: Buffer = buffer_cls(
            shape=shape, dtype=dtype, device=device
        )
        cache_lengths_host = buffer_cls(
            shape=(batch_size,), dtype=dtype, device=device
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
        # Fill value is load-bearing: must be exactly `total_num_pages` (the
        # null-block index). The SIMD `populate` path in `PagedKVCache`
        # (types.mojo) multiplies every LUT entry by `page_stride` with no
        # sentinel check, including tail-padding columns it over-reads for
        # SIMD alignment. `total_num_pages * page_stride` resolves to the
        # null-block page, which is in-bounds because `allocate_buffers`
        # allocates N+1 pages. Any other fill value (e.g. a magic constant)
        # computes an out-of-bounds GPU address → CUDA_ERROR_ILLEGAL_ADDRESS.
        lut_table_np.fill(self._total_num_pages)

        cache_lengths_np = cache_lengths_host.to_numpy()
        cache_lengths_np.fill(0)

        # Update cache_lengths and max prompt / cache lengths.
        max_prompt_len = 0
        absolute_max_cached_len = 0
        for batch_idx, ctx in enumerate(batch):
            # Get the blocks for this request.
            blocks = self.get_req_blocks(ctx.request_id, replica_idx)

            # Sanity check that we have enough blocks.
            seq_len = _compute_seq_len(
                ctx,
                self.params.num_draft_tokens,
                self.params.num_draft_tokens_per_step,
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

            # Update the maximum lengths seen so far. The shared helpers keep
            # this in lockstep with the graph-capture replay path's
            # upper-bound characteristics.
            max_prompt_len = max(max_prompt_len, prompt_tokens_for_context(ctx))
            absolute_max_cached_len = max(
                absolute_max_cached_len,
                cache_valid_length_for_context(
                    ctx, self.params.num_draft_tokens
                ),
            )

        # Order in-flight loads before the forward pass, since this runs before
        # the model executes. For dKV's off-stream remote (NIXL) READs this is a
        # host wait until they land; for dKV's same-host loads it enqueues a
        # cross-stream CUDA event wait so the compute stream is GPU-ordered after
        # the copies (no host block). No-op for host/disk tiers.
        replica.connector.wait_for_loads()

        # Initiate saves to external cache tiers.
        self._block_manager.offload(replica_idx)

        # Choose the shape used to prepare attention dispatch metadata. When
        # ``batch_characteristics`` is provided (e.g. graph-capture replay), the
        # dispatch key is resolved once from those (aligned, upper-bound) values
        # so it matches a captured graph; otherwise the real per-replica values
        # are used. LUT / cache_lengths always use the real values; only the
        # dispatch metadata and ``max_prompt_length`` / ``max_cache_length``
        # follow ``dispatch_*``.
        if batch_characteristics is not None:
            bc = batch_characteristics
            if (
                batch_size > bc.batch_size
                or max_prompt_len > bc.max_prompt_length
                or absolute_max_cached_len > bc.max_cache_valid_length
            ):
                raise ValueError(
                    f"Real batch size ({batch_size}) exceeds the requested dispatch batch size ({bc.batch_size})."
                )
            batch_size = bc.batch_size
            max_prompt_len = bc.max_prompt_length
            absolute_max_cached_len = bc.max_cache_valid_length

        max_prompt_length_host, max_cache_length_host = (
            build_max_lengths_tensors(
                max_prompt_len,
                absolute_max_cached_len,
            )
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

        return KVCacheAssignments(
            cache_lengths_by_device=cache_lengths_by_device,
            lookup_table_by_device=lut_table_by_device,
            max_prompt_length=max_prompt_length_host,
            max_cache_length=max_cache_length_host,
            batch_characteristics=BatchCharacteristics(
                batch_size=batch_size,
                max_prompt_length=max_prompt_len,
                max_cache_valid_length=absolute_max_cached_len,
            ),
        )

    def runtime_inputs(
        self,
        batches: Sequence[Sequence[TextContext]],
        *,
        max_cache_length: int | None = None,
        batch_characteristics: BatchCharacteristics | None = None,
    ) -> KVCacheInputsInterface[Buffer, Buffer]:
        """Gets the graph inputs for per-replica batches of requests.

        Returns a single ``KVCacheInputs`` leaf (or ``MultiKVCacheInputs``
        tree for multi-cache models) whose leaves hold every
        ``(DP replica, TP shard)`` device's inputs.

        This method will raise a RuntimeError if any request has insufficient blocks
        already allocated to it.

        Args:
            batches: Per-replica batches of requests
            max_cache_length: Optional explicit max cache length to size LUT
                views. If not provided, uses request-derived runtime length.
            batch_characteristics: Optional upper-bound batch shape applied
                uniformly across every replica when preparing attention dispatch
                metadata. When provided (e.g. graph-capture replay, where every
                DP replica must run the identical captured graph), the dispatch
                key is resolved once from these aligned values; the real
                per-replica values must not exceed them. When ``None``, each
                replica prepares metadata from its own real values (which may
                differ per replica).
        """
        if len(batches) != len(self._replica):
            raise ValueError(
                f"Number of batches must match number of replicas. Expected {len(self._replica)}, got {len(batches)}"
            )
        assignments = [
            self._compute_kv_cache_assignments(
                replica_idx=replica_idx,
                batch=ctxs,
                max_cache_length=max_cache_length,
                batch_characteristics=batch_characteristics,
            )
            for replica_idx, ctxs in enumerate(batches)
        ]
        return self.params.build_runtime_inputs(assignments, self._kv_buffers)

    def runtime_inputs_for_leaf(
        self,
        batches: Sequence[Sequence[TextContext]],
        *,
        max_cache_length: int | None = None,
        batch_characteristics: BatchCharacteristics | None = None,
    ) -> KVCacheInputs[Buffer, Buffer]:
        """Returns :meth:`runtime_inputs` narrowed to a single leaf cache.

        Convenience wrapper for single-cache (non-tree) models: it asserts the
        result is a :class:`KVCacheInputs` leaf and returns it, so callers can
        access ``.inputs`` directly without narrowing the
        :class:`KVCacheInputsInterface` themselves. Raises ``AssertionError``
        for tree (``MultiKVCacheInputs``) models.
        """
        inputs = self.runtime_inputs(
            batches,
            max_cache_length=max_cache_length,
            batch_characteristics=batch_characteristics,
        )
        assert isinstance(inputs, KVCacheInputs)
        return inputs

    def alloc_dummy(self, request_id: RequestID, replica_idx: int) -> None:
        """Claims a dummy request and maps it to the replica's null block."""
        self.claim(request_id, replica_idx)
        self._block_manager.register_dummy_request(request_id, replica_idx)

    def num_free_blocks(self, replica_idx: int = 0) -> int:
        """Returns the number of free KV cache blocks on the given replica."""
        return len(
            self._block_manager.device_block_pools[replica_idx].free_block_queue
        )

    def total_num_blocks(self, replica_idx: int = 0) -> int:
        """Returns the total number of KV cache blocks on the given replica."""
        return self._block_manager.total_num_blocks

    def release(self, request_id: RequestID, replica_idx: int) -> None:
        """Releases blocks for the request on the given replica."""
        replica = self._replica[replica_idx]
        if request_id not in replica.claimed_requests:
            raise ValueError(
                f"Attempted to release request ID {request_id} but it is not claimed"
            )

        replica.claimed_requests.remove(request_id)

        # Call the block manager release method with the request_id
        self._block_manager.release(request_id, replica_idx)

    def claim(self, request_id: RequestID, replica_idx: int) -> None:
        """Reserves a sequence ID for the given request ID."""
        replica = self._replica[replica_idx]
        if request_id in replica.claimed_requests:
            raise ValueError(f"Request ID {request_id} is already claimed")
        replica.claimed_requests.add(request_id)

    @contextmanager
    def reserve(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
    ) -> Iterator[None]:
        """Claims, allocates, and releases contexts within a scope.

        This helper is for ephemeral flows (for example, warmup capture) where
        request IDs should be released when leaving the scope.

        Args:
            replica_batches: Per-replica lists of contexts to reserve.
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
                    self.alloc(context, replica_idx=replica_idx)
            yield
        finally:
            for request_id, replica_idx in claimed:
                self.release(request_id, replica_idx=replica_idx)

    def step(self, batches: Sequence[Sequence[TextContext]]) -> None:
        """Commits new tokens into the prefix cache for per-replica batches."""
        for replica_idx, (replica, ctxs) in enumerate(
            zip(self._replica, batches, strict=True)
        ):
            # Drain offloads posted this iteration (post-forward-pass): the
            # host/disk tiers wait on their D2H copies and post disk writes; the
            # dKV connector awaits its NIXL WRITEs and registers the blocks.
            replica.connector.wait_for_offloads()
            for ctx in ctxs:
                self._block_manager.step(ctx, replica_idx)

    def contains(self, request_id: RequestID, replica_idx: int) -> bool:
        """Returns whether the request is present on the given replica."""
        replica = self._replica[replica_idx]
        return request_id in replica.claimed_requests

    def reset_metrics(self) -> None:
        """Resets metrics for the block manager."""
        self._block_manager.reset_metrics()

    def reset_prefix_cache(self) -> None:
        """Resets the device prefix caches and every connector's tiers."""
        self._block_manager.reset_prefix_cache()
        for replica in self._replica:
            replica.connector.reset_prefix_cache()

    def get_metrics_aggregated(self) -> KVCacheMetrics:
        """Returns aggregated metrics across all replicas."""
        return self._block_manager.metrics

    def get_req_blocks(
        self, request_id: RequestID, replica_idx: int
    ) -> list[int]:
        """Returns block IDs for the request on the given replica."""
        return self._block_manager.get_req_blocks(request_id, replica_idx)

    def get_num_pages(self, replica_idx: int) -> int:
        """Returns total number of pages for the replica."""
        return self._total_num_pages

    def get_num_used_pages(self, replica_idx: int) -> int:
        """Returns number of used pages for the replica."""
        free_blocks = self._block_manager.device_block_pools[
            replica_idx
        ].free_blocks
        return self._total_num_pages - len(free_blocks)

    def get_num_host_pages(self, replica_idx: int) -> int:
        """Returns number of host pages for the replica."""
        return self._total_num_host_pages

    def get_num_used_host_pages(self, replica_idx: int) -> int:
        """Returns number of used host pages for the replica."""
        replica = self._replica[replica_idx]
        return replica.connector.num_used_host_blocks

    def get_num_disk_pages(self, replica_idx: int) -> int:
        """Returns number of disk pages for the replica."""
        replica = self._replica[replica_idx]
        return replica.connector.num_disk_blocks

    def get_num_used_disk_pages(self, replica_idx: int) -> int:
        """Returns number of used disk pages for the replica."""
        replica = self._replica[replica_idx]
        return replica.connector.num_used_disk_blocks

    def get_device_buffer(self, replica_idx: int) -> KVCacheBufferInterface:
        """Returns the replica's KV buffer (single leaf or tree).

        HACK: this exists only for the transfer engine; callers flatten via
        :attr:`KVCacheBufferInterface.all_buffers`.
        """
        return self._kv_buffers[replica_idx]
