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

"""KV cache manager using a Jenga allocation strategy."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from max.driver import (
    Buffer,
    Device,
    DevicePinnedBuffer,
    copy_pinned_to_destinations,
)
from max.dtype import DType
from max.nn.kv_cache import (
    BatchCharacteristics,
    KVCacheInputs,
    KVCacheInputsInterface,
    KVCacheParamInterface,
)
from max.nn.kv_cache.cache_params import (
    KVCacheAssignments,
    KVCacheBufferInterface,
    KVConnectorType,
)
from max.nn.kv_cache.data_parallelism_utils import split_into_groups
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.nn.kv_cache.utils import build_max_lengths_tensors
from max.pipelines.context import TextContext
from max.pipelines.kv_cache.kv_connector import BlockCount, ByteCount
from max.profiler import traced
from max.support import to_human_readable_bytes
from max.support.math import ceildiv

from .block_manager import PrefixCacheHits, _compute_seq_len
from .cache_manager import (
    _contiguous_prefix_2d,
    _padded_lut_cols,
    cache_valid_length_for_context,
    prompt_tokens_for_context,
)
from .cache_manager_interface import PagedKVCacheManagerInterface
from .jenga_block_manager import JengaBlockManager, KVLeafInfo
from .jenga_block_pool import compute_jenga_ratios

logger = logging.getLogger("max.pipelines")


@dataclass(frozen=True)
class _PersistentKVDeviceInputBuffers:
    """Persistent device buffers backing runtime LUT/cache-length inputs."""

    lut_table_by_device: list[dict[str, Buffer]]
    """LUT on each device."""

    cache_lengths_by_device: list[Buffer]
    """Cache lengths on each device."""

    @classmethod
    def create(
        cls,
        max_batch_size: int,
        max_lut_size: int,
        leaf_infos: Mapping[str, KVLeafInfo],
        devices: Sequence[Device],
    ) -> _PersistentKVDeviceInputBuffers:
        """Creates a _PersistentKVDeviceInputBuffers."""
        lut_table_by_device: list[dict[str, Buffer]] = []
        cache_lengths_by_device: list[Buffer] = []
        # Pad the inner dim so the SIMD ``populate`` in ``PagedKVCache``
        # can always load up to 16 consecutive uint32s past any valid
        # ``first_lut_idx`` without going OOB of this backing allocation.
        padded_inner = _padded_lut_cols(max_lut_size)
        for device in devices:
            lut_table_by_device.append(
                {
                    leaf_id: Buffer(
                        shape=(max_batch_size, padded_inner),
                        dtype=DType.uint32,
                        device=device,
                    )
                    for leaf_id in leaf_infos
                }
            )
            cache_lengths_by_device.append(
                Buffer(
                    shape=(max_batch_size,),
                    dtype=DType.uint32,
                    device=device,
                )
            )

        return cls(
            lut_table_by_device=lut_table_by_device,
            cache_lengths_by_device=cache_lengths_by_device,
        )

    def view(
        self, batch_size: int, lut_num_pages: int
    ) -> tuple[list[dict[str, Buffer]], list[Buffer]]:
        padded_lut_num_pages = _padded_lut_cols(lut_num_pages)
        luts = [
            {
                leaf_id: _contiguous_prefix_2d(
                    buffer, batch_size, padded_lut_num_pages
                )
                for leaf_id, buffer in buffers_per_cache.items()
            }
            for buffers_per_cache in self.lut_table_by_device
        ]
        cache_lengths = [
            buffer[:batch_size] for buffer in self.cache_lengths_by_device
        ]
        assert all(
            all(buffer.is_contiguous for buffer in buffers_per_leaf.values())
            for buffers_per_leaf in luts
        )
        assert all(buffer.is_contiguous for buffer in cache_lengths)
        return luts, cache_lengths


class JengaKVCacheManager(JengaBlockManager, PagedKVCacheManagerInterface):
    """Paged KV cache manager backed by a single fungible memory slab.

    The slab is divided up based on the Jenga two-level huge/little page
    allocation strategy to allow multipple KV types to share the same physical
    memory.
    """

    @classmethod
    def create(
        cls,
        *,
        params: KVCacheParamInterface,
        available_bytes: int,
        max_batch_size: int,
        max_num_input_tokens: int | None = None,
    ) -> JengaKVCacheManager:
        """Creates a JengaKVCacheManager."""
        leaves = params.leaves()
        bytes_per_page = {
            leaf_id: leaf.bytes_per_page for leaf_id, leaf in leaves.items()
        }
        num_huge_blocks, huge_page_bytes, ratios = compute_jenga_ratios(
            available_bytes, bytes_per_page
        )
        leaf_infos = {
            leaf_id: KVLeafInfo(
                ratio=ratios[leaf_id],
                group_id=leaf.group_id,
            )
            for leaf_id, leaf in leaves.items()
        }

        devices = [d.to_device() for d in params.devices]
        slabs = [
            Buffer.zeros(
                shape=(num_huge_blocks, huge_page_bytes),
                dtype=DType.uint8,
                device=d,
            )
            for d in devices
        ]

        kv_buffers = [
            params.slab_to_buffer_views(bs)
            for bs in split_into_groups(slabs, params.data_parallel_degree)
        ]

        logger.info(
            f"Jenga KV manager: {num_huge_blocks} huge pages x {to_human_readable_bytes(huge_page_bytes)} = {to_human_readable_bytes(num_huge_blocks * huge_page_bytes)} (per device), page_size {params.page_size} tokens"
        )
        max_leaf_id_len = max(len(leaf_id) for leaf_id in leaf_infos)
        for leaf_id, leaf_info, leaf in zip(
            leaf_infos.keys(), leaf_infos.values(), leaves.values(), strict=True
        ):
            logger.info(
                f"\t{leaf_id:<{max_leaf_id_len}}: {leaf_info.ratio * num_huge_blocks} pages of {to_human_readable_bytes(leaf.bytes_per_page)}  ({leaf_info.ratio} per huge page)"
            )

        return cls(
            params=params,
            leaf_infos=leaf_infos,
            kv_buffers=kv_buffers,
            num_huge_blocks=num_huge_blocks,
            max_batch_size=max_batch_size,
            max_num_input_tokens=max_num_input_tokens,
        )

    def __init__(
        self,
        *,
        params: KVCacheParamInterface,
        leaf_infos: Mapping[str, KVLeafInfo],
        num_huge_blocks: int,
        kv_buffers: Sequence[KVCacheBufferInterface],
        max_batch_size: int,
        max_num_input_tokens: int | None = None,
    ) -> None:
        # Publicly accessible alias for the params object since it is accessed
        # by callers (e.g. scheduler).
        self.params = params

        if params.kv_connector_config.type != KVConnectorType.null:
            raise ValueError(
                "JengaKVCacheManager does not support KVConnectors."
            )

        self._leaf_infos = leaf_infos
        self._kv_buffers = kv_buffers
        self._num_huge_blocks = num_huge_blocks
        self._max_batch_size = max_batch_size
        self._max_num_input_tokens = max_num_input_tokens

        devices = [d.to_device() for d in self.params.devices]
        devices_per_replica = split_into_groups(
            devices, self.params.data_parallel_degree
        )
        self._staging_devices = [ds[0] for ds in devices_per_replica]
        self._persistent_kv_device_input_buffers = [
            _PersistentKVDeviceInputBuffers.create(
                max_batch_size=max_batch_size,
                max_lut_size=max(
                    leaf_info.ratio * num_huge_blocks
                    for leaf_info in leaf_infos.values()
                ),
                leaf_infos=leaf_infos,
                devices=ds,
            )
            for ds in devices_per_replica
        ]

        super().__init__(
            leaf_infos=leaf_infos,
            num_huge_blocks=num_huge_blocks,
            block_size=self.params.page_size,
            enable_prefix_caching=self.params.enable_prefix_caching,
            num_replicas=self.params.data_parallel_degree,
            kv_hash_algo=self.params.kv_hash_algo,
            kv_hash_seed=self.params.kv_hash_seed,
            max_num_input_tokens=self._max_num_input_tokens,
            num_draft_tokens=self.params.num_draft_tokens,
            num_draft_tokens_per_step=self.params.num_draft_tokens_per_step,
        )

    # ============================================================================
    # Graph Input Preparation
    # ============================================================================

    def runtime_inputs(
        self,
        batches: Sequence[Sequence[TextContext]],
        *,
        max_cache_length: int | None = None,
        batch_characteristics: BatchCharacteristics | None = None,
    ) -> KVCacheInputsInterface[Buffer, Buffer]:
        """Gets the graph inputs for per-replica batches of requests."""
        if len(batches) != self.params.data_parallel_degree:
            raise ValueError(
                f"Number of batches must match number of replicas. Expected {self.params.data_parallel_degree}, got {len(batches)}"
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

    @traced
    def _compute_kv_cache_assignments(
        self,
        *,
        replica_idx: int,
        batch: Sequence[TextContext],
        max_cache_length: int | None = None,
        batch_characteristics: BatchCharacteristics | None = None,
    ) -> KVCacheAssignments:
        max_seq_len = 0
        for ctx in batch:
            seq_len = _compute_seq_len(
                ctx,
                self.params.num_draft_tokens,
                self.params.num_draft_tokens_per_step,
                self._max_num_input_tokens,
            )
            num_blocks = min(
                len(bs) for bs in self.get_req_blocks_per_leaf(ctx).values()
            )
            if seq_len > num_blocks * self.params.page_size:
                raise ValueError(
                    f"Called runtime_inputs with request {ctx.request_id} but it does not have sufficient blocks. `alloc` must be called first."
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

        for leaf_id, leaf_info in self._leaf_infos.items():
            if (
                leaf_info.group_id.is_full()
                and lut_num_pages > self._num_huge_blocks * leaf_info.ratio
            ):
                raise ValueError(
                    f"Runtime LUT view exceeds allocated page capacity for leaf {leaf_id}: "
                    f"{lut_num_pages} > {self._num_huge_blocks * leaf_info.ratio}."
                )

        # Allocate pinned host staging each invocation so async H2D submissions
        # do not race with subsequent host writes to reused staging buffers.

        # Runtime lookup-table shape is [batch_size, padded_lut_num_pages]:
        # rows map to request slots in the current batch and columns map to
        # per-request page slots, padded so the SIMD ``populate`` in
        # ``PagedKVCache`` can safely over-read past any valid
        # ``first_lut_idx``. [0, total_num_pages) are the valid block ids
        # and total_num_pages denotes an unassigned block.
        padded_lut_num_pages = _padded_lut_cols(lut_num_pages)
        shape = (batch_size, padded_lut_num_pages)
        dtype = DType.uint32
        device = self._staging_devices[replica_idx]
        buffer_cls = Buffer if device.is_host else DevicePinnedBuffer
        cache_lengths_host = buffer_cls(
            shape=(batch_size,), dtype=dtype, device=device
        )
        luts, cache_lengths = self._persistent_kv_device_input_buffers[
            replica_idx
        ].view(batch_size, lut_num_pages)

        lut_host_by_leaf = {
            leaf_id: buffer_cls(shape=shape, dtype=dtype, device=device)
            for leaf_id in self._leaf_infos
        }
        lut_np_by_leaf = {
            leaf_id: buffer.to_numpy()
            for leaf_id, buffer in lut_host_by_leaf.items()
        }
        # Fill value is load-bearing: must be 0, Jenga's null-block index.
        for lut_np in lut_np_by_leaf.values():
            lut_np.fill(0)

        cache_lengths_np = cache_lengths_host.to_numpy()
        cache_lengths_np.fill(0)

        # Update cache_lengths and max prompt / cache lengths.
        max_prompt_len = 0
        absolute_max_cached_len = 0
        for batch_idx, ctx in enumerate(batch):
            # Sanity check that we have enough blocks.
            seq_len = _compute_seq_len(
                ctx,
                self.params.num_draft_tokens,
                self.params.num_draft_tokens_per_step,
                self._max_num_input_tokens,
            )
            num_required_blocks = ceildiv(seq_len, self.params.page_size)

            # Get the blocks for this request.
            blocks_per_leaf = self.get_req_blocks_per_leaf(ctx)
            for leaf_id, blocks in blocks_per_leaf.items():
                assert len(blocks) >= num_required_blocks, (
                    f"leaf {leaf_id!r} holds {len(blocks)} blocks, needs"
                    f" {num_required_blocks}"
                )

                # Vectorized assignment of block indices to lookup table
                lut_np_by_leaf[leaf_id][batch_idx, :num_required_blocks] = (
                    blocks[:num_required_blocks]
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
        # The pinned host staging is dropped when this method returns; the
        # memory manager defers its free until the owning device's stream
        # completes, and ``copy_pinned_to_destinations`` makes the owning
        # device wait for the other TP shards so the staging is not recycled
        # while their copies are still reading it.
        copy_pinned_to_destinations(cache_lengths_host, cache_lengths)
        for leaf_id, lut_host in lut_host_by_leaf.items():
            lut_per_device = [lut_per_leaf[leaf_id] for lut_per_leaf in luts]
            copy_pinned_to_destinations(lut_host, lut_per_device)

        return KVCacheAssignments(
            cache_lengths_by_device=cache_lengths,
            lookup_table_by_device=luts,
            max_prompt_length=max_prompt_length_host,
            max_cache_length=max_cache_length_host,
            batch_characteristics=BatchCharacteristics(
                batch_size=batch_size,
                max_prompt_length=max_prompt_len,
                max_cache_valid_length=absolute_max_cached_len,
            ),
        )

    # ============================================================================
    # Metrics
    # ============================================================================

    def get_metrics_aggregated(self) -> KVCacheMetrics:
        """Returns aggregated metrics across all replicas."""
        return self.metrics

    def block_count(self, replica_idx: int = 0) -> BlockCount:
        """Returns the device KV cache block occupancy for the given replica."""
        return self.huge_block_count(replica_idx)

    # ============================================================================
    # KVConnector APIs: Unimplemented as Jenga does not support KVConnector yet
    # ============================================================================

    def pending_transfers_exist(self, replica_idx: int = 0) -> bool:
        """Returns whether there are pending transfers for the replica."""
        return False

    def poll_transfers(self) -> None:
        """Polls for transfers to complete."""
        return

    def host_byte_count(self, replica_idx: int = 0) -> ByteCount:
        """Returns the host KV tier occupancy in bytes for the given replica."""
        return ByteCount(free=0, total=0)

    def disk_byte_count(self, replica_idx: int = 0) -> ByteCount:
        """Returns the disk KV tier occupancy in bytes for the given replica."""
        return ByteCount(free=0, total=0)

    def shutdown(self) -> None:
        """Shuts down the connector."""
        return

    # ============================================================================
    # Misc
    # ============================================================================

    def get_prefix_cache_hit_counts(
        self, ctx: TextContext
    ) -> list[PrefixCacheHits]:
        """Counts each replica's contiguous cached prefix for a request.

        Unimplemented: this only feeds prefix-aware replica selection, which
        fails open to weighting every replica by the full prompt.
        """
        raise NotImplementedError

    def runtime_inputs_for_leaf(
        self,
        batches: Sequence[Sequence[TextContext]],
        *,
        max_cache_length: int | None = None,
        batch_characteristics: BatchCharacteristics | None = None,
    ) -> KVCacheInputs[Buffer, Buffer]:
        """Returns :meth:`runtime_inputs` narrowed to a single leaf cache."""
        inputs = self.runtime_inputs(
            batches,
            max_cache_length=max_cache_length,
            batch_characteristics=batch_characteristics,
        )
        assert isinstance(inputs, KVCacheInputs)
        return inputs

    def get_device_buffer(self, replica_idx: int) -> KVCacheBufferInterface:
        """Returns the device buffer for the given replica."""
        return self._kv_buffers[replica_idx]

    def get_req_blocks(self, ctx: TextContext) -> list[int]:
        """Returns block IDs the request holds on the replica it was claimed on.

        Unimplemented: this interface method assumes one leaf (it backs the
        KVConnector / disaggregated-serving transfer path, which Jenga does
        not support -- see the ``kv_connector`` guard in ``__init__``).
        Jenga's own per-leaf equivalent is :meth:`get_req_blocks_per_leaf`.
        """
        raise NotImplementedError
