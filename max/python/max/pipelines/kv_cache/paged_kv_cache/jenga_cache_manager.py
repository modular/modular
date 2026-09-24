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

import numpy as np
from max import tree
from max.driver import Buffer
from max.dtype import DType
from max.nn.kv_cache import (
    BatchCharacteristics,
    KVCacheInputs,
    KVCacheInputsPerDevice,
    KVCacheParamInterface,
)
from max.nn.kv_cache.cache_params import (
    KVCacheAssignments,
    KVCacheBufferInterface,
    KVCacheMemory,
    KVConnectorType,
    spec_decode_cache_slack,
)
from max.nn.kv_cache.data_parallelism_utils import split_into_groups
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.nn.kv_cache.utils import (
    build_max_lengths_tensors,
)
from max.pipelines.context import TextContext
from max.pipelines.graph_input_stager import (
    GraphInputStager,
    GraphInputStaging,
    InputDescriptor,
)
from max.pipelines.kv_cache.kv_connector import (
    BlockCount,
    ByteCount,
    KVConnector,
)
from max.profiler import traced
from max.support import to_human_readable_bytes
from max.support.math import ceildiv

from ..connectors import create_connector
from .cache_manager import (
    KV_CACHE_LENGTHS,
    cache_valid_length_for_context,
    prompt_tokens_for_context,
)
from .cache_manager_interface import PagedKVCacheManagerInterface
from .jenga_block_manager import (
    JengaBlockManager,
    KVLeafInfo,
    _max_seq_len_fitting_in_geometry,
    create_groups,
    create_pools,
)
from .jenga_block_pool import (
    JengaBlockPool,
    JengaGeometry,
    plan_jenga_geometry,
)
from .kv_group_coordinator import KVGroupCoordinatorInterface

logger = logging.getLogger("max.pipelines")


class JengaKVCacheManager(JengaBlockManager, PagedKVCacheManagerInterface):
    """Paged KV cache manager backed by a single fungible memory slab.

    The slab is divided up based on the Jenga two-level huge/little page
    allocation strategy to allow multipple KV types to share the same physical
    memory.
    """

    @staticmethod
    def _plan_geometry(
        params: KVCacheParamInterface, available_bytes: int
    ) -> JengaGeometry:
        """Fits one device's share of the budget to a huge-block geometry."""
        leaves = params.leaves()
        # A leaf reports one device's page, which is what a slab tiles in.
        bytes_per_page = {
            leaf_id: leaf.bytes_per_page for leaf_id, leaf in leaves.items()
        }
        # A row-addressed leaf reports 1 and so constrains nothing.
        row_bytes = {
            leaf_id: leaf.row_bytes for leaf_id, leaf in leaves.items()
        }
        n_devices = len(params.devices)
        if n_devices < 1:
            raise ValueError("Jenga KV cache requires at least one device")
        return plan_jenga_geometry(
            available_bytes // n_devices, bytes_per_page, row_bytes
        )

    @classmethod
    def max_seq_len_fitting_in_cache(
        cls, params: KVCacheParamInterface, available_bytes: int
    ) -> int | None:
        """Returns the longest request a slab of ``available_bytes`` holds one of.

        Plans the geometry :meth:`create` allocates for the same budget, so
        the answer is a length ``create`` accepts. Net of the
        speculative-decode slack a request may occupy past its cap.

        Returns:
            The longest sequence, or ``None`` when no length exhausts the slab.
        """
        geometry = cls._plan_geometry(params, available_bytes)
        longest = _max_seq_len_fitting_in_geometry(
            params.leaves(),
            params.page_size,
            geometry.num_huge_blocks - 1,  # Huge block 0 is the null block.
            geometry.ratios,
        )
        if longest is None:
            return None
        return max(1, longest - spec_decode_cache_slack(params))

    @classmethod
    def create(
        cls,
        *,
        params: KVCacheParamInterface,
        available_bytes: int,
        max_batch_size: int,
        max_num_input_tokens: int | None = None,
        max_seq_len: int | None = None,
    ) -> JengaKVCacheManager:
        """Creates a JengaKVCacheManager.

        ``available_bytes`` is the KV budget across all devices from memory
        estimation (same contract as ``compute_num_device_blocks``). Each
        device slab is sized from ``available_bytes // len(params.devices)``.
        """
        leaves = params.leaves()
        bytes_per_page = {
            leaf_id: leaf.bytes_per_page for leaf_id, leaf in leaves.items()
        }
        is_kv_connector_enabled = (
            params.kv_connector_config.type.value != "null"
        )
        has_unservable_leaf = any(
            leaf.group_id.is_recurrent() for leaf in leaves.values()
        )
        if is_kv_connector_enabled and has_unservable_leaf:
            raise ValueError(
                "Recurrent KV cache group is incompatible with KVConnector."
                " Please disable KVConnector"
            )
        geometry = cls._plan_geometry(params, available_bytes)
        num_huge_blocks = geometry.num_huge_blocks
        huge_page_bytes = geometry.huge_page_bytes
        ratios = geometry.ratios
        leaf_infos = {
            leaf_id: KVLeafInfo(ratio=ratios[leaf_id], group_id=leaf.group_id)
            for leaf_id, leaf in leaves.items()
        }

        logger.info(
            f"Jenga KV manager: {num_huge_blocks} huge pages x {to_human_readable_bytes(huge_page_bytes)} = {to_human_readable_bytes(num_huge_blocks * huge_page_bytes)} (per device), page_size {params.page_size} tokens"
        )
        max_leaf_id_len = max(len(leaf_id) for leaf_id in leaf_infos)
        for leaf_id, leaf_info in leaf_infos.items():
            logger.info(
                f"\t{leaf_id:<{max_leaf_id_len}}: {leaf_info.ratio * num_huge_blocks} pages of {to_human_readable_bytes(geometry.padded_sizes[leaf_id])}  ({leaf_info.ratio} per huge page)"
                + (
                    ""
                    if geometry.padded_sizes[leaf_id] == bytes_per_page[leaf_id]
                    else f", padded from {to_human_readable_bytes(bytes_per_page[leaf_id])}"
                    f" (+{geometry.padding_fraction(leaf_id, bytes_per_page[leaf_id]):.2%})"
                )
            )

        devices = [d.to_device() for d in params.devices]
        slabs = [
            Buffer.zeros(
                shape=(num_huge_blocks, huge_page_bytes),
                dtype=DType.uint8,
                device=d,
            )
            for d in devices
        ]

        # The views carry their page stride, so nothing below here needs the
        # padded sizes.
        kv_buffers = [
            params.slab_to_buffer_views(bs, geometry.padded_sizes)
            for bs in split_into_groups(slabs, params.data_parallel_degree)
        ]

        pools = create_pools(
            leaf_infos, num_huge_blocks, params.data_parallel_degree
        )

        groups = create_groups(leaf_infos, pools, params.page_size)

        # A single connector serves every replica; each load/offload passes
        # the replica_idx that selects the device endpoint.
        replica_kv_memory = [buf.to_memory() for buf in kv_buffers]
        connector = create_connector(
            # A leaf carrying no hash has nothing an external tier can key on,
            # so the connector is never told about it.
            leaves={
                leaf_id: leaf.group_id
                for leaf_id, leaf in leaves.items()
                if leaf.cacheable
            },
            devices=devices,
            replica_kv_memory=replica_kv_memory,
            params=params,
            device_memory_bytes=num_huge_blocks * huge_page_bytes,
        )

        manager = cls(
            params=params,
            leaf_infos=leaf_infos,
            kv_buffers=kv_buffers,
            replica_kv_memory=replica_kv_memory,
            max_batch_size=max_batch_size,
            max_num_input_tokens=max_num_input_tokens,
            connector=connector,
            pools=pools,
            groups=groups,
            slabs=slabs,
        )
        if max_seq_len is not None:
            slack = spec_decode_cache_slack(params)
            seq_len_with_slack = max_seq_len + slack
            if not manager._fits_in_cache(seq_len_with_slack):
                effective = manager.effective_max_seq_length
                max_tokens = effective if effective is not None else 0
                slack_str = (
                    f" (plus {slack} speculative-decode slack tokens)"
                    if slack > 0
                    else ""
                )
                raise RuntimeError(
                    "Insufficient cache memory to support a batch containing one"
                    f" request at the max sequence length of {max_seq_len} tokens"
                    f"{slack_str}. A request approaching the max sequence length would"
                    " exhaust the KV cache and crash the model worker. Reduce"
                    f" --max-length to at most {max_tokens} or increase the available"
                    " KV cache memory (e.g. raise --device-memory-utilization)."
                )
        return manager

    def __init__(
        self,
        *,
        params: KVCacheParamInterface,
        leaf_infos: Mapping[str, KVLeafInfo],
        pools: Sequence[JengaBlockPool],
        kv_buffers: Sequence[KVCacheBufferInterface],
        replica_kv_memory: Sequence[Mapping[str, KVCacheMemory]],
        max_batch_size: int,
        max_num_input_tokens: int | None = None,
        connector: KVConnector | None = None,
        groups: Mapping[str, KVGroupCoordinatorInterface],
        slabs: Sequence[Buffer] = (),
    ) -> None:
        # Publicly accessible alias for the params object since it is accessed
        # by callers (e.g. scheduler).
        self.params = params
        # `create_connector` hands back a NullConnector when none is
        # configured; treat that as "no connector" so the guard and the
        # transfer paths below stay inert for the common case.
        if params.kv_connector_config.type == KVConnectorType.null:
            connector = None
        self._connector = connector

        # One zeroed buffer per (replica, device, bound input), shaped like a
        # single block's rows. Filled on first use and never written after.
        self._zero_rows: dict[tuple[int, int, str], Buffer] = {}

        self._leaf_infos = leaf_infos
        self._num_huge_blocks = pools[0].num_huge_blocks
        self._kv_buffers = kv_buffers
        self._max_batch_size = max_batch_size
        self._max_num_input_tokens = max_num_input_tokens

        devices = [d.to_device() for d in self.params.devices]
        devices_per_replica = split_into_groups(
            devices, self.params.data_parallel_degree
        )
        self._devices_per_replica = devices_per_replica

        leaves = params.leaves()
        max_num_blocks = max(
            leaf_info.ratio * self._num_huge_blocks
            for leaf_info in leaf_infos.values()
        )
        # One instance across every leaf, shard and replica. Leaf ids are
        # already unique, a replica's shards are the destinations of one
        # name, and the replica scopes the name because two replicas can be
        # handed the same device.
        #
        # Declared at the widest a forward can ask, because a graph input
        # that moved would have to be rebound. Batch-major, so a step's
        # slice of one is contiguous.
        # A row-addressed view onto the slab is a graph input the cache
        # already owns: bound once, never staged, and never rewritten by a
        # forward. Per replica, keyed the way its leaves read it.
        self._bound: list[Mapping[str, list[Buffer]]] = [
            params.slab_to_bound_views(replica_slabs)
            for replica_slabs in split_into_groups(
                slabs, self.params.data_parallel_degree
            )
        ]
        described: list[InputDescriptor] = []
        for replica_idx, replica_devices in enumerate(devices_per_replica):
            described.append(
                InputDescriptor(
                    name=f"{replica_idx}/{KV_CACHE_LENGTHS}",
                    dtype=DType.uint32,
                    max_shape=(max_batch_size,),
                    destinations=replica_devices,
                )
            )
            # Metadata the kernel reads on the device is a transfer like
            # any other, so it rides the forward's.
            described.extend(
                InputDescriptor(
                    name=f"{replica_idx}/{name}",
                    dtype=spec.dtype,
                    max_shape=spec.shape,
                    destinations=replica_devices,
                )
                for name, spec in params.staged_dispatch_metadata().items()
            )
            for leaf in leaves.values():
                described.extend(
                    InputDescriptor(
                        name=f"{replica_idx}/{key}",
                        dtype=dtype,
                        max_shape=shape,
                        destinations=replica_devices,
                    )
                    for key, (shape, dtype) in leaf.staged_input_shapes(
                        max_batch_size, max_num_blocks
                    ).items()
                )
        self._stager = GraphInputStager(described)

        super().__init__(
            pools=pools,
            block_size=self.params.page_size,
            enable_prefix_caching=self.params.enable_prefix_caching,
            kv_hash_algo=self.params.kv_hash_algo,
            kv_hash_seed=self.params.kv_hash_seed,
            max_num_input_tokens=self._max_num_input_tokens,
            num_draft_tokens=self.params.num_draft_tokens,
            num_draft_tokens_per_step=self.params.num_draft_tokens_per_step,
            connector=connector,
            replica_kv_memory=replica_kv_memory,
            enable_dp_cross_replica_prefix_copy=(
                self.params.enable_dp_cross_replica_prefix_copy
            ),
            groups=groups,
            leaves=leaves,
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
    ) -> KVCacheInputs[Buffer, Buffer]:
        """Gets the graph inputs for per-replica batches of requests."""
        if len(batches) != self.params.data_parallel_degree:
            raise ValueError(
                f"Number of batches must match number of replicas. Expected {self.params.data_parallel_degree}, got {len(batches)}"
            )

        # A row-addressed view is a graph input, and no request means no
        # rows to bind it with.
        binds_rows = any(self._bound)
        if binds_rows and not any(batches):
            raise ValueError("runtime_inputs called with an empty batch")

        if self._connector is not None:
            # Pre-forward load barrier (dKV-only): dKV posts its READs in
            # `load` and orders them here. Asynchronous connectors instead hold
            # a request out of the batch until its onload polls complete, so
            # this is a no-op for them.
            self._connector.wait_for_loads()
            for replica_idx in range(len(batches)):
                # Initiate saves of everything committed since the last forward.
                self.offload(replica_idx)

        # One scope per forward: every replica stages into it, and leaving
        # it sends the lot.
        with self._stager.stage() as staging:
            assignments = [
                self._compute_kv_cache_assignments(
                    staging,
                    replica_idx=replica_idx,
                    batch=ctxs,
                    max_cache_length=max_cache_length,
                    batch_characteristics=batch_characteristics,
                )
                for replica_idx, ctxs in enumerate(batches)
            ]
            # Built inside the scope because the dispatch metadata is staged
            # down there: its transfer folds into this forward's rather than
            # being one of its own.
            inputs = self.params.build_runtime_inputs(
                assignments, self._kv_buffers, staging=staging
            )
        self._resume_state(batches)
        return inputs

    @traced
    def _compute_kv_cache_assignments(
        self,
        staging: GraphInputStaging,
        *,
        replica_idx: int,
        batch: Sequence[TextContext],
        max_cache_length: int | None = None,
        batch_characteristics: BatchCharacteristics | None = None,
    ) -> KVCacheAssignments:
        # How far into each request's row this forward reaches, reused below
        # to size the staging and to tell each group where its rows end.
        num_blocks = [self._num_required_blocks(ctx) for ctx in batch]
        for ctx, required in zip(batch, num_blocks, strict=True):
            # Owing nothing is what "alloc has run" means. Every leaf's row is
            # `num_blocks` slots long; how many hold a distinct block is a
            # separate question the leaf answers.
            outstanding = {
                leaf_id: count
                for group in self._groups.values()
                for leaf_id, count in group.blocks_to_allocate(
                    ctx.request_id, required
                ).items()
                if count
            }
            if outstanding:
                raise ValueError(
                    f"Called runtime_inputs with request {ctx.request_id} but"
                    " it does not have sufficient blocks: it is short"
                    f" {outstanding}. `alloc` must be called first."
                )

        required_num_blocks = max(num_blocks, default=0)
        if max_cache_length is None:
            lut_num_blocks = required_num_blocks
        else:
            if max_cache_length < 1:
                raise ValueError("max_cache_length must be positive")
            lut_num_blocks = ceildiv(max_cache_length, self.params.page_size)
            if lut_num_blocks < required_num_blocks:
                raise ValueError(
                    "capture max_cache_length cannot be smaller than the "
                    "request-required runtime cache length: "
                    f"{lut_num_blocks} < {required_num_blocks} pages."
                )

        batch_size = len(batch)
        if batch_size > self._max_batch_size:
            raise ValueError(
                "Runtime batch size exceeds preallocated KV runtime "
                f"buffer capacity: {batch_size} > {self._max_batch_size}."
            )

        replica_devices = self._devices_per_replica[replica_idx]
        cache_lengths_host, cache_lengths_by_device = staging.get(
            f"{replica_idx}/{KV_CACHE_LENGTHS}", (batch_size,)
        )
        cache_lengths_np = cache_lengths_host.to_numpy()
        cache_lengths_np.fill(0)

        plans: dict[str, list[list[int]]] = {}
        for group in self._groups.values():
            plans.update(group.forward_blocks(batch, num_blocks))

        staged: dict[str, tuple[Buffer, ...]] = {}
        for leaf_id, leaf in self._leaves.items():
            into: dict[str, np.ndarray] = {}
            for key, (shape, _) in leaf.staged_input_shapes(
                batch_size, lut_num_blocks
            ).items():
                # Narrowed to the shape the group writes, whatever it is.
                host, staged[key] = staging.get(f"{replica_idx}/{key}", shape)
                into[key] = host.to_numpy()
            leaf.write_staged_inputs(plans[leaf_id], into)

        # Update cache_lengths and max prompt / cache lengths.
        max_prompt_len = 0
        absolute_max_cached_len = 0
        for batch_idx, ctx in enumerate(batch):
            # Get the existing cache length for this sequence.
            cache_length = ctx.tokens.processed_length + len(
                ctx.spec_decoding_state.maybe_accepted_draft_tokens
            )
            cache_lengths_np[batch_idx] = cache_length

            # Update the maximum lengths seen so far. The shared helpers
            # keep this in lockstep with the graph-capture replay path's
            # upper-bound characteristics.
            max_prompt_len = max(max_prompt_len, prompt_tokens_for_context(ctx))
            absolute_max_cached_len = max(
                absolute_max_cached_len,
                cache_valid_length_for_context(
                    ctx, self.params.num_draft_tokens
                ),
            )

        # Choose the shape used to prepare attention dispatch metadata.
        # When ``batch_characteristics`` is provided (e.g. graph-capture
        # replay), the dispatch key is resolved once from those (aligned,
        # upper-bound) values so it matches a captured graph; otherwise
        # the real per-replica values are used. LUT / cache_lengths always
        # use the real values; only the dispatch metadata and
        # ``max_prompt_length`` / ``max_cache_length`` follow
        # ``dispatch_*``.
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
        staged_by_device: list[dict[str, Buffer]] = [
            {key: value[device_idx] for key, value in staged.items()}
            for device_idx in range(len(replica_devices))
        ]
        # Inputs bound once ride along unnarrowed and uncopied.
        for key, views in self._bound[replica_idx].items():
            for device_idx, view in enumerate(views):
                staged_by_device[device_idx][key] = view

        return KVCacheAssignments(
            cache_lengths_by_device=list(cache_lengths_by_device),
            staged_by_device=staged_by_device,
            max_prompt_length=max_prompt_length_host,
            max_cache_length=max_cache_length_host,
            batch_characteristics=BatchCharacteristics(
                batch_size=batch_size,
                max_prompt_length=max_prompt_len,
                max_cache_valid_length=absolute_max_cached_len,
            ),
        )

    @traced
    def _copy_rows(
        self, replica_idx: int, rows: Mapping[str, tuple[range, range]]
    ) -> None:
        """Copies rows within the bound input holding them.

        Runs on the stream the forward wrote those rows on, so it cannot
        overtake the write it is copying.

        Args:
            replica_idx: Whose slab the rows belong to.
            rows: Per bound input, the source and destination rows.
        """
        for key, (src, dst) in rows.items():
            for view in self._bound[replica_idx][key]:
                # A rank-N buffer wants an index per dimension; every axis
                # but the row one is taken whole.
                rest = (slice(None),) * (view.rank - 1)
                view[(slice(dst.start, dst.stop), *rest)].inplace_copy_from(
                    view[(slice(src.start, src.stop), *rest)]
                )

    @traced
    def _wipe_rows(self, replica_idx: int, rows: Mapping[str, range]) -> None:
        """Zeroes rows within the bound input holding them.

        The driver copies but does not fill, so the zeroes come from a buffer
        allocated once per bound input and never written.

        Args:
            replica_idx: Whose slab the rows belong to.
            rows: Per bound input, the rows to zero.
        """
        for key, span in rows.items():
            for device_idx, view in enumerate(self._bound[replica_idx][key]):
                rest = (slice(None),) * (view.rank - 1)
                dst = view[(slice(span.start, span.stop), *rest)]
                cache_key = (replica_idx, device_idx, key)
                source = self._zero_rows.get(cache_key)
                if source is None:
                    source = Buffer.zeros(dst.shape, dst.dtype, dst.device)
                    self._zero_rows[cache_key] = source
                dst.inplace_copy_from(source)

    def _fill_state(
        self, replica_idx: int, fills: Mapping[str, tuple[int | None, int]]
    ) -> None:
        """Fills each state block from the block named for it, or with zeros.

        Args:
            replica_idx: Whose slab the blocks belong to.
            fills: Per leaf, the block to fill from and the block to fill.
        """
        for leaf_id, (src, dst) in fills.items():
            leaf = self._leaves[leaf_id]
            if src is None:
                span = leaf.bound_row_span(dst)
                if span:
                    self._wipe_rows(replica_idx, span)
            else:
                rows = leaf.bound_row_copies(src, dst)
                if rows:
                    self._copy_rows(replica_idx, rows)

    @traced
    def _resume_state(self, batches: Sequence[Sequence[TextContext]]) -> None:
        """Fills each request's state block before the forward reads it.

        A block drawn from the pool holds whatever its last request wrote, so
        a request resumes from the checkpoint it matched, or from zero when
        it has none.
        """
        for replica_idx, batch in enumerate(batches):
            for ctx in batch:
                for group in self._groups.values():
                    self._fill_state(
                        replica_idx, group.resume(ctx, replica_idx)
                    )

    @traced
    def step(self, ctx: TextContext) -> None:
        """Checkpoints the state onto its successor, then records the forward.

        The checkpoint runs before ``super().step`` so the copy reads a block
        the commit has not published and ``advance`` has not freed, and on the
        stream the forward wrote it on.

        Checkpoints are skipped when prefix caching is off, since nothing
        can publish them.
        """
        if not self._enable_prefix_caching:
            super().step(ctx)
            return
        replica_idx = self._replica_of(ctx)
        for group in self._groups.values():
            self._fill_state(replica_idx, group.checkpoint(ctx, replica_idx))
        super().step(ctx)

    # ============================================================================
    # Metrics
    # ============================================================================

    def get_metrics_aggregated(self) -> KVCacheMetrics:
        """Returns aggregated metrics across all replicas."""
        return self.metrics

    def take_metrics_aggregated(self) -> KVCacheMetrics:
        """Reads and clears aggregated metrics across all replicas."""
        return self.take_metrics()

    def block_count(self, replica_idx: int = 0) -> BlockCount:
        """Returns the device KV cache block occupancy for the given replica."""
        return self.huge_block_count(replica_idx)

    def pressure_pct(self, replica_idx: int = 0) -> float:
        """Returns the fullest leaf's page occupancy, in ``[0, 100]``.

        Reads 100 once no leaf can allocate another page.
        """
        return max(
            (
                count.used_pct
                for count in self.little_block_count(replica_idx).values()
            ),
            default=0.0,
        )

    # ============================================================================
    # KVConnector APIs (full-attention groups only -- see __init__)
    # ============================================================================

    def host_byte_count(self, replica_idx: int = 0) -> ByteCount:
        """Returns the host KV tier occupancy in bytes for the given replica."""
        if self._connector is None:
            return ByteCount(free=0, total=0)
        return self._connector.host_byte_count

    def disk_byte_count(self, replica_idx: int = 0) -> ByteCount:
        """Returns the disk KV tier occupancy in bytes for the given replica."""
        if self._connector is None:
            return ByteCount(free=0, total=0)
        return self._connector.disk_byte_count

    def shutdown(self) -> None:
        """Releases the connector's external resources.

        Drains in-flight host/disk transfers and frees the shared pinned host
        buffer; for the tiered connector this also removes its offload
        directory. One connector backs every replica, so this shuts it down
        once. A no-op for the ``null`` connector.
        """
        if self._connector is not None:
            self._connector.shutdown()

    # ============================================================================
    # Misc
    # ============================================================================

    def runtime_inputs_for_leaf(
        self,
        batches: Sequence[Sequence[TextContext]],
        *,
        max_cache_length: int | None = None,
        batch_characteristics: BatchCharacteristics | None = None,
    ) -> tuple[KVCacheInputsPerDevice[Buffer, Buffer], ...]:
        """Returns :meth:`runtime_inputs` narrowed to a single leaf cache."""
        inputs = self.runtime_inputs(
            batches,
            max_cache_length=max_cache_length,
            batch_characteristics=batch_characteristics,
        )
        return tuple(tree.leaves(inputs, leaf=KVCacheInputsPerDevice))

    def get_device_buffer(self, replica_idx: int) -> KVCacheBufferInterface:
        """Returns the device buffer for the given replica."""
        return self._kv_buffers[replica_idx]

    @property
    def chunk_alignment_tokens(self) -> int:
        """Returns the page size if any group needs the alignment, else zero.

        A recurrent checkpoint fires only on an exact page boundary. With
        prefix caching off there are no checkpoints.
        """
        if not self.params.enable_prefix_caching:
            return 0
        if any(
            group.group_id.is_recurrent() for group in self._groups.values()
        ):
            return self.params.page_size
        return 0
