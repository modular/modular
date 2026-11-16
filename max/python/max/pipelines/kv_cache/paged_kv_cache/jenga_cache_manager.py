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

import dataclasses
import logging
from collections.abc import Mapping, Sequence

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
    max_pages_per_request,
    spec_decode_cache_slack,
)
from max.nn.kv_cache.data_parallelism_utils import split_into_groups
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.pipelines.context import TextContext
from max.pipelines.graph_input_stager import GraphInputStaging
from max.pipelines.kv_cache.kv_connector import (
    BlockCount,
    ByteCount,
    KVConnector,
)
from max.profiler import traced
from max.support import to_human_readable_bytes
from max.support.math import ceildiv

from ..connectors import create_connector
from .block_utils import LittleKVCacheBlock
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
    _pristine_pool_can_satisfy,
    _stranding_warning,
    plan_jenga_geometry,
)
from .kv_graph_inputs import KVGraphInputs
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
        params: KVCacheParamInterface,
        available_bytes: int,
        *,
        warn_capped: bool = False,
    ) -> JengaGeometry:
        """Fits one device's share of the budget to a huge-block geometry.

        No more huge blocks than every leaf's kernels can address: a leaf
        with :attr:`~max.nn.kv_cache.cache_params.KVLeafRegion.max_pages`
        caps the slab at the blocks holding that many of its pages, and the
        budget past it goes unused.
        """
        leaves = params.leaves()
        # A leaf reports one device's page, which is what a slab tiles in.
        bytes_per_page = {
            leaf_id: leaf.bytes_per_page for leaf_id, leaf in leaves.items()
        }
        stride_alignments = {
            leaf_id: leaf.stride_alignment for leaf_id, leaf in leaves.items()
        }
        n_devices = len(params.devices)
        if n_devices < 1:
            raise ValueError("Jenga KV cache requires at least one device")
        geometry = plan_jenga_geometry(
            available_bytes // n_devices, bytes_per_page, stride_alignments
        )
        caps = {
            leaf_id: leaf.max_pages // geometry.ratios[leaf_id]
            for leaf_id, leaf in leaves.items()
            if leaf.max_pages is not None
        }
        if not caps or min(caps.values()) >= geometry.num_huge_blocks:
            return geometry
        leaf_id = min(caps, key=caps.__getitem__)
        (logger.warning if warn_capped else logger.debug)(
            "Jenga KV manager: capping the slab at %d of the %d huge blocks "
            "the budget holds, since leaf %r's kernels address at most %d "
            "pages",
            caps[leaf_id],
            geometry.num_huge_blocks,
            leaf_id,
            leaves[leaf_id].max_pages,
        )
        return dataclasses.replace(geometry, num_huge_blocks=caps[leaf_id])

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
            enable_prefix_caching=params.enable_prefix_caching,
        )
        if longest is None:
            return None
        return max(1, longest - spec_decode_cache_slack(params))

    @classmethod
    def memory_size(
        cls,
        params: KVCacheParamInterface,
        available_bytes: int,
        max_batch_size: int,
        max_seq_len: int,
    ) -> int:
        """Returns the slab bytes a full batch at ``max_seq_len`` needs.

        Every leaf, a state or scratch leaf included, is priced in the whole
        huge blocks an empty pool would carve for it, the same test the pool
        admits a request by. The answer is checked against the geometry
        :meth:`create` plans for it, since a smaller budget can choose a
        different huge block, and is never more than ``available_bytes``.

        Args:
            params: The cache the slab holds.
            available_bytes: The KV budget across all devices.
            max_batch_size: Requests held at once.
            max_seq_len: The longest request, before speculative slack.

        Returns:
            The bytes across all devices.

        Raises:
            RuntimeError: If ``available_bytes`` cannot hold a single block.
        """
        num_blocks = max_pages_per_request(params, max_seq_len)
        demand = {
            leaf_id: max_batch_size
            * leaf.blocks_to_reserve(
                num_blocks, enable_prefix_caching=params.enable_prefix_caching
            )
            for leaf_id, leaf in params.leaves().items()
        }
        n_devices = len(params.devices)

        def geometry_for(size: int) -> JengaGeometry:
            try:
                geometry = cls._plan_geometry(params, size)
            except ValueError as e:
                raise RuntimeError(
                    f"Insufficient cache memory to allocate even a single page: {e}"
                ) from None
            if geometry.num_huge_blocks < 2:
                raise RuntimeError(
                    "Insufficient cache memory to allocate even a single"
                    f" page: {to_human_readable_bytes(size)} holds"
                    f" {geometry.num_huge_blocks} huge block of"
                    f" {to_human_readable_bytes(geometry.huge_page_bytes)},"
                    " and one is the null block."
                )
            return geometry

        geometry = geometry_for(available_bytes)
        needed = 1 + sum(  # Huge block 0 is the null block.
            ceildiv(blocks, geometry.ratios[leaf_id])
            for leaf_id, blocks in demand.items()
        )
        size = min(
            available_bytes, needed * geometry.huge_page_bytes * n_devices
        )
        while size < available_bytes:
            planned = geometry_for(size)
            if _pristine_pool_can_satisfy(
                planned.num_huge_blocks - 1, planned.ratios, demand
            ):
                return size
            size = min(
                available_bytes, size + planned.huge_page_bytes * n_devices
            )
        return available_bytes

    @classmethod
    def create(
        cls,
        *,
        params: KVCacheParamInterface,
        available_bytes: int,
        max_batch_size: int,
        max_seq_len: int,
        max_num_input_tokens: int | None = None,
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
        geometry = cls._plan_geometry(params, available_bytes, warn_capped=True)
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

        warning = _stranding_warning(
            geometry, len(leaves), available_bytes // len(params.devices)
        )
        if warning is not None:
            logger.warning(warning)

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

        groups = create_groups(
            leaf_infos,
            pools,
            params.page_size,
            enable_prefix_caching=params.enable_prefix_caching,
        )

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
            max_seq_len=max_seq_len,
            connector=connector,
            pools=pools,
            groups=groups,
            slabs=slabs,
        )
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
        max_seq_len: int,
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

        self._replica_slabs = split_into_groups(
            slabs, self.params.data_parallel_degree
        )

        leaves = params.leaves()
        self._max_seq_len_with_slack = max_seq_len + spec_decode_cache_slack(
            params
        )
        self._inputs = KVGraphInputs(
            params=params,
            slabs=slabs,
            kv_buffers=kv_buffers,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )
        self._bound = self._inputs.bound

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
        for replica_idx in range(len(batches)):
            # Initiate saves of everything committed since the last forward.
            self.offload(replica_idx)
        inputs = self._inputs.build(
            batches,
            lambda staging, replica_idx, batch: (
                self._compute_kv_cache_assignments(
                    staging,
                    replica_idx=replica_idx,
                    batch=batch,
                    max_cache_length=max_cache_length,
                    batch_characteristics=batch_characteristics,
                )
            ),
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

        plans: dict[str, list[Sequence[int]]] = {}
        for group in self._groups.values():
            plans.update(group.forward_blocks(batch, num_blocks))
        return self._inputs.assign(
            staging,
            replica_idx=replica_idx,
            batch=batch,
            num_blocks=num_blocks,
            plans=plans,
            max_cache_length=max_cache_length,
            batch_characteristics=batch_characteristics,
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

    def allocate_block(
        self, replica_idx: int, leaf_id: str
    ) -> LittleKVCacheBlock:
        """Allocates initialized memory on the serial scheduler thread.

        Clears before any connector import or peer copy can write the block.
        A failed clear propagates to the existing fatal error handling.
        """
        pool = self.pools[replica_idx]
        block, retyped = pool.alloc_block_with_retype(leaf_id)
        if retyped:
            huge_bid = block.bid // pool.cache_ratios[leaf_id]
            # A valid value in one dtype may be a NaN under another leaf's dtype.
            # Clear the whole huge block, including little pages not yet allocated.
            for slab in self._replica_slabs[replica_idx]:
                slab[huge_bid, :].memset(0)
        return block

    @traced
    def _wipe_rows(self, replica_idx: int, rows: Mapping[str, range]) -> None:
        """Zeroes rows within the bound input holding them."""
        for key, span in rows.items():
            for view in self._bound[replica_idx][key]:
                rest = (slice(None),) * (view.rank - 1)
                view[(slice(span.start, span.stop), *rest)].view(
                    DType.uint8
                ).memset(0)

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
        if self._publishes_states(ctx):
            for group in self._groups.values():
                self._fill_state(
                    replica_idx,
                    group.checkpoint(ctx, replica_idx, allocation_owner=self),
                )
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
        """Returns how much of the replica's headroom is gone, in ``[0, 100]``.

        Headroom is how many more pages of every full leaf at once, and how
        many more requests' pages of every windowed, state or scratch leaf,
        the pool can serve, each against a pristine pool; the larger share
        gone wins. Committed pages no request holds count as free. Reads 100
        once either runs out.
        """
        pool = self.pools[replica_idx]
        per_request = self._blocks_to_reserve(self._max_seq_len_with_slack)
        units = [
            {
                leaf_id: 1
                for leaf_id, group in self.groups.items()
                if not group.demand_plateaus
            },
            {
                leaf_id: per_request[leaf_id]
                for leaf_id, group in self.groups.items()
                if group.demand_plateaus
            },
        ]
        return max(
            (
                100.0
                * (
                    1
                    - pool.max_satisfiable_multiple(unit)
                    / max(
                        1, pool.max_satisfiable_multiple(unit, at_capacity=True)
                    )
                )
                for unit in units
                if unit
            ),
            default=0.0,
        )

    # ============================================================================
    # KVConnector APIs (full-attention groups only -- see __init__)
    # ============================================================================

    def host_byte_count(self) -> ByteCount:
        """Returns the host KV tier occupancy in bytes, shared by every replica."""
        if self._connector is None:
            return ByteCount(free=0, total=0)
        return self._connector.host_byte_count

    def disk_byte_count(self) -> ByteCount:
        """Returns the disk KV tier occupancy in bytes, shared by every replica."""
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
