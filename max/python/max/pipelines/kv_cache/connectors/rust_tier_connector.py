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

"""KVConnector shim over the Rust ``kv_tier_connector`` extension.

The only host/disk tiered connector: it backs the ``rust_tiered`` connector type
as well as the retired ``tiered`` alias, whose Python implementation it
replaced. All of the host block pool, disk tier, and copy engine live in Rust
and run on Rust OS threads with the GIL released, so the connector never
contends for the GIL on the hot path (the Python lanes' GIL contention was
starving GPU utilization).

How it works:

* The Rust connector answers only which blocks it holds (``lookup``) and loads
  exactly the blocks it is handed (``load``), so it carries no notion of
  attention shape, window widths or null blocks. What that presence is worth is
  decided here, with the :mod:`~max.pipelines.kv_cache.prefix_hit` rules the
  device pools and the dKV connector also run.
* ``load``/``offload`` run on the scheduler thread (GIL released via pyo3) and
  do only cheap host block-pool bookkeeping, then hand the H2D/D2H copies and
  disk I/O to background Rust lanes. They return immediately with a transfer
  handle (the Rust ``TierTransfer`` wrapped in :class:`_RustTierTransfer`,
  which keys ``g0_blocks_per_leaf`` by leaf id so it satisfies
  :class:`~..kv_connector.KVConnectorTransfer` -- the Rust side has no notion
  of leaf names, only positional leaf indices); the block manager pins the
  device blocks and the scheduler cordons the request until the handle polls
  complete, so the GPU runs other ready work while the copy is in flight.
* Each copy lane does a blocking ``memcpy; cuStreamSynchronize`` per block on a
  dedicated copy engine (separate H2D and D2H aux streams per device). Keeping
  exactly one copy in flight yields the shared copy engine back to the forward
  pass after every block, so the connector never starves the forward's own
  (tiny) input/output copies -- copy-engine scheduling ignores CUDA stream
  priority, so this is the lever that matters.

This shim owns the host staging region (allocated the same way as
``BlockOffloadEngine``) and passes its address plus the per-replica device
buffer pointers and compute-stream handles to the Rust connector.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from typing import NamedTuple, Protocol

import psutil
from max.driver import Device, _ChunkedStagingRegion, accelerator_api
from max.nn.kv_cache import (
    KVCacheGroupId,
    KVCacheMemory,
    KVCacheParamInterface,
    KVConnectorType,
)
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.pipelines.kv_cache.paged_kv_cache.jenga_block_pool import (
    compute_jenga_ratios,
)
from max.support.human_readable_formatter import to_human_readable_bytes

from ..kv_connector import (
    ByteCount,
    KVConnector,
    KVConnectorTransfer,
    KVLoadRefused,
    KVTransfer,
)
from ..paged_kv_cache.block_manager import (
    _resolve_only_use_kv_connector_last_level_cache,
)
from ._offload_dir import OffloadDirectory, acquire_offload_dir

logger = logging.getLogger("max.pipelines")


def _check_disk_capacity(cache_dir: str, max_disk_size_bytes: int) -> None:
    """Raises when a disk offload budget exceeds free space at cache_dir."""
    available_bytes = psutil.disk_usage(cache_dir).free
    if max_disk_size_bytes > available_bytes:
        raise RuntimeError(
            "disk_offload_max_gb requests "
            f"{to_human_readable_bytes(max_disk_size_bytes)} at "
            f"{cache_dir} but only "
            f"{to_human_readable_bytes(available_bytes)} is available. Reduce "
            "disk_offload_max_gb or free space on the target filesystem."
        )


def _check_host_memory_capacity(requested_bytes: int) -> None:
    """Raises when a pinned host allocation exceeds host availability."""
    try:
        available_bytes = psutil.virtual_memory().available
    except (OSError, RuntimeError) as error:
        logger.warning(
            "Unable to determine available host memory; skipping KV cache "
            "host capacity preflight: %s",
            error,
        )
        return
    if requested_bytes > available_bytes:
        raise RuntimeError(
            "KV cache host offload buffer requires "
            f"{to_human_readable_bytes(requested_bytes)} of pinned host "
            f"memory but only {to_human_readable_bytes(available_bytes)} is "
            "available. Reduce "
            "host_offload_max_gb or provision more host memory."
        )


def _validate_leaves(leaves: Mapping[str, KVCacheGroupId]) -> None:
    """Rejects a leaf tree whose hit the prefix rules cannot decide.

    They cover full attention and sliding windows. A recurrent leaf's hit is
    the deepest published state rather than a run, so treating it as either
    shape would claim a prefix whose state pages are not the ones the row
    needs. Refuse instead of serving wrong KV, as ``_validate_dkv_leaves``
    does.

    Called from :meth:`RustTierConnector.create` before it acquires anything,
    so a tree this connector cannot serve leaks neither the offload directory
    nor the pinned staging region.
    """
    if not leaves:
        raise ValueError(
            "RustTierConnector requires at least one KV cache leaf"
        )
    unsupported = {
        leaf_id: group_id
        for leaf_id, group_id in leaves.items()
        if not (group_id.is_full() or group_id.is_sliding_window())
    }
    if unsupported:
        raise ValueError(
            "RustTierConnector supports full-attention and sliding-window "
            f"leaves only. Found: {unsupported}"
        )


# A device KV buffer endpoint the Rust connector copies to/from. These are
# ``NamedTuple``s (still plain tuples to pyo3, but self-documenting) that the
# Rust ``TierConnector`` extracts positionally.
class _Unit(NamedTuple):
    """One TP shard's device buffer endpoint, for FFI only."""

    device_id: int
    data_ptr: int
    bytes: int


class _Leaf(NamedTuple):
    """One leaf's device buffer endpoints, one per TP shard, for FFI only."""

    units: list[_Unit]
    replicated: bool

    @classmethod
    def from_mem(cls, mem: KVCacheMemory) -> _Leaf:
        """The ``_Leaf`` endpoint for a KV device buffer."""
        return cls(
            units=[
                _Unit(
                    device_id=b.device.id,
                    data_ptr=b._data_ptr(),
                    # Per-block stride, not the buffer's full size -- every
                    # shard shares one page width (`KVCacheMemory` requires it).
                    bytes=mem.bytes_per_page,
                )
                for b in mem.buffers
            ],
            replicated=mem.replicated,
        )


class _RawTierTransfer(Protocol):
    """Structural type for the Rust ``TierTransfer`` this module wraps.

    This is purely for FFI.

    ``g0_blocks_per_leaf`` is positional (one list per leaf index) -- the Rust
    connector has no notion of leaf names, only ``leaf_idx: usize``.
    """

    g0_blocks_per_leaf: list[Sequence[int]]

    def is_complete(self) -> bool: ...
    def synchronize(self) -> None: ...


class _RustTierTransfer:
    """Wraps the Rust ``TierTransfer`` to key ``g0_blocks_per_leaf`` by leaf id.

    ``leaves`` must be in the same order the connector built ``bytes_per_leaf``
    and ``replica_kv_memory`` from, since that's the order the Rust side's
    positional ``g0_blocks_per_leaf`` corresponds to.
    """

    def __init__(self, inner: _RawTierTransfer, leaves: Sequence[str]) -> None:
        self._inner = inner
        self._g0_blocks_per_leaf: Mapping[str, Sequence[int]] = dict(
            zip(leaves, inner.g0_blocks_per_leaf, strict=True)
        )

    @property
    def g0_blocks_per_leaf(self) -> Mapping[str, Sequence[int]]:
        return self._g0_blocks_per_leaf

    def is_complete(self) -> bool:
        return self._inner.is_complete()

    def synchronize(self) -> None:
        self._inner.synchronize()


def _alloc_chunked_staging_region(
    host_offload_num_huge_blocks: int,
    host_offload_huge_page_bytes: int,
    device: Device,
) -> _ChunkedStagingRegion:
    host_offload_bytes = (
        host_offload_num_huge_blocks * host_offload_huge_page_bytes
    )
    logger.info(
        "Allocating %s host KV cache staging region...",
        to_human_readable_bytes(host_offload_bytes),
    )
    start = time.perf_counter()
    # Rust copies a block at a time, so no chunk boundary may split one.
    region = _ChunkedStagingRegion(
        byte_size=host_offload_bytes,
        row_bytes=host_offload_huge_page_bytes,
        device=device,
    )
    elapsed = time.perf_counter() - start
    logger.info(
        "Allocated %s host KV cache staging region in %.1f s (%.2f GiB/s)",
        to_human_readable_bytes(host_offload_bytes),
        elapsed,
        host_offload_bytes / 1024**3 / elapsed,
    )
    return region


class RustTierConnector(KVConnector):
    """KVConnector backed by the Rust host/disk tiered connector."""

    def __init__(
        self,
        leaves: Mapping[str, KVCacheGroupId],
        leaf_cache_sizes: Mapping[str, int],
        replica_kv_memory: Sequence[Mapping[str, KVCacheMemory]],
        page_size: int,
        host_offload_num_huge_blocks: int,
        host_offload_huge_page_bytes: int,
        host_offload_cache_ratios: Mapping[str, int],
        disk_dir: OffloadDirectory | None,
        disk_offload_max_bytes: int,
        num_disk_workers: int = 32,
    ) -> None:
        """Initializes the connector over ``replica_kv_memory``'s device buffers.

        Takes ownership of ``disk_dir``, releasing it in :py:meth:`shutdown`.
        It is ``None`` for a host-only connector with no disk last level.
        """
        # Lazy import: OSS MAX can import this module without the extension.
        from kv_tier_connector import (  # type: ignore[import-not-found]
            TierConnector,
        )

        leaf0 = next(iter(leaves.keys()))
        gpu0 = replica_kv_memory[0][leaf0].buffers[0].device
        self._host_region = _alloc_chunked_staging_region(
            host_offload_num_huge_blocks,
            host_offload_huge_page_bytes,
            gpu0,
        )
        host_base = self._host_region.address

        replica_memories: list[list[_Leaf]] = []
        for memories in replica_kv_memory:
            replica_leaves = [
                _Leaf.from_mem(memories[leaf_id]) for leaf_id in leaves
            ]
            replica_memories.append(replica_leaves)

        device_to_stream: dict[int, int] = {}
        for memories in replica_kv_memory:
            for mem in memories.values():
                for b in mem.buffers:
                    device_to_stream[b.device.id] = (
                        b.device.default_queue.native_stream_handle
                    )

        self._leaves = leaves
        self._page_size = page_size
        self._disk_dir = disk_dir
        self._shutdown = False

        bytes_per_leaf = [leaf_cache_sizes[leaf_id] for leaf_id in leaves]
        cache_ratios = [
            host_offload_cache_ratios[leaf_id] for leaf_id in leaves
        ]

        only_last_level = _resolve_only_use_kv_connector_last_level_cache()
        if only_last_level and disk_dir is None:
            # Rust's `only_last_level` skips the host lookup, so with no disk
            # tier it would leave nothing to hit.
            only_last_level = False
            logger.warning(
                "Ignoring MODULAR_ONLY_USE_KV_CONNECTOR_LAST_LEVEL_CACHE: with "
                "no disk tier the host tier is the last level."
            )

        self._rust = TierConnector(
            bytes_per_leaf,
            host_offload_num_huge_blocks,
            cache_ratios,
            host_base,
            replica_memories,
            device_to_stream,
            only_last_level,
            disk_dir.path if disk_dir is not None else None,
            disk_offload_max_bytes,
            num_disk_workers,
        )

    @classmethod
    def create(
        cls,
        leaves: Mapping[str, KVCacheGroupId],
        replica_kv_memory: Sequence[Mapping[str, KVCacheMemory]],
        params: KVCacheParamInterface,
        device_memory_bytes: int,
    ) -> RustTierConnector:
        _validate_leaves(leaves)
        leaf0 = next(iter(leaves.keys()))
        cfg = params.kv_connector_config

        # Check the KV memory's own device before the build's accelerator API,
        # so a CPU-device pipeline fails the same way on every host rather than
        # reporting "no CUDA/HIP" only on GPU-less ones.
        if (
            replica_kv_memory
            and replica_kv_memory[0][leaf0].buffers[0].device.is_host
        ):
            raise ValueError("KVCacheMemory is on the CPU; cannot offload")
        # The Rust connector drives the GPU copy engines directly via its own
        # dlopen'd driver shim, supporting NVIDIA (CUDA) and AMD (HIP) but not
        # Metal/CPU.
        api = accelerator_api()
        if api not in ("cuda", "hip"):
            raise ValueError(
                f"kv_connector '{cfg.type.value}' requires a CUDA or HIP GPU, "
                f"found incompatible accelerator API: '{api}'."
            )

        if cfg.type != KVConnectorType.rust_tiered:
            logger.warning(
                "kv_connector '%s' is deprecated: its Python implementation "
                "was removed and it now runs the Rust 'rust_tiered' connector. "
                'Pass --kv-connector-config \'{"type": "rust_tiered"}\' '
                "instead.",
                cfg.type.value,
            )

        GiB = 1024**3
        host_offload_max_bytes: int = (
            int(cfg.host_offload_max_gb * GiB)
            if cfg.host_offload_max_gb is not None
            else int(1.5 * device_memory_bytes)
        )
        _check_host_memory_capacity(host_offload_max_bytes)

        disk_offload_max_bytes: int = (
            int(cfg.disk_offload_max_gb * GiB)
            if cfg.disk_offload_max_gb is not None
            else 2 * device_memory_bytes
        )
        # A zero disk budget means no disk last level. The tier sizes its
        # capacity from the budget, so a 0 that still opened one would disable
        # eviction rather than disable the tier.
        disk_dir = (
            None
            if disk_offload_max_bytes == 0
            else acquire_offload_dir(cfg.disk_offload_dir)
        )
        if disk_dir is not None:
            _check_disk_capacity(disk_dir.path, disk_offload_max_bytes)

        leaf_cache_sizes = {
            leaf_id: leaf_buffers.host_bytes_per_page
            for leaf_id, leaf_buffers in replica_kv_memory[0].items()
        }
        # The Rust pool hands out every huge block it is given -- unlike the
        # Python pool, it reserves no null block -- so one is enough.
        num_huge_blocks, huge_page_bytes, cache_ratios = compute_jenga_ratios(
            available_bytes=host_offload_max_bytes,
            cache_sizes=leaf_cache_sizes,
            include_null_block=False,
        )
        host_offload_max_bytes = num_huge_blocks * huge_page_bytes

        logger.info(
            "Creating RustTierConnector: "
            f"host_offload_max_bytes={to_human_readable_bytes(host_offload_max_bytes)}, "
            f"disk_cache_dir={disk_dir.path if disk_dir else 'disabled (host-only)'}, "
            f"disk_offload_max_bytes={to_human_readable_bytes(disk_offload_max_bytes)}, "
            f"num_disk_workers={cfg.num_disk_workers}"
        )
        logger.info(
            f"RustTierConnector: {num_huge_blocks} huge pages x {to_human_readable_bytes(huge_page_bytes)} = {to_human_readable_bytes(host_offload_max_bytes)}"
        )
        max_leaf_id_len = max(len(leaf_id) for leaf_id in leaves)
        for leaf_id in leaves:
            ratio = cache_ratios[leaf_id]
            mem = replica_kv_memory[0][leaf_id]
            logger.info(
                f"\t{leaf_id:<{max_leaf_id_len}}: {ratio * num_huge_blocks} pages of {mem.host_bytes_per_page}  ({ratio} per huge page)"
            )

        return cls(
            leaves=leaves,
            leaf_cache_sizes=leaf_cache_sizes,
            replica_kv_memory=replica_kv_memory,
            page_size=params.page_size,
            host_offload_num_huge_blocks=num_huge_blocks,
            host_offload_huge_page_bytes=huge_page_bytes,
            host_offload_cache_ratios=cache_ratios,
            disk_dir=disk_dir,
            disk_offload_max_bytes=disk_offload_max_bytes,
            num_disk_workers=cfg.num_disk_workers,
        )

    @property
    def leaves(self) -> Mapping[str, KVCacheGroupId]:
        return self._leaves

    @property
    def name(self) -> str:
        return "RustTieredConnector"

    def lookup(
        self,
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
        hint: bytes | None = None,
    ) -> Mapping[str, Sequence[bool]]:
        """Which of ``block_hashes`` the host and disk tiers hold, per leaf.

        ``hint`` is ignored for the same reason :meth:`load` ignores it. The
        Rust side knows leaves only by index, so the answer comes back in
        ``leaves()`` order and is keyed here.

        Takes no pin, and the disk index is not this thread's alone: an
        offload's write-through LRU-evicts from it on the disk-write lane. A
        disk-resident block reported here can therefore be gone by the time
        :meth:`load` asks for it, which raises :class:`KVLoadRefused`. The
        host tier has no such race.
        """
        leaf_ids = list(self._leaves)
        width = len(block_hashes)
        if not width:
            return {leaf_id: [] for leaf_id in leaf_ids}
        # Apply what the lanes handed back first: a block whose offload has
        # landed is not in the host prefix cache until its commit event is
        # drained, so a just-written block would read as absent. Moves no
        # data and allocates nothing.
        self._rust.reclaim()
        resident = self._rust.lookup(
            list(range(len(leaf_ids))), list(block_hashes)
        )
        # O(leaves), not O(blocks): the caller indexes these masks
        # positionally, so a mis-sized one has to fail here rather than as an
        # IndexError inside a serving request.
        if len(resident) != len(leaf_ids) or any(
            len(mask) != width for mask in resident
        ):
            raise ValueError(
                f"kv_tier_connector lookup answered {len(resident)} leaves "
                f"{[len(mask) for mask in resident]} blocks wide for a "
                f"{len(leaf_ids)}-leaf tree over {width} blocks"
            )
        return dict(zip(leaf_ids, resident, strict=True))

    def load(
        self,
        block_ids: Mapping[str, Sequence[int]],
        block_hashes: Mapping[str, Sequence[bytes]],
        replica_idx: int = 0,
        hint: bytes | None = None,
    ) -> KVTransfer:
        """Copies each leaf's hashes into the device blocks it was handed.

        Raises:
            KVLoadRefused: If a hash is resident in neither tier -- a disk
                block can be LRU-evicted between :meth:`lookup` and this -- or
                if the host pool has no room to stage the copy.
            ValueError: If the rows and hashes do not line up. A caller bug
                rather than a race, so it does not become a cache miss.
        """
        # ``hint`` is ignored: every tier this connector owns is host-local, so
        # a hint naming the instances that hold a prefix has nothing to route.
        if (
            block_ids.keys() != self._leaves.keys()
            or block_hashes.keys() != self._leaves.keys()
        ):
            raise ValueError(
                f"RustTierConnector.load was given block_ids {sorted(block_ids)} "
                f"and block_hashes {sorted(block_hashes)}, which do not both "
                f"match the connector's leaves {sorted(self._leaves)}"
            )
        leaf_ids = list(self._leaves)
        rows = [list(block_ids[leaf_id]) for leaf_id in leaf_ids]
        try:
            inner = self._rust.load(
                rows,
                [list(block_hashes[leaf_id]) for leaf_id in leaf_ids],
                replica_idx,
            )
        except RuntimeError as error:
            # A vanished hash. The Rust side raises ValueError for a shape
            # mismatch, which is a bug in the caller's sizing and deliberately
            # not caught here.
            raise KVLoadRefused(
                f"kv_tier_connector refused a load: {error}"
            ) from error
        posted = [list(blocks) for blocks in inner.g0_blocks_per_leaf]
        if posted != rows:
            # The one way this happens is the Rust side giving the whole load
            # up rather than truncating it -- its host pool is saturated by
            # in-flight transfers, so a disk promotion had nowhere to land. It
            # submits nothing and warns, which is what makes raising here safe;
            # a partial post would strand the blocks it pinned.
            assert not any(posted), (
                f"kv_tier_connector posted a partial load: asked {rows}, "
                f"it reports {posted}"
            )
            raise KVLoadRefused(
                "kv_tier_connector could not stage the load: its host pool is "
                "saturated by in-flight transfers"
            )
        return _RustTierTransfer(inner, leaf_ids)

    def offload(
        self,
        block_ids: Mapping[str, Sequence[int]],
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
    ) -> KVConnectorTransfer:
        if block_ids.keys() != self._leaves.keys():
            raise ValueError(
                f"RustTierConnector.offload block_ids keys {sorted(block_ids)} do not "
                f"match the connector's leaves {sorted(self._leaves)}"
            )
        block_ids_2d = [block_ids[leaf_id] for leaf_id in self.leaves]
        return _RustTierTransfer(
            self._rust.offload(block_ids_2d, list(block_hashes), replica_idx),
            list(self.leaves),
        )

    def wait_for_loads(self) -> None:
        # No-op: this connector reports load completion through the
        # KVConnectorTransfer it returns from ``load`` (the scheduler polls it),
        # so there is no pre-forward barrier.
        return None

    def wait_for_offloads(self) -> None:
        # No-op: offloads settle through ``poll_transfers`` (the returned
        # transfer's ``is_complete``), not a post-forward barrier.
        return None

    def wait_for_writes(self) -> None:
        """Blocks until all in-flight transfers (incl. disk write-through) drain.

        Not a scheduler hot-path barrier (see ``wait_for_offloads``); this is a
        real quiesce for tests and teardown that need a stable tier state (e.g.
        asserting disk residency after an offload's write-through has landed).
        """
        self._rust.wait_for_writes()

    def touch(
        self, block_hashes: Sequence[bytes], replica_idx: int = 0
    ) -> None:
        return None

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self._rust.shutdown()
        # Rust holds a raw pointer, so the shutdown above is what makes the
        # unmap safe.
        del self._host_region
        # Likewise the offload directory: the Rust shutdown above is what
        # guarantees no worker is still writing into it.
        if self._disk_dir is not None:
            self._disk_dir.release()

    @property
    def host_byte_count(self) -> ByteCount:
        return ByteCount(
            free=self._rust.free_host_bytes(),
            total=self._rust.host_bytes(),
        )

    @property
    def disk_byte_count(self) -> ByteCount:
        return ByteCount(
            free=self._rust.free_disk_bytes(),
            total=self._rust.disk_bytes(),
        )

    def reset_prefix_cache(self) -> None:
        self._rust.reset_prefix_cache()

    def _wrap_rust_metrics(
        self, h2d: int, d2h: int, disk_read: int, disk_write: int
    ) -> KVCacheMetrics:
        return KVCacheMetrics(
            h2d_bytes_copied=h2d,
            d2h_bytes_copied=d2h,
            disk_bytes_read=disk_read,
            disk_bytes_written=disk_write,
            inflight_disk_ops=self._rust.inflight_disk_ops(),
        )

    @property
    def metrics(self) -> KVCacheMetrics:
        return self._wrap_rust_metrics(*self._rust.metrics())

    def take_metrics(self) -> KVCacheMetrics:
        return self._wrap_rust_metrics(*self._rust.take_metrics())
