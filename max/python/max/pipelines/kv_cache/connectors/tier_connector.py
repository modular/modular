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

"""KVConnector shim over a native host/disk tiered connector extension.

The only host/disk tiered connector: it backs the ``rust_tiered`` connector
type, as well as the retired ``tiered`` alias for ``rust_tiered``, whose Python
implementation it replaced. The host block pool, disk tier, and copy engine run
on native OS threads inside the Rust ``kv_tier_connector`` extension with the
GIL released, so the connector never contends for the GIL on the hot path (the
Python lanes' GIL contention was starving GPU utilization).

How it works:

* The native connector answers only which blocks it holds (``lookup``) and loads
  exactly the blocks it is handed (``load``), so it carries no notion of
  attention shape, window widths or null blocks. What that presence is worth is
  decided here, with the :mod:`max._kv_core` prefix-hit rules the device
  pools and the dKV connector also run.
* ``load``/``offload`` run on the scheduler thread (GIL released via pyo3) and
  do only cheap host block-pool bookkeeping, then hand the H2D/D2H copies and
  disk I/O to background native lanes. They return immediately with a transfer
  handle (the native ``TierTransfer`` wrapped in :class:`_TierTransfer`,
  which keys ``g0_blocks_per_leaf`` by leaf id so it satisfies
  :class:`~..kv_connector.KVConnectorTransfer` -- the native side has no notion
  of leaf names, only positional leaf indices); the block manager pins the
  device blocks and the scheduler cordons the request until the handle polls
  complete, so the GPU runs other ready work while the copy is in flight.
* Each copy lane does a blocking ``memcpy; cuStreamSynchronize`` per block on a
  dedicated copy engine (separate H2D and D2H aux streams per device). Each
  H2D lane keeps one copy in flight on its own stream so it yields the copy
  engine after every block. ``num_h2d_onload_lanes`` (default 1) is how many
  of those lanes run at once.

This shim owns the host staging region (allocated the same way as
``BlockOffloadEngine``) and passes its chunk addresses plus the per-replica
device buffer pointers and compute queues to the native connector.
"""

from __future__ import annotations

import functools
import glob
import importlib
import logging
import os
import time
from collections.abc import Mapping, Sequence
from typing import NamedTuple, Protocol

import psutil
from max.driver import (
    Device,
    _ChunkedStagingRegion,
    accelerator_api,
)
from max.nn.kv_cache import (
    KVCacheGroupId,
    KVCacheMemory,
    KVCacheParamInterface,
    KVConnectorType,
)
from max.nn.kv_cache.cache_params import HostStagingAlloc
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.pipelines.kv_cache.paged_kv_cache.jenga_block_pool import (
    compute_jenga_ratios,
)
from max.support.host_memory import available_host_memory
from max.support.human_readable_formatter import to_human_readable_bytes

from ..kv_connector import (
    ByteCount,
    KVConnector,
    KVConnectorTransfer,
    KVLoadRefused,
    KVTransfer,
)
from ..paged_kv_cache._common import (
    _resolve_only_use_kv_connector_last_level_cache,
)
from ._offload_dir import OffloadDirectory, acquire_offload_dir

logger = logging.getLogger("max.pipelines")

# The default host tier is page-locked up front, so cap it below what the
# process can still allocate, leaving headroom for everything else.
_HOST_OFFLOAD_MAX_FRACTION_OF_AVAILABLE = 0.9

# Each driver-allocated chunk is its own driver call; the region rounds this
# down to whole huge blocks (at least one).
_DRIVER_STAGING_CHUNK_BYTES = 4 * 1024**3

# O_DIRECT requires the buffer address and length aligned to the device's
# logical block size; 4 KiB covers the devices MAX runs on and matches
# `DIRECT_IO_ALIGN` in rust_kv/kv-tier-connector/src/disk_tier.rs.
_DIRECT_IO_ALIGN_BYTES = 1 << 12


@functools.cache
def _native_connector_class() -> type:
    """Returns the native ``TierConnector`` class from the Rust extension.

    The extension is a private, optional module that is not on mypy's path,
    so it is imported lazily and dynamically: OSS MAX imports this module
    without it, and importing this module never loads it.

    Raises:
        ValueError: If this build does not ship the ``kv_tier_connector``
            extension. Any other import failure, such as a broken extension,
            propagates unchanged.
    """
    try:
        module = importlib.import_module("kv_tier_connector")
    except ModuleNotFoundError as e:
        if e.name != "kv_tier_connector":
            raise
        raise ValueError(
            "the tiered KV connector needs the kv_tier_connector extension, "
            "which this build does not ship"
        ) from e
    native_class = module.TierConnector
    assert isinstance(native_class, type)
    return native_class


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
    """Raises when a pinned host allocation exceeds what this process may use."""
    available_bytes = available_host_memory()
    if available_bytes is None:
        logger.warning(
            "Unable to determine available host memory; skipping KV cache "
            "host capacity preflight."
        )
        return
    if requested_bytes > available_bytes:
        raise RuntimeError(
            "KV cache host offload buffer requires "
            f"{to_human_readable_bytes(requested_bytes)} of pinned host "
            f"memory but only {to_human_readable_bytes(available_bytes)} is "
            "available to this process (free host memory, bounded by this "
            "process's cgroup memory limit where one is set). Reduce "
            "host_offload_max_gb, raise the container's memory limit, or "
            "provision more host memory."
        )


class _GttUsage(NamedTuple):
    """The node's GTT (GPU driver pinned host memory) pool, from sysfs."""

    total_bytes: int
    used_bytes: int


def _read_gtt_usage() -> _GttUsage | None:
    """Reads the node's GTT pool from sysfs, or ``None`` if unavailable.

    amdgpu exposes one GTT pool per *node*, mirrored identically under every
    GPU's own sysfs device node, so ``used_bytes`` already reflects every
    process on the node sharing it -- not just this one. Reading the first
    card is therefore as good as reading all of them.

    ``None`` when the sysfs files don't exist or aren't readable: non-AMD
    hardware, or a deployment that doesn't expose ``/sys/class/drm`` into the
    container.
    """
    for total_path in sorted(
        glob.glob("/sys/class/drm/card*/device/mem_info_gtt_total")
    ):
        used_path = total_path.replace(
            "mem_info_gtt_total", "mem_info_gtt_used"
        )
        try:
            with open(total_path) as total_file:
                total_bytes = int(total_file.read().strip())
            with open(used_path) as used_file:
                used_bytes = int(used_file.read().strip())
        except (OSError, ValueError):
            continue
        return _GttUsage(total_bytes, used_bytes)
    return None


# A GTT preflight (unlike the cgroup-bounded host-memory one) warns about a
# limit shared with every other process on the node, not just this one.
_GTT_SHARED_LIMIT_NOTE = (
    "GTT is a node-wide limit shared by every process on the node -- "
    "including any other engine pod co-located on it -- not a per-pod "
    "budget, so another process may be the one consuming it; reducing only "
    "this deployment's host_offload_max_gb will not free anything it isn't "
    "holding itself."
)


def _check_gtt_capacity(requested_bytes: int) -> None:
    """Raises when a driver-allocated host staging region exceeds the node's
    free GTT (GPU driver pinned memory) -- the limit that binds for
    ``host_staging_alloc='driver'`` on HIP, where the allocation is NOT
    charged to this process's memory cgroup."""
    usage = _read_gtt_usage()
    if usage is None:
        logger.warning(
            "Unable to read the node's GTT (GPU driver pinned memory) usage "
            "from sysfs; skipping the KV cache driver-staging capacity "
            "preflight."
        )
        return
    headroom = max(0, usage.total_bytes - usage.used_bytes)
    if requested_bytes > headroom:
        raise RuntimeError(
            "KV cache host offload buffer requests "
            f"{to_human_readable_bytes(requested_bytes)} of GPU driver "
            "pinned (GTT) memory (host_staging_alloc='driver'), but only "
            f"{to_human_readable_bytes(headroom)} of the node's "
            f"{to_human_readable_bytes(usage.total_bytes)} GTT pool is "
            f"free. {_GTT_SHARED_LIMIT_NOTE} Reduce host_offload_max_gb, or "
            "check what else on the node is holding GTT "
            "(/sys/class/drm/card*/device/mem_info_gtt_used)."
        )


def _default_host_offload_bytes(
    device_memory_bytes: int, gtt_bound: bool
) -> int:
    """Returns 1.5x the device page pool, capped to what's actually free.

    Only the default is capped; an explicit ``host_offload_max_gb`` that
    doesn't fit is refused instead (by :func:`_check_host_memory_capacity` or
    :func:`_check_gtt_capacity`). ``gtt_bound`` (``host_staging_alloc='driver'``
    on HIP) caps against the node's free GTT rather than cgroup-bounded host
    RAM: that allocation isn't charged to this process's memory cgroup, so
    the cgroup bound doesn't apply to it, and the GTT bound is what actually
    binds there instead.
    """
    requested = int(1.5 * device_memory_bytes)

    if gtt_bound:
        usage = _read_gtt_usage()
        if usage is None:
            return requested
        free_gtt_bytes = max(0, usage.total_bytes - usage.used_bytes)
        cap = int(free_gtt_bytes * _HOST_OFFLOAD_MAX_FRACTION_OF_AVAILABLE)
        if requested <= cap:
            return requested
        logger.warning(
            "Reduced the default KV cache host offload budget from %s to "
            "%s: it exceeded %.0f%% of the %s free GTT (GPU driver pinned "
            "memory) on the node. %s Set host_offload_max_gb to size the "
            "tier yourself.",
            to_human_readable_bytes(requested),
            to_human_readable_bytes(cap),
            _HOST_OFFLOAD_MAX_FRACTION_OF_AVAILABLE * 100,
            to_human_readable_bytes(free_gtt_bytes),
            _GTT_SHARED_LIMIT_NOTE,
        )
        return cap

    available_bytes = available_host_memory()
    if available_bytes is None:
        return requested
    cap = int(available_bytes * _HOST_OFFLOAD_MAX_FRACTION_OF_AVAILABLE)
    if requested <= cap:
        return requested
    logger.warning(
        "Reduced the default KV cache host offload budget from %s to %s: it "
        "exceeded %.0f%% of the %s this process can still allocate (free host "
        "memory, bounded by this process's cgroup memory limit where one is "
        "set). Set host_offload_max_gb to size the tier yourself.",
        to_human_readable_bytes(requested),
        to_human_readable_bytes(cap),
        _HOST_OFFLOAD_MAX_FRACTION_OF_AVAILABLE * 100,
        to_human_readable_bytes(available_bytes),
    )
    return cap


def _validate_leaves(leaves: Mapping[str, KVCacheGroupId]) -> None:
    """Rejects a leaf tree whose hit the prefix rules cannot decide.

    They cover full attention, sliding windows and recurrent states. Any other
    shape would serve wrong KV, so refuse, as ``_validate_dkv_leaves`` does.

    Called from :meth:`TierConnector.create` before it acquires anything,
    so a tree this connector cannot serve leaks neither the offload directory
    nor the pinned staging region.
    """
    if not leaves:
        raise ValueError("TierConnector requires at least one KV cache leaf")
    unsupported = {
        leaf_id: group_id
        for leaf_id, group_id in leaves.items()
        if not (
            group_id.is_full()
            or group_id.is_sliding_window()
            or group_id.is_recurrent()
        )
    }
    if unsupported:
        raise ValueError(
            "TierConnector supports full-attention, sliding-window and "
            f"recurrent leaves only. Found: {unsupported}"
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
    already_held: int
    """Blocks an offload skipped because the host tier already held them."""

    def is_complete(self) -> bool: ...
    def synchronize(self) -> None: ...


class _TierTransfer:
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


def _compact_unevictable_allowed(
    path: str = "/proc/sys/vm/compact_unevictable_allowed",
) -> bool:
    """Whether the sysctl reads as letting compaction migrate mlocked pages.

    False when it reads 0 or can't be read.
    """
    try:
        with open(path) as f:
            return f.read().strip() != "0"
    except OSError:
        return False


def _alloc_chunked_staging_region(
    host_offload_num_huge_blocks: int,
    host_offload_huge_page_bytes: int,
    device: Device,
    host_staging_alloc: HostStagingAlloc = "registered",
    host_staging_mlock: bool = False,
) -> _ChunkedStagingRegion:
    host_offload_bytes = (
        host_offload_num_huge_blocks * host_offload_huge_page_bytes
    )
    logger.info(
        "Allocating %s host KV cache staging region (%s%s)...",
        to_human_readable_bytes(host_offload_bytes),
        host_staging_alloc,
        ", mlocked" if host_staging_mlock else "",
    )
    start = time.perf_counter()
    # The connectors copy a block at a time, so no chunk boundary may split
    # one.
    if host_staging_alloc == "driver":
        # _ChunkedStagingRegion raises RuntimeError for three distinct
        # failures (a genuine allocation failure, the hardening check
        # refusing movable userptr memory, or this HAL build not supporting
        # driver_alloc at all) and its message is already specific and
        # actionable for each -- left to propagate unchanged rather than
        # replaced with generic advice that fits only the first case.
        region = _ChunkedStagingRegion(
            byte_size=host_offload_bytes,
            row_bytes=host_offload_huge_page_bytes,
            device=device,
            chunk_bytes=_DRIVER_STAGING_CHUNK_BYTES,
            driver_alloc=True,
        )
    else:
        region = _ChunkedStagingRegion(
            byte_size=host_offload_bytes,
            row_bytes=host_offload_huge_page_bytes,
            device=device,
            mlock_pages=host_staging_mlock,
        )
    elapsed = time.perf_counter() - start
    logger.info(
        "Allocated %s host KV cache staging region in %d chunks in %.1f s "
        "(%.2f GiB/s)",
        to_human_readable_bytes(host_offload_bytes),
        region.num_chunks,
        elapsed,
        host_offload_bytes / 1024**3 / elapsed,
    )
    return region


def _leaf_recency(group_id: KVCacheGroupId, page_size: int) -> tuple[str, int]:
    """Returns how the native ``touch`` orders a leaf's blocks.

    The kind, and for a sliding window its width in blocks.
    """
    if group_id.is_sliding_window():
        return ("sliding_window", group_id.blocks_in_window(page_size))
    if group_id.is_recurrent():
        return ("recurrent", 0)
    return ("full_sequence", 0)


class TierConnector(KVConnector):
    """KVConnector backed by a native host/disk tiered connector."""

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
        num_h2d_onload_lanes: int = 1,
        host_staging_alloc: HostStagingAlloc = "registered",
        use_disk_direct_io: bool = False,
        host_staging_mlock: bool = False,
        *,
        native_class: type,
    ) -> None:
        """Initializes the connector over ``replica_kv_memory``'s device buffers.

        Takes ownership of ``disk_dir``, releasing it in :py:meth:`shutdown`.
        It is ``None`` for a host-only connector with no disk last level.
        ``native_class`` is the extension's ``TierConnector``, which
        :meth:`create` has already resolved.
        """
        leaf0 = next(iter(leaves.keys()))
        gpu0 = replica_kv_memory[0][leaf0].buffers[0].device
        self._host_region = _alloc_chunked_staging_region(
            host_offload_num_huge_blocks,
            host_offload_huge_page_bytes,
            gpu0,
            host_staging_alloc,
            host_staging_mlock,
        )
        if self._host_region.driver_alloc:
            host_chunk_bases = list(self._host_region.chunk_addresses)
            host_rows_per_chunk = (
                self._host_region.chunk_bytes // host_offload_huge_page_bytes
            )
        else:
            # The registered region is one contiguous mapping, however many
            # chunks it was registered in.
            host_chunk_bases = [self._host_region.address]
            host_rows_per_chunk = host_offload_num_huge_blocks

        replica_memories: list[list[_Leaf]] = []
        for memories in replica_kv_memory:
            replica_leaves = [
                _Leaf.from_mem(memories[leaf_id]) for leaf_id in leaves
            ]
            replica_memories.append(replica_leaves)

        # The queue the model runs on, which every copy orders against.
        # Rust records its gate events on the raw driver stream.
        devices = {
            b.device.id: b.device
            for memories in replica_kv_memory
            for mem in memories.values()
            for b in mem.buffers
        }
        compute_stream_handles = {
            device_id: device.default_queue.native_stream_handle
            for device_id, device in devices.items()
        }

        self._leaves = leaves
        self._loads_refused = 0
        self._offload_blocks_dropped = 0
        self._page_size = page_size
        self._disk_dir = disk_dir

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

        disk_path = disk_dir.path if disk_dir is not None else None
        self._native = native_class(
            bytes_per_leaf,
            host_offload_num_huge_blocks,
            cache_ratios,
            host_chunk_bases,
            host_rows_per_chunk,
            replica_memories,
            compute_stream_handles,
            only_last_level,
            disk_path,
            disk_offload_max_bytes,
            num_disk_workers,
            num_h2d_onload_lanes,
            use_disk_direct_io,
            leaf_recency=[
                _leaf_recency(group_id, page_size)
                for group_id in leaves.values()
            ],
        )
        # Set only once the native connector exists, so `__del__` on a
        # half-built instance has nothing to stop.
        self._shutdown = False

    def __del__(self) -> None:
        # The native lanes copy into the host region until they are stopped.
        # A connector dropped without `shutdown` would otherwise free the
        # region first, since attributes are released in assignment order.
        if not getattr(self, "_shutdown", True):
            self.shutdown()

    @classmethod
    def create(
        cls,
        leaves: Mapping[str, KVCacheGroupId],
        replica_kv_memory: Sequence[Mapping[str, KVCacheMemory]],
        params: KVCacheParamInterface,
        device_memory_bytes: int,
    ) -> TierConnector:
        _validate_leaves(leaves)
        leaf0 = next(iter(leaves.keys()))
        cfg = params.kv_connector_config
        connector = cfg.type
        # Resolve the extension before anything is checked or acquired, so a
        # build without it fails before the offload directory is taken.
        native_class = _native_connector_class()

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
                f"kv_connector '{connector.value}' requires a CUDA or HIP GPU, "
                f"found incompatible accelerator API: '{api}'."
            )

        if (
            cfg.host_staging_alloc == "driver"
            and api == "hip"
            and os.environ.get("HSA_USERPTR_FOR_PAGED_MEM") != "0"
        ):
            raise RuntimeError(
                "kv_connector_config.host_staging_alloc='driver' requires "
                "HSA_USERPTR_FOR_PAGED_MEM=0 in the environment on HIP, and "
                "it is unset (or not '0'). Without it, hipHostMalloc silently "
                "returns ordinary movable host memory: the deployment pays "
                "the full cost of a driver allocation (system GTT memory, "
                "shared with every process on the node) and gets none of the "
                "protection from kernel memory compaction that 'driver' "
                "exists to provide. HIP reads HSA_USERPTR_FOR_PAGED_MEM once, "
                "at its own initialization, so it must be set in this "
                "process's environment before the process starts -- setting "
                "it after HIP has already initialized has no effect. "
                "hipHostRegister (host_staging_alloc='registered') does not "
                "read this variable either way."
            )

        if cfg.host_staging_mlock:
            if cfg.host_staging_alloc == "driver":
                raise ValueError(
                    "kv_connector_config.host_staging_mlock applies to "
                    "host_staging_alloc='registered'; driver-allocated "
                    "staging memory is already owned by the driver."
                )
            if _compact_unevictable_allowed():
                logger.warning(
                    "host_staging_mlock locks the staging pages, but with "
                    "vm.compact_unevictable_allowed=1 (the kernel default) "
                    "compaction still migrates them. Set it to 0 on the node "
                    "for the lock to protect the region."
                )

        disk_offload_max_bytes: int = (
            int(cfg.disk_offload_max_gb * 1024**3)
            if cfg.disk_offload_max_gb is not None
            else 2 * device_memory_bytes
        )
        # Direct I/O only touches the disk tier, so with a zero disk budget
        # (a host-only connector) none of its compatibility checks apply.
        use_disk_direct_io = (
            cfg.use_disk_direct_io and disk_offload_max_bytes > 0
        )
        if cfg.use_disk_direct_io and not use_disk_direct_io:
            logger.warning(
                "kv_connector_config: use_disk_direct_io has no effect with "
                "no disk tier (disk_offload_max_gb=0)."
            )

        if use_disk_direct_io and cfg.host_staging_alloc == "driver":
            # The native probe below catches this too (the direct write
            # EFAULTs on AMD's driver-owned GTT memory), but naming the
            # conflicting config here is a clearer error than an errno.
            # CUDA's driver-allocated staging is ordinary page-locked memory
            # that takes O_DIRECT fine, so only HIP is refused up front; the
            # probe remains the final word on both vendors.
            if api == "hip":
                raise ValueError(
                    "kv_connector_config.use_disk_direct_io is incompatible "
                    "with host_staging_alloc='driver' on AMD: the "
                    "driver-owned GTT staging memory rejects O_DIRECT "
                    "(EFAULT), and there is no buffered fallback. Use the "
                    "registered staging region to keep direct disk I/O, or "
                    "drop use_disk_direct_io to keep driver staging."
                )
            logger.warning(
                "use_disk_direct_io with host_staging_alloc='driver' on CUDA: "
                "driver staging is ordinary page-locked memory, so direct "
                "I/O should work. The startup probe will confirm it."
            )

        if connector == KVConnectorType.tiered:
            logger.warning(
                "kv_connector '%s' is deprecated: its Python implementation "
                "was removed and it now runs the Rust 'rust_tiered' connector. "
                'Pass --kv-connector-config \'{"type": "rust_tiered"}\' '
                "instead.",
                connector.value,
            )

        # A driver allocation draws from the node's shared GTT pool, not this
        # process's memory cgroup, only on HIP; on CUDA it's ordinary
        # cudaHostAlloc pinned memory, still cgroup-bounded like a
        # registered region.
        gtt_bound = cfg.host_staging_alloc == "driver" and api == "hip"

        GiB = 1024**3
        host_offload_max_bytes: int = (
            int(cfg.host_offload_max_gb * GiB)
            if cfg.host_offload_max_gb is not None
            else _default_host_offload_bytes(device_memory_bytes, gtt_bound)
        )
        if gtt_bound:
            _check_gtt_capacity(host_offload_max_bytes)
        else:
            _check_host_memory_capacity(host_offload_max_bytes)

        leaf_cache_sizes = {
            leaf_id: leaf_buffers.host_bytes_per_page
            for leaf_id, leaf_buffers in replica_kv_memory[0].items()
        }
        if use_disk_direct_io:
            # O_DIRECT needs every block length aligned to 4096 bytes. Check
            # all leaves before allocating anything, so one error names every
            # offender.
            unaligned = {
                leaf_id: size
                for leaf_id, size in leaf_cache_sizes.items()
                if size % _DIRECT_IO_ALIGN_BYTES != 0
            }
            if unaligned:
                raise ValueError(
                    "use_disk_direct_io requires every leaf's block size to "
                    f"be a multiple of 4096 bytes; offending leaves: {unaligned}. "
                    "Disable the flag."
                )
        # The host pool is built without a null block -- unlike the device
        # pool -- so it hands out every huge block, and one is enough.
        num_huge_blocks, huge_page_bytes, cache_ratios = compute_jenga_ratios(
            available_bytes=host_offload_max_bytes,
            cache_sizes=leaf_cache_sizes,
            include_null_block=False,
        )
        host_offload_max_bytes = num_huge_blocks * huge_page_bytes

        # A zero disk budget means no disk last level. The tier sizes its
        # capacity from the budget, so a 0 that still opened one would disable
        # eviction rather than disable the tier.
        disk_dir = (
            None
            if disk_offload_max_bytes == 0
            else acquire_offload_dir(cfg.disk_offload_dir)
        )
        # Everything past the acquisition can still raise (the native
        # direct-I/O probe, for one), and `__del__` skips a half-built
        # connector, so the directory is released here on any failure.
        try:
            if disk_dir is not None:
                _check_disk_capacity(disk_dir.path, disk_offload_max_bytes)
            logger.info(
                "Creating TierConnector: "
                f"type={connector.value}, "
                f"host_offload_max_bytes={to_human_readable_bytes(host_offload_max_bytes)}, "
                f"disk_cache_dir={disk_dir.path if disk_dir else 'disabled (host-only)'}, "
                f"disk_offload_max_bytes={to_human_readable_bytes(disk_offload_max_bytes)}, "
                f"num_disk_workers={cfg.num_disk_workers}, "
                f"num_h2d_onload_lanes={cfg.num_h2d_onload_lanes}, "
                f"disk_io={'direct (O_DIRECT)' if use_disk_direct_io else 'buffered'}"
            )
            logger.info(
                f"TierConnector: {num_huge_blocks} huge pages x {to_human_readable_bytes(huge_page_bytes)} = {to_human_readable_bytes(host_offload_max_bytes)}"
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
                num_h2d_onload_lanes=cfg.num_h2d_onload_lanes,
                host_staging_alloc=cfg.host_staging_alloc,
                use_disk_direct_io=use_disk_direct_io,
                host_staging_mlock=cfg.host_staging_mlock,
                native_class=native_class,
            )
        except BaseException:
            if disk_dir is not None:
                disk_dir.release()
            raise

    @property
    def leaves(self) -> Mapping[str, KVCacheGroupId]:
        return self._leaves

    @property
    def name(self) -> str:
        return "TieredConnector"

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
        self._native.reclaim()
        resident = self._native.lookup(
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
                f"TierConnector.load was given block_ids {sorted(block_ids)} "
                f"and block_hashes {sorted(block_hashes)}, which do not both "
                f"match the connector's leaves {sorted(self._leaves)}"
            )
        leaf_ids = list(self._leaves)
        rows = [list(block_ids[leaf_id]) for leaf_id in leaf_ids]
        try:
            inner = self._native.load(
                rows,
                [list(block_hashes[leaf_id]) for leaf_id in leaf_ids],
                replica_idx,
            )
        except RuntimeError as error:
            self._loads_refused += 1
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
            self._loads_refused += 1
            raise KVLoadRefused(
                "kv_tier_connector could not stage the load: its host pool is "
                "saturated by in-flight transfers"
            )
        return _TierTransfer(inner, leaf_ids)

    def offload(
        self,
        block_ids: Mapping[str, Sequence[int]],
        block_hashes: Mapping[str, Sequence[bytes]],
        replica_idx: int = 0,
    ) -> KVConnectorTransfer:
        """Offloads each leaf's blocks under that leaf's hashes.

        Raises:
            ValueError: If the keys do not match the connector's leaves.
        """
        if (
            block_ids.keys() != self._leaves.keys()
            or block_hashes.keys() != self._leaves.keys()
        ):
            raise ValueError(
                f"TierConnector.offload was given block_ids {sorted(block_ids)} "
                f"and block_hashes {sorted(block_hashes)}, which do not both "
                f"match the connector's leaves {sorted(self._leaves)}"
            )
        leaf_ids = list(self._leaves)
        rows = [list(block_hashes[leaf_id]) for leaf_id in leaf_ids]
        inner = self._native.offload(
            [list(block_ids[leaf_id]) for leaf_id in leaf_ids],
            rows,
            replica_idx,
        )
        # The host pool takes what it has room for and drops the rest, so a
        # starved pool shows up here before it shows up as a lower hit rate.
        # A block the host tier already held was not dropped.
        self._offload_blocks_dropped += (
            sum(len(row) for row in rows)
            - sum(len(blocks) for blocks in inner.g0_blocks_per_leaf)
            - inner.already_held
        )
        return _TierTransfer(inner, leaf_ids)

    def wait_for_writes(self) -> None:
        """Blocks until all in-flight transfers (incl. disk write-through) drain.

        Not a scheduler hot-path call: transfers settle through
        ``poll_transfers``. This is a real quiesce for tests and teardown that
        need a stable tier state (such as asserting disk residency after an
        offload's write-through has landed).
        """
        self._native.wait_for_writes()

    def touch(
        self, block_hashes: Sequence[bytes], replica_idx: int = 0
    ) -> None:
        """Refreshes each leaf's recency in the tiers.

        Does not raise. A missed touch costs a later refetch.
        """
        del replica_idx
        try:
            self._native.reclaim()
            self._native.touch(list(block_hashes))
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            logger.exception("Tier touch skipped")

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self._native.shutdown()
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
            free=self._native.free_host_bytes(),
            total=self._native.host_bytes(),
        )

    @property
    def disk_byte_count(self) -> ByteCount:
        return ByteCount(
            free=self._native.free_disk_bytes(),
            total=self._native.disk_bytes(),
        )

    def reset_prefix_cache(self) -> None:
        self._native.reset_prefix_cache()

    def _wrap_native_metrics(
        self, h2d: int, d2h: int, disk_read: int, disk_write: int
    ) -> KVCacheMetrics:
        return KVCacheMetrics(
            h2d_bytes_copied=h2d,
            d2h_bytes_copied=d2h,
            disk_bytes_read=disk_read,
            disk_bytes_written=disk_write,
            inflight_disk_ops=self._native.inflight_disk_ops(),
            connector_loads_refused=self._loads_refused,
            connector_offload_blocks_dropped=self._offload_blocks_dropped,
        )

    @property
    def metrics(self) -> KVCacheMetrics:
        return self._wrap_native_metrics(*self._native.metrics())

    def take_metrics(self) -> KVCacheMetrics:
        metrics = self._wrap_native_metrics(*self._native.take_metrics())
        self._loads_refused = 0
        self._offload_blocks_dropped = 0
        return metrics
