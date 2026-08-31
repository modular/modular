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

* ``load``/``offload`` run on the scheduler thread (GIL released via pyo3) and
  do only cheap host block-pool bookkeeping, then hand the H2D/D2H copies and
  disk I/O to background Rust lanes. They return immediately with a transfer
  handle (the Rust ``TierTransfer``, which duck-types
  :class:`~..kv_connector.KVConnectorTransfer`); the block manager pins the
  device blocks and the scheduler cordons the request until the handle polls
  complete, so the GPU runs other ready work while the copy is in flight.
* Each copy lane does a blocking ``memcpy; cuStreamSynchronize`` per block on a
  dedicated copy engine (separate H2D and D2H aux streams per device). Keeping
  exactly one copy in flight yields the shared copy engine back to the forward
  pass after every block, so the connector never starves the forward's own
  (tiny) input/output copies -- copy-engine scheduling ignores CUDA stream
  priority, so this is the lever that matters.

This shim owns the pinned host buffer (allocated the same way as
``BlockOffloadEngine``) and passes its address plus the per-replica device
buffer pointers and compute-stream handles to the Rust connector.
"""

from __future__ import annotations

import logging
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import psutil
from max.driver import (
    Buffer,
    _unsafe_alloc_fast_pinned_buffer,
    _unsafe_free_fast_pinned_buffer,
    accelerator_api,
)
from max.dtype import DType
from max.nn.kv_cache import KVCacheGroupId
from max.nn.kv_cache.cache_params import (
    KVCacheMemory,
    KVConnectorConfigInterface,
    KVConnectorType,
)
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.support.human_readable_formatter import to_human_readable_bytes

from ..kv_connector import ByteCount, KVConnector, KVConnectorTransfer
from ..paged_kv_cache.block_manager import (
    _resolve_only_use_kv_connector_last_level_cache,
)

logger = logging.getLogger("max.pipelines")

# Prefix for auto-created tiered-connector disk offload directories. Owned by
# the connectors package (which creates, warns about, and cleans up these
# dirs); the pipeline config imports it only to name the mkdtemp it creates.
KV_OFFLOAD_DIR_PREFIX = "max_kv_tiered_"


def warn_stale_offload_dirs(offload_dir: str) -> None:
    """Warns about leftover KV cache offload directories from previous runs.

    The tiered connectors delete their own offload directory on graceful
    shutdown, but a forceful shutdown (SIGKILL, OOM-kill, or a crash) skips
    that cleanup and leaves the directory (and its cached blocks) on disk.
    Scan the sibling directory for such leftovers and warn so operators can
    reclaim the space.

    Args:
        offload_dir: The offload directory this run will use. Its siblings
            matching ``{KV_OFFLOAD_DIR_PREFIX}*`` are treated as leftovers.
    """
    parent = Path(offload_dir).parent
    try:
        stale = sorted(
            str(p)
            for p in parent.glob(f"{KV_OFFLOAD_DIR_PREFIX}*")
            if p.is_dir() and str(p) != offload_dir
        )
    except OSError:
        return
    if not stale:
        return
    logger.warning(
        "Found %d leftover KV cache offload director%s from a previous run "
        "in %s:\n  %s\n"
        "MAX Serve deletes its offload directory on graceful shutdown, but a "
        "forceful shutdown (SIGKILL / OOM-kill) leaves it behind. If no MAX "
        "Serve process is currently using them, delete these directories to "
        "reclaim disk space.",
        len(stale),
        "y" if len(stale) == 1 else "ies",
        parent,
        "\n  ".join(stale),
    )


def _resolve_disk_offload_dir(cfg: KVConnectorConfigInterface) -> str:
    """Returns the disk offload dir, auto-creating one if unset.

    A single connector serves every DP replica, so the directory is created
    once here (not per replica). Warns about leftovers from previous runs.
    """
    disk_dir = cfg.disk_offload_dir
    if disk_dir is None:
        disk_dir = tempfile.mkdtemp(prefix=KV_OFFLOAD_DIR_PREFIX)
        logger.info(
            "Tiered connector: auto-created disk offload dir %s",
            disk_dir,
        )
    warn_stale_offload_dirs(disk_dir)
    return disk_dir


def host_bytes_per_page(memories: Sequence[KVCacheMemory]) -> int:
    """Returns the width of one host block row for a replica's KV memory.

    A replicated (MLA) unit contributes its stride once -- one copy is stored
    and broadcast back on load, so counting its peers would double the pinned
    host allocation. Must match across replicas, so a block written by one is
    readable by another.

    Args:
        memories: One replica's offload-ready KV memory units.

    Returns:
        The per-page byte width of the shared host buffer.
    """
    return sum(
        mem.bytes_per_page * (1 if mem.replicated else len(mem.buffers))
        for mem in memories
    )


def _check_disk_capacity(
    cache_dir: Path | str, max_disk_size_bytes: int
) -> None:
    """Raises when a disk offload budget exceeds free space at cache_dir."""
    available_bytes = psutil.disk_usage(str(cache_dir)).free
    if max_disk_size_bytes > available_bytes:
        raise RuntimeError(
            "disk_offload_max_gb requests "
            f"{max_disk_size_bytes / (1024**3):.1f} GiB at "
            f"{cache_dir} but only "
            f"{available_bytes / (1024**3):.1f} GiB is available. Reduce "
            "disk_offload_max_gb or free space on the target filesystem."
        )


_GIB = 1024**3


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
            f"{requested_bytes / _GIB:.1f} GiB of pinned host memory but only "
            f"{available_bytes / _GIB:.1f} GiB is available. Reduce "
            "host_offload_max_gb or provision more host memory."
        )


# A device KV buffer endpoint the Rust connector copies to/from. These are
# ``NamedTuple``s (still plain tuples to pyo3, but self-documenting) that the
# Rust ``TierConnector`` extracts positionally.
class _Unit(NamedTuple):
    device_id: int
    data_ptr: int
    len_bytes: int

    @classmethod
    def from_buffer(cls, buffer: Buffer) -> _Unit:
        """The ``_Unit`` endpoint for a KV device buffer."""
        return cls(
            device_id=buffer.device.id,
            data_ptr=buffer._data_ptr(),
            len_bytes=buffer.num_elements * buffer.dtype.size_in_bytes,
        )


class _Replica(NamedTuple):
    units: list[_Unit]
    # ``peers[i]`` are the MLA-replicated copies of ``units[i]`` on other
    # devices (empty for non-replicated units).
    peers: list[list[_Unit]]
    # ``(device_id, compute_stream_handle)`` for every device this replica uses.
    compute_streams: list[tuple[int, int]]


class RustTierConnector(KVConnector):
    """KVConnector backed by the Rust host/disk tiered connector."""

    @classmethod
    def create(
        cls,
        leaves: Mapping[str, KVCacheGroupId],
        replica_kv_memory: Sequence[Sequence[KVCacheMemory]],
        cfg: KVConnectorConfigInterface,
    ) -> RustTierConnector:
        """Builds the connector from a KV connector config.

        Both budgets may be ``None``: the connector then sizes each tier from
        its own device page pool. A 0 disk budget instead drops the tier.
        """
        # Check the KV memory's own device before the build's accelerator API,
        # so a CPU-device pipeline fails the same way on every host rather
        # than reporting "no CUDA/HIP" only on GPU-less ones.
        if (
            replica_kv_memory
            and replica_kv_memory[0][0].buffers[0].device.is_host
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

        # A zero disk budget drops the disk tier, so don't create a dir for it.
        disk_dir = (
            None
            if cfg.disk_offload_max_gb == 0
            else _resolve_disk_offload_dir(cfg)
        )
        if cfg.type != KVConnectorType.rust_tiered:
            logger.warning(
                "kv_connector '%s' is deprecated: its Python implementation "
                "was removed and it now runs the Rust 'rust_tiered' connector. "
                'Pass --kv-connector-config \'{"type": "rust_tiered"}\' '
                "instead.",
                cfg.type.value,
            )
        logger.debug(
            "Creating RustTierConnector: "
            f"host_max_gb={cfg.host_offload_max_gb}, "
            f"disk_dir={disk_dir}, "
            f"disk_max_gb={cfg.disk_offload_max_gb}, "
            f"num_disk_workers={cfg.num_disk_workers}"
        )
        return cls(
            leaves=leaves,
            replica_kv_memory=replica_kv_memory,
            disk_cache_dir=disk_dir,
            host_offload_max_gb=cfg.host_offload_max_gb,
            disk_offload_max_gb=cfg.disk_offload_max_gb,
            num_disk_workers=cfg.num_disk_workers,
        )

    def __init__(
        self,
        leaves: Mapping[str, KVCacheGroupId],
        replica_kv_memory: Sequence[Sequence[KVCacheMemory]],
        disk_cache_dir: str | None,
        host_offload_max_gb: float | None = None,
        disk_offload_max_gb: float | None = None,
        num_disk_workers: int = 32,
    ) -> None:
        """Initializes the connector over ``replica_kv_memory``'s device buffers.

        Args:
            leaves: The leaves / group ids for the connector.
            replica_kv_memory: Per-DP-replica offload-ready KV memory units.
            disk_cache_dir: Directory backing the disk last level, or
                ``None`` for a host-only connector with no disk last level.
            host_offload_max_gb: Host budget. ``None`` sizes the host pool to
                hold 1.5 times the device page pool.
            disk_offload_max_gb: Disk budget. ``None`` sizes it to hold twice
                the device page pool; 0 drops the disk last level.
            num_disk_workers: Disk I/O worker threads.
        """
        # Lazy import: OSS MAX can import this module without the extension.
        from kv_tier_connector import (  # type: ignore[import-not-found]
            TierConnector,
        )

        if not replica_kv_memory:
            raise ValueError("RustTierConnector requires at least one replica")

        if not leaves:
            raise ValueError("RustTierConnector requires at least one leaf")
        if not all(group_id.is_full() for group_id in leaves.values()):
            raise ValueError("RustTierConnector only supports full groups")
        self._leaves = leaves

        gpu0 = replica_kv_memory[0][0].buffers[0].device
        if gpu0.is_host:
            raise ValueError("KVCacheMemory is on the CPU; cannot offload")

        bytes_per_page = host_bytes_per_page(replica_kv_memory[0])
        total_num_pages = replica_kv_memory[0][0].total_num_pages

        # A zero disk budget means no disk last level, whichever arg set it. The
        # tier sizes its capacity from the budget, so a 0 that still opened one
        # would disable eviction rather than disable the tier.
        if disk_offload_max_gb == 0:
            disk_cache_dir = None
        if disk_cache_dir is None:
            disk_offload_max_gb = 0.0

        # Both tiers default to a multiple of the device pool: sizing them in
        # pages keeps the ratio meaningful across models, where a fixed byte
        # budget would be far too small for one and wasteful for another.
        GiB = 1024**3
        if host_offload_max_gb is None:
            total_num_host_blocks = int(1.5 * total_num_pages)
        else:
            total_num_host_blocks = (
                int(host_offload_max_gb * GiB) // bytes_per_page
            )
            if total_num_host_blocks == 0:
                raise RuntimeError(
                    "Insufficient host memory to allocate even a single KV "
                    f"page: one page needs "
                    f"{to_human_readable_bytes(bytes_per_page)} but "
                    f"host_offload_max_gb={host_offload_max_gb} gives "
                    f"{to_human_readable_bytes(int(host_offload_max_gb * GiB))}."
                )
        if disk_offload_max_gb is None:
            disk_offload_max_gb = 2 * total_num_pages * bytes_per_page / GiB

        # The shared pinned host buffer the Rust lanes copy to/from. It is not
        # GC-managed (see `_unsafe_alloc_fast_pinned_buffer`), so it must be
        # explicitly freed in `shutdown`.
        total_bytes = total_num_host_blocks * bytes_per_page
        _check_host_memory_capacity(total_bytes)
        if disk_cache_dir is not None:
            Path(disk_cache_dir).mkdir(parents=True, exist_ok=True)
            _check_disk_capacity(disk_cache_dir, int(disk_offload_max_gb * GiB))
        total_gib = total_bytes / (1024**3)
        start = time.perf_counter()
        logger.info("Allocating %.1f GiB pinned host KV cache...", total_gib)
        self._host_buffer = _unsafe_alloc_fast_pinned_buffer(
            DType.uint8, [total_num_host_blocks, bytes_per_page], gpu0
        )
        elapsed = time.perf_counter() - start
        logger.info(
            "Allocated %.1f GiB pinned host KV cache in %.1f s (%.2f GiB/s)",
            total_gib,
            elapsed,
            total_gib / elapsed if elapsed > 0 else float("inf"),
        )
        host_base = self._host_buffer._data_ptr()

        # Walked in the producer's unit order so the endpoints line up with the
        # host row `host_bytes_per_page` sized.
        replicas: list[_Replica] = []
        for memories in replica_kv_memory:
            units: list[_Unit] = []
            peers: list[list[_Unit]] = []
            for mem in memories:
                if mem.replicated:
                    # Stored once; the rest are H2D broadcast targets.
                    units.append(_Unit.from_buffer(mem.buffers[0]))
                    peers.append(
                        [_Unit.from_buffer(b) for b in mem.buffers[1:]]
                    )
                else:
                    units.extend(_Unit.from_buffer(b) for b in mem.buffers)
                    peers.extend([] for _ in mem.buffers)

            compute_streams = {
                b.device.id: b.device.default_queue.native_stream_handle
                for mem in memories
                for b in mem.buffers
            }
            replicas.append(
                _Replica(
                    units=units,
                    peers=peers,
                    compute_streams=list(compute_streams.items()),
                )
            )

        only_last_level = _resolve_only_use_kv_connector_last_level_cache()
        if only_last_level and disk_cache_dir is None:
            # Rust's `only_last_level` skips the host lookup, so with no disk
            # tier it would leave nothing to hit.
            only_last_level = False
            logger.warning(
                "Ignoring MODULAR_ONLY_USE_KV_CONNECTOR_LAST_LEVEL_CACHE: with "
                "no disk tier the host tier is the last level."
            )

        self._rust = TierConnector(
            list(self._leaves.keys()),
            total_num_host_blocks,
            host_base,
            bytes_per_page,
            total_num_pages,
            replicas,
            only_last_level,
            disk_cache_dir,
            disk_offload_max_gb,
            num_disk_workers,
        )
        self._shutdown = False
        logger.info(
            "RustTierConnector initialized: host=%d blocks, disk=%s, "
            "num_disk_workers=%d",
            total_num_host_blocks,
            disk_cache_dir or "disabled (host-only)",
            num_disk_workers,
        )

    @property
    def leaves(self) -> Mapping[str, KVCacheGroupId]:
        return self._leaves

    @property
    def name(self) -> str:
        return "RustTieredConnector"

    def load(
        self,
        block_ids: Mapping[str, Sequence[int]],
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
    ) -> KVConnectorTransfer:
        if block_ids.keys() != self._leaves.keys():
            raise ValueError(
                f"RustTierConnector.load block_ids keys {sorted(block_ids)} do not "
                f"match the connector's leaves {sorted(self._leaves)}"
            )
        unique_block_ids = {tuple(bids) for bids in block_ids.values()}
        if len(unique_block_ids) != 1:
            raise ValueError(
                "RustTierConnector.load expects identical block IDs across all leaves."
                f"Found {block_ids}"
            )
        leaf_block_ids = list(unique_block_ids.pop())
        return self._rust.load(
            leaf_block_ids,
            list(block_hashes),
            replica_idx,
        )

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
        unique_block_ids = {tuple(bids) for bids in block_ids.values()}
        if len(unique_block_ids) != 1:
            raise ValueError(
                "RustTierConnector.offload expects identical block IDs across all leaves."
                f"Found {block_ids}"
            )
        leaf_block_ids = list(unique_block_ids.pop())
        return self._rust.offload(
            leaf_block_ids,
            list(block_hashes),
            replica_idx,
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

    def count_cached_prefix(
        self, block_hashes: Sequence[bytes]
    ) -> tuple[int, int]:
        return self._rust.count_cached_prefix(list(block_hashes))

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self._rust.shutdown()
        # Free the pinned host buffer after all Rust lanes have been drained/stopped.
        _unsafe_free_fast_pinned_buffer(self._host_buffer)

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

    @property
    def metrics(self) -> KVCacheMetrics:
        h2d, d2h, disk_read, disk_write = self._rust.metrics()
        return KVCacheMetrics(
            h2d_bytes_copied=h2d,
            d2h_bytes_copied=d2h,
            disk_bytes_read=disk_read,
            disk_bytes_written=disk_write,
            inflight_disk_ops=self._rust.inflight_disk_ops(),
        )

    def reset_metrics(self) -> None:
        self._rust.reset_metrics()
