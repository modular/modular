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

The performant, CUDA-only replacement for the deprecated Python
:class:`~.local_connector.LocalConnector` / :class:`~.tiered_connector.TieredConnector`,
selected with ``--kv-connector rust_tiered``. All of the host block pool, disk
tier, and copy engine live in Rust and run on Rust OS threads with the GIL
released, so the connector never contends for the GIL on the hot path (the
Python lanes' GIL contention was starving GPU utilization).

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
import time
from collections.abc import Sequence
from typing import NamedTuple

from max.driver import Buffer
from max.dtype import DType
from max.nn.kv_cache.cache_params import (
    KVCacheMemory,
    KVHashAlgo,
    ReplicatedKVCacheMemory,
)
from max.nn.kv_cache.metrics import KVCacheMetrics

from ..kv_connector import KVConnectorTransfer
from ..paged_kv_cache.block_copy_engine import (
    _unsafe_alloc_fast_pinned_buffer,
    _unsafe_free_fast_pinned_buffer,
)
from ..paged_kv_cache.block_manager import (
    _resolve_only_use_kv_connector_last_level_cache,
)

logger = logging.getLogger("max.pipelines")


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


class RustTierConnector:
    """KVConnector backed by the Rust host/disk tiered connector."""

    def __init__(
        self,
        replica_kv_memory: Sequence[Sequence[KVCacheMemory]],
        total_num_host_blocks: int,
        disk_cache_dir: str,
        kv_hash_algo: KVHashAlgo = "ahash64",
        max_disk_size_gb: float = 50.0,
    ) -> None:
        # Lazy import: OSS MAX can import this module without the extension.
        from kv_tier_connector import (  # type: ignore[import-not-found]
            TierConnector,
        )

        if not replica_kv_memory:
            raise ValueError("RustTierConnector requires at least one replica")

        # The Rust tier stores blocks keyed by the caller-computed hash bytes,
        # so it is hash-algo agnostic; we only validate the caller's algo is one
        # this connector advertises.
        if kv_hash_algo not in self.supported_hash_algos:
            raise ValueError(
                f"RustTierConnector does not support kv_hash_algo="
                f"{kv_hash_algo!r}; supported: {sorted(self.supported_hash_algos)}"
            )

        gpu0 = replica_kv_memory[0][0].buffer.device
        if gpu0.is_host:
            raise ValueError("KVCacheMemory is on the CPU; cannot offload")

        # bytes_per_page (host row) = sum of each unit's per-page bytes; must
        # match across replicas so a block written by one is readable by another.
        bytes_per_page = sum(
            unit.buffer.shape[1] for unit in replica_kv_memory[0]
        )
        total_num_pages = replica_kv_memory[0][0].buffer.shape[0]

        # The shared pinned host buffer the Rust lanes copy to/from. It is not
        # GC-managed (see `_unsafe_alloc_fast_pinned_buffer`), so it must be
        # explicitly freed in `shutdown`.
        total_gib = total_num_host_blocks * bytes_per_page / (1024**3)
        start = time.perf_counter()
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

        # Per-replica device endpoints + p2p peers (MLA) + compute streams.
        replicas: list[_Replica] = []
        for units in replica_kv_memory:
            # Every buffer this replica touches (primary units + MLA peers), so
            # we can collect each device's compute stream once.
            peers = [
                list(u.peers) if isinstance(u, ReplicatedKVCacheMemory) else []
                for u in units
            ]
            all_buffers = [u.buffer for u in units] + [
                p for peer_list in peers for p in peer_list
            ]
            compute_streams = {
                b.device.id: b.device.default_stream.native_stream_handle
                for b in all_buffers
            }
            replicas.append(
                _Replica(
                    units=[_Unit.from_buffer(u.buffer) for u in units],
                    peers=[
                        [_Unit.from_buffer(p) for p in peer_list]
                        for peer_list in peers
                    ],
                    compute_streams=list(compute_streams.items()),
                )
            )

        self._rust = TierConnector(
            total_num_host_blocks,
            host_base,
            bytes_per_page,
            total_num_pages,
            replicas,
            _resolve_only_use_kv_connector_last_level_cache(),
            disk_cache_dir,
            max_disk_size_gb,
        )
        self._shutdown = False
        logger.info(
            "RustTierConnector initialized: host=%d blocks, disk=%s",
            total_num_host_blocks,
            disk_cache_dir,
        )

    @property
    def name(self) -> str:
        return "RustTieredConnector"

    def load(
        self,
        device_block_ids: list[int],
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
    ) -> KVConnectorTransfer:
        return self._rust.load(
            device_block_ids, list(block_hashes), replica_idx
        )

    def offload(
        self,
        block_ids: list[int],
        block_hashes: Sequence[bytes],
        parent_seq_hash: bytes | None = None,
        replica_idx: int = 0,
    ) -> KVConnectorTransfer:
        return self._rust.offload(block_ids, list(block_hashes), replica_idx)

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
    def num_host_blocks(self) -> int:
        return self._rust.num_host_blocks()

    @property
    def num_used_host_blocks(self) -> int:
        return self._rust.num_used_host_blocks()

    @property
    def num_disk_blocks(self) -> int:
        return self._rust.num_disk_blocks()

    @property
    def num_used_disk_blocks(self) -> int:
        return self._rust.num_used_disk_blocks()

    def reset_prefix_cache(self) -> None:
        self._rust.reset_prefix_cache()

    @property
    def metrics(self) -> KVCacheMetrics:
        h2d, d2h, disk_read, disk_write = self._rust.metrics()
        return KVCacheMetrics(
            h2d_blocks_copied=h2d,
            d2h_blocks_copied=d2h,
            disk_blocks_read=disk_read,
            disk_blocks_written=disk_write,
            inflight_disk_ops=self._rust.inflight_disk_ops(),
        )

    def reset_metrics(self) -> None:
        self._rust.reset_metrics()

    @property
    def supported_hash_algos(self) -> frozenset[KVHashAlgo]:
        return frozenset({"ahash64", "sha256", "sha256_64"})
