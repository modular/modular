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

"""Three-tier KV cache connector: GPU <-> CPU (pinned) <-> Disk.

Composes the CPU tier (pinned host ``Buffer`` + ``BlockOffloadEngine``) with a
``DiskTier`` that provides flat-file persistence. Write-through policy ensures
every block saved to CPU is also written to disk asynchronously, so CPU
eviction is always safe and disk coverage is maximised for warm restarts.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Sequence
from concurrent.futures import Future, wait
from dataclasses import dataclass

from max.driver import Buffer, Device, DeviceEvent
from max.dtype import DType
from max.kv_cache.memory_tier import MemoryTier
from max.nn.kv_cache import KVCacheParams
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.profiler import Tracer, traced

from ..paged_kv_cache.block_copy_engine import BlockOffloadEngine
from ..paged_kv_cache.block_manager import (
    _resolve_only_use_kv_connector_last_level_cache,
)
from ..paged_kv_cache.block_pool import BlockPool
from ..paged_kv_cache.block_utils import KVCacheBlock
from .disk_tier import DiskTier

logger = logging.getLogger("max.pipelines")

GiB = 1024**3


@dataclass
class _CacheHit:
    block_hash: int
    host_block: KVCacheBlock
    device_block_id: int
    future: Future[None] | None = None


@dataclass
class _PendingDiskWrite:
    d2h_copy_complete_event: DeviceEvent
    host_blocks: list[KVCacheBlock]


class TieredConnector:
    """Three-tier KV cache connector: GPU <-> CPU (pinned) <-> Disk.

    Uses write-through: every block saved to CPU is also async-written to disk.
    Blocks are stored in the native paged format at every tier — no reshape.
    """

    @traced
    def __init__(
        self,
        params: KVCacheParams,
        devices: Sequence[Device],
        device_buffers: list[Buffer],
        total_num_host_blocks: int,
        disk_cache_dir: str,
        max_disk_size_gb: float,
        use_direct_io: bool = False,
    ) -> None:
        if not params.enable_prefix_caching:
            raise ValueError(
                "TieredConnector requires prefix caching to be enabled"
            )
        if total_num_host_blocks <= 0:
            raise ValueError("TieredConnector requires host blocks")

        self._devices = list(devices)
        self._block_size = params.page_size
        self._total_num_host_blocks = total_num_host_blocks

        self._block_copy_engine = BlockOffloadEngine(
            total_num_host_blocks,
            device_buffers,
            replicate_kv_across_tp=params.replicates_kv_across_tp,
        )
        self._host_buffer = self._block_copy_engine.host_buffer

        if self._host_buffer.dtype != DType.uint8:
            raise ValueError("TieredConnector requires uint8 host buffer")
        if len(self._host_buffer.shape) != 2:
            raise ValueError("TieredConnector requires 2D host buffer")
        self._block_disk_bytes = self._host_buffer.shape[1]

        self._host_block_pool = BlockPool(
            MemoryTier.MEMORY_TIER_CPU,
            total_num_host_blocks,
            enable_prefix_caching=True,
            enable_runtime_checks=False,
        )

        # -- Disk tier --
        self._disk_tier = DiskTier(
            cache_dir=disk_cache_dir,
            block_nbytes=self._block_disk_bytes,
            max_disk_size_bytes=int(max_disk_size_gb * GiB),
            use_direct_io=use_direct_io,
        )

        logger.info(
            "TieredConnector initialized: "
            f"CPU={total_num_host_blocks} blocks, "
            f"Disk={disk_cache_dir} (max {max_disk_size_gb:.1f} GB), "
            f"block_size={self._block_disk_bytes / (1024 * 1024):.1f} MB"
        )

        # -- State --
        # host_block kept at ref_cnt=1 (pinned until disk write completes).
        self._pending_disk_writes: deque[_PendingDiskWrite] = deque()
        # Blocks with in-flight disk writes. Holds ref_cnt=1 until the
        # write Future completes so the host memory can't be evicted.
        self._write_locked_blocks: list[tuple[Future[None], KVCacheBlock]] = []

        # Metrics
        self._h2d_blocks_copied: int = 0
        self._d2h_blocks_copied: int = 0
        self._disk_blocks_written: int = 0
        self._disk_blocks_read: int = 0

        # Whether to only use the KVConnector last level cache.
        # When this is set, cache hits will only be served from the disk tier.
        self._only_use_kv_connector_last_level_cache = (
            _resolve_only_use_kv_connector_last_level_cache()
        )

    @property
    def name(self) -> str:
        """Connector name for logging/debugging."""
        return "TieredConnector"

    @property
    def num_host_blocks(self) -> int:
        """Get the total number of host blocks."""
        return self._total_num_host_blocks

    @property
    def num_used_host_blocks(self) -> int:
        """Get the number of host blocks currently in use."""
        return len(self._host_block_pool.hash_to_committed_block)

    @traced
    def load(
        self,
        device_block_ids: list[int],
        block_hashes: list[int],
    ) -> int:
        """Load data from host or disk cache into device blocks.

        Returns:
            Number of blocks loaded from host cache.
        """

        host_cache = self._host_block_pool.hash_to_committed_block
        hits: list[_CacheHit] = []
        disk_reads = 0

        for device_block_id, block_hash in zip(
            device_block_ids, block_hashes, strict=True
        ):
            # Skip the host tier if env var is set
            if (
                not self._only_use_kv_connector_last_level_cache
                and block_hash in host_cache
            ):
                # CPU hit
                host_block = host_cache[block_hash]
                # Touch the host block to ensure it does not get evicted / recycled
                # in subsequent iterations of this loop.
                self._host_block_pool.touch(host_block)
                hits.append(_CacheHit(block_hash, host_block, device_block_id))

            elif (
                self._disk_tier.contains(block_hash)
                and len(self._host_block_pool.free_block_queue) > 0
            ):
                # Disk hit -> async promote to CPU
                host_block, _ = self._host_block_pool.alloc_block()

                # Use uint8 view to avoid bfloat16 numpy incompatibility
                assert self._host_buffer.dtype == DType.uint8
                dest = self._host_buffer.to_numpy()[host_block.bid]
                future = self._disk_tier.read_block_async(block_hash, dest)
                hits.append(
                    _CacheHit(block_hash, host_block, device_block_id, future)
                )

                self._disk_blocks_read += 1
                disk_reads += 1

            else:
                break  # prefix chain broken

        # Unpin the host blocks now that we stopped allocating them and there
        # is no more risk of accidently evicting one.
        for hit in hits:
            self._host_block_pool.free_block(hit.host_block)

        # Process hits in FIFO order: wait on each disk read individually,
        # then immediately enqueue H2D + broadcast.  Disk reads complete in
        # submission order, so this pipelines GPU DMA with remaining I/O.
        num_loaded = 0
        with Tracer(f"{disk_reads} disk reads + H2D"):
            for i, hit in enumerate(hits):
                if hit.future is not None:
                    try:
                        with Tracer(f"Sync on disk read {i}"):
                            hit.future.result()
                    except Exception as exc:
                        logger.error(
                            "Disk read failed for hash %s: %s",
                            hit.block_hash,
                            exc,
                        )
                        break  # prefix chain broken

                    if (
                        hit.block_hash
                        not in self._host_block_pool.hash_to_committed_block
                    ):
                        self._host_block_pool.commit_into_prefix_cache(
                            hit.block_hash, hit.host_block
                        )

                self._block_copy_engine.memcpy_h2d(
                    hit.device_block_id, hit.host_block.bid
                )
                self._h2d_blocks_copied += 1
                num_loaded += 1

        # Wait for in-flight disk reads on unprocessed hits so their
        # worker threads finish writing into the host buffer before we return.
        # This is only triggered if there is a failed disk read that breaks
        # the prefix chain.
        remaining = [
            h.future for h in hits[num_loaded:] if h.future is not None
        ]
        if remaining:
            wait(remaining)

        return num_loaded

    @traced
    def sync(self) -> None:
        """Wait for pending loads/offloads to complete and post disk writes.

        Uses zero-copy: host blocks are kept pinned (ref_cnt=1) from D2H
        through disk write completion.  Numpy views (no ``.copy()``) are
        passed to the disk writer thread — safe because the block can't be
        evicted while pinned.  Blocks are released on the *main* thread in
        ``_drain_completed_writes()``.
        """
        self._block_copy_engine.wait_for_completion()

        # 1. Release blocks from previously completed disk writes.
        self._drain_completed_writes()

        # 2. Submit new writes with numpy for any pending disk writes that
        # have completed.
        while self._pending_disk_writes:
            pending_disk_write = self._pending_disk_writes[-1]
            if not pending_disk_write.d2h_copy_complete_event.is_ready():
                break
            self._pending_disk_writes.pop()
            for host_block in pending_disk_write.host_blocks:
                self._write_block_to_disk(host_block)

    @traced
    def _drain_completed_writes(self) -> None:
        """Release host blocks whose disk writes have completed.

        Always called on the main thread so BlockPool access is safe.
        """
        still_pending: list[tuple[Future[None], KVCacheBlock]] = []
        for future, host_block in self._write_locked_blocks:
            if future.done():
                exc = future.exception()
                if exc is not None:
                    logger.error("Disk write failed: %s", exc)
                self._host_block_pool.free_block(host_block)
            else:
                still_pending.append((future, host_block))
        self._write_locked_blocks = still_pending

    @traced
    def offload(
        self,
        block_ids: list[int],
        block_hashes: list[int],
    ) -> None:
        """Offload the device blocks to the external cache."""
        self._block_copy_engine.wait_for_completion()

        host_blocks: list[KVCacheBlock] = []
        for device_block_id, block_hash in zip(
            block_ids, block_hashes, strict=True
        ):
            host_block = self._maybe_offload_to_host(
                device_block_id, block_hash
            )
            if host_block is not None:
                host_blocks.append(host_block)

        if host_blocks:
            event = DeviceEvent(self._devices[0])
            pending_disk_write = _PendingDiskWrite(
                d2h_copy_complete_event=event,
                host_blocks=host_blocks,
            )
            self._pending_disk_writes.appendleft(pending_disk_write)

    def shutdown(self) -> None:
        """Clean shutdown of connector resources."""
        self._block_copy_engine.wait_for_completion()
        # Wait for in-flight disk writes and release their pinned blocks.
        self._disk_tier.wait_for_writes()
        for _, host_block in self._write_locked_blocks:
            self._host_block_pool.free_block(host_block)
        self._write_locked_blocks.clear()
        self._disk_tier.shutdown()
        # Release any host blocks still pinned in pending disk writes.
        for pending_disk_write in self._pending_disk_writes:
            for host_block in pending_disk_write.host_blocks:
                self._host_block_pool.free_block(host_block)
        self._pending_disk_writes.clear()

        d2h_gb = self._d2h_blocks_copied * self._block_disk_bytes / GiB
        h2d_gb = self._h2d_blocks_copied * self._block_disk_bytes / GiB
        disk_w_gb = self._disk_blocks_written * self._block_disk_bytes / GiB
        disk_r_gb = self._disk_blocks_read * self._block_disk_bytes / GiB
        logger.info(
            "TieredConnector shutdown: "
            f"D2H={self._d2h_blocks_copied} blocks ({d2h_gb:.2f} GB), "
            f"H2D={self._h2d_blocks_copied} blocks ({h2d_gb:.2f} GB), "
            f"Disk written={self._disk_blocks_written} blocks "
            f"({disk_w_gb:.2f} GB), "
            f"Disk read={self._disk_blocks_read} blocks "
            f"({disk_r_gb:.2f} GB)"
        )

    def reset_prefix_cache(self) -> None:
        """Reset the host prefix cache and disk cache."""
        # Wait for in-flight disk writes and release their pinned blocks
        # before resetting, otherwise blocks with ref_cnt>0 survive the reset.
        self._disk_tier.wait_for_writes()
        self._drain_completed_writes()
        self._host_block_pool.reset_prefix_cache()
        self._disk_tier.reset()

    @property
    def metrics(self) -> KVCacheMetrics:
        """Transfer metrics for host memory and disk operations."""
        return KVCacheMetrics(
            h2d_blocks_copied=self._h2d_blocks_copied,
            d2h_blocks_copied=self._d2h_blocks_copied,
            disk_blocks_written=self._disk_blocks_written,
            disk_blocks_read=self._disk_blocks_read,
        )

    @traced
    def _maybe_offload_to_host(
        self, device_block_id: int, block_hash: int
    ) -> KVCacheBlock | None:
        """Offload a device block to host memory if not already cached.

        Returns the host block if a new D2H copy was initiated, None
        otherwise.  The returned block stays at ref_cnt=1 so it can't be
        evicted while an async disk write reads from its memory.  The
        caller is responsible for calling ``free_block()`` when the write
        completes (via ``_drain_completed_writes()``).
        """
        # Skip if already in host cache
        if block_hash in self._host_block_pool.hash_to_committed_block:
            return None

        # Skip if no free host blocks are available. This is possible if there
        # are many disk writes inflight that are holding on to host blocks.
        if len(self._host_block_pool.free_block_queue) == 0:
            return None

        host_block, _ = self._host_block_pool.alloc_block()  # ref_cnt=1

        self._block_copy_engine.memcpy_d2h(host_block.bid, device_block_id)
        self._d2h_blocks_copied += 1

        self._host_block_pool.commit_into_prefix_cache(block_hash, host_block)
        # Do NOT call free_block() — keep ref_cnt=1 so the block can't be
        # evicted while the disk write thread reads from its memory.

        return host_block

    @traced
    def _write_block_to_disk(self, host_block: KVCacheBlock) -> None:
        """Write a host block to disk.

        Args:
            host_block: The host block to write to disk. We should have already
            bumped the ref_cnt by 1 prior to calling this method.
        """
        block_hash = host_block.block_hash
        assert block_hash is not None
        src = self._host_buffer.to_numpy()[host_block.bid]
        future = self._disk_tier.write_block_async(block_hash, src)
        if future is not None:
            self._disk_blocks_written += 1
            self._write_locked_blocks.append((future, host_block))
        else:
            # write_block_async returned None (already on disk / pending)
            self._host_block_pool.free_block(host_block)
