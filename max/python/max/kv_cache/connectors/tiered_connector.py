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
from collections.abc import Sequence
from concurrent.futures import Future, wait

import numpy as np
from max.driver import Device
from max.dtype import DType
from max.interfaces import RequestID, TextGenerationContext
from max.nn.kv_cache import KVCacheBuffer, KVCacheParams
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.profiler import traced
from max.serve.kvcache_agent.kvcache_agent_service_v1_pb2 import (  # type: ignore
    MemoryTier,
)

from ..paged_kv_cache.block_copy_engine import BlockOffloadEngine
from ..paged_kv_cache.block_pool import BlockPool
from ..paged_kv_cache.block_utils import KVCacheBlock
from .disk_tier import DiskTier

logger = logging.getLogger("max.pipelines")

GiB = 1024**3


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
        device_buffer: KVCacheBuffer,
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

        shape_per_block = params.shape_per_block
        dtype = params.dtype

        self._block_copy_engine = BlockOffloadEngine(
            total_num_host_blocks, device_buffer
        )
        self._host_buffer = self._block_copy_engine.host_buffer

        self._host_block_pool = BlockPool(
            MemoryTier.MEMORY_TIER_CPU,
            total_num_host_blocks,
            enable_prefix_caching=True,
            enable_runtime_checks=False,
        )

        # -- Disk tier --

        block_nbytes = int(np.prod(shape_per_block) * dtype.size_in_bytes)
        has_scales = device_buffer.scales is not None
        scale_block_nbytes = 0
        if has_scales and params.kvcache_quant_config is not None:
            scale_block_nbytes = int(
                np.prod(params.shape_per_scale_block)
                * params.kvcache_quant_config.scale_dtype.size_in_bytes
            )

        self._disk_tier = DiskTier(
            cache_dir=disk_cache_dir,
            block_nbytes=block_nbytes,
            num_devices=len(devices),
            max_disk_size_bytes=int(max_disk_size_gb * GiB),
            has_scales=has_scales,
            scale_block_nbytes=scale_block_nbytes,
            use_direct_io=use_direct_io,
        )

        # Per-block size on disk (all TP shards)
        self._block_disk_bytes = block_nbytes * len(devices)
        if has_scales:
            self._block_disk_bytes += scale_block_nbytes * len(devices)

        logger.info(
            "TieredConnector initialized: "
            f"CPU={total_num_host_blocks} blocks, "
            f"Disk={disk_cache_dir} (max {max_disk_size_gb:.1f} GB), "
            f"block_size={self._block_disk_bytes / (1024 * 1024):.1f} MB"
        )

        # -- State --
        self._pending_saves: list[tuple[int, int]] = []  # (block_id, hash)
        self._pending_loads: dict[str, list[tuple[KVCacheBlock, int]]] = {}
        # (bid, hash, host_block) — host_block kept at ref_cnt=1 for
        # zero-copy disk writes (pinned until write completes).
        self._pending_disk_writes: list[tuple[int, int, KVCacheBlock]] = []
        self._pending_disk_reads: dict[
            str, list[tuple[Future[None], int]]
        ] = {}  # (future, block_hash)
        # Blocks with in-flight disk writes.  Holds ref_cnt=1 until the
        # write Future completes so the host memory can't be evicted.
        self._write_locked_blocks: list[tuple[Future[None], KVCacheBlock]] = []

        # Metrics
        self._h2d_blocks_copied: int = 0
        self._d2h_blocks_copied: int = 0
        self._disk_blocks_written: int = 0
        self._disk_blocks_read: int = 0

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
    def lookup(
        self,
        ctx: TextGenerationContext,
        block_hashes: list[int],
    ) -> int:
        """Look up blocks in CPU and disk caches. Returns tokens available."""
        if not block_hashes:
            return 0

        request_id = str(ctx.request_id)

        # Clear any previous state for this request
        self._pending_loads.pop(request_id, None)
        self._pending_disk_reads.pop(request_id, None)

        host_cache = self._host_block_pool.hash_to_committed_block

        hits: list[tuple[KVCacheBlock, int]] = []
        read_futures: list[tuple[Future[None], int]] = []

        for block_hash in block_hashes:
            if block_hash in host_cache:
                # CPU hit
                host_block = host_cache[block_hash]
                self._host_block_pool.touch(host_block)
                hits.append((host_block, block_hash))

            elif self._disk_tier.contains(block_hash):
                # Disk hit -> async promote to CPU
                # ref_cnt=1 from alloc_block().  Do NOT call free_block()
                # here — the block must stay pinned so the async disk read
                # thread can safely write into its memory.  free_block()
                # is deferred to load() (after H2D) or on_request_complete().
                host_block, _ = self._host_block_pool.alloc_block()

                # Use uint8 view to avoid bfloat16 numpy incompatibility
                dest = [
                    ht.view(DType.uint8).to_numpy()[host_block.bid]
                    for ht in self._host_buffer.values
                ]
                scale_dest = (
                    [
                        st.view(DType.uint8).to_numpy()[host_block.bid]
                        for st in self._host_buffer.scales
                    ]
                    if self._host_buffer.scales is not None
                    else None
                )
                future = self._disk_tier.read_block_async(
                    block_hash, dest, scale_dest
                )
                read_futures.append((future, block_hash))

                # Commit to prefix cache (data will be valid before load())
                self._host_block_pool.commit_into_prefix_cache(
                    block_hash, host_block
                )
                hits.append((host_block, block_hash))
                self._disk_blocks_read += 1

            else:
                break  # prefix chain broken

        if hits:
            self._pending_loads[request_id] = hits
        if read_futures:
            self._pending_disk_reads[request_id] = read_futures

        return len(hits) * self._block_size

    @traced
    def load(
        self,
        ctx: TextGenerationContext,
        target_block_ids: list[int],
    ) -> list[int]:
        """Load data from host cache into device blocks.

        Waits for any pending disk reads before issuing H2D copies.

        Returns:
            List of block hashes for the loaded blocks.
        """
        request_id = str(ctx.request_id)

        # Wait for async disk reads to complete before H2D
        failed_hashes: set[int] = set()
        read_entries = self._pending_disk_reads.pop(request_id, None)
        if read_entries:
            futures = [f for f, _ in read_entries]
            wait(futures)

            # Check for read failures and evict corrupt prefix cache entries
            for future, block_hash in read_entries:
                exc = future.exception()
                if exc is not None:
                    logger.error(
                        "Disk read failed for hash %s: %s", block_hash, exc
                    )
                    failed_hashes.add(block_hash)
                    # Evict the corrupt entry from prefix cache
                    host_cache = self._host_block_pool.hash_to_committed_block
                    if block_hash in host_cache:
                        del host_cache[block_hash]

        pending = self._pending_loads.pop(request_id, None)
        if not pending:
            return []

        loaded_hashes: list[int] = []
        for (host_block, block_hash), device_block_id in zip(
            pending, target_block_ids, strict=False
        ):
            if block_hash in failed_hashes:
                # Release the pinned block for failed reads.
                self._host_block_pool.free_block(host_block)
                continue
            self._block_copy_engine.memcpy_h2d(device_block_id, host_block.bid)
            self._h2d_blocks_copied += 1
            loaded_hashes.append(block_hash)
            # Release the pin set in lookup().  The block is now in the
            # prefix cache (ref_cnt → 0) and can be evicted if needed.
            self._host_block_pool.free_block(host_block)

        return loaded_hashes

    @traced
    def save(
        self,
        block_ids: list[int],
        block_hashes: list[int],
    ) -> None:
        """Queue device blocks for offload to host. Executed in flush()."""
        for block_id, block_hash in zip(block_ids, block_hashes, strict=True):
            self._pending_saves.append((block_id, block_hash))

    @traced
    def sync(self) -> None:
        """Wait for D2H transfers, then write-through to disk.

        Uses zero-copy: host blocks are kept pinned (ref_cnt=1) from D2H
        through disk write completion.  Numpy views (no ``.copy()``) are
        passed to the disk writer thread — safe because the block can't be
        evicted while pinned.  Blocks are released on the *main* thread in
        ``_drain_completed_writes()``.
        """
        self._block_copy_engine.wait_for_completion()

        # 1. Release blocks from previously completed disk writes.
        self._drain_completed_writes()

        # 2. Submit new writes with numpy.
        for bid, block_hash, host_block in self._pending_disk_writes:
            # Zero-copy: pass numpy view directly. Safe because
            # ref_cnt=1 prevents the block from being evicted.
            src = [
                ht.view(DType.uint8).to_numpy()[bid]
                for ht in self._host_buffer.values
            ]
            scale_src = (
                [
                    st.view(DType.uint8).to_numpy()[bid]
                    for st in self._host_buffer.scales
                ]
                if self._host_buffer.scales is not None
                else None
            )
            future = self._disk_tier.write_block_async(
                block_hash, src, scale_src
            )
            if future is not None:
                self._write_locked_blocks.append((future, host_block))
            else:
                # write_block_async returned None (already on disk / pending)
                self._host_block_pool.free_block(host_block)

            self._disk_blocks_written += 1

        self._pending_disk_writes.clear()

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
    def flush(self) -> None:
        """Execute pending D2H copies and record blocks for disk write-through."""
        if not self._pending_saves:
            return

        for device_block_id, block_hash in self._pending_saves:
            host_block = self._maybe_offload_to_host(
                device_block_id, block_hash
            )
            if host_block is not None:
                self._pending_disk_writes.append(
                    (host_block.bid, block_hash, host_block)
                )

        self._pending_saves.clear()

    def on_request_complete(
        self,
        request_id: RequestID,
        block_ids: list[int],
    ) -> None:
        """Clean up request-specific state.

        Waits for any pending disk reads, then releases host blocks that
        were pinned in lookup() but never consumed by load().
        """
        req_id = str(request_id)
        # Wait for any pending disk reads before freeing their target blocks.
        read_futures = self._pending_disk_reads.pop(req_id, None)
        if read_futures:
            futures = [f for f, _ in read_futures]
            wait(futures)

        # Free blocks that were never consumed by load().
        pending = self._pending_loads.pop(req_id, None)
        if pending:
            for host_block, _hash in pending:
                self._host_block_pool.free_block(host_block)

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
        for _, _, host_block in self._pending_disk_writes:
            self._host_block_pool.free_block(host_block)
        self._pending_saves.clear()
        self._pending_loads.clear()
        self._pending_disk_writes.clear()
        self._pending_disk_reads.clear()

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
        if block_hash in self._host_block_pool.hash_to_committed_block:
            return None

        host_block, _ = self._host_block_pool.alloc_block()  # ref_cnt=1

        self._block_copy_engine.memcpy_d2h(host_block.bid, device_block_id)
        self._d2h_blocks_copied += 1

        self._host_block_pool.commit_into_prefix_cache(block_hash, host_block)
        # Do NOT call free_block() — keep ref_cnt=1 so the block can't be
        # evicted while the disk write thread reads from its memory.

        return host_block
