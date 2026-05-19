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

from __future__ import annotations

import logging
from collections.abc import Sequence

from max.driver import Buffer, Device
from max.dtype import DType
from max.nn.kv_cache import KVCacheParams
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.pipelines.kv_cache.memory_tier import MemoryTier
from max.profiler import traced

from ..paged_kv_cache.block_manager import (
    _resolve_only_use_kv_connector_last_level_cache,
)
from ..paged_kv_cache.block_pool import BlockPool
from ..paged_kv_cache.block_utils import KVCacheBlock
from ..paged_kv_cache.debug_block_copy_engine import DebugBlockOffloadEngine
from .debug_disk_tier import DebugDiskTier

logger = logging.getLogger("max.pipelines")

GiB = 1024**3


class DebugTieredConnector:
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

        self._block_copy_engine = DebugBlockOffloadEngine(
            total_num_host_blocks, device_buffers
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
        self._disk_tier = DebugDiskTier(
            cache_dir=disk_cache_dir,
            block_nbytes=self._block_disk_bytes,
            max_disk_size_bytes=int(max_disk_size_gb * GiB),
        )

        logger.info(
            "DebugTieredConnector initialized: "
            f"CPU={total_num_host_blocks} blocks, "
            f"Disk={disk_cache_dir} (max {max_disk_size_gb:.1f} GB), "
            f"block_size={self._block_disk_bytes / (1024 * 1024):.1f} MB"
        )

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
        num_loaded = 0

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

                self._block_copy_engine.memcpy_h2d(
                    device_block_id, host_block.bid
                )
                self._h2d_blocks_copied += 1
                num_loaded += 1

            elif (
                self._disk_tier.contains(block_hash)
                and len(self._host_block_pool.free_block_queue) > 0
            ):
                # Disk hit -> async promote to CPU
                host_block, _ = self._host_block_pool.alloc_block()

                # Use uint8 view to avoid bfloat16 numpy incompatibility
                assert self._host_buffer.dtype == DType.uint8
                dest = self._host_buffer.to_numpy()[host_block.bid]

                self._disk_tier.read_block(block_hash, dest)
                self._disk_blocks_read += 1

                if (
                    block_hash
                    not in self._host_block_pool.hash_to_committed_block
                ):
                    self._host_block_pool.commit_into_prefix_cache(
                        block_hash, host_block
                    )

                self._block_copy_engine.memcpy_h2d(
                    device_block_id, host_block.bid
                )
                self._h2d_blocks_copied += 1
                self._host_block_pool.free_block(host_block)
                num_loaded += 1

            else:
                break  # prefix chain broken

        return num_loaded

    @traced
    def offload(
        self,
        block_ids: list[int],
        block_hashes: list[int],
    ) -> None:
        """Offload the device blocks to the external cache."""
        host_blocks: list[KVCacheBlock] = []
        for device_block_id, block_hash in zip(
            block_ids, block_hashes, strict=True
        ):
            host_block = self._maybe_offload_to_host(
                device_block_id, block_hash
            )
            if host_block is not None:
                host_blocks.append(host_block)

    @traced
    def _maybe_offload_to_host(
        self, device_block_id: int, block_hash: int
    ) -> KVCacheBlock | None:
        # Skip if already in host cache
        if block_hash in self._host_block_pool.hash_to_committed_block:
            return None

        assert len(self._host_block_pool.free_block_queue) > 0

        host_block, _ = self._host_block_pool.alloc_block()  # ref_cnt=1

        self._block_copy_engine.memcpy_d2h(host_block.bid, device_block_id)
        self._d2h_blocks_copied += 1

        self._host_block_pool.commit_into_prefix_cache(block_hash, host_block)

        if not self._disk_tier.contains(block_hash):
            src = self._host_buffer.to_numpy()[host_block.bid]
            self._disk_tier.write_block(block_hash, src)
            self._disk_blocks_written += 1

        self._host_block_pool.free_block(host_block)

        return host_block

    @traced
    def sync(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def reset_prefix_cache(self) -> None:
        pass

    @property
    def name(self) -> str:
        """Connector name for logging/debugging."""
        return "DebugTieredConnector"

    @property
    def num_host_blocks(self) -> int:
        """Get the total number of host blocks."""
        return self._total_num_host_blocks

    @property
    def num_used_host_blocks(self) -> int:
        """Get the number of host blocks currently in use."""
        return len(self._host_block_pool.hash_to_committed_block)

    @property
    def num_disk_blocks(self) -> int:
        """Get the total number of disk blocks."""
        return self._disk_tier.num_blocks

    @property
    def num_used_disk_blocks(self) -> int:
        """Get the number of disk blocks currently in use."""
        return self._disk_tier.num_used_blocks

    @property
    def metrics(self) -> KVCacheMetrics:
        """Transfer metrics for host memory and disk operations."""
        return KVCacheMetrics(
            h2d_blocks_copied=self._h2d_blocks_copied,
            d2h_blocks_copied=self._d2h_blocks_copied,
            disk_blocks_written=self._disk_blocks_written,
            disk_blocks_read=self._disk_blocks_read,
        )
