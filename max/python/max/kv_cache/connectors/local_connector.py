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

"""Local connector for KV cache host memory offloading.

Provides a connector implementation that manages host memory as a secondary
cache tier. Committed blocks can be offloaded to host memory and loaded back
to device when needed for prefix cache hits.
"""

from __future__ import annotations

import logging

from max.driver import Buffer
from max.kv_cache.memory_tier import MemoryTier
from max.nn.kv_cache import KVCacheParams
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.profiler import traced

from ..paged_kv_cache.block_copy_engine import BlockOffloadEngine
from ..paged_kv_cache.block_pool import BlockPool

logger = logging.getLogger("max.pipelines")


class LocalConnector:
    """Host memory connector for KV cache offloading.

    Manages host memory as a secondary cache tier. Committed device blocks
    can be offloaded via save() and loaded back via lookup()/load().
    """

    @traced
    def __init__(
        self,
        params: KVCacheParams,
        device_buffers: list[Buffer],
        total_num_host_blocks: int,
    ) -> None:
        """Initialize the local host memory connector."""
        if not params.enable_prefix_caching:
            raise ValueError(
                "LocalConnector requires prefix caching to be enabled"
            )
        if total_num_host_blocks <= 0:
            raise ValueError("LocalConnector requires host blocks")

        self._block_size = params.page_size

        self._total_num_host_blocks = total_num_host_blocks

        # Create BlockOffloadEngine for memory transfers
        self._block_copy_engine = BlockOffloadEngine(
            total_num_host_blocks,
            device_buffers,
            replicate_kv_across_tp=params.replicates_kv_across_tp,
        )

        # Host block pool for managing host memory
        self._host_block_pool = BlockPool(
            MemoryTier.MEMORY_TIER_CPU,
            total_num_host_blocks,
            enable_prefix_caching=True,
            enable_runtime_checks=False,
        )

        # Metrics tracking
        self._h2d_blocks_copied: int = 0
        self._d2h_blocks_copied: int = 0

    @property
    def name(self) -> str:
        """Connector name for logging/debugging."""
        return "LocalConnector"

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
        """Load data from host cache into device blocks.

        Returns:
            Number of blocks loaded from host cache.
        """
        hit = 0
        host_cache = self._host_block_pool.hash_to_committed_block
        for block_hash, device_block_id in zip(
            block_hashes, device_block_ids, strict=True
        ):
            if block_hash in host_cache:
                host_block = host_cache[block_hash]
                self._block_copy_engine.memcpy_h2d(
                    device_block_id, host_block.bid
                )
                self._h2d_blocks_copied += 1
                hit += 1
            else:
                break

        return hit

    @traced
    def offload(
        self,
        block_ids: list[int],
        block_hashes: list[int],
    ) -> None:
        """Queue device blocks for offload to host. Executed in flush()."""
        for block_id, block_hash in zip(block_ids, block_hashes, strict=True):
            self._maybe_offload_to_host(block_id, block_hash)

    @traced
    def sync(self) -> None:
        """Wait for pending H2D transfers to complete."""
        self._block_copy_engine.wait_for_completion()

    def shutdown(self) -> None:
        """Clean shutdown of connector resources."""
        # Wait for any pending transfers
        self._block_copy_engine.wait_for_completion()

    def reset_prefix_cache(self) -> None:
        """Reset the host prefix cache."""
        self._host_block_pool.reset_prefix_cache()

    @property
    def metrics(self) -> KVCacheMetrics:
        """Transfer metrics for host memory operations."""
        return KVCacheMetrics(
            h2d_blocks_copied=self._h2d_blocks_copied,
            d2h_blocks_copied=self._d2h_blocks_copied,
        )

    @traced
    def _maybe_offload_to_host(
        self, device_block_id: int, block_hash: int
    ) -> None:
        """Offload a device block to host memory if not already cached."""
        # Skip if already in host cache
        if block_hash in self._host_block_pool.hash_to_committed_block:
            return

        # Allocate host block. This should never fail!
        assert (
            len(self._host_block_pool.free_block_queue)
            == self._host_block_pool.total_num_blocks
        )
        assert self._host_block_pool.total_num_blocks > 0
        host_block, _ = self._host_block_pool.alloc_block()

        # Copy from device to host
        self._block_copy_engine.memcpy_d2h(host_block.bid, device_block_id)
        self._d2h_blocks_copied += 1

        # Commit to host prefix cache
        self._host_block_pool.commit_into_prefix_cache(block_hash, host_block)

        # Mark as free (host blocks are never "active", only cached)
        self._host_block_pool.free_block(host_block)
