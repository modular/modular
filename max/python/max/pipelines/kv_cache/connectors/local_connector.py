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
from collections.abc import Sequence

from max.nn.kv_cache.cache_params import KVCacheMemory, KVHashAlgo
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.pipelines.kv_cache.memory_tier import MemoryTier
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
        replica_kv_memory: Sequence[Sequence[KVCacheMemory]],
        total_num_host_blocks: int,
    ) -> None:
        """Initialize the local host memory connector.

        The host tier is shared across every DP replica: ``replica_kv_memory``
        holds each replica's device buffers, and a single pinned host buffer +
        host block pool serve them all (SERVOPT-1501).
        """
        if total_num_host_blocks <= 0:
            raise ValueError("LocalConnector requires host blocks")

        self._total_num_host_blocks = total_num_host_blocks

        # Create BlockOffloadEngine for memory transfers
        self._block_copy_engine = BlockOffloadEngine(
            total_num_host_blocks,
            replica_kv_memory,
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
        return len(self._host_block_pool.prefix_cache)

    @property
    def num_disk_blocks(self) -> int:
        """LocalConnector has no disk tier."""
        return 0

    @property
    def num_used_disk_blocks(self) -> int:
        """LocalConnector has no disk tier."""
        return 0

    @traced
    def load(
        self,
        device_block_ids: list[int],
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
    ) -> int:
        """Load data from host cache into ``replica_idx``'s device blocks.

        Returns:
            Number of blocks loaded from host cache.
        """
        host_cache = self._host_block_pool.prefix_cache
        dsts: list[int] = []
        srcs: list[int] = []
        for block_hash, device_block_id in zip(
            block_hashes, device_block_ids, strict=True
        ):
            if block_hash in host_cache:
                dsts.append(device_block_id)
                srcs.append(host_cache[block_hash].bid)
            else:
                break

        self._block_copy_engine.memcpy_h2d(dsts, srcs, replica_idx)
        self._h2d_blocks_copied += len(dsts)
        return len(dsts)

    @traced
    def offload(
        self,
        block_ids: list[int],
        block_hashes: Sequence[bytes],
        parent_seq_hash: bytes | None = None,
        replica_idx: int = 0,
    ) -> None:
        """Offload ``replica_idx``'s device blocks to the host cache.

        ``parent_seq_hash`` is ignored: host blocks are keyed by hash.

        Kicks off the D2H copies on the auxiliary stream without synchronizing.
        The main/aux stream sync runs once per forward pass in
        ``wait_for_loads``; ``offload`` is now called once per request
        (multiple times per forward pass), so syncing here would re-serialize
        the copies against the forward pass and destroy the overlap.
        """
        dsts: list[int] = []
        srcs: list[int] = []
        for block_id, block_hash in zip(block_ids, block_hashes, strict=True):
            pair = self._maybe_offload_to_host(block_id, block_hash)
            if pair is not None:
                dsts.append(pair[0])
                srcs.append(pair[1])

        self._block_copy_engine.memcpy_d2h(dsts, srcs, replica_idx)
        self._d2h_blocks_copied += len(dsts)

    def wait_for_loads(self) -> None:
        """Synchronize the main and auxiliary streams once per forward pass.

        Called once before the forward pass and before the per-request
        ``offload`` calls. This duplex sync makes the forward pass wait for
        in-flight H2D loads and the previous step's D2H offloads (so reused
        blocks are safe), and orders subsequent D2H copies after the forward
        pass. Doing it here once — rather than at the head of every ``offload``
        — keeps the forward pass overlapping with the D2H transfers.
        """
        self._block_copy_engine.wait_for_completion()

    @traced
    def wait_for_offloads(self) -> None:
        """Drain offloads posted this step by syncing the copy streams.

        Called after the forward pass. Waits for in-flight D2H copies on the
        auxiliary stream to complete.
        """
        self._block_copy_engine.wait_for_completion()

    def shutdown(self) -> None:
        """Clean shutdown of connector resources."""
        # Syncs the copy streams and frees the (non-GC-freed) host buffer.
        self._block_copy_engine.close()

    def reset_prefix_cache(self) -> None:
        """Reset the host prefix cache."""
        self._host_block_pool.reset_prefix_cache()

    def _maybe_offload_to_host(
        self, device_block_id: int, block_hash: bytes
    ) -> tuple[int, int] | None:
        """Reserve a host slot for device_block_id if not already cached.

        Returns ``(host_block_id, device_block_id)`` when a slot was reserved,
        or ``None`` if the block is already in the host cache. D2H copy is
        NOT issued here; the caller batches all pairs and calls memcpy_d2h once.
        """
        if block_hash in self._host_block_pool.prefix_cache:
            return None

        assert self._host_block_pool.total_num_blocks > 0
        host_block, _ = self._host_block_pool.alloc_block()
        self._host_block_pool.commit_into_prefix_cache(block_hash, host_block)
        self._host_block_pool.free_block(host_block)
        return host_block.bid, device_block_id

    @property
    def metrics(self) -> KVCacheMetrics:
        """Transfer metrics for host memory operations."""
        return KVCacheMetrics(
            h2d_blocks_copied=self._h2d_blocks_copied,
            d2h_blocks_copied=self._d2h_blocks_copied,
        )

    def reset_metrics(self) -> None:
        """Clear transfer counters after the scheduler samples a batch."""
        self._h2d_blocks_copied = 0
        self._d2h_blocks_copied = 0

    @property
    def supported_hash_algos(self) -> frozenset[KVHashAlgo]:
        return frozenset({"ahash64", "sha256", "sha256_64"})
