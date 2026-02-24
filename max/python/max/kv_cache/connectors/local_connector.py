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

from max.driver import Buffer, Device
from max.interfaces import RequestID, TextGenerationContext
from max.nn.legacy.kv_cache import KVCacheParams
from max.nn.legacy.kv_cache.metrics import KVCacheMetrics
from max.profiler import traced
from max.serve.kvcache_agent.kvcache_agent_service_v1_pb2 import (  # type: ignore
    MemoryTier,
)

from ..paged_kv_cache.block_copy_engine import BlockCopyEngine
from ..paged_kv_cache.block_pool import BlockPool
from ..paged_kv_cache.block_utils import KVCacheBlock

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
        devices: Sequence[Device],
        device_tensors: list[Buffer],
        device_scale_tensors: list[Buffer] | None,
        total_num_host_blocks: int,
    ) -> None:
        """Initialize the local host memory connector."""
        if not params.enable_prefix_caching:
            raise ValueError(
                "LocalConnector requires prefix caching to be enabled"
            )
        if total_num_host_blocks <= 0:
            raise ValueError("LocalConnector requires host blocks")

        self._devices = list(devices)
        self._block_size = params.page_size
        self._total_num_host_blocks = total_num_host_blocks

        shape_per_block = params.shape_per_block
        dtype = params.dtype

        # Create host tensors (pinned memory for efficient transfers)
        self._host_tensors: list[Buffer] = []
        self._host_scale_tensors: list[Buffer] | None = None
        if (
            params.quantized_kv_cache
            and params.kvcache_quant_config is not None
        ):
            self._host_scale_tensors = []

        for device in devices:
            if device.is_host:
                raise ValueError(
                    "Host device detected. Paging to host is not supported "
                    "when executing on CPU."
                )
            # Pinned memory for efficient H2D/D2H transfers
            self._host_tensors.append(
                Buffer(
                    shape=[total_num_host_blocks, *shape_per_block],
                    dtype=dtype,
                    device=device,
                    pinned=True,
                )
            )
            if self._host_scale_tensors is not None:
                assert params.kvcache_quant_config is not None
                self._host_scale_tensors.append(
                    Buffer(
                        shape=[
                            total_num_host_blocks,
                            *params.shape_per_scale_block,
                        ],
                        dtype=params.kvcache_quant_config.scale_dtype,
                        device=device,
                        pinned=True,
                    )
                )

        # Create BlockCopyEngine for memory transfers
        self._block_copy_engine = BlockCopyEngine(
            block_size=self._block_size,
            num_device_blocks=device_tensors[0].shape[0]
            if device_tensors
            else 0,
            device_tensors=device_tensors,
            num_host_blocks=total_num_host_blocks,
            host_tensors=self._host_tensors,
            device_scale_tensors=device_scale_tensors,
            host_scale_tensors=self._host_scale_tensors,
        )

        # Host block pool for managing host memory
        self._host_block_pool = BlockPool(
            MemoryTier.MEMORY_TIER_CPU,
            total_num_host_blocks,
            enable_prefix_caching=True,
            enable_runtime_checks=False,
        )

        # Pending saves to host (accumulated between step() and flush())
        self._pending_saves: list[tuple[int, int]] = []  # (block_id, hash)

        # Lookup state for pending loads (keyed by request_id)
        self._pending_loads: dict[str, list[tuple[KVCacheBlock, int]]] = {}

        # Metrics tracking
        self._h2d_blocks_copied: int = 0
        self._d2h_blocks_copied: int = 0

    @property
    def name(self) -> str:
        """Connector name for logging/debugging."""
        return "LocalConnector"

    @property
    def host_scale_tensors(self) -> list[Buffer] | None:
        """Get the host scale tensors for FP8 quantization swapping."""
        return self._host_scale_tensors

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
        """Look up blocks in host prefix cache. Returns tokens available."""
        if not block_hashes:
            return 0

        request_id = str(ctx.request_id)

        # Clear any previous lookup state for this request
        self._pending_loads.pop(request_id, None)

        host_cache = self._host_block_pool.hash_to_committed_block

        hits: list[tuple[KVCacheBlock, int]] = []
        for block_hash in block_hashes:
            if block_hash not in host_cache:
                break
            host_block = host_cache[block_hash]
            hits.append((host_block, block_hash))
            self._host_block_pool.touch(host_block)

        if hits:
            self._pending_loads[request_id] = hits

        return len(hits) * self._block_size

    @traced
    def load(
        self,
        ctx: TextGenerationContext,
        target_block_ids: list[int],
    ) -> list[int]:
        """Load data from host cache into device blocks.

        Returns:
            List of block hashes for the loaded blocks.
        """
        request_id = str(ctx.request_id)
        pending = self._pending_loads.pop(request_id, None)

        if not pending:
            return []

        loaded_hashes: list[int] = []
        for (host_block, block_hash), device_block_id in zip(
            pending, target_block_ids, strict=False
        ):
            self._block_copy_engine.memcpy_h2d(device_block_id, host_block.bid)
            self._h2d_blocks_copied += 1
            loaded_hashes.append(block_hash)
            # Balance the touch() in lookup() so the block returns to the
            # free queue and can be evicted when the host pool is full.
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
        """Wait for pending H2D transfers to complete."""
        self._block_copy_engine.wait_for_completion()

    @traced
    def flush(self) -> None:
        """Execute pending D2H copies to host cache."""
        if not self._pending_saves:
            return

        for device_block_id, block_hash in self._pending_saves:
            self._maybe_offload_to_host(device_block_id, block_hash)

        self._pending_saves.clear()

    def on_request_complete(
        self,
        request_id: RequestID,
        block_ids: list[int],
    ) -> None:
        """Clean up request-specific state."""
        # Free host blocks that were touched in lookup() but never consumed
        # by load() (e.g. if the request was cancelled before load()).
        pending = self._pending_loads.pop(str(request_id), None)
        if pending:
            for host_block, _hash in pending:
                self._host_block_pool.free_block(host_block)

    def shutdown(self) -> None:
        """Clean shutdown of connector resources."""
        # Wait for any pending transfers
        self._block_copy_engine.wait_for_completion()
        self._pending_saves.clear()
        self._pending_loads.clear()

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

        # Allocate host block
        host_block, _ = self._host_block_pool.alloc_block()

        # Copy from device to host
        self._block_copy_engine.memcpy_d2h(host_block.bid, device_block_id)
        self._d2h_blocks_copied += 1

        # Commit to host prefix cache
        self._host_block_pool.commit_into_prefix_cache(block_hash, host_block)

        # Mark as free (host blocks are never "active", only cached)
        self._host_block_pool.free_block(host_block)
