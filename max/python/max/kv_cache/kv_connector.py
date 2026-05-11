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

"""Connector protocol for external KV cache tiers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from max.nn.kv_cache.metrics import KVCacheMetrics


@runtime_checkable
class KVConnector(Protocol):
    """Protocol for KV cache connectors managing external (non-device) tiers.

    The manager owns device tensors, block allocation, and device-side prefix
    cache. Connectors handle external tier operations (e.g., host memory)
    via lookup/load/save methods.

    Required call ordering per inference step:
      1. connector.sync()   — wait for H2D loads before model execution
      2. [model executes]
      3. connector.flush()  — initiate D2H saves after execution
      connector.save() is called by BlockManager between steps.
    """

    @property
    def name(self) -> str:
        """Connector name for logging/debugging."""
        ...

    def load(
        self,
        device_block_ids: list[int],
        block_hashes: list[int],
    ) -> int:
        """Load data from external cache into device blocks.

        Args:
            device_block_ids: Device block IDs to load data into.
            block_hashes: Hashes to load data for.

        Returns:
            Number of blocks loaded from external cache.
        """
        ...

    def save(
        self,
        block_ids: list[int],
        block_hashes: list[int],
        parent_seq_hash: int = 0,
    ) -> None:
        """Queue device blocks for save to external cache.

        Args:
            block_ids: Device block IDs to save.
            block_hashes: Hashes for the blocks being saved.
            parent_seq_hash: Hash of the parent block for the first block
                in this sequence (0 means root). Subsequent blocks chain
                to their predecessor within the sequence.
        """
        ...

    def sync(self) -> None:
        """Wait for pending loads to complete."""
        ...

    def flush(self) -> None:
        """Initiate queued async saves."""
        ...

    def shutdown(self) -> None:
        """Clean shutdown of connector resources."""
        ...

    # Optional properties with default implementations
    @property
    def num_host_blocks(self) -> int:
        """Number of host blocks. Returns 0 if not applicable."""
        return 0

    @property
    def num_used_host_blocks(self) -> int:
        """Number of used host blocks. Returns 0 if not applicable."""
        return 0

    def reset_prefix_cache(self) -> None:
        """Reset prefix cache. No-op by default."""
        return None

    @property
    def metrics(self) -> KVCacheMetrics:
        """Transfer metrics for this connector. Returns empty metrics by default."""
        return KVCacheMetrics()
