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
    via load/offload methods.

    Required call ordering per inference step:
      1. connector.load() # on main stream
      1. connector.offload() # sync main + aux stream, kick off prev batch offloads
      2. [model executes]
      3. connector.sync() # sync main + aux stream
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

    def offload(
        self,
        block_ids: list[int],
        block_hashes: list[int],
        parent_seq_hash: int = 0,
    ) -> None:
        """Offload the device blocks to the external cache.

        The blocks form one ordered sequence whose first block chains onto
        ``parent_seq_hash`` (``0`` = root). Connectors that key blocks purely by
        hash (host/disk tiers) ignore ``parent_seq_hash``; the dKV connector
        uses it to chain the sequence server-side.

        Args:
            block_ids: Device block IDs to offload, in prefix order.
            block_hashes: Hashes for the blocks being offloaded, in prefix order.
            parent_seq_hash: Hash of the block preceding ``block_hashes[0]`` in
                the prefix, or ``0`` if it begins at the root.
        """
        ...

    def sync(self) -> None:
        """Wait for pending loads/offloads to complete."""
        ...

    def wait_for_loads(self) -> None:
        """Block until all posted loads have landed in device memory.

        Called before the forward pass. Connectors whose loads are ordered on
        the device stream (host/disk tiers) need no work here; the dKV
        connector blocks on its off-stream NIXL READs. No-op by default.
        """
        return None

    def wait_for_offloads(self) -> None:
        """Drain offloads posted since the last call.

        Called after the forward pass. No-op by default.
        """
        return None

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

    @property
    def num_disk_blocks(self) -> int:
        """Number of disk blocks. Returns 0 if not applicable."""
        return 0

    @property
    def num_used_disk_blocks(self) -> int:
        """Number of used disk blocks. Returns 0 if not applicable."""
        return 0

    def reset_prefix_cache(self) -> None:
        """Reset prefix cache. No-op by default."""
        return None

    @property
    def metrics(self) -> KVCacheMetrics:
        """Transfer metrics for this connector. Returns empty metrics by default."""
        return KVCacheMetrics()
