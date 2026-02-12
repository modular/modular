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

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from max.driver import Buffer
from max.nn.legacy.kv_cache.metrics import KVCacheMetrics

if TYPE_CHECKING:
    from max.interfaces import RequestID, TextGenerationContext


@runtime_checkable
class KVConnector(Protocol):
    """Protocol for KV cache connectors managing external (non-device) tiers.

    The manager owns device tensors, block allocation, and device-side prefix
    cache. Connectors handle external tier operations (e.g., host memory)
    via lookup/load/save methods.

    Async coordination:
    - sync(): Wait for pending loads before model execution
    - flush(): Initiate async saves after get_runtime_inputs()
    """

    @property
    def name(self) -> str:
        """Connector name for logging/debugging."""
        ...

    def lookup(
        self,
        ctx: TextGenerationContext,
        block_hashes: list[int],
    ) -> int:
        """Look up blocks in external cache tiers.

        Args:
            ctx: The request context.
            block_hashes: Hashes to look up in external cache.

        Returns:
            Number of tokens available from external cache.
        """
        ...

    def load(
        self,
        ctx: TextGenerationContext,
        target_block_ids: list[int],
        device_tensors: list[Buffer],
    ) -> list[int]:
        """Load data from external cache into device blocks.

        Args:
            ctx: The request context.
            target_block_ids: Device block IDs to load data into.
            device_tensors: Device KV cache tensors to copy into.

        Returns:
            List of block hashes for the loaded blocks, in the same order
            as target_block_ids. Returns empty list if no loads occurred.
        """
        ...

    def save(
        self,
        block_ids: list[int],
        block_hashes: list[int],
    ) -> None:
        """Queue device blocks for save to external cache.

        Args:
            block_ids: Device block IDs to save.
            block_hashes: Hashes for the blocks being saved.
        """
        ...

    def sync(self) -> None:
        """Wait for pending loads to complete."""
        ...

    def flush(self) -> None:
        """Initiate queued async saves."""
        ...

    def on_request_complete(
        self,
        request_id: RequestID,
        block_ids: list[int],
    ) -> None:
        """Called when a request completes to clean up tracking state."""
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

    @property
    def host_tensors(self) -> list[Buffer] | None:
        """Host tensors, or None if not applicable."""
        return None

    @property
    def host_scale_tensors(self) -> list[Buffer] | None:
        """Host scale tensors, or None if not applicable."""
        return None

    def reset_prefix_cache(self) -> None:
        """Reset prefix cache. No-op by default."""
        ...

    @property
    def metrics(self) -> KVCacheMetrics:
        """Transfer metrics for this connector. Returns empty metrics by default."""
        return KVCacheMetrics()
