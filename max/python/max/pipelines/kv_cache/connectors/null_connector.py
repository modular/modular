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

"""Null connector implementation for KV cache.

Provides a no-op connector for use when external caching is disabled.
All operations are no-ops that return immediately.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from max.nn.legacy.kv_cache import KVCacheGroupId
from max.nn.legacy.kv_cache.metrics import KVCacheMetrics
from max.pipelines.kv_cache.kv_connector import (
    CompletedTransfer,
    KVConnector,
    KVConnectorTransfer,
    KVTransfer,
)


class NullConnector(KVConnector):
    """No-op connector for when external caching is disabled."""

    @property
    def leaves(self) -> Mapping[str, KVCacheGroupId]:
        """Returns the single ``full`` leaf this connector serves."""
        return {"full": KVCacheGroupId.full()}

    @property
    def name(self) -> str:
        """Reports ``NullConnector`` as the connector name."""
        return "NullConnector"

    def lookup(
        self,
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
        hint: bytes | None = None,
        chain_start: int | None = None,
    ) -> Mapping[str, Sequence[bool]]:
        """Holds nothing, so every block is absent."""
        return {leaf_id: [False] * len(block_hashes) for leaf_id in self.leaves}

    def load(
        self,
        block_ids: Mapping[str, Sequence[int]],
        block_hashes: Mapping[str, Sequence[bytes]],
        replica_idx: int = 0,
        hint: bytes | None = None,
    ) -> KVTransfer:
        """Returns a completed transfer without reading any block."""
        return CompletedTransfer()

    def offload(
        self,
        block_ids: Mapping[str, Sequence[int]],
        block_hashes: Mapping[str, Sequence[bytes]],
        replica_idx: int = 0,
    ) -> KVConnectorTransfer:
        """Returns a completed transfer without writing any block."""
        return CompletedTransfer()

    @property
    def metrics(self) -> KVCacheMetrics:
        """Reports empty transfer metrics."""
        return KVCacheMetrics()
