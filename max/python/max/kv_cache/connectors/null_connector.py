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

from max.interfaces import RequestID, TextGenerationContext
from max.nn.legacy.kv_cache.metrics import KVCacheMetrics


class NullConnector:
    """No-op connector for when external caching is disabled."""

    @property
    def name(self) -> str:
        return "NullConnector"

    def lookup(
        self,
        ctx: TextGenerationContext,
        block_hashes: list[int],
    ) -> int:
        return 0

    def load(
        self,
        ctx: TextGenerationContext,
        target_block_ids: list[int],
    ) -> list[int]:
        return []

    def save(
        self,
        block_ids: list[int],
        block_hashes: list[int],
    ) -> None:
        pass

    def sync(self) -> None:
        pass

    def flush(self) -> None:
        pass

    def on_request_complete(
        self,
        request_id: RequestID,
        block_ids: list[int],
    ) -> None:
        pass

    def shutdown(self) -> None:
        pass

    @property
    def num_host_blocks(self) -> int:
        return 0

    @property
    def num_used_host_blocks(self) -> int:
        return 0

    def reset_prefix_cache(self) -> None:
        pass

    @property
    def metrics(self) -> KVCacheMetrics:
        return KVCacheMetrics()
