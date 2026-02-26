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

"""Provides a no-op :class:`DummyKVCache` implementation for testing when KV caching is disabled."""

from __future__ import annotations

from typing import Any

from max.interfaces import RequestID

from .cache_manager import KVCacheMetrics, PagedKVCacheManager


class DummyKVCache(PagedKVCacheManager):
    """No-op KV cache implementation for testing or when cache is disabled."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the dummy cache with a single replica and no host swapping."""
        self.reqs = set[RequestID]()
        self.num_replicas = 1
        self.enable_kvcache_swapping_to_host = False

    def get_pct_used_blocks_after_allocation(
        self, *args: Any, **kwargs: Any
    ) -> float:
        """Returns a fixed low percentage (0.01)."""
        return 0.01

    def claim(self, request_id: RequestID, replica_idx: int) -> None:
        """No-op."""
        pass

    def alloc(self, *args: Any, **kwargs: Any) -> None:
        """No-op."""
        pass

    def step(self, *args: Any, **kwargs: Any) -> None:
        """No-op."""
        pass

    def contains(self, request_id: RequestID, replica_idx: int) -> bool:
        """Returns True for any request."""
        return True

    def release(self, request_id: RequestID, replica_idx: int) -> None:
        """No-op."""
        pass

    def get_num_pages(self, replica_idx: int) -> int:
        """Returns 1."""
        return 1

    def get_num_used_pages(self, replica_idx: int) -> int:
        """Returns 1."""
        return 1

    def get_num_host_pages(self, replica_idx: int) -> int:
        """Returns 1."""
        return 1

    def get_num_used_host_pages(self, replica_idx: int) -> int:
        """Returns 1."""
        return 1

    def get_metrics(self, replica_idx: int) -> KVCacheMetrics:
        """Returns empty metrics."""
        return KVCacheMetrics()

    def reset_metrics(self) -> None:
        """No-op."""
        pass
