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
from dataclasses import dataclass

from typing_extensions import Self


@dataclass
class KVCacheMetrics:
    """Metrics for the KV cache.

    Tracks token usage and block transfer statistics for KV cache operations.
    """

    input_tokens: int = 0
    """Number of tokens processed as new input (cache misses)."""
    cache_tokens: int = 0
    """Number of tokens retrieved from cache (cache hits)."""
    h2d_blocks_copied: int = 0
    """Number of cache blocks copied from host to device."""
    d2h_blocks_copied: int = 0
    """Number of cache blocks copied from device to host."""
    disk_blocks_written: int = 0
    """Number of cache blocks written to disk."""
    disk_blocks_read: int = 0
    """Number of cache blocks read from disk."""
    nixl_read_blocks: int = 0
    """Number of cache blocks read via NIXL (dKV GET)."""
    nixl_write_blocks: int = 0
    """Number of cache blocks written via NIXL (dKV PUT)."""

    # dKV latency pairs: total_ms + count sum correctly across DP replicas.
    nixl_read_latency_total_ms: float = 0.0
    """Cumulative NIXL READ transfer latency in milliseconds."""
    nixl_read_latency_count: int = 0
    """Number of NIXL READ transfer completions."""
    nixl_write_latency_total_ms: float = 0.0
    """Cumulative NIXL WRITE transfer latency in milliseconds."""
    nixl_write_latency_count: int = 0
    """Number of NIXL WRITE transfer completions."""
    rpc_acquire_latency_total_ms: float = 0.0
    """Cumulative dKV acquire_blocks RPC latency in milliseconds."""
    rpc_acquire_latency_count: int = 0
    """Number of acquire_blocks RPC calls."""
    rpc_read_latency_total_ms: float = 0.0
    """Cumulative dKV read_blocks RPC latency in milliseconds."""
    rpc_read_latency_count: int = 0
    """Number of read_blocks RPC calls."""
    nixl_read_bytes: int = 0
    """Total bytes transferred via NIXL READ."""
    nixl_write_bytes: int = 0
    """Total bytes transferred via NIXL WRITE."""
    nixl_read_blocks_local: int = 0
    """NIXL reads from co-located (default) block store."""
    nixl_read_blocks_remote: int = 0
    """NIXL reads from non-default (remote) block stores."""

    @property
    def prompt_tokens(self) -> int:
        """Total number of prompt tokens (input + cached).

        Returns:
            Sum of input_tokens and cache_tokens.
        """
        return self.input_tokens + self.cache_tokens

    @property
    def cache_hit_rate(self) -> float:
        """Proportion of prompt tokens that were retrieved from cache.

        Returns:
            Ratio of cache_tokens to total prompt_tokens, or 0.0 if no tokens
            were processed.
        """
        if self.prompt_tokens == 0:
            return 0.0
        return self.cache_tokens / self.prompt_tokens

    @property
    def nixl_read_latency_avg_ms(self) -> float:
        """Average NIXL READ transfer latency in milliseconds."""
        if self.nixl_read_latency_count == 0:
            return 0.0
        return self.nixl_read_latency_total_ms / self.nixl_read_latency_count

    @property
    def nixl_write_latency_avg_ms(self) -> float:
        """Average NIXL WRITE transfer latency in milliseconds."""
        if self.nixl_write_latency_count == 0:
            return 0.0
        return self.nixl_write_latency_total_ms / self.nixl_write_latency_count

    @property
    def rpc_acquire_latency_avg_ms(self) -> float:
        """Average dKV acquire_blocks RPC latency in milliseconds."""
        if self.rpc_acquire_latency_count == 0:
            return 0.0
        return (
            self.rpc_acquire_latency_total_ms / self.rpc_acquire_latency_count
        )

    @property
    def rpc_read_latency_avg_ms(self) -> float:
        """Average dKV read_blocks RPC latency in milliseconds."""
        if self.rpc_read_latency_count == 0:
            return 0.0
        return self.rpc_read_latency_total_ms / self.rpc_read_latency_count

    @property
    def nixl_read_gib_per_s(self) -> float:
        """NIXL READ throughput in GiB/s."""
        if self.nixl_read_latency_total_ms <= 0:
            return 0.0
        return (self.nixl_read_bytes / (1 << 30)) / (
            self.nixl_read_latency_total_ms / 1000
        )

    @property
    def nixl_write_gib_per_s(self) -> float:
        """NIXL WRITE throughput in GiB/s."""
        if self.nixl_write_latency_total_ms <= 0:
            return 0.0
        return (self.nixl_write_bytes / (1 << 30)) / (
            self.nixl_write_latency_total_ms / 1000
        )

    @property
    def remote_read_ratio(self) -> float:
        """Fraction of NIXL reads hitting non-default (remote) block stores."""
        total = self.nixl_read_blocks_local + self.nixl_read_blocks_remote
        if total == 0:
            return 0.0
        return self.nixl_read_blocks_remote / total

    def __add__(self, other: Self) -> Self:
        """Combine two KVCacheMetrics by summing their respective fields.

        Args:
            other: Another KVCacheMetrics instance to add.

        Returns:
            A new KVCacheMetrics instance with summed values.
        """
        return type(self)(
            input_tokens=self.input_tokens + other.input_tokens,
            cache_tokens=self.cache_tokens + other.cache_tokens,
            h2d_blocks_copied=self.h2d_blocks_copied + other.h2d_blocks_copied,
            d2h_blocks_copied=self.d2h_blocks_copied + other.d2h_blocks_copied,
            disk_blocks_written=self.disk_blocks_written
            + other.disk_blocks_written,
            disk_blocks_read=self.disk_blocks_read + other.disk_blocks_read,
            nixl_read_blocks=self.nixl_read_blocks + other.nixl_read_blocks,
            nixl_write_blocks=self.nixl_write_blocks + other.nixl_write_blocks,
            nixl_read_latency_total_ms=self.nixl_read_latency_total_ms
            + other.nixl_read_latency_total_ms,
            nixl_read_latency_count=self.nixl_read_latency_count
            + other.nixl_read_latency_count,
            nixl_write_latency_total_ms=self.nixl_write_latency_total_ms
            + other.nixl_write_latency_total_ms,
            nixl_write_latency_count=self.nixl_write_latency_count
            + other.nixl_write_latency_count,
            rpc_acquire_latency_total_ms=self.rpc_acquire_latency_total_ms
            + other.rpc_acquire_latency_total_ms,
            rpc_acquire_latency_count=self.rpc_acquire_latency_count
            + other.rpc_acquire_latency_count,
            rpc_read_latency_total_ms=self.rpc_read_latency_total_ms
            + other.rpc_read_latency_total_ms,
            rpc_read_latency_count=self.rpc_read_latency_count
            + other.rpc_read_latency_count,
            nixl_read_bytes=self.nixl_read_bytes + other.nixl_read_bytes,
            nixl_write_bytes=self.nixl_write_bytes + other.nixl_write_bytes,
            nixl_read_blocks_local=self.nixl_read_blocks_local
            + other.nixl_read_blocks_local,
            nixl_read_blocks_remote=self.nixl_read_blocks_remote
            + other.nixl_read_blocks_remote,
        )
