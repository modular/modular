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
        )
