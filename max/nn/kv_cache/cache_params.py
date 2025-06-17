# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
from enum import Enum
from typing import Optional

from max.dtype import DType


class KVCacheStrategy(str, Enum):
    MODEL_DEFAULT = "model_default"
    CONTINUOUS = "continuous"
    """Deprecated. Use ``PAGED`` instead."""
    PAGED = "paged"

    def kernel_substring(self) -> str:
        """Returns the common substring that we include in the kernel name for this caching strategy."""
        if self == KVCacheStrategy.CONTINUOUS:
            return "continuous_batching"
        return self.value

    def uses_opaque(self) -> bool:
        return True


@dataclass
class KVCacheParams:
    dtype: DType
    n_kv_heads: int
    head_dim: int
    enable_prefix_caching: bool = False
    enable_kvcache_swapping_to_host: bool = False
    host_kvcache_swap_space_gb: Optional[float] = None
    cache_strategy: KVCacheStrategy = KVCacheStrategy.CONTINUOUS
    page_size: Optional[int] = None
    n_devices: int = 1

    def __post_init__(self):
        self.n_kv_heads_per_device = max(self.n_kv_heads // self.n_devices, 1)

        # Validate inputs
        if (
            self.enable_prefix_caching
            and self.cache_strategy != KVCacheStrategy.PAGED
        ):
            raise ValueError(
                "Prefix caching is only supported for paged cache strategy"
            )
        if (
            self.enable_kvcache_swapping_to_host
            and self.cache_strategy != KVCacheStrategy.PAGED
        ):
            raise ValueError(
                "KVCache swapping to host is only supported for paged cache strategy"
            )
        if (
            self.enable_kvcache_swapping_to_host
            and not self.enable_prefix_caching
        ):
            raise ValueError(
                "KVCache swapping to host is only supported when prefix caching is enabled"
            )
        if (
            self.enable_kvcache_swapping_to_host
            and self.host_kvcache_swap_space_gb is None
        ):
            raise ValueError(
                "host_kvcache_swap_space_gb is required when kvcache_swapping_to_host is enabled"
            )
        if (
            self.page_size is None
            and self.cache_strategy == KVCacheStrategy.PAGED
        ):
            raise ValueError("Page size is required for paged cache strategy")

    @property
    def dtype_shorthand(self) -> str:
        """The textual representation in shorthand of the dtype."""
        return "bf16" if self.dtype == DType.bfloat16 else "f32"

    @property
    def static_cache_shape(self) -> tuple[str, str, str, str, str]:
        return (
            "num_layers",
            "batch_size",
            "seq_len",
            "n_kv_heads",
            "head_dim",
        )
