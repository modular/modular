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

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum

from max.dtype import DType


class KVCacheStrategy(str, Enum):
    MODEL_DEFAULT = "model_default"
    PAGED = "paged"

    def kernel_substring(self) -> str:
        """Returns the common substring that we include in the kernel name for this caching strategy."""
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
    host_kvcache_swap_space_gb: float | None = None
    cache_strategy: KVCacheStrategy = KVCacheStrategy.PAGED
    page_size: int | None = None
    n_devices: int = 1
    is_mla: bool = False

    data_parallel_degree: int = 1

    # Computed fields (set in __post_init__)
    n_kv_heads_per_device: int = 0  # Will be computed

    def __post_init__(self):
        if self.data_parallel_degree > 1:
            if self.n_devices < self.data_parallel_degree:
                raise ValueError(
                    f"Data parallelism degree ({self.data_parallel_degree}) cannot be greater than the number of devices ({self.n_devices})"
                )
            if self.data_parallel_degree < self.n_devices:
                raise ValueError(
                    f"We do not yet support DP + TP at the same time. Found {self.data_parallel_degree=} and {self.n_devices=}"
                )
            self.n_kv_heads_per_device = self.n_kv_heads
        else:
            # Tensor parallel mode: shard by heads, keep all layers per device
            if self.n_kv_heads % self.n_devices != 0:
                raise ValueError(
                    f"Number of KV heads ({self.n_kv_heads}) must be divisible by the number of devices ({self.n_devices})"
                )
            self.n_kv_heads_per_device = max(
                self.n_kv_heads // self.n_devices, 1
            )

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

    def copy_as_dp_1(self) -> KVCacheParams:
        """Create a copy of the KVCacheParams as if data parallelism is disabled."""
        cloned = copy.deepcopy(self)
        assert cloned.n_devices % self.data_parallel_degree == 0
        cloned.n_devices //= self.data_parallel_degree
        cloned.data_parallel_degree = 1
        return cloned
