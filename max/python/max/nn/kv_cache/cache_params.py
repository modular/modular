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

from dataclasses import dataclass
from enum import Enum

from max.dtype import DType


class KVCacheStrategy(str, Enum):
    """Enumeration of supported KV cache strategies for attention mechanisms.

    This enum defines the different strategies for managing key-value caches
    in transformer models during inference.
    """

    MODEL_DEFAULT = "model_default"
    """Use the model's default caching strategy."""

    PAGED = "paged"
    """Use paged attention for efficient memory management."""

    def kernel_substring(self) -> str:
        """Returns the common substring included in the kernel name for this caching strategy.

        Returns:
            The string representation of the cache strategy value.
        """
        return self.value

    def uses_opaque(self) -> bool:
        """Determines if this cache strategy uses opaque cache implementations.

        Returns:
            True if the strategy uses opaque caching, False otherwise.
        """
        return True


@dataclass
class KVCacheParams:
    """Configuration parameters for key-value cache management in transformer models.

    This class encapsulates all configuration options for managing KV caches during
    inference, including parallelism settings, memory management, and cache strategy.
    """

    dtype: DType
    """Data type for storing key and value tensors in the cache."""

    n_kv_heads: int
    """Total number of key-value attention heads across all devices."""

    head_dim: int
    """Dimensionality of each attention head."""

    enable_prefix_caching: bool = False
    """Whether to enable prefix caching for efficient reuse of common prompt prefixes."""

    enable_kvcache_swapping_to_host: bool = False
    """Whether to enable swapping of KV cache blocks to host memory when device memory is full."""

    host_kvcache_swap_space_gb: float | None = None
    """Amount of host memory (in GB) to reserve for KV cache swapping. Required when swapping is enabled."""

    cache_strategy: KVCacheStrategy = KVCacheStrategy.PAGED
    """Strategy to use for managing the KV cache."""

    page_size: int | None = None
    """Number of tokens per page (block) when using the paged cache strategy.

    This value is expressed in tokens, not bytes. The byte footprint of a page is
    derived from pipeline configuration.

    Current constraints: the page size must be a multiple of 128 and at least 128.
    Required when ``cache_strategy`` is ``KVCacheStrategy.PAGED``.
    """

    n_devices: int = 1
    """Total number of devices (GPUs/accelerators) available for inference."""

    is_mla: bool = False
    """Whether the model uses Multi-Latent Attention (MLA) architecture."""

    data_parallel_degree: int = 1
    """Degree of data parallelism. Must be 1 or equal to n_devices (DP+TP not yet supported)."""

    n_kv_heads_per_device: int = 0
    """Number of KV heads allocated to each device. Computed automatically in __post_init__."""

    def __post_init__(self):
        """Validates configuration and computes derived fields after initialization.

        This method:
        - Validates parallelism configuration (data parallel vs tensor parallel)
        - Computes n_kv_heads_per_device based on parallelism strategy
        - Validates cache strategy compatibility with enabled features

        Raises:
            ValueError: If configuration parameters are invalid or incompatible.
        """
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
        elif self.is_mla:
            # MLA always caches one latent vector per device.
            self.n_kv_heads_per_device = 1
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
        """Returns a shorthand textual representation of the data type.

        Returns:
            "bf16" for bfloat16 dtype, "f32" otherwise.
        """
        return "bf16" if self.dtype == DType.bfloat16 else "f32"

    def copy_as_dp_1(self) -> KVCacheParams:
        """Creates a copy of the KVCacheParams with data parallelism disabled.

        This method creates a new instance of the current configuration and adjusts
        the device count to reflect a tensor-parallel-only setup (data_parallel_degree=1).
        The number of devices is divided by the current data parallel degree.

        Returns:
            A new KVCacheParams instance with data_parallel_degree set to 1.

        Raises:
            ValueError: If n_devices is not evenly divisible by data_parallel_degree.
        """
        if self.n_devices % self.data_parallel_degree != 0:
            raise ValueError(
                f"Number of devices ({self.n_devices}) must be evenly divisible "
                f"by data parallel degree ({self.data_parallel_degree})"
            )

        new_n_devices = self.n_devices // self.data_parallel_degree

        return KVCacheParams(
            dtype=self.dtype,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            enable_prefix_caching=self.enable_prefix_caching,
            enable_kvcache_swapping_to_host=self.enable_kvcache_swapping_to_host,
            host_kvcache_swap_space_gb=self.host_kvcache_swap_space_gb,
            cache_strategy=self.cache_strategy,
            page_size=self.page_size,
            n_devices=new_n_devices,
            is_mla=self.is_mla,
            data_parallel_degree=1,
        )
