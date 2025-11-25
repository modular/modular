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

import logging
import math
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from operator import mul

from max.dtype import DType
from max.support.human_readable_formatter import to_human_readable_bytes

logger = logging.getLogger("max.pipelines")


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

    num_layers: int
    """Number of layers in the model."""

    enable_prefix_caching: bool = False
    """Whether to enable prefix caching for efficient reuse of common prompt prefixes."""

    enable_kvcache_swapping_to_host: bool = False
    """Whether to enable swapping of KV cache blocks to host memory when device memory is full."""

    host_kvcache_swap_space_gb: float | None = None
    """Amount of host memory (in GB) to reserve for KV cache swapping. Required when swapping is enabled."""

    cache_strategy: KVCacheStrategy = KVCacheStrategy.PAGED
    """Strategy to use for managing the KV cache."""

    page_size: int = 128
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
    def tensor_parallel_degree(self) -> int:
        """Returns the tensor parallel degree.

        Returns:
            The tensor parallel degree.
        """
        return self.n_devices // self.data_parallel_degree

    @property
    def dtype_shorthand(self) -> str:
        """Returns a shorthand textual representation of the data type.

        Returns:
            "bf16" for bfloat16 dtype, "f32" otherwise.
        """
        return "bf16" if self.dtype == DType.bfloat16 else "f32"

    @property
    def shape_per_block(self) -> list[int]:
        """Returns the shape of each cache block.

        Returns:
            The shape of the cache block.
        """
        # split k and v caches across a single dim
        # 0 = key
        # 1 = value
        kv_dim = 2 if not self.is_mla else 1
        return [
            kv_dim,
            self.num_layers,
            self.page_size,
            self.n_kv_heads_per_device,
            self.head_dim,
        ]

    @property
    def bytes_per_block(self) -> int:
        """Returns the number of bytes per cache block.

        When TP>1, each block is sharded across the devices in the tensor parallel group.
        This method returns the total memory needed to store a block across these devices.

        Returns:
            The number of bytes per cache block.
        """
        return (
            reduce(mul, self.shape_per_block, 1)
            * self.dtype.size_in_bytes
            * self.tensor_parallel_degree
        )

    def compute_num_device_blocks(
        self,
        available_cache_memory: int,
        max_batch_size: int | None,
        max_seq_len: int | None,
    ) -> int:
        """Computes the number of blocks that can be allocated based on the available cache memory.

        The number of blocks returned is for a single replica. Each replica will
        have the same number of blocks.

        Args:
            available_cache_memory: The amount of cache memory available across all devices.
            max_batch_size: The maximum batch size, or None.
            max_seq_len: The maximum sequence length, or None.

        Returns:
            The number of blocks that can be allocated for a single replica.
        """
        # Compute upper bound of total number of pages required.
        max_blocks_per_req: int | None = None
        max_total_blocks: int | None = None
        if max_seq_len is not None and max_batch_size is not None:
            max_blocks_per_req = math.ceil(max_seq_len / self.page_size)
            max_total_blocks = max_blocks_per_req * max_batch_size

        # Compute total number of blocks allocatable based on available memory.
        available_cache_memory_per_replica = (
            available_cache_memory // self.data_parallel_degree
        )
        num_allocable_blocks = (
            available_cache_memory_per_replica // self.bytes_per_block
        )

        if max_total_blocks is not None:
            num_blocks = min(num_allocable_blocks, max_total_blocks)
        else:
            num_blocks = num_allocable_blocks

        # Check if we are allocating sufficient blocks.
        # If not, raise a warning or error.
        single_page_size_bytes_str = to_human_readable_bytes(
            self.bytes_per_block
        )
        cache_memory_str = to_human_readable_bytes(
            available_cache_memory_per_replica
        )
        devices_per_replica = self.n_devices // self.data_parallel_degree
        across_x_devices_str = (
            f" across {devices_per_replica} devices"
            if devices_per_replica > 1
            else ""
        )
        if num_allocable_blocks == 0:
            raise RuntimeError(
                f"Insufficient cache memory to allocate even a single page.\n"
                f"One page requires {single_page_size_bytes_str} but only "
                f"{cache_memory_str} are available{across_x_devices_str}."
            )

        if max_batch_size is not None and max_batch_size > num_allocable_blocks:
            memory_needed_str = to_human_readable_bytes(
                max_batch_size * self.bytes_per_block
            )
            logger.warning(
                f"Insufficient cache memory to support a batch containing {max_batch_size} "
                f"requests with one token per request. Need to allocate at least {max_batch_size} "
                f"pages ({memory_needed_str}), but only have enough memory for {num_allocable_blocks} "
                f"pages ({cache_memory_str}{across_x_devices_str})."
            )

        if (
            max_blocks_per_req is not None
            and max_blocks_per_req > num_allocable_blocks
        ):
            memory_needed_str = to_human_readable_bytes(
                max_blocks_per_req * self.bytes_per_block
            )
            logger.warning(
                f"Insufficient cache memory to support a batch containing one request "
                f"at the max sequence length of {max_seq_len} tokens. "
                f"Need to allocate at least {max_blocks_per_req} "
                f"pages ({memory_needed_str}), but only have enough memory for "
                f"{num_allocable_blocks} pages ({cache_memory_str}{across_x_devices_str})."
            )

        return num_blocks

    def estimated_memory_size(
        self, available_cache_memory: int, max_batch_size: int, max_seq_len: int
    ) -> int:
        """Computes the estimated memory size of the KV cache used by all replicas.

        Args:
            available_cache_memory: The amount of cache memory available across all devices.
            max_batch_size: The maximum batch size.
            max_seq_len: The maximum sequence length.

        Returns:
            The estimated memory usage of the KV cache in bytes.
        """
        num_device_blocks = self.compute_num_device_blocks(
            available_cache_memory=available_cache_memory,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )
        return (
            num_device_blocks * self.bytes_per_block * self.data_parallel_degree
        )

    def compute_max_seq_len_fitting_in_cache(
        self, available_cache_memory: int
    ) -> int:
        """Computes the maximum sequence length that can fit in the available cache memory.

        Args:
            available_cache_memory: The amount of cache memory available across all devices.

        Returns:
            The maximum sequence length that can fit in the available cache memory.
        """
        num_blocks = self.compute_num_device_blocks(
            available_cache_memory=available_cache_memory,
            max_batch_size=1,
            # Do not limit the sequence length.
            max_seq_len=None,
        )
        return num_blocks * self.page_size

    def compute_num_host_blocks(self) -> int:
        """Computes the number of blocks that can be allocated to the host.

        Returns:
            The number of blocks that can be allocated to the host.
        """
        if not self.enable_kvcache_swapping_to_host:
            return 0
        assert self.host_kvcache_swap_space_gb is not None
        GiB = 1024 * 1024 * 1024
        host_gb_per_replica = self.host_kvcache_swap_space_gb
        host_bytes_per_replica = host_gb_per_replica * GiB
        num_host_blocks = int(host_bytes_per_replica // self.bytes_per_block)

        if num_host_blocks == 0:
            raise RuntimeError(
                f"Insufficient cache memory to allocate even a single page.\n"
                f"One page requires {to_human_readable_bytes(self.bytes_per_block)} but only "
                f"{to_human_readable_bytes(host_gb_per_replica * GiB)} are available on host."
            )

        return num_host_blocks

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
            num_layers=self.num_layers,
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
