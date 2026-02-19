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

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Literal, Protocol, runtime_checkable

from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.support.human_readable_formatter import to_human_readable_bytes

from .data_parallelism_utils import split_into_groups
from .input_types import (
    InputSymbolInterface,
    MultiKVCacheInputSymbols,
    PagedCacheInputSymbols,
    PagedCacheInputSymbolsByReplica,
)

logger = logging.getLogger("max.pipelines")


KVCacheStrategy = Literal["model_default", "paged"]
"""Supported KV cache strategies for attention mechanisms.

This defines the different strategies for managing key-value caches
in transformer models during inference.
"""


def kernel_substring(strategy: KVCacheStrategy) -> str:
    """Returns the common substring included in the kernel name for this caching strategy.

    Args:
        strategy: The cache strategy.

    Returns:
        The string representation of the cache strategy value.
    """
    return strategy


def uses_opaque(strategy: KVCacheStrategy) -> bool:
    """Determines if this cache strategy uses opaque cache implementations.

    Args:
        strategy: The cache strategy to check.

    Returns:
        True if the strategy uses opaque caching, False otherwise.
    """
    return True


@dataclass
class KVCacheQuantizationConfig:
    """Configuration for KVCache quantization.

    Currently only FP8 Quantization is supported.
    """

    scale_dtype: DType = DType.float32
    """Data type of quantization scales, if quantization is enabled"""

    quantization_granularity: int = 128
    """Block-size used for KVCache quantization along head-dimension (e.g. 128)."""


@runtime_checkable
class KVCacheParamInterface(Protocol):
    """Interface for KV cache parameters."""

    cache_strategy: KVCacheStrategy
    page_size: int
    data_parallel_degree: int
    n_devices: int

    @property
    def bytes_per_block(self) -> int:
        """Number of bytes per cache block."""
        ...

    def get_symbolic_inputs(self) -> InputSymbolInterface:
        """Returns the symbolic inputs for the KV cache."""
        ...


@dataclass
class KVCacheParams(KVCacheParamInterface):
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

    devices: Sequence[DeviceRef]
    """Devices to use for the KV cache."""

    enable_prefix_caching: bool = False
    """Whether to enable prefix caching for efficient reuse of common prompt prefixes."""

    enable_kvcache_swapping_to_host: bool = False
    """Whether to enable swapping of KV cache blocks to host memory when device memory is full."""

    host_kvcache_swap_space_gb: float | None = None
    """Amount of host memory (in GB) to reserve for KV cache swapping. Required when swapping is enabled."""

    cache_strategy: KVCacheStrategy = "paged"
    """Strategy to use for managing the KV cache."""

    page_size: int = 128
    """Number of tokens per page (block) when using the paged cache strategy.

    This value is expressed in tokens, not bytes. The byte footprint of a page is
    derived from pipeline configuration.

    Current constraints: the page size must be a multiple of 128 and at least 128.
    Required when ``cache_strategy`` is ``"paged"``.
    """

    is_mla: bool = False
    """Whether the model uses Multi-Latent Attention (MLA) architecture."""

    data_parallel_degree: int = 1
    """Degree of data parallelism. Must be 1 or equal to n_devices (DP+TP not yet supported)."""

    n_kv_heads_per_device: int = 0
    """Number of KV heads allocated to each device. Computed automatically in __post_init__."""

    kvcache_quant_config: KVCacheQuantizationConfig | None = None
    """KVCache quantization config. Currently only FP8 quantization supported."""

    lmcache_config_file: str | None = None
    """Path to LMCache YAML config file. Enables LMCache external KV cache tiering."""

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
        if self.enable_prefix_caching and self.cache_strategy != "paged":
            raise ValueError(
                "Prefix caching is only supported for paged cache strategy"
            )
        if (
            self.enable_kvcache_swapping_to_host
            and self.cache_strategy != "paged"
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
        if self.page_size is None and self.cache_strategy == "paged":
            raise ValueError("Page size is required for paged cache strategy")

        if self.quantized_kv_cache and self.kvcache_quant_config is not None:
            # Validate FP8 KVCache quantization granularity.
            if (
                self.head_dim
                % self.kvcache_quant_config.quantization_granularity
                != 0
            ):
                raise ValueError(
                    "KVCache quantization granularity must evenly divide KV head dimension."
                )
            if self.kvcache_quant_config is None:
                raise ValueError("KVCache quantization config required.")

    @property
    def quantized_kv_cache(self) -> bool:
        # Currently only FP8_E4M3 KVCache quantization supported.
        return self.dtype in (DType.float8_e4m3fn, DType.float8_e4m3fnuz)

    @property
    def n_devices(self) -> int:
        """Returns the number of devices.

        Returns:
            The number of devices.
        """
        return len(self.devices)

    @n_devices.setter  # Required for protocol.
    def n_devices(self, value: int) -> None:
        raise ValueError("n_devices is read-only")

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
        if self.dtype == DType.bfloat16:
            return "bf16"
        elif self.dtype == DType.float8_e4m3fn:
            return "f8_m4e3fn"
        else:
            return "f32"

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
    def shape_per_scale_block(self) -> list[int]:
        """Returns the shape of each scale block used for KVCache quantization

        Returns:
            The shape of the KVCache quantization scales block.
        """
        assert self.kvcache_quant_config is not None
        shape_per_block = self.shape_per_block
        # The final dimension is (head_dim / quantization_granularity).
        shape_per_block[4] = (
            shape_per_block[4]
            // self.kvcache_quant_config.quantization_granularity
        )
        return shape_per_block

    @property
    def bytes_per_block(self) -> int:
        """Returns the number of bytes per cache block.

        When TP>1, each block is sharded across the devices in the tensor parallel group.
        This method returns the total memory needed to store a block across these devices.
        Includes memory needed for scales if quantization is enabled.

        Returns:
            The number of bytes per cache block.
        """
        base_bytes = (
            reduce(mul, self.shape_per_block, 1)
            * self.dtype.size_in_bytes
            * self.tensor_parallel_degree
        )
        if self.quantized_kv_cache and self.kvcache_quant_config is not None:
            # Add bytes needed to store the quantization scales.
            scale_bytes = (
                reduce(mul, self.shape_per_scale_block, 1)
                * self.kvcache_quant_config.scale_dtype.size_in_bytes
                * self.tensor_parallel_degree
            )
            base_bytes += scale_bytes
        return base_bytes

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

        devices_per_replica = split_into_groups(
            self.devices, self.data_parallel_degree
        )

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
            devices=devices_per_replica[0],
            is_mla=self.is_mla,
            data_parallel_degree=1,
            kvcache_quant_config=self.kvcache_quant_config,
            lmcache_config_file=self.lmcache_config_file,
        )

    def _get_symbolic_inputs_for_replica(
        self, devices: Sequence[DeviceRef], replica_idx: int
    ) -> list[PagedCacheInputSymbols]:
        """Computes the symbolic inputs for a single replica.

        Returns:
            The symbolic inputs for the KV cache.
        """

        dynamic_dim_prefix = f"replica_{replica_idx}_"

        kv_cache_scale_dtype = DType.float32
        if self.quantized_kv_cache and self.kvcache_quant_config is not None:
            kv_cache_scale_dtype = self.kvcache_quant_config.scale_dtype
        return [
            PagedCacheInputSymbols(
                kv_blocks=BufferType(
                    self.dtype,
                    shape=[
                        "total_num_pages",
                        *self.shape_per_block,
                    ],
                    device=device,
                ),
                cache_lengths=TensorType(
                    DType.uint32,
                    shape=[dynamic_dim_prefix + "batch_size"],
                    device=device,
                ),
                lookup_table=TensorType(
                    DType.uint32,
                    shape=[
                        dynamic_dim_prefix + "batch_size",
                        dynamic_dim_prefix + "max_num_pages",
                    ],
                    device=device,
                ),
                max_lengths=TensorType(
                    DType.uint32,
                    shape=[dynamic_dim_prefix + "steps_remaining", 2],
                    device=DeviceRef.CPU(),
                ),
                kv_scales=BufferType(
                    kv_cache_scale_dtype,
                    shape=["total_num_pages", *self.shape_per_scale_block],
                    device=device,
                )
                if self.quantized_kv_cache
                else None,
            )
            for device in devices
        ]

    def get_symbolic_inputs(self) -> PagedCacheInputSymbolsByReplica:
        """Computes the symbolic inputs for the KV cache.

        This method returns a list of PagedCacheInputSymbols for each replica.
        This is used when constructing the model graph.

        Returns:
            The symbolic inputs for the KV cache.
        """
        devices_per_replica = split_into_groups(
            self.devices, self.data_parallel_degree
        )
        input_symbols: list[PagedCacheInputSymbols] = []
        for replica_idx, devices in enumerate(devices_per_replica):
            symbols = self._get_symbolic_inputs_for_replica(
                devices,
                replica_idx,
            )
            input_symbols.extend(symbols)
        return PagedCacheInputSymbolsByReplica(values=input_symbols)


@dataclass(frozen=True)
class MultiKVCacheParams(KVCacheParamInterface):
    """Aggregates multiple KV cache parameter sets.

    This class implements KVCacheParamInterface by aggregating multiple
    KVCacheParamInterface instances. Useful for models with multiple distinct
    KV caches (e.g., different cache configurations for different layers).
    """

    params: Sequence[KVCacheParamInterface]
    """List of KV cache parameter sets to aggregate."""

    cache_strategy: KVCacheStrategy
    page_size: int
    data_parallel_degree: int
    n_devices: int

    @classmethod
    def from_params(cls, *params: KVCacheParamInterface) -> MultiKVCacheParams:
        if len(params) == 0:
            raise ValueError("MultiKVCacheParams requires at least one param.")
        return cls(
            params=params,
            cache_strategy=params[0].cache_strategy,
            page_size=params[0].page_size,
            data_parallel_degree=params[0].data_parallel_degree,
            n_devices=params[0].n_devices,
        )

    def __post_init__(self) -> None:
        """Validates that all params have consistent cache strategy and page size."""
        if not self.params:
            raise ValueError(
                "MultiKVCacheParams requires at least one param set."
            )

        strategies = {p.cache_strategy for p in self.params}
        if len(strategies) > 1:
            raise ValueError(
                f"All params must use the same cache strategy, got: {strategies}"
            )

        page_sizes = {p.page_size for p in self.params}
        if len(page_sizes) > 1:
            raise ValueError(
                f"All params must use the same page size, got: {page_sizes}"
            )

        data_parallel_degrees = {p.data_parallel_degree for p in self.params}
        if len(data_parallel_degrees) > 1:
            raise ValueError(
                f"All params must use the same data parallel degree, got: {data_parallel_degrees}"
            )

        n_devices = {p.n_devices for p in self.params}
        if len(n_devices) > 1:
            raise ValueError(
                f"All params must use the same number of devices, got: {n_devices}"
            )

    @property
    def bytes_per_block(self) -> int:
        """Total bytes per block across all KV caches.

        Since all caches allocate memory for the same sequence, the total
        memory cost per block is the sum across all param sets.
        """
        return sum(p.bytes_per_block for p in self.params)

    def get_symbolic_inputs(self) -> MultiKVCacheInputSymbols:
        """Returns the symbolic inputs for the KV cache."""
        return MultiKVCacheInputSymbols(
            [p.get_symbolic_inputs() for p in self.params]
        )


def compute_num_device_blocks(
    params: KVCacheParamInterface,
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
        max_blocks_per_req = math.ceil(max_seq_len / params.page_size)
        max_total_blocks = max_blocks_per_req * max_batch_size

    # Compute total number of blocks allocatable based on available memory.
    available_cache_memory_per_replica = (
        available_cache_memory // params.data_parallel_degree
    )
    num_allocable_blocks = (
        available_cache_memory_per_replica // params.bytes_per_block
    )

    if max_total_blocks is not None:
        num_blocks = min(num_allocable_blocks, max_total_blocks)
    else:
        num_blocks = num_allocable_blocks

    # Check if we are allocating sufficient blocks.
    # If not, raise a warning or error.
    single_page_size_bytes_str = to_human_readable_bytes(params.bytes_per_block)
    cache_memory_str = to_human_readable_bytes(
        available_cache_memory_per_replica
    )
    devices_per_replica = params.n_devices // params.data_parallel_degree
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
            max_batch_size * params.bytes_per_block
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
            max_blocks_per_req * params.bytes_per_block
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
    params: KVCacheParamInterface,
    available_cache_memory: int,
    max_batch_size: int,
    max_seq_len: int,
) -> int:
    """Computes the estimated memory size of the KV cache used by all replicas.

    Args:
        available_cache_memory: The amount of cache memory available across all devices.
        max_batch_size: The maximum batch size.
        max_seq_len: The maximum sequence length.

    Returns:
        The estimated memory usage of the KV cache in bytes.
    """
    num_device_blocks = compute_num_device_blocks(
        available_cache_memory=available_cache_memory,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        params=params,
    )
    return (
        num_device_blocks * params.bytes_per_block * params.data_parallel_degree
    )


def compute_max_seq_len_fitting_in_cache(
    params: KVCacheParamInterface,
    available_cache_memory: int,
) -> int:
    """Computes the maximum sequence length that can fit in the available memory.

    Args:
        available_cache_memory: The amount of cache memory available across
        all devices.

    Returns:
        The maximum sequence length that can fit in the available cache memory.
    """
    if params.bytes_per_block == 0:
        raise ValueError("bytes_per_block cannot be zero")
    num_blocks = compute_num_device_blocks(
        params=params,
        available_cache_memory=available_cache_memory,
        max_batch_size=1,
        # Do not limit the sequence length.
        max_seq_len=None,
    )
    return num_blocks * params.page_size
