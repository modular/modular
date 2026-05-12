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
import os
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from operator import mul
from typing import Any, Protocol, runtime_checkable

from max.driver import Buffer, DevicePinnedBuffer
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.support.human_readable_formatter import to_human_readable_bytes

from .data_parallelism_utils import split_into_groups
from .input_types import KVCacheInputs, KVCacheInputsPerDevice

logger = logging.getLogger("max.pipelines")


class KVConnectorType(str, Enum):
    """Type of KV cache connector to use."""

    null = "null"
    local = "local"
    tiered = "tiered"
    dkv = "dkv"


@dataclass
class KVCacheBuffer:
    """This is a collection of the KVCache buffers.

    There are two types of supported buffers today: values and scales.
    The scales are optional and used for FP8 quantization.

    The length of the list of buffers correspond to the tensor parallel degree
    where each buffer in the list corresponds to a single TP shard.

    For DP, we would have multiple instances of KVCacheBuffer per replica.
    """

    total_num_pages: int
    values: list[Buffer]
    scales: list[Buffer] | None = None

    def __post_init__(self) -> None:
        if self.total_num_pages <= 0:
            raise ValueError("Total number of pages must be strictly positive")

        if len(self.values) == 0:
            raise ValueError("List of values must be non-empty")

        if self.scales is None:
            return

        if len(self.scales) != len(self.values):
            raise ValueError("Scales must be the same length as values")

        for value, scale in zip(self.values, self.scales, strict=True):
            if value.device != scale.device:
                raise ValueError(
                    "Corresponding values and scales must be on the same device"
                )
            if isinstance(value, DevicePinnedBuffer) != isinstance(
                scale, DevicePinnedBuffer
            ):
                raise ValueError(
                    "Corresponding values and scales must be either both pinned or both non-pinned"
                )

    @property
    def all_buffers(self) -> list[Buffer]:
        """Returns all value and scale buffers in a single flat list.

        Returns:
            A list containing every value buffer followed by every scale
            buffer (if scales are present).
        """
        return [*self.values, *(self.scales if self.scales is not None else [])]


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

    page_size: int
    data_parallel_degree: int
    n_devices: int
    kv_connector: KVConnectorType | None
    host_kvcache_swap_space_gb: float | None
    num_eagle_speculative_tokens: int = 0

    @property
    def bytes_per_block(self) -> int:
        """Number of bytes per cache block."""
        ...

    def get_symbolic_inputs(
        self, prefix: str = ""
    ) -> KVCacheInputs[TensorType, BufferType]:
        """Returns the symbolic inputs for the KV cache."""
        ...


@dataclass
class KVCacheParams(KVCacheParamInterface):
    """Configuration parameters for key-value cache management in transformer models.

    This class encapsulates all configuration options for managing KV caches during
    inference, including parallelism settings, and memory management.
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

    kv_connector: KVConnectorType | None = None
    """Type of KV cache connector to use (null, local, tiered, dkv)."""

    kv_connector_config: Any = None
    """Connector-specific configuration (KVConnectorConfig from the pipelines layer)."""

    host_kvcache_swap_space_gb: float | None = None
    """Amount of host memory (in GB) to reserve for KV cache swapping. Required when local or tiered connector is used."""

    page_size: int = 128
    """Number of tokens per page (block).

    This value is expressed in tokens, not bytes. The byte footprint of a page is
    derived from pipeline configuration.

    Current constraints: the page size must be a multiple of 128 and at least 128.
    """

    is_mla: bool = False
    """Whether the model uses Multi-Latent Attention (MLA) architecture."""

    num_q_heads: int | None = None
    """Number of query attention heads. Required when ``is_mla`` is True so
    that the attention dispatch resolver can call the MLA-specific kernel."""

    data_parallel_degree: int = 1
    """Degree of data parallelism. Must be 1 or equal to n_devices (DP+TP not yet supported)."""

    n_kv_heads_per_device: int = 0
    """Number of KV heads allocated to each device. Computed automatically in __post_init__."""

    num_q_heads_per_device: int | None = None
    """Number of query heads per device. Computed automatically in __post_init__
    from ``num_q_heads`` and the parallelism configuration."""

    kvcache_quant_config: KVCacheQuantizationConfig | None = None
    """KVCache quantization config. Currently only FP8 quantization supported."""

    num_eagle_speculative_tokens: int = 0
    """Number of draft tokens to generate for EAGLE speculative decoding."""

    def __post_init__(self):
        """Validates configuration and computes derived fields after initialization.

        This method:
        - Validates parallelism configuration (data parallel vs tensor parallel)
        - Computes n_kv_heads_per_device based on parallelism strategy

        Raises:
            ValueError: If configuration parameters are invalid or incompatible.
        """
        if self.is_mla and self.num_q_heads is None:
            raise ValueError(
                "num_q_heads is required when is_mla=True so the attention"
                "dispatch resolver can use the MLA kernel."
            )

        if self.data_parallel_degree > 1:
            # Data parallel mode: simply duplicate the heads across all devices
            if self.n_devices < self.data_parallel_degree:
                raise ValueError(
                    f"Data parallelism degree ({self.data_parallel_degree}) cannot be greater than the number of devices ({self.n_devices})"
                )
            if self.data_parallel_degree < self.n_devices:
                raise ValueError(
                    f"We do not yet support DP + TP at the same time. Found {self.data_parallel_degree=} and {self.n_devices=}"
                )
            self.n_kv_heads_per_device = self.n_kv_heads
            self.num_q_heads_per_device = self.num_q_heads

        else:
            # Tensor parallel mode: shard by heads, keep all layers per device
            # First, resolve the number of KV heads per device
            if self.is_mla:
                self.n_kv_heads_per_device = 1
            else:
                if self.n_kv_heads % self.n_devices != 0:
                    raise ValueError(
                        f"Number of KV heads ({self.n_kv_heads}) must be divisible by the number of devices ({self.n_devices})"
                    )
                self.n_kv_heads_per_device = max(
                    self.n_kv_heads // self.n_devices, 1
                )

            # Then, resolve the number of query heads per device if it
            # is provided.
            if self.num_q_heads is not None:
                if self.num_q_heads % self.n_devices != 0:
                    raise ValueError(
                        f"Number of query heads ({self.num_q_heads}) must be divisible by the number of devices ({self.n_devices})"
                    )
                self.num_q_heads_per_device = max(
                    self.num_q_heads // self.n_devices, 1
                )

        # Validate connector configuration
        if self.kv_connector in (
            KVConnectorType.local,
            KVConnectorType.tiered,
        ):
            if not self.enable_prefix_caching:
                raise ValueError(
                    f"KV connector '{self.kv_connector.value}' requires prefix caching to be enabled"
                )
            if self.host_kvcache_swap_space_gb is None:
                raise ValueError(
                    f"host_kvcache_swap_space_gb is required when kv_connector is '{self.kv_connector.value}'"
                )

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
    def is_fp8_kv_dtype(self) -> bool:
        """Whether the KV cache stores FP8 data, for dispatch resolution.

        Unlike ``quantized_kv_cache`` (which also requires valid scale config),
        this checks only the storage dtype—matching the compile-time detection
        in the MLA decode kernel.

        TODO(SERVOPT-1094): Once SnapMLA uses a valid scale_dtype, this
        can be replaced by ``quantized_kv_cache``.
        """
        return self.dtype in (DType.float8_e4m3fn, DType.float8_e4m3fnuz)

    @property
    def quantized_kv_cache(self) -> bool:
        """Returns whether FP8 KV cache quantization is enabled.

        Returns:
            ``True`` when the cache dtype is ``float8_e4m3fn`` or
            ``float8_e4m3fnuz`` and a valid quantization scale dtype is
            configured; ``False`` otherwise.
        """
        # Currently only FP8_E4M3 KVCache quantization is supported.
        valid_scale = False
        if self.kvcache_quant_config is not None:
            valid_scale = self.kvcache_quant_config.scale_dtype in (
                DType.float32,
                DType.float8_e8m0fnu,
            )
        return (
            self.dtype in (DType.float8_e4m3fn, DType.float8_e4m3fnuz)
            and valid_scale
        )

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
    def replicates_kv_across_tp(self) -> bool:
        """Whether every device holds identical KV state."""
        return (
            self.is_mla
            and self.data_parallel_degree == 1
            and self.n_devices > 1
        )

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
        # The final dimension is ceil(head_dim / quantization_granularity).
        granularity = self.kvcache_quant_config.quantization_granularity
        shape_per_block[4] = math.ceil(shape_per_block[4] / granularity)
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

    def copy_as_dp_1(self, replica_idx: int = 0) -> KVCacheParams:
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

        # Build per-replica connector config with replica-specific disk path.
        replica_cfg = self.kv_connector_config
        if replica_cfg is not None and hasattr(replica_cfg, "model_copy"):
            disk_dir = getattr(replica_cfg, "disk_offload_dir", None)
            if disk_dir is not None:
                replica_cfg = replica_cfg.model_copy(
                    update={
                        "disk_offload_dir": os.path.join(
                            disk_dir, f"replica_{replica_idx}"
                        )
                    }
                )

        return KVCacheParams(
            dtype=self.dtype,
            num_layers=self.num_layers,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            enable_prefix_caching=self.enable_prefix_caching,
            kv_connector=self.kv_connector,
            kv_connector_config=replica_cfg,
            host_kvcache_swap_space_gb=self.host_kvcache_swap_space_gb,
            page_size=self.page_size,
            devices=devices_per_replica[replica_idx],
            is_mla=self.is_mla,
            num_q_heads=self.num_q_heads,
            data_parallel_degree=1,
            kvcache_quant_config=self.kvcache_quant_config,
        )

    def _get_symbolic_inputs_for_replica(
        self,
        devices: Sequence[DeviceRef],
        replica_idx: int,
        prefix: str = "",
        draft_attention_group: KVCacheParams | None = None,
    ) -> list[KVCacheInputsPerDevice[TensorType, BufferType]]:
        """Computes the symbolic inputs for a single replica.

        Returns:
            The symbolic inputs for the KV cache.
        """
        dynamic_dim_prefix = f"{prefix}replica_{replica_idx}_"

        kv_cache_scale_dtype = DType.float32
        if self.quantized_kv_cache and self.kvcache_quant_config is not None:
            kv_cache_scale_dtype = self.kvcache_quant_config.scale_dtype

        draft_params: KVCacheParams | None = (
            draft_attention_group
            if draft_attention_group is not None
            else (self if self.num_eagle_speculative_tokens > 0 else None)
        )

        return [
            KVCacheInputsPerDevice(
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
                attention_dispatch_metadata=TensorType(
                    DType.int64,
                    shape=[3] if self.is_mla else [4],
                    # MLA kernels consume 3-value dispatch metadata on GPU;
                    # MHA reads 4-value metadata on CPU.
                    device=device if self.is_mla else DeviceRef.CPU(),
                ),
                draft_attention_dispatch_metadata=TensorType(
                    DType.int64,
                    shape=[3] if draft_params.is_mla else [4],
                    device=device if draft_params.is_mla else DeviceRef.CPU(),
                )
                if draft_params is not None
                else None,
            )
            for device in devices
        ]

    def get_symbolic_inputs(
        self,
        prefix: str = "",
        *,
        draft_attention_group: KVCacheParams | None = None,
    ) -> KVCacheInputs[TensorType, BufferType]:
        """Computes the symbolic inputs for the KV cache.

        Args:
            prefix: Prefix for dynamic dim names.
            draft_attention_group: When set, sizes
                ``draft_attention_dispatch_metadata`` by the drafter's
                ``is_mla`` rather than ``self``'s. Use for unified spec-dec
                graphs with asymmetric attention types.

        Returns:
            The symbolic inputs for the KV cache.
        """
        devices_per_replica = split_into_groups(
            self.devices, self.data_parallel_degree
        )
        input_symbols: list[KVCacheInputsPerDevice[TensorType, BufferType]] = []
        for replica_idx, devices in enumerate(devices_per_replica):
            symbols = self._get_symbolic_inputs_for_replica(
                devices,
                replica_idx,
                prefix,
                draft_attention_group=draft_attention_group,
            )
            input_symbols.extend(symbols)
        return KVCacheInputs(inputs=input_symbols)

    def allocate_buffers(self, total_num_pages: int) -> list[KVCacheBuffer]:
        """Allocates the buffers for the KV cache."""
        devices_per_replica = split_into_groups(
            x=[d.to_device() for d in self.devices],
            groups=self.data_parallel_degree,
        )
        kv_cache_buffers: list[KVCacheBuffer] = []
        for devices in devices_per_replica:
            values = []
            for device in devices:
                value = Buffer.zeros(
                    shape=[total_num_pages, *self.shape_per_block],
                    dtype=self.dtype,
                    device=device,
                )
                values.append(value)

            scales: list[Buffer] | None = None
            if self.quantized_kv_cache:
                scales = []
                assert self.kvcache_quant_config is not None
                scale_dtype = self.kvcache_quant_config.scale_dtype
                for device in devices:
                    scale = Buffer.zeros(
                        shape=[total_num_pages, *self.shape_per_scale_block],
                        dtype=scale_dtype,
                        device=device,
                    )
                    scales.append(scale)

            kv_cache_buffer = KVCacheBuffer(
                values=values, scales=scales, total_num_pages=total_num_pages
            )
            kv_cache_buffers.append(kv_cache_buffer)
        return kv_cache_buffers


@dataclass(frozen=True)
class MultiKVCacheParams(KVCacheParamInterface):
    """Aggregates multiple KV cache parameter sets.

    This class implements KVCacheParamInterface by aggregating multiple
    KVCacheParamInterface instances. Useful for models with multiple distinct
    KV caches (e.g., different cache configurations for different layers).
    """

    params: Sequence[KVCacheParams]
    """List of KV cache parameter sets to aggregate."""

    page_size: int
    data_parallel_degree: int
    n_devices: int
    kv_connector: KVConnectorType | None
    host_kvcache_swap_space_gb: float | None
    num_eagle_speculative_tokens: int = 0

    @classmethod
    def from_params(cls, *params: KVCacheParams) -> MultiKVCacheParams:
        """Creates a :class:`MultiKVCacheParams` from one or more :class:`KVCacheParams`.

        Args:
            params: One or more :class:`KVCacheParams` instances to aggregate.
                All params must share the same ``page_size``,
                ``data_parallel_degree``, ``n_devices``,
                ``enable_kvcache_swapping_to_host``, and
                ``host_kvcache_swap_space_gb`` values.

        Returns:
            A new :class:`MultiKVCacheParams` aggregating all provided params.

        Raises:
            ValueError: If no params are provided.
        """
        if len(params) == 0:
            raise ValueError("MultiKVCacheParams requires at least one param.")
        return cls(
            params=params,
            page_size=params[0].page_size,
            data_parallel_degree=params[0].data_parallel_degree,
            n_devices=params[0].n_devices,
            kv_connector=params[0].kv_connector,
            host_kvcache_swap_space_gb=params[0].host_kvcache_swap_space_gb,
            num_eagle_speculative_tokens=params[0].num_eagle_speculative_tokens,
        )

    def __post_init__(self) -> None:
        """Validates that all params have consistent page size."""
        if not self.params:
            raise ValueError(
                "MultiKVCacheParams requires at least one param set."
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

        kv_connectors = {p.kv_connector for p in self.params}
        if len(kv_connectors) > 1:
            raise ValueError(
                f"All params must use the same kv_connector, got: {kv_connectors}"
            )

        host_kvcache_swap_space_gb = {
            p.host_kvcache_swap_space_gb for p in self.params
        }
        if len(host_kvcache_swap_space_gb) > 1:
            raise ValueError(
                f"All params must use the same host_kvcache_swap_space_gb, got: {host_kvcache_swap_space_gb}"
            )

        num_eagle_speculative_tokens = {
            p.num_eagle_speculative_tokens for p in self.params
        }
        if len(num_eagle_speculative_tokens) > 1:
            raise ValueError(
                f"All params must use the same num_eagle_speculative_tokens, got: {num_eagle_speculative_tokens}"
            )

    @property
    def bytes_per_block(self) -> int:
        """Total bytes per block across all KV caches.

        Since all caches allocate memory for the same sequence, the total
        memory cost per block is the sum across all param sets.
        """
        return sum(p.bytes_per_block for p in self.params)

    def get_symbolic_inputs(
        self, prefix: str = ""
    ) -> KVCacheInputs[TensorType, BufferType]:
        """Returns the symbolic inputs for the KV cache."""
        inputs: list[KVCacheInputsPerDevice[TensorType, BufferType]] = []
        for i, p in enumerate(self.params):
            inputs.extend(p.get_symbolic_inputs(f"{prefix}cache{i}_").inputs)
        return KVCacheInputs(inputs=inputs)


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


def compute_num_host_blocks(params: KVCacheParamInterface) -> int:
    """Computes the number of blocks that can be allocated on the host.

    Returns:
        The number of blocks that can be allocated on the host.
    """
    if params.kv_connector not in (
        KVConnectorType.local,
        KVConnectorType.tiered,
    ):
        return 0
    assert params.host_kvcache_swap_space_gb is not None
    GiB = 1024 * 1024 * 1024
    host_gb_per_replica = params.host_kvcache_swap_space_gb
    host_bytes_per_replica = host_gb_per_replica * GiB
    num_host_blocks = int(host_bytes_per_replica // params.bytes_per_block)

    if num_host_blocks == 0:
        raise RuntimeError(
            f"Insufficient cache memory to allocate even a single page.\n"
            f"One page requires {to_human_readable_bytes(params.bytes_per_block)} but only "
            f"{to_human_readable_bytes(host_gb_per_replica * GiB)} are available on host."
        )

    return num_host_blocks
