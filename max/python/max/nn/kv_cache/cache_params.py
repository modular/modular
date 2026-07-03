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
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import cached_property, reduce
from operator import mul
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from max._kv_cache_ops import (
    mha_decode_num_partitions,
    mla_dispatch_args_scalar,
)
from max.driver import Buffer, Device, DevicePinnedBuffer
from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    TensorType,
    TensorValue,
)
from max.support.human_readable_formatter import to_human_readable_bytes

from .data_parallelism_utils import split_into_groups
from .input_types import (
    KVCacheInputs,
    KVCacheInputsInterface,
    KVCacheInputsPerDevice,
    MultiKVCacheInputs,
)
from .utils import (
    AttnKeyInterface,
    MHAAttnKey,
    MLAAttnKey,
    MSAAttnKey,
    MultiAttnKey,
)

# Mirror of max.pipelines.speculative.config.SpeculativeMethod. Defined
# inline rather than imported because max.pipelines.speculative depends
# on max.nn (BUILD.bazel), so importing back would create a circular
# bazel dependency. The two definitions are structurally identical
# Literals, so mypy treats them as the same type at use sites.
SpeculativeMethod = Literal["eagle", "mtp", "dflash"]

KVHashAlgo = Literal["ahash64", "sha256", "sha256_64"]
"""Supported hash algorithms for KV-cache block identity."""

logger = logging.getLogger("max.pipelines")


def _filter_tiny_cache_lengths(
    probe_lengths: list[int], num_draft_tokens: int
) -> list[int]:
    min_cache_length = 1 + 2 * num_draft_tokens
    return [cl for cl in probe_lengths if cl >= min_cache_length]


class KVConnectorType(str, Enum):
    """Identifies which off-device backing store the KV cache uses.

    Set on :attr:`KVCacheParams.kv_connector` to control whether evicted
    cache pages stay on device only, spill to host memory, tier across host
    and disk, or route through a distributed block store.
    """

    null = "null"
    """No off-device backing store. Pages live on device only."""

    local = "local"
    """Spills evicted pages to host memory.

    Requires ``enable_prefix_caching`` and ``host_kvcache_swap_space_gb``
    to be set on :class:`KVCacheParams`.
    """

    tiered = "tiered"
    """Tiers evicted pages across host memory and disk.

    Requires ``enable_prefix_caching``, ``host_kvcache_swap_space_gb``,
    and a ``disk_offload_dir`` on the connector config.
    """

    dkv = "dkv"
    """Routes pages through a distributed KV block store.

    Requires a ``block_store_endpoint`` on the connector config.
    """


def _validate_is_2d_uint8_buffer(buffer: Buffer) -> None:
    if len(buffer.shape) != 2:
        raise ValueError("KVCacheMemory buffer must have 2 dimensions")
    if buffer.dtype != DType.uint8:
        raise ValueError("KVCacheMemory buffer must have dtype uint8")


@dataclass
class KVCacheMemory:
    """A single KV cache shard as a 2-D ``uint8`` view.

    ``buffer`` has shape ``[num_pages, bytes_per_page]`` with dtype
    ``uint8``.  This is the form consumed by the offload engine and KV
    connectors.  :class:`ReplicatedKVCacheMemory` subclasses this for
    caches that are replicated across TP shards (MLA).
    """

    buffer: Buffer

    def __post_init__(self) -> None:
        _validate_is_2d_uint8_buffer(self.buffer)

    @property
    def total_num_pages(self) -> int:
        """Returns the total number of pages."""
        return self.buffer.shape[0]

    @property
    def all_buffers(self) -> list[Buffer]:
        """Returns every shard buffer backing this unit (rank-0 + peers)."""
        return [self.buffer]

    def copy_block_to(
        self,
        dst: KVCacheMemory,
        dst_block_id: int,
        src_block_id: int,
    ) -> None:
        """Copies one page from this unit into ``dst`` (device-to-device).

        ``dst`` must be the same logical unit on another replica (identical
        page layout, same number of shards). The copy may cross devices; the
        driver issues a peer-to-peer (or host-bounced) transfer. Used to
        materialize a prefix-cache block on a different DP replica
        (SERVOPT-1500).

        Args:
            dst: Destination memory unit (same logical unit, another replica).
            dst_block_id: Destination page index.
            src_block_id: Source page index in this unit.
        """
        for dst_buffer, src_buffer in zip(
            dst.all_buffers, self.all_buffers, strict=True
        ):
            dst_buffer[dst_block_id, :].inplace_copy_from(
                src_buffer[src_block_id, :]
            )


@dataclass
class ReplicatedKVCacheMemory(KVCacheMemory):
    """A replicated KV cache unit (rank-0 shard plus its TP peers).

    All shards hold identical data (MLA); D2H reads from ``buffer``
    (rank-0) and H2D broadcasts back to ``buffer`` and every entry in
    ``peers``.  Each buffer has shape ``[num_pages, bytes_per_page]``
    with dtype ``uint8``.
    """

    peers: list[Buffer]

    def __post_init__(self) -> None:
        super().__post_init__()

        if len(self.peers) == 0:
            raise ValueError(
                "ReplicatedKVCacheMemory must have at least one peer"
            )

        for peer in self.peers:
            _validate_is_2d_uint8_buffer(peer)

        unique_shapes = set(
            buffer.shape for buffer in [self.buffer, *self.peers]
        )
        if len(unique_shapes) > 1:
            raise ValueError("All buffers must have the same shape")

    @property
    def all_buffers(self) -> list[Buffer]:
        """Returns the rank-0 shard buffer followed by every peer buffer.

        For replicated caches (MLA) all shards hold identical data; the
        inherited :meth:`KVCacheMemory.copy_block_to` zips these against the
        destination's shards (``strict=True``), so each shard is fanned out
        with an independent point-to-point copy (no broadcast collective).
        """
        return [self.buffer, *self.peers]


@runtime_checkable
class KVCacheBufferInterface(Protocol):
    """Interface for a KV cache buffer (single leaf or a tree of leaves)."""

    @property
    def total_num_pages(self) -> int:
        """Returns the total number of pages."""
        ...

    @property
    def all_buffers(self) -> list[Buffer]:
        """Returns all buffers."""
        ...

    def to_memory(self) -> list[KVCacheMemory]:
        """Returns the offload-ready KV cache memory units."""
        ...


@dataclass
class MultiKVCacheBuffer(KVCacheBufferInterface):
    """A tree of KVCache buffers for one data-parallel replica.

    ``children`` maps a cache name (e.g. ``"target"``/``"draft"`` for
    speculative decoding, or ``"sliding"``/``"global"`` for hybrid models) to
    that cache's buffer for this replica.
    """

    children: dict[str, KVCacheBufferInterface]

    @property
    def total_num_pages(self) -> int:
        """Returns the total number of pages."""
        first = next(iter(self.children.values()))
        return first.total_num_pages

    @property
    def all_buffers(self) -> list[Buffer]:
        """Returns all buffers across every child cache."""
        bufs: list[Buffer] = []
        for child in self.children.values():
            bufs.extend(child.all_buffers)
        return bufs

    def to_memory(self) -> list[KVCacheMemory]:
        """Returns the offload-ready KV cache memory units for all children."""
        memories: list[KVCacheMemory] = []
        for child in self.children.values():
            memories.extend(child.to_memory())
        return memories


@dataclass
class KVCacheBuffer(KVCacheBufferInterface):
    """A collection of KVCache buffers for one data-parallel replica.

    Two buffer kinds are supported: ``values`` and (optionally, for FP8
    quantization) ``scales``. The length of each list corresponds to the
    tensor-parallel degree, with one buffer per TP shard.

    ``page_size`` and ``replicates_kv_across_tp`` describe the physical layout
    so KV connectors can offload this cache without a separate
    ``KVCacheParams`` reference: ``replicates_kv_across_tp`` is ``True`` when
    the KV data is replicated identically across TP shards (MLA) and ``False``
    when it is sharded (MHA).
    """

    replicates_kv_across_tp: bool
    values: list[Buffer]
    scales: list[Buffer] | None = None

    def __post_init__(self) -> None:
        all_buffers = self.all_buffers

        if len(self.values) == 0:
            raise ValueError("List of values must be non-empty")

        if self.replicates_kv_across_tp and len(self.values) <= 1:
            raise ValueError(
                "replicates_kv_across_tp=True requires at least 2 TP shards "
                "(len(values) > 1)"
            )

        unique_dtype = set(b.dtype for b in self.values)
        if len(unique_dtype) > 1:
            raise ValueError("All values must have the same dtype")

        unique_shapes = set(b.shape for b in self.values)
        if len(unique_shapes) > 1:
            raise ValueError("All values must have the same shape")

        unique_is_pinned = set(
            isinstance(b, DevicePinnedBuffer) for b in all_buffers
        )
        if len(unique_is_pinned) > 1:
            raise ValueError(
                "All values (and scales if present) must be either all pinned "
                "or all non-pinned"
            )

        if self.scales is None:
            return

        if len(self.scales) != len(self.values):
            raise ValueError("Scales must be the same length as values")

        unique_dtype = set(b.dtype for b in self.scales)
        if len(unique_dtype) > 1:
            raise ValueError("All scales must have the same dtype")

        unique_shapes = set(b.shape for b in self.scales)
        if len(unique_shapes) > 1:
            raise ValueError("All scales must have the same shape")

        unique_num_pages = set(b.shape[0] for b in all_buffers)
        if len(unique_num_pages) > 1:
            raise ValueError(
                "Values and scales must have the same number of pages"
            )
        for value, scale in zip(self.values, self.scales, strict=True):
            if value.device != scale.device:
                raise ValueError(
                    "Corresponding values and scales must be on the same device"
                )

    @property
    def total_num_pages(self) -> int:
        """Returns the total number of pages across all values and scales."""
        return self.values[0].shape[0]

    @property
    def all_buffers(self) -> list[Buffer]:
        """Returns all value and scale buffers in a single flat list.

        Returns:
            A list containing every value buffer followed by every scale
            buffer (if scales are present).
        """
        return [
            *self.values,
            *(self.scales if self.scales is not None else []),
        ]

    def to_memory(self) -> list[KVCacheMemory]:
        """Convert to a flat list of offload-ready memory units.

        Each unit covers one buffer kind (values or scales) and one
        logical TP group.  Non-replicated shards become individual
        :class:`KVCacheMemory` entries; replicated shards become one
        :class:`ReplicatedKVCacheMemory` entry (root + peers).

        Every buffer is re-viewed as a 2-D ``[num_pages, bytes_per_page]``
        ``uint8`` array so the offload engine can treat all caches
        uniformly regardless of original dtype or shape.

        Returns:
            A list of memory units ready for use by KV connectors and the
            offload engine.
        """
        result: list[KVCacheMemory] = []
        shard_lists: list[list[Buffer]] = [self.values]
        if self.scales is not None:
            shard_lists.append(self.scales)
        for shards in shard_lists:
            viewed = [
                b.view(
                    dtype=DType.uint8,
                    shape=[
                        b.shape[0],
                        b.num_elements * b.dtype.size_in_bytes // b.shape[0],
                    ],
                )
                for b in shards
            ]
            if self.replicates_kv_across_tp:
                result.append(
                    ReplicatedKVCacheMemory(buffer=viewed[0], peers=viewed[1:])
                )
            else:
                result.extend(KVCacheMemory(buffer=v) for v in viewed)
        return result


@dataclass
class KVCacheQuantizationConfig:
    """Configuration for KVCache quantization.

    Currently only FP8 Quantization is supported.
    """

    scale_dtype: DType = DType.float32
    """Data type of quantization scales, if quantization is enabled"""

    quantization_granularity: int = 128
    """Block-size used for KVCache quantization along head-dimension (e.g. 128)."""


@dataclass(frozen=True)
class BatchCharacteristics:
    """Upper-bound batch shape used to prepare decode attention metadata.

    Captures the ``(batch_size, max_prompt_length, max_cache_valid_length)`` a
    decode forward should prepare its attention dispatch metadata *for*, which
    may exceed the batch's real per-request values.
    :meth:`PagedKVCacheManager.runtime_inputs` uses it to resolve the dispatch
    key once: e.g. for graph-capture replay, ``max_cache_valid_length`` is
    aligned up to a cache length recorded during capture and every data-parallel
    replica must run the identical captured graph. The batch's real values must
    not exceed these.
    """

    batch_size: int
    max_prompt_length: int
    max_cache_valid_length: int


@dataclass
class KVCacheAssignments:
    """Assignments of request blocks to KV cache pages for a replica.

    ``batch_characteristics`` carries the effective ``(batch_size,
    max_prompt_length, max_cache_valid_length)`` used to build this
    assignment (after any graph-capture upper-bound override) so that
    :meth:`KVCacheParamInterface.build_runtime_inputs` can resolve the decode
    attention dispatch keys from the same values.
    """

    cache_lengths_by_device: list[Buffer]
    lookup_table_by_device: list[Buffer]
    max_prompt_length: Buffer
    max_cache_length: Buffer
    batch_characteristics: BatchCharacteristics


@runtime_checkable
class KVCacheParamInterface(Protocol):
    """Interface for KV cache parameters."""

    page_size: int
    data_parallel_degree: int
    devices: Sequence[DeviceRef]
    kv_connector: KVConnectorType | None
    kv_connector_config: Any
    host_kvcache_swap_space_gb: float | None
    speculative_method: SpeculativeMethod | None = None
    num_draft_tokens: int = 0

    @property
    def n_devices(self) -> int:
        """Returns the total number of devices."""
        ...

    @property
    def enable_prefix_caching(self) -> bool:
        """Whether prefix caching is enabled."""
        ...

    @property
    def num_draft_tokens_per_step(self) -> int:
        """Number of draft tokens written per draft forward.

        One for autoregressive drafts (``eagle``, ``mtp``);
        equal to ``num_draft_tokens`` for block drafts (``dflash``).
        """
        if self.speculative_method == "dflash":
            return self.num_draft_tokens
        return 1

    @property
    def bytes_per_block(self) -> int:
        """Number of bytes per cache block."""
        ...

    def get_symbolic_inputs(
        self,
    ) -> KVCacheInputsInterface[TensorType, BufferType]:
        """Returns the symbolic inputs for the KV cache."""
        ...

    def flattened_kv_inputs(self) -> list[TensorType | BufferType]:
        """Flattens the symbolic inputs for the KV cache."""
        return self.get_symbolic_inputs().flatten()

    def unflatten_kv_inputs(
        self, it: Iterator[Any]
    ) -> KVCacheInputsInterface[TensorValue, BufferValue]:
        """Unflattens the symbolic inputs for the KV cache."""
        ...

    @property
    def replicates_kv_across_tp(self) -> bool:
        """Whether every device holds identical KV state."""
        ...

    @property
    def tensor_parallel_degree(self) -> int:
        """Returns the tensor parallel degree."""
        ...

    def resolve_attn_key(
        self,
        batch_size: int,
        max_prompt_length: int,
        max_cache_valid_length: int,
    ) -> AttnKeyInterface:
        """Resolves the decode dispatch shape for the given shape.

        Returns a :class:`AttnKeyInterface` for a single cache, or a
        :class:`MultiAttnKey` tree mirroring the cache tree.
        """
        ...

    def graph_capture_probe_cache_lengths(
        self, max_cache_length: int, q_max_seq_len: int = 1
    ) -> list[int]:
        """Returns the cache lengths to probe during decode graph capture."""
        ...

    @property
    def kv_hash_algo(self) -> KVHashAlgo:
        """Hash algorithm used for KV-cache block identity."""
        ...

    @property
    def kv_hash_seed(self) -> bytes | None:
        """Resolved 32-byte cluster seed for sha256/sha256_64. None for ahash64."""
        ...

    def allocate_buffers(
        self, total_num_pages: int
    ) -> Sequence[KVCacheBufferInterface]:
        """Allocates the buffers for the KV cache."""
        ...

    def build_runtime_inputs(
        self,
        assignments: Sequence[KVCacheAssignments],
        buffers: Sequence[KVCacheBufferInterface],
    ) -> KVCacheInputsInterface[Buffer, Buffer]:
        """Builds the runtime KV-cache inputs spanning all replicas.

        ``assignments`` and ``buffers`` are indexed by data-parallel replica.
        Returns a single :class:`KVCacheInputs` leaf (or a
        :class:`MultiKVCacheInputs` tree) whose leaves each hold every
        ``(replica, TP shard)`` device's inputs."""
        ...

    def unflatten_basic_kv_tree(
        self, it: Iterator[Any]
    ) -> tuple[list[KVCacheInputsPerDevice[TensorValue, BufferValue]], ...]:
        """Unflattens a basic KV tree from a graph-input iterator.

        Requires that the model is a basic height-1 tree. This method does not work
        on nested trees.
        """
        ...


@dataclass
class KVCacheParams(KVCacheParamInterface):
    """Configuration parameters for key-value cache management in transformer models.

    This class encapsulates all configuration options for managing KV caches during
    inference, including parallelism settings, and memory management.
    """

    dtype: DType
    """Data type for storing key and value tensors in the cache."""

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

    kv_hash_algo: KVHashAlgo = "ahash64"
    """Hash algorithm used for KV-cache block identity."""

    kv_hash_seed: bytes | None = None
    """Resolved 32-byte cluster seed for sha256/sha256_64. None for ahash64.

    Set by ``KVCacheConfig.to_params`` via ``resolve_kv_hash_seed``.
    """

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

    data_parallel_degree: int = 1
    """Degree of data parallelism. Devices are grouped replica-major, with
    ``n_devices // data_parallel_degree`` TP shards per replica."""

    kvcache_quant_config: KVCacheQuantizationConfig | None = None
    """KVCache quantization config. Currently only FP8 quantization supported."""

    speculative_method: SpeculativeMethod | None = None
    """Speculative decoding method propagated from
    SpeculativeConfig"""

    num_draft_tokens: int = 0
    """Total draft tokens generated per speculative iteration.

    Zero when no speculative decoding is configured."""

    def __post_init__(self):
        """Validates configuration and computes derived fields after initialization.

        Raises:
            ValueError: If configuration parameters are invalid or incompatible.
        """
        if self.data_parallel_degree < 1:
            raise ValueError(
                f"Data parallelism degree ({self.data_parallel_degree})"
                " must be at least 1"
            )

        if self.n_devices < self.data_parallel_degree:
            raise ValueError(
                f"Data parallelism degree ({self.data_parallel_degree})"
                " cannot be greater than the number of devices"
                f" ({self.n_devices})"
            )

        if self.n_devices % self.data_parallel_degree != 0:
            raise ValueError(
                f"Number of devices ({self.n_devices}) must be divisible by"
                " data parallelism degree"
                f" ({self.data_parallel_degree})"
            )

        # Validate connector configuration
        if self.kv_connector in (
            KVConnectorType.local,
            KVConnectorType.tiered,
        ):
            if not self.enable_prefix_caching:
                raise ValueError(
                    f"KV connector '{self.kv_connector.value}' requires prefix"
                    " caching to be enabled"
                )
            if self.host_kvcache_swap_space_gb is None:
                raise ValueError(
                    "host_kvcache_swap_space_gb is required when kv_connector"
                    f" is '{self.kv_connector.value}'"
                )

        if self.quantized_kv_cache and self.kvcache_quant_config is not None:
            # Validate FP8 KVCache quantization granularity.
            if (
                self.head_dim
                % self.kvcache_quant_config.quantization_granularity
                != 0
            ):
                raise ValueError(
                    "KVCache quantization granularity must evenly divide KV"
                    " head dimension."
                )
            if self.kvcache_quant_config is None:
                raise ValueError("KVCache quantization config required.")

    @cached_property
    def devices_per_replica(self) -> Sequence[Sequence[DeviceRef]]:
        """Returns the devices per replica."""
        return split_into_groups(self.devices, self.data_parallel_degree)

    @cached_property
    def _primary_device(self) -> Device | None:
        """Concrete primary device for decode-dispatch kernels.

        The decode dispatch custom ops are GPU kernels needing a concrete
        :class:`~max.driver.Device`. Built lazily (and cached) so constructing
        params for a GPU ``DeviceRef`` on a CPU-only host does not require a
        device context. Returns ``None`` on a CPU-only host; callers then fall
        back to the sentinel dispatch key (``num_partitions=1``).
        """
        device_ref = self.devices[0]
        if device_ref.is_cpu():
            return None
        return device_ref.to_device()

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
    def kv_cache_scale_dtype(self) -> DType:
        """Returns the dtype of the KV cache scales.

        Returns:
            The dtype of the KV cache scales.
        """
        if self.quantized_kv_cache and self.kvcache_quant_config is not None:
            return self.kvcache_quant_config.scale_dtype
        else:
            return DType.float32

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
        raise NotImplementedError

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
    def kv_dim(self) -> int:
        raise NotImplementedError

    @property
    def n_kv_heads_per_device(self) -> int:
        raise NotImplementedError

    @property
    def shape_per_block(self) -> list[int]:
        """Returns the shape of each cache block.

        Returns:
            The shape of the cache block.
        """
        # split k and v caches across a single dim
        # 0 = key
        # 1 = value
        return [
            self.kv_dim,
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

    def _get_symbolic_inputs_for_replica(
        self, replica_idx: int, prefix: str
    ) -> list[KVCacheInputsPerDevice[TensorType, BufferType]]:
        raise NotImplementedError

    def get_symbolic_inputs(self) -> KVCacheInputs[TensorType, BufferType]:
        """Computes the symbolic inputs for the KV cache.

        Returns:
            The symbolic inputs for the KV cache.
        """
        input_symbols: list[KVCacheInputsPerDevice[TensorType, BufferType]] = []
        for replica_idx in range(len(self.devices_per_replica)):
            prefix = f"replica_{replica_idx}_"
            symbols = self._get_symbolic_inputs_for_replica(replica_idx, prefix)
            input_symbols.extend(symbols)
        return KVCacheInputs(inputs=input_symbols)

    def unflatten_kv_inputs(
        self, it: Iterator[Any]
    ) -> KVCacheInputs[TensorValue, BufferValue]:
        """Unflattens the KV cache inputs from a graph-input iterator."""
        return self.get_symbolic_inputs().unflatten(it)

    def allocate_buffers(self, total_num_pages: int) -> list[KVCacheBuffer]:
        """Allocates the buffers for the KV cache."""
        # ``Buffer.zeros`` needs concrete devices, so materialize the per-replica
        # device groups from the ``DeviceRef``s here.
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
                values=values,
                scales=scales,
                replicates_kv_across_tp=self.replicates_kv_across_tp,
            )
            kv_cache_buffers.append(kv_cache_buffer)
        return kv_cache_buffers

    def _build_kvcache_inputs_per_device(
        self,
        device: Device,
        blocks: Buffer,
        cache_lengths: Buffer,
        lookup_table: Buffer,
        max_prompt_length: Buffer,
        max_cache_length: Buffer,
        kv_scales: Buffer | None,
        target_key: AttnKeyInterface,
        draft_key: AttnKeyInterface | None,
        max_cache_valid_length: int,
    ) -> KVCacheInputsPerDevice[Buffer, Buffer]:
        raise NotImplementedError

    def build_runtime_inputs(
        self,
        assignments: Sequence[KVCacheAssignments],
        buffers: Sequence[KVCacheBufferInterface],
    ) -> KVCacheInputsInterface[Buffer, Buffer]:
        """Builds the runtime KV-cache leaf spanning all replicas.

        ``assignments`` and ``buffers`` are indexed by data-parallel replica.
        The returned :class:`KVCacheInputs` lists one
        :class:`KVCacheInputsPerDevice` per ``(replica, TP shard)``, in the
        same replica-major order as :meth:`get_symbolic_inputs`.
        """
        tp_shards: list[KVCacheInputsPerDevice[Buffer, Buffer]] = []
        for assignment, buffer in zip(assignments, buffers, strict=True):
            assert isinstance(buffer, KVCacheBuffer)
            bc = assignment.batch_characteristics
            batch_size = bc.batch_size
            max_cl = bc.max_cache_valid_length

            target_key = self.resolve_attn_key(
                batch_size, bc.max_prompt_length, max_cl
            )
            draft_key = (
                self.resolve_attn_key(
                    batch_size, self.num_draft_tokens_per_step, max_cl
                )
                if self.speculative_method is not None
                else None
            )

            for i, (cl, lut, blocks) in enumerate(
                zip(
                    assignment.cache_lengths_by_device,
                    assignment.lookup_table_by_device,
                    buffer.values,
                    strict=True,
                )
            ):
                device = blocks.device
                kv_scales = (
                    buffer.scales[i] if buffer.scales is not None else None
                )
                tp_shards.append(
                    self._build_kvcache_inputs_per_device(
                        device,
                        blocks,
                        cl,
                        lut,
                        assignment.max_prompt_length,
                        assignment.max_cache_length,
                        kv_scales,
                        target_key,
                        draft_key,
                        max_cl,
                    )
                )
        return KVCacheInputs(inputs=tp_shards)

    def unflatten_basic_kv_tree(
        self, it: Iterator[Any]
    ) -> tuple[list[KVCacheInputsPerDevice[TensorValue, BufferValue]], ...]:
        """Unflattens a basic KV tree from a graph-input iterator.

        Requires that the model is a basic height-1 tree. This method does not work
        on nested trees.
        """
        raise ValueError(
            "Unflattening a basic KV tree is only supported for MultiKVCacheParams"
        )


@dataclass(kw_only=True)
class MHAKVCacheParams(KVCacheParams):
    n_kv_heads: int
    """Total number of key-value attention heads across all devices."""

    allow_kv_head_replication: bool = False
    """Allow TP wider than ``n_kv_heads``: when set and ``n_devices`` is a
    multiple of ``n_kv_heads``, replicate each KV head across a group of
    devices (``n_kv_heads_per_device == 1``)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        tp_degree = self.tensor_parallel_degree
        if self.n_kv_heads % tp_degree == 0:
            return
        # Fewer heads than devices: replicate each head across a device group.
        if self.allow_kv_head_replication and tp_degree % self.n_kv_heads == 0:
            return
        raise ValueError(
            f"Number of KV heads ({self.n_kv_heads}) must be divisible by"
            f" the tensor parallel degree ({tp_degree})"
        )

    @property
    def kv_dim(self) -> int:
        return 2

    @property
    def n_kv_heads_per_device(self) -> int:
        tp_degree = self.tensor_parallel_degree
        if self.n_kv_heads % tp_degree == 0:
            return max(self.n_kv_heads // tp_degree, 1)
        # ``allow_kv_head_replication``: each head spans a group of devices.
        return 1

    @property
    def replicates_kv_across_tp(self) -> bool:
        """Whether every device holds identical KV state."""
        return False

    def resolve_attn_key(
        self,
        batch_size: int,
        max_prompt_length: int,
        max_cache_valid_length: int,
    ) -> AttnKeyInterface:
        """Resolves the decode attention dispatch shape for the given shape.

        Args:
            batch_size: Number of requests in the decode batch.
            max_prompt_length: Per-step query width (``1`` for plain decode,
                ``1 + num_spec_tokens`` for speculative verify).
            max_cache_valid_length: Maximum valid cache length in the batch.

        Returns:
            The resolved :class:`~max.nn.kv_cache.AttnKeyInterface`
        """
        device = self._primary_device
        if batch_size <= 0 or device is None:
            # Sentinel for empty / degenerate replicas or a CPU-only host;
            # skip the GPU dispatch kernel.
            num_partitions = 1
        else:
            num_partitions = mha_decode_num_partitions(
                batch_size,
                max_cache_valid_length,
                self.n_kv_heads_per_device,
                device,
            )
        return MHAAttnKey(
            batch_size=batch_size,
            max_prompt_length=max_prompt_length,
            num_partitions=num_partitions,
        )

    def graph_capture_probe_cache_lengths(
        self, max_cache_length: int, q_max_seq_len: int = 1
    ) -> list[int]:
        """Returns cache lengths to probe for distinct num_partitions."""
        granularity = 256
        probe_lengths = (
            [1]
            + list(range(granularity, max_cache_length, granularity))
            + [max_cache_length]
        )
        return _filter_tiny_cache_lengths(probe_lengths, self.num_draft_tokens)

    def _attn_metadata_buffer(self, device: DeviceRef) -> TensorType:
        # MHA decode kernels read a 4-int dispatch buffer on the host (CPU),
        # matching ``MHAAttnKey.pack_into_buffer``. ``device`` is accepted so
        # subclasses can emit device-resident metadata of a different shape.
        return TensorType(DType.int64, shape=[4], device=DeviceRef.CPU())

    def _get_symbolic_inputs_for_replica(
        self, replica_idx: int, prefix: str
    ) -> list[KVCacheInputsPerDevice[TensorType, BufferType]]:
        devices = self.devices_per_replica[replica_idx]

        return [
            KVCacheInputsPerDevice(
                kv_blocks=BufferType(
                    self.dtype,
                    shape=["total_num_pages", *self.shape_per_block],
                    device=device,
                ),
                cache_lengths=TensorType(
                    DType.uint32,
                    shape=[prefix + "batch_size"],
                    device=device,
                ),
                lookup_table=TensorType(
                    DType.uint32,
                    shape=[prefix + "batch_size", prefix + "max_num_pages"],
                    device=device,
                ),
                max_prompt_length=TensorType(
                    DType.uint32,
                    shape=[1],
                    device=DeviceRef.CPU(),
                ),
                max_cache_length=TensorType(
                    DType.uint32,
                    shape=[1],
                    device=DeviceRef.CPU(),
                ),
                kv_scales=BufferType(
                    self.kv_cache_scale_dtype,
                    shape=["total_num_pages", *self.shape_per_scale_block],
                    device=device,
                )
                if self.quantized_kv_cache
                else None,
                attention_dispatch_metadata=self._attn_metadata_buffer(device),
                draft_attention_dispatch_metadata=self._attn_metadata_buffer(
                    device
                )
                if self.speculative_method is not None
                else None,
            )
            for device in devices
        ]

    def _build_kvcache_inputs_per_device(
        self,
        device: Device,
        blocks: Buffer,
        cache_lengths: Buffer,
        lookup_table: Buffer,
        max_prompt_length: Buffer,
        max_cache_length: Buffer,
        kv_scales: Buffer | None,
        target_key: AttnKeyInterface,
        draft_key: AttnKeyInterface | None,
        max_cache_valid_length: int,
    ) -> KVCacheInputsPerDevice[Buffer, Buffer]:
        return KVCacheInputsPerDevice(
            kv_blocks=blocks,
            cache_lengths=cache_lengths,
            lookup_table=lookup_table,
            max_prompt_length=max_prompt_length,
            max_cache_length=max_cache_length,
            kv_scales=kv_scales,
            attention_dispatch_metadata=target_key.pack_into_buffer(
                device, max_cache_valid_length
            ),
            draft_attention_dispatch_metadata=draft_key.pack_into_buffer(
                device, max_cache_valid_length
            )
            if draft_key is not None
            else None,
        )


@dataclass(kw_only=True)
class MLAKVCacheParams(KVCacheParams):
    num_q_heads: int
    """Number of query attention heads, required so the MLA decode kernel can
    resolve its dispatch metadata."""

    def __post_init__(self) -> None:
        super().__post_init__()
        tp_degree = self.tensor_parallel_degree
        if self.num_q_heads % tp_degree != 0:
            raise ValueError(
                f"Number of query heads ({self.num_q_heads}) must be"
                " divisible by the tensor parallel degree"
                f" ({tp_degree})"
            )

    @property
    def kv_dim(self) -> int:
        return 1

    @property
    def replicates_kv_across_tp(self) -> bool:
        """Whether every device holds identical KV state."""
        return self.tensor_parallel_degree > 1

    @property
    def n_kv_heads_per_device(self) -> int:
        return 1

    @property
    def num_q_heads_per_device(self) -> int:
        return max(self.num_q_heads // self.tensor_parallel_degree, 1)

    def resolve_attn_key(
        self,
        batch_size: int,
        max_prompt_length: int,
        max_cache_valid_length: int,
    ) -> AttnKeyInterface:
        """Resolves the decode attention dispatch shape for the given shape.

        Args:
            batch_size: Number of requests in the decode batch.
            max_prompt_length: Per-step query width (``1`` for plain decode,
                ``1 + num_spec_tokens`` for speculative verify).
            max_cache_valid_length: Maximum valid cache length in the batch.

        Returns:
            The resolved :class:`~max.nn.kv_cache.AttnKeyInterface`
        """
        device = self._primary_device
        if batch_size <= 0 or device is None:
            # Sentinel for empty / degenerate replicas or a CPU-only host;
            # skip the GPU dispatch kernel.
            return MLAAttnKey(
                batch_size=batch_size,
                max_prompt_length=max_prompt_length,
                num_partitions=1,
            )
        # ``mla_dispatch_args_scalar`` may adjust batch_size / max_prompt_length
        # alongside the resolved num_partitions; carry the adjusted values.
        adj_batch_size, adj_max_prompt_length, num_partitions = (
            mla_dispatch_args_scalar(
                batch_size,
                max_cache_valid_length,
                max_prompt_length,
                self.num_q_heads_per_device,
                self.is_fp8_kv_dtype,
                device,
            )
        )
        return MLAAttnKey(
            batch_size=int(adj_batch_size),
            max_prompt_length=int(adj_max_prompt_length),
            num_partitions=int(num_partitions),
        )

    def graph_capture_probe_cache_lengths(
        self, max_cache_length: int, q_max_seq_len: int = 1
    ) -> list[int]:
        """Returns cache lengths to probe for distinct num_partitions."""
        granularity = 64
        probe_lengths = (
            [1]
            + list(range(granularity, max_cache_length, granularity))
            + [max_cache_length]
        )
        if q_max_seq_len > 1:
            # With spec decoding, probe a few more entries to hit all viable
            # (num_partition, draft_num_partition) pairs. Determined
            # experimentally; brute-forcing all pairs would inflate capture
            # time significantly.
            probe_lengths.extend(
                range(granularity - 1, max_cache_length, granularity)
            )
        return _filter_tiny_cache_lengths(probe_lengths, self.num_draft_tokens)

    def _get_symbolic_inputs_for_replica(
        self, replica_idx: int, prefix: str
    ) -> list[KVCacheInputsPerDevice[TensorType, BufferType]]:
        devices = self.devices_per_replica[replica_idx]

        return [
            KVCacheInputsPerDevice(
                kv_blocks=BufferType(
                    self.dtype,
                    shape=["total_num_pages", *self.shape_per_block],
                    device=device,
                ),
                cache_lengths=TensorType(
                    DType.uint32,
                    shape=[prefix + "batch_size"],
                    device=device,
                ),
                lookup_table=TensorType(
                    DType.uint32,
                    shape=[prefix + "batch_size", prefix + "max_num_pages"],
                    device=device,
                ),
                max_prompt_length=TensorType(
                    DType.uint32,
                    shape=[1],
                    device=DeviceRef.CPU(),
                ),
                max_cache_length=TensorType(
                    DType.uint32,
                    shape=[1],
                    device=DeviceRef.CPU(),
                ),
                kv_scales=BufferType(
                    self.kv_cache_scale_dtype,
                    shape=["total_num_pages", *self.shape_per_scale_block],
                    device=device,
                )
                if self.quantized_kv_cache
                else None,
                # MLA decode kernels read a 3-int dispatch buffer on the
                # accelerator, matching ``MLAAttnKey.pack_into_buffer``.
                attention_dispatch_metadata=TensorType(
                    DType.int64, shape=[3], device=device
                ),
                draft_attention_dispatch_metadata=TensorType(
                    DType.int64, shape=[3], device=device
                )
                if self.speculative_method is not None
                else None,
                mla_num_partitions=TensorType(
                    DType.int64, shape=[1], device=DeviceRef.CPU()
                ),
                draft_mla_num_partitions=TensorType(
                    DType.int64, shape=[1], device=DeviceRef.CPU()
                )
                if self.speculative_method is not None
                else None,
            )
            for device in devices
        ]

    def _build_kvcache_inputs_per_device(
        self,
        device: Device,
        blocks: Buffer,
        cache_lengths: Buffer,
        lookup_table: Buffer,
        max_prompt_length: Buffer,
        max_cache_length: Buffer,
        kv_scales: Buffer | None,
        target_key: AttnKeyInterface,
        draft_key: AttnKeyInterface | None,
        max_cache_valid_length: int,
    ) -> KVCacheInputsPerDevice[Buffer, Buffer]:
        assert isinstance(target_key, MLAAttnKey)
        assert draft_key is None or isinstance(draft_key, MLAAttnKey)
        return KVCacheInputsPerDevice(
            kv_blocks=blocks,
            cache_lengths=cache_lengths,
            lookup_table=lookup_table,
            max_prompt_length=max_prompt_length,
            max_cache_length=max_cache_length,
            kv_scales=kv_scales,
            attention_dispatch_metadata=target_key.pack_into_buffer(
                device, max_cache_valid_length
            ),
            draft_attention_dispatch_metadata=draft_key.pack_into_buffer(
                device, max_cache_valid_length
            )
            if draft_key is not None
            else None,
            mla_num_partitions=Buffer.from_numpy(
                np.array([target_key.num_partitions], dtype=np.int64)
            ),
            draft_mla_num_partitions=Buffer.from_numpy(
                np.array([draft_key.num_partitions], dtype=np.int64)
            )
            if draft_key is not None
            else None,
        )


@dataclass(kw_only=True)
class MSAKVCacheParams(MHAKVCacheParams):
    # TODO(SERVOPT-1502): MSA does not actually consume attention dispatch
    # metadata in its kernel. Once the indexer graph is migrated to a dedicated
    # MSA input record, drop ``attention_dispatch_metadata`` from the symbolic
    # and runtime inputs entirely instead of carrying the 1-int placeholder.
    def resolve_attn_key(
        self,
        batch_size: int,
        max_prompt_length: int,
        max_cache_valid_length: int,
    ) -> AttnKeyInterface:
        """Resolves the decode attention dispatch shape for the given shape."""
        return MSAAttnKey()

    def graph_capture_probe_cache_lengths(
        self, max_cache_length: int, q_max_seq_len: int = 1
    ) -> list[int]:
        """Returns cache lengths to probe for distinct num_partitions."""
        return [1, max_cache_length]

    def _attn_metadata_buffer(self, device: DeviceRef) -> TensorType:
        # ``MSAAttnKey.pack_into_buffer`` emits a single sentinel int.
        return TensorType(DType.int64, shape=[1], device=DeviceRef.CPU())


@dataclass(frozen=True)
class MultiKVCacheParams(KVCacheParamInterface):
    """Aggregates multiple KV cache parameter sets into a recursive tree.

    Children may be leaf :class:`KVCacheParams` instances or nested
    :class:`MultiKVCacheParams` subtrees, so arbitrarily deep hierarchies
    are supported (e.g. ``{target: {sliding, mla}, draft: mha}``). The
    whole tree is consumed through the :class:`KVCacheParamInterface` —
    callers never need to know the depth.
    """

    children: dict[str, KVCacheParamInterface]
    """KV cache parameter sets to aggregate. Values may be leaf
    :class:`KVCacheParams` or nested :class:`MultiKVCacheParams` trees."""

    page_size: int
    data_parallel_degree: int
    devices: Sequence[DeviceRef]
    kv_connector: KVConnectorType | None
    host_kvcache_swap_space_gb: float | None
    speculative_method: SpeculativeMethod | None = None
    num_draft_tokens: int = 0

    @classmethod
    def from_params(
        cls, params: Mapping[str, KVCacheParamInterface]
    ) -> MultiKVCacheParams:
        """Creates a :class:`MultiKVCacheParams` from one or more param sets.

        Children may be leaf :class:`KVCacheParams` instances or nested
        :class:`MultiKVCacheParams` trees, enabling arbitrarily deep KV
        cache hierarchies (e.g. ``{target: {sliding, mla}, draft: mha}``).
        All children must share the same ``page_size``,
        ``data_parallel_degree``, ``n_devices``, ``kv_connector``, and
        ``host_kvcache_swap_space_gb`` values.

        Args:
            params: Named mapping of :class:`KVCacheParamInterface` instances
                to aggregate.

        Returns:
            A new :class:`MultiKVCacheParams` aggregating all provided params.

        Raises:
            ValueError: If no params are provided.
        """
        if len(params) == 0:
            raise ValueError("MultiKVCacheParams requires at least one param.")
        first = next(iter(params.values()))
        return cls(
            children=dict(params),
            page_size=first.page_size,
            data_parallel_degree=first.data_parallel_degree,
            devices=first.devices,
            kv_connector=first.kv_connector,
            host_kvcache_swap_space_gb=first.host_kvcache_swap_space_gb,
            speculative_method=first.speculative_method,
            num_draft_tokens=first.num_draft_tokens,
        )

    def __post_init__(self) -> None:
        """Validates that all params have consistent page size."""
        if not self.children:
            raise ValueError(
                "MultiKVCacheParams requires at least one param set."
            )

        params = list(self.children.values())
        page_sizes = {p.page_size for p in params}
        if len(page_sizes) > 1:
            raise ValueError(
                f"All params must use the same page size, got: {page_sizes}"
            )

        data_parallel_degrees = {p.data_parallel_degree for p in params}
        if len(data_parallel_degrees) > 1:
            raise ValueError(
                "All params must use the same data parallel degree, got:"
                f" {data_parallel_degrees}"
            )

        devices = {tuple(p.devices) for p in params}
        if len(devices) > 1:
            raise ValueError(
                "All params must use the same number of devices, got:"
                f" {devices}"
            )

        enable_prefix_caching = {p.enable_prefix_caching for p in params}
        if len(enable_prefix_caching) > 1:
            raise ValueError(
                "All params must use the same enable_prefix_caching, got:"
                f" {enable_prefix_caching}"
            )

        kv_connectors = {p.kv_connector for p in params}
        if len(kv_connectors) > 1:
            raise ValueError(
                "All params must use the same kv_connector, got:"
                f" {kv_connectors}"
            )

        # ``KVConnectorConfig`` is not hashable, so compare by equality against
        # the first rather than collapsing into a set.
        first_kv_connector_config = params[0].kv_connector_config
        if any(
            p.kv_connector_config != first_kv_connector_config for p in params
        ):
            raise ValueError(
                "All params must use the same kv_connector_config, got:"
                f" {[p.kv_connector_config for p in params]}"
            )

        host_kvcache_swap_space_gb = {
            p.host_kvcache_swap_space_gb for p in params
        }
        if len(host_kvcache_swap_space_gb) > 1:
            raise ValueError(
                "All params must use the same host_kvcache_swap_space_gb, got:"
                f" {host_kvcache_swap_space_gb}"
            )

        speculative_methods = {p.speculative_method for p in params}
        if len(speculative_methods) > 1:
            raise ValueError(
                "All params must use the same speculative_method, got:"
                f" {speculative_methods}"
            )

        num_draft_tokens_set = {p.num_draft_tokens for p in params}
        if len(num_draft_tokens_set) > 1:
            raise ValueError(
                "All params must use the same num_draft_tokens, got:"
                f" {num_draft_tokens_set}"
            )

        kv_hash_algos = {p.kv_hash_algo for p in params}
        if len(kv_hash_algos) > 1:
            raise ValueError(
                "All params must use the same kv_hash_algo, got:"
                f" {kv_hash_algos}"
            )

        kv_hash_seeds = {p.kv_hash_seed for p in params}
        if len(kv_hash_seeds) > 1:
            raise ValueError(
                "All params must use the same kv_hash_seed, got:"
                f" {kv_hash_seeds}"
            )

    @property
    def _first(self) -> KVCacheParamInterface:
        """Returns the first child param set."""
        return next(iter(self.children.values()))

    @property
    def n_devices(self) -> int:
        """Returns the number of devices."""
        return len(self.devices)

    @property
    def enable_prefix_caching(self) -> bool:
        """Whether prefix caching is enabled (shared across all caches)."""
        return self._first.enable_prefix_caching

    @property
    def kv_connector_config(self) -> Any:
        """Connector config (shared across all caches)."""
        return self._first.kv_connector_config

    @property
    def kv_hash_algo(self) -> KVHashAlgo:
        """Hash algorithm used for KV-cache block identity."""
        return self._first.kv_hash_algo

    @property
    def kv_hash_seed(self) -> bytes | None:
        """Resolved 32-byte cluster seed for sha256/sha256_64. None for ahash64."""
        return self._first.kv_hash_seed

    @property
    def bytes_per_block(self) -> int:
        """Total bytes per block across all KV caches.

        Since all caches allocate memory for the same sequence, the total
        memory cost per block is the sum across all param sets.
        """
        return sum(p.bytes_per_block for p in self.children.values())

    def get_symbolic_inputs(self) -> MultiKVCacheInputs[TensorType, BufferType]:
        """Returns the symbolic inputs for the KV cache tree."""
        return MultiKVCacheInputs(
            children={
                k: p.get_symbolic_inputs() for k, p in self.children.items()
            }
        )

    def unflatten_kv_inputs(
        self, it: Iterator[Any]
    ) -> MultiKVCacheInputs[TensorValue, BufferValue]:
        """Unflattens the KV cache inputs from a graph-input iterator."""
        return self.get_symbolic_inputs().unflatten(it)

    def unflatten_basic_kv_tree(
        self, it: Iterator[Any]
    ) -> tuple[list[KVCacheInputsPerDevice[TensorValue, BufferValue]], ...]:
        """Unflattens a basic KV tree from a graph-input iterator.

        Requires that the model is a basic height-1 tree. This method does not work
        on nested trees.
        """
        tree = self.unflatten_kv_inputs(it)
        assert isinstance(tree, MultiKVCacheInputs)
        out: list[list[KVCacheInputsPerDevice[TensorValue, BufferValue]]] = []
        for child in tree.children.values():
            if not isinstance(child, KVCacheInputs):
                raise ValueError("Unable to flatten nested KV tree")
            out.append(list(child.inputs))
        return tuple(out)

    @property
    def replicates_kv_across_tp(self) -> bool:
        """Whether every device holds identical KV state."""
        return self._first.replicates_kv_across_tp

    @property
    def tensor_parallel_degree(self) -> int:
        """Returns the tensor parallel degree."""
        return self._first.tensor_parallel_degree

    def resolve_attn_key(
        self,
        batch_size: int,
        max_prompt_length: int,
        max_cache_valid_length: int,
    ) -> AttnKeyInterface:
        """Resolves the dispatch shape tree mirroring the cache tree."""
        return MultiAttnKey.from_dict(
            {
                k: p.resolve_attn_key(
                    batch_size, max_prompt_length, max_cache_valid_length
                )
                for k, p in self.children.items()
            }
        )

    def graph_capture_probe_cache_lengths(
        self, max_cache_length: int, q_max_seq_len: int = 1
    ) -> list[int]:
        """Returns the union of probe cache lengths across all child caches."""
        lengths: set[int] = set()
        for p in self.children.values():
            lengths.update(
                p.graph_capture_probe_cache_lengths(
                    max_cache_length, q_max_seq_len
                )
            )
        return sorted(lengths)

    def allocate_buffers(
        self, total_num_pages: int
    ) -> list[KVCacheBufferInterface]:
        """Allocates per-replica buffers for every cache in the tree.

        Returns one :class:`MultiKVCacheBuffer` per data-parallel replica,
        each holding that replica's :class:`KVCacheBuffer` for every child
        cache.
        """
        per_key = {
            k: p.allocate_buffers(total_num_pages)
            for k, p in self.children.items()
        }
        return [
            MultiKVCacheBuffer(
                children={k: per_key[k][replica_idx] for k in self.children}
            )
            for replica_idx in range(self.data_parallel_degree)
        ]

    def build_runtime_inputs(
        self,
        assignments: Sequence[KVCacheAssignments],
        buffers: Sequence[KVCacheBufferInterface],
    ) -> KVCacheInputsInterface[Buffer, Buffer]:
        """Builds the runtime KV-cache tree spanning all replicas.

        Each child leaf is built from every replica's assignment plus that
        replica's child buffer; the per-replica assignment (cache lengths /
        lookup table / dispatch shape) is shared across child caches since
        they all map the same sequence.
        """
        multi_buffers: list[MultiKVCacheBuffer] = []
        for buffer in buffers:
            assert isinstance(buffer, MultiKVCacheBuffer)
            multi_buffers.append(buffer)
        return MultiKVCacheInputs(
            children={
                k: p.build_runtime_inputs(
                    assignments, [b.children[k] for b in multi_buffers]
                )
                for k, p in self.children.items()
            }
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
            "Insufficient cache memory to allocate even a single page.\n"
            f"One page requires {single_page_size_bytes_str} but only "
            f"{cache_memory_str} are available{across_x_devices_str}."
        )

    if max_batch_size is not None and max_batch_size > num_allocable_blocks:
        memory_needed_str = to_human_readable_bytes(
            max_batch_size * params.bytes_per_block
        )
        logger.warning(
            "Insufficient cache memory to support a batch containing"
            f" {max_batch_size} requests with one token per request. Need to"
            f" allocate at least {max_batch_size} pages ({memory_needed_str}),"
            f" but only have enough memory for {num_allocable_blocks} pages"
            f" ({cache_memory_str}{across_x_devices_str})."
        )

    if (
        max_blocks_per_req is not None
        and max_blocks_per_req > num_allocable_blocks
    ):
        memory_needed_str = to_human_readable_bytes(
            max_blocks_per_req * params.bytes_per_block
        )
        logger.warning(
            "Insufficient cache memory to support a batch containing one"
            f" request at the max sequence length of {max_seq_len} tokens. Need"
            f" to allocate at least {max_blocks_per_req} pages"
            f" ({memory_needed_str}), but only have enough memory for"
            f" {num_allocable_blocks} pages"
            f" ({cache_memory_str}{across_x_devices_str})."
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

    The host (CPU/disk) tier is a single pool shared across all data-parallel
    replicas via one connector (SERVOPT-1501), sized at
    ``host_kvcache_swap_space_gb`` total (independent of
    ``data_parallel_degree``). The connector is replica-agnostic, so a block
    offloaded by one replica can be served as a cache hit for another.

    Returns:
        The total number of blocks that can be allocated in the shared host
        pool.
    """
    if params.kv_connector not in (
        KVConnectorType.local,
        KVConnectorType.tiered,
    ):
        return 0
    assert params.host_kvcache_swap_space_gb is not None
    GiB = 1024 * 1024 * 1024
    host_bytes = params.host_kvcache_swap_space_gb * GiB

    bytes_per_block = params.bytes_per_block
    if params.replicates_kv_across_tp:
        # On cpu/disk, we don't need multiple replicas of the same KV state.
        assert bytes_per_block % params.tensor_parallel_degree == 0
        bytes_per_block = bytes_per_block // params.tensor_parallel_degree
    num_host_blocks = int(host_bytes // bytes_per_block)

    if num_host_blocks == 0:
        raise RuntimeError(
            "Insufficient cache memory to allocate even a single page.\nOne"
            " page requires"
            f" {to_human_readable_bytes(params.bytes_per_block)} but only"
            f" {to_human_readable_bytes(host_bytes)} are"
            " available on host."
        )

    return num_host_blocks
