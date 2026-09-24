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
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from typing import (
    Any,
    ClassVar,
    Literal,
    Protocol,
    TypeGuard,
    runtime_checkable,
)

import numpy as np
from max import tree
from max._kv_cache_ops import (
    mha_decode_num_partitions,
    mla_dispatch_args_scalar,
)
from max.driver import Buffer, Device, DevicePinnedBuffer
from max.dtype import DType
from max.experimental.sharding import TensorLayout
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    TensorType,
    TensorValue,
    ops,
)
from max.support.human_readable_formatter import to_human_readable_bytes
from max.support.math import ceildiv

from .data_parallelism_utils import split_into_groups
from .input_types import (
    KVCacheInputs,
    KVCacheInputsPerDevice,
    RecurrentLeafInputs,
    RecurrentStateInputsPerDevice,
)
from .utils import (
    AttnKeyInterface,
    MHAAttnKey,
    MLAAttnKey,
    MSAAttnKey,
    MultiAttnKey,
    padded_lut_cols,
)

# Mirror of max.pipelines.speculative.config.SpeculativeMethod. Defined
# inline rather than imported because max.pipelines.speculative depends
# on max.nn (BUILD.bazel), so importing back would create a circular
# bazel dependency. The two definitions are structurally identical
# Literals, so mypy treats them as the same type at use sites.
SpeculativeMethod = Literal["eagle", "mtp", "dflash", "dflash2"]

KVHashAlgo = Literal["ahash64", "sha256", "sha256_64"]
"""Supported hash algorithms for KV-cache block identity."""

logger = logging.getLogger("max.pipelines")


def _filter_tiny_cache_lengths(
    probe_lengths: list[int], num_draft_tokens: int
) -> list[int]:
    min_cache_length = 1 + 2 * num_draft_tokens
    return [cl for cl in probe_lengths if cl >= min_cache_length]


@dataclass(frozen=True)
class KVCacheGroupId:
    """Identifies the caches a model reuses and evicts together.

    Caches behind the same attention pattern share a prefix-cache hit and an
    external tier namespace, so this doubles as the key for both.

    ``recurrent`` names a cache whose entry is a state rather than a span of
    tokens. It shares the page pool, the prefix cache and the eviction order
    with the attention caches.

    ``scratch`` names a cache that is never published: one block per request,
    drawn when the request first grows and freed when it is released. It
    draws from the same page pool as the others, so its bytes stay fungible
    with theirs and admission counts them, but it carries no hash and so
    takes no part in prefix hits, commits, eviction order or an external
    tier.
    """

    type: Literal["full", "sliding_window", "recurrent", "scratch"]
    window_size: int = -1

    def __post_init__(self):
        if self.type == "full":
            if self.window_size != -1:
                raise ValueError("Window size must be -1 for full groups.")
        elif self.type == "sliding_window":
            # A window of 1 reads no history back: the query token attends to
            # itself alone, so every block is a prefix hit that attention
            # never looks at. The prefix rules used to serve that as a 100%
            # hit; they no longer carry the case, so it is rejected here
            # instead (SERVOPT-1627).
            if self.window_size <= 1:
                raise ValueError(
                    "Window size must be greater than 1 for sliding window"
                    f" groups, got {self.window_size}."
                )
        elif self.type == "recurrent":
            if self.window_size != -1:
                raise ValueError("Window size must be -1 for recurrent groups.")
        elif self.type == "scratch":
            if self.window_size != -1:
                raise ValueError("Window size must be -1 for scratch groups.")

    def is_sliding_window(self) -> bool:
        return self.type == "sliding_window"

    def is_full(self) -> bool:
        return self.type == "full"

    def blocks_in_window(self, page_size: int) -> int:
        if self.is_scratch():
            raise ValueError(
                "a scratch group has no window: it holds one block per request,"
                " and nothing downstream of a window should reach it"
            )
        if self.is_full():
            return -1
        return ceildiv(self.window_size - 1, page_size)

    def longest_hit(self, page_size: int, resident: Sequence[bool]) -> int:
        """How much of ``resident`` a leaf of this shape can reuse.

        Residency in, prefix length out. The answer is counted from the start
        even for a shape that only reads the tail of it, so the caller can
        compare shapes against one another -- which
        :func:`~max.pipelines.kv_cache.prefix_hit.longest_joint_prefix_hit`
        does to settle what a whole tree serves at once.

        ``len(resident)`` bounds the candidate, so narrowing means passing a
        shorter view, not recomputing residency.

        Computing eligible Prefix Cache hits for sliding window differs greatly
        from full attn. Recall that the window size includes the query token.
        Say the query token is idx=42 and the window size is 10. This means
        that the query token will attend to tokens from idx=32 to idx=41.

        For a concrete example:

        .. code-block:: text

            [X]: Token is in Prefix Cache
             . : Token is not in Prefix Cache
             ^ : Eligible Prefix Cache hit

              Tokens [A]  [B]   .   [D]  [E]  [F]   .    .   [I]  [J]  [K]  [L]  [M]
            w_size=2  ^    ^         ^    ^    ^              ^    ^    ^    ^    ^
            w_size=3  ^    ^              ^    ^                   ^    ^    ^    ^
            w_size=4  ^    ^                   ^                        ^    ^    ^
            w_size=5  ^    ^                                                 ^    ^
            w_size=6  ^    ^                                                      ^
            w_size=7  ^    ^

        Notice that as window_size increases, the number of indices eligible for
        a cache hit decreases. Additionally, we can count consecutive runs of
        window_size-1 tokens to determine eligibility. For example, [DEF] is a
        run of 3 tokens so token F is a valid cache hit for w_size=4 and below.

        Additionally, partial window cache hits is possible if the run starts from
        the start of sequence. For example, [A] and [AB] are valid cache hits for
        any window size.

        window_size=1 is not a case this has to serve: ``__post_init__``
        rejects it, since a query token attending to no historical tokens
        would make every block a hit attention never reads back.

        Args:
            page_size: Tokens per block, which turns ``window_size`` into a
                block count.
            resident: Whether each block of the chain is held, positionally.
        """
        num_hashes = len(resident)
        if self.is_scratch():
            # Never published, so it has no opinion and must not shorten what
            # the leaves that do cache agree on.
            return num_hashes
        if self.is_recurrent():
            # A state is a single published boundary rather than a run, so
            # the deepest one that stands is the answer.
            for idx in range(num_hashes - 1, -1, -1):
                if resident[idx]:
                    return idx + 1
            return 0
        if self.is_full():
            # Reads its whole history, so the hit is the run from the root.
            for idx in range(num_hashes):
                if not resident[idx]:
                    return idx
            return num_hashes

        # Sliding window: the hit is a SUFFIX run, so walk back counting a
        # consecutive run and take the first place it fills the window. A
        # shallower stopping point covers different blocks, so this cannot be
        # found by shortening a full-attention answer.
        blocks_in_window = self.blocks_in_window(page_size)
        if blocks_in_window < 1:
            # window_size == 1, a query attending to no history, which would
            # make every block a hit attention never reads back. __post_init__
            # rejects that window and so does Mach's KVCacheConfig::validate,
            # so one arriving here is a caller bug (SERVOPT-1627).
            raise ValueError(
                "A sliding-window group spans at least one block; got"
                f" blocks_in_window={blocks_in_window}. window_size must be"
                " greater than 1."
            )
        run = 0
        for idx in range(num_hashes - 1, -1, -1):
            if not resident[idx]:
                run = 0
                continue
            run += 1
            if run >= blocks_in_window:
                return idx + run
        # A run reaching the root is a hit of just that run: nothing sits
        # below it to be missing.
        return run

    def is_recurrent(self) -> bool:
        return self.type == "recurrent"

    def is_scratch(self) -> bool:
        return self.type == "scratch"

    @classmethod
    def full(cls) -> KVCacheGroupId:
        return cls(type="full")

    @classmethod
    def recurrent(cls) -> KVCacheGroupId:
        return cls(type="recurrent")

    @classmethod
    def scratch(cls) -> KVCacheGroupId:
        return cls(type="scratch")

    def __repr__(self) -> str:
        if self.type == "full":
            return "full_group"
        elif self.type == "sliding_window":
            return f"sliding_window_group({self.window_size})"
        elif self.type == "recurrent":
            return "recurrent_group"
        elif self.type == "scratch":
            return "scratch_group"


class KVConnectorType(str, Enum):
    """Identifies which off-device backing store the KV cache uses.

    Set on the connector config's ``type`` field to control whether evicted
    cache pages stay on device only, tier across host and disk, or route
    through a distributed block store. MAX currently supports only
    :attr:`null`.
    """

    null = "null"
    """No off-device backing store. Pages live on device only."""

    tiered = "tiered"
    """Tiers evicted pages across host memory and disk.

    MAX currently doesn't support this connector type.

    Requires ``enable_prefix_caching``.

    .. deprecated::
        A backward-compatible alias for :attr:`rust_tiered`; the Python
        implementation was removed.
    """

    rust_tiered = "rust_tiered"
    """Tiers evicted pages across host memory and disk, backed by the Rust
    ``kv_tier_connector`` extension.

    MAX currently doesn't support this connector type.

    The only host/disk tiered implementation, and what :attr:`tiered` now
    resolves to: it runs its copies and disk I/O on Rust threads (no GIL
    contention) and overlaps onloads with GPU compute via asynchronous
    transfer handles. Requires ``enable_prefix_caching``. Raises on
    non-CUDA/HIP devices.
    """

    dkv = "dkv"
    """Routes pages through a distributed KV block store.

    MAX currently doesn't support this connector type.

    Requires a ``block_store_endpoint`` on the connector config.
    """


@runtime_checkable
class KVConnectorConfigInterface(Protocol):
    """The KV connector configuration contract: a type plus per-tier settings.

    Declared here because :class:`KVCacheParams` carries it, and implemented by
    the Pydantic ``KVConnectorConfig`` in the pipelines layer (which owns CLI
    and config-file parsing). Structural typing keeps ``max.nn`` free of a
    Pydantic dependency, which the base ``max`` wheel does not ship, while
    still giving every consumer a checked type instead of ``Any``.
    """

    @property
    def type(self) -> KVConnectorType:
        """Which off-device backing store to use."""
        ...

    @property
    def host_offload_max_gb(self) -> float | None:
        """Host budget in GiB; ``None`` sizes it from the device pool."""
        ...

    @property
    def disk_offload_max_gb(self) -> float | None:
        """Disk budget in GiB; ``None`` sizes it from the device pool."""
        ...

    @property
    def disk_offload_dir(self) -> str | None:
        """Disk cache directory; ``None`` means auto-create one."""
        ...

    @property
    def num_disk_workers(self) -> int:
        """Disk I/O worker threads for the tiered connectors."""
        ...

    @property
    def block_store_endpoint(self) -> str | None:
        """Endpoint for the co-located dKV service."""
        ...


@dataclass(frozen=True)
class NullKVConnectorConfig:
    """Connector config for no off-device backing store.

    The default for :attr:`KVCacheParams.kv_connector_config`, so the field is
    never ``None`` and every reader can go straight to ``.type``.
    """

    type: KVConnectorType = KVConnectorType.null
    host_offload_max_gb: float | None = None
    disk_offload_max_gb: float | None = None
    disk_offload_dir: str | None = None
    num_disk_workers: int = 32
    block_store_endpoint: str | None = None


def _validate_is_2d_uint8_buffer(buffer: Buffer) -> None:
    if len(buffer.shape) != 2:
        raise ValueError("KVCacheMemory buffer must have 2 dimensions")
    if buffer.dtype != DType.uint8:
        raise ValueError("KVCacheMemory buffer must have dtype uint8")
    # TODO(MXSERV-502): the offload path addresses a page as
    # `base + idx * bytes_per_page`, so a padded leaf needs the stride and the
    # copy length separated before it can be offloaded.
    if not buffer.is_contiguous:
        raise ValueError(
            "KVCacheMemory buffer must be contiguous, but this one has "
            f"shape {tuple(buffer.shape)} with strides "
            f"{tuple(buffer.strides)}: its pages are padded, and the offload "
            "and transfer paths cannot yet stride by a distance that differs "
            "from the page they copy. Serve without a KVConnector, or without "
            "disaggregation, until MXSERV-502 lands."
        )


# The flat scale TMA starts a copy at a key index, so the scale pool's
# page-to-page stride has to stay 16-byte aligned; see `scale_align_elems` and
# the `create_index_scale_tma_tile` assert in `kv_cache/types.mojo`.
_SCALE_TMA_ALIGN_BYTES = 16

PACKED_PAGE_STRIDE = -1
"""``page_stride`` sentinel meaning the pages are packed."""


def packed_page_stride(like: Any) -> Any:
    """Returns the packed ``page_stride`` sentinel, typed to match ``like``."""
    if isinstance(like, Buffer):
        return Buffer.from_numpy(np.array([PACKED_PAGE_STRIDE], dtype=np.int64))
    if isinstance(like, BufferType):
        return TensorType(DType.int64, shape=[1], device=DeviceRef.CPU())
    if isinstance(like, TensorLayout):
        return TensorLayout(DType.int64, [1], DeviceRef.CPU())
    return ops.constant(
        [PACKED_PAGE_STRIDE], DType.int64, device=DeviceRef.CPU()
    )


def _page_stride_buffer(pages: Buffer) -> Buffer:
    """Wraps a page view's stride as the rank-1 int64 tensor the ops take."""
    return Buffer.from_numpy(np.array([page_stride_of(pages)], dtype=np.int64))


def page_view(
    slab: Buffer,
    shape: Sequence[int],
    dtype: DType,
    padded_page_bytes: int | None = None,
) -> Buffer:
    """Views ``slab`` as ``[num_pages, *shape]`` pages.

    A padded leaf gets a strided view sliced out of the padded grid, so its
    dim-0 stride is the padded distance while its extent is only the data.

    Args:
        slab: The backing allocation, typically the raw ``uint8`` arena.
        shape: One page's data shape, without the leading page dimension.
        dtype: Element type of the page data.
        padded_page_bytes: The pool's padded page size, or ``None`` if packed.

    Returns:
        The page view, strided iff the leaf was padded.
    """
    data_bytes = math.prod(shape) * dtype.size_in_bytes
    total_bytes = slab.num_elements * slab.dtype.size_in_bytes

    if padded_page_bytes is None or padded_page_bytes == data_bytes:
        num_pages = total_bytes // data_bytes
        if num_pages * data_bytes != total_bytes:
            raise ValueError(
                f"Packed pages must tile the slab exactly, but {total_bytes} "
                f"bytes does not divide into pages of {data_bytes}."
            )
        return slab.view(shape=(num_pages, *shape), dtype=dtype)

    if padded_page_bytes < data_bytes:
        raise ValueError(
            f"A padded page cannot be smaller than the data it holds: "
            f"{padded_page_bytes} < {data_bytes}."
        )
    if padded_page_bytes % dtype.size_in_bytes:
        raise ValueError(
            f"Padded page of {padded_page_bytes} bytes is not a whole number "
            f"of {dtype} elements, so it has no stride in elements."
        )

    num_pages = total_bytes // padded_page_bytes
    # Slicing keeps the grid's dim-0 stride, which is exactly the page-to-page
    # distance this view needs to carry.
    grid = slab.view(
        dtype=dtype,
        shape=(num_pages, padded_page_bytes // dtype.size_in_bytes),
    )
    return grid[:, : data_bytes // dtype.size_in_bytes].view(
        dtype=dtype, shape=(num_pages, *shape)
    )


def page_stride_of(pages: Buffer) -> int:
    """Returns the page-to-page distance a page view carries, in elements.

    ``PACKED_PAGE_STRIDE`` when the view is contiguous.
    """
    return PACKED_PAGE_STRIDE if pages.is_contiguous else pages.strides[0]


def contiguous_page_view_and_stride(
    slab: Buffer,
    shape: Sequence[int],
    dtype: DType,
    padded_page_bytes: int | None = None,
) -> tuple[Buffer, int]:
    """Builds a contiguous page view for model execution, plus its stride.

    Model execution rejects non-contiguous buffers, so a padded leaf is bound
    as a packed view with the stride passed beside it. That view is shorter
    than the span the stride reaches; the memory past it belongs to the arena.

    Args:
        slab: The backing allocation, typically the raw ``uint8`` arena.
        shape: One page's data shape, without the leading page dimension.
        dtype: Element type of the page data.
        padded_page_bytes: The pool's padded page size, or ``None`` if packed.

    Returns:
        A contiguous ``[num_pages, *shape]`` buffer, and the page stride in
        elements -- ``PACKED_PAGE_STRIDE`` when the leaf was never padded.
    """
    data_bytes = math.prod(shape) * dtype.size_in_bytes
    if padded_page_bytes is None or padded_page_bytes == data_bytes:
        return page_view(slab, shape, dtype), PACKED_PAGE_STRIDE

    total_bytes = slab.num_elements * slab.dtype.size_in_bytes
    num_pages = total_bytes // padded_page_bytes
    flat = slab.view(dtype=dtype, shape=(total_bytes // dtype.size_in_bytes,))
    packed = math.prod(shape)
    return (
        flat[: num_pages * packed].view(dtype=dtype, shape=(num_pages, *shape)),
        padded_page_bytes // dtype.size_in_bytes,
    )


def _scales_leaf_id(leaf_id: str) -> str:
    """Returns the id of the leaf holding ``leaf_id``'s quantization scales."""
    return leaf_id + "/scales"


def _view_as_uint8_pages(buffer: Buffer) -> Buffer:
    """Re-view a KV buffer as a 2-D ``[num_pages, bytes_per_page]`` uint8 array.

    The original dtype and per-page element count are folded into a flat
    per-page byte stride so the offload engine and transfer engine can treat
    every cache uniformly regardless of dtype or shape.
    """
    return buffer.view(
        dtype=DType.uint8,
        shape=[
            buffer.shape[0],
            buffer.num_elements * buffer.dtype.size_in_bytes // buffer.shape[0],
        ],
    )


@dataclass
class KVCacheMemory:
    """One logical ``(child, kind)`` KV tensor as per-TP-shard ``uint8`` views.

    A unit is one logical tensor — a cache's ``values`` or its ``scales`` —
    holding a 2-D ``[num_pages, bytes_per_page]`` view per TP shard in canonical
    device order.

    ``replicated`` indicates that all buffers hold identical bytes. This is true
    for certain cases like TP + MLA, TP + MiniMaxM3IndexerAttn, etc.
    """

    replicated: bool
    buffers: list[Buffer]

    def __post_init__(self) -> None:
        if len(self.buffers) == 0:
            raise ValueError("KVCacheMemory must have at least one buffer")
        for buffer in self.buffers:
            _validate_is_2d_uint8_buffer(buffer)
        first_shape = self.buffers[0].shape
        for i, buffer in enumerate(self.buffers):
            if buffer.shape != first_shape:
                raise ValueError(
                    f"All buffers in a KVCacheMemory must share a shape, "
                    f"but shard {i} has shape {buffer.shape} vs shard 0's "
                    f"{first_shape}. bytes_per_page/total_num_pages are read "
                    f"off shard 0 and would silently report the wrong value "
                    f"for a mismatched shard."
                )
        if self.replicated and len(self.buffers) <= 1:
            raise ValueError(
                "replicated=True requires at least 2 TP-shard buffers"
            )

    @property
    def bytes_per_page(self) -> int:
        """Returns the per-page byte stride shared by every shard."""
        return self.buffers[0].shape[1]

    @property
    def host_bytes_per_page(self) -> int:
        """Returns the width of one host block row holding this unit's page.

        A replicated (MLA) unit contributes its stride once -- one copy is
        stored and broadcast back on load, so counting its peers would double
        the pinned host allocation. Must match across replicas, so a block
        written by one is readable by another.
        """
        return self.bytes_per_page * (
            1 if self.replicated else len(self.buffers)
        )

    @property
    def total_num_pages(self) -> int:
        """Returns the total number of pages (including the null block)."""
        return self.buffers[0].shape[0]


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

    def to_memory(self) -> Mapping[str, KVCacheMemory]:
        """Returns the offload-ready memory units, keyed by pool leaf id."""
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

    def to_memory(self) -> Mapping[str, KVCacheMemory]:
        """Returns the offload-ready memory units for all children.

        Raises:
            ValueError: If two children claim the same leaf.
        """
        memories: dict[str, KVCacheMemory] = {}
        for child in self.children.values():
            for leaf_id, memory in child.to_memory().items():
                if leaf_id in memories:
                    raise ValueError(f"Duplicate cache leaf {leaf_id!r}")
                memories[leaf_id] = memory
        return memories


@dataclass
class KVCacheBuffer(KVCacheBufferInterface):
    """A collection of KVCache buffers for one data-parallel replica.

    Two buffer kinds are supported: ``values`` and (optionally, for FP8
    quantization) ``scales``. The length of each list corresponds to the
    tensor-parallel degree, with one buffer per TP shard.

    ``replicates_kv_across_tp`` is ``True`` when the KV data is replicated
    identically across TP shards and ``False`` when it is sharded. The data is
    replicated in certain cases like TP + MLA, TP + MiniMaxM3IndexerAttn, etc.
    """

    leaf_id: str = field(kw_only=True)
    """The pool leaf these pages belong to, as ``leaves()`` names it."""

    replicates_kv_across_tp: bool
    values: list[Buffer]
    """Page views, strided when the pool padded this leaf. The canonical form:
    the page stride is read off these."""
    values_packed: list[Buffer] | None = None
    """Contiguous aliases of :attr:`values`, for binding as graph inputs.

    Model execution rejects a non-contiguous buffer, so a padded leaf cannot be
    bound as its strided view. These cover the same allocation packed, and are
    deliberately *shorter* than the span their stride reaches -- see
    ``contiguous_page_view_and_stride``. ``None`` when nothing was padded,
    in which case :attr:`values` is already contiguous."""
    scales: list[Buffer] | None = None
    """Per-TP-shard scale buffers for a quantized cache; ``None`` when
    unquantized."""
    scales_packed: list[Buffer] | None = None
    """Contiguous aliases of :attr:`scales`; see :attr:`values_packed`."""
    values_per_layer: list[list[Buffer]] | None = None
    """Per-TP-shard, per-layer value buffers when the pool uses
    :attr:`~max.nn.kv_cache.KVCacheParams.per_layer_buffers`.

    ``values_per_layer[shard]`` is the list of single-layer buffers for that
    shard, and ``values[shard]`` aliases ``values_per_layer[shard][0]`` so the
    single-buffer ``values`` invariants (and consumers) stay valid. ``None``
    for a normal single multi-layer buffer."""
    scales_per_layer: list[list[Buffer]] | None = None
    """Per-TP-shard, per-layer scale buffers for a quantized KV cache backed by
    :attr:`~max.nn.kv_cache.KVCacheParams.per_layer_buffers` (mirrors
    :attr:`values_per_layer`). ``scales[shard]`` aliases
    ``scales_per_layer[shard][0]``. ``None`` for a single multi-layer scale
    buffer or an unquantized cache."""
    is_jenga: bool = False
    """Whether this buffer is associated with Jenga KV cache

    TODO: Delete this field after reworking KVCacheBufferInterface.
    """

    def __post_init__(self) -> None:
        all_buffers = self.all_buffers

        if len(self.values) == 0:
            raise ValueError("List of values must be non-empty")

        if self.values_per_layer is not None:
            if len(self.values_per_layer) != len(self.values):
                raise ValueError(
                    "values_per_layer must have one entry per TP shard"
                )
            for shard_layers, value in zip(
                self.values_per_layer, self.values, strict=True
            ):
                if len(shard_layers) == 0:
                    raise ValueError(
                        "each values_per_layer shard must be non-empty"
                    )
                if shard_layers[0] is not value:
                    raise ValueError(
                        "values[i] must alias values_per_layer[i][0]"
                    )

        if self.scales_per_layer is not None:
            assert self.scales is not None
            if len(self.scales_per_layer) != len(self.scales):
                raise ValueError(
                    "scales_per_layer must have one entry per TP shard"
                )
            for shard_layers, scale in zip(
                self.scales_per_layer, self.scales, strict=True
            ):
                if len(shard_layers) == 0:
                    raise ValueError(
                        "each scales_per_layer shard must be non-empty"
                    )
                if shard_layers[0] is not scale:
                    raise ValueError(
                        "scales[i] must alias scales_per_layer[i][0]"
                    )

        if self.replicates_kv_across_tp and len(self.values) <= 1:
            raise ValueError(
                "replicates_kv_across_tp=True requires at least 2 TP shards "
                "(len(values) > 1)"
            )

        unique_dtype = {b.dtype for b in self.values}
        if len(unique_dtype) > 1:
            raise ValueError("All values must have the same dtype")

        unique_shapes = {b.shape for b in self.values}
        if len(unique_shapes) > 1:
            raise ValueError("All values must have the same shape")

        unique_is_pinned = {
            isinstance(b, DevicePinnedBuffer) for b in all_buffers
        }
        if len(unique_is_pinned) > 1:
            raise ValueError(
                "All values (and scales if present) must be either all pinned "
                "or all non-pinned"
            )

        if self.scales is None:
            return

        if len(self.scales) != len(self.values):
            raise ValueError("Scales must be the same length as values")

        unique_dtype = {b.dtype for b in self.scales}
        if len(unique_dtype) > 1:
            raise ValueError("All scales must have the same dtype")

        unique_shapes = {b.shape for b in self.scales}
        if len(unique_shapes) > 1:
            raise ValueError("All scales must have the same shape")

        # Allow the number of pages to be different between values / scales only
        # for Jenga KV cache.
        # TODO: Get rid of this hack.
        unique_num_pages = {b.shape[0] for b in all_buffers}
        if not self.is_jenga and len(unique_num_pages) > 1:
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

    def to_memory(self) -> Mapping[str, KVCacheMemory]:
        """Converts to offload-ready memory units, keyed by pool leaf id.

        Every buffer is re-viewed as 2-D ``uint8`` pages so consumers can treat
        all caches uniformly regardless of dtype or shape.

        Per-layer buffers are deliberately not enumerated -- only each shard's
        layer-0 alias -- which is why ``allocate_buffers`` rejects
        ``per_layer_buffers`` alongside off-device connectors and DP > 1.

        Returns:
            This cache's values leaf, plus its scales leaf when quantized.
        """
        shards_by_leaf: dict[str, list[Buffer]] = {self.leaf_id: self.values}
        if self.scales is not None:
            shards_by_leaf[_scales_leaf_id(self.leaf_id)] = self.scales
        return {
            leaf_id: KVCacheMemory(
                replicated=self.replicates_kv_across_tp,
                buffers=[_view_as_uint8_pages(b) for b in shards],
            )
            for leaf_id, shards in shards_by_leaf.items()
        }


@dataclass
class RecurrentStateBuffer(KVCacheBufferInterface):
    """One replica's recurrent state pool, viewed as the pages it holds."""

    pages: dict[str, list[Buffer]]
    """Each state leaf's ``[num_blocks, bytes_per_page]`` uint8 pages, one
    buffer per device."""

    def __post_init__(self) -> None:
        if not self.pages:
            raise ValueError("RecurrentStateBuffer needs at least one leaf")
        for leaf_id, buffers in self.pages.items():
            if not buffers:
                raise ValueError(f"state leaf {leaf_id!r} has no device pages")

    @property
    def total_num_pages(self) -> int:
        """Returns the first leaf's block count, including the null block."""
        return next(iter(self.pages.values()))[0].shape[0]

    @property
    def all_buffers(self) -> list[Buffer]:
        """Returns every leaf's pages, leaf-major then device order."""
        return [buffer for buffers in self.pages.values() for buffer in buffers]

    def to_memory(self) -> Mapping[str, KVCacheMemory]:
        """Returns one unit per state leaf, keyed as ``leaves()`` names it."""
        return {
            leaf_id: KVCacheMemory(replicated=False, buffers=list(buffers))
            for leaf_id, buffers in self.pages.items()
        }


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

    :meth:`~max.pipelines.kv_cache.PagedKVCacheManager.runtime_inputs` uses
    it to resolve the dispatch
    key once: e.g. for graph-capture replay, ``max_cache_valid_length`` is
    aligned up to a cache length recorded during capture and every data-parallel
    replica must run the identical captured graph. The batch's real values must
    not exceed these.
    """

    batch_size: int
    """Upper bound on requests in the decode batch."""
    max_prompt_length: int
    """Upper bound on the batch's prompt length."""
    max_cache_valid_length: int
    """Upper bound on the batch's valid cache length."""


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

    staged_by_device: list[dict[str, Buffer]]
    """Each leaf's own inputs, per device.

    A leaf's own key holds the table its ops index the pool by. A leaf
    wanting more than one keys them under itself, the way quantized
    attention keys ``<leaf>/scales``.
    """

    max_prompt_length: Buffer
    max_cache_length: Buffer
    batch_characteristics: BatchCharacteristics


@dataclass(frozen=True)
class KVLeafRegion:
    """One addressable region of the cache pool."""

    leaf_id: str
    group_id: KVCacheGroupId
    bytes_per_page: int
    row_bytes: int = field(default=1, kw_only=True)
    """The granularity a padded page of this leaf must be a multiple of.

    One row, ``num_heads * head_size * dtype_size``, widened where a kernel
    needs more alignment than a row gives. Defaults to 1 for leaves not
    addressed by row."""

    @property
    def cacheable(self) -> bool:
        """Whether this leaf's blocks are addressed by content.

        False makes the leaf invisible to everything keyed on a hash: prefix
        hits, commits, and both directions of an external tier. It still
        draws from the pool and still counts against admission.
        """
        return not self.group_id.is_scratch()

    def blocks_to_reserve(self, num_blocks: int) -> int:
        """Returns how many blocks one request draws to fill ``num_blocks`` slots.

        Fewer than ``num_blocks`` when some slots hold the null block: a
        sliding window keeps only its span, and a state keeps only its live
        block and one checkpoint.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def staged_input_shapes(
        self, batch_size: int, num_blocks: int
    ) -> Mapping[str, tuple[tuple[int, ...], DType]]:
        """Returns the shape and dtype of each input a forward stages.

        Called once at the widest bounds a forward can reach, to allocate the
        tensors the graph binds, and again per forward to size host staging.

        Args:
            batch_size: Requests the forward runs.
            num_blocks: Blocks the deepest row of the batch reaches.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def write_staged_inputs(
        self,
        plans: Sequence[Sequence[int]],
        into: Mapping[str, np.ndarray],
    ) -> None:
        """Writes one forward's blocks into the arrays declared for them.

        Args:
            plans: The blocks each row touches, in batch-row order.
            into: The array to fill, per staged key.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def bound_row_copies(
        self, src: int, dst: int
    ) -> Mapping[str, tuple[range, range]]:
        """Returns the rows to copy between, per bound input.

        Empty for a leaf the graph reaches through a per-forward table, whose
        blocks are never overwritten in place.

        Args:
            src: The block to copy.
            dst: The block to copy it to.
        """
        return {}

    def bound_row_span(self, block: int) -> Mapping[str, range]:
        """Returns the rows one block occupies, per bound input.

        Empty for the same leaves :meth:`bound_row_copies` is empty for.

        Args:
            block: The block whose rows to name.
        """
        return {}


@dataclass(frozen=True)
class PagedKVLeafRegion(KVLeafRegion):
    """A leaf the graph reaches through a per-forward page table."""

    page_size: int
    """Number of tokens per page."""

    def blocks_to_reserve(self, num_blocks: int) -> int:
        """Returns ``num_blocks``, capped by the window when the leaf has one."""
        if self.group_id.is_full():
            return num_blocks
        return min(
            num_blocks, ceildiv(self.group_id.window_size, self.page_size)
        )

    def staged_input_shapes(
        self, batch_size: int, num_blocks: int
    ) -> Mapping[str, tuple[tuple[int, ...], DType]]:
        """Returns this leaf's lookup table."""
        # Padded so ``PagedKVCache``'s SIMD ``populate`` can load 16 uint32s
        # past any valid ``first_lut_idx`` without leaving the allocation.
        shape = (batch_size, padded_lut_cols(num_blocks))
        return {self.leaf_id: (shape, DType.uint32)}

    def write_staged_inputs(
        self,
        plans: Sequence[Sequence[int]],
        into: Mapping[str, np.ndarray],
    ) -> None:
        """Writes each row's block ids, zeroing the slots it does not reach."""
        table = into[self.leaf_id]
        # 0 is the pool's null block, so an unreached slot reads nothing.
        table.fill(0)
        for batch_idx, blocks in enumerate(plans):
            table[batch_idx, : len(blocks)] = blocks


@dataclass(frozen=True)
class RecurrentStateRegion:
    """Shape and dtype of one kind of recurrent state, for one pool leaf."""

    leaf_id: str
    num_layers: int
    row_shape: tuple[int, ...]
    """Shape of one layer's state, per device."""
    dtype: DType
    scratch: bool = False
    """Whether the region is per-request scratch rather than a checkpoint.

    A scratch region is drawn once per request and never published, so it is
    invisible to prefix hits, to eviction order and to an external tier, and
    its block does not rotate at a page boundary. Its rows are addressed
    exactly like a published state's, so a kernel reads it the same way.
    """

    @property
    def rows_dim(self) -> str:
        """Symbolic dim naming this leaf's row count."""
        return f"{self.leaf_id.replace('/', '_')}_rows"

    @property
    def row_elements(self) -> int:
        """Elements in one layer's state."""
        return math.prod(self.row_shape)

    @property
    def pool_key(self) -> str:
        """Key the leaf's flat pool view is staged under."""
        return f"{self.leaf_id}/pool"

    @property
    def bytes_per_page(self) -> int:
        """Bytes one page of this leaf holds on one device."""
        return self.num_layers * self.row_elements * self.dtype.size_in_bytes

    def rows_of(self, page: int) -> range:
        """Returns the rows a page's layers occupy, layer ``l`` at index ``l``."""
        base = page * self.num_layers
        return range(base, base + self.num_layers)


@dataclass(frozen=True)
class RecurrentKVLeafRegion(KVLeafRegion):
    """A leaf addressed by row: one block holds one request's whole state."""

    region: RecurrentStateRegion
    """The state region that names this leaf and sizes its rows."""

    def blocks_to_reserve(self, num_blocks: int) -> int:
        """Returns two: the live block, and at most one checkpoint behind it."""
        return 2

    def staged_input_shapes(
        self, batch_size: int, num_blocks: int
    ) -> Mapping[str, tuple[tuple[int, ...], DType]]:
        """Returns the row table each layer slices.

        The table is layer-major, one layer per row, so a layer's slice is a
        contiguous row and the graph keeps it a view instead of materializing
        a gather kernel. ``num_blocks`` is unused: a state is addressed by
        row, not by block.
        """
        rows = (self.region.num_layers, batch_size)
        return {self.region.leaf_id: (rows, DType.uint32)}

    def write_staged_inputs(
        self,
        plans: Sequence[Sequence[int]],
        into: Mapping[str, np.ndarray],
    ) -> None:
        """Folds each request's block into the rows its layers index."""
        live = into[self.region.leaf_id]
        for batch_idx, blocks in enumerate(plans):
            (block,) = blocks
            live[:, batch_idx] = self.region.rows_of(block)

    def bound_row_copies(
        self, src: int, dst: int
    ) -> Mapping[str, tuple[range, range]]:
        """Returns the state rows to copy, keyed where the pool is bound."""
        return {
            self.region.pool_key: (
                self.region.rows_of(src),
                self.region.rows_of(dst),
            )
        }

    def bound_row_span(self, block: int) -> Mapping[str, range]:
        """Returns the rows one block's layers occupy."""
        return {self.region.pool_key: self.region.rows_of(block)}


@dataclass(frozen=True)
class ScratchKVLeafRegion(RecurrentKVLeafRegion):
    """A row-addressed leaf drawn once per request and never published.

    Addressed exactly like :class:`RecurrentKVLeafRegion`, one row per layer,
    but its block does not rotate at a page boundary and carries no hash, so
    a request holds the same one from its first forward to its release.
    """

    def blocks_to_reserve(self, num_blocks: int) -> int:
        """Returns one: the block the request holds for its whole life."""
        return 1


class CacheLeafKind(Enum):
    """What a child of a cache tree holds."""

    ATTENTION = "attention"
    """Keys and values a span of tokens wrote, read through an attention op."""

    RECURRENT = "recurrent"
    """One fixed-size state carrying every token before it."""


@runtime_checkable
class CacheLeafParamInterface(Protocol):
    """What every child of a cache tree contributes: leaves, inputs, cost."""

    data_parallel_degree: int
    """Degree of data parallelism."""
    devices: Sequence[DeviceRef]
    """Devices to use for the cache."""

    page_size: int
    """Tokens a block covers, or zero where this child declares no pool."""

    @property
    def bytes_per_block(self) -> int:
        """Number of bytes per cache block.

        Zero for a cache whose entry is not a span of tokens.
        """
        ...

    kv_connector_config: KVConnectorConfigInterface
    speculative_method: SpeculativeMethod | None
    num_draft_tokens: int

    @property
    def enable_prefix_caching(self) -> bool: ...

    @property
    def enable_dp_cross_replica_prefix_copy(self) -> bool: ...

    @property
    def kv_hash_algo(self) -> KVHashAlgo: ...

    @property
    def kv_hash_seed(self) -> bytes | None: ...

    @property
    def replicates_kv_across_tp(self) -> bool: ...

    @property
    def leaf_kind(self) -> CacheLeafKind:
        """What this child holds."""
        ...

    @property
    def n_devices(self) -> int:
        """Returns the total number of devices."""
        ...

    def get_symbolic_inputs(
        self, namespace: str = ""
    ) -> KVCacheInputs[TensorType, BufferType]:
        """Returns the symbolic inputs for this cache.

        Args:
            namespace: Prefix that disambiguates this cache's page-pool
                symbolic dim from sibling caches in a multi-group tree. Empty
                for a single-group cache, leaving its names unchanged.
        """
        ...

    def flattened_kv_inputs(self) -> list[TensorType | BufferType]:
        """Flattens the symbolic inputs for this cache."""
        return tree.leaves(self.get_symbolic_inputs())

    def unflatten_kv_inputs(
        self, it: Iterator[Any]
    ) -> KVCacheInputs[TensorValue, BufferValue]:
        """Unflattens the symbolic inputs for this cache."""
        ...

    def build_runtime_inputs(
        self,
        assignments: Sequence[KVCacheAssignments],
        buffers: Sequence[KVCacheBufferInterface],
        _prefix: str = "",
    ) -> KVCacheInputs[Buffer, Buffer]:
        """Builds the runtime cache inputs spanning all replicas.

        ``assignments`` and ``buffers`` are indexed by data-parallel replica.
        Returns the :class:`KVCacheInputs` pytree (a tuple of per-device
        leaves, or a dict of named subtrees for multi-cache models) whose
        leaves each hold one ``(replica, TP shard)`` device's inputs."""
        ...

    def leaves(self, _prefix: str = "") -> Mapping[str, KVLeafRegion]:
        """Returns the leaves this cache contributes to the pool."""
        ...

    def allocate_buffers(
        self, total_num_pages: int, _prefix: str = ""
    ) -> Sequence[KVCacheBufferInterface]:
        """Allocates the buffers for the cache, one per replica.

        Empty for a cache with no buffer an op indexes.

        Args:
            total_num_pages: Pages the pool holds, including the null block.
            _prefix: Names the buffers' leaves the way :meth:`leaves` does.
        """
        ...

    def slab_to_bound_views(
        self, slabs: Sequence[Buffer]
    ) -> Mapping[str, list[Buffer]]:
        """Returns the views this cache binds once and never restages.

        Keyed the way its leaves read them back. Empty for a cache whose
        pages the graph reaches through a per-forward table.

        Args:
            slabs: One replica's slab per device.
        """
        ...

    def slab_to_buffer_views(
        self,
        buffers: Sequence[Buffer],
        padded_page_bytes: Mapping[str, int] | None = None,
        _prefix: str = "",
    ) -> KVCacheBufferInterface:
        """Converts one replica's slabs into the pages its leaves occupy.

        Args:
            buffers: One replica's slab per device.
            padded_page_bytes: Each padded leaf's page stride, or ``None``.
            _prefix: Names the views' leaves the way :meth:`leaves` does.
        """
        ...


@runtime_checkable
class KVCacheParamInterface(CacheLeafParamInterface, Protocol):
    """A cache leaf a model reads through an attention op.

    It resolves a dispatch shape, names the cache lengths worth probing at
    graph capture, and hands out the paged buffers the op indexes. It also
    defines the pool: its page size, external tier, and block hash.
    """

    page_size: int
    """Number of tokens per page (block)."""
    kv_connector_config: KVConnectorConfigInterface
    """The KV connector's type and settings."""
    speculative_method: SpeculativeMethod | None = None
    num_draft_tokens: int = 0

    @property
    def enable_prefix_caching(self) -> bool:
        """Whether prefix caching is enabled."""
        ...

    @property
    def enable_dp_cross_replica_prefix_copy(self) -> bool:
        """Whether a prefix-cache hit resident on another data-parallel
        replica's device may be served by a device-to-device copy."""
        ...

    @property
    def num_draft_tokens_per_step(self) -> int:
        """Number of draft tokens written per draft forward.

        Zero when speculative decoding is disabled; one for autoregressive
        drafts (``eagle``, ``mtp``); equal to ``num_draft_tokens`` for block
        drafts (``dflash``, ``dflash2``).
        """
        if self.speculative_method is None:
            return 0
        elif self.speculative_method in ("dflash", "dflash2"):
            return self.num_draft_tokens
        elif self.speculative_method in ("mtp", "eagle"):
            return 1
        else:
            raise ValueError(
                f"Unrecognized speculative_method: {self.speculative_method!r}"
            )

    @property
    def kv_hash_algo(self) -> KVHashAlgo:
        """Hash algorithm used for KV-cache block identity."""
        ...

    @property
    def kv_hash_seed(self) -> bytes | None:
        """Resolved 32-byte cluster seed for ``sha256``/``sha256_64``.
        ``None`` for ``ahash64``."""
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

        Returns an :class:`AttnKeyInterface` for a single cache, or a
        ``MultiAttnKey`` tree mirroring the cache tree.
        """
        ...

    def graph_capture_probe_cache_lengths(
        self, max_cache_length: int, q_max_seq_len: int = 1
    ) -> list[int]:
        """Returns the cache lengths to probe during decode graph capture."""
        ...

    def unflatten_basic_kv_tree(
        self, it: Iterator[Any]
    ) -> tuple[list[KVCacheInputsPerDevice[TensorValue, BufferValue]], ...]:
        """Unflattens a basic KV tree from a graph-input iterator.

        Requires that the model is a basic height-1 tree. This method does not work
        on nested trees.
        """
        ...

    def per_request_row_bytes(self, blocks_per_request: int = 1) -> int:
        """Returns the bytes one request holds in row-addressed leaves.

        These are recurrent state and scratch leaves, which
        :meth:`bytes_per_block` does not cover. Summed across the
        tensor-parallel group.

        Args:
            blocks_per_request: Blocks a request spans, passed to each leaf's
                ``blocks_to_reserve``.
        """
        total = sum(
            leaf.blocks_to_reserve(blocks_per_request) * leaf.bytes_per_page
            for leaf in self.leaves().values()
            if leaf.group_id.is_recurrent() or leaf.group_id.is_scratch()
        )
        return total * self.tensor_parallel_degree


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

    leaf_kind: ClassVar[CacheLeafKind] = CacheLeafKind.ATTENTION

    num_layers: int
    """Number of layers in the model."""

    devices: Sequence[DeviceRef]
    """Devices to use for the KV cache."""

    enable_prefix_caching: bool = False
    """Whether to enable prefix caching for efficient reuse of common prompt prefixes."""

    enable_dp_cross_replica_prefix_copy: bool = True
    """Whether a prefix-cache block resident on another data-parallel (DP)
    replica's device may be materialized locally via a device-to-device copy
    to serve a cache hit. When False, cross-replica reuse is only served from
    the shared external tier via the KV connector (or recomputed). Only
    relevant when ``data_parallel_degree > 1`` and prefix caching is enabled."""

    per_layer_buffers: bool = False
    """When ``True``, allocate one standalone single-layer buffer per layer
    instead of one ``[..., num_layers, ...]`` multi-layer buffer.

    Each attention dispatch then binds only its own per-layer buffer, so the
    pool total can exceed a per-allocation size cap (e.g. a device's maximum
    single allocation) while every individual buffer stays under it. Defaults
    to ``False`` (one multi-layer buffer), keeping all other backends and
    models byte-identical."""

    kv_hash_algo: KVHashAlgo = "ahash64"
    """Hash algorithm used for KV-cache block identity."""

    kv_hash_seed: bytes | None = None
    """Resolved 32-byte cluster seed for ``sha256``/``sha256_64``. ``None``
    for ``ahash64``.

    Set by ``KVCacheConfig.to_params`` via ``resolve_kv_hash_seed``.
    """

    kv_connector_config: KVConnectorConfigInterface = field(
        default_factory=NullKVConnectorConfig
    )
    """Holds the connector type and its settings. The default is a ``null``
    connector (no external caching)."""

    page_size: int = 128
    """Number of tokens per page (block).

    This value is expressed in tokens, not bytes. The byte footprint of a page is
    derived from pipeline configuration.

    Current constraints: the page size must be a multiple of 128 and at least 128.
    """

    slots_per_page: int | None = None
    """Number of storage slots a page holds, which may be fewer than the
    :attr:`page_size` tokens it covers.

    ``None`` (the default) means one slot per token; construction resolves it
    to ``page_size``, so every later reader sees an ``int``. An architecture
    whose attention compresses a run of tokens into a single cached entry -- a
    pooled key, a compressed latent -- sets this to the number of entries a
    page's tokens compress into, and gets a page that many slots deep instead.
    It must divide :attr:`page_size` evenly, so a page never ends mid-slot.

    Only the physical buffer shape and the kernels reading it are affected:
    block allocation, the page table and prefix caching stay in tokens, and a
    block still covers ``page_size`` of them.
    """

    data_parallel_degree: int = 1
    """Degree of data parallelism. Devices are grouped replica-major, with
    ``n_devices // data_parallel_degree`` TP shards per replica."""

    kvcache_quant_config: KVCacheQuantizationConfig | None = None
    """KVCache quantization config. Currently only FP8 quantization supported."""

    speculative_method: SpeculativeMethod | None = None
    """Speculative decoding method propagated from
    ``SpeculativeConfig``."""

    num_draft_tokens: int = 0
    """Total draft tokens generated per speculative iteration.

    Zero when no speculative decoding is configured."""

    window_size: int | None = None
    """Window size for the sliding window attention. None for global attention."""

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

        if self.slots_per_page is None:
            self.slots_per_page = self.page_size
        if (
            self.slots_per_page <= 0
            or self.page_size % self.slots_per_page != 0
        ):
            raise ValueError(
                f"Slots per page ({self.slots_per_page}) must be a positive"
                f" divisor of the page size ({self.page_size})."
            )

        # Validate connector configuration
        connector = self.kv_connector_config.type
        if connector in (
            KVConnectorType.tiered,
            KVConnectorType.rust_tiered,
        ):
            if not self.enable_prefix_caching:
                raise ValueError(
                    f"KV connector '{connector.value}' requires prefix"
                    " caching to be enabled"
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
        """Returns whether KV cache quantization is enabled."""
        # Supported quantized-KV storage schemes: FP8_E4M3 (fp32 / e8m0 scales)
        # and int8 (fp16 per-block absmax scales).
        if self.kvcache_quant_config is None:
            return False
        value_dtypes = (
            DType.float8_e4m3fn,
            DType.float8_e4m3fnuz,
            DType.int8,
        )
        scale_dtypes = (
            DType.float32,
            DType.float8_e8m0fnu,
            DType.float16,
        )
        return (
            self.dtype in value_dtypes
            and self.kvcache_quant_config.scale_dtype in scale_dtypes
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
        """Returns the number of key/value tensors each cache slot holds."""
        raise NotImplementedError

    @property
    def n_kv_heads_per_device(self) -> int:
        """Returns the number of KV attention heads on one device."""
        raise NotImplementedError

    @property
    def shape_per_block(self) -> list[int]:
        """Returns the shape of each cache block.

        The slot dimension is :attr:`slots_per_page`, which equals
        :attr:`page_size` unless the architecture stores a page's tokens
        compressed.

        Returns:
            The shape of the cache block.
        """
        assert self.slots_per_page is not None
        # split k and v caches across a single dim
        # 0 = key
        # 1 = value
        return [
            self.kv_dim,
            self.num_layers,
            self.slots_per_page,
            self.n_kv_heads_per_device,
            self.head_dim,
        ]

    @property
    def shape_per_layer_block(self) -> list[int]:
        """Returns the block shape for a single-layer buffer.

        Same as :attr:`shape_per_block` but with the layer dimension pinned to
        ``1``. Used when :attr:`per_layer_buffers` is set: the pool allocates
        ``num_layers`` such buffers per device instead of one multi-layer
        buffer. The attention kernel derives ``num_layers`` from this dim, so a
        single-layer buffer (``num_layers == 1``) with ``layer_idx == 0`` is
        self-consistent.
        """
        kv_dim, _num_layers, slots, n_kv_heads, head_dim = self.shape_per_block
        return [kv_dim, 1, slots, n_kv_heads, head_dim]

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
    def shape_per_layer_scale_block(self) -> list[int]:
        """Scale-block shape for a single-layer buffer (layer dim pinned to 1).

        The scale analog of :attr:`shape_per_layer_block`: used with
        :attr:`per_layer_buffers` on a quantized cache, where the pool allocates
        one single-layer scale buffer per layer instead of one multi-layer one.
        """
        shape = self.shape_per_scale_block
        shape[1] = 1
        return shape

    @property
    def bytes_per_block(self) -> int:
        """Returns the number of bytes per cache block.

        When TP>1, each block is sharded across the devices in the tensor parallel group.
        This method returns the total memory needed to store a block across these devices.
        Includes memory needed for scales if quantization is enabled.

        Returns:
            The number of bytes per cache block.
        """
        return self.bytes_per_value_block + self.bytes_per_scale_block

    @property
    def bytes_per_value_block(self) -> int:
        """Returns the number of bytes per value block."""
        return (
            math.prod(self.shape_per_block)
            * self.dtype.size_in_bytes
            * self.tensor_parallel_degree
        )

    @property
    def row_bytes(self) -> int:
        """Returns one value row, ``num_heads * head_size * dtype_size``."""
        return (
            self.n_kv_heads_per_device
            * self.head_dim
            * self.dtype.size_in_bytes
        )

    @property
    def scale_row_bytes(self) -> int:
        """Returns one scale row; see :attr:`row_bytes`."""
        if not (
            self.quantized_kv_cache and self.kvcache_quant_config is not None
        ):
            return 1
        *_, n_kv_heads, granular_dim = self.shape_per_scale_block
        return (
            n_kv_heads
            * granular_dim
            * self.kvcache_quant_config.scale_dtype.size_in_bytes
        )

    @property
    def bytes_per_scale_block(self) -> int:
        """Returns the number of bytes per scale block."""
        if not (
            self.quantized_kv_cache and self.kvcache_quant_config is not None
        ):
            return 0
        return (
            math.prod(self.shape_per_scale_block)
            * self.kvcache_quant_config.scale_dtype.size_in_bytes
            * self.tensor_parallel_degree
        )

    def _get_symbolic_inputs_for_replica(
        self, replica_idx: int, prefix: str, page_namespace: str = ""
    ) -> list[KVCacheInputsPerDevice[TensorType, BufferType]]:
        raise NotImplementedError

    def get_symbolic_inputs(
        self, namespace: str = ""
    ) -> tuple[KVCacheInputsPerDevice[TensorType, BufferType], ...]:
        """Computes the symbolic inputs for the KV cache.

        Args:
            namespace: Prefix disambiguating this cache's per-pool page-count
                dim from sibling caches in a multi-group tree (empty for a
                single-group cache).

        Returns:
            The symbolic inputs for the KV cache.
        """
        input_symbols: list[KVCacheInputsPerDevice[TensorType, BufferType]] = []
        for replica_idx in range(len(self.devices_per_replica)):
            prefix = f"replica_{replica_idx}_"
            symbols = self._get_symbolic_inputs_for_replica(
                replica_idx, prefix, namespace
            )
            input_symbols.extend(symbols)
        return tuple(input_symbols)

    @cached_property
    def _kv_symbolic_treedef(self) -> tree.TreeDef:
        # TODO(SERVOPT-1505): avoid flattening symbolic inputs only to retain TreeDef for unflatten.
        return tree.flatten(self.get_symbolic_inputs())[1]

    def unflatten_kv_inputs(
        self, it: Iterator[Any]
    ) -> tuple[KVCacheInputsPerDevice[TensorValue, BufferValue], ...]:
        """Unflattens the KV cache inputs from a graph-input iterator."""
        return tuple(
            tree.leaves(
                tree.unflatten(self._kv_symbolic_treedef, it, exact=False),
                leaf=KVCacheInputsPerDevice,
            )
        )

    def allocate_buffers(
        self, total_num_pages: int, _prefix: str = ""
    ) -> list[KVCacheBuffer]:
        """Allocates the buffers for the KV cache."""
        if self.per_layer_buffers:
            # Validate the per-layer configuration before materializing any
            # device buffers. These guards live here, not in ``__post_init__``,
            # because call sites set ``per_layer_buffers`` after construction.
            if self.num_layers < 1:
                # ``values`` aliases layer 0, so the pool needs at least one
                # layer; otherwise ``layer_buffers[0]`` below raises an opaque
                # IndexError.
                raise ValueError(
                    f"per_layer_buffers requires num_layers >= 1, got {self.num_layers}"
                )
            connector = self.kv_connector_config.type
            if connector in (
                KVConnectorType.tiered,
                KVConnectorType.rust_tiered,
                KVConnectorType.dkv,
            ):
                # KVCacheBuffer.all_buffers / to_memory enumerate only the
                # layer-0 alias, so an off-device connector would move layers
                # 1..N-1 nowhere. Reject until that enumeration covers per-layer
                # buffers.
                raise NotImplementedError(
                    "per_layer_buffers is not supported with an off-device KV"
                    f" connector ('{connector.value}')"
                )
            if self.data_parallel_degree > 1:
                # Cross-replica block copy enumerates the same layer-0 alias.
                raise NotImplementedError(
                    "per_layer_buffers is not supported with data parallelism"
                    " (data_parallel_degree > 1)"
                )
        # ``Buffer.zeros`` needs concrete devices, so materialize the per-replica
        # device groups from the ``DeviceRef``s here.
        devices_per_replica = split_into_groups(
            x=[d.to_device() for d in self.devices],
            groups=self.data_parallel_degree,
        )
        kv_cache_buffers: list[KVCacheBuffer] = []
        for devices in devices_per_replica:
            values: list[Buffer] = []
            values_per_layer: list[list[Buffer]] | None = None
            if self.per_layer_buffers:
                # One standalone single-layer buffer per layer (num_layers==1 in
                # dim-2). ``values`` aliases each shard's layer-0 buffer so the
                # single-buffer invariants and consumers stay valid.
                values_per_layer = []
                for device in devices:
                    layer_buffers = [
                        Buffer.zeros(
                            shape=[
                                total_num_pages,
                                *self.shape_per_layer_block,
                            ],
                            dtype=self.dtype,
                            device=device,
                        )
                        for _ in range(self.num_layers)
                    ]
                    values_per_layer.append(layer_buffers)
                    values.append(layer_buffers[0])
            else:
                for device in devices:
                    value = Buffer.zeros(
                        shape=[total_num_pages, *self.shape_per_block],
                        dtype=self.dtype,
                        device=device,
                    )
                    values.append(value)

            scales: list[Buffer] | None = None
            scales_per_layer: list[list[Buffer]] | None = None
            if self.quantized_kv_cache:
                scales = []
                assert self.kvcache_quant_config is not None
                scale_dtype = self.kvcache_quant_config.scale_dtype
                if self.per_layer_buffers:
                    # One single-layer scale buffer per layer, parallel to
                    # ``values_per_layer``. ``scales`` aliases each shard's
                    # layer-0 scale so single-buffer consumers stay valid.
                    scales_per_layer = []
                    for device in devices:
                        layer_scales = [
                            Buffer.zeros(
                                shape=[
                                    total_num_pages,
                                    *self.shape_per_layer_scale_block,
                                ],
                                dtype=scale_dtype,
                                device=device,
                            )
                            for _ in range(self.num_layers)
                        ]
                        scales_per_layer.append(layer_scales)
                        scales.append(layer_scales[0])
                else:
                    for device in devices:
                        scale = Buffer.zeros(
                            shape=[
                                total_num_pages,
                                *self.shape_per_scale_block,
                            ],
                            dtype=scale_dtype,
                            device=device,
                        )
                        scales.append(scale)

            kv_cache_buffer = KVCacheBuffer(
                leaf_id=self.leaf_id(_prefix),
                values=values,
                scales=scales,
                replicates_kv_across_tp=self.replicates_kv_across_tp,
                values_per_layer=values_per_layer,
                scales_per_layer=scales_per_layer,
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
        scales_lookup_table: Buffer | None,
        target_key: AttnKeyInterface,
        draft_key: AttnKeyInterface | None,
        max_cache_valid_length: int,
        blocks_per_layer: list[Buffer] | None = None,
        scales_per_layer: list[Buffer] | None = None,
        *,
        page_stride: Buffer,
        scales_page_stride: Buffer | None = None,
    ) -> KVCacheInputsPerDevice[Buffer, Buffer]:
        raise NotImplementedError

    def build_runtime_inputs(
        self,
        assignments: Sequence[KVCacheAssignments],
        buffers: Sequence[KVCacheBufferInterface],
        _prefix: str = "",
    ) -> tuple[KVCacheInputsPerDevice[Buffer, Buffer], ...]:
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

            for i, (cl, luts, blocks) in enumerate(
                zip(
                    assignment.cache_lengths_by_device,
                    assignment.staged_by_device,
                    buffer.values,
                    strict=True,
                )
            ):
                device = blocks.device
                lut = luts[buffer.leaf_id]
                kv_scales = (
                    buffer.scales[i] if buffer.scales is not None else None
                )
                scales_lut = (
                    luts[_scales_leaf_id(buffer.leaf_id)]
                    if buffer.scales is not None
                    else None
                )
                blocks_per_layer = (
                    buffer.values_per_layer[i]
                    if buffer.values_per_layer is not None
                    else None
                )
                scales_per_layer = (
                    buffer.scales_per_layer[i]
                    if buffer.scales_per_layer is not None
                    else None
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
                        scales_lut,
                        target_key,
                        draft_key,
                        max_cl,
                        # Read off the buffers rather than recomputed, so the
                        # distance the kernel addresses with is the one the
                        # allocation was built with.
                        page_stride=_page_stride_buffer(blocks),
                        scales_page_stride=(
                            _page_stride_buffer(kv_scales)
                            if kv_scales is not None
                            else None
                        ),
                        blocks_per_layer=blocks_per_layer,
                        scales_per_layer=scales_per_layer,
                    )
                )
        return tuple(tp_shards)

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

    @property
    def group_id(self) -> KVCacheGroupId:
        """Returns the group id this cache pools under: ``sliding_window``
        with the window size when one is set, else ``full``."""
        if self.window_size is not None:
            return KVCacheGroupId(
                type="sliding_window", window_size=self.window_size
            )
        else:
            return KVCacheGroupId(type="full")

    def leaf_id(self, _prefix: str = "") -> str:
        """Returns the id this cache's pages are pooled under."""
        return _prefix + str(self.group_id)

    def leaves(self, _prefix: str = "") -> Mapping[str, KVLeafRegion]:
        """Returns the leaves of the KV cache.

        A leaf reports what one device holds, the unit a slab tiles in, where
        :attr:`bytes_per_block` counts a block across the whole replica. The
        division is exact: that figure is this one times the degree.
        """
        leaf_id = self.leaf_id(_prefix)
        leaves = {
            leaf_id: PagedKVLeafRegion(
                leaf_id=leaf_id,
                group_id=self.group_id,
                bytes_per_page=self.bytes_per_value_block
                // self.tensor_parallel_degree,
                row_bytes=self.row_bytes,
                page_size=self.page_size,
            )
        }

        if self.quantized_kv_cache:
            scales_id = _scales_leaf_id(leaf_id)
            leaves[scales_id] = PagedKVLeafRegion(
                leaf_id=scales_id,
                group_id=self.group_id,
                bytes_per_page=self.bytes_per_scale_block
                // self.tensor_parallel_degree,
                row_bytes=math.lcm(
                    self.scale_row_bytes, _SCALE_TMA_ALIGN_BYTES
                ),
                page_size=self.page_size,
            )

        return leaves

    def slab_to_bound_views(
        self, slabs: Sequence[Buffer]
    ) -> Mapping[str, list[Buffer]]:
        """Returns nothing: these pages are addressed by the lookup table."""
        return {}

    def slab_to_buffer_views(
        self,
        buffers: Sequence[Buffer],
        padded_page_bytes: Mapping[str, int] | None = None,
        _prefix: str = "",
    ) -> KVCacheBufferInterface:
        """Converts a slab of memory into per-leaf page views.

        Each view carries its own page-to-page distance in its dim-0 stride,
        so nothing downstream -- the pool, the connector, the transfer engine
        -- has to be told whether this pool padded its pages.

        Args:
            buffers: One raw arena slab per TP shard.
            padded_page_bytes: The pool's padded page size per leaf, keyed as
                :meth:`leaves` keys it. ``None`` means nothing was padded.
            _prefix: Leaf-id prefix identifying this node in a params tree.

        Returns:
            The buffer views for this cache.
        """
        padded = padded_page_bytes or {}
        quant_config = self.kvcache_quant_config
        values_id = self.leaf_id(_prefix)
        scales_id = _scales_leaf_id(values_id)
        quantized = self.quantized_kv_cache and quant_config is not None

        def views(
            shape: Sequence[int], dtype: DType, leaf_id: str
        ) -> tuple[list[Buffer], list[Buffer] | None]:
            page_bytes = padded.get(leaf_id)
            strided = [page_view(b, shape, dtype, page_bytes) for b in buffers]
            if page_bytes is None:
                return strided, None
            packed = [
                contiguous_page_view_and_stride(b, shape, dtype, page_bytes)[0]
                for b in buffers
            ]
            return strided, packed

        values, values_packed = views(
            self.shape_per_block, self.dtype, values_id
        )
        scales, scales_packed = (
            views(
                self.shape_per_scale_block,
                quant_config.scale_dtype,
                scales_id,
            )
            if quantized and quant_config is not None
            else (None, None)
        )
        return KVCacheBuffer(
            leaf_id=values_id,
            replicates_kv_across_tp=self.replicates_kv_across_tp,
            values=values,
            values_packed=values_packed,
            scales=scales,
            scales_packed=scales_packed,
            is_jenga=True,
        )


@dataclass(kw_only=True)
class MHAKVCacheParams(KVCacheParams):
    """KV cache parameters for multi-head attention (MHA)."""

    n_kv_heads: int
    """Total number of key-value attention heads across all devices."""

    allow_kv_head_replication: bool = False
    """Allows TP wider than ``n_kv_heads``. When set and ``n_devices`` is a
    multiple of ``n_kv_heads``, each KV head is replicated across a group of
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
        """Returns two: each slot holds a key and a value tensor."""
        return 2

    @property
    def n_kv_heads_per_device(self) -> int:
        """Returns the KV heads on one device, or ``1`` per device group when
        heads are replicated (``allow_kv_head_replication``)."""
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
            The resolved :class:`AttnKeyInterface`.
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
        self, replica_idx: int, prefix: str, page_namespace: str = ""
    ) -> list[KVCacheInputsPerDevice[TensorType, BufferType]]:
        devices = self.devices_per_replica[replica_idx]
        # Sibling cache groups may size their page pools independently.
        page_dim = page_namespace + "total_num_pages"
        # A Jenga pool tiles one slab at every leaf's own page size, so a
        # quantized leaf holds more scale pages than value pages. A legacy pool
        # binds the two counts equal, which a separate symbol still accepts.
        scale_page_dim = page_namespace + "total_num_scale_pages"

        def _blocks_per_layer(
            device: DeviceRef,
        ) -> list[BufferType] | None:
            # One single-layer BufferType per layer. Must stay in exact
            # lock-step with ``flatten``/``unflatten`` (tail order) and with the
            # runtime buffers built by ``allocate_buffers`` / ``build_runtime_inputs``.
            if not self.per_layer_buffers:
                return None
            return [
                BufferType(
                    self.dtype,
                    shape=[page_dim, *self.shape_per_layer_block],
                    device=device,
                )
                for _ in range(self.num_layers)
            ]

        def _scales_per_layer(
            device: DeviceRef,
        ) -> list[BufferType] | None:
            # Scale analog of ``_blocks_per_layer`` (per-layer + quantized).
            # Same lock-step requirement; appended after kv_blocks_per_layer.
            if not (self.per_layer_buffers and self.quantized_kv_cache):
                return None
            return [
                BufferType(
                    self.kv_cache_scale_dtype,
                    shape=[scale_page_dim, *self.shape_per_layer_scale_block],
                    device=device,
                )
                for _ in range(self.num_layers)
            ]

        def _kv_blocks(device: DeviceRef) -> BufferType:
            # ``per_layer_buffers`` aliases ``kv_blocks`` to the first per-layer
            # buffer so single-buffer consumers stay valid.
            if self.per_layer_buffers:
                return BufferType(
                    self.dtype,
                    shape=[page_dim, *self.shape_per_layer_block],
                    device=device,
                )
            return BufferType(
                self.dtype,
                shape=[page_dim, *self.shape_per_block],
                device=device,
            )

        def _lookup_table(device: DeviceRef) -> TensorType:
            return TensorType(
                DType.uint32,
                shape=[
                    prefix + "batch_size",
                    prefix + page_namespace + "max_num_pages",
                ],
                device=device,
            )

        return [
            KVCacheInputsPerDevice(
                kv_blocks=_kv_blocks(device),
                # Read off the buffer's stride when the inputs are bound.
                page_stride=TensorType(
                    DType.int64, shape=[1], device=DeviceRef.CPU()
                ),
                cache_lengths=TensorType(
                    DType.uint32,
                    shape=[prefix + "batch_size"],
                    device=device,
                ),
                lookup_table=_lookup_table(device),
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
                    # Per-layer buffers alias ``kv_scales`` to a single-layer
                    # scale (mirrors ``_kv_blocks`` for the KV data).
                    shape=[
                        scale_page_dim,
                        *(
                            self.shape_per_layer_scale_block
                            if self.per_layer_buffers
                            else self.shape_per_scale_block
                        ),
                    ],
                    device=device,
                )
                if self.quantized_kv_cache
                else None,
                # Present exactly when the scales are.
                scales_page_stride=TensorType(
                    DType.int64, shape=[1], device=DeviceRef.CPU()
                )
                if self.quantized_kv_cache
                else None,
                # Scales share the values' block-id space, so their lookup table
                # matches ``lookup_table``. Present exactly when the scales are.
                scales_lookup_table=_lookup_table(device)
                if self.quantized_kv_cache
                else None,
                attention_dispatch_metadata=self._attn_metadata_buffer(device),
                draft_attention_dispatch_metadata=self._attn_metadata_buffer(
                    device
                )
                if self.speculative_method is not None
                else None,
                kv_blocks_per_layer=_blocks_per_layer(device),
                kv_scales_per_layer=_scales_per_layer(device),
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
        scales_lookup_table: Buffer | None,
        target_key: AttnKeyInterface,
        draft_key: AttnKeyInterface | None,
        max_cache_valid_length: int,
        blocks_per_layer: list[Buffer] | None = None,
        scales_per_layer: list[Buffer] | None = None,
        *,
        page_stride: Buffer,
        scales_page_stride: Buffer | None = None,
    ) -> KVCacheInputsPerDevice[Buffer, Buffer]:
        return KVCacheInputsPerDevice(
            kv_blocks=blocks,
            page_stride=page_stride,
            cache_lengths=cache_lengths,
            lookup_table=lookup_table,
            max_prompt_length=max_prompt_length,
            max_cache_length=max_cache_length,
            kv_scales=kv_scales,
            scales_page_stride=scales_page_stride,
            scales_lookup_table=scales_lookup_table,
            attention_dispatch_metadata=target_key.pack_into_buffer(
                device, max_cache_valid_length
            ),
            draft_attention_dispatch_metadata=draft_key.pack_into_buffer(
                device, max_cache_valid_length
            )
            if draft_key is not None
            else None,
            kv_blocks_per_layer=blocks_per_layer,
            kv_scales_per_layer=scales_per_layer,
        )


@dataclass(kw_only=True)
class MLAKVCacheParams(KVCacheParams):
    """KV cache parameters for multi-latent attention (MLA)."""

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
        """Returns one, the single latent tensor each cache slot holds."""
        return 1

    @property
    def replicates_kv_across_tp(self) -> bool:
        """Whether every device holds identical KV state."""
        return self.tensor_parallel_degree > 1

    @property
    def n_kv_heads_per_device(self) -> int:
        """Returns one, the single latent head every device holds."""
        return 1

    @property
    def num_q_heads_per_device(self) -> int:
        """Returns the query attention heads on one device."""
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
            The resolved :class:`AttnKeyInterface`.
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
        return _filter_tiny_cache_lengths(probe_lengths, self.num_draft_tokens)

    def _get_symbolic_inputs_for_replica(
        self, replica_idx: int, prefix: str, page_namespace: str = ""
    ) -> list[KVCacheInputsPerDevice[TensorType, BufferType]]:
        devices = self.devices_per_replica[replica_idx]
        # Sibling cache groups may size their page pools independently.
        page_dim = page_namespace + "total_num_pages"
        # A Jenga pool tiles one slab at every leaf's own page size, so a
        # quantized leaf holds more scale pages than value pages. A legacy pool
        # binds the two counts equal, which a separate symbol still accepts.
        scale_page_dim = page_namespace + "total_num_scale_pages"

        def _lookup_table(device: DeviceRef) -> TensorType:
            return TensorType(
                DType.uint32,
                shape=[
                    prefix + "batch_size",
                    prefix + page_namespace + "max_num_pages",
                ],
                device=device,
            )

        return [
            KVCacheInputsPerDevice(
                kv_blocks=BufferType(
                    self.dtype,
                    shape=[page_dim, *self.shape_per_block],
                    device=device,
                ),
                # Read off the buffer's stride when the inputs are bound.
                page_stride=TensorType(
                    DType.int64, shape=[1], device=DeviceRef.CPU()
                ),
                cache_lengths=TensorType(
                    DType.uint32,
                    shape=[prefix + "batch_size"],
                    device=device,
                ),
                lookup_table=_lookup_table(device),
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
                    shape=[scale_page_dim, *self.shape_per_scale_block],
                    device=device,
                )
                if self.quantized_kv_cache
                else None,
                # Present exactly when the scales are.
                scales_page_stride=TensorType(
                    DType.int64, shape=[1], device=DeviceRef.CPU()
                )
                if self.quantized_kv_cache
                else None,
                # Scales share the values' block-id space, so their lookup table
                # matches ``lookup_table``. Present exactly when the scales are.
                scales_lookup_table=_lookup_table(device)
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
        scales_lookup_table: Buffer | None,
        target_key: AttnKeyInterface,
        draft_key: AttnKeyInterface | None,
        max_cache_valid_length: int,
        blocks_per_layer: list[Buffer] | None = None,
        scales_per_layer: list[Buffer] | None = None,
        *,
        page_stride: Buffer,
        scales_page_stride: Buffer | None = None,
    ) -> KVCacheInputsPerDevice[Buffer, Buffer]:
        # MLA never uses per-layer buffers; the parameters exist only to match
        # the base signature threaded by ``build_runtime_inputs``.
        assert blocks_per_layer is None
        assert scales_per_layer is None
        assert isinstance(target_key, MLAAttnKey)
        assert draft_key is None or isinstance(draft_key, MLAAttnKey)
        return KVCacheInputsPerDevice(
            kv_blocks=blocks,
            page_stride=page_stride,
            cache_lengths=cache_lengths,
            lookup_table=lookup_table,
            max_prompt_length=max_prompt_length,
            max_cache_length=max_cache_length,
            kv_scales=kv_scales,
            scales_page_stride=scales_page_stride,
            scales_lookup_table=scales_lookup_table,
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
    """KV cache parameters for multi-step attention (MSA)."""

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


@dataclass
class RecurrentStateParams(CacheLeafParamInterface):
    """A cache leaf whose entry is a state rather than a span of tokens.

    One fixed-size value carrying every token before it, drawn from the same
    slab, prefix index and eviction order as the attention caches.

    A cache may hold more than one -- a speculative pair keeps a state for
    the target and one for the draft. They are told apart by their regions'
    leaf ids, which name both the pool entry a state binds and the symbolic
    dim it declares, so each state chooses its own.
    """

    leaf_kind: ClassVar[CacheLeafKind] = CacheLeafKind.RECURRENT

    regions: tuple[RecurrentStateRegion, ...]
    """The state leaves one request occupies, in flatten order."""

    devices: Sequence[DeviceRef]
    """Devices to use for the cache."""

    data_parallel_degree: int = 1
    """Degree of data parallelism."""

    page_size: int = 0
    """Tokens a block covers, set only by a cache with no attention leaf.

    Zero otherwise: the attention leaves beside a state declare the pool, and
    no page covers zero tokens, so nothing real collides with it.
    """

    kv_connector_config: KVConnectorConfigInterface = field(
        default_factory=NullKVConnectorConfig
    )
    enable_prefix_caching: bool = False
    enable_dp_cross_replica_prefix_copy: bool = True
    kv_hash_algo: KVHashAlgo = "ahash64"
    kv_hash_seed: bytes | None = None
    speculative_method: SpeculativeMethod | None = None
    num_draft_tokens: int = 0
    """Read only from the leaf a tree takes its pool configuration off."""

    def __post_init__(self) -> None:
        if not self.regions:
            raise ValueError("RecurrentStateParams needs at least one region")
        leaf_ids = [region.leaf_id for region in self.regions]
        if len(set(leaf_ids)) != len(leaf_ids):
            raise ValueError(f"Region leaf ids must be unique, got: {leaf_ids}")
        self.regions = tuple(self.regions)

    @property
    def n_devices(self) -> int:
        return len(self.devices)

    @property
    def tensor_parallel_degree(self) -> int:
        return self.n_devices // self.data_parallel_degree

    @property
    def replicates_kv_across_tp(self) -> bool:
        """False: a state's rows hold one device's shard of the heads."""
        return False

    @cached_property
    def devices_per_replica(self) -> Sequence[Sequence[DeviceRef]]:
        return split_into_groups(self.devices, self.data_parallel_degree)

    @property
    def bytes_per_state(self) -> int:
        """Bytes one request's state occupies on one device, every layer."""
        return sum(region.bytes_per_page for region in self.regions)

    def slab_to_row_views(
        self, slabs: Sequence[Buffer]
    ) -> dict[str, list[Buffer]]:
        """Converts one replica's slabs into the rows its kernels index.

        Args:
            slabs: That replica's ``[num_huge_blocks, huge_page_bytes]``
                uint8 slab, one per device.

        Returns:
            One entry per leaf, holding a view per device in the order
            ``slabs`` came in.
        """
        views: dict[str, list[Buffer]] = {}
        for region in self.regions:
            row_bytes = region.row_elements * region.dtype.size_in_bytes
            views[region.leaf_id] = []
            for slab in slabs:
                num_rows, remainder = divmod(slab.num_elements, row_bytes)
                # A page is a whole number of rows, so the flat view a
                # kernel indexes is uniformly strided.
                assert remainder == 0, (
                    f"leaf {region.leaf_id!r} has {row_bytes} B rows, which"
                    f" do not tile a {slab.num_elements} B slab"
                )
                views[region.leaf_id].append(
                    slab.view(region.dtype, [num_rows, *region.row_shape])
                )
        return views

    def slab_to_bound_views(
        self, slabs: Sequence[Buffer]
    ) -> Mapping[str, list[Buffer]]:
        """Returns each leaf's rows, keyed where its layers read them."""
        rows = self.slab_to_row_views(slabs)
        return {
            region.pool_key: rows[region.leaf_id] for region in self.regions
        }

    @property
    def bytes_per_block(self) -> int:
        """Returns zero because a state's page is a per-request cost, not a
        cost per token."""
        return 0

    def allocate_buffers(
        self, total_num_pages: int, _prefix: str = ""
    ) -> list[KVCacheBufferInterface]:
        """Returns nothing: a state draws from a slab it does not allocate."""
        return []

    def slab_to_buffer_views(
        self,
        buffers: Sequence[Buffer],
        padded_page_bytes: Mapping[str, int] | None = None,
        _prefix: str = "",
    ) -> KVCacheBufferInterface:
        """Returns each leaf's pages, one state per page."""
        padded = padded_page_bytes or {}
        return RecurrentStateBuffer(
            pages={
                region.leaf_id: [
                    page_view(
                        slab,
                        (region.bytes_per_page,),
                        DType.uint8,
                        padded.get(region.leaf_id),
                    )
                    for slab in buffers
                ]
                for region in self.regions
            }
        )

    def get_symbolic_inputs(
        self, namespace: str = ""
    ) -> tuple[RecurrentStateInputsPerDevice[TensorType, BufferType], ...]:
        """Returns the symbolic inputs for the state leaves.

        ``namespace`` is unused: the region ids are already distinct.
        """

        def leaf(
            region: RecurrentStateRegion, device: DeviceRef, batch_dim: str
        ) -> RecurrentLeafInputs[TensorType, BufferType]:
            pool_shape: list[str | int] = [region.rows_dim]
            pool_shape.extend(region.row_shape)
            rows_shape: list[str | int] = [region.num_layers, batch_dim]
            return RecurrentLeafInputs(
                pool=BufferType(region.dtype, shape=pool_shape, device=device),
                live_row_ids=TensorType(
                    DType.uint32, shape=rows_shape, device=device
                ),
            )

        per_device: list[
            RecurrentStateInputsPerDevice[TensorType, BufferType]
        ] = []
        for replica_idx, devices in enumerate(self.devices_per_replica):
            batch_dim = f"replica_{replica_idx}_batch_size"
            for device in devices:
                per_device.append(
                    RecurrentStateInputsPerDevice(
                        leaves=tuple(
                            leaf(region, device, batch_dim)
                            for region in self.regions
                        ),
                    )
                )
        return tuple(per_device)

    @cached_property
    def _kv_symbolic_treedef(self) -> tree.TreeDef:
        # TODO(SERVOPT-1505): avoid flattening symbolic inputs only to retain TreeDef for unflatten.
        return tree.flatten(self.get_symbolic_inputs())[1]

    def unflatten_kv_inputs(
        self, it: Iterator[Any]
    ) -> tuple[RecurrentStateInputsPerDevice[TensorValue, BufferValue], ...]:
        return tuple(
            tree.leaves(
                tree.unflatten(self._kv_symbolic_treedef, it, exact=False),
                leaf=RecurrentStateInputsPerDevice,
            )
        )

    def build_runtime_inputs(
        self,
        assignments: Sequence[KVCacheAssignments],
        buffers: Sequence[KVCacheBufferInterface],
        _prefix: str = "",
    ) -> tuple[RecurrentStateInputsPerDevice[Buffer, Buffer], ...]:
        """Gathers this forward's state rows, replica-major.

        ``buffers`` is unused: a state's pool is staged in the assignment
        alongside the rows that address it.
        """
        inputs: list[RecurrentStateInputsPerDevice[Buffer, Buffer]] = []
        for replica_idx, assignment in enumerate(assignments):
            for staged in assignment.staged_by_device:
                leaves: list[RecurrentLeafInputs[Buffer, Buffer]] = []
                for region in self.regions:
                    missing = [
                        key
                        for key in (region.leaf_id, region.pool_key)
                        if key not in staged
                    ]
                    if missing:
                        raise ValueError(
                            f"Replica {replica_idx} staged no {missing} for"
                            f" state leaf {region.leaf_id!r}"
                        )
                    leaves.append(
                        RecurrentLeafInputs(
                            pool=staged[region.pool_key],
                            live_row_ids=staged[region.leaf_id],
                        )
                    )
                inputs.append(
                    RecurrentStateInputsPerDevice(leaves=tuple(leaves))
                )
        return tuple(inputs)

    def leaves(self, _prefix: str = "") -> Mapping[str, KVLeafRegion]:
        """Returns one pool leaf per state leaf, each one state wide.

        A region marked ``scratch`` lands in the scratch group rather than the
        recurrent one, so the same tree can declare a published state and the
        per-request scratch that rides beside it.

        ``_prefix`` is unused, so the pool key and the name a layer asks for
        stay the same string.
        """

        def leaf(region: RecurrentStateRegion) -> KVLeafRegion:
            cls = (
                ScratchKVLeafRegion if region.scratch else RecurrentKVLeafRegion
            )
            group_id = (
                KVCacheGroupId.scratch()
                if region.scratch
                else KVCacheGroupId.recurrent()
            )
            return cls(
                leaf_id=region.leaf_id,
                group_id=group_id,
                bytes_per_page=region.bytes_per_page,
                region=region,
            )

        return {region.leaf_id: leaf(region) for region in self.regions}


def recurrent_leaves(
    params: CacheLeafParamInterface,
) -> list[RecurrentStateParams]:
    """Returns every state a cache keeps, in tree order."""
    if isinstance(params, RecurrentStateParams):
        return [params]
    if isinstance(params, MultiKVCacheParams):
        return [
            state
            for child in params.children.values()
            for state in recurrent_leaves(child)
        ]
    return []


def recurrent_leaf(
    params: CacheLeafParamInterface,
) -> RecurrentStateParams | None:
    """Returns the first state a cache keeps, or ``None`` if it keeps none."""
    states = recurrent_leaves(params)
    return states[0] if states else None


def _is_attention(
    child: CacheLeafParamInterface,
) -> TypeGuard[KVCacheParamInterface]:
    """Returns whether an attention op reads this child of a cache tree."""
    return child.leaf_kind is CacheLeafKind.ATTENTION


def _agreed_pool(
    children: Mapping[str, CacheLeafParamInterface],
) -> CacheLeafParamInterface:
    """Returns a child the pool's configuration is read off.

    Every child that declares one must agree, so which is returned cannot
    matter.

    Raises:
        ValueError: If no child declares a pool, or if two disagree.
    """
    params = [child for child in children.values() if child.page_size]
    if not params:
        raise ValueError(
            "A cache tree takes its page size and pool configuration from a"
            " child that declares one, and none of these does."
        )
    first = params[0]
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
            f"All params must use the same number of devices, got: {devices}"
        )

    enable_prefix_caching = {p.enable_prefix_caching for p in params}
    if len(enable_prefix_caching) > 1:
        raise ValueError(
            "All params must use the same enable_prefix_caching, got:"
            f" {enable_prefix_caching}"
        )

    enable_dp_cross_replica_prefix_copy = {
        p.enable_dp_cross_replica_prefix_copy for p in params
    }
    if len(enable_dp_cross_replica_prefix_copy) > 1:
        raise ValueError(
            "All params must use the same"
            " enable_dp_cross_replica_prefix_copy, got:"
            f" {enable_dp_cross_replica_prefix_copy}"
        )

    # ``KVConnectorConfig`` is not hashable, so compare by equality against
    # the first rather than collapsing into a set.
    if any(p.kv_connector_config != first.kv_connector_config for p in params):
        raise ValueError(
            "All params must use the same kv_connector_config, got:"
            f" {[p.kv_connector_config for p in params]}"
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
            f"All params must use the same kv_hash_algo, got: {kv_hash_algos}"
        )

    kv_hash_seeds = {p.kv_hash_seed for p in params}
    if len(kv_hash_seeds) > 1:
        raise ValueError(
            f"All params must use the same kv_hash_seed, got: {kv_hash_seeds}"
        )
    return first


@dataclass(frozen=True)
class MultiKVCacheParams(KVCacheParamInterface):
    """Aggregates multiple cache parameter sets into a recursive tree.

    Children may be leaf :class:`KVCacheParams` instances or nested
    :class:`MultiKVCacheParams` subtrees, so arbitrarily deep hierarchies
    are supported (e.g. ``{target: {sliding, mla}, draft: mha}``). The
    whole tree is consumed through the :class:`KVCacheParamInterface` —
    callers never need to know how many blocks that is.

    A :class:`RecurrentStateParams` is a child like any other, but answers
    fewer questions, so attention-only aggregates run over
    ``_attention_children``.
    """

    children: dict[str, CacheLeafParamInterface]
    """Cache parameter sets to aggregate. Values may be leaf
    :class:`KVCacheParams` or :class:`RecurrentStateParams` instances, or
    nested :class:`MultiKVCacheParams` trees."""

    @property
    def leaf_kind(self) -> CacheLeafKind:
        """Attention where the subtree holds any, else recurrent."""
        return (
            CacheLeafKind.ATTENTION
            if self._attention_children
            else CacheLeafKind.RECURRENT
        )

    page_size: int
    """Number of tokens per page, a value every child cache must share."""
    data_parallel_degree: int
    """Degree of data parallelism, a value every child cache must share."""
    devices: Sequence[DeviceRef]
    """Devices to use for the KV caches."""
    kv_connector_config: KVConnectorConfigInterface
    """The KV connector's type and settings, a value every child must
    share."""
    enable_prefix_caching: bool = False
    """Whether prefix caching is enabled, a value every child must share."""
    enable_dp_cross_replica_prefix_copy: bool = True
    """Whether a DP cross-replica prefix copy may serve a hit."""
    kv_hash_algo: KVHashAlgo = "ahash64"
    """Hash algorithm used for block identity."""
    kv_hash_seed: bytes | None = None
    """Resolved cluster seed for ``sha256``/``sha256_64``."""
    speculative_method: SpeculativeMethod | None = None
    """Speculative decoding method propagated from ``SpeculativeConfig``."""
    num_draft_tokens: int = 0
    """Total draft tokens generated per speculative iteration."""

    @classmethod
    def from_params(
        cls,
        params: Mapping[str, CacheLeafParamInterface],
    ) -> MultiKVCacheParams:
        """Creates a :class:`MultiKVCacheParams` from one or more param sets.

        Children may be leaf :class:`KVCacheParams` instances, one
        :class:`RecurrentStateParams`, or nested
        :class:`MultiKVCacheParams` trees, enabling arbitrarily deep KV
        cache hierarchies (e.g. ``{target: {sliding, mla}, draft: mha}``).
        All children must share the same ``page_size``,
        ``data_parallel_degree``, ``n_devices``, and
        ``kv_connector_config`` values.

        Args:
            params: Named mapping of :class:`CacheLeafParamInterface`
                instances to aggregate. At least one must be a cache an
                attention op reads, since the pool's configuration is read
                off one.

        Returns:
            A new :class:`MultiKVCacheParams` aggregating all provided params.

        Raises:
            ValueError: If no params are provided, or if none of them is a
                cache an attention op reads.
        """
        if len(params) == 0:
            raise ValueError("MultiKVCacheParams requires at least one param.")
        first = _agreed_pool(params)
        return cls(
            children=dict(params),
            page_size=first.page_size,
            data_parallel_degree=first.data_parallel_degree,
            devices=first.devices,
            kv_connector_config=first.kv_connector_config,
            enable_prefix_caching=first.enable_prefix_caching,
            enable_dp_cross_replica_prefix_copy=(
                first.enable_dp_cross_replica_prefix_copy
            ),
            kv_hash_algo=first.kv_hash_algo,
            kv_hash_seed=first.kv_hash_seed,
            speculative_method=first.speculative_method,
            num_draft_tokens=first.num_draft_tokens,
        )

    def __post_init__(self) -> None:
        """Validates that all params have consistent page size."""
        if not self.children:
            raise ValueError(
                "MultiKVCacheParams requires at least one param set."
            )

        # A leaf id keys the pool's leaves and names the symbolic dim sizing
        # its rows, so a duplicate collides in both.
        seen: dict[str, str] = {}
        for key, child in self.children.items():
            for state in recurrent_leaves(child):
                for region in state.regions:
                    clash = seen.setdefault(region.leaf_id, key)
                    if clash != key:
                        raise ValueError(
                            f"Recurrent states {clash!r} and {key!r} both"
                            f" declare leaf {region.leaf_id!r}; give each"
                            " state its own leaf ids."
                        )

    @cached_property
    def _attention_children(self) -> dict[str, KVCacheParamInterface]:
        """The children an attention op reads."""
        return {
            key: child
            for key, child in self.children.items()
            if _is_attention(child)
        }

    @property
    def n_devices(self) -> int:
        """Returns the number of devices."""
        return len(self.devices)

    @property
    def bytes_per_block(self) -> int:
        """Total bytes per block across all KV caches.

        Since all caches allocate memory for the same sequence, the total
        memory cost per block is the sum across all param sets.
        """
        return sum(p.bytes_per_block for p in self.children.values())

    def get_symbolic_inputs(
        self, namespace: str = ""
    ) -> dict[str, KVCacheInputs[TensorType, BufferType]]:
        """Returns the symbolic inputs for the KV cache tree.

        Each child inherits a distinct namespace so sibling groups' page-pool
        dims stay independent; nested subtrees compose the prefix.
        """
        # OrderedDict: the graph declares its inputs in child-declaration order
        # (children, then state), and tree.flatten preserves an OrderedDict's
        # order rather than sorting keys like a plain dict.
        return OrderedDict(
            (k, p.get_symbolic_inputs(namespace=f"{namespace}{k}_"))
            for k, p in self.children.items()
        )

    @cached_property
    def _kv_symbolic_treedef(self) -> tree.TreeDef:
        # TODO(SERVOPT-1505): avoid flattening symbolic inputs only to retain TreeDef for unflatten.
        return tree.flatten(self.get_symbolic_inputs())[1]

    def unflatten_kv_inputs(
        self, it: Iterator[Any]
    ) -> dict[str, KVCacheInputs[TensorValue, BufferValue]]:
        """Unflattens the KV cache inputs from a graph-input iterator."""
        inputs = tree.unflatten(self._kv_symbolic_treedef, it, exact=False)
        assert isinstance(inputs, dict)
        return inputs

    def unflatten_basic_kv_tree(
        self, it: Iterator[Any]
    ) -> tuple[list[KVCacheInputsPerDevice[TensorValue, BufferValue]], ...]:
        """Unflattens a basic KV tree from a graph-input iterator.

        Requires that the model is a basic height-1 tree. This method does not work
        on nested trees.

        Returns one entry per attention child, in declaration order.
        """
        kv_tree = self.unflatten_kv_inputs(it)
        out: list[list[KVCacheInputsPerDevice[TensorValue, BufferValue]]] = []
        for key in self._attention_children:
            child = kv_tree[key]
            # A nested (height > 1) child unflattens to a dict subtree, not the
            # flat per-device tuple this shortcut requires.
            if not isinstance(child, tuple):
                raise ValueError("Unable to flatten nested KV tree")
            out.append(tree.leaves(child, leaf=KVCacheInputsPerDevice))
        return tuple(out)

    @property
    def replicates_kv_across_tp(self) -> bool:
        """Whether every device holds identical KV state.

        A leaf's own answer, not the tree's: an MHA leaf shards its heads
        where an MLA one replicates its latent, so siblings can differ.
        """
        attention = list(self._attention_children.values())
        if attention:
            return attention[0].replicates_kv_across_tp
        states = recurrent_leaves(self)
        assert states, "a cache tree holds attention leaves or state leaves"
        return states[0].replicates_kv_across_tp

    @property
    def tensor_parallel_degree(self) -> int:
        """Returns the tensor parallel degree."""
        return self.n_devices // self.data_parallel_degree

    def resolve_attn_key(
        self,
        batch_size: int,
        max_prompt_length: int,
        max_cache_valid_length: int,
    ) -> AttnKeyInterface:
        """Resolves the dispatch shape tree mirroring the attention caches."""
        return MultiAttnKey.from_dict(
            {
                k: p.resolve_attn_key(
                    batch_size, max_prompt_length, max_cache_valid_length
                )
                for k, p in self._attention_children.items()
            }
        )

    def graph_capture_probe_cache_lengths(
        self, max_cache_length: int, q_max_seq_len: int = 1
    ) -> list[int]:
        """Returns the union of probe cache lengths across all child caches."""
        lengths: set[int] = set()
        for p in self._attention_children.values():
            lengths.update(
                p.graph_capture_probe_cache_lengths(
                    max_cache_length, q_max_seq_len
                )
            )
        return sorted(lengths)

    def allocate_buffers(
        self, total_num_pages: int, _prefix: str = ""
    ) -> list[KVCacheBufferInterface]:
        """Allocates per-replica buffers for every cache in the tree.

        Returns one :class:`MultiKVCacheBuffer` per data-parallel replica,
        each holding that replica's :class:`KVCacheBuffer` for every child
        that allocates one.
        """
        per_key = {
            k: p.allocate_buffers(total_num_pages, _prefix + k + ".")
            for k, p in self.children.items()
        }
        return [
            MultiKVCacheBuffer(
                children={
                    k: buffers[replica_idx]
                    for k, buffers in per_key.items()
                    if buffers
                }
            )
            for replica_idx in range(self.data_parallel_degree)
        ]

    def build_runtime_inputs(
        self,
        assignments: Sequence[KVCacheAssignments],
        buffers: Sequence[KVCacheBufferInterface],
        _prefix: str = "",
    ) -> KVCacheInputs[Buffer, Buffer]:
        """Builds the runtime KV-cache tree spanning all replicas.

        Each child builds itself from every replica's assignment plus that
        replica's child buffer, if it allocated one; the per-replica
        assignment (cache lengths / lookup table / dispatch shape / state
        rows) is shared across child caches since they all map the same
        sequence. The tree comes out in the order the graph declared its
        inputs.
        """
        multi_buffers: list[MultiKVCacheBuffer] = []
        for buffer in buffers:
            assert isinstance(buffer, MultiKVCacheBuffer)
            multi_buffers.append(buffer)
        # OrderedDict so the runtime tree flattens in the same
        # child-declaration order the graph declared its inputs (see
        # get_symbolic_inputs).
        return OrderedDict(
            (
                k,
                p.build_runtime_inputs(
                    assignments,
                    [b.children[k] for b in multi_buffers if k in b.children],
                    _prefix=_prefix + k + ".",
                ),
            )
            for k, p in self.children.items()
        )

    def leaves(self, _prefix: str = "") -> Mapping[str, KVLeafRegion]:
        """Returns the leaves of every child, prefixed by the child's name."""
        contributions = [
            v.leaves(_prefix + k + ".") for k, v in self.children.items()
        ]

        leaves: dict[str, KVLeafRegion] = {}
        for contribution in contributions:
            for leaf_id, leaf in contribution.items():
                if leaf_id in leaves:
                    raise ValueError(f"Duplicate cache leaf {leaf_id!r}")
                leaves[leaf_id] = leaf
        return leaves

    def slab_to_bound_views(
        self, slabs: Sequence[Buffer]
    ) -> Mapping[str, list[Buffer]]:
        """Returns whatever the children bind, in one mapping.

        Keys come from the leaves, which are unique across the tree.
        """
        bound: dict[str, list[Buffer]] = {}
        for child in self.children.values():
            bound.update(child.slab_to_bound_views(slabs))
        return bound

    def slab_to_buffer_views(
        self,
        buffers: Sequence[Buffer],
        padded_page_bytes: Mapping[str, int] | None = None,
        _prefix: str = "",
    ) -> KVCacheBufferInterface:
        """Converts a slab of memory into every child's view of it."""
        return MultiKVCacheBuffer(
            children={
                child_id: child.slab_to_buffer_views(
                    buffers, padded_page_bytes, _prefix + child_id + "."
                )
                for child_id, child in self.children.items()
            },
        )


def spec_decode_cache_slack(params: KVCacheParamInterface) -> int:
    """Computes the extra KV positions a request may occupy past ``max_seq_len``.

    A speculative-decode step can over-speculate past the per-request
    ``max_seq_len`` cap into this slack (the KV pool reserves it beyond
    ``max_seq_len``), so any per-request sizing derived from ``max_seq_len``
    -- the pool's page budget, the sparse-indexer score scratch, etc. -- must
    add it. Centralized here so ``_compute_seq_len``, pool sizing, and
    ``OverlapTextGenerationPipeline._effective_max_cache_length`` stay in sync.

    Args:
        params: The KV cache parameters. The speculative-decoding fields
            (``num_draft_tokens`` and ``num_draft_tokens_per_step``) determine
            the slack.

    Returns:
        The number of extra KV positions to reserve past ``max_seq_len``, or
        ``0`` when speculative decoding is off.
    """
    if params.num_draft_tokens <= 0:
        return 0
    # Worst case matching ``_compute_seq_len``: drafts verified and written next
    # batch (2x), the prior overlap batch's drafts assumed accepted (1x), the
    # DFlash block-draft slot, and the FUTURE_TOKEN placeholder.
    block_draft_extra = (
        1 if params.num_draft_tokens_per_step == params.num_draft_tokens else 0
    )
    return 3 * params.num_draft_tokens + block_draft_extra + 1


def compute_num_device_blocks(
    params: KVCacheParamInterface,
    available_cache_memory: int,
    max_batch_size: int | None,
    max_seq_len: int | None,
    require_max_seq_len_fits: bool = False,
    include_null_block: bool = False,
) -> int:
    """Computes the number of blocks that can be allocated based on the available cache memory.

    The number of blocks returned is for a single replica. Each replica will
    have the same number of blocks.

    Args:
        available_cache_memory: The amount of cache memory available across all devices.
        max_batch_size: The maximum batch size, or None.
        max_seq_len: The maximum sequence length, or None.
        require_max_seq_len_fits: When True, raise if a single request at
            ``max_seq_len`` cannot fit in the allocable device blocks. Set
            only when allocating a uniform pool; memory estimation probes
            oversized configs on purpose.
        include_null_block: Whether to include room for the null block.

    Returns:
        The number of blocks that can be allocated for a single replica.
    """
    # Compute upper bound of total number of pages required. A speculative
    # step can grow a request past max_seq_len into the pool's draft-token
    # slack, so budget the per-request pages on that same bound; otherwise the
    # pool caps short of what the scheduler is allowed to reserve.
    max_blocks_per_req: int | None = None
    max_total_blocks: int | None = None
    if max_seq_len is not None and max_batch_size is not None:
        max_seq_len_with_slack = max_seq_len + spec_decode_cache_slack(params)
        max_blocks_per_req = math.ceil(
            max_seq_len_with_slack / params.page_size
        )
        max_total_blocks = max_blocks_per_req * max_batch_size
        if include_null_block:
            max_total_blocks += 1

    if params.bytes_per_block == 0:
        # Nothing here costs per token, so memory does not divide into equal
        # blocks. Every leaf plateaus instead -- a state keeps its live block
        # and one checkpoint however long a request runs.
        slots = ceildiv(max_seq_len, params.page_size) if max_seq_len else 1
        per_request = max(
            (
                leaf.blocks_to_reserve(slots)
                for leaf in params.leaves().values()
            ),
            default=0,
        )
        blocks = per_request * (max_batch_size or 1)
        return blocks + 1 if include_null_block else blocks

    # Compute total number of blocks allocatable based on available memory.
    available_cache_memory_per_replica = (
        available_cache_memory // params.data_parallel_degree
    )
    # ``bytes_per_block`` excludes row-addressed leaves, so reserve them first.
    row_bytes = params.per_request_row_bytes(
        max_blocks_per_req if max_blocks_per_req is not None else 1
    ) * (max_batch_size or 0)
    num_allocable_blocks = (
        max(0, available_cache_memory_per_replica - row_bytes)
        // params.bytes_per_block
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

    # A page of every leaf per slot is the uniform pool's cost model, so only
    # its allocator may judge whether a request fits.
    if (
        require_max_seq_len_fits
        and max_blocks_per_req is not None
        and max_blocks_per_req > num_allocable_blocks
    ):
        memory_needed_str = to_human_readable_bytes(
            max_blocks_per_req * params.bytes_per_block
        )
        slack = spec_decode_cache_slack(params)
        slack_str = (
            f" (plus {slack} speculative-decode slack tokens)"
            if slack > 0
            else ""
        )
        raise RuntimeError(
            "Insufficient cache memory to support a batch containing one"
            f" request at the max sequence length of {max_seq_len} tokens"
            f"{slack_str}. Need to allocate at least {max_blocks_per_req} pages"
            f" ({memory_needed_str}), but only have enough memory for"
            f" {num_allocable_blocks} pages"
            f" ({cache_memory_str}{across_x_devices_str})."
            " A request approaching the max sequence length would"
            " exhaust the KV cache and crash the model worker. Reduce"
            " --max-length to at most"
            f" {num_allocable_blocks * params.page_size} or increase the"
            " available KV cache memory (e.g. raise"
            " --device-memory-utilization)."
        )

    return num_blocks


def estimated_memory_size(
    params: KVCacheParamInterface,
    available_cache_memory: int,
    max_batch_size: int,
    max_seq_len: int,
    include_null_block: bool = False,
) -> int:
    """Computes the estimated memory size of the KV cache used by all replicas.

    Args:
        available_cache_memory: The amount of cache memory available across all devices.
        max_batch_size: The maximum batch size.
        max_seq_len: The maximum sequence length.
        include_null_block: Whether to include room for the null block.

    Returns:
        The estimated memory usage of the KV cache in bytes.
    """
    num_device_blocks = compute_num_device_blocks(
        available_cache_memory=available_cache_memory,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        params=params,
        include_null_block=include_null_block,
    )
    bytes_per_block = params.bytes_per_block or (
        sum(leaf.bytes_per_page for leaf in params.leaves().values())
        * params.tensor_parallel_degree
    )
    # The block count excludes row-addressed leaves, except under the
    # zero-price fallback above.
    row_bytes = (
        params.per_request_row_bytes(ceildiv(max_seq_len, params.page_size))
        * max_batch_size
        if params.bytes_per_block
        else 0
    )
    return (
        num_device_blocks * bytes_per_block + row_bytes
    ) * params.data_parallel_degree


def compute_max_seq_len_fitting_in_cache(
    params: KVCacheParamInterface,
    available_cache_memory: int,
    include_null_block: bool = False,
) -> int | None:
    """Computes the maximum sequence length that can fit in the available memory.

    Args:
        available_cache_memory: The amount of cache memory available across
            all devices.
        include_null_block: Whether to include room for the null block.

    Returns:
        The maximum sequence length that fits, or None where no length
        exhausts the cache. A recurrent state costs the same however long a
        request runs, which is what the manager reports once it exists.
    """
    if params.bytes_per_block == 0:
        return None
    num_blocks = compute_num_device_blocks(
        params=params,
        available_cache_memory=available_cache_memory,
        max_batch_size=1,
        # Do not limit the sequence length.
        max_seq_len=None,
        include_null_block=include_null_block,
    )
    # Reserve the speculative-decode slack a request may occupy past its
    # advertised max_seq_len (see spec_decode_cache_slack). Without this the
    # auto-derived cap would equal the whole pool, so _effective_max_cache_length
    # (min(max_seq_len + slack, pool)) collapses back to max_seq_len and
    # over-speculation is silently disabled near the top of context. No-op when
    # speculative decoding is off (slack == 0).
    max_seq_len = num_blocks * params.page_size - spec_decode_cache_slack(
        params
    )
    return max(1, max_seq_len)
