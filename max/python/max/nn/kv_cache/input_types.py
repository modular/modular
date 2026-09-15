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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeAlias, TypeVar

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.experimental.tensor import Tensor
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    TensorType,
    TensorValue,
    ops,
)
from max.tree import Tree
from typing_extensions import Self

PACKED_PAGE_STRIDE = -1
"""``page_stride`` sentinel: the pages are packed, so the distance from one page
to the next is the product of the dimensions inside a page."""

_Tensor = TypeVar("_Tensor", TensorValue, TensorType, Buffer, Tensor)
_Buffer = TypeVar("_Buffer", BufferValue, BufferType, Buffer, Tensor)


def _verify_rank1_int64_tensor(name: str, t: _Tensor | None) -> None:
    if t is None:
        return
    if t.dtype != DType.int64:
        raise ValueError(
            f"Expected dtype int64, got {t.dtype} for tensor {name}"
        )
    if t.rank != 1:
        raise ValueError(f"Expected rank 1, got {t.rank} for tensor {t}")


@dataclass(frozen=True)
class _KVCacheInputsPerDeviceMeta:
    """Metadata for trees over :class:`KVCacheInputsPerDevice`."""

    emit_kv_scales: bool
    emit_attention_dispatch_metadata: bool
    emit_draft_attention_dispatch_metadata: bool
    emit_mla_num_partitions: bool
    emit_draft_mla_num_partitions: bool
    num_kv_blocks_per_layer: int
    num_kv_scales_per_layer: int


@dataclass
class KVCacheInputsPerDevice(Generic[_Tensor, _Buffer]):
    """Symbolic graph input types for a single device's paged KV cache."""

    kv_blocks: _Buffer
    """The device's paged KV cache blocks."""
    cache_lengths: _Tensor
    """Per-request cache lengths, one rank-1 entry per request."""
    lookup_table: _Tensor
    """Per-request page lookup table, each row holding the block ids its
    request reads."""
    max_prompt_length: _Tensor
    """The batch's maximum prompt length, as a scalar tensor."""
    max_cache_length: _Tensor
    """The batch's maximum cache length, as a scalar tensor."""
    kv_scales: _Buffer | None = None
    """KV scales for FP8 quantization."""
    page_stride_input: _Tensor | None = None
    """Page-to-page distance for ``kv_blocks``, as a rank-1 int64 tensor.

    ``None`` means packed; read it through :meth:`values_page_stride`.
    """
    scales_page_stride_input: _Tensor | None = None
    """Page-to-page distance for ``kv_scales``; ``None`` means packed."""
    scales_lookup_table: _Tensor | None = None
    """Page lookup table for ``kv_scales``, present when the scales are paged
    independently of the values so a request's scale pages carry their own
    ids. ``None`` means the two share one block-id space and ``lookup_table``
    resolves both, which is what every non-pooled cache does."""
    attention_dispatch_metadata: _Tensor | None = None
    """The device's attention dispatch metadata, as a rank-1 int64 tensor."""
    draft_attention_dispatch_metadata: _Tensor | None = None
    """The draft cache's attention dispatch metadata, as a rank-1 int64
    tensor."""
    mla_num_partitions: _Tensor | None = None
    """Capturable-graph scalar the SM100 MLA dispatcher uses to align
    grid-time partition decisions with the kernel's divmod. Populated only
    for MLA paths; ``None`` otherwise."""
    draft_mla_num_partitions: _Tensor | None = None
    """The draft cache's analog of :attr:`mla_num_partitions`."""
    kv_blocks_per_layer: list[_Buffer] | None = None
    """One single-layer KV buffer per layer, used when the backing pool
    allocates a standalone buffer per layer
    (``KVCacheParams.per_layer_buffers``) instead of one multi-layer buffer.
    ``kv_blocks`` aliases ``kv_blocks_per_layer[0]`` so single-buffer
    consumers stay valid; a per-layer attention dispatch picks
    ``kv_blocks_per_layer[layer_idx]``. ``None`` (the default) for every
    non-per-layer cache."""
    kv_scales_per_layer: list[_Buffer] | None = None
    """One single-layer scale buffer per layer, the quantized-scale analog of
    ``kv_blocks_per_layer`` (used with ``per_layer_buffers`` + a quantized
    KV cache). ``kv_scales`` aliases ``kv_scales_per_layer[0]``; a per-layer
    attention dispatch picks ``kv_scales_per_layer[layer_idx]``. ``None``
    for every non-per-layer / unquantized cache."""

    def __post_init__(self) -> None:
        _verify_rank1_int64_tensor(
            "attention_dispatch_metadata", self.attention_dispatch_metadata
        )
        _verify_rank1_int64_tensor(
            "draft_attention_dispatch_metadata",
            self.draft_attention_dispatch_metadata,
        )
        _verify_rank1_int64_tensor(
            "mla_num_partitions", self.mla_num_partitions
        )
        _verify_rank1_int64_tensor(
            "draft_mla_num_partitions", self.draft_mla_num_partitions
        )
        if self.scales_lookup_table is not None and self.kv_scales is None:
            raise ValueError(
                "kv_scales must be provided when scales_lookup_table is provided"
            )

    def _packed_page_stride(self) -> Any:
        """Returns the packed sentinel, typed like the rest of this collection.

        ``flatten`` serves the symbolic, graph and runtime paths, so the
        sentinel has to be a type, a graph value or a buffer to match.
        """
        if isinstance(self.kv_blocks, Buffer):
            return Buffer.from_numpy(
                np.array([PACKED_PAGE_STRIDE], dtype=np.int64)
            )
        if isinstance(self.kv_blocks, BufferType):
            return TensorType(DType.int64, shape=[1], device=DeviceRef.CPU())
        return ops.constant(
            [PACKED_PAGE_STRIDE], DType.int64, device=DeviceRef.CPU()
        )

    def values_page_stride(self) -> Any:
        """Returns the ``page_stride`` operand for ``kv_blocks``."""
        if self.page_stride_input is not None:
            return self.page_stride_input
        return self._packed_page_stride()

    def scales_page_stride(self) -> Any:
        """Returns the ``page_stride`` operand for ``kv_scales``."""
        if self.scales_page_stride_input is not None:
            return self.scales_page_stride_input
        return self._packed_page_stride()

    def __tree_flatten__(
        self,
    ) -> tuple[tuple[Any, ...], _KVCacheInputsPerDeviceMeta]:
        parts: list[Any] = [
            self.kv_blocks,
            # Each stride follows the buffer it describes.
            self.values_page_stride(),
            self.cache_lengths,
            self.lookup_table,
            self.max_prompt_length,
            self.max_cache_length,
        ]
        emit_kv_scales = bool(self.kv_scales)
        if emit_kv_scales:
            assert self.kv_scales is not None
            parts.append(self.kv_scales)
            # The scales stride follows the scales buffer it describes.
            parts.append(self.scales_page_stride())
            parts.append(self.scales_lookup_table or self.lookup_table)
        if self.attention_dispatch_metadata is not None:
            parts.append(self.attention_dispatch_metadata)
        if self.draft_attention_dispatch_metadata is not None:
            parts.append(self.draft_attention_dispatch_metadata)
        if self.mla_num_partitions is not None:
            parts.append(self.mla_num_partitions)
        if self.draft_mla_num_partitions is not None:
            parts.append(self.draft_mla_num_partitions)
        parts.extend(self.kv_blocks_per_layer or ())
        parts.extend(self.kv_scales_per_layer or ())
        meta = _KVCacheInputsPerDeviceMeta(
            emit_kv_scales=emit_kv_scales,
            emit_attention_dispatch_metadata=self.attention_dispatch_metadata
            is not None,
            emit_draft_attention_dispatch_metadata=self.draft_attention_dispatch_metadata
            is not None,
            emit_mla_num_partitions=self.mla_num_partitions is not None,
            emit_draft_mla_num_partitions=self.draft_mla_num_partitions
            is not None,
            num_kv_blocks_per_layer=len(self.kv_blocks_per_layer or ()),
            num_kv_scales_per_layer=len(self.kv_scales_per_layer or ()),
        )
        return (tuple(parts), meta)

    @classmethod
    def __tree_unflatten__(
        cls,
        meta: _KVCacheInputsPerDeviceMeta,
        children: tuple[Any, ...],
    ) -> Self:
        it = iter(children)
        kv_blocks = next(it)
        page_stride_input = next(it)
        cache_lengths = next(it)
        lookup_table = next(it)
        max_prompt_length = next(it)
        max_cache_length = next(it)
        kv_scales = next(it) if meta.emit_kv_scales else None
        scales_page_stride_input = next(it) if meta.emit_kv_scales else None
        scales_lookup_table = next(it) if meta.emit_kv_scales else None
        attention_dispatch_metadata = (
            next(it) if meta.emit_attention_dispatch_metadata else None
        )
        draft_attention_dispatch_metadata = (
            next(it) if meta.emit_draft_attention_dispatch_metadata else None
        )
        mla_num_partitions = next(it) if meta.emit_mla_num_partitions else None
        draft_mla_num_partitions = (
            next(it) if meta.emit_draft_mla_num_partitions else None
        )
        kv_blocks_per_layer = (
            [next(it) for _ in range(meta.num_kv_blocks_per_layer)]
            if meta.num_kv_blocks_per_layer
            else None
        )
        kv_scales_per_layer = (
            [next(it) for _ in range(meta.num_kv_scales_per_layer)]
            if meta.num_kv_scales_per_layer
            else None
        )
        return cls(
            kv_blocks=kv_blocks,
            page_stride_input=page_stride_input,
            cache_lengths=cache_lengths,
            lookup_table=lookup_table,
            max_prompt_length=max_prompt_length,
            max_cache_length=max_cache_length,
            kv_scales=kv_scales,
            scales_page_stride_input=scales_page_stride_input,
            scales_lookup_table=scales_lookup_table,
            attention_dispatch_metadata=attention_dispatch_metadata,
            draft_attention_dispatch_metadata=draft_attention_dispatch_metadata,
            mla_num_partitions=mla_num_partitions,
            draft_mla_num_partitions=draft_mla_num_partitions,
            kv_blocks_per_layer=kv_blocks_per_layer,
            kv_scales_per_layer=kv_scales_per_layer,
        )

    # TODO: FIX THIS HACK!!!
    def flatten_without_attention_dispatch_metadata(
        self,
    ) -> list[_Tensor | _Buffer]:
        """Serializes fields into a flat list, minus the attention dispatch
        metadata fields."""
        return [
            self.kv_blocks,
            self.values_page_stride(),
            self.cache_lengths,
            self.lookup_table,
            self.max_prompt_length,
            self.max_cache_length,
            *((self.kv_scales,) if self.kv_scales else ()),
            *((self.scales_page_stride(),) if self.kv_scales else ()),
            *(
                (self.scales_lookup_table or self.lookup_table,)
                if self.kv_scales
                else ()
            ),
            # Tail per-layer buffers (see ``flatten``). Attention dispatch clears
            # this field before calling an op, so this is ``()`` at op sites.
            *(self.kv_blocks_per_layer or ()),
            *(self.kv_scales_per_layer or ()),
        ]


PagedCacheValues = KVCacheInputsPerDevice[TensorValue, BufferValue]


# ===--------------------------------------------------------------------=== #
# Recurrent state
# ===--------------------------------------------------------------------=== #


@dataclass(frozen=True)
class RecurrentLeafInputs(Generic[_Tensor, _Buffer]):
    """One state leaf's graph inputs on one device."""

    pool: _Buffer
    live_row_ids: _Tensor

    def live_row_id(self, layer: int) -> TensorValue:
        """Returns the ``[batch_size]`` pool row this layer runs in."""
        return _layer_row_ids(self.live_row_ids, layer)

    def __tree_flatten__(
        self,
    ) -> tuple[tuple[_Buffer, _Tensor], None]:
        return (self.pool, self.live_row_ids), None

    @classmethod
    def __tree_unflatten__(
        cls,
        meta: Any,
        children: Sequence[Any],
        /,
    ) -> RecurrentLeafInputs[Any, Any]:
        pool, live_row_ids = children
        return cls(pool=pool, live_row_ids=live_row_ids)


def _layer_row_ids(ids: Any, layer: int) -> TensorValue:
    """Returns one layer's column of a ``[batch_size, num_layers]`` id tensor."""
    assert isinstance(ids, TensorValue), (
        "per-layer row ids can only be taken from a graph value, not from "
        f"{type(ids).__name__}"
    )
    return ids[:, layer]


@dataclass
class RecurrentStateInputsPerDevice(Generic[_Tensor, _Buffer]):
    """One device's recurrent-state leaves."""

    leaves: tuple[RecurrentLeafInputs[_Tensor, _Buffer], ...]
    """In the order the regions were declared."""

    def __tree_flatten__(
        self,
    ) -> tuple[list[RecurrentLeafInputs[Any, Any]], None]:
        return list(self.leaves), None

    @classmethod
    def __tree_unflatten__(
        cls,
        meta: Any,
        children: Sequence[Any],
        /,
    ) -> RecurrentStateInputsPerDevice[Any, Any]:
        return cls(leaves=tuple(children))


KVCacheInputs: TypeAlias = Tree[
    KVCacheInputsPerDevice[_Tensor, _Buffer]
    | RecurrentStateInputsPerDevice[_Tensor, _Buffer]
]
