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

from typing import Any, Generic, TypeAlias, TypeVar

from max import tree
from max.driver import Buffer
from max.dtype import DType
from max.experimental.tensor import Tensor
from max.graph import (
    BufferType,
    BufferValue,
    TensorType,
    TensorValue,
)
from max.tree import Tree

_Tensor = TypeVar("_Tensor", TensorValue, TensorType, Buffer, Tensor)
_Buffer = TypeVar("_Buffer", BufferValue, BufferType, Buffer, Tensor)


@tree.dataclass
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
    page_stride: _Tensor
    """Page-to-page distance for ``kv_blocks``, as a rank-1 int64 tensor."""
    kv_scales: _Buffer | None = None
    """KV scales for FP8 quantization."""
    scales_page_stride: _Tensor | None = None
    """The same as ``page_stride``, for ``kv_scales``."""
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
        def verify_rank1_int64_tensor(name: str, t: _Tensor | None) -> None:
            if t is None:
                return
            if t.dtype != DType.int64:
                raise ValueError(
                    f"Expected dtype int64, got {t.dtype} for tensor {name}"
                )
            if t.rank != 1:
                raise ValueError(
                    f"Expected rank 1, got {t.rank} for tensor {t}"
                )

        verify_rank1_int64_tensor(
            "attention_dispatch_metadata", self.attention_dispatch_metadata
        )
        verify_rank1_int64_tensor(
            "draft_attention_dispatch_metadata",
            self.draft_attention_dispatch_metadata,
        )
        verify_rank1_int64_tensor("mla_num_partitions", self.mla_num_partitions)
        verify_rank1_int64_tensor(
            "draft_mla_num_partitions", self.draft_mla_num_partitions
        )

        if not (
            (self.kv_scales is None)
            == (self.scales_page_stride is None)
            == (self.scales_lookup_table is None)
        ):
            raise ValueError(
                "kv_scales, scales_page_stride and scales_lookup_table must "
                "be provided together"
            )

    # TODO: FIX THIS HACK!!!
    def flatten_without_attention_dispatch_metadata(
        self,
    ) -> list[_Tensor | _Buffer]:
        """Serializes fields into a flat list, minus the attention dispatch
        metadata fields."""
        scales: tuple[_Tensor | _Buffer, ...] = ()
        if self.kv_scales is not None:
            assert (
                self.scales_page_stride is not None
                and self.scales_lookup_table is not None
            )
            scales = (
                self.kv_scales,
                self.scales_page_stride,
                self.scales_lookup_table,
            )
        return [
            self.kv_blocks,
            self.page_stride,
            self.cache_lengths,
            self.lookup_table,
            self.max_prompt_length,
            self.max_cache_length,
            *scales,
            # Tail per-layer buffers (see ``flatten``). Attention dispatch clears
            # this field before calling an op, so this is ``()`` at op sites.
            *(self.kv_blocks_per_layer or ()),
            *(self.kv_scales_per_layer or ()),
        ]


PagedCacheValues = KVCacheInputsPerDevice[TensorValue, BufferValue]


# ===--------------------------------------------------------------------=== #
# Recurrent state
# ===--------------------------------------------------------------------=== #


@tree.dataclass(frozen=True)
class RecurrentLeafInputs(Generic[_Tensor, _Buffer]):
    """One state leaf's graph inputs on one device."""

    pool: _Buffer
    live_row_ids: _Tensor

    def live_row_id(self, layer: int) -> TensorValue:
        """Returns the ``[batch_size]`` pool row this layer runs in."""
        return _layer_row_ids(self.live_row_ids, layer)


def _layer_row_ids(ids: Any, layer: int) -> TensorValue:
    """Returns one layer's column of a ``[batch_size, num_layers]`` id tensor."""
    assert isinstance(ids, TensorValue), (
        "per-layer row ids can only be taken from a graph value, not from "
        f"{type(ids).__name__}"
    )
    return ids[:, layer]


@tree.dataclass
class RecurrentStateInputsPerDevice(Generic[_Tensor, _Buffer]):
    """One device's recurrent-state leaves."""

    leaves: tuple[RecurrentLeafInputs[_Tensor, _Buffer], ...]
    """In the order the regions were declared."""


KVCacheInputs: TypeAlias = Tree[
    KVCacheInputsPerDevice[_Tensor, _Buffer]
    | RecurrentStateInputsPerDevice[_Tensor, _Buffer]
]
