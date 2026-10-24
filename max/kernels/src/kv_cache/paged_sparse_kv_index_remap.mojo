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
# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Map logical sequence token indices to MLA sparse ``physical row`` encoding.

Sparse MLA kernels expect each selected key position as::

    Int32(physical_block_id * page_size + token_offset_within_page)

where ``physical_block_id`` comes from the paged ``lookup_table``. The indexer
instead emits logical positions ``t`` in ``[0, cache_length)``. This module
implements that remapping on GPU (or CPU) without device↔host staging of the
full sparse index or LUT tensors.

Invalid sparse slots conventionally use ``-1`` and are copied through.
If the LUT entry is ``>= invalid_block_id`` (runtime sentinel ``total_num_pages``),
the output slot is written ``-1``.
"""

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.math import ceildiv
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_cpu

from extensibility import InputTensor
from extensibility import (
    _MutableInputTensor as MutableInputTensor,
)


@inline(.always)
def _remap_one(
    log_t: Int32,
    batch_u32: UInt32,
    lut: ImmPointer[UInt32, _],
    lut_cols: Int,
    lut_rows: Int,
    page_size: Int,
    invalid_block_id: UInt32,
) -> Int32:
    """Single-element remap; shared by CPU loop and GPU kernel."""
    if log_t < 0:
        return log_t
    var t = Int(log_t)
    var bi = Int(batch_u32)
    if bi >= lut_rows:
        return Int32(-1)
    var page_idx = t // page_size
    if page_idx >= lut_cols:
        return Int32(-1)
    var tok_in_page = t % page_size
    var block_id = lut[bi * lut_cols + page_idx]
    if block_id >= invalid_block_id:
        return Int32(-1)
    return Int32(Int(block_id) * page_size + tok_in_page)


trait SparseIndexRemap(DevicePassable, TrivialRegisterPassable):
    """A second output a sparse-index producer writes beside each logical slot.

    Lets the kernel that selects the logical indices also emit what a
    consumer would otherwise derive from them in a separate pass.
    """

    comptime enabled: Bool
    """Whether `store` writes anything; `False` lets callers skip the batch
    lookup it would need."""

    def offset(self, slots: Int) -> Self:
        """Returns this remap with its output advanced by `slots` slots.

        Args:
            slots: Flat slots to advance by.

        Returns:
            The advanced remap.
        """
        ...

    def store(self, idx: Int, batch: Int, logical: Int32):
        """Writes the remapped value of output slot `idx`.

        Args:
            idx: Flat output slot.
            batch: Batch entry the slot's row belongs to.
            logical: The slot's logical token position, or `-1`.
        """
        ...


struct NoRemap(SparseIndexRemap):
    """Writes nothing: the producer emits logical indices only."""

    comptime device_type: AnyType = Self
    """Device-side type alias; the struct is trivially device-passable."""

    comptime enabled = False
    """Nothing is written."""

    @inline(.always)
    def __init__(out self):
        """Creates the empty remap."""
        pass

    @inline(.always)
    def offset(self, slots: Int) -> Self:
        """Returns `self`; there is no output to advance.

        Args:
            slots: Ignored.

        Returns:
            `self`.
        """
        return self

    @inline(.always)
    def store(self, idx: Int, batch: Int, logical: Int32):
        """Does nothing.

        Args:
            idx: Ignored.
            batch: Ignored.
            logical: Ignored.
        """
        pass

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode_fields[Self](self, target)

    @staticmethod
    def get_type_name() -> String:
        """Returns the type name for runtime diagnostics.

        Returns:
            Always `"NoRemap"`.
        """
        return "NoRemap"


struct PagedKVRemap[page_size: Int](SparseIndexRemap):
    """Writes each slot's physical paged-cache row, as the remap kernel would.

    The value written for slot `idx` is exactly what
    `paged_sparse_kv_index_remap` produces for it, so a consumer can take this
    buffer in place of running that kernel.

    Parameters:
        page_size: Tokens per page of the cache `lut` indexes.
    """

    comptime device_type: AnyType = Self
    """Device-side type alias; the struct is trivially device-passable."""

    comptime enabled = True
    """Physical rows are written."""

    var physical_out: MutPointer[Int32, MutUntrackedOrigin]
    """Flat output buffer, laid out like the logical indices."""
    var lut: ImmPointer[UInt32, ImmUntrackedOrigin]
    """Row-major `[lut_rows, lut_cols]` page lookup table."""
    var lut_cols: Int32
    """Logical pages per batch entry."""
    var lut_rows: Int32
    """Batch entries in `lut`."""
    var invalid_block_id: UInt32
    """Block ids at or above this (the page count) are invalid."""

    @inline(.always)
    def __init__(
        out self,
        physical_out: MutPointer[Int32, _],
        lut: Pointer[UInt32, _],
        lut_cols: Int,
        lut_rows: Int,
        invalid_block_id: UInt32,
    ):
        """Describes the cache to remap into and where to write.

        Args:
            physical_out: Flat output buffer, laid out like the logical
                indices.
            lut: Row-major `[lut_rows, lut_cols]` page lookup table.
            lut_cols: Logical pages per batch entry.
            lut_rows: Batch entries in `lut`.
            invalid_block_id: Block ids at or above this are invalid; the
                cache's page count.
        """
        self.physical_out = physical_out.unsafe_origin_cast[
            MutUntrackedOrigin
        ]()
        self.lut = lut.as_imm().unsafe_origin_cast[ImmUntrackedOrigin]()
        self.lut_cols = Int32(lut_cols)
        self.lut_rows = Int32(lut_rows)
        self.invalid_block_id = invalid_block_id

    @inline(.always)
    def offset(self, slots: Int) -> Self:
        """Returns this remap with `physical_out` advanced by `slots`.

        Args:
            slots: Flat slots to advance by.

        Returns:
            The advanced remap.
        """
        var r = self
        r.physical_out = self.physical_out.unsafe_offset(slots)
        return r

    @inline(.always)
    def store(self, idx: Int, batch: Int, logical: Int32):
        """Writes `physical_out[idx]` for logical position `logical`.

        Args:
            idx: Flat output slot.
            batch: Batch entry the slot's row belongs to.
            logical: The slot's logical token position, or `-1`.
        """
        self.physical_out[idx] = _remap_one(
            logical,
            UInt32(batch),
            self.lut,
            Int(self.lut_cols),
            Int(self.lut_rows),
            Self.page_size,
            self.invalid_block_id,
        )

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode_fields[Self](self, target)

    @staticmethod
    def get_type_name() -> String:
        """Returns the type name for runtime diagnostics.

        Returns:
            Always `"PagedKVRemap"`.
        """
        return "PagedKVRemap"


@inline(.always)
def _remap_scale_one(
    log_t: Int32,
    batch_u32: UInt32,
    lut: ImmPointer[UInt32, _],
    lut_cols: Int,
    lut_rows: Int,
    page_size: Int,
    invalid_block_id: UInt32,
    rows_per_block: Int,
) -> Int32:
    """Like `_remap_one`, for a pool whose blocks sit `rows_per_block` rows
    apart: returns `block * rows_per_block + offset`, a pool row index rather
    than the `block * page_size + offset` encoding."""
    if log_t < 0:
        return log_t
    var t = Int(log_t)
    var bi = Int(batch_u32)
    if bi >= lut_rows:
        return Int32(-1)
    var page_idx = t // page_size
    if page_idx >= lut_cols:
        return Int32(-1)
    var block_id = lut[bi * lut_cols + page_idx]
    if block_id >= invalid_block_id:
        return Int32(-1)
    return Int32(Int(block_id) * rows_per_block + t % page_size)


struct PagedKVScaleRemap[page_size: Int, top_k: Int](SparseIndexRemap):
    """Writes each slot's physical row and its row in the scale pool.

    For an NVFP4 cache the scales are paged apart from the values: they have
    their own block ids and page stride, so a consumer cannot derive a
    selected token's scale row from its value row. Each output row holds
    `2 * top_k` slots: the `PagedKVRemap` encoding of the slot (the
    `top_k` value rows), then the slot's scale row (the next `top_k`), which is
    `scale_block * scale_rows_per_block + offset`, the same row
    `PagedKVCache.scale_row_idx` returns for the position. Either is `-1` where
    the position is `-1` or its page is unmapped.

    Parameters:
        page_size: Tokens per page of both pools.
        top_k: Slots per output row.
    """

    comptime device_type: AnyType = Self
    """Device-side type alias; the struct is trivially device-passable."""

    comptime enabled = True
    """Physical and scale rows are written."""

    var physical_out: MutPointer[Int32, MutUntrackedOrigin]
    """Flat output buffer, `[rows, 2 * top_k]`."""
    var lut: ImmPointer[UInt32, ImmUntrackedOrigin]
    """Row-major `[lut_rows, lut_cols]` value page lookup table."""
    var scales_lut: ImmPointer[UInt32, ImmUntrackedOrigin]
    """Row-major `[lut_rows, lut_cols]` scale page lookup table."""
    var lut_cols: Int32
    """Logical pages per batch entry in `lut`."""
    var scales_lut_cols: Int32
    """Logical pages per batch entry in `scales_lut`."""
    var lut_rows: Int32
    """Batch entries in the tables."""
    var invalid_block_id: UInt32
    """Value block ids at or above this (the page count) are invalid."""
    var scales_invalid_block_id: UInt32
    """Scale block ids at or above this (the scale page count) are invalid."""
    var scale_rows_per_block: Int32
    """Rows between consecutive blocks of the scale pool."""

    @inline(.always)
    def __init__(
        out self,
        physical_out: MutPointer[Int32, _],
        lut: ImmPointer[UInt32, _],
        scales_lut: ImmPointer[UInt32, _],
        lut_cols: Int,
        scales_lut_cols: Int,
        lut_rows: Int,
        invalid_block_id: UInt32,
        scales_invalid_block_id: UInt32,
        scale_rows_per_block: Int,
    ):
        """Describes the two pools to remap into and where to write.

        Args:
            physical_out: Flat output buffer, `[rows, 2 * top_k]`.
            lut: Row-major `[lut_rows, lut_cols]` value page lookup table.
            scales_lut: Row-major `[lut_rows, lut_cols]` scale page lookup
                table.
            lut_cols: Logical pages per batch entry in `lut`.
            scales_lut_cols: Logical pages per batch entry in `scales_lut`.
            lut_rows: Batch entries in the tables.
            invalid_block_id: Value block ids at or above this are invalid;
                the value pool's page count.
            scales_invalid_block_id: Scale block ids at or above this are
                invalid; the scale pool's page count.
            scale_rows_per_block: Rows between consecutive scale blocks.
        """
        self.physical_out = physical_out.unsafe_origin_cast[
            MutUntrackedOrigin
        ]()
        self.lut = lut.as_imm().unsafe_origin_cast[ImmUntrackedOrigin]()
        self.scales_lut = scales_lut.as_imm().unsafe_origin_cast[
            ImmUntrackedOrigin
        ]()
        self.lut_cols = Int32(lut_cols)
        self.scales_lut_cols = Int32(scales_lut_cols)
        self.lut_rows = Int32(lut_rows)
        self.invalid_block_id = invalid_block_id
        self.scales_invalid_block_id = scales_invalid_block_id
        self.scale_rows_per_block = Int32(scale_rows_per_block)

    @inline(.always)
    def offset(self, slots: Int) -> Self:
        """Returns this remap with `physical_out` advanced past `slots` slots.

        Args:
            slots: Slots of the logical output to advance by, a whole number
                of rows.

        Returns:
            The advanced remap.
        """
        var r = self
        r.physical_out = self.physical_out.unsafe_offset(2 * slots)
        return r

    @inline(.always)
    def store(self, idx: Int, batch: Int, logical: Int32):
        """Writes both rows of slot `idx` for logical position `logical`.

        Args:
            idx: Flat slot of the logical output, `row * top_k + column`.
            batch: Batch entry the slot's row belongs to.
            logical: The slot's logical token position, or `-1`.
        """
        var row = idx // Self.top_k
        var out = self.physical_out.unsafe_offset(
            row * 2 * Self.top_k + (idx - row * Self.top_k)
        )
        out[0] = _remap_one(
            logical,
            UInt32(batch),
            self.lut,
            Int(self.lut_cols),
            Int(self.lut_rows),
            Self.page_size,
            self.invalid_block_id,
        )
        out[Self.top_k] = _remap_scale_one(
            logical,
            UInt32(batch),
            self.scales_lut,
            Int(self.scales_lut_cols),
            Int(self.lut_rows),
            Self.page_size,
            self.scales_invalid_block_id,
            Int(self.scale_rows_per_block),
        )

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode_fields[Self](self, target)

    @staticmethod
    def get_type_name() -> String:
        """Returns the type name for runtime diagnostics.

        Returns:
            Always `"PagedKVScaleRemap"`.
        """
        return "PagedKVScaleRemap"


@inline(.always)
def _find_batch_for_row(
    r: Int,
    row_offsets: ImmPointer[UInt32, _],
    num_batches: Int,
) -> UInt32:
    """Map ragged row ``r`` to batch ``b`` with ``row_offsets[b] <= r < row_offsets[b+1]``.
    """
    var ru = UInt32(r)
    for b in range(num_batches):
        if ru >= row_offsets[b] and ru < row_offsets[b + 1]:
            return UInt32(b)
    return UInt32(0)


@__name(t"paged_sparse_kv_index_remap_row_offs_kernel")
def _paged_sparse_kv_index_remap_row_offs_kernel(
    logical: ImmPointer[Int32, ImmutAnyOrigin],
    row_offsets: ImmPointer[UInt32, ImmutAnyOrigin],
    lut: ImmPointer[UInt32, ImmutAnyOrigin],
    physical_out: MutPointer[Int32, MutAnyOrigin],
    num_indices: Int32,
    lut_cols: Int32,
    lut_rows: Int32,
    page_size: Int32,
    invalid_block_id: UInt32,
    indices_stride: Int32,
    num_batches: Int32,
    logical_stride0: Int32,
    logical_stride1: Int32,
):
    var _num_indices = Int(num_indices)
    var _lut_cols = Int(lut_cols)
    var _lut_rows = Int(lut_rows)
    var _page_size = Int(page_size)
    var _indices_stride = Int(indices_stride)
    var _num_batches = Int(num_batches)
    var _logical_stride0 = Int(logical_stride0)
    var _logical_stride1 = Int(logical_stride1)
    var tid = block_idx.x * block_dim.x + thread_idx.x
    if tid >= _num_indices:
        return
    var r = tid // _indices_stride
    var c = tid - r * _indices_stride
    var loff = r * _logical_stride0 + c * _logical_stride1
    var batch_u32 = _find_batch_for_row(r, row_offsets, _num_batches)
    physical_out[tid] = _remap_one(
        logical[loff],
        batch_u32,
        lut,
        _lut_cols,
        _lut_rows,
        _page_size,
        invalid_block_id,
    )


def paged_sparse_kv_logical_to_physical_indices_from_row_offsets_dispatch[
    target: StaticString,
    page_size: Int,
](
    physical_out: MutPointer[Int32, _],
    logical: ImmPointer[Int32, _],
    input_row_offsets: ImmPointer[UInt32, _],
    lut: ImmPointer[UInt32, _],
    num_indices: Int,
    lut_cols: Int,
    lut_rows: Int,
    indices_stride: Int,
    invalid_block_id: UInt32,
    num_batches: Int,
    logical_stride0: Int,
    logical_stride1: Int,
    ctx: DeviceContext,
) raises:
    """Remap logical sparse slots using ragged ``input_row_offsets`` (not per-slot batch ids).

    Each flattened slot ``tid`` maps to row ``r = tid // indices_stride`` in the logical
    sparse matrix; batch index is found by scanning ``input_row_offsets``. Logical loads
    use ``(logical_stride0, logical_stride1)`` like MOGG graph tensors.

    Parameters:
        target: StaticString identifying the dispatch target, used to select CPU versus GPU.
        page_size: Number of tokens per KV cache page.

    Args:
        physical_out: Output buffer of physical row indices, one per sparse slot.
        logical: Pointer to the flattened logical sparse index tensor.
        input_row_offsets: Pointer to per-batch row offsets of length
            ``num_batches + 1``.
        lut: Pointer to the paged KV cache lookup table.
        num_indices: Total number of sparse slots to remap.
        lut_cols: Number of columns in the lookup table (logical pages per batch).
        lut_rows: Number of rows in the lookup table (number of batches).
        indices_stride: Stride used to split a flat slot index into row and column.
        invalid_block_id: Sentinel block id marking invalid LUT entries; slots
            referencing it are written ``-1``.
        num_batches: Number of batches in the ragged batch dimension.
        logical_stride0: Row stride for indexing the logical tensor.
        logical_stride1: Column stride for indexing the logical tensor.
        ctx: Device context used to enqueue the GPU kernel.
    """
    comptime if is_cpu[target]():
        for i in range(num_indices):
            var r = i // indices_stride
            var c = i - r * indices_stride
            var loff = r * logical_stride0 + c * logical_stride1
            var batch_u32 = _find_batch_for_row(
                r, input_row_offsets, num_batches
            )
            physical_out[i] = _remap_one(
                logical[loff],
                batch_u32,
                lut,
                lut_cols,
                lut_rows,
                page_size,
                invalid_block_id,
            )
    else:
        if num_indices == 0:
            return
        var gpu_ctx = ctx
        comptime BLOCK = 256
        var grid = ceildiv(num_indices, BLOCK)
        comptime kernel = _paged_sparse_kv_index_remap_row_offs_kernel
        gpu_ctx.enqueue_function[kernel](
            logical,
            input_row_offsets,
            lut,
            physical_out,
            Int32(num_indices),
            Int32(lut_cols),
            Int32(lut_rows),
            Int32(page_size),
            invalid_block_id,
            Int32(indices_stride),
            Int32(num_batches),
            Int32(logical_stride0),
            Int32(logical_stride1),
            grid_dim=grid,
            block_dim=BLOCK,
        )


@inline(.always)
def paged_sparse_kv_index_remap[
    target: StaticString,
    page_size: Int,
    indices_stride: Int,
    cache_dtype: DType,
](
    physical_out: MutPointer[Int32, _],
    sparse_indices: InputTensor[dtype=.int32, rank=2, ...],
    input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
    kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
    kv_blocks: MutableInputTensor[dtype=cache_dtype, rank=6, ...],
    ctx: DeviceContext,
) raises:
    """High-level remap for sparse MLA MOGG ops (logical indices → physical rows).

    Unpacks graph tensors, sets ``invalid_block_id`` from ``kv_blocks.dim_size(0)``,
    derives batch count from row offsets, and dispatches
    ``paged_sparse_kv_logical_to_physical_indices_from_row_offsets_dispatch``.

    Parameters:
        target: StaticString identifying the dispatch target, used to select CPU versus GPU.
        page_size: Number of tokens per KV cache page.
        indices_stride: Stride used to split a flat slot index into row and column.
        cache_dtype: DType of the ``kv_blocks`` tensor.

    Args:
        physical_out: Output buffer of physical row indices, one per sparse slot.
        sparse_indices: Rank-2 ``int32`` tensor of logical sparse token positions.
        input_row_offsets: Rank-1 ``uint32`` tensor of per-batch row offsets.
        kv_lookup_table: Rank-2 ``uint32`` lookup table mapping logical page index
            to physical block id.
        kv_blocks: Rank-6 KV cache blocks; ``dim_size(0)`` supplies the invalid
            block id sentinel.
        ctx: Device context used to enqueue the GPU kernel.
    """
    var si_tt = sparse_indices.to_tile_tensor()
    var num_batches = input_row_offsets.dim_size(0) - 1
    var invalid_block_id = UInt32(kv_blocks.dim_size(0))
    var log_stride0 = Int(si_tt.dynamic_stride(0))
    var log_stride1 = Int(si_tt.dynamic_stride(1))
    paged_sparse_kv_logical_to_physical_indices_from_row_offsets_dispatch[
        target, page_size
    ](
        physical_out,
        rebind[MutPointer[Int32, MutAnyOrigin]](si_tt.ptr),
        rebind[MutPointer[UInt32, MutAnyOrigin]](
            input_row_offsets.to_tile_tensor().ptr,
        ),
        rebind[MutPointer[UInt32, MutAnyOrigin]](
            kv_lookup_table.to_tile_tensor().ptr,
        ),
        sparse_indices.size(),
        kv_lookup_table.dim_size(1),
        kv_lookup_table.dim_size(0),
        indices_stride,
        invalid_block_id,
        num_batches,
        log_stride0,
        log_stride1,
        ctx,
    )
