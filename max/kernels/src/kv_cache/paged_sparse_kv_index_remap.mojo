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

from std.math import ceildiv
from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host.info import is_cpu
from std.memory import UnsafePointer
from std.runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor
from tensor.managed_tensor_slice import (
    _MutableInputTensor as MutableInputTensor,
)


@always_inline
def _remap_one(
    log_t: Int32,
    batch_u32: UInt32,
    lut: UnsafePointer[UInt32, MutAnyOrigin],
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


@always_inline
def _find_batch_for_row(
    r: Int,
    row_offsets: UnsafePointer[UInt32, MutAnyOrigin],
    num_batches: Int,
) -> UInt32:
    """Map ragged row ``r`` to batch ``b`` with ``row_offsets[b] <= r < row_offsets[b+1]``.
    """
    var ru = UInt32(r)
    for b in range(num_batches):
        if ru >= row_offsets[b] and ru < row_offsets[b + 1]:
            return UInt32(b)
    return UInt32(0)


@__name(t"paged_sparse_kv_index_remap_row_offs_kernel", mangle=True)
def _paged_sparse_kv_index_remap_row_offs_kernel(
    logical: UnsafePointer[Int32, MutAnyOrigin],
    row_offsets: UnsafePointer[UInt32, MutAnyOrigin],
    lut: UnsafePointer[UInt32, MutAnyOrigin],
    physical_out: UnsafePointer[Int32, MutAnyOrigin],
    num_indices: Int,
    lut_cols: Int,
    lut_rows: Int,
    page_size: Int,
    invalid_block_id: UInt32,
    indices_stride: Int,
    num_batches: Int,
    logical_stride0: Int,
    logical_stride1: Int,
):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    if tid >= num_indices:
        return
    var r = tid // indices_stride
    var c = tid - r * indices_stride
    var loff = r * logical_stride0 + c * logical_stride1
    var batch_u32 = _find_batch_for_row(r, row_offsets, num_batches)
    physical_out[tid] = _remap_one(
        logical[loff],
        batch_u32,
        lut,
        lut_cols,
        lut_rows,
        page_size,
        invalid_block_id,
    )


def paged_sparse_kv_logical_to_physical_indices_from_row_offsets_dispatch[
    target: StaticString,
    page_size: Int,
](
    physical_out: UnsafePointer[Int32, MutAnyOrigin],
    logical: UnsafePointer[Int32, MutAnyOrigin],
    input_row_offsets: UnsafePointer[UInt32, MutAnyOrigin],
    lut: UnsafePointer[UInt32, MutAnyOrigin],
    num_indices: Int,
    lut_cols: Int,
    lut_rows: Int,
    indices_stride: Int,
    invalid_block_id: UInt32,
    num_batches: Int,
    logical_stride0: Int,
    logical_stride1: Int,
    ctx: DeviceContextPtr,
) raises:
    """Remap logical sparse slots using ragged ``input_row_offsets`` (not per-slot batch ids).

    Each flattened slot ``tid`` maps to row ``r = tid // indices_stride`` in the logical
    sparse matrix; batch index is found by scanning ``input_row_offsets``. Logical loads
    use ``(logical_stride0, logical_stride1)`` like MOGG graph tensors.
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
        var gpu_ctx = ctx.get_device_context()
        comptime BLOCK = 256
        var grid = ceildiv(num_indices, BLOCK)
        comptime kernel = _paged_sparse_kv_index_remap_row_offs_kernel
        gpu_ctx.enqueue_function[kernel](
            logical,
            input_row_offsets,
            lut,
            physical_out,
            num_indices,
            lut_cols,
            lut_rows,
            page_size,
            invalid_block_id,
            indices_stride,
            num_batches,
            logical_stride0,
            logical_stride1,
            grid_dim=grid,
            block_dim=BLOCK,
        )


@always_inline
def paged_sparse_kv_index_remap[
    target: StaticString,
    page_size: Int,
    indices_stride: Int,
    cache_dtype: DType,
](
    physical_out: UnsafePointer[Int32, MutAnyOrigin],
    sparse_indices: InputTensor[dtype=DType.int32, rank=2, ...],
    input_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
    kv_lookup_table: InputTensor[dtype=DType.uint32, rank=2, ...],
    kv_blocks: MutableInputTensor[dtype=cache_dtype, rank=6, ...],
    ctx: DeviceContextPtr,
) raises:
    """High-level remap for sparse MLA MOGG ops (logical indices → physical rows).

    Unpacks graph tensors, sets ``invalid_block_id`` from ``kv_blocks.dim_size(0)``,
    derives batch count from row offsets, and dispatches
    ``paged_sparse_kv_logical_to_physical_indices_from_row_offsets_dispatch``.
    """
    var si_lt = sparse_indices.to_layout_tensor()
    var num_batches = input_row_offsets.dim_size(0) - 1
    var invalid_block_id = UInt32(kv_blocks.dim_size(0))
    var log_stride0 = Int(si_lt.runtime_layout.stride.value[0])
    var log_stride1 = Int(si_lt.runtime_layout.stride.value[1])
    paged_sparse_kv_logical_to_physical_indices_from_row_offsets_dispatch[
        target, page_size
    ](
        physical_out,
        rebind[UnsafePointer[Int32, MutAnyOrigin]](si_lt.ptr),
        rebind[UnsafePointer[UInt32, MutAnyOrigin]](
            input_row_offsets.to_layout_tensor().ptr,
        ),
        rebind[UnsafePointer[UInt32, MutAnyOrigin]](
            kv_lookup_table.to_layout_tensor().ptr,
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
