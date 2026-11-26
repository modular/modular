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
"""GPU test for row-offset sparse KV remap (same dispatch as sparse MLA MOGG path)."""

from max.gpu.host import DeviceContext
from std.memory import alloc
from std.memory.alloc import unsafe_alloc

from std.testing import assert_equal

from kv_cache.paged_sparse_kv_index_remap import (
    PagedKVRemap,
    PagedKVScaleRemap,
    paged_sparse_kv_logical_to_physical_indices_from_row_offsets_dispatch,
    sparse_selection_rows,
)


@inline(.always)
def _find_batch_for_row_ref(
    r: Int,
    row_offsets: ImmPointer[UInt32, _],
    num_batches: Int,
) -> UInt32:
    """Matches production ``_find_batch_for_row`` (test golden only)."""
    var ru = UInt32(r)
    for b in range(num_batches):
        if ru >= row_offsets[b] and ru < row_offsets[b + 1]:
            return UInt32(b)
    return UInt32(0)


@inline(.always)
def _remap_one_ref(
    log_t: Int32,
    batch_u32: UInt32,
    lut: ImmPointer[UInt32, _],
    lut_cols: Int,
    lut_rows: Int,
    page_size: Int,
    invalid_block_id: UInt32,
) -> Int32:
    """Matches production ``_remap_one`` (test golden only)."""
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


def _reference_row_offsets_remap(
    logical: ImmPointer[Int32, _],
    row_offsets: ImmPointer[UInt32, _],
    lut: ImmPointer[UInt32, _],
    physical_out: MutPointer[Int32, _],
    num_indices: Int,
    lut_cols: Int,
    lut_rows: Int,
    indices_stride: Int,
    invalid_block_id: UInt32,
    num_batches: Int,
    logical_stride0: Int,
    logical_stride1: Int,
    page_size: Int,
):
    """CPU golden mirroring ``paged_sparse_kv_logical_to_physical_indices_from_row_offsets_dispatch``.
    """
    for i in range(num_indices):
        var r = i // indices_stride
        var c = i - r * indices_stride
        var loff = r * logical_stride0 + c * logical_stride1
        var batch_u32 = _find_batch_for_row_ref(r, row_offsets, num_batches)
        physical_out[i] = _remap_one_ref(
            logical[loff],
            batch_u32,
            lut,
            lut_cols,
            lut_rows,
            page_size,
            invalid_block_id,
        )


def test_sparse_selection_rows_resolve_value_rows(ctx: DeviceContext) raises:
    """A selection's positions map to the rows of an FP8 cache's pages.

    Each row's positions resolve through its own batch entry's table, and a
    position that is `-1` or on an unmapped page gives `-1`.
    """
    comptime PAGE_SIZE = 128
    comptime TOP_K = 2
    comptime ROWS = 3
    comptime LUT_COLS = 4
    comptime NUM_BATCHES = 2
    comptime INVALID_BLOCK = 512

    var h_row_off = unsafe_alloc[UInt32](NUM_BATCHES + 1)
    h_row_off[0] = 0
    h_row_off[1] = 2
    h_row_off[2] = 3

    var h_lut = unsafe_alloc[UInt32](NUM_BATCHES * LUT_COLS)
    var blocks: List[UInt32] = [
        10,
        5,
        INVALID_BLOCK,
        INVALID_BLOCK,
        3,
        20,
        INVALID_BLOCK,
        INVALID_BLOCK,
    ]
    for i in range(NUM_BATCHES * LUT_COLS):
        h_lut[i] = blocks[i]

    # Rows 0 and 1 belong to batch entry 0, row 2 to entry 1.
    var positions: List[Int32] = [0, 129, -1, 256, 200, 5]
    var h_pos = unsafe_alloc[Int32](ROWS * TOP_K)
    for i in range(ROWS * TOP_K):
        h_pos[i] = positions[i]

    var expected: List[Int32] = [
        10 * PAGE_SIZE + 0,
        5 * PAGE_SIZE + 1,
        -1,
        -1,
        20 * PAGE_SIZE + 72,
        3 * PAGE_SIZE + 5,
    ]
    var h_out = unsafe_alloc[Int32](ROWS * TOP_K)

    var d_pos = ctx.enqueue_create_buffer[.int32](ROWS * TOP_K)
    var d_row_off = ctx.enqueue_create_buffer[.uint32](NUM_BATCHES + 1)
    var d_lut = ctx.enqueue_create_buffer[.uint32](NUM_BATCHES * LUT_COLS)
    var d_out = ctx.enqueue_create_buffer[.int32](ROWS * TOP_K)
    ctx.enqueue_copy(d_pos, h_pos)
    ctx.enqueue_copy(d_row_off, h_row_off)
    ctx.enqueue_copy(d_lut, h_lut)

    var remap = PagedKVRemap[PAGE_SIZE](
        d_out.unsafe_ptr(),
        d_lut.unsafe_ptr().as_imm(),
        LUT_COLS,
        NUM_BATCHES,
        UInt32(INVALID_BLOCK),
    )
    sparse_selection_rows(
        remap,
        d_pos.unsafe_ptr().as_imm(),
        d_row_off.unsafe_ptr().as_imm(),
        ROWS,
        TOP_K,
        NUM_BATCHES,
        ctx,
    )
    ctx.enqueue_copy(h_out, d_out)
    # TODO(GPUA-92): keep the inputs alive until the kernel has run, as in
    # `main`.
    _ = d_pos^
    _ = d_row_off^
    _ = d_lut^
    ctx.synchronize()

    for i in range(ROWS * TOP_K):
        assert_equal(Int(h_out[i]), Int(expected[i]), "selection row mismatch")

    h_row_off.unsafe_free()
    h_lut.unsafe_free()
    h_pos.unsafe_free()
    h_out.unsafe_free()


def test_sparse_selection_rows_resolve_scale_rows(ctx: DeviceContext) raises:
    """A selection's positions map to value rows and scale-pool rows.

    The scale pool has its own block ids and a block stride other than the
    page size, so each half needs its own table.
    """
    comptime PAGE_SIZE = 128
    comptime TOP_K = 2
    comptime ROWS = 3
    comptime LUT_COLS = 4
    comptime NUM_BATCHES = 2
    comptime INVALID_BLOCK = 512
    comptime INVALID_SCALE_BLOCK = 64
    comptime SCALE_ROWS_PER_BLOCK = 384

    var h_row_off = unsafe_alloc[UInt32](NUM_BATCHES + 1)
    h_row_off[0] = 0
    h_row_off[1] = 2
    h_row_off[2] = 3

    # Value and scale block ids of each batch entry's logical pages.
    var h_lut = unsafe_alloc[UInt32](NUM_BATCHES * LUT_COLS)
    var h_scales_lut = unsafe_alloc[UInt32](NUM_BATCHES * LUT_COLS)
    var value_blocks: List[UInt32] = [
        10,
        5,
        INVALID_BLOCK,
        INVALID_BLOCK,
        3,
        20,
        INVALID_BLOCK,
        INVALID_BLOCK,
    ]
    var scale_blocks: List[UInt32] = [
        7,
        2,
        INVALID_SCALE_BLOCK,
        INVALID_SCALE_BLOCK,
        4,
        9,
        INVALID_SCALE_BLOCK,
        INVALID_SCALE_BLOCK,
    ]
    for i in range(NUM_BATCHES * LUT_COLS):
        h_lut[i] = value_blocks[i]
        h_scales_lut[i] = scale_blocks[i]

    # Rows 0 and 1 belong to batch entry 0, row 2 to entry 1.
    var positions: List[Int32] = [0, 129, -1, 256, 200, 5]
    var h_pos = unsafe_alloc[Int32](ROWS * TOP_K)
    for i in range(ROWS * TOP_K):
        h_pos[i] = positions[i]

    var expected: List[Int32] = [
        10 * PAGE_SIZE + 0,
        5 * PAGE_SIZE + 1,
        7 * SCALE_ROWS_PER_BLOCK + 0,
        2 * SCALE_ROWS_PER_BLOCK + 1,
        -1,
        -1,
        -1,
        -1,
        20 * PAGE_SIZE + 72,
        3 * PAGE_SIZE + 5,
        9 * SCALE_ROWS_PER_BLOCK + 72,
        4 * SCALE_ROWS_PER_BLOCK + 5,
    ]
    var h_out = unsafe_alloc[Int32](ROWS * 2 * TOP_K)

    var d_pos = ctx.enqueue_create_buffer[.int32](ROWS * TOP_K)
    var d_row_off = ctx.enqueue_create_buffer[.uint32](NUM_BATCHES + 1)
    var d_lut = ctx.enqueue_create_buffer[.uint32](NUM_BATCHES * LUT_COLS)
    var d_scales_lut = ctx.enqueue_create_buffer[.uint32](
        NUM_BATCHES * LUT_COLS
    )
    var d_out = ctx.enqueue_create_buffer[.int32](ROWS * 2 * TOP_K)
    ctx.enqueue_copy(d_pos, h_pos)
    ctx.enqueue_copy(d_row_off, h_row_off)
    ctx.enqueue_copy(d_lut, h_lut)
    ctx.enqueue_copy(d_scales_lut, h_scales_lut)

    var remap = PagedKVScaleRemap[PAGE_SIZE, TOP_K](
        d_out.unsafe_ptr(),
        d_lut.unsafe_ptr().as_imm(),
        d_scales_lut.unsafe_ptr().as_imm(),
        LUT_COLS,
        LUT_COLS,
        NUM_BATCHES,
        UInt32(INVALID_BLOCK),
        UInt32(INVALID_SCALE_BLOCK),
        SCALE_ROWS_PER_BLOCK,
    )
    sparse_selection_rows(
        remap,
        d_pos.unsafe_ptr().as_imm(),
        d_row_off.unsafe_ptr().as_imm(),
        ROWS,
        TOP_K,
        NUM_BATCHES,
        ctx,
    )
    ctx.enqueue_copy(h_out, d_out)
    # TODO(GPUA-92): keep the inputs alive until the kernel has run, as in
    # `main`.
    _ = d_pos^
    _ = d_row_off^
    _ = d_lut^
    _ = d_scales_lut^
    ctx.synchronize()

    for i in range(ROWS * 2 * TOP_K):
        assert_equal(Int(h_out[i]), Int(expected[i]), "selection row mismatch")

    h_row_off.unsafe_free()
    h_lut.unsafe_free()
    h_scales_lut.unsafe_free()
    h_pos.unsafe_free()
    h_out.unsafe_free()


def main() raises:
    comptime PAGE_SIZE = 128
    comptime INVALID_BLOCK = 512
    comptime LUT_ROWS = 2
    comptime LUT_COLS = 4
    comptime NUM_IDX = 5
    comptime INDICES_STRIDE = 1
    comptime NUM_BATCHES = 2
    comptime LOG_S0 = 1
    comptime LOG_S1 = 1

    # Sparse rows 0..3 → batch 0; row 4 → batch 1 (indices_stride=1 ⇒ tid i → row i).
    var h_row_off = unsafe_alloc[UInt32](NUM_BATCHES + 1)
    h_row_off[0] = 0
    h_row_off[1] = 4
    h_row_off[2] = 5

    var h_lut = unsafe_alloc[UInt32](LUT_ROWS * LUT_COLS)
    h_lut[0 * LUT_COLS + 0] = 10
    h_lut[0 * LUT_COLS + 1] = 5
    h_lut[0 * LUT_COLS + 2] = INVALID_BLOCK
    h_lut[0 * LUT_COLS + 3] = INVALID_BLOCK
    h_lut[1 * LUT_COLS + 0] = 3
    h_lut[1 * LUT_COLS + 1] = 20
    h_lut[1 * LUT_COLS + 2] = INVALID_BLOCK
    h_lut[1 * LUT_COLS + 3] = INVALID_BLOCK

    var h_log = unsafe_alloc[Int32](NUM_IDX)
    h_log[0] = 0
    h_log[1] = 129
    h_log[2] = -1
    h_log[3] = 256
    h_log[4] = 200

    var h_expected = unsafe_alloc[Int32](NUM_IDX)
    _reference_row_offsets_remap(
        h_log,
        h_row_off,
        h_lut,
        h_expected,
        NUM_IDX,
        LUT_COLS,
        LUT_ROWS,
        INDICES_STRIDE,
        UInt32(INVALID_BLOCK),
        NUM_BATCHES,
        LOG_S0,
        LOG_S1,
        PAGE_SIZE,
    )

    assert_equal(Int(h_expected[0]), 10 * PAGE_SIZE + 0)
    assert_equal(Int(h_expected[1]), 5 * PAGE_SIZE + 1)
    assert_equal(Int(h_expected[2]), -1)
    assert_equal(Int(h_expected[3]), -1)
    assert_equal(Int(h_expected[4]), 20 * PAGE_SIZE + 72)

    var h_out = unsafe_alloc[Int32](NUM_IDX)

    with DeviceContext() as ctx:
        var d_log = ctx.enqueue_create_buffer[.int32](NUM_IDX)
        var d_row_off = ctx.enqueue_create_buffer[.uint32](NUM_BATCHES + 1)
        var d_lut = ctx.enqueue_create_buffer[.uint32](LUT_ROWS * LUT_COLS)
        var d_out = ctx.enqueue_create_buffer[.int32](NUM_IDX)

        ctx.enqueue_copy(d_log, h_log)
        ctx.enqueue_copy(d_row_off, h_row_off)
        ctx.enqueue_copy(d_lut, h_lut)

        paged_sparse_kv_logical_to_physical_indices_from_row_offsets_dispatch[
            PAGE_SIZE
        ](
            d_out.unsafe_ptr(),
            d_log.unsafe_ptr(),
            d_row_off.unsafe_ptr(),
            d_lut.unsafe_ptr(),
            NUM_IDX,
            LUT_COLS,
            LUT_ROWS,
            INDICES_STRIDE,
            UInt32(INVALID_BLOCK),
            NUM_BATCHES,
            LOG_S0,
            LOG_S1,
            ctx,
        )

        ctx.enqueue_copy(h_out, d_out)

        # TODO(GPUA-92)
        # `DeviceBuffer.unsafe_ptr()` returns an untracked-origin pointer, so
        # these buffers are otherwise destroyed at the `unsafe_ptr()` calls in
        # the dispatch above — before the async kernel actually runs — freeing
        # their device memory and causing a use-after-free.
        _ = d_log^
        _ = d_row_off^
        _ = d_lut^

        ctx.synchronize()
        test_sparse_selection_rows_resolve_value_rows(ctx)
        test_sparse_selection_rows_resolve_scale_rows(ctx)

    for i in range(NUM_IDX):
        assert_equal(Int(h_out[i]), Int(h_expected[i]), "gpu remap mismatch")

    h_row_off.unsafe_free()
    h_log.unsafe_free()
    h_lut.unsafe_free()
    h_expected.unsafe_free()
    h_out.unsafe_free()
