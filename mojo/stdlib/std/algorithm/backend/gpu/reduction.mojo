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
"""Implements GPU reduction algorithms for parallel data aggregation."""

from std.math import align_up
from std.math.uutils import udivmod, ufloordiv

from std.algorithm.reduction import _get_nd_indices_from_flat_index
from std.gpu.primitives.block import broadcast
from std.gpu.host import DeviceContext
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    block_dim,
    block_idx,
    grid_dim,
    global_idx,
    lane_id,
    thread_idx,
    warp_id,
    AddressSpace,
)
from std.gpu.primitives.grid_controls import (
    PDL,
    PDLLevel,
    pdl_launch_attributes,
)
from std.gpu.host import DeviceContext, get_gpu_target
from std.gpu.primitives import warp
from std.memory import stack_allocation
from std.atomic import Atomic

from std.utils import IndexList
from std.utils.numerics import get_accum_type
from std.utils.static_tuple import StaticTuple
from std.sys import get_defined_int
from std.sys.info import simd_width_of


comptime _PDL_LEVEL = PDLLevel(1)


@always_inline
def block_reduce[
    BLOCK_SIZE: Int,
    reduce_fn: def[dtype: DType, width: SIMDSize](
        SIMD[dtype, width], SIMD[dtype, width]
    ) capturing[_] -> SIMD[dtype, width],
    dtype: DType,
    simd_width: SIMDSize,
](val: SIMD[dtype, simd_width], init: Scalar[dtype]) -> Scalar[dtype]:
    """Performs a block-level reduction of a single SIMD value across all
    threads in a GPU thread block using warp-level primitives and shared memory.

    Parameters:
        BLOCK_SIZE: The number of threads per block.
        reduce_fn: The binary reduction function.
        dtype: The data type of the elements.
        simd_width: The SIMD vector width.

    Args:
        val: The per-thread SIMD value to reduce.
        init: The identity value for the reduction.

    Returns:
        The reduced scalar result (valid on thread 0).
    """
    comptime num_reductions = 1

    @always_inline
    @parameter
    def reduce_wrapper[
        dtype: DType, width: SIMDSize, reduction_idx: Int
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert (
            reduction_idx < num_reductions
        ), "invalid reduction index"
        return reduce_fn(lhs, rhs)

    var val_tup = StaticTuple[SIMD[dtype, simd_width], num_reductions](val)
    var init_tup = StaticTuple[Scalar[dtype], num_reductions](init)

    return block_reduce[
        BLOCK_SIZE,
        num_reductions,
        reduce_wrapper,
        dtype,
        simd_width,
    ](val_tup, init_tup)[0]


@always_inline
def block_reduce[
    BLOCK_SIZE: Int,
    num_reductions: Int,
    reduce_fn: def[dtype: DType, width: SIMDSize, reduction_idx: Int](
        SIMD[dtype, width], SIMD[dtype, width]
    ) capturing[_] -> SIMD[dtype, width],
    dtype: DType,
    simd_width: SIMDSize,
](
    val: StaticTuple[SIMD[dtype, simd_width], num_reductions],
    init: StaticTuple[Scalar[dtype], num_reductions],
) -> StaticTuple[Scalar[dtype], num_reductions]:
    """Performs a block-level reduction of multiple fused SIMD values across all
    threads in a GPU thread block using warp shuffles and shared memory.

    Parameters:
        BLOCK_SIZE: The number of threads per block.
        num_reductions: The number of fused reductions to perform.
        reduce_fn: The binary reduction function, parameterized by reduction
          index.
        dtype: The data type of the elements.
        simd_width: The SIMD vector width.

    Args:
        val: The per-thread SIMD values to reduce, one per reduction.
        init: The identity values for each reduction.

    Returns:
        The reduced scalar results (valid on thread 0).
    """
    comptime assert (
        BLOCK_SIZE % WARP_SIZE == 0
    ), "block size must be a multiple of the warp size"

    @always_inline
    @parameter
    def do_warp_reduce(
        val: StaticTuple[SIMD[dtype, simd_width], num_reductions]
    ) -> StaticTuple[SIMD[dtype, simd_width], num_reductions]:
        var result = StaticTuple[SIMD[dtype, simd_width], num_reductions]()

        comptime for i in range(num_reductions):

            @always_inline
            @parameter
            def reduce_wrapper[
                dtype: DType, width: SIMDSize
            ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
                dtype, width
            ]:
                return reduce_fn[dtype, width, i](lhs, rhs)

            result[i] = warp.reduce[warp.shuffle_down, reduce_wrapper](val[i])

        return result

    var shared = stack_allocation[
        (BLOCK_SIZE // WARP_SIZE) * num_reductions * simd_width,
        dtype,
        address_space=AddressSpace.SHARED,
    ]()

    var warp = warp_id()

    var warp_accum = do_warp_reduce(val)

    if lane_id() == 0:
        comptime for i in range(num_reductions):
            # bank conflict for sub 4 byte data elems
            shared.store(
                (warp * num_reductions + i) * simd_width,
                warp_accum[i],
            )

    barrier()

    var last_accum = StaticTuple[SIMD[dtype, simd_width], num_reductions]()

    if thread_idx.x < ufloordiv(block_dim.x, WARP_SIZE):
        comptime for i in range(num_reductions):
            last_accum[i] = shared.load[width=simd_width](
                (num_reductions * lane_id() + i) * simd_width
            )
    else:
        comptime for i in range(num_reductions):
            last_accum[i] = init[i]

    var result_packed = do_warp_reduce(last_accum)
    var result = StaticTuple[Scalar[dtype], num_reductions]()

    comptime for i in range(num_reductions):
        result[i] = result_packed[i].reduce[
            reduce_fn[dtype, reduction_idx=i, ...]
        ]()

    return result


@always_inline
def row_reduce[
    BLOCK_SIZE: Int,
    input_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank]
    ) capturing[_] -> SIMD[dtype, width],
    reduce_fn: def[dtype: DType, width: SIMDSize](
        SIMD[dtype, width], SIMD[dtype, width]
    ) capturing[_] -> SIMD[dtype, width],
    dtype: DType,
    simd_width: Int,
    rank: Int,
    accum_type: DType = get_accum_type[dtype](),
](
    mut row_coords: IndexList[rank],
    axis: Int,
    init: Scalar[dtype],
    row_size: Int,
) -> Scalar[accum_type]:
    """Reduces a single row along the given axis using block-level cooperative
    reduction. Delegates to the multi-reduction `row_reduce` overload with
    `num_reductions=1`.

    Parameters:
        BLOCK_SIZE: The number of threads per block.
        input_fn: The lambda to load input elements.
        reduce_fn: The binary reduction function.
        dtype: The data type of the input elements.
        simd_width: The SIMD vector width.
        rank: The tensor rank.
        accum_type: The accumulator data type (defaults to widened type).

    Args:
        row_coords: The ND coordinates identifying the row.
        axis: The axis along which to reduce.
        init: The identity value for the reduction.
        row_size: The number of elements in the row.

    Returns:
        The reduced scalar result for the row.
    """
    comptime num_reductions = 1

    @always_inline
    @parameter
    def reduce_wrapper[
        dtype: DType, width: SIMDSize, reduction_idx: Int
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert (
            reduction_idx < num_reductions
        ), "invalid reduction index"
        return reduce_fn(lhs, rhs)

    var init_tup = StaticTuple[Scalar[dtype], num_reductions](init)

    return row_reduce[
        BLOCK_SIZE,
        num_reductions,
        input_fn,
        reduce_wrapper,
        dtype,
        simd_width,
        rank,
        accum_type=accum_type,
    ](row_coords, axis, init_tup, row_size)[0]


@always_inline
def row_reduce[
    BLOCK_SIZE: Int,
    num_reductions: Int,
    input_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank]
    ) capturing[_] -> SIMD[dtype, width],
    reduce_fn: def[dtype: DType, width: SIMDSize, reduction_idx: Int](
        SIMD[dtype, width], SIMD[dtype, width]
    ) capturing[_] -> SIMD[dtype, width],
    dtype: DType,
    simd_width: Int,
    rank: Int,
    accum_type: DType = get_accum_type[dtype](),
](
    mut row_coords: IndexList[rank],
    axis: Int,
    init: StaticTuple[Scalar[dtype], num_reductions],
    row_size: Int,
) -> StaticTuple[Scalar[accum_type], num_reductions]:
    """Reduces a row along the given axis with multiple fused reductions using
    cooperative SIMD-width reads across threads, a `block_reduce`, and scalar
    tail handling.

    Parameters:
        BLOCK_SIZE: The number of threads per block.
        num_reductions: The number of fused reductions to perform.
        input_fn: The lambda to load input elements.
        reduce_fn: The binary reduction function parameterized by reduction
          index.
        dtype: The data type of the input elements.
        simd_width: The SIMD vector width.
        rank: The tensor rank.
        accum_type: The accumulator data type (defaults to widened type).

    Args:
        row_coords: The ND coordinates identifying the row.
        axis: The axis along which to reduce.
        init: The identity values for each reduction.
        row_size: The number of elements in the row.

    Returns:
        The reduced scalar results, one per fused reduction.
    """
    var num_tail_values = row_size % simd_width
    var rounded_row_size = row_size - num_tail_values
    var row_size_padded = align_up(row_size // simd_width, BLOCK_SIZE)

    var accum = StaticTuple[SIMD[accum_type, simd_width], num_reductions]()
    var init_cast = StaticTuple[Scalar[accum_type], num_reductions]()

    comptime for i in range(num_reductions):
        init_cast[i] = init[i].cast[accum_type]()
        accum[i] = init_cast[i]

    var tid = thread_idx.x
    for offset_in_row in range(0, row_size_padded, BLOCK_SIZE):
        var idx_in_padded_row = (tid + offset_in_row) * simd_width

        if idx_in_padded_row >= rounded_row_size:
            break

        row_coords[axis] = idx_in_padded_row
        var val = input_fn[dtype, simd_width, rank](row_coords).cast[
            accum_type
        ]()

        comptime for i in range(num_reductions):
            accum[i] = reduce_fn[accum_type, simd_width, i](val, accum[i])

    var scalar_accum = block_reduce[
        BLOCK_SIZE,
        num_reductions,
        reduce_fn,
        accum_type,
        simd_width,
    ](accum, init_cast)

    # handle trailing values
    for idx_in_padded_row in range(rounded_row_size, row_size):
        row_coords[axis] = idx_in_padded_row
        var val = input_fn[dtype, 1, rank](row_coords).cast[accum_type]()

        comptime for i in range(num_reductions):
            scalar_accum[i] = reduce_fn[accum_type, 1, i](val, scalar_accum[i])

    return scalar_accum


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
def reduce_kernel[
    rank: Int,
    axis: Int,
    num_reductions: Int,
    BLOCK_SIZE: Int,
    input_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank]
    ) capturing[_] -> SIMD[dtype, width],
    output_fn: def[dtype: DType, width: SIMDSize, rank: Int](
        IndexList[rank], StaticTuple[SIMD[dtype, width], num_reductions]
    ) capturing[_] -> None,
    reduce_fn: def[ty: DType, width: SIMDSize, reduction_idx: Int](
        SIMD[ty, width], SIMD[ty, width]
    ) capturing[_] -> SIMD[ty, width],
    dtype: DType,
    simd_width: Int,
    accum_type: DType = get_accum_type[dtype](),
](shape: IndexList[rank], init: StaticTuple[Scalar[dtype], num_reductions],):
    """GPU kernel that reduces rows along a given axis. Each block reduces one
    row at a time using `row_reduce` and writes the result via `output_fn`.
    Uses a grid-stride loop to handle more rows than blocks.

    Parameters:
        rank: The tensor rank.
        axis: The axis along which to reduce.
        num_reductions: The number of fused reductions to perform.
        BLOCK_SIZE: The number of threads per block.
        input_fn: The lambda to load input elements.
        output_fn: The lambda to store output elements.
        reduce_fn: The binary reduction function.
        dtype: The data type of the elements.
        simd_width: The SIMD vector width.
        accum_type: The accumulator data type.

    Args:
        shape: The shape of the input tensor.
        init: The identity values for each reduction.
    """
    var row_size = shape[axis]
    var num_rows = shape.flattened_length() // row_size

    with PDL():
        # grid stride loop over rows
        # each block reduces a row, which requires no partial reductions
        for row_idx in range(block_idx.x, num_rows, grid_dim.x):
            var row_coords = _get_nd_indices_from_flat_index(
                row_idx, shape, axis
            )

            var row_accum = row_reduce[
                BLOCK_SIZE,
                num_reductions,
                input_fn,
                reduce_fn,
                dtype,
                simd_width,
                rank,
                accum_type=accum_type,
            ](row_coords, axis, init, row_size)

            if thread_idx.x == 0:
                var row_accum_cast = StaticTuple[
                    Scalar[dtype], num_reductions
                ]()

                comptime for i in range(num_reductions):
                    row_accum_cast[i] = row_accum[i].cast[dtype]()

                row_coords[axis] = 0
                output_fn[dtype, 1, rank](row_coords, row_accum_cast)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
def small_reduce_kernel[
    rank: Int,
    axis: Int,
    num_reductions: Int,
    BLOCK_SIZE: Int,
    input_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank]
    ) capturing[_] -> SIMD[dtype, width],
    output_fn: def[dtype: DType, width: SIMDSize, rank: Int](
        IndexList[rank], StaticTuple[SIMD[dtype, width], num_reductions]
    ) capturing[_] -> None,
    reduce_fn: def[ty: DType, width: SIMDSize, reduction_idx: Int](
        SIMD[ty, width], SIMD[ty, width]
    ) capturing[_] -> SIMD[ty, width],
    dtype: DType,
    simd_width: Int,
    accum_type: DType = get_accum_type[dtype](),
](shape: IndexList[rank], init: StaticTuple[Scalar[dtype], num_reductions],):
    """GPU kernel optimized for rows smaller than the warp size. Each warp
    reduces an entire row independently, allowing multiple rows to be reduced
    per block without shared-memory synchronization.

    Parameters:
        rank: The tensor rank.
        axis: The axis along which to reduce.
        num_reductions: The number of fused reductions to perform.
        BLOCK_SIZE: The number of threads per block.
        input_fn: The lambda to load input elements.
        output_fn: The lambda to store output elements.
        reduce_fn: The binary reduction function.
        dtype: The data type of the elements.
        simd_width: The SIMD vector width.
        accum_type: The accumulator data type.

    Args:
        shape: The shape of the input tensor.
        init: The identity values for each reduction.
    """
    var row_size = shape[axis]
    var num_rows = shape.flattened_length() // row_size

    comptime warps_per_block = BLOCK_SIZE // WARP_SIZE

    with PDL():
        # grid stride loop over rows
        # each block reduces as many rows as warps,
        # No need to partial reduction because this is the degenerated case of
        # rows smaller than warp size
        #
        for row_idx in range(
            block_idx.x * warps_per_block,
            num_rows,
            grid_dim.x * warps_per_block,
        ):
            var row_coords = _get_nd_indices_from_flat_index(
                row_idx + warp_id(), shape, axis
            )

            # One row per warp, warp collectively reads from global
            if warp_id() < warps_per_block:
                var val = InlineArray[
                    SIMD[accum_type, simd_width], num_reductions
                ](fill=0)

                comptime for i in range(num_reductions):
                    val[i] = init[i].cast[accum_type]()

                if lane_id() < row_size:
                    row_coords[axis] = lane_id()
                    var t = input_fn[dtype, simd_width, rank](row_coords).cast[
                        accum_type
                    ]()

                    val = type_of(val)(fill=t)
                else:
                    comptime for i in range(num_reductions):
                        val[i] = init[i].cast[accum_type]()

                var result = InlineArray[
                    SIMD[accum_type, simd_width], num_reductions
                ](fill=0)

                comptime for i in range(num_reductions):

                    @always_inline
                    @parameter
                    def reduce_wrapper[
                        dtype: DType, width: SIMDSize
                    ](
                        x: SIMD[dtype, width], y: SIMD[dtype, width]
                    ) capturing -> SIMD[dtype, width]:
                        return reduce_fn[dtype, width, i](x, y)

                    result[i] = warp.reduce[warp.shuffle_down, reduce_wrapper](
                        val[i]
                    )

                if lane_id() == 0:
                    var row_accum_cast = StaticTuple[
                        Scalar[dtype], num_reductions
                    ]()

                    comptime for i in range(num_reductions):
                        row_accum_cast[i] = result[i][0].cast[dtype]()

                    row_coords[axis] = 0
                    output_fn[dtype, 1, rank](row_coords, row_accum_cast)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
def twophase_reduce_kernel[
    rank: Int,
    axis: Int,
    num_reductions: Int,
    BLOCK_SIZE: Int,
    input_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank]
    ) capturing[_] -> SIMD[dtype, width],
    output_fn: def[dtype: DType, width: SIMDSize, rank: Int](
        IndexList[rank], StaticTuple[SIMD[dtype, width], num_reductions]
    ) capturing[_] -> None,
    reduce_fn: def[ty: DType, width: SIMDSize, reduction_idx: Int](
        SIMD[ty, width], SIMD[ty, width]
    ) capturing[_] -> SIMD[ty, width],
    dtype: DType,
    simd_width: Int,
    accum_type: DType = get_accum_type[dtype](),
](
    shape: IndexList[rank],
    init: StaticTuple[Scalar[dtype], num_reductions],
    partials: UnsafePointer[Scalar[accum_type], MutAnyOrigin],
    counters: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],
    blocks_per_row: Int,
):
    """GPU kernel for reductions when there are too few rows to saturate the
    device at one block per row. Assigns multiple blocks per row and uses a
    two-phase approach: each block reduces a chunk via cooperative block-level
    reduction, then the last block to finish (detected via a per-row atomic
    counter) reduces all partial results for its row.

    Parameters:
        rank: The tensor rank.
        axis: The axis along which to reduce.
        num_reductions: The number of fused reductions to perform.
        BLOCK_SIZE: The number of threads per block.
        input_fn: The lambda to load input elements.
        output_fn: The lambda to store output elements.
        reduce_fn: The binary reduction function.
        dtype: The data type of the elements.
        simd_width: The SIMD vector width.
        accum_type: The accumulator data type.

    Args:
        shape: The shape of the input tensor.
        init: The identity values for each reduction.
        partials: Global memory buffer for per-block partial results.
            Size: grid_dim.x * num_reductions elements of accum_type.
        counters: Global memory buffer for per-row atomic completion counters.
            Size: num_rows elements of int32, zero-initialized.
        blocks_per_row: The number of blocks assigned to each row.
    """
    comptime assert (
        simd_width == 1
    ), "twophase_reduce_kernel only currently supports simd_width == 1"
    var row_size = shape[axis]
    var num_rows = shape.flattened_length() // row_size

    var row_idx, block_in_row = udivmod(block_idx.x, blocks_per_row)

    if row_idx >= num_rows:
        return

    var row_coords = _get_nd_indices_from_flat_index(row_idx, shape, axis)

    # --- Phase 1: Each block reduces its portion of the row ---
    # Threads are striped across ALL blocks for this row to coalesce reads.
    var row_tid = block_in_row * BLOCK_SIZE + thread_idx.x
    var row_total_threads = blocks_per_row * BLOCK_SIZE

    var accum = StaticTuple[SIMD[accum_type, simd_width], num_reductions]()
    var init_cast = StaticTuple[Scalar[accum_type], num_reductions]()

    comptime for i in range(num_reductions):
        init_cast[i] = init[i].cast[accum_type]()
        accum[i] = init_cast[i]

    with PDL():
        for elem_idx in range(row_tid, row_size, row_total_threads):
            row_coords[axis] = elem_idx
            var val = input_fn[dtype, simd_width, rank](row_coords).cast[
                accum_type
            ]()

            comptime for i in range(num_reductions):
                accum[i] = reduce_fn[accum_type, simd_width, i](val, accum[i])

        var partial = block_reduce[
            BLOCK_SIZE, num_reductions, reduce_fn, accum_type, simd_width
        ](accum, init_cast)

        # Thread 0 writes partial result for this block and signals completion.
        var is_last_block: Scalar[DType.bool] = False
        if thread_idx.x == 0:
            var base = block_idx.x * num_reductions
            comptime for i in range(num_reductions):
                partials[base + i] = partial[i]

            var finished = Atomic[DType.int32].fetch_add(
                counters + row_idx, Int32(1)
            )
            is_last_block = finished == Int32(blocks_per_row - 1)

        # --- Phase 2: Last block reduces all partials for this row ---
        # Broadcast is_last_block from thread 0 to all threads via shared memory
        # so the entire block can participate cooperatively.
        is_last_block = broadcast[block_size=BLOCK_SIZE](is_last_block)
        if is_last_block:
            # Each thread loads a stripe of the partials and reduces locally.
            var thread_accum = StaticTuple[Scalar[accum_type], num_reductions]()

            comptime for i in range(num_reductions):
                thread_accum[i] = init_cast[i]

            var row_base = row_idx * blocks_per_row * num_reductions
            for b in range(thread_idx.x, blocks_per_row, BLOCK_SIZE):
                comptime for i in range(num_reductions):
                    thread_accum[i] = reduce_fn[accum_type, 1, i](
                        thread_accum[i],
                        partials[row_base + b * num_reductions + i],
                    )

            # Note this is currently no-op since we insist simd_width==1
            var accum_simd = StaticTuple[
                SIMD[accum_type, simd_width], num_reductions
            ]()
            comptime for i in range(num_reductions):
                accum_simd[i] = thread_accum[i]

            var final_result = block_reduce[
                BLOCK_SIZE, num_reductions, reduce_fn, accum_type, simd_width
            ](accum_simd, init_cast)

            if thread_idx.x == 0:
                var result_cast = StaticTuple[Scalar[dtype], num_reductions]()

                comptime for i in range(num_reductions):
                    result_cast[i] = final_result[i].cast[dtype]()

                row_coords[axis] = 0
                output_fn[dtype, 1, rank](row_coords, result_cast)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
def saturated_reduce_kernel[
    rank: Int,
    axis: Int,
    num_reductions: Int,
    BLOCK_SIZE: Int,
    input_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank]
    ) capturing[_] -> SIMD[dtype, width],
    output_fn: def[dtype: DType, width: SIMDSize, rank: Int](
        IndexList[rank], StaticTuple[SIMD[dtype, width], num_reductions]
    ) capturing[_] -> None,
    reduce_fn: def[ty: DType, width: SIMDSize, reduction_idx: Int](
        SIMD[ty, width], SIMD[ty, width]
    ) capturing[_] -> SIMD[ty, width],
    dtype: DType,
    simd_width: Int,
    accum_type: DType = get_accum_type[dtype](),
](shape: IndexList[rank], init: StaticTuple[Scalar[dtype], num_reductions],):
    """GPU kernel for reductions when the device is saturated with enough rows.
    Each thread independently reduces an entire row using SIMD packing,
    avoiding shared-memory synchronization entirely. Used when reducing along
    a non-contiguous axis.

    Parameters:
        rank: The tensor rank.
        axis: The axis along which to reduce.
        num_reductions: The number of fused reductions to perform.
        BLOCK_SIZE: The number of threads per block.
        input_fn: The lambda to load input elements.
        output_fn: The lambda to store output elements.
        reduce_fn: The binary reduction function.
        dtype: The data type of the elements.
        simd_width: The SIMD vector width.
        accum_type: The accumulator data type.

    Args:
        shape: The shape of the input tensor.
        init: The identity values for each reduction.
    """
    var row_size = shape[axis]
    var num_rows = shape.flattened_length() // row_size

    var global_dim_x = grid_dim.x * block_dim.x

    with PDL():
        # Loop over rows
        for row_idx in range(
            global_idx.x * simd_width,
            num_rows,
            global_dim_x * simd_width,
        ):
            # Reduce the whole row
            var row_coords = _get_nd_indices_from_flat_index(
                row_idx, shape, axis
            )

            # Declare & initialize registers
            var val = InlineArray[SIMD[accum_type, simd_width], num_reductions](
                uninitialized=True
            )

            comptime for i in range(num_reductions):
                val[i] = init[i].cast[accum_type]()

            # Load data & reduce
            for val_idx in range(row_size):
                row_coords[axis] = val_idx
                var t = input_fn[dtype, simd_width, rank](row_coords).cast[
                    accum_type
                ]()

                comptime for i in range(num_reductions):
                    val[i] = reduce_fn[reduction_idx=i](val[i], t)

            # Cast to output type
            var row_accum_cast = StaticTuple[
                SIMD[dtype, simd_width], num_reductions
            ]()

            comptime for i in range(num_reductions):
                row_accum_cast[i] = rebind[SIMD[dtype, simd_width]](
                    val[i].cast[dtype]()
                )

            # Write output
            row_coords[axis] = 0
            output_fn[dtype, simd_width, rank](row_coords, row_accum_cast)


def reduce_launch[
    num_reductions: Int,
    input_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank]
    ) capturing[_] -> SIMD[dtype, width],
    output_fn: def[dtype: DType, width: SIMDSize, rank: Int](
        IndexList[rank], StaticTuple[SIMD[dtype, width], num_reductions]
    ) capturing[_] -> None,
    reduce_fn: def[ty: DType, width: SIMDSize, reduction_idx: Int](
        SIMD[ty, width], SIMD[ty, width]
    ) capturing[_] -> SIMD[ty, width],
    rank: Int,
    dtype: DType,
](
    shape: IndexList[rank],
    axis: Int,
    init: StaticTuple[Scalar[dtype], num_reductions],
    ctx: DeviceContext,
) raises:
    """Selects and launches the appropriate GPU reduction kernel based on the
    tensor shape, axis, and device saturation level.

    Three-tier dispatch:
    1. Thread-saturated (many rows, non-contiguous axis): one row per thread
       via `saturated_reduce_kernel`.
    2. Block-saturated (enough rows to fill SMs at one block per row):
       `reduce_kernel` or `small_reduce_kernel`.
    3. Under-saturated (too few rows to fill the device): multiple blocks per
       row via `twophase_reduce_kernel` with a two-phase atomic finish.

    Parameters:
        num_reductions: The number of fused reductions to perform.
        input_fn: The lambda to load input elements.
        output_fn: The lambda to store output elements.
        reduce_fn: The binary reduction function.
        rank: The tensor rank.
        dtype: The data type of the elements.

    Args:
        shape: The shape of the input tensor.
        axis: The axis along which to reduce.
        init: The identity values for each reduction.
        ctx: The device context for GPU execution.

    Raises:
        If the GPU kernel launch fails.
    """
    comptime register_width = 32
    comptime sm_count = ctx.default_device_info.sm_count

    comptime packing_factor = 1

    var num_rows = shape.flattened_length() // shape[axis] // packing_factor
    comptime sm_overprovision_factor = 32  # tunable
    var num_blocks = min(num_rows, sm_overprovision_factor * sm_count)

    # Do not launch gpu kernels with grid_dim = 0
    if num_blocks == 0:
        return

    # 256 is a proxy for BLOCK_SIZE, because having BLOCK_SIZE affect kernel
    # selection is likely to confound autotuning.
    comptime num_persistent_threads = 256 * sm_count

    # enough rows that each thread can handle a whole row.
    var thread_saturated: Bool = num_rows >= num_persistent_threads
    # enough rows that each block/SM can handle a whole row.
    var block_saturated: Bool = num_rows >= sm_count
    # enough work to justify two-phase kernel
    comptime unsaturated_block_size = 128
    var more_values_than_threads: Bool = shape[axis] > unsaturated_block_size

    # This assumes row-major layout:
    var reduce_contig_dim: Bool = axis == rank - 1

    # --- Tier 1: Thread-saturated, non-contiguous axis ---
    # Each thread handles a whole row. SIMD packing across adjacent rows.
    if thread_saturated and not reduce_contig_dim:
        # TODO: a shape which *only just* saturates the device might be
        # more performant without SIMD, but the dispatch is more complicated
        comptime simd_packing_factor = simd_width_of[dtype, get_gpu_target()]()
        comptime BLOCK_SIZE = get_defined_int["MOJO_REDUCTION_BLOCK_SIZE", 32]()

        comptime for ax in range(rank):
            if axis == ax:
                comptime kernel = saturated_reduce_kernel[
                    rank,
                    ax,
                    num_reductions,
                    BLOCK_SIZE,
                    input_fn,
                    output_fn,
                    reduce_fn,
                    dtype,
                    simd_packing_factor,
                ]
                ctx.enqueue_function[kernel](
                    shape,
                    init,
                    grid_dim=num_blocks,
                    block_dim=BLOCK_SIZE,
                    attributes=pdl_launch_attributes(_PDL_LEVEL),
                )

    # --- Tier 3: Under-saturated ---
    # Too few rows to fill the device. Assign multiple blocks per row.
    # The canonical case here is a complete reduction (i.e. output size = 1)
    # Only use twophase when there are more_values_than_threads to
    # justify the overhead. Otherwise fall through to standard kernels.
    elif not block_saturated and more_values_than_threads:
        comptime BLOCK_SIZE = get_defined_int[
            "MOJO_REDUCTION_BLOCK_SIZE", unsaturated_block_size
        ]()

        # Round down to avoid a second wave
        var target_blocks = sm_count * sm_overprovision_factor
        var blocks_per_row = target_blocks // num_rows
        var total_blocks = num_rows * blocks_per_row
        comptime _accum_type = get_accum_type[dtype]()
        var partials_buf = ctx.enqueue_create_buffer[_accum_type](
            total_blocks * num_reductions
        )
        var counter_buf = ctx.enqueue_create_buffer[DType.int32](num_rows)
        ctx.enqueue_memset(counter_buf, Int32(0))

        comptime for ax in range(rank):
            if axis == ax:
                comptime kernel = twophase_reduce_kernel[
                    rank,
                    ax,
                    num_reductions,
                    BLOCK_SIZE,
                    input_fn,
                    output_fn,
                    reduce_fn,
                    dtype,
                    packing_factor,
                ]
                ctx.enqueue_function[kernel](
                    shape,
                    init,
                    partials_buf,
                    counter_buf,
                    blocks_per_row,
                    grid_dim=total_blocks,
                    block_dim=BLOCK_SIZE,
                    attributes=pdl_launch_attributes(_PDL_LEVEL),
                )

        _ = partials_buf
        _ = counter_buf

    # --- Tier 2: Block-saturated ---
    # Enough rows for one block per row. Standard cooperative reduction.
    else:
        comptime BLOCK_SIZE = get_defined_int[
            "MOJO_REDUCTION_BLOCK_SIZE", 128
        ]()
        if shape[axis] < WARP_SIZE:
            comptime for ax in range(rank):
                if axis == ax:
                    comptime kernel = small_reduce_kernel[
                        rank,
                        ax,
                        num_reductions,
                        BLOCK_SIZE,
                        input_fn,
                        output_fn,
                        reduce_fn,
                        dtype,
                        packing_factor,
                    ]
                    ctx.enqueue_function[kernel](
                        shape,
                        init,
                        grid_dim=num_blocks,
                        block_dim=BLOCK_SIZE,
                        attributes=pdl_launch_attributes(_PDL_LEVEL),
                    )
        else:
            comptime for ax in range(rank):
                if axis == ax:
                    comptime kernel = reduce_kernel[
                        rank,
                        ax,
                        num_reductions,
                        BLOCK_SIZE,
                        input_fn,
                        output_fn,
                        reduce_fn,
                        dtype,
                        packing_factor,
                    ]
                    ctx.enqueue_function[kernel](
                        shape,
                        init,
                        grid_dim=num_blocks,
                        block_dim=BLOCK_SIZE,
                        attributes=pdl_launch_attributes(_PDL_LEVEL),
                    )


@always_inline
def _reduce_generator_gpu[
    num_reductions: Int,
    init_type: DType,
    input_0_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank]
    ) capturing[_] -> SIMD[dtype, width],
    output_0_fn: def[dtype: DType, width: SIMDSize, rank: Int](
        IndexList[rank], StaticTuple[SIMD[dtype, width], num_reductions]
    ) capturing[_] -> None,
    reduce_function: def[ty: DType, width: SIMDSize, reduction_idx: Int](
        SIMD[ty, width], SIMD[ty, width]
    ) capturing[_] -> SIMD[ty, width],
    /,
](
    shape: IndexList[_, element_type=DType.int64],
    init: StaticTuple[Scalar[init_type], num_reductions],
    reduce_dim: Int,
    ctx: DeviceContext,
) raises:
    """Reduce the given tensor using the given reduction function on GPU. The
    num_reductions parameter enables callers to execute fused reductions. The
    reduce_0_fn and output_0_fn should be implemented in a way which routes
    between the fused reduction methods using their reduction_idx parameter.

    Parameters:
        num_reductions: The number of fused reductions to perform.
        init_type: The initial accumulator value for each reduction.
        input_0_fn: The lambda to use to access the incoming tensor.
        output_0_fn: The lambda to use to storing to the output tensor.
        reduce_function: The lambda implementing the reduction.

    Args:
        shape: The shape of the tensor we are reducing.
        init: The value to start the reduction from.
        reduce_dim: The dimension we are reducing.
        ctx: The pointer to DeviceContext.

    Raises:
        If the GPU kernel launch fails.
    """

    var reduce_dim_normalized = (
        len(shape) + reduce_dim
    ) if reduce_dim < 0 else reduce_dim

    reduce_launch[
        num_reductions,
        input_0_fn,
        output_0_fn,
        reduce_function,
        shape.size,
        init_type,
    ](shape, reduce_dim_normalized, init, ctx)
