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

from math import align_up

from algorithm.reduction import _get_nd_indices_from_flat_index
from gpu import (
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
    PDLLevel,
    launch_dependent_grids,
    wait_on_dependent_grids,
    AddressSpace,
)
from gpu.host import DeviceContext, get_gpu_target
from gpu.primitives import warp
from gpu.primitives.grid_controls import pdl_launch_attributes  # @doc_private
from memory import stack_allocation

from utils import IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple
from sys import env_get_int
from sys.info import simd_width_of


@always_inline
fn block_reduce[
    BLOCK_SIZE: Int,
    reduce_fn: fn[dtype: DType, width: Int](
        SIMD[dtype, width], SIMD[dtype, width]
    ) capturing[_] -> SIMD[dtype, width],
    dtype: DType,
    simd_width: Int,
](val: SIMD[dtype, simd_width], init: Scalar[dtype]) -> Scalar[dtype]:
    comptime num_reductions = 1

    @always_inline
    @parameter
    fn reduce_wrapper[
        dtype: DType, width: Int, reduction_idx: Int
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
fn block_reduce[
    BLOCK_SIZE: Int,
    num_reductions: Int,
    reduce_fn: fn[dtype: DType, width: Int, reduction_idx: Int](
        SIMD[dtype, width], SIMD[dtype, width]
    ) capturing[_] -> SIMD[dtype, width],
    dtype: DType,
    simd_width: Int,
](
    val: StaticTuple[SIMD[dtype, simd_width], num_reductions],
    init: StaticTuple[Scalar[dtype], num_reductions],
) -> StaticTuple[Scalar[dtype], num_reductions]:
    comptime assert (
        BLOCK_SIZE % WARP_SIZE == 0
    ), "block size must be a multiple of the warp size"

    @always_inline
    @parameter
    fn do_warp_reduce(
        val: StaticTuple[SIMD[dtype, simd_width], num_reductions]
    ) -> StaticTuple[SIMD[dtype, simd_width], num_reductions]:
        var result = StaticTuple[SIMD[dtype, simd_width], num_reductions]()

        comptime for i in range(num_reductions):

            @always_inline
            @parameter
            fn reduce_wrapper[
                dtype: DType, width: Int
            ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
                dtype, width
            ]:
                return reduce_fn[dtype, width, i](lhs, rhs)

            result[i] = warp.reduce[warp.shuffle_down, reduce_wrapper](val[i])

        return result

    var shared = stack_allocation[
        (BLOCK_SIZE // WARP_SIZE) * num_reductions * simd_width,
        dtype,
        address_space = AddressSpace.SHARED,
    ]()

    var warp = warp_id()

    var warp_accum = do_warp_reduce(val)

    if lane_id() == 0:
        comptime for i in range(num_reductions):
            # bank conflict for sub 4 byte data elems
            shared.store(
                (Int(warp) * num_reductions + i) * simd_width,
                warp_accum[i],
            )

    barrier()

    var last_accum = StaticTuple[SIMD[dtype, simd_width], num_reductions]()

    if thread_idx.x < (block_dim.x // UInt(WARP_SIZE)):
        comptime for i in range(num_reductions):
            last_accum[i] = shared.load[width=simd_width](
                (num_reductions * Int(lane_id()) + i) * simd_width
            )
    else:
        comptime for i in range(num_reductions):
            last_accum[i] = init[i]

    var result_packed = do_warp_reduce(last_accum)
    var result = StaticTuple[Scalar[dtype], num_reductions]()

    comptime for i in range(num_reductions):
        result[i] = result_packed[i].reduce[reduce_fn[dtype, reduction_idx=i]]()

    return result


@always_inline
fn row_reduce[
    BLOCK_SIZE: Int,
    input_fn: fn[dtype: DType, width: Int, rank: Int](
        IndexList[rank]
    ) capturing[_] -> SIMD[dtype, width],
    reduce_fn: fn[dtype: DType, width: Int](
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
    comptime num_reductions = 1

    @always_inline
    @parameter
    fn reduce_wrapper[
        dtype: DType, width: Int, reduction_idx: Int
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
fn row_reduce[
    BLOCK_SIZE: Int,
    num_reductions: Int,
    input_fn: fn[dtype: DType, width: Int, rank: Int](
        IndexList[rank]
    ) capturing[_] -> SIMD[dtype, width],
    reduce_fn: fn[dtype: DType, width: Int, reduction_idx: Int](
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
    var num_tail_values = row_size % simd_width
    var rounded_row_size = row_size - num_tail_values
    var row_size_padded = align_up(row_size // simd_width, BLOCK_SIZE)

    var accum = StaticTuple[SIMD[accum_type, simd_width], num_reductions]()
    var init_cast = StaticTuple[Scalar[accum_type], num_reductions]()

    comptime for i in range(num_reductions):
        init_cast[i] = init[i].cast[accum_type]()
        accum[i] = init_cast[i]

    var tid: UInt = thread_idx.x
    for offset_in_row in range(0, row_size_padded, BLOCK_SIZE):
        var idx_in_padded_row = (tid + UInt(offset_in_row)) * UInt(simd_width)

        if idx_in_padded_row >= UInt(rounded_row_size):
            break

        row_coords[axis] = Int(idx_in_padded_row)
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
fn reduce_kernel[
    rank: Int,
    axis: Int,
    num_reductions: Int,
    BLOCK_SIZE: Int,
    input_fn: fn[dtype: DType, width: Int, rank: Int](
        IndexList[rank]
    ) capturing[_] -> SIMD[dtype, width],
    output_fn: fn[dtype: DType, width: Int, rank: Int](
        IndexList[rank], StaticTuple[SIMD[dtype, width], num_reductions]
    ) capturing[_] -> None,
    reduce_fn: fn[ty: DType, width: Int, reduction_idx: Int](
        SIMD[ty, width], SIMD[ty, width]
    ) capturing[_] -> SIMD[ty, width],
    dtype: DType,
    simd_width: Int,
    accum_type: DType = get_accum_type[dtype](),
](shape: IndexList[rank], init: StaticTuple[Scalar[dtype], num_reductions],):
    var row_size = shape[axis]
    var num_rows = shape.flattened_length() // row_size

    comptime if PDLLevel() == PDLLevel.OVERLAP_AT_BEGINNING:
        launch_dependent_grids()

    comptime if PDLLevel() > PDLLevel.OFF:
        wait_on_dependent_grids()

    # grid stride loop over rows
    # each block reduces a row, which requires no partial reductions
    for row_idx in range(block_idx.x, UInt(num_rows), grid_dim.x):
        var row_coords = _get_nd_indices_from_flat_index(
            Int(row_idx), shape, axis
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
            var row_accum_cast = StaticTuple[Scalar[dtype], num_reductions]()

            comptime for i in range(num_reductions):
                row_accum_cast[i] = row_accum[i].cast[dtype]()

            row_coords[axis] = 0
            output_fn[dtype, 1, rank](row_coords, row_accum_cast)

    comptime if PDLLevel() == PDLLevel.OVERLAP_AT_END:
        launch_dependent_grids()


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
fn small_reduce_kernel[
    rank: Int,
    axis: Int,
    num_reductions: Int,
    BLOCK_SIZE: Int,
    input_fn: fn[dtype: DType, width: Int, rank: Int](
        IndexList[rank]
    ) capturing[_] -> SIMD[dtype, width],
    output_fn: fn[dtype: DType, width: Int, rank: Int](
        IndexList[rank], StaticTuple[SIMD[dtype, width], num_reductions]
    ) capturing[_] -> None,
    reduce_fn: fn[ty: DType, width: Int, reduction_idx: Int](
        SIMD[ty, width], SIMD[ty, width]
    ) capturing[_] -> SIMD[ty, width],
    dtype: DType,
    simd_width: Int,
    accum_type: DType = get_accum_type[dtype](),
](shape: IndexList[rank], init: StaticTuple[Scalar[dtype], num_reductions],):
    var row_size = shape[axis]
    var num_rows = shape.flattened_length() // row_size

    comptime if PDLLevel() == PDLLevel.OVERLAP_AT_BEGINNING:
        launch_dependent_grids()

    comptime if PDLLevel() > PDLLevel.OFF:
        wait_on_dependent_grids()

    comptime warps_per_block = BLOCK_SIZE // WARP_SIZE

    # grid stride loop over rows
    # each block reduces as many rows as warps,
    # No need to partial reduction because this is the degenerated case of
    # rows smaller than warp size
    #
    for row_idx in range(
        block_idx.x * UInt(warps_per_block),
        UInt(num_rows),
        grid_dim.x * UInt(warps_per_block),
    ):
        var row_coords = _get_nd_indices_from_flat_index(
            Int(row_idx) + Int(warp_id()), shape, axis
        )

        # One row per warp, warp collectively reads from global
        if warp_id() < UInt(warps_per_block):
            var val = InlineArray[SIMD[accum_type, simd_width], num_reductions](
                fill=0
            )

            comptime for i in range(num_reductions):
                val[i] = init[i].cast[accum_type]()

            if lane_id() < UInt(row_size):
                row_coords[axis] = Int(lane_id())
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
                fn reduce_wrapper[
                    dtype: DType, width: Int
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

    comptime if PDLLevel() == PDLLevel.OVERLAP_AT_END:
        launch_dependent_grids()


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
fn saturated_reduce_kernel[
    rank: Int,
    axis: Int,
    num_reductions: Int,
    BLOCK_SIZE: Int,
    input_fn: fn[dtype: DType, width: Int, rank: Int](
        IndexList[rank]
    ) capturing[_] -> SIMD[dtype, width],
    output_fn: fn[dtype: DType, width: Int, rank: Int](
        IndexList[rank], StaticTuple[SIMD[dtype, width], num_reductions]
    ) capturing[_] -> None,
    reduce_fn: fn[ty: DType, width: Int, reduction_idx: Int](
        SIMD[ty, width], SIMD[ty, width]
    ) capturing[_] -> SIMD[ty, width],
    dtype: DType,
    simd_width: Int,
    accum_type: DType = get_accum_type[dtype](),
](shape: IndexList[rank], init: StaticTuple[Scalar[dtype], num_reductions],):
    var row_size = shape[axis]
    var num_rows = shape.flattened_length() // row_size

    comptime if PDLLevel() == PDLLevel.OVERLAP_AT_BEGINNING:
        launch_dependent_grids()

    comptime if PDLLevel() > PDLLevel.OFF:
        wait_on_dependent_grids()

    var global_dim_x = grid_dim.x * block_dim.x
    # Loop over rows
    for row_idx in range(
        global_idx.x * UInt(simd_width),
        UInt(num_rows),
        global_dim_x * UInt(simd_width),
    ):
        # Reduce the whole row
        var row_coords = _get_nd_indices_from_flat_index(
            Int(row_idx), shape, axis
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

    comptime if PDLLevel() == PDLLevel.OVERLAP_AT_END:
        launch_dependent_grids()


fn reduce_launch[
    num_reductions: Int,
    input_fn: fn[dtype: DType, width: Int, rank: Int](
        IndexList[rank]
    ) capturing[_] -> SIMD[dtype, width],
    output_fn: fn[dtype: DType, width: Int, rank: Int](
        IndexList[rank], StaticTuple[SIMD[dtype, width], num_reductions]
    ) capturing[_] -> None,
    reduce_fn: fn[ty: DType, width: Int, reduction_idx: Int](
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
    var saturated: Bool = num_rows >= num_persistent_threads

    # This assumes row-major layout:
    var reduce_contig_dim: Bool = axis == rank - 1

    # If we have enough work to saturate the device, have each thread handle a
    # separate output value (i.e. a whole row), unless we are reducing the
    # contiguous (stride 1) dimension, in which case original approach coalesces
    # memory better.
    if saturated and not reduce_contig_dim:
        # If we are sufficiently above the saturation threshold, use SIMD to give
        # each thread multiple values per iter

        # TODO: a shape which *only just* saturates the device might be
        # more performant without SIMD, but the dispatch is more complicated
        comptime simd_packing_factor = simd_width_of[dtype, get_gpu_target()]()
        comptime BLOCK_SIZE = env_get_int["MOJO_REDUCTION_BLOCK_SIZE", 32]()

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
                ctx.enqueue_function[kernel, kernel](
                    shape,
                    init,
                    grid_dim=num_blocks,
                    block_dim=BLOCK_SIZE,
                    attributes=pdl_launch_attributes(),
                )

    # When the row size is smaller than the warp so we can use
    # multiple warps within a block to reduce rows and save shared memory sync
    else:
        comptime BLOCK_SIZE = env_get_int["MOJO_REDUCTION_BLOCK_SIZE", 128]()
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
                    ctx.enqueue_function[kernel, kernel](
                        shape,
                        init,
                        grid_dim=num_blocks,
                        block_dim=BLOCK_SIZE,
                        attributes=pdl_launch_attributes(),
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
                    ctx.enqueue_function[kernel, kernel](
                        shape,
                        init,
                        grid_dim=num_blocks,
                        block_dim=BLOCK_SIZE,
                        attributes=pdl_launch_attributes(),
                    )
