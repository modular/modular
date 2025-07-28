# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

import gpu.warp as warp
from algorithm.reduction import _get_nd_indices_from_flat_index
from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    block_dim,
    block_idx,
    grid_dim,
    lane_id,
    thread_idx,
    warp_id,
)
from gpu.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from memory import stack_allocation

from utils import IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple


@always_inline
fn block_reduce[
    BLOCK_SIZE: Int,
    reduce_fn: fn[dtype: DType, width: Int] (
        SIMD[dtype, width], SIMD[dtype, width]
    ) capturing [_] -> SIMD[dtype, width],
    dtype: DType,
    simd_width: Int,
](val: SIMD[dtype, simd_width], init: Scalar[dtype]) -> Scalar[dtype]:
    alias num_reductions = 1

    @always_inline
    @parameter
    fn reduce_wrapper[
        dtype: DType, width: Int, reduction_idx: Int
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        constrained[reduction_idx < num_reductions, "invalid reduction index"]()
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
    reduce_fn: fn[dtype: DType, width: Int, reduction_idx: Int] (
        SIMD[dtype, width], SIMD[dtype, width]
    ) capturing [_] -> SIMD[dtype, width],
    dtype: DType,
    simd_width: Int,
](
    val: StaticTuple[SIMD[dtype, simd_width], num_reductions],
    init: StaticTuple[Scalar[dtype], num_reductions],
) -> StaticTuple[Scalar[dtype], num_reductions]:
    constrained[
        BLOCK_SIZE % WARP_SIZE == 0,
        "block size must be a multiple of the warp size",
    ]()

    @always_inline
    @parameter
    fn do_warp_reduce(
        val: StaticTuple[SIMD[dtype, simd_width], num_reductions]
    ) -> StaticTuple[SIMD[dtype, simd_width], num_reductions]:
        var result = StaticTuple[SIMD[dtype, simd_width], num_reductions]()

        @parameter
        for i in range(num_reductions):

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

        @parameter
        for i in range(num_reductions):
            # bank conflict for sub 4 byte data elems
            shared.store(
                (Int(warp) * num_reductions + i) * simd_width,
                warp_accum[i],
            )

    barrier()

    var last_accum = StaticTuple[SIMD[dtype, simd_width], num_reductions]()

    if thread_idx.x < (block_dim.x // WARP_SIZE):

        @parameter
        for i in range(num_reductions):
            last_accum[i] = shared.load[width=simd_width](
                (num_reductions * lane_id() + i) * simd_width
            )
    else:

        @parameter
        for i in range(num_reductions):
            last_accum[i] = init[i]

    var result_packed = do_warp_reduce(last_accum)
    var result = StaticTuple[Scalar[dtype], num_reductions]()

    @parameter
    for i in range(num_reductions):
        result[i] = result_packed[i].reduce[reduce_fn[dtype, reduction_idx=i]]()

    return result


@always_inline
fn row_reduce[
    BLOCK_SIZE: Int,
    input_fn: fn[dtype: DType, width: Int, rank: Int] (
        IndexList[rank]
    ) capturing [_] -> SIMD[dtype, width],
    reduce_fn: fn[dtype: DType, width: Int] (
        SIMD[dtype, width], SIMD[dtype, width]
    ) capturing [_] -> SIMD[dtype, width],
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
    alias num_reductions = 1

    @always_inline
    @parameter
    fn reduce_wrapper[
        dtype: DType, width: Int, reduction_idx: Int
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        constrained[reduction_idx < num_reductions, "invalid reduction index"]()
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
    input_fn: fn[dtype: DType, width: Int, rank: Int] (
        IndexList[rank]
    ) capturing [_] -> SIMD[dtype, width],
    reduce_fn: fn[dtype: DType, width: Int, reduction_idx: Int] (
        SIMD[dtype, width], SIMD[dtype, width]
    ) capturing [_] -> SIMD[dtype, width],
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

    @parameter
    for i in range(num_reductions):
        init_cast[i] = init[i].cast[accum_type]()
        accum[i] = init_cast[i]

    var tid: UInt = thread_idx.x
    for offset_in_row in range(0, row_size_padded, BLOCK_SIZE):
        var idx_in_padded_row: UInt = (tid + offset_in_row) * simd_width

        if idx_in_padded_row >= UInt(rounded_row_size):
            break

        row_coords[axis] = Int(idx_in_padded_row)
        var val = input_fn[dtype, simd_width, rank](row_coords).cast[
            accum_type
        ]()

        @parameter
        for i in range(num_reductions):
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

        @parameter
        for i in range(num_reductions):
            scalar_accum[i] = reduce_fn[accum_type, 1, i](val, scalar_accum[i])

    return scalar_accum


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BLOCK_SIZE)
)
fn reduce_kernel[
    rank: Int,
    num_reductions: Int,
    BLOCK_SIZE: Int,
    input_fn: fn[dtype: DType, width: Int, rank: Int] (
        IndexList[rank]
    ) capturing [_] -> SIMD[dtype, width],
    output_fn: fn[dtype: DType, width: Int, rank: Int] (
        IndexList[rank], StaticTuple[SIMD[dtype, width], num_reductions]
    ) capturing [_] -> None,
    reduce_fn: fn[ty: DType, width: Int, reduction_idx: Int] (
        SIMD[ty, width], SIMD[ty, width]
    ) capturing [_] -> SIMD[ty, width],
    dtype: DType,
    simd_width: Int,
    accum_type: DType = get_accum_type[dtype](),
](
    shape: IndexList[rank],
    axis: Int,
    init: StaticTuple[Scalar[dtype], num_reductions],
):
    var row_size = shape[axis]
    var num_rows = shape.flattened_length() // row_size

    @parameter
    if PDLLevel() == PDLLevel.OVERLAP_AT_BEGINNING:
        launch_dependent_grids()

    @parameter
    if PDLLevel() > PDLLevel.OFF:
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

            @parameter
            for i in range(num_reductions):
                row_accum_cast[i] = row_accum[i].cast[dtype]()

            row_coords[axis] = 0
            output_fn[dtype, 1, rank](row_coords, row_accum_cast)

    @parameter
    if PDLLevel() == PDLLevel.OVERLAP_AT_END:
        launch_dependent_grids()


fn reduce_launch[
    num_reductions: Int,
    input_fn: fn[dtype: DType, width: Int, rank: Int] (
        IndexList[rank]
    ) capturing [_] -> SIMD[dtype, width],
    output_fn: fn[dtype: DType, width: Int, rank: Int] (
        IndexList[rank], StaticTuple[SIMD[dtype, width], num_reductions]
    ) capturing [_] -> None,
    reduce_fn: fn[ty: DType, width: Int, reduction_idx: Int] (
        SIMD[ty, width], SIMD[ty, width]
    ) capturing [_] -> SIMD[ty, width],
    rank: Int,
    dtype: DType,
](
    shape: IndexList[rank],
    axis: Int,
    init: StaticTuple[Scalar[dtype], num_reductions],
    ctx: DeviceContext,
) raises:
    alias BLOCK_SIZE = 128
    alias register_width = 32
    alias sm_count = ctx.default_device_info.sm_count

    alias packing_factor = 1

    var num_rows = shape.flattened_length() // shape[axis] // packing_factor
    alias sm_overprovision_factor = 32  # tunable
    var num_blocks = min(num_rows, sm_overprovision_factor * sm_count)

    ctx.enqueue_function[
        reduce_kernel[
            rank,
            num_reductions,
            BLOCK_SIZE,
            input_fn,
            output_fn,
            reduce_fn,
            dtype,
            packing_factor,
        ]
    ](
        shape,
        axis,
        init,
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE,
        attributes=pdl_launch_attributes(),
    )
