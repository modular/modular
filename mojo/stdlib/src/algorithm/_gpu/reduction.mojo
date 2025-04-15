# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
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
    warp_id,
    thread_idx,
)
from gpu.grid_controls import (
    pdl_launch_attributes,
    launch_dependent_grids,
    wait_on_dependent_grids,
    PDLLevel,
)
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from memory import stack_allocation

from utils import IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple
from sys import env_get_int


@always_inline
fn block_reduce[
    BLOCK_SIZE: Int,
    reduce_fn: fn[type: DType, width: Int] (
        SIMD[type, width], SIMD[type, width]
    ) capturing [_] -> SIMD[type, width],
    type: DType,
    simd_width: Int,
](val: SIMD[type, simd_width], init: Scalar[type]) -> Scalar[type]:
    alias num_reductions = 1

    @always_inline
    @parameter
    fn reduce_wrapper[
        type: DType, width: Int, reduction_idx: Int
    ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[type, width]:
        constrained[reduction_idx < num_reductions, "invalid reduction index"]()
        return reduce_fn(lhs, rhs)

    var val_tup = StaticTuple[SIMD[type, simd_width], num_reductions](val)
    var init_tup = StaticTuple[Scalar[type], num_reductions](init)

    return block_reduce[
        BLOCK_SIZE,
        num_reductions,
        reduce_wrapper,
        type,
        simd_width,
    ](val_tup, init_tup)[0]


@always_inline
fn block_reduce[
    BLOCK_SIZE: Int,
    num_reductions: Int,
    reduce_fn: fn[type: DType, width: Int, reduction_idx: Int] (
        SIMD[type, width], SIMD[type, width]
    ) capturing [_] -> SIMD[type, width],
    type: DType,
    simd_width: Int,
](
    val: StaticTuple[SIMD[type, simd_width], num_reductions],
    init: StaticTuple[Scalar[type], num_reductions],
) -> StaticTuple[Scalar[type], num_reductions]:
    constrained[
        BLOCK_SIZE % WARP_SIZE == 0,
        "block size must be a multiple of the warp size",
    ]()

    @always_inline
    @parameter
    fn do_warp_reduce(
        val: StaticTuple[SIMD[type, simd_width], num_reductions]
    ) -> StaticTuple[SIMD[type, simd_width], num_reductions]:
        var result = StaticTuple[SIMD[type, simd_width], num_reductions]()

        @parameter
        for i in range(num_reductions):

            @always_inline
            @parameter
            fn reduce_wrapper[
                type: DType, width: Int
            ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[
                type, width
            ]:
                return reduce_fn[type, width, i](lhs, rhs)

            result[i] = warp.reduce[warp.shuffle_down, reduce_wrapper](val[i])

        return result

    var shared = stack_allocation[
        (BLOCK_SIZE // WARP_SIZE) * num_reductions * simd_width,
        type,
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

    var last_accum = StaticTuple[SIMD[type, simd_width], num_reductions]()

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
    var result = StaticTuple[Scalar[type], num_reductions]()

    @parameter
    for i in range(num_reductions):

        @always_inline
        @parameter
        fn reduce_wrapper[
            type: DType, width: Int
        ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[type, width]:
            return reduce_fn[type, width, i](lhs, rhs)

        result[i] = result_packed[i].reduce[reduce_wrapper]()

    return result


@always_inline
fn row_reduce[
    BLOCK_SIZE: Int,
    input_fn: fn[type: DType, width: Int, rank: Int] (
        IndexList[rank]
    ) capturing [_] -> SIMD[type, width],
    reduce_fn: fn[type: DType, width: Int] (
        SIMD[type, width], SIMD[type, width]
    ) capturing [_] -> SIMD[type, width],
    type: DType,
    simd_width: Int,
    rank: Int,
    accum_type: DType = get_accum_type[type](),
](
    mut row_coords: IndexList[rank],
    axis: Int,
    init: Scalar[type],
    row_size: Int,
) -> Scalar[accum_type]:
    alias num_reductions = 1

    @always_inline
    @parameter
    fn reduce_wrapper[
        type: DType, width: Int, reduction_idx: Int
    ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[type, width]:
        constrained[reduction_idx < num_reductions, "invalid reduction index"]()
        return reduce_fn(lhs, rhs)

    var init_tup = StaticTuple[Scalar[type], num_reductions](init)

    return row_reduce[
        BLOCK_SIZE,
        num_reductions,
        input_fn,
        reduce_wrapper,
        type,
        simd_width,
        rank,
        accum_type=accum_type,
    ](row_coords, axis, init_tup, row_size)[0]


@always_inline
fn row_reduce[
    BLOCK_SIZE: Int,
    num_reductions: Int,
    input_fn: fn[type: DType, width: Int, rank: Int] (
        IndexList[rank]
    ) capturing [_] -> SIMD[type, width],
    reduce_fn: fn[type: DType, width: Int, reduction_idx: Int] (
        SIMD[type, width], SIMD[type, width]
    ) capturing [_] -> SIMD[type, width],
    type: DType,
    simd_width: Int,
    rank: Int,
    accum_type: DType = get_accum_type[type](),
](
    mut row_coords: IndexList[rank],
    axis: Int,
    init: StaticTuple[Scalar[type], num_reductions],
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
        var val = input_fn[type, simd_width, rank](row_coords).cast[
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
        var val = input_fn[type, 1, rank](row_coords).cast[accum_type]()

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
    input_fn: fn[type: DType, width: Int, rank: Int] (
        IndexList[rank]
    ) capturing [_] -> SIMD[type, width],
    output_fn: fn[type: DType, width: Int, rank: Int] (
        IndexList[rank], StaticTuple[SIMD[type, width], num_reductions]
    ) capturing [_] -> None,
    reduce_fn: fn[ty: DType, width: Int, reduction_idx: Int] (
        SIMD[ty, width], SIMD[ty, width]
    ) capturing [_] -> SIMD[ty, width],
    type: DType,
    simd_width: Int,
    accum_type: DType = get_accum_type[type](),
](
    shape: IndexList[rank],
    axis: Int,
    init: StaticTuple[Scalar[type], num_reductions],
):
    var row_size = shape[axis]
    var num_rows = shape.flattened_length() // row_size

    alias pdl_level = PDLLevel(env_get_int["PDL_LEVEL", 1]())

    @parameter
    if pdl_level == PDLLevel.OVERLAP_AT_BEGINNING:
        launch_dependent_grids()

    @parameter
    if pdl_level > PDLLevel.OFF:
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
            type,
            simd_width,
            rank,
            accum_type=accum_type,
        ](row_coords, axis, init, row_size)

        if thread_idx.x == 0:
            var row_accum_cast = StaticTuple[Scalar[type], num_reductions]()

            @parameter
            for i in range(num_reductions):
                row_accum_cast[i] = row_accum[i].cast[type]()

            row_coords[axis] = 0
            output_fn[type, 1, rank](row_coords, row_accum_cast)

    @parameter
    if pdl_level == PDLLevel.OVERLAP_AT_END:
        launch_dependent_grids()


fn reduce_launch[
    num_reductions: Int,
    input_fn: fn[type: DType, width: Int, rank: Int] (
        IndexList[rank]
    ) capturing [_] -> SIMD[type, width],
    output_fn: fn[type: DType, width: Int, rank: Int] (
        IndexList[rank], StaticTuple[SIMD[type, width], num_reductions]
    ) capturing [_] -> None,
    reduce_fn: fn[ty: DType, width: Int, reduction_idx: Int] (
        SIMD[ty, width], SIMD[ty, width]
    ) capturing [_] -> SIMD[ty, width],
    rank: Int,
    type: DType,
](
    shape: IndexList[rank],
    axis: Int,
    init: StaticTuple[Scalar[type], num_reductions],
    ctx: DeviceContext,
) raises:
    alias BLOCK_SIZE = 128
    alias register_width = 32
    alias sm_count = ctx.device_info.sm_count

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
            type,
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
