# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import align_up, min

from algorithm.reduction import _get_nd_indices_from_flat_index
from builtin.io import _printf
from gpu import (
    WARP_SIZE,
    BlockDim,
    BlockIdx,
    GridDim,
    ThreadIdx,
    barrier,
    lane_id,
    shuffle_down,
    warp_reduce,
)
from gpu.host import Context, Device, DeviceAttribute, Dim, Function, Stream
from gpu.memory import AddressSpace
from memory import stack_allocation

from utils.static_tuple import StaticTuple


@always_inline
fn block_reduce[
    BLOCK_SIZE: Int,
    reduce_fn: fn[type: DType, width: Int] (
        SIMD[type, width], SIMD[type, width]
    ) capturing -> SIMD[type, width],
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
    ) capturing -> SIMD[type, width],
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

        @always_inline
        @parameter
        fn unrolled_warp_reduce_helper[i: Int]():
            @always_inline
            @parameter
            fn reduce_wrapper[
                type: DType, width: Int
            ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[
                type, width
            ]:
                return reduce_fn[type, width, i](lhs, rhs)

            result[i] = warp_reduce[shuffle_down, reduce_wrapper](val[i])

        unroll[unrolled_warp_reduce_helper, num_reductions]()
        return result

    var shared = stack_allocation[
        (BLOCK_SIZE // WARP_SIZE) * num_reductions * simd_width,
        type,
        address_space = AddressSpace.SHARED,
    ]()

    var warp = ThreadIdx.x() // WARP_SIZE

    var warp_accum = do_warp_reduce(val)

    if lane_id() == 0:

        @unroll
        for i in range(num_reductions):
            # bank conflict for sub 4 byte data elems
            shared.store[width=simd_width](
                (warp * num_reductions + i) * simd_width, warp_accum[i]
            )

    barrier()

    var last_accum = StaticTuple[SIMD[type, simd_width], num_reductions]()

    if ThreadIdx.x() < (BlockDim.x() // WARP_SIZE):

        @unroll
        for i in range(num_reductions):
            last_accum[i] = shared.load[width=simd_width](
                (num_reductions * lane_id() + i) * simd_width
            )
    else:

        @unroll
        for i in range(num_reductions):
            last_accum[i] = init[i]

    var result_packed = do_warp_reduce(last_accum)
    var result = StaticTuple[Scalar[type], num_reductions]()

    @always_inline
    @parameter
    fn unrolled_simd_reduce_helper[i: Int]():
        @always_inline
        @parameter
        fn reduce_wrapper[
            type: DType, width: Int
        ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[type, width]:
            return reduce_fn[type, width, i](lhs, rhs)

        result[i] = result_packed[i].reduce[reduce_wrapper]()

    unroll[unrolled_simd_reduce_helper, num_reductions]()

    return result


@always_inline
fn row_reduce[
    BLOCK_SIZE: Int,
    input_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    reduce_fn: fn[type: DType, width: Int] (
        SIMD[type, width], SIMD[type, width]
    ) capturing -> SIMD[type, width],
    accum_type: DType,
    type: DType,
    simd_width: Int,
    rank: Int,
](
    inout row_coords: StaticIntTuple[rank],
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
        accum_type,
        type,
        simd_width,
        rank,
    ](row_coords, axis, init_tup, row_size)[0]


@always_inline
fn row_reduce[
    BLOCK_SIZE: Int,
    num_reductions: Int,
    input_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    reduce_fn: fn[type: DType, width: Int, reduction_idx: Int] (
        SIMD[type, width], SIMD[type, width]
    ) capturing -> SIMD[type, width],
    accum_type: DType,
    type: DType,
    simd_width: Int,
    rank: Int,
](
    inout row_coords: StaticIntTuple[rank],
    axis: Int,
    init: StaticTuple[Scalar[type], num_reductions],
    row_size: Int,
) -> StaticTuple[Scalar[accum_type], num_reductions]:
    var num_tail_values = row_size % simd_width
    var rounded_row_size = row_size - num_tail_values
    var row_size_padded = align_up(row_size // simd_width, BLOCK_SIZE)

    var accum = StaticTuple[SIMD[accum_type, simd_width], num_reductions]()
    var init_cast = StaticTuple[Scalar[accum_type], num_reductions]()

    @unroll
    for i in range(num_reductions):
        init_cast[i] = init[i].cast[accum_type]()
        accum[i] = init_cast[i]

    var tid = ThreadIdx.x()
    for offset_in_row in range(0, row_size_padded, BLOCK_SIZE):
        var idx_in_padded_row = (tid + offset_in_row) * simd_width

        if idx_in_padded_row >= rounded_row_size:
            break

        row_coords[axis] = idx_in_padded_row
        var val = input_fn[type, simd_width, rank](row_coords).cast[
            accum_type
        ]()

        @always_inline
        @__copy_capture(val)
        @parameter
        fn unrolled_reduce_wrapper[i: Int]():
            accum[i] = reduce_fn[accum_type, simd_width, i](val, accum[i])

        unroll[unrolled_reduce_wrapper, num_reductions]()

    var scalar_accum = block_reduce[
        BLOCK_SIZE, num_reductions, reduce_fn, accum_type, simd_width
    ](accum, init_cast)

    # handle trailing values
    for idx_in_padded_row in range(rounded_row_size, row_size):
        row_coords[axis] = idx_in_padded_row
        var val = input_fn[type, 1, rank](row_coords).cast[accum_type]()

        @always_inline
        @__copy_capture(val)
        @parameter
        fn unrolled_scalar_reduce_wrapper[i: Int]():
            scalar_accum[i] = reduce_fn[accum_type, 1, i](val, scalar_accum[i])

        unroll[unrolled_scalar_reduce_wrapper, num_reductions]()

    return scalar_accum


@__llvm_metadata(`nvvm.maxntid`=StaticTuple[Int32, 1](BLOCK_SIZE))
fn reduce_kernel[
    rank: Int,
    num_reductions: Int,
    BLOCK_SIZE: Int,
    input_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    output_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank], StaticTuple[SIMD[type, width], num_reductions]
    ) capturing -> None,
    reduce_fn: fn[ty: DType, width: Int, reduction_idx: Int] (
        SIMD[ty, width], SIMD[ty, width]
    ) capturing -> SIMD[ty, width],
    type: DType,
    simd_width: Int,
    accum_type: DType = type,
](
    shape: StaticIntTuple[rank],
    axis: Int,
    init: StaticTuple[Scalar[type], num_reductions],
):
    var row_size = shape[axis]
    var num_rows = shape.flattened_length() // row_size

    # grid stride loop over rows
    # each block reduces a row, which requires no partial reductions
    for row_idx in range(
        BlockIdx.x(),
        num_rows,
        GridDim.x(),
    ):
        var row_coords = _get_nd_indices_from_flat_index(row_idx, shape, axis)

        var row_accum = row_reduce[
            BLOCK_SIZE,
            num_reductions,
            input_fn,
            reduce_fn,
            accum_type,
            type,
            simd_width,
            rank,
        ](row_coords, axis, init, row_size)

        if ThreadIdx.x() == 0:
            var row_accum_cast = StaticTuple[Scalar[type], num_reductions]()

            @unroll
            for i in range(num_reductions):
                row_accum_cast[i] = row_accum[i].cast[type]()

            row_coords[axis] = 0
            output_fn[type, 1, rank](row_coords, row_accum_cast)


fn reduce_launch[
    num_reductions: Int,
    input_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    output_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank], StaticTuple[SIMD[type, width], num_reductions]
    ) capturing -> None,
    reduce_fn: fn[ty: DType, width: Int, reduction_idx: Int] (
        SIMD[ty, width], SIMD[ty, width]
    ) capturing -> SIMD[ty, width],
    rank: Int,
    type: DType,
](
    shape: StaticIntTuple[rank],
    axis: Int,
    init: StaticTuple[Scalar[type], num_reductions],
    stream: Stream,
) raises:
    alias BLOCK_SIZE = 128
    alias register_width = 32

    alias packing_factor = 1
    alias accum_type = DType.float32 if type.is_bfloat16() or type.is_float16() else type

    var func = Function[
        fn (
            StaticIntTuple[rank],
            Int,
            StaticTuple[Scalar[type], num_reductions],
        ) capturing -> None, reduce_kernel[
            rank,
            num_reductions,
            BLOCK_SIZE,
            input_fn,
            output_fn,
            reduce_fn,
            type,
            packing_factor,
        ]
    ]()

    var num_rows = shape.flattened_length() // shape[axis] // packing_factor
    var sm_count = Device()._query(DeviceAttribute.MULTIPROCESSOR_COUNT)
    alias sm_overprovision_factor = 32  # tunable
    var num_blocks = min(num_rows, sm_overprovision_factor * sm_count)

    func(
        shape,
        axis,
        init,
        grid_dim=(num_blocks,),
        block_dim=(BLOCK_SIZE,),
        stream=stream,
    )
