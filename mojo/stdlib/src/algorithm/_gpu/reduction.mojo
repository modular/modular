# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from algorithm.reduction import _get_nd_indices_from_flat_index
from math import align_up, min

from builtin.io import _printf
from gpu.memory import AddressSpace
from gpu import (
    ThreadIdx,
    BlockIdx,
    BlockDim,
    GridDim,
    barrier,
    lane_id,
    shuffle_down,
    WARP_SIZE,
    warp_reduce,
)
from gpu.host import Context, Dim, Function, Stream, Device, DeviceAttribute
from memory import stack_allocation


@always_inline
fn block_reduce[
    BLOCK_SIZE: Int,
    reduce_fn: fn[type: DType, width: Int] (
        SIMD[type, width], SIMD[type, width]
    ) capturing -> SIMD[type, width],
    type: DType,
](val: SIMD[type, 1], init: SIMD[type, 1]) -> SIMD[type, 1]:
    constrained[
        BLOCK_SIZE % WARP_SIZE == 0,
        "block size must be a multiple of the warp size",
    ]()
    let shared = stack_allocation[
        BLOCK_SIZE // WARP_SIZE,
        type,
        address_space = AddressSpace.SHARED,
    ]()

    let lane = lane_id()
    let warp = ThreadIdx.x() // WARP_SIZE

    let warp_accum = warp_reduce[shuffle_down, reduce_fn](val)

    if lane == 0:
        shared.store(
            warp, warp_accum
        )  # bank conflict for sub 4 byte data elems

    barrier()

    return warp_reduce[shuffle_down, reduce_fn](
        shared.load(lane) if ThreadIdx.x()
        < (BlockDim.x() // WARP_SIZE) else init
    )


@always_inline
fn row_reduce[
    BLOCK_SIZE: Int,
    input_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    reduce_fn: fn[type: DType, width: Int] (
        SIMD[type, width], SIMD[type, width]
    ) capturing -> SIMD[type, width],
    type: DType,
    rank: Int,
](
    inout row_coords: StaticIntTuple[rank],
    axis: Int,
    init: SIMD[type, 1],
    row_size: Int,
) -> SIMD[type, 1]:
    let row_size_padded = align_up(row_size, BLOCK_SIZE)
    var accum = init
    let tid = ThreadIdx.x()
    for offset_in_row in range(0, row_size_padded, BLOCK_SIZE):
        let idx_in_padded_row = tid + offset_in_row

        if idx_in_padded_row >= row_size:
            break

        row_coords[axis] = idx_in_padded_row
        let val = input_fn[type, 1, rank](row_coords)

        accum = reduce_fn(val, accum)

    return block_reduce[BLOCK_SIZE, reduce_fn](accum, init)


@__llvm_metadata(`nvvm.maxntid`=[BLOCK_SIZE])
fn reduce_kernel[
    rank: Int,
    BLOCK_SIZE: Int,
    input_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    output_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
    reduce_fn: fn[type: DType, width: Int] (
        SIMD[type, width], SIMD[type, width]
    ) capturing -> SIMD[type, width],
    type: DType,
](shape: StaticIntTuple[rank], axis: Int, init: SIMD[type, 1],):
    let row_size = shape[axis]
    let num_rows = shape.flattened_length() // row_size

    # grid stride loop over rows
    # each block reduces a row, which requires no partial reductions
    for row_idx in range(
        BlockIdx.x(),
        num_rows,
        GridDim.x(),
    ):
        var row_coords = _get_nd_indices_from_flat_index(row_idx, shape, axis)
        let row_accum = row_reduce[BLOCK_SIZE, input_fn, reduce_fn](
            row_coords, axis, init, row_size
        )

        if ThreadIdx.x() == 0:
            row_coords[axis] = 0
            output_fn(row_coords, row_accum)


fn reduce_launch[
    input_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    output_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
    reduce_fn: fn[type: DType, width: Int] (
        SIMD[type, width], SIMD[type, width]
    ) capturing -> SIMD[type, width],
    rank: Int,
    type: DType,
](
    shape: StaticIntTuple[rank],
    axis: Int,
    init: SIMD[type, 1],
    stream: Stream,
) raises:
    alias BLOCK_SIZE = 128
    let func = Function[
        fn (
            StaticIntTuple[rank],
            Int,
            SIMD[type, 1],
        ) capturing -> None, reduce_kernel[
            rank,
            BLOCK_SIZE,
            input_fn,
            output_fn,
            reduce_fn,
            type,
        ]
    ]()

    let num_rows = shape.flattened_length() // shape[axis]
    let sm_count = Device()._query(DeviceAttribute.MULTIPROCESSOR_COUNT)
    alias sm_overprovision_factor = 32  # tunable
    let num_blocks = min(num_rows, sm_overprovision_factor * sm_count)

    func(
        (num_blocks,),
        (BLOCK_SIZE,),
        shape,
        axis,
        init,
        stream=stream,
    )
