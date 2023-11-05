# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo %s | FileCheck %s

from math import div_ceil, exp, align_up, min, max
from math.limit import min_or_neginf
from pathlib import Path
from sys.info import triple_is_nvidia_cuda
from memory.buffer import NDBuffer
from memory import stack_allocation
from algorithm.reduction import _get_nd_indices_from_flat_index

from builtin.io import _printf
from gpu import (
    ThreadIdx,
    BlockIdx,
    BlockDim,
    GridDim,
    barrier,
    AddressSpace,
    lane_id,
    warp_id,
    shuffle_down,
    WARP_SIZE,
    warp_reduce,
)
from gpu.host import Context, Dim, Function, Stream, Device, DeviceAttribute
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
    _memset,
)
from tensor import Tensor

from utils.index import Index


@always_inline
fn block_reduce[
    BLOCK_SIZE: Int,
    reduce_fn: fn[type: DType, width: Int] (
        SIMD[type, width], SIMD[type, width]
    ) -> SIMD[type, width],
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

    let warp_accum = warp_reduce[type, shuffle_down, reduce_fn](val)

    if lane == 0:
        shared.store(
            warp, warp_accum
        )  # bank conflict for sub 4 byte data elems

    barrier()

    return warp_reduce[type, shuffle_down, reduce_fn](
        shared.load(lane) if ThreadIdx.x() < (BlockDim.x() // WARP_SIZE) else 0
    )


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
    ) -> SIMD[type, width],
    type: DType,
](shape: StaticIntTuple[rank], axis: Int, init: SIMD[type, 1],):
    @parameter
    if not triple_is_nvidia_cuda():
        return

    let row_size = shape[axis]
    let row_size_padded = align_up(row_size, BLOCK_SIZE)
    let num_rows = shape.flattened_length() // row_size

    # grid stride loop over rows
    # each block reduces a row, which requires no partial reductions
    for row_idx in range(
        BlockIdx.x(),
        num_rows,
        GridDim.x(),
    ):
        var accum = init
        var row_coords = _get_nd_indices_from_flat_index(row_idx, shape, axis)

        # thread block takes a row
        for block_in_row in range(0, row_size_padded // BLOCK_SIZE):
            let idx_in_padded_row = ThreadIdx.x() + block_in_row * BLOCK_SIZE

            row_coords[axis] = idx_in_padded_row
            let val = init if idx_in_padded_row >= row_size else input_fn[
                type, 1, rank
            ](row_coords)

            accum = reduce_fn(val, accum)

        barrier()

        let final_accum = block_reduce[BLOCK_SIZE, reduce_fn](accum, init)
        if ThreadIdx.x() == 0:
            row_coords[axis] = 0
            output_fn(row_coords, final_accum)


fn reduce_host_launch[
    input_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    output_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
    reduce_fn: fn[type: DType, width: Int] (
        SIMD[type, width], SIMD[type, width]
    ) -> SIMD[type, width],
    rank: Int,
    type: DType,
](
    shape: StaticIntTuple[rank],
    axis: Int,
    init: SIMD[type, 1],
    inout stream: Stream,
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
    ](verbose=True, dump_ptx=False)

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


fn reduce_inner_test[
    reduce_fn: fn[type: DType, width: Int] (
        SIMD[type, width], SIMD[type, width]
    ) -> SIMD[type, width],
    rank: Int,
    type: DType,
](shape: StaticIntTuple[rank], init: SIMD[type, 1]) raises:
    print("== run_inner_test")

    let axis = rank - 1
    var out_shape = shape
    out_shape[axis] = 1

    let in_size = shape.flattened_length()
    let out_size = shape.flattened_length() // shape[axis]

    var stream = Stream()

    var vec_host = Tensor[type](in_size)
    var res_host = Tensor[type](out_size)

    for i in range(in_size):
        vec_host[i] = i // shape[axis] + 1

    let vec_device = _malloc[type](in_size)
    let res_device = _malloc[type](out_size)
    let input_buf_device = NDBuffer[
        rank,
        DimList.create_unknown[rank](),
        type,
    ](vec_device, shape)
    let output_buf_device = NDBuffer[
        rank,
        DimList.create_unknown[rank](),
        type,
    ](res_device, out_shape)

    _copy_host_to_device(vec_device, vec_host.data(), in_size)

    @parameter
    fn input_fn[
        type: DType, width: Int, _rank: Int
    ](coords: StaticIntTuple[_rank]) -> SIMD[type, width]:
        return rebind[SIMD[type, width]](
            input_buf_device[rebind[StaticIntTuple[rank]](coords)]
        )

    @parameter
    fn output_fn[
        _type: DType, width: Int, _rank: Int
    ](coords: StaticIntTuple[_rank], val: SIMD[_type, width]):
        output_buf_device.__setitem__(
            rebind[StaticIntTuple[rank]](coords), rebind[SIMD[type, 1]](val)
        )

    reduce_host_launch[input_fn, output_fn, reduce_fn, rank, type](
        shape, axis, init, stream
    )

    _copy_device_to_host(res_host.data(), res_device, out_size)

    for i in range(out_shape.flattened_length()):
        print("res =", res_host[i])

    _free(vec_device)
    _free(res_device)

    _ = vec_host
    _ = res_host

    _ = stream ^


# CHECK-NOT: CUDA_ERROR
def main():
    @parameter
    @noncapturing
    fn reduce_add[
        type: DType,
        width: Int,
    ](x: SIMD[type, width], y: SIMD[type, width]) -> SIMD[type, width]:
        return x + y

    @parameter
    @noncapturing
    fn reduce_max[
        type: DType,
        width: Int,
    ](x: SIMD[type, width], y: SIMD[type, width]) -> SIMD[type, width]:
        return max(x, y)

    try:
        with Context() as ctx:
            # CHECK-LABEL: run_inner_test
            # CHECK: res = 257.0
            # CHECK: res = 514.0
            # CHECK: res = 771.0
            # CHECK: res = 1028.0
            # CHECK: res = 1285.0
            # CHECK: res = 1542.0

            reduce_inner_test[reduce_add](
                StaticIntTuple[3](2, 3, 257), Float32(0)
            )

            # CHECK-LABEL: run_inner_test
            # CHECK: res = 257.0
            # CHECK: res = 514.0
            # CHECK: res = 771.0
            # CHECK: res = 1028.0
            # CHECK: res = 1285.0
            reduce_inner_test[reduce_add](StaticIntTuple[2](5, 257), Float32(0))

            # CHECK-LABEL: run_inner_test
            # CHECK: res = 1029.0
            # CHECK: res = 2058.0
            # CHECK: res = 3087.0
            # CHECK: res = 4116.0
            # CHECK: res = 5145.0
            # CHECK: res = 6174.0
            # CHECK: res = 7203.0
            # CHECK: res = 8232.0
            reduce_inner_test[reduce_add](
                StaticIntTuple[4](2, 2, 2, 1029), Float32(0)
            )

            # CHECK-LABEL: run_inner_test
            # CHECK: res = 1.0
            # CHECK: res = 2.0
            # CHECK: res = 3.0
            # CHECK: res = 4.0
            # CHECK: res = 5.0
            reduce_inner_test[reduce_max](
                StaticIntTuple[2](5, 3), min_or_neginf[DType.float32]()
            )
    except e:
        print("CUDA_ERROR:", e)
