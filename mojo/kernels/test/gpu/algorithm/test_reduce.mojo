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
from memory.buffer import NDBuffer
from algorithm._gpu.reduction import reduce_launch

from builtin.io import _printf
from gpu.host import Context, Stream
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
    _memset,
)
from tensor import Tensor

from utils.index import Index


fn reduce_inner_test[
    reduce_fn: fn[type: DType, width: Int] (
        SIMD[type, width], SIMD[type, width]
    ) capturing -> SIMD[type, width],
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

    reduce_launch[input_fn, output_fn, reduce_fn, rank, type](
        shape, axis, init, stream
    )

    stream.synchronize()
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
    fn reduce_add[
        type: DType,
        width: Int,
    ](x: SIMD[type, width], y: SIMD[type, width]) -> SIMD[type, width]:
        return x + y

    @parameter
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
