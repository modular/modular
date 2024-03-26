# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo %s

from math import align_up, div_ceil, exp, max, min
from pathlib import Path

from algorithm._gpu.reduction import reduce_launch
from buffer import NDBuffer
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
from testing import assert_equal

from utils.index import Index

alias num_reductions = 2


fn fused_reduce_inner_test[
    reduce_fn: fn[ty: DType, width: Int, reduction_idx: Int] (
        SIMD[ty, width], SIMD[ty, width]
    ) capturing -> SIMD[ty, width],
    rank: Int,
    type: DType,
](
    shape: StaticIntTuple[rank],
    init: StaticTuple[Scalar[type], num_reductions],
    expected_vals0: List[Float32],
    expected_vals1: List[Float32],
) raises:
    var axis = rank - 1
    var out_shape = shape
    out_shape[axis] = 1

    var in_size = shape.flattened_length()
    var out_size = out_shape.flattened_length()

    debug_assert(
        len(expected_vals0) == out_size,
        "expected vals must match output shape",
    )
    debug_assert(
        len(expected_vals1) == out_size,
        "expected vals must match output shape",
    )
    var stream = Stream()

    var vec_host = Tensor[type](in_size)
    var res_host0 = Tensor[type](out_size)
    var res_host1 = Tensor[type](out_size)

    for i in range(in_size):
        vec_host[i] = i // shape[axis] + 1

    var vec_device = _malloc[type](in_size)
    var res_device0 = _malloc[type](out_size)
    var res_device1 = _malloc[type](out_size)
    var input_buf_device = NDBuffer[type, rank](vec_device, shape)
    var output_buf_device0 = NDBuffer[type, rank](res_device0, out_shape)
    var output_buf_device1 = NDBuffer[type, rank](res_device1, out_shape)

    _copy_host_to_device(vec_device, vec_host.data(), in_size)

    @__copy_capture(input_buf_device)
    @parameter
    fn input_fn[
        type: DType,
        width: Int,
        _rank: Int,
    ](coords: StaticIntTuple[_rank]) -> SIMD[type, width]:
        return rebind[SIMD[type, width]](
            input_buf_device.load[width=width](
                rebind[StaticIntTuple[rank]](coords)
            )
        )

    @__copy_capture(output_buf_device0, output_buf_device1)
    @parameter
    fn output_fn[
        _type: DType, width: Int, _rank: Int
    ](
        coords: StaticIntTuple[_rank],
        val: StaticTuple[SIMD[_type, width], num_reductions],
    ):
        output_buf_device0.__setitem__(
            rebind[StaticIntTuple[rank]](coords), rebind[Scalar[type]](val[0])
        )
        output_buf_device1.__setitem__(
            rebind[StaticIntTuple[rank]](coords), rebind[Scalar[type]](val[1])
        )

    reduce_launch[num_reductions, input_fn, output_fn, reduce_fn, rank, type](
        shape, axis, init, stream
    )

    stream.synchronize()
    _copy_device_to_host(res_host0.data(), res_device0, out_size)
    _copy_device_to_host(res_host1.data(), res_device1, out_size)

    for i in range(out_shape.flattened_length()):
        assert_equal(res_host0[i], expected_vals0[i])

    for i in range(out_shape.flattened_length()):
        assert_equal(res_host1[i], expected_vals1[i])

    _free(vec_device)
    _free(res_device0)
    _free(res_device1)

    _ = vec_host
    _ = res_host0
    _ = res_host1

    _ = stream^


fn reduce_inner_test[
    reduce_fn: fn[type: DType, width: Int] (
        SIMD[type, width], SIMD[type, width]
    ) capturing -> SIMD[type, width],
    rank: Int,
    type: DType,
](
    shape: StaticIntTuple[rank],
    init: Scalar[type],
    expected_vals: List[Float32],
) raises:
    alias num_reductions = 1

    var axis = rank - 1
    var out_shape = shape
    out_shape[axis] = 1

    var in_size = shape.flattened_length()
    var out_size = shape.flattened_length() // shape[axis]
    debug_assert(
        len(expected_vals) == out_size, "expected vals must match output shape"
    )

    var stream = Stream()

    var vec_host = Tensor[type](in_size)
    var res_host = Tensor[type](out_size)

    for i in range(in_size):
        vec_host[i] = i // shape[axis] + 1

    var vec_device = _malloc[type](in_size)
    var res_device = _malloc[type](out_size)
    var input_buf_device = NDBuffer[type, rank](vec_device, shape)
    var output_buf_device = NDBuffer[type, rank](res_device, out_shape)

    _copy_host_to_device(vec_device, vec_host.data(), in_size)

    @always_inline
    @parameter
    fn reduce_wrapper[
        type: DType, width: Int, reduction_idx: Int
    ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[type, width]:
        constrained[reduction_idx < num_reductions, "invalid reduction idx"]()

        return reduce_fn[type, width](lhs, rhs)

    @__copy_capture(input_buf_device)
    @parameter
    fn input_fn[
        type: DType,
        width: Int,
        _rank: Int,
    ](coords: StaticIntTuple[_rank]) -> SIMD[type, width]:
        return rebind[SIMD[type, width]](
            input_buf_device.load[width=width](
                rebind[StaticIntTuple[rank]](coords)
            )
        )

    @__copy_capture(output_buf_device)
    @parameter
    fn output_fn[
        _type: DType, width: Int, _rank: Int
    ](
        coords: StaticIntTuple[_rank],
        val: StaticTuple[SIMD[_type, width], num_reductions],
    ):
        output_buf_device.__setitem__(
            rebind[StaticIntTuple[rank]](coords), rebind[Scalar[type]](val[0])
        )

    reduce_launch[
        num_reductions, input_fn, output_fn, reduce_wrapper, rank, type
    ](shape, axis, init, stream)

    stream.synchronize()
    _copy_device_to_host(res_host.data(), res_device, out_size)

    for i in range(out_shape.flattened_length()):
        assert_equal(res_host[i], expected_vals[i])

    _free(vec_device)
    _free(res_device)

    _ = vec_host
    _ = res_host

    _ = stream^


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

    @parameter
    fn fused_reduce_add_max[
        type: DType,
        width: Int,
        reduction_idx: Int,
    ](x: SIMD[type, width], y: SIMD[type, width]) -> SIMD[type, width]:
        constrained[reduction_idx < 2, "reduction idx OOB"]()

        alias func = reduce_max if reduction_idx == 0 else reduce_add
        return func(x, y)

    try:
        with Context() as ctx:
            reduce_inner_test[reduce_add](
                StaticIntTuple[3](2, 3, 257),
                Float32(0),
                List[Float32](257.0, 514.0, 771.0, 1028.0, 1285.0, 1542.0),
            )

            reduce_inner_test[reduce_add](
                StaticIntTuple[2](5, 257),
                Float32(0),
                List[Float32](257.0, 514.0, 771.0, 1028.0, 1285.0),
            )

            reduce_inner_test[reduce_add](
                StaticIntTuple[4](2, 2, 2, 1029),
                Float32(0),
                List[Float32](
                    1029.0,
                    2058.0,
                    3087.0,
                    4116.0,
                    5145.0,
                    6174.0,
                    7203.0,
                    8232.0,
                ),
            )

            reduce_inner_test[reduce_max](
                StaticIntTuple[2](5, 3),
                Scalar[DType.float32].MIN,
                List[Float32](1.0, 2.0, 3.0, 4.0, 5.0),
            )

            fused_reduce_inner_test[fused_reduce_add_max, 2, DType.float32](
                StaticIntTuple[2](5, 3),
                StaticTuple[Scalar[DType.float32], 2](
                    Scalar[DType.float32].MIN, 0.0
                ),
                List[Float32](1.0, 2.0, 3.0, 4.0, 5.0),
                List[Float32](3.0, 6.0, 9.0, 12.0, 15.0),
            )

            # bf16 tests
            reduce_inner_test[reduce_max](
                StaticIntTuple[2](5, 5),
                BFloat16.MIN,
                List[Float32](1.0, 2.0, 3.0, 4.0, 5.0),
            )

            fused_reduce_inner_test[fused_reduce_add_max, 2, DType.bfloat16](
                StaticIntTuple[2](5, 3),
                StaticTuple[BFloat16, 2](BFloat16.MIN, 0.0),
                List[Float32](1.0, 2.0, 3.0, 4.0, 5.0),
                List[Float32](3.0, 6.0, 9.0, 12.0, 15.0),
            )

            # fp16 tests
            reduce_inner_test[reduce_max](
                StaticIntTuple[2](5, 5),
                Float16.MIN,
                List[Float32](1.0, 2.0, 3.0, 4.0, 5.0),
            )

            fused_reduce_inner_test[fused_reduce_add_max, 2, DType.float16](
                StaticIntTuple[2](5, 3),
                StaticTuple[Float16, 2](Float16.MIN, 0.0),
                List[Float32](1.0, 2.0, 3.0, 4.0, 5.0),
                List[Float32](3.0, 6.0, 9.0, 12.0, 15.0),
            )
    except e:
        print("CUDA_ERROR:", e)
