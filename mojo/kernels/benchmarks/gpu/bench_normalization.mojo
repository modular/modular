# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# REQUIRES: has_cuda_device
# RUN: %mojo-build-no-debug %s

from random import random_float64
from nn.normalization import layer_norm_gpu, rms_norm_gpu
from benchmark import Bench, BenchConfig, Bencher, BenchId
from gpu.host import DeviceContext
from internal_utils import (
    DeviceNDBuffer,
    env_get_shape,
    env_get_dtype,
    int_list_to_tuple,
)
from buffer import NDBuffer
from utils.index import StaticTuple, StaticIntTuple, Index
from sys import env_get_int


fn bench_layer_norm_gpu[
    type: DType, rank: Int
](
    ctx: DeviceContext,
    inout b: Bench,
    fn_name: String,
    shape: StaticIntTuple[rank],
) raises:
    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols

    var data_h = UnsafePointer[Scalar[type]].alloc(rows * cols)
    var res = UnsafePointer[Scalar[type]].alloc(rows * cols)
    var gamma_h = UnsafePointer[Scalar[type]].alloc(cols)
    var beta_h = UnsafePointer[Scalar[type]].alloc(cols)

    for i in range(rows * cols):
        var val = Scalar[type](random_float64(0, 100).cast[type]())
        data_h[i] = val

    for i in range(cols):
        gamma_h[i] = ((i + cols) / cols).cast[type]()
        beta_h[i] = (i / cols).cast[type]()

    var data_d = ctx.create_buffer[type](rows * cols)
    var gamma_d = ctx.create_buffer[type](cols)
    var beta_d = ctx.create_buffer[type](cols)

    var param_shape = StaticIntTuple[1](cols)

    var data_buf = NDBuffer[type, rank](data_d.ptr, shape)
    var gamma = NDBuffer[type, 1](gamma_d.ptr, param_shape)
    var beta = NDBuffer[type, 1](beta_d.ptr, param_shape)
    var epsilon = Scalar[type]()

    ctx.enqueue_copy_to_device(data_d, data_h)
    ctx.enqueue_copy_to_device(gamma_d, gamma_h)
    ctx.enqueue_copy_to_device(beta_d, beta_h)

    @__copy_capture(data_buf)
    @always_inline
    @parameter
    fn input_fn[
        width: Int, _rank: Int
    ](idx: StaticIntTuple[_rank]) -> SIMD[type, width]:
        return data_buf.load[width=width](rebind[StaticIntTuple[rank]](idx))

    @__copy_capture(gamma)
    @always_inline
    @parameter
    fn gamma_fn[
        width: Int, rank: Int
    ](idx: StaticIntTuple[rank]) -> SIMD[type, width]:
        return gamma.load[width=width](idx[0])

    @always_inline
    @__copy_capture(shape, beta, epsilon, data_buf)
    @parameter
    fn bench_fn(inout b: Bencher) raises:
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            layer_norm_gpu[input_fn, gamma_fn](
                shape, beta, epsilon, data_buf, ctx
            )

        b.iter_custom[kernel_launch](ctx)

    b.bench_function[bench_fn](
        BenchId(
            "layer_norm", input_id=fn_name + "/" + str(type) + "/" + str(shape)
        ),
    )

    ctx.synchronize()

    _ = data_d
    _ = gamma_d
    _ = beta_d

    data_h.free()
    res.free()
    gamma_h.free()
    beta_h.free()


fn bench_rms_norm_gpu[
    type: DType, rank: Int
](
    ctx: DeviceContext,
    inout b: Bench,
    fn_name: String,
    shape: StaticIntTuple[rank],
) raises:
    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols

    var data_h = UnsafePointer[Scalar[type]].alloc(rows * cols)
    var res = UnsafePointer[Scalar[type]].alloc(rows * cols)
    var gamma_h = UnsafePointer[Scalar[type]].alloc(cols)

    for i in range(rows * cols):
        var val = Scalar[type](random_float64(0, 100).cast[type]())
        data_h[i] = val

    for i in range(cols):
        gamma_h[i] = ((i + cols) / cols).cast[type]()

    var data_d = ctx.create_buffer[type](rows * cols)
    var gamma_d = ctx.create_buffer[type](cols)

    var param_shape = StaticIntTuple[1](cols)

    var data_buf = NDBuffer[type, rank](data_d.ptr, shape)
    var gamma = NDBuffer[type, 1](gamma_d.ptr, param_shape)
    var epsilon = Scalar[type](0.001)

    ctx.enqueue_copy_to_device(data_d, data_h)
    ctx.enqueue_copy_to_device(gamma_d, gamma_h)

    @__copy_capture(data_buf)
    @always_inline
    @parameter
    fn input_fn[
        width: Int, _rank: Int
    ](idx: StaticIntTuple[_rank]) -> SIMD[type, width]:
        return data_buf.load[width=width](rebind[StaticIntTuple[rank]](idx))

    @always_inline
    @__copy_capture(shape, gamma, epsilon, data_buf)
    @parameter
    fn bench_fn(inout b: Bencher) raises:
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            rms_norm_gpu[input_fn](shape, gamma, epsilon, data_buf, ctx)

        b.iter_custom[kernel_launch](ctx)

    b.bench_function[bench_fn](
        BenchId(
            "rms_norm", input_id=fn_name + "/" + str(type) + "/" + str(shape)
        ),
    )

    ctx.synchronize()

    _ = data_d
    _ = gamma_d

    data_h.free()
    res.free()
    gamma_h.free()


def main():
    alias dtype = env_get_dtype["dtype", DType.bfloat16]()
    alias shape = int_list_to_tuple[env_get_shape["shape", "256x256"]()]()

    var m = Bench(BenchConfig(num_repetitions=1))
    with DeviceContext() as ctx:

        @parameter
        if len(shape) == 2:
            bench_layer_norm_gpu[dtype](ctx, m, "layer_norm_gpu", shape)
        elif len(shape) == 3:
            bench_rms_norm_gpu[dtype](ctx, m, "rms_norm_gpu", shape)

    m.dump_report()
