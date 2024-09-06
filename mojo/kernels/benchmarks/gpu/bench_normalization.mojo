# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# REQUIRES: has_cuda_device
# RUN: %mojo-build %s

from random import random_float64
from nn.normalization import layer_norm_reshape, layer_norm_gpu
from benchmark import Bench, BenchConfig, Bencher, BenchId
from gpu.host import DeviceContext
from internal_utils import DeviceNDBuffer, bench_compile_time
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

    alias rank_rs = 2
    var data_buf_rs = layer_norm_reshape[rank_rs](shape, data_buf)

    @__copy_capture(data_buf_rs)
    @always_inline
    @parameter
    fn input_fn[
        width: Int, _r: Int
    ](idx: StaticIntTuple[_r]) -> SIMD[type, width]:
        var r_idx = rebind[StaticIntTuple[rank_rs]](idx)
        return data_buf_rs.load[width=width](r_idx)

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

    _ = data_h
    _ = gamma_h
    _ = beta_h
    _ = data_d
    _ = gamma_d
    _ = beta_d
    _ = res


def main():
    # TODO: expand to all the params
    alias phony = env_get_int["phony", 1]()
    constrained[phony == 1]()

    with DeviceContext() as ctx:
        var m = Bench(BenchConfig(num_repetitions=1))

        var shape_list = List[StaticIntTuple[2]](
            StaticIntTuple[2](256, 256),
            StaticIntTuple[2](1024, 1024),
            StaticIntTuple[2](8192, 8192),
            StaticIntTuple[2](1, 3072),
            StaticIntTuple[2](512, 3072),
            StaticIntTuple[2](1024, 3072),
            StaticIntTuple[2](4096, 3072),
            StaticIntTuple[2](5072, 256),
        )

        for s in range(len(shape_list)):
            bench_layer_norm_gpu[DType.bfloat16, 2](
                ctx, m, "layer_norm_gpu", shape_list[s]
            )
        m.dump_report()
