# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s

from math import ceildiv, isqrt
from random import random_float64
from sys import simdwidthof

from algorithm import mean, variance
from buffer import Buffer, NDBuffer
from buffer.dimlist import DimList
from gpu import WARP_SIZE
from gpu.host._compile import _compile_code, _get_nvptx_target
from gpu.host.device_context import DeviceContext
from memory import UnsafePointer
from nn.normalization import *
from testing import assert_almost_equal

from utils.index import Index, StaticIntTuple, StaticTuple


fn welford_mean_var[
    type: DType
](vector: Buffer[type], size: Int) -> StaticTuple[Scalar[type], 2]:
    var mean: Scalar[type] = 0
    var s: Scalar[type] = 0

    for i in range(1, size + 1):
        var x: Scalar[type] = vector[i - 1]
        var old_mean: Scalar[type] = mean
        mean = mean + ((x - mean) / i)
        s = s + ((x - mean) * (x - old_mean))
    return StaticTuple[Scalar[type], 2](mean, s / (size - 1))


fn run_layer_norm_block[
    type: DType,
    *,
    simd_width: Int = simdwidthof[type, target = _get_nvptx_target()](),
](ctx: DeviceContext, rows: Int, cols: Int, rtol: Scalar[type] = 0.01) raises:
    print("== run_layer_norm_gpu block kernel")

    alias rank = 2
    var data_h = UnsafePointer[Scalar[type]].alloc(rows * cols)
    var res = UnsafePointer[Scalar[type]].alloc(rows * cols)
    var gamma_h = UnsafePointer[Scalar[type]].alloc(cols)
    var beta_h = UnsafePointer[Scalar[type]].alloc(cols)

    for i in range(rows * cols):
        var val = Scalar[type](i)
        data_h[i] = val

    for i in range(cols):
        gamma_h[i] = ((i + cols) / cols).cast[type]()
        beta_h[i] = (i / cols).cast[type]()

    var data_d = ctx.create_buffer[type](rows * cols)
    var gamma_d = ctx.create_buffer[type](cols)
    var beta_d = ctx.create_buffer[type](cols)

    var data_shape = StaticIntTuple[rank](rows, cols)
    var param_shape = StaticIntTuple[1](cols)

    var data_buf = NDBuffer[type, rank](data_d.ptr, data_shape)
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
        width: Int, _r: Int
    ](idx: StaticIntTuple[_r]) -> SIMD[type, width]:
        var r_idx = rebind[StaticIntTuple[rank]](idx)
        return data_buf.load[width=width](r_idx)

    @__copy_capture(gamma)
    @always_inline
    @parameter
    fn gamma_fn[
        width: Int, rank: Int
    ](idx: StaticIntTuple[rank]) -> SIMD[type, width]:
        return gamma.load[width=width](idx[0])

    var func_ln = ctx.compile_function[
        layer_norm_gpu_block[type, simd_width, input_fn, gamma_fn]
    ]()

    var max_warps_per_block = 32

    @always_inline
    @parameter
    @__copy_capture(data_buf, gamma, beta, epsilon)
    fn run_func_ln() raises:
        ctx.enqueue_function(
            func_ln,
            data_buf,
            beta,
            epsilon,
            grid_dim=(rows, 1),
            block_dim=min(
                ceildiv(ceildiv(cols, simd_width), WARP_SIZE) * WARP_SIZE,
                WARP_SIZE * max_warps_per_block,
            ),
        )

    run_func_ln()
    ctx.enqueue_copy_from_device(res, data_d)
    ctx.synchronize()

    for r in range(rows):
        var vec = Buffer[type](data_h + r * cols, cols)
        var mean_ref = mean(vec)
        var var_ref = variance(vec, 1)
        var norm_factor_ref = isqrt(var_ref + epsilon)
        for c in range(cols):
            var idx = r * cols + c
            var val = ((data_h[idx] - mean_ref) * norm_factor_ref) * gamma_h[
                c
            ] + beta_h[c]
            assert_almost_equal(val, res[idx], rtol=rtol)

    _ = data_h
    _ = gamma_h
    _ = beta_h
    _ = data_d
    _ = gamma_d
    _ = beta_d
    _ = res
    _ = func_ln^


fn run_layer_norm_gpu[
    type: DType, rank: Int
](
    ctx: DeviceContext, shape: StaticIntTuple[rank], rtol: Scalar[type] = 0.01
) raises:
    print("== run_layer_norm_gpu")

    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols

    var data_h = UnsafePointer[Scalar[type]].alloc(rows * cols)
    var res = UnsafePointer[Scalar[type]].alloc(rows * cols)
    var gamma_h = UnsafePointer[Scalar[type]].alloc(cols)
    var beta_h = UnsafePointer[Scalar[type]].alloc(cols)

    for i in range(rows * cols):
        var val = Scalar[type](i)
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

    layer_norm_gpu[input_fn, gamma_fn](shape, beta, epsilon, data_buf, ctx)
    ctx.enqueue_copy_from_device(res, data_d)
    ctx.synchronize()

    for r in range(rows):
        var vec = Buffer[type](data_h + r * cols, cols)
        var mean_ref = mean(vec)
        var var_ref = variance(vec, 1)
        var norm_factor_ref = isqrt(var_ref + epsilon)
        for c in range(cols):
            var idx = r * cols + c
            var val = ((data_h[idx] - mean_ref) * norm_factor_ref) * gamma_h[
                c
            ] + beta_h[c]
            assert_almost_equal(val, res[idx], rtol=rtol)

    _ = data_h
    _ = gamma_h
    _ = beta_h
    _ = data_d
    _ = gamma_d
    _ = beta_d
    _ = res


fn run_layer_norm_warp_tiling[
    type: DType,
    *,
    simd_width: Int = simdwidthof[type, target = _get_nvptx_target()](),
](ctx: DeviceContext, rows: Int, cols: Int, rtol: Scalar[type] = 0.01) raises:
    print("== run_layer_norm_gpu warp tiling kernel")

    alias rank = 2
    var data_h = UnsafePointer[Scalar[type]].alloc(rows * cols)
    var res = UnsafePointer[Scalar[type]].alloc(rows * cols)
    var gamma_h = UnsafePointer[Scalar[type]].alloc(cols)
    var beta_h = UnsafePointer[Scalar[type]].alloc(cols)

    for i in range(rows * cols):
        var val = Scalar[type](i)
        data_h[i] = val

    for i in range(cols):
        gamma_h[i] = ((i + cols) / cols).cast[type]()
        beta_h[i] = (i / cols).cast[type]()

    var data_d = ctx.create_buffer[type](rows * cols)
    var gamma_d = ctx.create_buffer[type](cols)
    var beta_d = ctx.create_buffer[type](cols)

    var data_shape = StaticIntTuple[rank](rows, cols)
    var param_shape = StaticIntTuple[1](cols)

    var data_buf = NDBuffer[type, rank](data_d.ptr, data_shape)
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
        width: Int, _r: Int
    ](idx: StaticIntTuple[_r]) -> SIMD[type, width]:
        var r_idx = rebind[StaticIntTuple[rank]](idx)
        return data_buf.load[width=width](r_idx)

    @__copy_capture(gamma)
    @always_inline
    @parameter
    fn gamma_fn[
        width: Int, rank: Int
    ](idx: StaticIntTuple[rank]) -> SIMD[type, width]:
        return gamma.load[width=width](idx[0])

    var func_ln = ctx.compile_function[
        layer_norm_gpu_warp_tiling[type, simd_width, input_fn, gamma_fn]
    ]()

    var max_warps_per_block = 32

    @always_inline
    @parameter
    @__copy_capture(data_buf, gamma, beta, epsilon)
    fn run_func_ln() raises:
        ctx.enqueue_function(
            func_ln,
            data_buf,
            beta,
            epsilon,
            grid_dim=(rows, 1),
            block_dim=min(
                ceildiv(ceildiv(cols, simd_width), WARP_SIZE) * WARP_SIZE,
                WARP_SIZE * max_warps_per_block,
            ),
        )

    run_func_ln()
    ctx.enqueue_copy_from_device(res, data_d)
    ctx.synchronize()

    for r in range(rows):
        var vec = Buffer[type](data_h + r * cols, cols)
        var mean_ref = mean(vec)
        var var_ref = variance(vec, 1)
        var norm_factor_ref = isqrt(var_ref + epsilon)
        for c in range(cols):
            var idx = r * cols + c
            var val = ((data_h[idx] - mean_ref) * norm_factor_ref) * gamma_h[
                c
            ] + beta_h[c]
            assert_almost_equal(val, res[idx], rtol=rtol)

    _ = data_h
    _ = gamma_h
    _ = beta_h
    _ = data_d
    _ = gamma_d
    _ = beta_d
    _ = res
    _ = func_ln^


def main():
    with DeviceContext() as ctx:
        run_layer_norm_block[DType.float32, simd_width=1](ctx, rows=3, cols=5)
        run_layer_norm_block[DType.float32](ctx, rows=3, cols=8)
        run_layer_norm_block[DType.float32, simd_width=1](ctx, rows=7, cols=33)
        run_layer_norm_block[DType.float32](ctx, rows=1, cols=1024)
        run_layer_norm_block[DType.float32](ctx, rows=1, cols=8192, rtol=0.1)

        run_layer_norm_warp_tiling[DType.float32, simd_width=1](
            ctx, rows=3, cols=5
        )
        run_layer_norm_warp_tiling[DType.float32](ctx, rows=3, cols=8)
        run_layer_norm_warp_tiling[DType.float32, simd_width=1](
            ctx, rows=7, cols=33
        )
        run_layer_norm_warp_tiling[DType.float32](ctx, rows=1, cols=1024)
        run_layer_norm_warp_tiling[DType.float32](ctx, rows=10, cols=4096)

        # variable rank
        run_layer_norm_gpu[DType.float32, 1](ctx, StaticIntTuple[1](0))
        run_layer_norm_gpu[DType.float32, 1](ctx, StaticIntTuple[1](5))
        run_layer_norm_gpu[DType.float32, 5](
            ctx, StaticIntTuple[5](3, 4, 10, 20, 8)
        )
        run_layer_norm_gpu[DType.float32, 5](
            ctx, StaticIntTuple[5](1, 5, 6, 10, 128)
        )
