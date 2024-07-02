# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s

from algorithm import mean, variance
from math import ceildiv, isclose, sqrt
from random import random_float64
from nn.normalization import layer_norm_gpu_block, layer_norm_gpu_warp_tiling
from buffer import Buffer, NDBuffer
from buffer.list import DimList
from gpu.host.device_context import DeviceBuffer, DeviceContext
from memory.unsafe import DTypePointer
from testing import assert_true
from utils.index import StaticTuple, StaticIntTuple, Index


fn welford_mean_var[
    type: DType, size: Int
](vector: Buffer[type, size]) -> StaticTuple[Scalar[type], 2]:
    var mean: Scalar[type] = Float32(0)
    var s: Scalar[type] = Float32(0)

    @parameter
    for i in range(1, size + 1):
        var x: Scalar[type] = vector[i - 1]
        var old_mean: Scalar[type] = mean
        mean = mean + ((x - mean) / i)
        s = s + ((x - mean) * (x - old_mean))
    return StaticTuple[Scalar[type], 2](mean, s / (size - 1))


fn run_layer_norm_block[
    type: DType
](ctx: DeviceContext, rows: Int, cols: Int) raises:
    print("== run_layer_norm_gpu block kernel")

    alias rank = 2
    var data_h = Pointer[Scalar[type]].alloc(rows * cols)
    var res = Pointer[Scalar[type]].alloc(rows * cols)
    var gamma_h = Pointer[Scalar[type]].alloc(1)
    var beta_h = Pointer[Scalar[type]].alloc(1)
    var epsilon_h = Pointer[Scalar[type]].alloc(1)

    for i in range(rows * cols):
        var val = Scalar[type](i)
        data_h[i] = val

    gamma_h[0] = Scalar[type](1)
    beta_h[0] = Scalar[type](0)
    epsilon_h[0] = Scalar[type](0)

    var data_d = ctx.create_buffer[type](rows * cols)
    var gamma_d = ctx.create_buffer[type](1)
    var beta_d = ctx.create_buffer[type](1)
    var epsilon_d = ctx.create_buffer[type](1)

    var data_shape = StaticIntTuple[rank](rows, cols)
    alias param_shape = DimList(1)

    var data_buf = NDBuffer[type, rank](data_d.ptr, data_shape)
    var gamma = NDBuffer[type, 1, param_shape](gamma_d.ptr)
    var beta = NDBuffer[type, 1, param_shape](beta_d.ptr)
    var epsilon = NDBuffer[type, 1, param_shape](epsilon_d.ptr)

    ctx.enqueue_copy_to_device(data_d, data_h)
    ctx.enqueue_copy_to_device(gamma_d, gamma_h)
    ctx.enqueue_copy_to_device(beta_d, beta_h)
    ctx.enqueue_copy_to_device(epsilon_d, epsilon_h)

    alias simd_width = 4
    var func_ln = ctx.compile_function[
        layer_norm_gpu_block[
            type,
            simd_width,
            rank,
        ]
    ](dump_ptx=False)

    @always_inline
    @parameter
    @__copy_capture(data_buf, gamma, beta, epsilon)
    fn run_func_ln() raises:
        ctx.enqueue_function(
            func_ln,
            data_buf,
            gamma,
            beta,
            epsilon,
            grid_dim=(rows, 1),
            block_dim=(min(cols // simd_width, 1024), 1),
        )

    run_func_ln()
    ctx.synchronize()

    ctx.enqueue_copy_from_device(res, data_d)

    for r in range(rows):
        var vec = Buffer[type](data_h + r * cols, cols)
        var mean_ref = mean(vec)
        var var_ref = variance(vec, 1)
        var norm_factor_ref = 1 / sqrt(var_ref + epsilon_h[0])
        for c in range(cols):
            var idx = r * cols + c
            var val = ((data_h[idx] - mean_ref) * norm_factor_ref) * gamma_h[
                0
            ] + beta_h[0]
            assert_true(isclose(val, res[idx], rtol=0.01))

    _ = data_h
    _ = gamma_h
    _ = beta_h
    _ = epsilon_h
    _ = data_d
    _ = gamma_d
    _ = beta_d
    _ = epsilon_d
    _ = res
    _ = func_ln^


fn run_layer_norm_warp_tiling[
    type: DType
](ctx: DeviceContext, rows: Int, cols: Int) raises:
    print("== run_layer_norm_gpu warp tiling kernel")

    alias rank = 2
    var data_h = Pointer[Scalar[type]].alloc(rows * cols)
    var res = Pointer[Scalar[type]].alloc(rows * cols)
    var gamma_h = Pointer[Scalar[type]].alloc(1)
    var beta_h = Pointer[Scalar[type]].alloc(1)
    var epsilon_h = Pointer[Scalar[type]].alloc(1)

    for i in range(rows * cols):
        var val = Scalar[type](i)
        data_h[i] = val

    gamma_h[0] = Scalar[type](1)
    beta_h[0] = Scalar[type](0)
    epsilon_h[0] = Scalar[type](0)

    var data_d = ctx.create_buffer[type](rows * cols)
    var gamma_d = ctx.create_buffer[type](1)
    var beta_d = ctx.create_buffer[type](1)
    var epsilon_d = ctx.create_buffer[type](1)

    var data_shape = StaticIntTuple[rank](rows, cols)
    alias param_shape = DimList(1)

    var data_buf = NDBuffer[type, rank](data_d.ptr, data_shape)
    var gamma = NDBuffer[type, 1, param_shape](gamma_d.ptr)
    var beta = NDBuffer[type, 1, param_shape](beta_d.ptr)
    var epsilon = NDBuffer[type, 1, param_shape](epsilon_d.ptr)

    ctx.enqueue_copy_to_device(data_d, data_h)
    ctx.enqueue_copy_to_device(gamma_d, gamma_h)
    ctx.enqueue_copy_to_device(beta_d, beta_h)
    ctx.enqueue_copy_to_device(epsilon_d, epsilon_h)

    alias simd_width = 4
    var func_ln = ctx.compile_function[
        layer_norm_gpu_warp_tiling[
            type,
            simd_width,
            rank,
        ]
    ](dump_ptx=False)

    @always_inline
    @parameter
    @__copy_capture(data_buf, gamma, beta, epsilon)
    fn run_func_ln() raises:
        ctx.enqueue_function(
            func_ln,
            data_buf,
            gamma,
            beta,
            epsilon,
            grid_dim=(rows, 1),
            block_dim=(min(cols // simd_width, 1024), 1),
        )

    run_func_ln()
    ctx.synchronize()

    ctx.enqueue_copy_from_device(res, data_d)

    for r in range(rows):
        var vec = Buffer[type](data_h + r * cols, cols)
        var mean_ref = mean(vec)
        var var_ref = variance(vec, 1)
        var norm_factor_ref = 1 / sqrt(var_ref + epsilon_h[0])
        for c in range(cols):
            var idx = r * cols + c
            var val = ((data_h[idx] - mean_ref) * norm_factor_ref) * gamma_h[
                0
            ] + beta_h[0]
            assert_true(isclose(val, res[idx], rtol=0.01))

    _ = data_h
    _ = gamma_h
    _ = beta_h
    _ = epsilon_h
    _ = data_d
    _ = gamma_d
    _ = beta_d
    _ = epsilon_d
    _ = res
    _ = func_ln^


def main():
    try:
        with DeviceContext() as ctx:
            run_layer_norm_block[DType.float32](ctx, rows=1, cols=128)
            run_layer_norm_block[DType.float32](ctx, rows=10, cols=1024)
            run_layer_norm_warp_tiling[DType.float32](ctx, rows=1, cols=128)
            run_layer_norm_warp_tiling[DType.float32](ctx, rows=10, cols=1024)
    except e:
        print("CUDA_ERROR:", e)
