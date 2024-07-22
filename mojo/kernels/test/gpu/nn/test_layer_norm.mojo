# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s

from algorithm import mean, variance
from math import ceildiv, rsqrt
from random import random_float64
from nn.normalization import *
from buffer import Buffer, NDBuffer
from buffer.dimlist import DimList
from gpu.host.device_context import DeviceBuffer, DeviceContext
from gpu import WARP_SIZE
from memory import UnsafePointer
from testing import assert_almost_equal
from utils.index import StaticTuple, StaticIntTuple, Index
from gpu.host._compile import _compile_code, _get_nvptx_target


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


fn run_layer_norm_block_vector[
    type: DType
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
        gamma_h[i] = Scalar[type](1)
        beta_h[i] = Scalar[type](0)

    var data_d = ctx.create_buffer[type](rows * cols)
    var gamma_d = ctx.create_buffer[type](cols)
    var beta_d = ctx.create_buffer[type](rows)

    var data_shape = StaticIntTuple[rank](rows, cols)
    var param_shape = StaticIntTuple[1](cols)

    var data_buf = NDBuffer[type, rank](data_d.ptr, data_shape)
    var gamma = NDBuffer[type, 1](gamma_d.ptr, param_shape)
    var beta = NDBuffer[type, 1](beta_d.ptr, param_shape)
    var epsilon = Scalar[type]()

    ctx.enqueue_copy_to_device(data_d, data_h.address)
    ctx.enqueue_copy_to_device(gamma_d, gamma_h.address)
    ctx.enqueue_copy_to_device(beta_d, beta_h.address)

    alias simd_width = simdwidthof[type, target = _get_nvptx_target()]()
    var func_ln = ctx.compile_function[
        layer_norm_gpu_block_vector[
            type,
            simd_width,
            rank,
        ]
    ](dump_ptx=False)

    var max_warps_per_block = 32

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
            block_dim=min(
                ceildiv(ceildiv(cols, simd_width), WARP_SIZE) * WARP_SIZE,
                WARP_SIZE * max_warps_per_block,
            ),
        )

    run_func_ln()
    ctx.synchronize()

    ctx.enqueue_copy_from_device(res.address, data_d)

    for r in range(rows):
        var vec = Buffer[type](data_h + r * cols, cols)
        var mean_ref = mean(vec)
        var var_ref = variance(vec, 1)
        var norm_factor_ref = rsqrt(var_ref + epsilon)
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


fn run_layer_norm_block_scalar[
    type: DType
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
        gamma_h[i] = Scalar[type](1)
        beta_h[i] = Scalar[type](0)

    var data_d = ctx.create_buffer[type](rows * cols)
    var gamma_d = ctx.create_buffer[type](cols)
    var beta_d = ctx.create_buffer[type](rows)

    var data_shape = StaticIntTuple[rank](rows, cols)
    var param_shape = StaticIntTuple[1](cols)

    var data_buf = NDBuffer[type, rank](data_d.ptr, data_shape)
    var gamma = NDBuffer[type, 1](gamma_d.ptr, param_shape)
    var beta = NDBuffer[type, 1](beta_d.ptr, param_shape)
    var epsilon = Scalar[type]()

    ctx.enqueue_copy_to_device(data_d, data_h.address)
    ctx.enqueue_copy_to_device(gamma_d, gamma_h.address)
    ctx.enqueue_copy_to_device(beta_d, beta_h.address)

    alias simd_width = 1
    var func_ln = ctx.compile_function[
        layer_norm_gpu_block_scalar[
            type,
            simd_width,
            rank,
        ]
    ](dump_ptx=False)

    var max_warps_per_block = 32

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
            block_dim=min(
                ceildiv(ceildiv(cols, simd_width), WARP_SIZE) * WARP_SIZE,
                WARP_SIZE * max_warps_per_block,
            ),
        )

    run_func_ln()
    ctx.synchronize()

    ctx.enqueue_copy_from_device(res.address, data_d)

    for r in range(rows):
        var vec = Buffer[type](data_h + r * cols, cols)
        var mean_ref = mean(vec)
        var var_ref = variance(vec, 1)
        var norm_factor_ref = rsqrt(var_ref + epsilon)
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


fn run_layer_norm_warp_tiling_vector[
    type: DType
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
        gamma_h[i] = Scalar[type](1)
        beta_h[i] = Scalar[type](0)

    var data_d = ctx.create_buffer[type](rows * cols)
    var gamma_d = ctx.create_buffer[type](cols)
    var beta_d = ctx.create_buffer[type](rows)

    var data_shape = StaticIntTuple[rank](rows, cols)
    var param_shape = StaticIntTuple[1](cols)

    var data_buf = NDBuffer[type, rank](data_d.ptr, data_shape)
    var gamma = NDBuffer[type, 1](gamma_d.ptr, param_shape)
    var beta = NDBuffer[type, 1](beta_d.ptr, param_shape)
    var epsilon = Scalar[type]()

    ctx.enqueue_copy_to_device(data_d, data_h.address)
    ctx.enqueue_copy_to_device(gamma_d, gamma_h.address)
    ctx.enqueue_copy_to_device(beta_d, beta_h.address)

    alias simd_width = simdwidthof[type, target = _get_nvptx_target()]()
    var func_ln = ctx.compile_function[
        layer_norm_gpu_warp_tiling_vector[
            type,
            simd_width,
            rank,
        ]
    ](dump_ptx=False)

    var max_warps_per_block = 32

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
            block_dim=min(
                ceildiv(ceildiv(cols, simd_width), WARP_SIZE) * WARP_SIZE,
                WARP_SIZE * max_warps_per_block,
            ),
        )

    run_func_ln()
    ctx.synchronize()

    ctx.enqueue_copy_from_device(res.address, data_d)

    for r in range(rows):
        var vec = Buffer[type](data_h + r * cols, cols)
        var mean_ref = mean(vec)
        var var_ref = variance(vec, 1)
        var norm_factor_ref = rsqrt(var_ref + epsilon)
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


fn run_layer_norm_warp_tiling_scalar[
    type: DType
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
        gamma_h[i] = Scalar[type](1)
        beta_h[i] = Scalar[type](0)

    var data_d = ctx.create_buffer[type](rows * cols)
    var gamma_d = ctx.create_buffer[type](cols)
    var beta_d = ctx.create_buffer[type](rows)

    var data_shape = StaticIntTuple[rank](rows, cols)
    var param_shape = StaticIntTuple[1](cols)

    var data_buf = NDBuffer[type, rank](data_d.ptr, data_shape)
    var gamma = NDBuffer[type, 1](gamma_d.ptr, param_shape)
    var beta = NDBuffer[type, 1](beta_d.ptr, param_shape)
    var epsilon = Scalar[type]()

    ctx.enqueue_copy_to_device(data_d, data_h.address)
    ctx.enqueue_copy_to_device(gamma_d, gamma_h.address)
    ctx.enqueue_copy_to_device(beta_d, beta_h.address)

    alias simd_width = 1
    var func_ln = ctx.compile_function[
        layer_norm_gpu_warp_tiling_scalar[
            type,
            simd_width,
            rank,
        ]
    ](dump_ptx=False)

    var max_warps_per_block = 32

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
            block_dim=min(
                ceildiv(ceildiv(cols, simd_width), WARP_SIZE) * WARP_SIZE,
                WARP_SIZE * max_warps_per_block,
            ),
        )

    run_func_ln()
    ctx.synchronize()

    ctx.enqueue_copy_from_device(res.address, data_d)

    for r in range(rows):
        var vec = Buffer[type](data_h + r * cols, cols)
        var mean_ref = mean(vec)
        var var_ref = variance(vec, 1)
        var norm_factor_ref = rsqrt(var_ref + epsilon)
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
    try:
        with DeviceContext() as ctx:
            run_layer_norm_block_scalar[DType.float32](ctx, rows=3, cols=5)
            run_layer_norm_block_vector[DType.float32](ctx, rows=3, cols=8)
            run_layer_norm_block_scalar[DType.float32](ctx, rows=7, cols=33)
            run_layer_norm_block_vector[DType.float32](ctx, rows=1, cols=1024)
            run_layer_norm_block_vector[DType.float32](
                ctx, rows=1, cols=8192, rtol=0.1
            )

            run_layer_norm_warp_tiling_scalar[DType.float32](
                ctx, rows=3, cols=5
            )
            run_layer_norm_warp_tiling_vector[DType.float32](
                ctx, rows=3, cols=8
            )
            run_layer_norm_warp_tiling_scalar[DType.float32](
                ctx, rows=7, cols=33
            )
            run_layer_norm_warp_tiling_vector[DType.float32](
                ctx, rows=1, cols=1024
            )
            run_layer_norm_warp_tiling_vector[DType.float32](
                ctx, rows=10, cols=4096
            )
    except e:
        print("CUDA_ERROR:", e)
