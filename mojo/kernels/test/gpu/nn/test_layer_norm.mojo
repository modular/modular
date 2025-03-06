# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug-no-assert %s

from math import ceildiv, isqrt
from sys import simdwidthof

from algorithm import mean, variance
from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu import WARP_SIZE
from gpu.host import DeviceContext
from gpu.host._compile import _get_gpu_target
from memory import UnsafePointer
from nn.normalization import *
from testing import assert_almost_equal

from utils.index import Index, IndexList


fn run_layer_norm_block[
    type: DType,
    *,
    simd_width: Int = simdwidthof[type, target = _get_gpu_target()](),
](ctx: DeviceContext, rows: Int, cols: Int, rtol: Float64 = 0.01) raises:
    print("== run_layer_norm_gpu block kernel")

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

    var data_d = ctx.enqueue_create_buffer[type](rows * cols)
    var gamma_d = ctx.enqueue_create_buffer[type](cols)
    var beta_d = ctx.enqueue_create_buffer[type](cols)

    var data_shape = Index(rows, cols)
    var param_shape = Index(cols)

    var data_buf = NDBuffer[type, 2](data_d.unsafe_ptr(), data_shape)
    var gamma = NDBuffer[type, 1](gamma_d.unsafe_ptr(), param_shape)
    var beta = NDBuffer[type, 1](beta_d.unsafe_ptr(), param_shape)
    var epsilon = Float32(0)

    ctx.enqueue_copy(data_d, data_h)
    ctx.enqueue_copy(gamma_d, gamma_h)
    ctx.enqueue_copy(beta_d, beta_h)

    @__copy_capture(data_buf)
    @always_inline
    @parameter
    fn input_fn[width: Int](row: Int, col: Int) -> SIMD[type, width]:
        return data_buf.load[width=width](Index(row, col))

    @__copy_capture(gamma)
    @always_inline
    @parameter
    fn gamma_fn[
        width: Int, rank: Int
    ](idx: IndexList[rank]) -> SIMD[type, width]:
        return gamma.load[width=width](idx[0])

    var max_warps_per_block = ctx.device_info.max_thread_block_size // WARP_SIZE

    @always_inline
    @parameter
    @__copy_capture(data_buf, gamma, beta, epsilon)
    fn run_func_ln() raises:
        ctx.enqueue_function[
            layer_norm_gpu_block[simd_width, input_fn, gamma_fn]
        ](
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
    ctx.enqueue_copy(res, data_d)
    ctx.synchronize()

    for r in range(rows):
        var vec = NDBuffer[type, 1](data_h + r * cols, cols)
        var mean_ref = mean(vec)
        var var_ref = variance(vec, correction=0)
        var norm_factor_ref = isqrt(var_ref + epsilon.cast[type]())
        for c in range(cols):
            var idx = r * cols + c
            var val = ((data_h[idx] - mean_ref) * norm_factor_ref) * gamma_h[
                c
            ] + beta_h[c]
            assert_almost_equal(val, res[idx], rtol=rtol)

    _ = data_d
    _ = gamma_d
    _ = beta_d

    data_h.free()
    res.free()
    gamma_h.free()
    beta_h.free()


fn run_layer_norm_gpu[
    type: DType, rank: Int
](ctx: DeviceContext, shape: IndexList[rank], rtol: Float64 = 0.01) raises:
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

    var data_d = ctx.enqueue_create_buffer[type](rows * cols)
    var gamma_d = ctx.enqueue_create_buffer[type](cols)
    var beta_d = ctx.enqueue_create_buffer[type](cols)

    var param_shape = Index(cols)

    var data_buf = NDBuffer[type, rank](data_d.unsafe_ptr(), shape)
    var gamma = NDBuffer[type, 1](gamma_d.unsafe_ptr(), param_shape)
    var beta = NDBuffer[type, 1](beta_d.unsafe_ptr(), param_shape)
    var epsilon = Float32(0)

    ctx.enqueue_copy(data_d, data_h)
    ctx.enqueue_copy(gamma_d, gamma_h)
    ctx.enqueue_copy(beta_d, beta_h)

    @__copy_capture(data_buf)
    @always_inline
    @parameter
    fn input_fn[
        width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[type, width]:
        return data_buf.load[width=width](rebind[IndexList[rank]](idx))

    @__copy_capture(gamma)
    @always_inline
    @parameter
    fn gamma_fn[
        width: Int, rank: Int
    ](idx: IndexList[rank]) -> SIMD[type, width]:
        return gamma.load[width=width](idx[0])

    layer_norm_gpu[input_fn, gamma_fn](shape, beta, epsilon, data_buf, ctx=ctx)
    ctx.enqueue_copy(res, data_d)
    ctx.synchronize()

    for r in range(rows):
        var vec = NDBuffer[type, 1](data_h + r * cols, cols)
        var mean_ref = mean(vec)
        var var_ref = variance(vec, correction=0)
        var norm_factor_ref = isqrt(var_ref + epsilon.cast[type]())
        for c in range(cols):
            var idx = r * cols + c
            var val = ((data_h[idx] - mean_ref) * norm_factor_ref) * gamma_h[
                c
            ] + beta_h[c]
            assert_almost_equal(val, res[idx], rtol=rtol)

    _ = data_d
    _ = gamma_d
    _ = beta_d

    data_h.free()
    res.free()
    gamma_h.free()
    beta_h.free()


fn run_layer_norm_warp_tiling[
    type: DType,
    *,
    simd_width: Int = simdwidthof[type, target = _get_gpu_target()](),
](ctx: DeviceContext, rows: Int, cols: Int, rtol: Float64 = 0.01) raises:
    print("== run_layer_norm_gpu warp tiling kernel")

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

    var data_d = ctx.enqueue_create_buffer[type](rows * cols)
    var gamma_d = ctx.enqueue_create_buffer[type](cols)
    var beta_d = ctx.enqueue_create_buffer[type](cols)

    var data_shape = Index(rows, cols)
    var param_shape = Index(cols)

    var data_buf = NDBuffer[type, 2](data_d.unsafe_ptr(), data_shape)
    var gamma = NDBuffer[type, 1](gamma_d.unsafe_ptr(), param_shape)
    var beta = NDBuffer[type, 1](beta_d.unsafe_ptr(), param_shape)
    var epsilon = Float32(0)

    ctx.enqueue_copy(data_d, data_h)
    ctx.enqueue_copy(gamma_d, gamma_h)
    ctx.enqueue_copy(beta_d, beta_h)

    @__copy_capture(data_buf)
    @always_inline
    @parameter
    fn input_fn[width: Int](row: Int, col: Int) -> SIMD[type, width]:
        return data_buf.load[width=width](Index(row, col))

    @__copy_capture(gamma)
    @always_inline
    @parameter
    fn gamma_fn[
        width: Int, rank: Int
    ](idx: IndexList[rank]) -> SIMD[type, width]:
        return gamma.load[width=width](idx[0])

    var max_warps_per_block = ctx.device_info.max_thread_block_size // WARP_SIZE

    @always_inline
    @parameter
    @__copy_capture(data_buf, gamma, beta, epsilon)
    fn run_func_ln() raises:
        ctx.enqueue_function[
            layer_norm_gpu_warp_tiling[simd_width, input_fn, gamma_fn]
        ](
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
    ctx.enqueue_copy(res, data_d)
    ctx.synchronize()

    for r in range(rows):
        var vec = NDBuffer[type, 1](data_h + r * cols, cols)
        var mean_ref = mean(vec)
        var var_ref = variance(vec, correction=0)
        var norm_factor_ref = isqrt(var_ref + epsilon.cast[type]())
        for c in range(cols):
            var idx = r * cols + c
            var val = ((data_h[idx] - mean_ref) * norm_factor_ref) * gamma_h[
                c
            ] + beta_h[c]
            assert_almost_equal(val, res[idx], rtol=rtol)

    _ = data_d
    _ = gamma_d
    _ = beta_d

    data_h.free()
    res.free()
    gamma_h.free()
    beta_h.free()


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
        run_layer_norm_gpu[DType.float32](ctx, Index(5))
        run_layer_norm_gpu[DType.float32](ctx, Index(3, 4, 10, 20, 8))
        run_layer_norm_gpu[DType.float32](ctx, Index(1, 5, 6, 10, 128))
