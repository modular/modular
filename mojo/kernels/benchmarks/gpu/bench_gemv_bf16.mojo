# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %bare-mojo build %s

from math import ceildiv
from random import randn

from benchmark import Bench, BenchConfig, Bencher, BenchId
from gpu import WARP_SIZE
from gpu.host import DeviceContext
from linalg.matmul_gpu import gemv_kernel, gemv_tc_kernel, matmul_kernel_naive
from memory import memset
from gpu.host._compile import _get_nvptx_target
from internal_utils import DeviceNDBuffer, bench_compile_time
from buffer import DimList, NDBuffer
from utils import StaticIntTuple
from utils.index import Index


fn bench_gemv_tc[
    type: DType
](
    inout m: Bench, fn_name: String, dims: StaticIntTuple[3], ctx: DeviceContext
) raises:
    var M = dims[0]
    var N = dims[1]
    var K = dims[2]
    var a_host = UnsafePointer[Scalar[type]].alloc(M * K)
    var b_host = UnsafePointer[Scalar[type]].alloc(K * N)
    var c_host = UnsafePointer[Scalar[type]].alloc(M * N)

    randn(a_host, M * K)
    randn(b_host, K * N)
    memset(c_host, 0, M * N)

    var a_buf_h = NDBuffer[type, 2](a_host, StaticIntTuple[2](M, K))
    var b_buf_h = NDBuffer[type, 2](b_host, StaticIntTuple[2](K, N))
    var c_buf_h = NDBuffer[type, 2](c_host, StaticIntTuple[2](M, N))

    var a_buf = DeviceNDBuffer[type, 2](StaticIntTuple[2](M, K), ctx=ctx)
    var b_buf = DeviceNDBuffer[type, 2](StaticIntTuple[2](K, N), ctx=ctx)
    var c_buf = DeviceNDBuffer[type, 2](StaticIntTuple[2](M, N), ctx=ctx)

    ctx.enqueue_copy_to_device(a_buf.buffer, a_buf_h.data)
    ctx.enqueue_copy_to_device(b_buf.buffer, b_buf_h.data)

    alias BLOCK_DIM = 16
    alias WARPS_PER_BLOCK = 32
    var func_gemv = ctx.compile_function[
        gemv_tc_kernel[
            type,
            type,
            type,
        ]
    ]()

    @always_inline
    @__copy_capture(M, N, K)
    @parameter
    fn bench_fn(inout b: Bencher) raises:
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            ctx.enqueue_function(
                func_gemv,
                c_buf.buffer.ptr,
                a_buf.buffer.ptr,
                b_buf.buffer.ptr,
                M,
                N,
                K,
                grid_dim=ceildiv(M, WARPS_PER_BLOCK),
                block_dim=WARP_SIZE * WARPS_PER_BLOCK,
            )

        b.iter_custom[kernel_launch](ctx)

    m.bench_function[bench_fn](
        BenchId("gemv", input_id=fn_name + "/" + str(type) + "/" + str(dims)),
    )

    ctx.synchronize()

    _ = a_buf
    _ = b_buf
    _ = c_buf

    a_host.free()
    b_host.free()
    c_host.free()

    _ = func_gemv^


fn bench_gemv_ws[
    type: DType
](
    inout m: Bench, fn_name: String, dims: StaticIntTuple[3], ctx: DeviceContext
) raises:
    var M = dims[0]
    var N = dims[1]
    var K = dims[2]
    var a_host = UnsafePointer[Scalar[type]].alloc(M * K)
    var b_host = UnsafePointer[Scalar[type]].alloc(K * N)
    var c_host = UnsafePointer[Scalar[type]].alloc(M * N)

    randn(a_host, M * K)
    randn(b_host, K * N)
    memset(c_host, 0, M * N)

    var a_buf_h = NDBuffer[type, 2](a_host, StaticIntTuple[2](M, K))
    var b_buf_h = NDBuffer[type, 2](b_host, StaticIntTuple[2](K, N))
    var c_buf_h = NDBuffer[type, 2](c_host, StaticIntTuple[2](M, N))

    var a_buf = DeviceNDBuffer[type, 2](StaticIntTuple[2](M, K), ctx=ctx)
    var b_buf = DeviceNDBuffer[type, 2](StaticIntTuple[2](K, N), ctx=ctx)
    var c_buf = DeviceNDBuffer[type, 2](StaticIntTuple[2](M, N), ctx=ctx)

    ctx.enqueue_copy_to_device(a_buf.buffer, a_buf_h.data)
    ctx.enqueue_copy_to_device(b_buf.buffer, b_buf_h.data)

    alias BLOCK_DIM = 16
    alias WARPS_PER_BLOCK = 32
    var func_gemv = ctx.compile_function[
        gemv_kernel[
            type,
            type,
            type,
        ]
    ]()

    @always_inline
    @__copy_capture(M, N, K)
    @parameter
    fn bench_fn(inout b: Bencher) raises:
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            ctx.enqueue_function(
                func_gemv,
                c_buf.buffer.ptr,
                a_buf.buffer.ptr,
                b_buf.buffer.ptr,
                M,
                N,
                K,
                grid_dim=ceildiv(M, WARPS_PER_BLOCK),
                block_dim=WARP_SIZE * WARPS_PER_BLOCK,
            )

        b.iter_custom[kernel_launch](ctx)

    m.bench_function[bench_fn](
        BenchId("gemv", input_id=fn_name + "/" + str(type) + "/" + str(dims)),
    )

    ctx.synchronize()

    _ = a_buf
    _ = b_buf
    _ = c_buf

    a_host.free()
    b_host.free()
    c_host.free()

    _ = func_gemv^


fn bench_gemv_naive[
    type: DType
](
    inout m: Bench, fn_name: String, dims: StaticIntTuple[3], ctx: DeviceContext
) raises:
    var M = dims[0]
    var N = dims[1]
    var K = dims[2]
    var a_host = UnsafePointer[Scalar[type]].alloc(M * K)
    var b_host = UnsafePointer[Scalar[type]].alloc(K * N)
    var c_host = UnsafePointer[Scalar[type]].alloc(M * N)

    randn(a_host, M * K)
    randn(b_host, K * N)
    memset(c_host, 0, M * N)

    var a_buf_h = NDBuffer[type, 2](a_host, StaticIntTuple[2](M, K))
    var b_buf_h = NDBuffer[type, 2](b_host, StaticIntTuple[2](K, N))
    var c_buf_h = NDBuffer[type, 2](c_host, StaticIntTuple[2](M, N))

    var a_buf = DeviceNDBuffer[type, 2](StaticIntTuple[2](M, K), ctx=ctx)
    var b_buf = DeviceNDBuffer[type, 2](StaticIntTuple[2](K, N), ctx=ctx)
    var c_buf = DeviceNDBuffer[type, 2](StaticIntTuple[2](M, N), ctx=ctx)

    ctx.enqueue_copy_to_device(a_buf.buffer, a_buf_h.data)
    ctx.enqueue_copy_to_device(b_buf.buffer, b_buf_h.data)

    alias BLOCK_DIM = 16
    alias WARPS_PER_BLOCK = 32
    var func_gemv = ctx.compile_function[
        matmul_kernel_naive[
            type,
            type,
            type,
            BLOCK_DIM,
        ]
    ]()

    @always_inline
    @__copy_capture(M, N, K)
    @parameter
    fn bench_fn(inout b: Bencher) raises:
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            ctx.enqueue_function(
                func_gemv,
                c_buf.buffer.ptr,
                a_buf.buffer.ptr,
                b_buf.buffer.ptr,
                M,
                N,
                K,
                grid_dim=(ceildiv(M, WARPS_PER_BLOCK), ceildiv(N, BLOCK_DIM)),
                block_dim=(BLOCK_DIM, BLOCK_DIM),
            )

        b.iter_custom[kernel_launch](ctx)

    m.bench_function[bench_fn](
        BenchId("gemv", input_id=fn_name + "/" + str(type) + "/" + str(dims)),
    )

    ctx.synchronize()

    _ = a_buf
    _ = b_buf
    _ = c_buf

    a_host.free()
    b_host.free()
    c_host.free()

    _ = func_gemv^


@value
struct GemvSpec(Stringable):
    var m: Int
    var n: Int
    var k: Int

    @no_inline
    fn __str__(self) -> String:
        return "m=" + str(self.m) + ";n=" + str(self.n) + ";k=" + str(self.k)

    fn flops(self) -> Int:
        return 2 * self.m * self.n * self.k


def main():
    with DeviceContext() as ctx:
        var m = Bench(BenchConfig(num_repetitions=1))

        var shape_list = List[StaticIntTuple[3]](
            StaticIntTuple[3](256, 1, 256),
        )

        for s in range(len(shape_list)):
            bench_gemv_naive[DType.bfloat16](
                m, "gemv_naive", shape_list[s], ctx
            )
            bench_gemv_ws[DType.bfloat16](m, "gemv_ws", shape_list[s], ctx)
            bench_gemv_tc[DType.bfloat16](m, "gemv_tc", shape_list[s], ctx)

        bench_compile_time[bench_gemv_naive[DType.bfloat16]](m, "gemv_naive")
        bench_compile_time[bench_gemv_ws[DType.bfloat16]](m, "gemv_ws")
        bench_compile_time[bench_gemv_tc[DType.bfloat16]](m, "gemv_tc")

        m.dump_report()
