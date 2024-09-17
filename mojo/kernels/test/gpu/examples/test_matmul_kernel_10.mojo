# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# TODO (#33518): -t flag is required right now because the kernel assumes C is zeroed
# RUN: %mojo-no-debug %s -t | FileCheck %s

from math import ceildiv

from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu import WARP_SIZE, BlockDim, BlockIdx, ThreadIdx
from gpu.host.device_context import DeviceContext
from linalg.matmul_gpu import sgemm_warp_tiling_kernel
from memory import UnsafePointer

from utils.index import Index
from utils.numerics import isnan

alias BLOCK_DIM = 8


fn matmul_naive(
    a_ptr: UnsafePointer[Float32],
    b_ptr: UnsafePointer[Float32],
    c_ptr: UnsafePointer[Float32],
    m: Int,
    n: Int,
    k: Int,
):
    var x: UInt = BlockIdx.x() * BlockDim.x() + ThreadIdx.x()
    var y: UInt = BlockIdx.y() * BlockDim.y() + ThreadIdx.y()

    if x >= m or y >= n:
        return

    var a = NDBuffer[DType.float32, 2](a_ptr, Index(m, k))
    var b = NDBuffer[DType.float32, 2](b_ptr, Index(k, n))
    var c = NDBuffer[DType.float32, 2](c_ptr, Index(m, n))

    var accum = Float32(0)
    for i in range(k):
        accum = a[x, i] * b[i, y] + accum
    c[Index(x, y)] = accum


# CHECK-LABEL: run_matmul_kernel_10
fn bench_matmuls(inout m: Bench, ctx: DeviceContext) raises:
    print("== run_matmul_kernel_10")

    alias M = 4096
    alias N = 4096
    alias K = 4096

    # TODO: Find best for target GPU.
    #       For A100 see below (based on siboehm repo):
    # alias K10_NUM_THREADS = 128
    # alias K10_BN = 128
    # alias K10_BM = 64
    # alias K10_BK = 16
    # alias K10_WN = 64
    # alias K10_WM = 32
    # alias K10_WNITER = 1
    # alias K10_TN = 4
    # alias K10_TM = 4
    # Settings for A6000
    alias K10_NUM_THREADS = 128
    alias K10_BN = 128
    alias K10_BM = 128
    alias K10_BK = 16
    alias K10_WN = 64
    alias K10_WM = 64
    alias K10_WNITER = 4
    alias K10_TN = 4
    alias K10_TM = 8

    alias NUM_WARPS = K10_NUM_THREADS / 32
    alias K10_WMITER = (K10_WM * K10_WN) // (32 * K10_TM * K10_TN * K10_WNITER)

    # Warptile in threadblocktile.
    constrained[(K10_BN % K10_WN == 0) and (K10_BM % K10_WM == 0)]()
    constrained[(K10_BN / K10_WN) * (K10_BM / K10_WM) == NUM_WARPS]()

    # Threads in warpsubtile.
    constrained[
        (K10_WM * K10_WN) % (WARP_SIZE * K10_TM * K10_TN * K10_WNITER) == 0
    ]()

    # Warpsubtile in warptile.
    constrained[(K10_WM % K10_WMITER == 0) and (K10_WN % K10_WNITER == 0)]()

    constrained[
        (K10_NUM_THREADS * 4) % K10_BK == 0,
        (
            "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
            "issues during GMEM->SMEM tiling (loading only parts of the "
            "final row of Bs during each iteraion)"
        ),
    ]()
    constrained[
        (K10_NUM_THREADS * 4) % K10_BN == 0,
        (
            "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
            "issues during GMEM->SMEM tiling (loading only parts of the "
            "final row of As during each iteration)"
        ),
    ]()

    constrained[
        K10_BN % (16 * K10_TN) == 0,
        "BN must be a multiple of 16*TN to avoid quantization effects",
    ]()
    constrained[
        K10_BM % (16 * K10_TM) == 0,
        "BM must be a multiple of 16*TM to avoid quantization effects",
    ]()

    constrained[
        (K10_BM * K10_BK) % (4 * K10_NUM_THREADS) == 0,
        "BM*BK must be a multiple of 4*256 to vectorize loads",
    ]()
    constrained[
        (K10_BN * K10_BK) % (4 * K10_NUM_THREADS) == 0,
        "BN*BK must be a multiple of 4*256 to vectorize loads",
    ]()

    constrained[
        K10_TM % 4 == 0,
        "TM must be a multiple of 4",
    ]()

    constrained[
        K10_TN % 4 == 0,
        "TN must be a multiple of 4",
    ]()

    var a_host = UnsafePointer[Float32].alloc(M * K)
    var b_host = UnsafePointer[Float32].alloc(K * N)
    var c_host = UnsafePointer[Float32].alloc(M * N)
    var c_host_naive = UnsafePointer[Float32].alloc(M * N)

    for i in range(M * K):
        a_host[i] = i

    for i in range(K * N):
        b_host[i] = i + 1

    for i in range(M * N):
        c_host[i] = 0

    for i in range(M * N):
        c_host_naive[i] = 0

    var a_device = ctx.create_buffer[DType.float32](M * K)
    var b_device = ctx.create_buffer[DType.float32](K * N)
    var c_device = ctx.create_buffer[DType.float32](M * N)

    ctx.enqueue_copy_to_device(a_device, a_host)
    ctx.enqueue_copy_to_device(b_device, b_host)

    var c_buffer = NDBuffer[DType.float32, 2, DimList(M, N)](c_device.ptr)
    var a_buffer = NDBuffer[DType.float32, 2, DimList(M, K)](a_device.ptr)
    var b_buffer = NDBuffer[DType.float32, 2, DimList(K, N)](b_device.ptr)

    alias sgemm_type = sgemm_warp_tiling_kernel[
        DType.float32,
        DimList(M, N),
        DType.float32,
        DimList(M, K),
        DType.float32,
        DimList(K, N),
        BM=K10_BM,
        BN=K10_BN,
        BK=K10_BK,
        WM=K10_WM,
        WN=K10_WN,
        WMITER=K10_WMITER,
        WNITER=K10_WNITER,
        TM=K10_TM,
        TN=K10_TN,
        NUM_THREADS=K10_NUM_THREADS,
    ]
    var func = ctx.compile_function[sgemm_type](
        threads_per_block=K10_NUM_THREADS
    )

    @parameter
    @always_inline
    fn bench_matmul_10(inout b: Bencher):
        @parameter
        @always_inline
        fn run_func(ctx: DeviceContext) raises:
            ctx.enqueue_function(
                func,
                c_buffer,
                a_buffer,
                b_buffer,
                Scalar[DType.float32](1),
                Scalar[DType.float32](0),
                grid_dim=(ceildiv(N, K10_BN), ceildiv(M, K10_BM)),
                block_dim=(K10_NUM_THREADS,),
            )

        b.iter_custom[run_func](ctx)

    m.bench_function[bench_matmul_10](
        BenchId("matmul_sgemm_10"),
        ThroughputMeasure(BenchMetric.elements, 2 * M * N * K),
    )
    _ = a_buffer
    _ = b_buffer
    _ = c_buffer

    ctx.enqueue_copy_from_device(c_host, c_device)

    # Perform naive matmul to compare results & performance.

    ctx.enqueue_copy_to_device(a_device, a_host)
    ctx.enqueue_copy_to_device(b_device, b_host)

    var func_naive = ctx.compile_function[matmul_naive]()

    @parameter
    @always_inline
    fn bench_naive(inout b: Bencher):
        @parameter
        @always_inline
        fn run_func_naive(ctx: DeviceContext) raises:
            ctx.enqueue_function(
                func_naive,
                a_device,
                b_device,
                c_device,
                M,
                N,
                K,
                grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
                block_dim=(BLOCK_DIM, BLOCK_DIM),
            )

        b.iter_custom[run_func_naive](ctx)

    m.bench_function[bench_naive](
        BenchId("matmul_naive"),
        # TODO: Pick relevant benchmetric
        ThroughputMeasure(BenchMetric.elements, 2 * M * N * K),
    )

    ctx.enqueue_copy_from_device(c_host_naive, c_device)

    for i in range(M * N):
        if (
            c_host[i] != c_host_naive[i]
            or isnan(c_host_naive[i])
            or isnan(c_host[i])
        ):
            print(c_host[i])
            print(c_host_naive[i])
            raise "Failed ❌: results mismatch"

    _ = a_device
    _ = b_device
    _ = c_device

    _ = a_host
    _ = b_host
    _ = c_host
    _ = c_host_naive

    _ = func^
    _ = func_naive^


def main():
    with DeviceContext() as ctx:
        var m = Bench()
        bench_matmuls(m, ctx)
        m.dump_report()
