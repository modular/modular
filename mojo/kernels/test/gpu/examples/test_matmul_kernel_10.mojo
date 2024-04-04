# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# TODO (#33518): -t flag is required right now because the kernel assumes C is zeroed
# RUN: %mojo %s -t | FileCheck %s

from math import div_ceil

from benchmark import Bench, Bencher, BenchId
from benchmark._cuda import time_async_cuda_kernel
from buffer import NDBuffer
from buffer.list import DimList
from gpu import WARP_SIZE, BlockDim, BlockIdx, ThreadIdx, barrier
from gpu.host import Context, Dim, Function, Stream, synchronize
from gpu.host.event import time_function
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
)
from gpu.memory import AddressSpace
from LinAlg.MatmulGPU import sgemm_warp_tiling_kernel
from memory import memset_zero, stack_allocation
from memory.unsafe import DTypePointer, bitcast
from tensor import Tensor

from utils.index import Index

alias BLOCK_DIM = 8


fn matmul_naive(
    a_ptr: DTypePointer[DType.float32],
    b_ptr: DTypePointer[DType.float32],
    c_ptr: DTypePointer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
):
    var x = BlockIdx.x() * BlockDim.x() + ThreadIdx.x()
    var y = BlockIdx.y() * BlockDim.y() + ThreadIdx.y()

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
fn bench_matmuls(inout m: Bench) raises:
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

    var stream = Stream()

    var a_host = Pointer[Float32].alloc(M * K)
    var b_host = Pointer[Float32].alloc(K * N)
    var c_host = Pointer[Float32].alloc(M * N)
    var c_host_naive = Pointer[Float32].alloc(M * N)

    for i in range(M * K):
        a_host[i] = i

    for i in range(K * N):
        b_host[i] = i + 1

    for i in range(M * N):
        c_host[i] = 0

    for i in range(M * N):
        c_host_naive[i] = 0

    var a_device = _malloc[Float32](M * K)
    var b_device = _malloc[Float32](K * N)
    var c_device = _malloc[Float32](M * N)

    _copy_host_to_device(a_device, a_host, M * K)
    _copy_host_to_device(b_device, b_host, K * N)

    var c_buffer = NDBuffer[DType.float32, 2, DimList(M, N)](c_device)
    var a_buffer = NDBuffer[DType.float32, 2, DimList(M, K)](a_device)
    var b_buffer = NDBuffer[DType.float32, 2, DimList(K, N)](b_device)

    alias sgemm_type = sgemm_warp_tiling_kernel[
        DType.float32,
        DimList(M, N),
        DType.float32,
        DimList(M, K),
        DType.float32,
        DimList(K, N),
        indexing_integral_dtype = DType.uint32,
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
    var func = Function[__type_of(sgemm_type), sgemm_type](
        threads_per_block=K10_NUM_THREADS
    )

    @parameter
    fn bench_matmul_10(inout b: Bencher):
        @always_inline
        @__copy_capture(func, a_buffer, b_buffer, c_buffer)
        @parameter
        fn run_func(stream: Stream) raises:
            func(
                c_buffer,
                a_buffer,
                b_buffer,
                Scalar[DType.float32](1),
                Scalar[DType.float32](0),
                grid_dim=(div_ceil(N, K10_BN), div_ceil(M, K10_BM)),
                block_dim=(K10_NUM_THREADS,),
                stream=stream,
            )

        b.iter_custom[time_async_cuda_kernel[run_func]]()

    m.bench_function[bench_matmul_10](
        BenchId("matmul_sgemm_10"), throughput_elems=2 * M * N * K
    )

    _copy_device_to_host(c_host, c_device, M * N)

    # Perform naive matmul to compare results & performance.

    _copy_host_to_device(a_device, a_host, M * K)
    _copy_host_to_device(b_device, b_host, K * N)

    var func_naive = Function[__type_of(matmul_naive), matmul_naive]()

    @parameter
    fn bench_naive(inout b: Bencher):
        @always_inline
        @__copy_capture(func_naive, a_device, b_device, c_device)
        @parameter
        fn run_func_naive(stream: Stream) raises:
            func_naive(
                a_device,
                b_device,
                c_device,
                M,
                N,
                K,
                grid_dim=(div_ceil(M, BLOCK_DIM), div_ceil(N, BLOCK_DIM)),
                block_dim=(BLOCK_DIM, BLOCK_DIM),
                stream=stream,
            )

        b.iter_custom[time_async_cuda_kernel[run_func_naive]]()

    m.bench_function[bench_naive](
        BenchId("matmul_naive"), throughput_elems=2 * M * N * K
    )

    _copy_device_to_host(c_host_naive, c_device, M * N)

    var failed = False
    for i in range(M * N):
        if (
            c_host.load(i) != c_host_naive.load(i)
            or math.isnan(c_host_naive.load(i))
            or math.isnan(c_host.load(i))
        ):
            print(c_host.load(i))
            print(c_host_naive.load(i))
            raise "Failed ❌: results mismatch"

    _free(a_device)
    _free(b_device)
    _free(c_device)

    _ = a_host
    _ = b_host
    _ = c_host
    _ = c_host_naive

    _ = func^
    _ = stream^


# CHECK-NOT: CUDA_ERROR
def main():
    var m = Bench()
    try:
        with Context() as ctx:
            bench_matmuls(m)
    except e:
        print("CUDA_ERROR:", e)
    m.dump_report()
