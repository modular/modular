# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo %s | FileCheck %s

from math import div_ceil
from gpu.host import Context, Dim, Function, Stream, synchronize
import gpu.host.benchmark
from gpu.host.event import time_function
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
)
from gpu import BlockIdx, BlockDim, ThreadIdx, barrier, WARP_SIZE
from gpu.memory import AddressSpace
from memory import memset_zero, stack_allocation
from memory.buffer import NDBuffer
from memory.unsafe import DTypePointer, bitcast
from tensor import Tensor

from Matmul import sgemm_warp_tiling_kernel

from utils.index import Index
from utils.list import DimList

alias BLOCK_DIM = 8


fn matmul_naive(
    a_ptr: DTypePointer[DType.float32],
    b_ptr: DTypePointer[DType.float32],
    c_ptr: DTypePointer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
):
    let x = BlockIdx.x() * BlockDim.x() + ThreadIdx.x()
    let y = BlockIdx.y() * BlockDim.y() + ThreadIdx.y()

    if x >= m or y >= n:
        return

    let a = NDBuffer[2, DimList.create_unknown[2](), DType.float32](
        a_ptr, Index(m, k)
    )
    let b = NDBuffer[2, DimList.create_unknown[2](), DType.float32](
        b_ptr, Index(k, n)
    )
    let c = NDBuffer[2, DimList.create_unknown[2](), DType.float32](
        c_ptr, Index(m, n)
    )

    var accum = Float32(0)
    for i in range(k):
        accum = a[x, i] * b[i, y] + accum
    c[Index(x, y)] = accum


# CHECK-LABEL: run_matmul_kernel_10
fn run_matmul_kernel_10() raises:
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

    let stream = Stream()

    var a_host = Pointer[Float32].alloc(M * K)
    var b_host = Pointer[Float32].alloc(K * N)
    var c_host = Pointer[Float32].alloc(M * N)
    var c_host_naive = Pointer[Float32].alloc(M * N)

    for i in range(M * K):
        a_host.store(i, i)

    for i in range(K * N):
        b_host.store(i, i + 1)

    for i in range(M * N):
        c_host.store(i, 0)

    for i in range(M * N):
        c_host_naive.store(i, 0)

    let a_device = _malloc[Float32](M * K)
    let b_device = _malloc[Float32](K * N)
    let c_device = _malloc[Float32](M * N)

    _copy_host_to_device(a_device, a_host, M * K)
    _copy_host_to_device(b_device, b_host, K * N)

    let c_buffer = NDBuffer[2, DimList(M, N), DType.float32](c_device)
    let a_buffer = NDBuffer[2, DimList(M, K), DType.float32](a_device)
    let b_buffer = NDBuffer[2, DimList(K, N), DType.float32](b_device)

    let func = Function[
        fn (
            NDBuffer[2, DimList(M, N), DType.float32],
            NDBuffer[2, DimList(M, K), DType.float32],
            NDBuffer[2, DimList(K, N), DType.float32],
        ) -> None, sgemm_warp_tiling_kernel[
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
    ](threads_per_block=K10_NUM_THREADS)

    @always_inline
    @parameter
    fn run_func() raises:
        func(
            (div_ceil(N, K10_BN), div_ceil(M, K10_BM)),
            (K10_NUM_THREADS,),
            c_buffer,
            a_buffer,
            b_buffer,
            stream=stream,
        )

    var nstime = time_function[run_func]()
    let flops = 2 * M * N * K
    let sectime = nstime / 1000000000
    print("WARP-TILING MATMUL:")
    print(sectime, "sec")
    print(flops * 1e-9 / sectime, " GFLOPS")
    print()

    _copy_device_to_host(c_host, c_device, M * N)

    # Perform naive matmul to compare results & performance.

    _copy_host_to_device(a_device, a_host, M * K)
    _copy_host_to_device(b_device, b_host, K * N)

    let func_naive = Function[
        fn (
            DTypePointer[DType.float32],
            DTypePointer[DType.float32],
            DTypePointer[DType.float32],
            Int,
            Int,
            Int,
        ) -> None, matmul_naive
    ]()

    @always_inline
    @parameter
    fn run_func_naive() raises:
        func_naive(
            (div_ceil(M, BLOCK_DIM), div_ceil(N, BLOCK_DIM)),
            (BLOCK_DIM, BLOCK_DIM),
            a_device,
            b_device,
            c_device,
            M,
            N,
            K,
            stream=stream,
        )

    nstime = time_function[run_func_naive]()
    let sectime2 = nstime / 1000000000
    print("NAIVE MATMUL:")
    print(sectime2, "sec")
    print(flops * 1e-9 / sectime2, " GFLOPS")
    print()

    _copy_device_to_host(c_host_naive, c_device, M * N)

    var failed = False
    for i in range(M * N):
        if (
            c_host.load(i) != c_host_naive.load(i)
            or math.isnan(c_host_naive.load(i))
            or math.isnan(c_host.load(i))
        ):
            failed = True

    # CHECK: Success
    if not failed:
        print("Success 🎉: results match")
        print("Performance warp-tiling vs. naive: ", sectime2 / sectime, "x")
    else:
        print("Failed ❌: results mismatch")

    _free(a_device)
    _free(b_device)
    _free(c_device)

    _ = a_host
    _ = b_host
    _ = c_host
    _ = c_host_naive

    _ = func ^
    _ = stream ^


# CHECK-NOT: CUDA_ERROR
def main():
    try:
        with Context() as ctx:
            run_matmul_kernel_10()
    except e:
        print("CUDA_ERROR:", e)
