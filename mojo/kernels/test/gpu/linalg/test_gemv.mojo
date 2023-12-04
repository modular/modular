# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo %s | FileCheck %s

from math import div_ceil, min, max
from gpu.host import Context, Dim, Function, Stream, synchronize
import gpu.host.benchmark
from gpu.host.event import time_function
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
)
from gpu import GridDim, BlockIdx, BlockDim, ThreadIdx, barrier, WARP_SIZE
from gpu.sync import syncwarp
from memory.buffer import NDBuffer
from memory.unsafe import DTypePointer, bitcast
from Matmul import gemv_kernel, gevm_kernel, matmul_kernel
from utils.index import Index
from utils.list import DimList


fn run_matvec(M: Int, N: Int, K: Int) raises:
    print("== run_matvec kernel")

    let iterations = 100
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

    alias WARPS_PER_BLOCK = 32
    let func_gemv = Function[
        fn (
            DTypePointer[DType.float32],
            DTypePointer[DType.float32],
            DTypePointer[DType.float32],
            Int,
            Int,
            Int,
        ) capturing -> None, gemv_kernel[
            DType.float32,
            DType.float32,
            DType.float32,
        ]
    ]()

    let func_gevm = Function[
        fn (
            DTypePointer[DType.float32],
            DTypePointer[DType.float32],
            DTypePointer[DType.float32],
            Int,
            Int,
            Int,
        ) capturing -> None, gevm_kernel[
            DType.float32,
            DType.float32,
            DType.float32,
            WARP_SIZE * WARPS_PER_BLOCK,
        ]
    ]()

    @always_inline
    @parameter
    fn run_func_gemv(stream: Stream) raises:
        func_gemv(
            div_ceil(M, WARPS_PER_BLOCK),
            WARP_SIZE * WARPS_PER_BLOCK,
            c_device,
            a_device,
            b_device,
            M,
            N,
            K,
            stream=stream,
        )

    @always_inline
    @parameter
    fn run_func_gevm(stream: Stream) raises:
        func_gevm(
            div_ceil(N, WARPS_PER_BLOCK),
            WARP_SIZE * WARPS_PER_BLOCK,
            c_device,
            a_device,
            b_device,
            M,
            N,
            K,
            stream=stream,
        )

    var nstime = 0.0
    var kernelType = ""
    if N == 1:
        for i in range(iterations):
            nstime += time_function[run_func_gemv](stream)
        kernelType = "GEMV"
    elif M == 1:
        for i in range(iterations):
            nstime += time_function[run_func_gevm](stream)
        kernelType = "GEVM"
    else:
        print("Incorrect input shape [MNK]")
        return
    let flops = 2 * M * N * K
    let sectime = ((nstime / iterations) / 1000000000)
    print(kernelType, "KERNEL:")
    print(sectime, "sec")
    print(flops * 1e-9 / sectime, " GFLOPS")
    print()

    _copy_device_to_host(c_host, c_device, M * N)

    # running naive
    _copy_host_to_device(a_device, a_host, M * K)
    _copy_host_to_device(b_device, b_host, K * N)

    alias BLOCK_DIM = 16
    let func_naive = Function[
        fn (
            DTypePointer[DType.float32],
            DTypePointer[DType.float32],
            DTypePointer[DType.float32],
            Int,
            Int,
            Int,
        ) capturing -> None, matmul_kernel[
            DType.float32,
            DType.float32,
            DType.float32,
            BLOCK_DIM,
        ]
    ]()

    @always_inline
    @parameter
    fn run_func_naive(stream: Stream) raises:
        func_naive(
            (div_ceil(M, BLOCK_DIM), div_ceil(N, BLOCK_DIM)),
            (BLOCK_DIM, BLOCK_DIM),
            c_device,
            a_device,
            b_device,
            M,
            N,
            K,
            stream=stream,
        )

    nstime = 0.0
    for i in range(iterations):
        nstime += time_function[run_func_naive](stream)
    let sectime2 = ((nstime / iterations) / 1000000000)
    print("SHMEM MATMUL:")
    print(sectime2, "sec")
    print(flops * 1e-9 / sectime2, " GFLOPS")
    print()

    _copy_device_to_host(c_host_naive, c_device, M * N)

    # Due to varied pattern of FP32 arith the accumulated sum isn't exactly accurate. Hence relative tolerance needs to be checked.
    let errorTolerance = 0.0001
    var failed = False
    for i in range(M * N):
        var outVal = c_host.load(i)
        var outRef = c_host_naive.load(i)
        let relDiff = (max(outVal, outRef) / min(outVal, outRef)) - 1.0
        if (
            (relDiff > errorTolerance)
            or math.isnan(outVal)
            or math.isnan(outRef)
        ):
            failed = True

    # CHECK: Success
    if not failed:
        print("Success 🎉: results match")
        print(
            "Performance warp-shuffle matvec vs. shmem matmul: ",
            sectime2 / sectime,
            "x",
        )
    else:
        print("Failed ❌: results mismatch")

    _free(a_device)
    _free(b_device)
    _free(c_device)

    _ = a_host
    _ = b_host
    _ = c_host
    _ = c_host_naive

    _ = func_gemv ^
    _ = func_gevm ^
    _ = func_naive ^
    _ = stream ^


# CHECK-NOT: CUDA_ERROR
def main():
    try:
        with Context() as ctx:
            # gemv for matrix vector multiply and gevm for vector matrix multiply
            run_matvec(4096, 1, 4096)
            run_matvec(1, 4096, 4096)

    except e:
        print("CUDA_ERROR:", e)
