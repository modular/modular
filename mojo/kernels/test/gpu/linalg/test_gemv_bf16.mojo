# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s | FileCheck %s

from math import ceildiv

from gpu import WARP_SIZE
from gpu.host import Context, Function, Stream
from gpu.host.event import time_function
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
)
from linalg.matmul_gpu import gemv_tc_kernel, matmul_kernel_naive
from memory.unsafe import DTypePointer

from utils.numerics import isnan


fn run_matvec(M: Int, N: Int, K: Int) raises:
    print("== run_matvec kernel")

    var iterations = 100
    var stream = Stream()
    var a_host = UnsafePointer[BFloat16].alloc(M * K)
    var b_host = UnsafePointer[BFloat16].alloc(K * N)
    var c_host = UnsafePointer[Float32].alloc(M * N)
    var a_host_n = UnsafePointer[Float32].alloc(M * K)
    var b_host_n = UnsafePointer[Float32].alloc(K * N)
    var c_host_n = UnsafePointer[Float32].alloc(M * N)

    for i in range(M * K):
        a_host[i] = i
        a_host_n[i] = i

    for i in range(K * N):
        b_host[i] = i + 1
        b_host_n[i] = i + 1

    for i in range(M * N):
        c_host[i] = 0

    for i in range(M * N):
        c_host_n[i] = 0

    var a_device = _malloc[BFloat16](M * K)
    var b_device = _malloc[BFloat16](K * N)
    var c_device = _malloc[Float32](M * N)
    var a_device_n = _malloc[Float32](M * K)
    var b_device_n = _malloc[Float32](K * N)
    var c_device_n = _malloc[Float32](M * N)

    _copy_host_to_device(a_device, a_host, M * K)
    _copy_host_to_device(b_device, b_host, K * N)

    alias WARPS_PER_BLOCK = 32
    var func_gemv = Function[
        gemv_tc_kernel[
            DType.float32,
            DType.bfloat16,
            DType.bfloat16,
        ]
    ]()

    @always_inline
    @__copy_capture(func_gemv, c_device, a_device, b_device)
    @parameter
    fn run_func_gemv(stream: Stream) raises:
        func_gemv(
            c_device,
            a_device,
            b_device,
            M,
            N,
            K,
            grid_dim=ceildiv(M, WARPS_PER_BLOCK),
            block_dim=WARP_SIZE * WARPS_PER_BLOCK,
            stream=stream,
        )

    var nstime = 0.0
    var kernelType = "GEMV"
    for i in range(iterations):
        nstime += time_function[run_func_gemv](stream)
    var flops = 2 * M * N * K
    var sectime = ((nstime / iterations) / 1000000000)
    print(kernelType, "KERNEL:")
    print(sectime, "sec")
    print(flops * 1e-9 / sectime, " GFLOPS")
    print()

    _copy_device_to_host(c_host, c_device, M * N)

    # running naive
    _copy_host_to_device(a_device_n, a_host_n, M * K)
    _copy_host_to_device(b_device_n, b_host_n, K * N)

    alias BLOCK_DIM = 16
    var func_naive = Function[
        matmul_kernel_naive[
            DType.float32,
            DType.float32,
            DType.float32,
            BLOCK_DIM,
        ]
    ]()

    @always_inline
    @__copy_capture(func_naive, c_device_n, a_device_n, b_device_n)
    @parameter
    fn run_func_naive(stream: Stream) raises:
        func_naive(
            c_device_n,
            a_device_n,
            b_device_n,
            M,
            N,
            K,
            grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
            block_dim=(BLOCK_DIM, BLOCK_DIM),
            stream=stream,
        )

    nstime = 0.0
    for i in range(iterations):
        nstime += time_function[run_func_naive](stream)
    var sectime2 = ((nstime / iterations) / 1000000000)
    print("SHMEM MATMUL:")
    print(sectime2, "sec")
    print(flops * 1e-9 / sectime2, " GFLOPS")
    print()

    _copy_device_to_host(c_host_n, c_device_n, M * N)

    # Due to varied pattern of FP arith the accumulated sum isn't exactly accurate. Hence relative tolerance needs to be checked.
    var errorTolerance = 0.1
    var failed = False
    for i in range(M * N):
        var outVal = c_host[i]
        var outRef = c_host_n[i]
        var relDiff = (max(outVal, outRef) / min(outVal, outRef)) - 1.0
        if (relDiff > errorTolerance) or isnan(outVal) or isnan(outRef):
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

    _free(a_device_n)
    _free(b_device_n)
    _free(c_device_n)

    _ = a_host
    _ = b_host
    _ = c_host

    _ = a_host_n
    _ = b_host_n
    _ = c_host_n

    _ = func_gemv^
    _ = func_naive^
    _ = stream^


# CHECK-NOT: CUDA_ERROR
def main():
    try:
        with Context() as ctx:
            run_matvec(4096, 1, 4096)

    except e:
        print("CUDA_ERROR:", e)
