# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s | FileCheck %s

from math import ceildiv
from random import randn, seed

from buffer import NDBuffer
from gpu import WARP_SIZE
from gpu.host import Context, Function, Stream
from gpu.host.event import time_function
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
)
from LinAlg.MatmulGPU import (
    gemv_kernel,
    gevm_kernel,
    matmul_kernel,
    matmul_kernel_naive,
)
from memory.unsafe import DTypePointer

from utils.index import Index
from utils.numerics import isnan


fn run_matvec(M: Int, N: Int, K: Int) raises:
    print("== run_matvec kernel")

    var iterations = 100
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

    alias WARPS_PER_BLOCK = 32
    var func_gemv = Function[
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

    var func_gevm = Function[
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

    @always_inline
    @__copy_capture(func_gevm, c_device, a_device, b_device)
    @parameter
    fn run_func_gevm(stream: Stream) raises:
        func_gevm(
            c_device,
            a_device,
            b_device,
            M,
            N,
            K,
            grid_dim=ceildiv(N, WARPS_PER_BLOCK),
            block_dim=WARP_SIZE * WARPS_PER_BLOCK,
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
    var flops = 2 * M * N * K
    var sectime = ((nstime / iterations) / 1000000000)
    print(kernelType, "KERNEL:")
    print(sectime, "sec")
    print(flops * 1e-9 / sectime, " GFLOPS")
    print()

    _copy_device_to_host(c_host, c_device, M * N)

    # running naive
    _copy_host_to_device(a_device, a_host, M * K)
    _copy_host_to_device(b_device, b_host, K * N)

    alias BLOCK_DIM = 16
    var func_naive = Function[
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
    @__copy_capture(func_naive, c_device, a_device, b_device)
    @parameter
    fn run_func_naive(stream: Stream) raises:
        func_naive(
            c_device,
            a_device,
            b_device,
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

    _copy_device_to_host(c_host_naive, c_device, M * N)

    # Due to varied pattern of FP32 arith the accumulated sum isn't exactly accurate. Hence relative tolerance needs to be checked.
    var errorTolerance = 0.0001
    var failed = False
    for i in range(M * N):
        var outVal = c_host.load(i)
        var outRef = c_host_naive.load(i)
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

    _ = a_host
    _ = b_host
    _ = c_host
    _ = c_host_naive

    _ = func_gemv^
    _ = func_gevm^
    _ = func_naive^
    _ = stream^


fn test_gevm_with_epilogue_fn(M: Int, N: Int, K: Int) raises:
    alias c_stride = 5
    alias seed_val = 42

    var iterations = 100
    var stream = Stream()
    var a_host = DTypePointer[DType.float32].alloc(M * K)
    var b_host = DTypePointer[DType.float32].alloc(K * N)

    seed(seed_val)

    # over-allocate C to simulate a view tensor
    var c_host = DTypePointer[DType.float32].alloc(M * N * c_stride)
    var c_host_naive = DTypePointer[DType.float32].alloc(M * N * c_stride)

    randn(a_host, M * K)

    randn(b_host, K * N)

    for i in range(M * N * c_stride):
        c_host[i] = 0

    for i in range(M * N * c_stride):
        c_host_naive[i] = 0

    var a_device = _malloc[Float32](M * K)
    var b_device = _malloc[Float32](K * N)
    var c_device = _malloc[Float32](M * N * c_stride)

    var c_device_nd = NDBuffer[DType.float32, 2](
        c_device, Index(M, N), Index(N * c_stride, c_stride)
    )
    _copy_host_to_device(a_device, a_host, M * K)
    _copy_host_to_device(b_device, b_host, K * N)

    @parameter
    @__copy_capture(c_device_nd)
    fn epilogue_fn[
        type: DType, width: Int
    ](idx: StaticIntTuple[2], val: SIMD[type, width]):
        c_device_nd.store[width=width](
            idx, rebind[SIMD[DType.float32, width]](val + 4.0)
        )

    alias WARPS_PER_BLOCK = 32
    var func_gemv = Function[
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
            elementwise_lambda_fn=epilogue_fn,
        ]
    ]()

    var func_gevm = Function[
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
            elementwise_lambda_fn=epilogue_fn,
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

    @always_inline
    @__copy_capture(func_gevm, c_device, a_device, b_device)
    @parameter
    fn run_func_gevm(stream: Stream) raises:
        func_gevm(
            c_device,
            a_device,
            b_device,
            M,
            N,
            K,
            grid_dim=ceildiv(N, WARPS_PER_BLOCK),
            block_dim=WARP_SIZE * WARPS_PER_BLOCK,
            stream=stream,
        )

    _copy_host_to_device(c_device, c_host, M * N * c_stride)

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

    var flops = 2 * M * N * K
    var sectime = ((nstime / iterations) / 1000000000)

    print(kernelType, "KERNEL:")
    print(sectime, "sec")
    print(flops * 1e-9 / sectime, " GFLOPS")
    print()

    _copy_device_to_host(c_host, c_device, M * N * c_stride)

    # running naive
    _copy_host_to_device(a_device, a_host, M * K)
    _copy_host_to_device(b_device, b_host, K * N)

    alias BLOCK_DIM = 16
    var func_naive = Function[
        fn (
            DTypePointer[DType.float32],
            DTypePointer[DType.float32],
            DTypePointer[DType.float32],
            Int,
            Int,
            Int,
        ) capturing -> None, matmul_kernel_naive[
            DType.float32,
            DType.float32,
            DType.float32,
            BLOCK_DIM,
            elementwise_lambda_fn=epilogue_fn,
        ]
    ]()

    @always_inline
    @__copy_capture(func_naive, c_device, a_device, b_device)
    @parameter
    fn run_func_naive(stream: Stream) raises:
        func_naive(
            c_device,
            a_device,
            b_device,
            M,
            N,
            K,
            grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
            block_dim=(BLOCK_DIM, BLOCK_DIM),
            stream=stream,
        )

    _copy_host_to_device(c_device, c_host_naive, M * N * c_stride)

    nstime = 0.0
    for i in range(iterations):
        nstime += time_function[run_func_naive](stream)
    var sectime2 = ((nstime / iterations) / 1000000000)
    print("NAIVE MATMUL:")
    print(sectime2, "sec")
    print(flops * 1e-9 / sectime2, " GFLOPS")
    print()

    _copy_device_to_host(c_host_naive, c_device, M * N * c_stride)

    # Due to varied pattern of FP32 arith the accumulated sum isn't exactly accurate. Hence relative tolerance needs to be checked.
    var errorTolerance = 0.005
    var failed = False
    for i in range(M * N * c_stride):
        var outVal = c_host.load(i)
        var outRef = c_host_naive.load(i)
        var relDiff = (max(outVal, outRef) / min(outVal, outRef)) - 1.0
        if (relDiff > errorTolerance) or isnan(outVal) or isnan(outRef):
            print(i, relDiff, outVal, outRef)
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

    _ = func_gevm^
    _ = func_naive^
    _ = stream^


# CHECK-NOT: CUDA_ERROR
def main():
    try:
        with Context() as ctx:
            # gemv for matrix vector multiply and gevm for vector matrix multiply
            run_matvec(4096, 1, 4096)
            run_matvec(1, 4096, 4096)
            test_gevm_with_epilogue_fn(1, 4096, 4096)
            test_gevm_with_epilogue_fn(4096, 1, 4096)
    except e:
        print("CUDA_ERROR:", e)
