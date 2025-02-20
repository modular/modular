# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# FIXME: KERN-1377
# UNSUPPORTED: AMD-GPU
# RUN: %mojo-no-debug-no-assert %s | FileCheck %s

from math import ceildiv

from gpu import WARP_SIZE
import gpu.warp as warp
from gpu.host import DeviceContext
from linalg.gemv import gemv_kernel
from linalg.matmul_gpu import matmul_kernel_naive
from memory import UnsafePointer

from utils.numerics import isnan


fn run_matvec[
    reduction_method: warp.ReductionMethod
](M: Int, N: Int, K: Int, *, ctx: DeviceContext) raises:
    print("== run_matvec kernel")

    var iterations = 100
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

    var a_device = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
    var b_device = ctx.enqueue_create_buffer[DType.bfloat16](K * N)
    var c_device = ctx.enqueue_create_buffer[DType.float32](M * N)
    var a_device_n = ctx.enqueue_create_buffer[DType.float32](M * K)
    var b_device_n = ctx.enqueue_create_buffer[DType.float32](K * N)
    var c_device_n = ctx.enqueue_create_buffer[DType.float32](M * N)

    ctx.enqueue_copy_to_device(a_device, a_host)
    ctx.enqueue_copy_to_device(b_device, b_host)

    alias WARPS_PER_BLOCK = 32
    alias kernel = gemv_kernel[
        DType.float32,
        DType.bfloat16,
        DType.bfloat16,
        reduction_method=reduction_method,
    ]

    @always_inline
    @parameter
    fn run_func_gemv(ctx: DeviceContext) raises:
        ctx.enqueue_function[kernel](
            c_device,
            a_device,
            b_device,
            UInt(M),
            UInt(N),
            UInt(K),
            grid_dim=ceildiv(M, WARPS_PER_BLOCK),
            block_dim=WARP_SIZE * WARPS_PER_BLOCK,
        )

    var nstime = 0.0
    var kernelType = "GEMV"
    nstime = ctx.execution_time[run_func_gemv](iterations)
    var flops = 2 * M * N * K
    var sectime = ((nstime / iterations) / 1000000000)
    print(kernelType, "KERNEL:")
    print(sectime, "sec")
    print(flops * 1e-9 / sectime, " GFLOPS")
    print()

    ctx.enqueue_copy_from_device(c_host, c_device)

    # running naive
    ctx.enqueue_copy_to_device(a_device_n, a_host_n)
    ctx.enqueue_copy_to_device(b_device_n, b_host_n)

    alias BLOCK_DIM = 16

    @always_inline
    @parameter
    fn run_func_naive(ctx: DeviceContext) raises:
        ctx.enqueue_function[
            matmul_kernel_naive[
                DType.float32,
                DType.float32,
                DType.float32,
                BLOCK_DIM,
            ]
        ](
            c_device_n,
            a_device_n,
            b_device_n,
            UInt(M),
            UInt(N),
            UInt(K),
            grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
            block_dim=(BLOCK_DIM, BLOCK_DIM),
        )

    nstime = ctx.execution_time[run_func_naive](iterations)
    var sectime2 = ((nstime / iterations) / 1000000000)
    print("SHMEM MATMUL:")
    print(sectime2, "sec")
    print(flops * 1e-9 / sectime2, " GFLOPS")
    print()

    ctx.enqueue_copy_from_device(c_host_n, c_device_n)
    ctx.synchronize()

    # Due to varied pattern of FP arith the accumulated sum isn't exactly
    # accurate. Hence relative tolerance needs to be checked.
    alias errorTolerance = 0.1
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

    _ = a_device
    _ = b_device
    _ = c_device

    _ = a_device_n
    _ = b_device_n
    _ = c_device_n

    _ = a_host
    _ = b_host
    _ = c_host

    _ = a_host_n
    _ = b_host_n
    _ = c_host_n


def main():
    with DeviceContext() as ctx:
        run_matvec[reduction_method = warp.ReductionMethod.WARP](
            4096, 1, 4096, ctx=ctx
        )
        run_matvec[reduction_method = warp.ReductionMethod.TENSOR_CORE](
            4096, 1, 4096, ctx=ctx
        )
