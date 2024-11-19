# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from math import ceildiv
from random import random_si64

from sys.info import is_amd_gpu
from gpu import WARP_SIZE, BlockDim, BlockIdx, ThreadIdx, barrier
from gpu.host import DeviceContext
from gpu.mma import mma
from linalg.matmul_gpu import matmul_kernel_naive
from memory import UnsafePointer
from utils.numerics import isnan

alias M = 16
alias N = 16
alias K = 16
alias LDA = K
alias LDB = N
alias LDD = N


fn matmul_naive[
    a_type: DType, b_type: DType, c_type: DType
](
    a: UnsafePointer[Scalar[a_type]],
    b: UnsafePointer[Scalar[b_type]],
    c: UnsafePointer[Scalar[c_type]],
    m: Int,
    n: Int,
    k: Int,
):
    for i in range(m):
        for l in range(k):
            for j in range(n):
                var av = a[k * i + l].cast[c_type]()
                var bv = b[n * l + j].cast[c_type]()
                c[n * i + j] += av * bv


# FP32-FP32 Matrix core Matmul with shape m16n16k4
fn mma_kernel_fp32_fp32(
    a_ptr: UnsafePointer[Float32],
    b_ptr: UnsafePointer[Float32],
    c_ptr: UnsafePointer[Float32],
    m: Int,
    n: Int,
    k: Int,
):
    var d_reg: SIMD[DType.float32, 4] = 0

    var a_idx = LDA * ThreadIdx.x() + ThreadIdx.y()
    var b_idx = ThreadIdx.x() + LDB * ThreadIdx.y()

    for i in range(4):
        var a_reg = a_ptr[a_idx + 4 * i]
        var b_reg = b_ptr[b_idx + 4 * LDB * i]

        mma(d_reg, a_reg, b_reg, d_reg)

    for i in range(4):
        var d_idx = ThreadIdx.x() + i * LDD + 4 * LDD * ThreadIdx.y()
        c_ptr[d_idx] = d_reg[i]


fn run_mma_fp32_fp32(
    M: Int,
    N: Int,
    K: Int,
    rand_min: Int64,
    rand_max: Int64,
    ctx: DeviceContext,
) raises:
    print("== run_matmul fp32.fp32 matrix core kernel")

    var errors = 0

    @parameter
    if is_amd_gpu():
        var a_host = UnsafePointer[Float32].alloc(M * K)
        var b_host = UnsafePointer[Float32].alloc(K * N)
        var c_host = UnsafePointer[Float32].alloc(M * N)
        var c_host_ref = UnsafePointer[Float32].alloc(M * N)

        for i in range(M * K):
            var val = random_si64(rand_min, rand_max)
            a_host[i] = val.cast[DType.float32]()

        for i in range(K * N):
            var val = random_si64(rand_min, rand_max)
            b_host[i] = val.cast[DType.float32]()

        for i in range(M * N):
            c_host[i] = 0
            c_host_ref[i] = 0

        var a_device = ctx.enqueue_create_buffer[DType.float32](M * K)
        var b_device = ctx.enqueue_create_buffer[DType.float32](K * N)
        var c_device = ctx.enqueue_create_buffer[DType.float32](M * N)

        ctx.enqueue_copy_to_device(a_device, a_host)
        ctx.enqueue_copy_to_device(b_device, b_host)

        var func_mma = ctx.compile_function[mma_kernel_fp32_fp32]()

        ctx.enqueue_function(
            func_mma,
            a_device,
            b_device,
            c_device,
            M,
            N,
            K,
            grid_dim=1,
            block_dim=(16, 4),
        )

        ctx.enqueue_copy_from_device(c_host, c_device)

        matmul_naive(a_host, b_host, c_host_ref, M, N, K)

        for i in range(M * N):
            if c_host[i] != c_host_ref[i]:
                errors += 1

        _ = a_device
        _ = b_device
        _ = c_device
        _ = a_host
        _ = b_host
        _ = c_host
        _ = c_host_ref
        _ = func_mma^

    # CHECK: Success
    if errors == 0:
        print("Success 🎉: Results match.")
    else:
        print("Failed ❌: results mismatch.")


# FP32-FP32 Matrix core Matmul with shape m16n16k4
fn mma_kernel_fp32_fp16(
    a_ptr: UnsafePointer[Float16],
    b_ptr: UnsafePointer[Float16],
    c_ptr: UnsafePointer[Float32],
    m: Int,
    n: Int,
    k: Int,
):
    var a_reg: SIMD[DType.float16, 4] = 0
    var b_reg: SIMD[DType.float16, 4] = 0
    var d_reg: SIMD[DType.float32, 4] = 0
    for i in range(4):
        var a_idx = LDA * ThreadIdx.x() + i + 4 * ThreadIdx.y()
        var b_idx = ThreadIdx.x() + LDB * i + 4 * LDB * ThreadIdx.y()
        a_reg[i] = a_ptr[a_idx]
        b_reg[i] = b_ptr[b_idx]
    mma(d_reg, a_reg, b_reg, d_reg)
    for i in range(4):
        var d_idx = ThreadIdx.x() + i * LDD + 4 * LDD * ThreadIdx.y()
        c_ptr[d_idx] = d_reg[i]


fn run_mma_fp32_fp16(
    M: Int,
    N: Int,
    K: Int,
    rand_min: Int64,
    rand_max: Int64,
    ctx: DeviceContext,
) raises:
    print("== run_matmul fp32.fp16 matrix core kernel")

    var errors = 0

    @parameter
    if is_amd_gpu():
        var a_host = UnsafePointer[Float16].alloc(M * K)
        var b_host = UnsafePointer[Float16].alloc(K * N)
        var c_host = UnsafePointer[Float32].alloc(M * N)
        var c_host_ref = UnsafePointer[Float32].alloc(M * N)

        for i in range(M * K):
            var val = random_si64(rand_min, rand_max)
            a_host[i] = val.cast[DType.float16]()

        for i in range(K * N):
            var val = random_si64(rand_min, rand_max)
            b_host[i] = val.cast[DType.float16]()

        for i in range(M * N):
            c_host[i] = 0
            c_host_ref[i] = 0

        var a_device = ctx.enqueue_create_buffer[DType.float16](M * K)
        var b_device = ctx.enqueue_create_buffer[DType.float16](K * N)
        var c_device = ctx.enqueue_create_buffer[DType.float32](M * N)

        ctx.enqueue_copy_to_device(a_device, a_host)
        ctx.enqueue_copy_to_device(b_device, b_host)

        var func_mma = ctx.compile_function[mma_kernel_fp32_fp16]()

        ctx.enqueue_function(
            func_mma,
            a_device,
            b_device,
            c_device,
            M,
            N,
            K,
            grid_dim=1,
            block_dim=(16, 4),
        )

        ctx.enqueue_copy_from_device(c_host, c_device)

        matmul_naive(a_host, b_host, c_host_ref, M, N, K)

        for i in range(M * N):
            if c_host[i] != c_host_ref[i]:
                errors += 1
        _ = a_device
        _ = b_device
        _ = c_device
        _ = a_host
        _ = b_host
        _ = c_host
        _ = c_host_ref
        _ = func_mma^

    # CHECK: Success
    if errors == 0:
        print("Success 🎉: Results match.")
    else:
        print("Failed ❌: results mismatch.")


def main():
    with DeviceContext() as ctx:
        run_mma_fp32_fp32(16, 16, 16, -100, 100, ctx)
        run_mma_fp32_fp16(16, 16, 16, -100, 100, ctx)
