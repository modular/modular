# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: AMD-GPU
# RUN: %mojo-no-debug-no-assert %s | FileCheck %s

from math import ceildiv
from random import random_si64

from gpu import WARP_SIZE, block_idx
from gpu.host import DeviceContext
from gpu.mma import mma
from gpu.mma_util import load_matrix_a_amd as load_matrix_a
from gpu.mma_util import load_matrix_b_amd as load_matrix_b
from gpu.mma_util import store_matrix_d
from memory import UnsafePointer

from utils.numerics import isnan


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


fn mma_kernel_fp32_fp32(
    a_ptr: UnsafePointer[Float32],
    b_ptr: UnsafePointer[Float32],
    c_ptr: UnsafePointer[Float32],
    m: Int,
    n: Int,
    k: Int,
):
    alias mma_m = 16
    alias mma_n = 16
    alias mma_k = 4

    var d_reg: SIMD[DType.float32, 4] = 0
    var tile_loops = k // (4 * mma_k)

    for l in range(tile_loops):
        for i in range(4):
            var a_tile_row = block_idx.x * mma_m
            var a_tile_col = 4 * (l * mma_k + i)
            var b_tile_row = 4 * (l * mma_k + i)
            var b_tile_col = block_idx.y * mma_n
            var a_reg = load_matrix_a[mma_m, mma_n, mma_k](
                a_ptr, a_tile_row, a_tile_col, k
            )
            var b_reg = load_matrix_b[mma_m, mma_n, mma_k](
                b_ptr, b_tile_row, b_tile_col, n
            )

            # Perform mma (d = a * b + d)
            mma(d_reg, a_reg, b_reg, d_reg)

    var c_tile_row = block_idx.x * mma_m
    var c_tile_col = block_idx.y * mma_n
    store_matrix_d[mma_m, mma_n, mma_k](c_ptr, d_reg, c_tile_row, c_tile_col, n)


fn mma_kernel_fp32_fp16(
    a_ptr: UnsafePointer[Float16],
    b_ptr: UnsafePointer[Float16],
    c_ptr: UnsafePointer[Float32],
    m: Int,
    n: Int,
    k: Int,
):
    alias mma_m = 16
    alias mma_n = 16
    alias mma_k = 16

    var d_reg: SIMD[DType.float32, 4] = 0
    var tile_loops = k // mma_k

    for l in range(tile_loops):
        var a_tile_row = block_idx.x * mma_m
        var a_tile_col = l * mma_k
        var b_tile_row = l * mma_k
        var b_tile_col = block_idx.y * mma_n

        var a_reg = load_matrix_a[mma_m, mma_n, mma_k](
            a_ptr, a_tile_row, a_tile_col, k
        )
        var b_reg = load_matrix_b[mma_m, mma_n, mma_k](
            b_ptr, b_tile_row, b_tile_col, n
        )
        mma(d_reg, a_reg, b_reg, d_reg)

    var c_tile_row = block_idx.x * mma_m
    var c_tile_col = block_idx.y * mma_n
    store_matrix_d[mma_m, mma_n, mma_k](c_ptr, d_reg, c_tile_row, c_tile_col, n)


fn mma_kernel_fp32_bf16(
    a_ptr: UnsafePointer[BFloat16],
    b_ptr: UnsafePointer[BFloat16],
    c_ptr: UnsafePointer[Float32],
    m: Int,
    n: Int,
    k: Int,
):
    alias mma_m = 16
    alias mma_n = 16
    alias mma_k = 16

    var d_reg: SIMD[DType.float32, 4] = 0
    var tile_loops = k // mma_k

    for l in range(tile_loops):
        var a_tile_row = block_idx.x * mma_m
        var a_tile_col = l * mma_k
        var b_tile_row = l * mma_k
        var b_tile_col = block_idx.y * mma_n

        var a_reg = load_matrix_a[mma_m, mma_n, mma_k](
            a_ptr, a_tile_row, a_tile_col, k
        )
        var b_reg = load_matrix_b[mma_m, mma_n, mma_k](
            b_ptr, b_tile_row, b_tile_col, n
        )
        mma(d_reg, a_reg, b_reg, d_reg)

    var c_tile_row = block_idx.x * mma_m
    var c_tile_col = block_idx.y * mma_n
    store_matrix_d[mma_m, mma_n, mma_k](c_ptr, d_reg, c_tile_row, c_tile_col, n)


fn run_mma_fp32_fp32(
    M: Int,
    N: Int,
    K: Int,
    rand_min: Int64,
    rand_max: Int64,
    ctx: DeviceContext,
) raises:
    print("== run_matmul fp32.fp32 matrix core kernel")

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
    ctx.enqueue_copy_to_device(c_device, c_host)

    alias WARP_PER_BLOCK = 1
    alias MMA_M = 16
    alias MMA_N = 16
    alias MMA_K = 4

    ctx.enqueue_function[mma_kernel_fp32_fp32](
        a_device,
        b_device,
        c_device,
        M,
        N,
        K,
        grid_dim=(ceildiv(M, MMA_M), ceildiv(N, MMA_N)),
        block_dim=WARP_PER_BLOCK * WARP_SIZE,
    )

    ctx.enqueue_copy_from_device(c_host, c_device)

    matmul_naive(a_host, b_host, c_host_ref, M, N, K)

    var errors = 0
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
    a_host.free()
    b_host.free()
    c_host.free()
    c_host_ref.free()

    # CHECK: Success
    if errors == 0:
        print("Success 🎉: Results match.")
    else:
        print("Failed ❌: results mismatch.")


fn run_mma_fp32_fp16(
    M: Int,
    N: Int,
    K: Int,
    rand_min: Int64,
    rand_max: Int64,
    ctx: DeviceContext,
) raises:
    print("== run_matmul fp32.fp16 matrix core kernel")

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
    ctx.enqueue_copy_to_device(c_device, c_host)

    alias WARP_PER_BLOCK = 1
    alias MMA_M = 16
    alias MMA_N = 16
    alias MMA_K = 16

    ctx.enqueue_function[mma_kernel_fp32_fp16](
        a_device,
        b_device,
        c_device,
        M,
        N,
        K,
        grid_dim=(ceildiv(M, MMA_M), ceildiv(N, MMA_N)),
        block_dim=WARP_PER_BLOCK * WARP_SIZE,
    )

    ctx.enqueue_copy_from_device(c_host, c_device)

    matmul_naive(a_host, b_host, c_host_ref, M, N, K)

    var errors = 0
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
    a_host.free()
    b_host.free()
    c_host.free()
    c_host_ref.free()

    # CHECK: Success
    if errors == 0:
        print("Success 🎉: Results match.")
    else:
        print("Failed ❌: results mismatch.")


fn run_mma_fp32_bf16(
    M: Int,
    N: Int,
    K: Int,
    rand_min: Int64,
    rand_max: Int64,
    ctx: DeviceContext,
) raises:
    print("== run_matmul fp32.bf16 matrix core kernel")

    var a_host = UnsafePointer[BFloat16].alloc(M * K)
    var b_host = UnsafePointer[BFloat16].alloc(K * N)
    var c_host = UnsafePointer[Float32].alloc(M * N)
    var c_host_ref = UnsafePointer[Float32].alloc(M * N)

    for i in range(M * K):
        var val = random_si64(rand_min, rand_max)
        a_host[i] = val.cast[DType.bfloat16]()

    for i in range(K * N):
        var val = random_si64(rand_min, rand_max)
        b_host[i] = val.cast[DType.bfloat16]()

    for i in range(M * N):
        c_host[i] = 0
        c_host_ref[i] = 0

    var a_device = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
    var b_device = ctx.enqueue_create_buffer[DType.bfloat16](K * N)
    var c_device = ctx.enqueue_create_buffer[DType.float32](M * N)

    ctx.enqueue_copy_to_device(a_device, a_host)
    ctx.enqueue_copy_to_device(b_device, b_host)
    ctx.enqueue_copy_to_device(c_device, c_host)

    alias WARP_PER_BLOCK = 1
    alias MMA_M = 16
    alias MMA_N = 16
    alias MMA_K = 16

    ctx.enqueue_function[mma_kernel_fp32_bf16](
        a_device,
        b_device,
        c_device,
        M,
        N,
        K,
        grid_dim=(ceildiv(M, MMA_M), ceildiv(N, MMA_N)),
        block_dim=WARP_PER_BLOCK * WARP_SIZE,
    )

    ctx.enqueue_copy_from_device(c_host, c_device)

    matmul_naive(a_host, b_host, c_host_ref, M, N, K)

    var errors = 0
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
    a_host.free()
    b_host.free()
    c_host.free()
    c_host_ref.free()

    # CHECK: Success
    if errors == 0:
        print("Success 🎉: Results match.")
    else:
        print("Failed ❌: results mismatch.")


def main():
    with DeviceContext() as ctx:
        run_mma_fp32_fp32(16, 16, 16, -100, 100, ctx)
        run_mma_fp32_fp32(1024, 1024, 1024, -100, 100, ctx)
        run_mma_fp32_fp32(1024, 4096, 2048, -100, 100, ctx)

        run_mma_fp32_fp16(16, 16, 16, -100, 100, ctx)
        run_mma_fp32_fp16(1024, 1024, 1024, -100, 100, ctx)
        run_mma_fp32_fp16(1024, 4096, 2048, -100, 100, ctx)

        run_mma_fp32_bf16(16, 16, 16, -100, 100, ctx)
        run_mma_fp32_bf16(1024, 1024, 1024, -100, 100, ctx)
        run_mma_fp32_bf16(1024, 4096, 2048, -100, 100, ctx)
