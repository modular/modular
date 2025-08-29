# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from math import ceildiv
from random import random_si64

from gpu import WARP_SIZE, block_idx
from gpu.host import DeviceContext
from gpu.mma import mma
from gpu.mma_util import load_matrix_a, load_matrix_b, store_matrix_d
from linalg.matmul_gpu import matmul_kernel_naive
from testing import assert_false
from layout import Layout, UNKNOWN_VALUE, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from utils.numerics import isnan
from utils.index import IndexList


# TF32 Tensor core Matmul with shape m16n8k8
fn mma_kernel_fp32_tf32(
    c_ptr: UnsafePointer[Float32],
    a_ptr: UnsafePointer[Float32],
    b_ptr: UnsafePointer[Float32],
    m: Int,
    n: Int,
    k: Int,
):
    alias mma_m = 16
    alias mma_n = 8
    alias mma_k = 8

    var d_reg = SIMD[DType.float32, 4](0)
    var tile_loops = k // mma_k

    for i in range(tile_loops):
        var a_tile_row = block_idx.x * mma_m
        var a_tile_col = i * mma_k
        var b_tile_row = i * mma_k
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


# FP32-BF16 (mixed precision) Tensor core Matmul with shape m16n8k8
fn mma_kernel_fp32_bf16(
    c_ptr: UnsafePointer[Float32],
    a_ptr: UnsafePointer[BFloat16],
    b_ptr: UnsafePointer[BFloat16],
    m: Int,
    n: Int,
    k: Int,
):
    alias mma_m = 16
    alias mma_n = 8
    alias mma_k = 8

    var d_reg = SIMD[DType.float32, 4](0)
    var tile_loops = k // mma_k

    for i in range(tile_loops):
        var a_tile_row = block_idx.x * mma_m
        var a_tile_col = i * mma_k
        var b_tile_row = i * mma_k
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


# FP32-BF16 (mixed precision) Tensor core Matmul with shape m16n8k16
fn mma_kernel_fp32_bf16_2(
    c_ptr: UnsafePointer[Float32],
    a_ptr: UnsafePointer[BFloat16],
    b_ptr: UnsafePointer[BFloat16],
    m: Int,
    n: Int,
    k: Int,
):
    alias mma_m = 16
    alias mma_n = 8
    alias mma_k = 16

    var d_reg = SIMD[DType.float32, 4](0)
    var tile_loops = k // mma_k

    for i in range(tile_loops):
        var a_tile_row = block_idx.x * mma_m
        var a_tile_col = i * mma_k
        var b_tile_row = i * mma_k
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


# FP32-FP16 (mixed precision) Tensor core Matmul with shape m16n8k8
fn mma_kernel_fp32_fp16(
    c_ptr: UnsafePointer[Float32],
    a_ptr: UnsafePointer[Float16],
    b_ptr: UnsafePointer[Float16],
    m: Int,
    n: Int,
    k: Int,
):
    alias mma_m = 16
    alias mma_n = 8
    alias mma_k = 8

    var d_reg = SIMD[DType.float32, 4](0)
    var tile_loops = k // mma_k

    for i in range(tile_loops):
        var a_tile_row = block_idx.x * mma_m
        var a_tile_col = i * mma_k
        var b_tile_row = i * mma_k
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


# FP16 Tensor core Matmul with shape m16n8k8
fn mma_kernel_fp16_fp16(
    c_ptr: UnsafePointer[Float16],
    a_ptr: UnsafePointer[Float16],
    b_ptr: UnsafePointer[Float16],
    m: Int,
    n: Int,
    k: Int,
):
    alias mma_m = 16
    alias mma_n = 8
    alias mma_k = 8

    var d_reg = SIMD[DType.float16, 4](0)
    var tile_loops = k // mma_k

    for i in range(tile_loops):
        var a_tile_row = block_idx.x * mma_m
        var a_tile_col = i * mma_k
        var b_tile_row = i * mma_k
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


fn run_mma_fp32_tf32(
    M: Int,
    N: Int,
    K: Int,
    rand_min: Int64,
    rand_max: Int64,
    iterations: Int,
    errorTolerance: Float32,
    ctx: DeviceContext,
) raises:
    print("== run_matmul fp32.tf32 tensor core kernel")

    var a_host = UnsafePointer[Float32].alloc(M * K)
    var b_host = UnsafePointer[Float32].alloc(K * N)
    var c_host = UnsafePointer[Float32].alloc(M * N)
    var a_host_ref = UnsafePointer[Float32].alloc(M * K)
    var b_host_ref = UnsafePointer[Float32].alloc(K * N)
    var c_host_ref = UnsafePointer[Float32].alloc(M * N)

    for i in range(M * K):
        var val = random_si64(rand_min, rand_max)
        a_host[i] = val.cast[DType.float32]()
        a_host_ref[i] = val.cast[DType.float32]()

    for i in range(K * N):
        var val = random_si64(rand_min, rand_max)
        b_host[i] = val.cast[DType.float32]()
        b_host_ref[i] = val.cast[DType.float32]()

    for i in range(M * N):
        c_host[i] = 0
        c_host_ref[i] = 0

    var a_device = ctx.enqueue_create_buffer[DType.float32](M * K)
    var b_device = ctx.enqueue_create_buffer[DType.float32](K * N)
    var c_device = ctx.enqueue_create_buffer[DType.float32](M * N)
    var a_device_ref = ctx.enqueue_create_buffer[DType.float32](M * K)
    var b_device_ref = ctx.enqueue_create_buffer[DType.float32](K * N)
    var c_device_ref = ctx.enqueue_create_buffer[DType.float32](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    alias WARP_PER_BLOCK = 1
    alias MMA_M = 16
    alias MMA_N = 8
    alias MMA_K = 8

    @always_inline
    @parameter
    fn run_func_mma(ctx: DeviceContext) raises:
        ctx.enqueue_function[mma_kernel_fp32_tf32](
            c_device,
            a_device,
            b_device,
            M,
            N,
            K,
            grid_dim=(ceildiv(M, MMA_M), ceildiv(N, MMA_N)),
            block_dim=WARP_PER_BLOCK * WARP_SIZE,
        )

    var nstime = ctx.execution_time[run_func_mma](iterations)
    var flops = 2 * M * N * K
    var sectime = (nstime / iterations) / 1000000000
    print("Basic Tensor core kernel:")
    print(sectime, "sec")
    print(flops * 1e-9 / sectime, " GFLOPS")
    print()

    ctx.enqueue_copy(c_host, c_device)

    # Run naive matmul.
    ctx.enqueue_copy(a_device_ref, a_host_ref)
    ctx.enqueue_copy(b_device_ref, b_host_ref)

    alias layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    var a_tensor = LayoutTensor[DType.float32, layout, MutableAnyOrigin](
        a_device_ref._unsafe_ptr(),
        RuntimeLayout[layout].row_major(IndexList[2](M, K)),
    )

    var b_tensor = LayoutTensor[DType.float32, layout, MutableAnyOrigin](
        b_device_ref._unsafe_ptr(),
        RuntimeLayout[layout].row_major(IndexList[2](K, N)),
    )

    var c_tensor = LayoutTensor[DType.float32, layout, MutableAnyOrigin](
        c_device_ref._unsafe_ptr(),
        RuntimeLayout[layout].row_major(IndexList[2](M, N)),
    )

    alias BLOCK_DIM = 16

    @always_inline
    @parameter
    fn run_func_naive(ctx: DeviceContext) raises:
        ctx.enqueue_function[
            matmul_kernel_naive[
                DType.float32,
                DType.float32,
                DType.float32,
                c_tensor.layout,
                a_tensor.layout,
                b_tensor.layout,
                BLOCK_DIM,
            ]
        ](
            c_tensor,
            a_tensor,
            b_tensor,
            M,
            N,
            K,
            grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
            block_dim=(BLOCK_DIM, BLOCK_DIM),
        )

    nstime = ctx.execution_time[run_func_naive](iterations)
    var sectime2 = (nstime / iterations) / 1000000000
    print("Naive matmul kernel:")
    print(sectime2, "sec")
    print(flops * 1e-9 / sectime2, " GFLOPS")
    print()

    ctx.enqueue_copy(c_host_ref, c_device_ref)

    ctx.synchronize()

    # Check correctness.
    var failed = False
    for i in range(M * N):
        var outVal = c_host[i]
        var outRef = c_host_ref[i]
        var relDiff = (max(outVal, outRef) / min(outVal, outRef)) - 1.0
        if (relDiff > errorTolerance) or isnan(outVal) or isnan(outRef):
            failed = True
            print(i, outVal, outRef)

    if not failed:
        print("Success 🎉: Results match.")
        print(
            "Performance basic tensor core matmul vs. naive matmul: ",
            sectime2 / sectime,
            "x",
        )
    else:
        print("Failed ❌: results mismatch.")

    assert_false(failed)

    _ = a_device
    _ = b_device
    _ = c_device
    _ = a_device_ref
    _ = b_device_ref
    _ = c_device_ref

    _ = a_host
    _ = b_host
    _ = c_host
    _ = a_host_ref
    _ = a_host_ref
    _ = c_host_ref


fn run_mma_fp32_bf16(
    M: Int,
    N: Int,
    K: Int,
    rand_min: Int64,
    rand_max: Int64,
    iterations: Int,
    errorTolerance: Float32,
    ctx: DeviceContext,
) raises:
    print("== run_matmul fp32.bf16 1688 tensor core kernel")

    var a_host = UnsafePointer[BFloat16].alloc(M * K)
    var b_host = UnsafePointer[BFloat16].alloc(K * N)
    var c_host = UnsafePointer[Float32].alloc(M * N)
    var a_host_ref = UnsafePointer[Float32].alloc(M * K)
    var b_host_ref = UnsafePointer[Float32].alloc(K * N)
    var c_host_ref = UnsafePointer[Float32].alloc(M * N)

    for i in range(M * K):
        var val = random_si64(rand_min, rand_max)
        a_host[i] = val.cast[DType.bfloat16]()
        a_host_ref[i] = val.cast[DType.float32]()

    for i in range(K * N):
        var val = random_si64(rand_min, rand_max)
        b_host[i] = val.cast[DType.bfloat16]()
        b_host_ref[i] = val.cast[DType.float32]()

    for i in range(M * N):
        c_host[i] = 0
        c_host_ref[i] = 0

    var a_device = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
    var b_device = ctx.enqueue_create_buffer[DType.bfloat16](K * N)
    var c_device = ctx.enqueue_create_buffer[DType.float32](M * N)
    var a_device_ref = ctx.enqueue_create_buffer[DType.float32](M * K)
    var b_device_ref = ctx.enqueue_create_buffer[DType.float32](K * N)
    var c_device_ref = ctx.enqueue_create_buffer[DType.float32](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    alias WARP_PER_BLOCK = 1
    alias MMA_M = 16
    alias MMA_N = 8
    alias MMA_K = 8

    @always_inline
    @parameter
    fn run_func_mma(ctx: DeviceContext) raises:
        ctx.enqueue_function[mma_kernel_fp32_bf16](
            c_device,
            a_device,
            b_device,
            M,
            N,
            K,
            grid_dim=(ceildiv(M, MMA_M), ceildiv(N, MMA_N)),
            block_dim=WARP_PER_BLOCK * WARP_SIZE,
        )

    var nstime = ctx.execution_time[run_func_mma](iterations)
    var flops = 2 * M * N * K
    var sectime = (nstime / iterations) / 1000000000
    print("Basic Tensor core kernel:")
    print(sectime, "sec")
    print(flops * 1e-9 / sectime, " GFLOPS")
    print()

    ctx.enqueue_copy(c_host, c_device)

    # Run naive matmul.
    ctx.enqueue_copy(a_device_ref, a_host_ref)
    ctx.enqueue_copy(b_device_ref, b_host_ref)

    alias BLOCK_DIM = 16
    alias layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    var a_tensor = LayoutTensor[DType.float32, layout, MutableAnyOrigin](
        a_device_ref._unsafe_ptr(),
        RuntimeLayout[layout].row_major(IndexList[2](M, K)),
    )

    var b_tensor = LayoutTensor[DType.float32, layout, MutableAnyOrigin](
        b_device_ref._unsafe_ptr(),
        RuntimeLayout[layout].row_major(IndexList[2](K, N)),
    )

    var c_tensor = LayoutTensor[DType.float32, layout, MutableAnyOrigin](
        c_device_ref._unsafe_ptr(),
        RuntimeLayout[layout].row_major(IndexList[2](M, N)),
    )

    @always_inline
    @parameter
    fn run_func_naive(ctx: DeviceContext) raises:
        ctx.enqueue_function[
            matmul_kernel_naive[
                DType.float32,
                DType.float32,
                DType.float32,
                c_tensor.layout,
                a_tensor.layout,
                b_tensor.layout,
                BLOCK_DIM,
            ]
        ](
            c_tensor,
            a_tensor,
            b_tensor,
            M,
            N,
            K,
            grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
            block_dim=(BLOCK_DIM, BLOCK_DIM),
        )

    nstime = ctx.execution_time[run_func_naive](iterations)
    var sectime2 = (nstime / iterations) / 1000000000
    print("Naive matmul kernel:")
    print(sectime2, "sec")
    print(flops * 1e-9 / sectime2, " GFLOPS")
    print()

    ctx.enqueue_copy(c_host_ref, c_device_ref)

    # Check correctness.
    var failed = False
    for i in range(M * N):
        var outVal = c_host[i]
        var outRef = c_host_ref[i]
        var relDiff = (max(outVal, outRef) / min(outVal, outRef)) - 1.0
        if (relDiff > errorTolerance) or isnan(outVal) or isnan(outRef):
            failed = True
            print(i, outVal, outRef)

    if not failed:
        print("Success 🎉: Results match.")
        print(
            "Performance basic tensor core matmul vs. naive matmul: ",
            sectime2 / sectime,
            "x",
        )
    else:
        print("Failed ❌: results mismatch.")

    assert_false(failed)

    _ = a_device
    _ = b_device
    _ = c_device
    _ = a_device_ref
    _ = b_device_ref
    _ = c_device_ref

    _ = a_host
    _ = b_host
    _ = c_host
    _ = a_host_ref
    _ = a_host_ref
    _ = c_host_ref


fn run_mma_fp32_bf16_2(
    M: Int,
    N: Int,
    K: Int,
    rand_min: Int64,
    rand_max: Int64,
    iterations: Int,
    errorTolerance: Float32,
    ctx: DeviceContext,
) raises:
    print("== run_matmul fp32.bf16 16816 tensor core kernel")

    var a_host = UnsafePointer[BFloat16].alloc(M * K)
    var b_host = UnsafePointer[BFloat16].alloc(K * N)
    var c_host = UnsafePointer[Float32].alloc(M * N)
    var a_host_ref = UnsafePointer[Float32].alloc(M * K)
    var b_host_ref = UnsafePointer[Float32].alloc(K * N)
    var c_host_ref = UnsafePointer[Float32].alloc(M * N)

    for i in range(M * K):
        var val = random_si64(rand_min, rand_max)
        a_host[i] = val.cast[DType.bfloat16]()
        a_host_ref[i] = val.cast[DType.float32]()

    for i in range(K * N):
        var val = random_si64(rand_min, rand_max)
        b_host[i] = val.cast[DType.bfloat16]()
        b_host_ref[i] = val.cast[DType.float32]()

    for i in range(M * N):
        c_host[i] = 0
        c_host_ref[i] = 0

    var a_device = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
    var b_device = ctx.enqueue_create_buffer[DType.bfloat16](K * N)
    var c_device = ctx.enqueue_create_buffer[DType.float32](M * N)
    var a_device_ref = ctx.enqueue_create_buffer[DType.float32](M * K)
    var b_device_ref = ctx.enqueue_create_buffer[DType.float32](K * N)
    var c_device_ref = ctx.enqueue_create_buffer[DType.float32](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    alias WARP_PER_BLOCK = 1
    alias MMA_M = 16
    alias MMA_N = 8
    alias MMA_K = 8

    @always_inline
    @parameter
    fn run_func_mma(ctx: DeviceContext) raises:
        ctx.enqueue_function[mma_kernel_fp32_bf16_2](
            c_device,
            a_device,
            b_device,
            M,
            N,
            K,
            grid_dim=(ceildiv(M, MMA_M), ceildiv(N, MMA_N)),
            block_dim=WARP_PER_BLOCK * WARP_SIZE,
        )

    var nstime = ctx.execution_time[run_func_mma](iterations)
    var flops = 2 * M * N * K
    var sectime = (nstime / iterations) / 1000000000
    print("Basic Tensor core kernel:")
    print(sectime, "sec")
    print(flops * 1e-9 / sectime, " GFLOPS")
    print()

    ctx.enqueue_copy(c_host, c_device)

    # Run naive matmul.
    ctx.enqueue_copy(a_device_ref, a_host_ref)
    ctx.enqueue_copy(b_device_ref, b_host_ref)

    alias BLOCK_DIM = 16

    alias layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    var a_tensor = LayoutTensor[DType.float32, layout, MutableAnyOrigin](
        a_device_ref._unsafe_ptr(),
        RuntimeLayout[layout].row_major(IndexList[2](M, K)),
    )

    var b_tensor = LayoutTensor[DType.float32, layout, MutableAnyOrigin](
        b_device_ref._unsafe_ptr(),
        RuntimeLayout[layout].row_major(IndexList[2](K, N)),
    )

    var c_tensor = LayoutTensor[DType.float32, layout, MutableAnyOrigin](
        c_device_ref._unsafe_ptr(),
        RuntimeLayout[layout].row_major(IndexList[2](M, N)),
    )

    @always_inline
    @parameter
    fn run_func_naive(ctx: DeviceContext) raises:
        ctx.enqueue_function[
            matmul_kernel_naive[
                DType.float32,
                DType.float32,
                DType.float32,
                a_tensor.layout,
                b_tensor.layout,
                c_tensor.layout,
                BLOCK_DIM,
            ]
        ](
            c_tensor,
            a_tensor,
            b_tensor,
            M,
            N,
            K,
            grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
            block_dim=(BLOCK_DIM, BLOCK_DIM),
        )

    nstime = ctx.execution_time[run_func_naive](iterations)
    var sectime2 = (nstime / iterations) / 1000000000
    print("Naive matmul kernel:")
    print(sectime2, "sec")
    print(flops * 1e-9 / sectime2, " GFLOPS")
    print()

    ctx.enqueue_copy(c_host_ref, c_device_ref)

    # Check correctness.
    var failed = False
    for i in range(M * N):
        var outVal = c_host[i]
        var outRef = c_host_ref[i]
        var relDiff = (max(outVal, outRef) / min(outVal, outRef)) - 1.0
        if (relDiff > errorTolerance) or isnan(outVal) or isnan(outRef):
            failed = True
            print(i, outVal, outRef)

    if not failed:
        print("Success 🎉: Results match.")
        print(
            "Performance basic tensor core matmul vs. naive matmul: ",
            sectime2 / sectime,
            "x",
        )
    else:
        print("Failed ❌: results mismatch.")

    assert_false(failed)

    _ = a_device
    _ = b_device
    _ = c_device
    _ = a_device_ref
    _ = b_device_ref
    _ = c_device_ref

    _ = a_host
    _ = b_host
    _ = c_host
    _ = a_host_ref
    _ = a_host_ref
    _ = c_host_ref


fn run_mma_fp32_fp16(
    M: Int,
    N: Int,
    K: Int,
    rand_min: Int64,
    rand_max: Int64,
    iterations: Int,
    errorTolerance: Float32,
    ctx: DeviceContext,
) raises:
    print("== run_matmul fp32.fp16 tensor core kernel")

    var a_host = UnsafePointer[Float16].alloc(M * K)
    var b_host = UnsafePointer[Float16].alloc(K * N)
    var c_host = UnsafePointer[Float32].alloc(M * N)
    var a_host_ref = UnsafePointer[Float32].alloc(M * K)
    var b_host_ref = UnsafePointer[Float32].alloc(K * N)
    var c_host_ref = UnsafePointer[Float32].alloc(M * N)

    for i in range(M * K):
        var val = random_si64(rand_min, rand_max)
        a_host[i] = val.cast[DType.float16]()
        a_host_ref[i] = val.cast[DType.float32]()

    for i in range(K * N):
        var val = random_si64(rand_min, rand_max)
        b_host[i] = val.cast[DType.float16]()
        b_host_ref[i] = val.cast[DType.float32]()

    for i in range(M * N):
        c_host[i] = 0
        c_host_ref[i] = 0

    var a_device = ctx.enqueue_create_buffer[DType.float16](M * K)
    var b_device = ctx.enqueue_create_buffer[DType.float16](K * N)
    var c_device = ctx.enqueue_create_buffer[DType.float32](M * N)
    var a_device_ref = ctx.enqueue_create_buffer[DType.float32](M * K)
    var b_device_ref = ctx.enqueue_create_buffer[DType.float32](K * N)
    var c_device_ref = ctx.enqueue_create_buffer[DType.float32](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    alias WARP_PER_BLOCK = 1
    alias MMA_M = 16
    alias MMA_N = 8
    alias MMA_K = 8

    @always_inline
    @parameter
    fn run_func_mma(ctx: DeviceContext) raises:
        ctx.enqueue_function[mma_kernel_fp32_fp16](
            c_device,
            a_device,
            b_device,
            M,
            N,
            K,
            grid_dim=(ceildiv(M, MMA_M), ceildiv(N, MMA_N)),
            block_dim=WARP_PER_BLOCK * WARP_SIZE,
        )

    var nstime = ctx.execution_time[run_func_mma](iterations)
    var flops = 2 * M * N * K
    var sectime = (nstime / iterations) / 1000000000
    print("Basic Tensor core kernel:")
    print(sectime, "sec")
    print(flops * 1e-9 / sectime, " GFLOPS")
    print()

    ctx.enqueue_copy(c_host, c_device)

    # Run naive matmul.
    ctx.enqueue_copy(a_device_ref, a_host_ref)
    ctx.enqueue_copy(b_device_ref, b_host_ref)

    alias BLOCK_DIM = 16
    alias layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    var a_tensor = LayoutTensor[DType.float32, layout, MutableAnyOrigin](
        a_device_ref._unsafe_ptr(),
        RuntimeLayout[layout].row_major(IndexList[2](M, K)),
    )

    var b_tensor = LayoutTensor[DType.float32, layout, MutableAnyOrigin](
        b_device_ref._unsafe_ptr(),
        RuntimeLayout[layout].row_major(IndexList[2](K, N)),
    )

    var c_tensor = LayoutTensor[DType.float32, layout, MutableAnyOrigin](
        c_device_ref._unsafe_ptr(),
        RuntimeLayout[layout].row_major(IndexList[2](M, N)),
    )

    @always_inline
    @parameter
    fn run_func_naive(ctx: DeviceContext) raises:
        ctx.enqueue_function[
            matmul_kernel_naive[
                DType.float32,
                DType.float32,
                DType.float32,
                a_tensor.layout,
                b_tensor.layout,
                c_tensor.layout,
                BLOCK_DIM,
            ]
        ](
            c_tensor,
            a_tensor,
            b_tensor,
            M,
            N,
            K,
            grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
            block_dim=(BLOCK_DIM, BLOCK_DIM),
        )

    nstime = ctx.execution_time[run_func_naive](iterations)
    var sectime2 = (nstime / iterations) / 1000000000
    print("Naive matmul kernel:")
    print(sectime2, "sec")
    print(flops * 1e-9 / sectime2, " GFLOPS")
    print()

    ctx.enqueue_copy(c_host_ref, c_device_ref)

    # Check correctness.
    var failed = False
    for i in range(M * N):
        var outVal = c_host[i]
        var outRef = c_host_ref[i]
        var relDiff = (max(outVal, outRef) / min(outVal, outRef)) - 1.0
        if (relDiff > errorTolerance) or isnan(outVal) or isnan(outRef):
            failed = True
            print(i, outVal, outRef)

    if not failed:
        print("Success 🎉: Results match.")
        print(
            "Performance basic tensor core matmul vs. naive matmul: ",
            sectime2 / sectime,
            "x",
        )
    else:
        print("Failed ❌: results mismatch.")

    assert_false(failed)

    _ = a_device
    _ = b_device
    _ = c_device
    _ = a_device_ref
    _ = b_device_ref
    _ = c_device_ref

    _ = a_host
    _ = b_host
    _ = c_host
    _ = a_host_ref
    _ = a_host_ref
    _ = c_host_ref


fn run_mma_fp16_fp16(
    M: Int,
    N: Int,
    K: Int,
    rand_min: Int64,
    rand_max: Int64,
    iterations: Int,
    errorTolerance: Float32,
    ctx: DeviceContext,
) raises:
    print("== run_matmul fp16.fp16 tensor core kernel")

    var a_host = UnsafePointer[Float16].alloc(M * K)
    var b_host = UnsafePointer[Float16].alloc(K * N)
    var c_host = UnsafePointer[Float16].alloc(M * N)
    var a_host_ref = UnsafePointer[Float32].alloc(M * K)
    var b_host_ref = UnsafePointer[Float32].alloc(K * N)
    var c_host_ref = UnsafePointer[Float32].alloc(M * N)

    for i in range(M * K):
        var val = random_si64(rand_min, rand_max)
        a_host[i] = val.cast[DType.float16]()
        a_host_ref[i] = val.cast[DType.float32]()

    for i in range(K * N):
        var val = random_si64(rand_min, rand_max)
        b_host[i] = val.cast[DType.float16]()
        b_host_ref[i] = val.cast[DType.float32]()

    for i in range(M * N):
        c_host[i] = 0
        c_host_ref[i] = 0

    var a_device = ctx.enqueue_create_buffer[DType.float16](M * K)
    var b_device = ctx.enqueue_create_buffer[DType.float16](K * N)
    var c_device = ctx.enqueue_create_buffer[DType.float16](M * N)
    var a_device_ref = ctx.enqueue_create_buffer[DType.float32](M * K)
    var b_device_ref = ctx.enqueue_create_buffer[DType.float32](K * N)
    var c_device_ref = ctx.enqueue_create_buffer[DType.float32](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    alias WARP_PER_BLOCK = 1
    alias MMA_M = 16
    alias MMA_N = 8
    alias MMA_K = 8

    @always_inline
    @parameter
    fn run_func_mma(ctx: DeviceContext) raises:
        ctx.enqueue_function[mma_kernel_fp16_fp16](
            c_device,
            a_device,
            b_device,
            M,
            N,
            K,
            grid_dim=(ceildiv(M, MMA_M), ceildiv(N, MMA_N)),
            block_dim=WARP_PER_BLOCK * WARP_SIZE,
        )

    var nstime = ctx.execution_time[run_func_mma](iterations)
    var flops = 2 * M * N * K
    var sectime = (nstime / iterations) / 1000000000
    print("Basic Tensor core kernel:")
    print(sectime, "sec")
    print(flops * 1e-9 / sectime, " GFLOPS")
    print()

    ctx.enqueue_copy(c_host, c_device)

    # Run naive matmul.
    ctx.enqueue_copy(a_device_ref, a_host_ref)
    ctx.enqueue_copy(b_device_ref, b_host_ref)

    alias BLOCK_DIM = 16
    alias layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    var a_tensor = LayoutTensor[DType.float32, layout, MutableAnyOrigin](
        a_device_ref._unsafe_ptr(),
        RuntimeLayout[layout].row_major(IndexList[2](M, K)),
    )

    var b_tensor = LayoutTensor[DType.float32, layout, MutableAnyOrigin](
        b_device_ref._unsafe_ptr(),
        RuntimeLayout[layout].row_major(IndexList[2](K, N)),
    )

    var c_tensor = LayoutTensor[DType.float32, layout, MutableAnyOrigin](
        c_device_ref._unsafe_ptr(),
        RuntimeLayout[layout].row_major(IndexList[2](M, N)),
    )

    @always_inline
    @parameter
    fn run_func_naive(ctx: DeviceContext) raises:
        ctx.enqueue_function[
            matmul_kernel_naive[
                DType.float32,
                DType.float32,
                DType.float32,
                a_tensor.layout,
                b_tensor.layout,
                c_tensor.layout,
                BLOCK_DIM,
            ]
        ](
            c_tensor,
            a_tensor,
            b_tensor,
            M,
            N,
            K,
            grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
            block_dim=(BLOCK_DIM, BLOCK_DIM),
        )

    nstime = ctx.execution_time[run_func_naive](iterations)
    var sectime2 = (nstime / iterations) / 1000000000
    print("Naive matmul kernel:")
    print(sectime2, "sec")
    print(flops * 1e-9 / sectime2, " GFLOPS")
    print()

    ctx.enqueue_copy(c_host_ref, c_device_ref)

    # Check correctness.
    var failed = False
    for i in range(M * N):
        var outVal = c_host[i].cast[DType.float32]()
        # var outVal = c_host[i]
        var outRef = c_host_ref[i]
        var relDiff = (max(outVal, outRef) / min(outVal, outRef)) - 1.0
        if (relDiff > errorTolerance) or isnan(outVal) or isnan(outRef):
            failed = True
            print(i, outVal, outRef)

    if not failed:
        print("Success 🎉: Results match.")
        print(
            "Performance basic tensor core matmul vs. naive matmul: ",
            sectime2 / sectime,
            "x",
        )
    else:
        print("Failed ❌: results mismatch.")

    assert_false(failed)

    _ = a_device
    _ = b_device
    _ = c_device
    _ = a_device_ref
    _ = b_device_ref
    _ = c_device_ref

    _ = a_host
    _ = b_host
    _ = c_host
    _ = a_host_ref
    _ = a_host_ref
    _ = c_host_ref


def main():
    with DeviceContext() as ctx:
        # Run tensor core versions of matmul, verify correctness & compare to naive.
        run_mma_fp32_fp16(16, 8, 8, -100, 100, 10, 0.01, ctx)
        run_mma_fp32_fp16(1024, 1024, 1024, -100, 100, 10, 0.01, ctx)
        run_mma_fp32_fp16(1024, 4096, 2048, -100, 100, 10, 0.01, ctx)

        run_mma_fp32_bf16(16, 8, 8, -100, 100, 10, 0.01, ctx)
        run_mma_fp32_bf16(1024, 1024, 1024, -100, 100, 10, 0.01, ctx)
        run_mma_fp32_bf16(1024, 4096, 2048, -100, 100, 10, 0.01, ctx)

        run_mma_fp32_bf16_2(16, 8, 16, -100, 100, 10, 0.01, ctx)
        run_mma_fp32_bf16_2(2048, 1024, 2048, -100, 100, 10, 0.01, ctx)
        run_mma_fp32_bf16_2(2048, 4096, 2048, -100, 100, 10, 0.01, ctx)

        run_mma_fp32_tf32(16, 8, 8, -100, 100, 10, 0.01, ctx)
        run_mma_fp32_tf32(1024, 1024, 1024, -100, 100, 10, 0.01, ctx)
        run_mma_fp32_tf32(1024, 4096, 2048, -100, 100, 10, 0.01, ctx)

        run_mma_fp16_fp16(16, 8, 8, -100, 100, 10, 0.01, ctx)
        run_mma_fp16_fp16(512, 128, 32, -10, 10, 10, 0.01, ctx)
        run_mma_fp16_fp16(128, 256, 64, -10, 10, 10, 0.01, ctx)
