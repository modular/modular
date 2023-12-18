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
from gpu import (
    GridDim,
    BlockIdx,
    BlockDim,
    ThreadIdx,
    barrier,
    WARP_SIZE,
    lane_id,
)
from gpu.sync import syncwarp
from memory.buffer import NDBuffer
from memory.unsafe import DTypePointer, bitcast
from utils.index import Index
from utils.list import DimList
from random import seed, random_si64
from gpu.mma import mma
from Matmul import matmul_kernel, matmul_kernel_naive
from gpu.mma_util import load_matrix_a, load_matrix_b, store_matrix_d
from memory import memset_zero, stack_allocation
from gpu.memory import AddressSpace


# TF32 Tensor core Matmul with shape m16n8k8
fn mma_kernel_fp32_tf32(
    c_ptr: DTypePointer[DType.float32],
    a_ptr: DTypePointer[DType.float32],
    b_ptr: DTypePointer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
):
    alias mma_m = 16
    alias mma_n = 8
    alias mma_k = 8

    var a_reg = SIMD[DType.float32, 4]()
    var b_reg = SIMD[DType.float32, 2]()
    var d_reg = SIMD[DType.float32, 4]()
    let tile_loops = k // mma_k

    for i in range(tile_loops):
        let a_tile_row = BlockIdx.x() * mma_m
        let a_tile_col = i * mma_k
        let b_tile_row = i * mma_k
        let b_tile_col = BlockIdx.y() * mma_n

        load_matrix_a[mma_m, mma_n, mma_k](
            a_reg, a_ptr, a_tile_row, a_tile_col, k
        )
        load_matrix_b[mma_m, mma_n, mma_k](
            b_reg, b_ptr, b_tile_row, b_tile_col, n
        )

        # Perform mma (d = a * b + d)
        mma(d_reg, a_reg, b_reg, d_reg)

    let c_tile_row = BlockIdx.x() * mma_m
    let c_tile_col = BlockIdx.y() * mma_n
    store_matrix_d[DType.float32, mma_m, mma_n, mma_k](
        c_ptr, d_reg, c_tile_row, c_tile_col, n
    )


# FP32-FP16 (mixed precision) Tensor core Matmul with shape m16n8k8
fn mma_kernel_fp32_fp16(
    c_ptr: DTypePointer[DType.float32],
    a_ptr: DTypePointer[DType.float16],
    b_ptr: DTypePointer[DType.float16],
    m: Int,
    n: Int,
    k: Int,
):
    alias mma_m = 16
    alias mma_n = 8
    alias mma_k = 8

    var a_reg = SIMD[DType.float16, 4]()
    var b_reg = SIMD[DType.float16, 2]()
    var d_reg = SIMD[DType.float32, 4]()
    let tile_loops = k // mma_k

    for i in range(tile_loops):
        let a_tile_row = BlockIdx.x() * mma_m
        let a_tile_col = i * mma_k
        let b_tile_row = i * mma_k
        let b_tile_col = BlockIdx.y() * mma_n

        load_matrix_a[mma_m, mma_n, mma_k](
            a_reg, a_ptr, a_tile_row, a_tile_col, k
        )
        load_matrix_b[mma_m, mma_n, mma_k](
            b_reg, b_ptr, b_tile_row, b_tile_col, n
        )

        # Perform mma (d = a * b + d)
        mma(d_reg, a_reg, b_reg, d_reg)

    let c_tile_row = BlockIdx.x() * mma_m
    let c_tile_col = BlockIdx.y() * mma_n
    store_matrix_d[DType.float32, mma_m, mma_n, mma_k](
        c_ptr, d_reg, c_tile_row, c_tile_col, n
    )


# FP16 Tensor core Matmul with shape m16n8k8
fn mma_kernel_fp16_fp16(
    c_ptr: DTypePointer[DType.float16],
    a_ptr: DTypePointer[DType.float16],
    b_ptr: DTypePointer[DType.float16],
    m: Int,
    n: Int,
    k: Int,
):
    alias mma_m = 16
    alias mma_n = 8
    alias mma_k = 8

    var a_reg = SIMD[DType.float16, 4]()
    var b_reg = SIMD[DType.float16, 2]()
    var d_reg = SIMD[DType.float16, 4]()
    let tile_loops = k // mma_k

    for i in range(tile_loops):
        let a_tile_row = BlockIdx.x() * mma_m
        let a_tile_col = i * mma_k
        let b_tile_row = i * mma_k
        let b_tile_col = BlockIdx.y() * mma_n

        load_matrix_a[mma_m, mma_n, mma_k](
            a_reg, a_ptr, a_tile_row, a_tile_col, k
        )
        load_matrix_b[mma_m, mma_n, mma_k](
            b_reg, b_ptr, b_tile_row, b_tile_col, n
        )

        # Perform mma (d = a * b + d)
        mma(d_reg, a_reg, b_reg, d_reg)

    let c_tile_row = BlockIdx.x() * mma_m
    let c_tile_col = BlockIdx.y() * mma_n
    store_matrix_d[DType.float16, mma_m, mma_n, mma_k](
        c_ptr, d_reg, c_tile_row, c_tile_col, n
    )


fn run_mma_fp32_tf32(
    M: Int,
    N: Int,
    K: Int,
    rand_min: Int64,
    rand_max: Int64,
    iterations: Int,
    errorTolerance: Float32,
) raises:
    print("== run_matmul fp32.tf32 tensor core kernel")

    let stream = Stream()
    let a_host = Pointer[Float32].alloc(M * K)
    let b_host = Pointer[Float32].alloc(K * N)
    let c_host = Pointer[Float32].alloc(M * N)
    let a_host_ref = Pointer[Float32].alloc(M * K)
    let b_host_ref = Pointer[Float32].alloc(K * N)
    let c_host_ref = Pointer[Float32].alloc(M * N)

    for i in range(M * K):
        let val = random_si64(rand_min, rand_max)
        a_host.store(i, val.cast[DType.float32]())
        a_host_ref.store(i, val.cast[DType.float32]())

    for i in range(K * N):
        let val = random_si64(rand_min, rand_max)
        b_host.store(i, val.cast[DType.float32]())
        b_host_ref.store(i, val.cast[DType.float32]())

    for i in range(M * N):
        c_host.store(i, 0)
        c_host_ref.store(i, 0)

    let a_device = _malloc[Float32](M * K)
    let b_device = _malloc[Float32](K * N)
    let c_device = _malloc[Float32](M * N)
    let a_device_ref = _malloc[Float32](M * K)
    let b_device_ref = _malloc[Float32](K * N)
    let c_device_ref = _malloc[Float32](M * N)

    _copy_host_to_device(a_device, a_host, M * K)
    _copy_host_to_device(b_device, b_host, K * N)

    let func_mma = Function[
        fn (
            DTypePointer[DType.float32],
            DTypePointer[DType.float32],
            DTypePointer[DType.float32],
            Int,
            Int,
            Int,
        ) -> None, mma_kernel_fp32_tf32
    ](dump_ptx=False)

    alias WARP_PER_BLOCK = 1
    alias MMA_M = 16
    alias MMA_N = 8
    alias MMA_K = 8

    @always_inline
    @parameter
    fn run_func_mma(stream: Stream) raises:
        func_mma(
            (div_ceil(M, MMA_M), div_ceil(N, MMA_N)),
            WARP_PER_BLOCK * WARP_SIZE,
            c_device,
            a_device,
            b_device,
            M,
            N,
            K,
            stream=stream,
        )

    var nstime = 0.0
    for i in range(iterations):
        nstime += time_function[run_func_mma](stream)
    let flops = 2 * M * N * K
    let sectime = ((nstime / iterations) / 1000000000)
    print("Basic Tensor core kernel:")
    print(sectime, "sec")
    print(flops * 1e-9 / sectime, " GFLOPS")
    print()

    _copy_device_to_host(c_host, c_device, M * N)

    # Run naive matmul.
    _copy_host_to_device(a_device_ref, a_host_ref, M * K)
    _copy_host_to_device(b_device_ref, b_host_ref, K * N)

    alias BLOCK_DIM = 16
    let func_naive = Function[
        fn (
            DTypePointer[DType.float32],
            DTypePointer[DType.float32],
            DTypePointer[DType.float32],
            Int,
            Int,
            Int,
        ) capturing -> None, matmul_kernel_naive[
            DType.float32, DType.float32, DType.float32, BLOCK_DIM
        ]
    ]()

    @always_inline
    @parameter
    fn run_func_naive(stream: Stream) raises:
        func_naive(
            (div_ceil(M, BLOCK_DIM), div_ceil(N, BLOCK_DIM)),
            (BLOCK_DIM, BLOCK_DIM),
            c_device_ref,
            a_device_ref,
            b_device_ref,
            M,
            N,
            K,
            stream=stream,
        )

    nstime = 0.0
    for i in range(iterations):
        nstime += time_function[run_func_naive](stream)
    let sectime2 = ((nstime / iterations) / 1000000000)
    print("Naive matmul kernel:")
    print(sectime2, "sec")
    print(flops * 1e-9 / sectime2, " GFLOPS")
    print()

    _copy_device_to_host(c_host_ref, c_device_ref, M * N)

    # Check correctness.
    var failed = False
    for i in range(M * N):
        let outVal = c_host.load(i)
        let outRef = c_host_ref.load(i)
        let relDiff = (max(outVal, outRef) / min(outVal, outRef)) - 1.0
        if (
            (relDiff > errorTolerance)
            or math.isnan(outVal)
            or math.isnan(outRef)
        ):
            failed = True
            print(i, outVal, outRef)

    # CHECK: Success
    if not failed:
        print("Success 🎉: Results match.")
        print(
            "Performance basic tensor core matmul vs. naive matmul: ",
            sectime2 / sectime,
            "x",
        )
    else:
        print("Failed ❌: results mismatch.")

    _free(a_device)
    _free(b_device)
    _free(c_device)
    _free(a_device_ref)
    _free(b_device_ref)
    _free(c_device_ref)

    _ = a_host
    _ = b_host
    _ = c_host
    _ = a_host_ref
    _ = a_host_ref
    _ = c_host_ref

    _ = func_mma ^
    _ = func_naive ^
    _ = stream ^


fn run_mma_fp32_fp16(
    M: Int,
    N: Int,
    K: Int,
    rand_min: Int64,
    rand_max: Int64,
    iterations: Int,
    errorTolerance: Float32,
) raises:
    print("== run_matmul fp32.fp16 tensor core kernel")

    let stream = Stream()
    let a_host = Pointer[Float16].alloc(M * K)
    let b_host = Pointer[Float16].alloc(K * N)
    let c_host = Pointer[Float32].alloc(M * N)
    let a_host_ref = Pointer[Float32].alloc(M * K)
    let b_host_ref = Pointer[Float32].alloc(K * N)
    let c_host_ref = Pointer[Float32].alloc(M * N)

    for i in range(M * K):
        let val = random_si64(rand_min, rand_max)
        a_host.store(i, val.cast[DType.float16]())
        a_host_ref.store(i, val.cast[DType.float32]())

    for i in range(K * N):
        let val = random_si64(rand_min, rand_max)
        b_host.store(i, val.cast[DType.float16]())
        b_host_ref.store(i, val.cast[DType.float32]())

    for i in range(M * N):
        c_host.store(i, 0)
        c_host_ref.store(i, 0)

    let a_device = _malloc[Float16](M * K)
    let b_device = _malloc[Float16](K * N)
    let c_device = _malloc[Float32](M * N)
    let a_device_ref = _malloc[Float32](M * K)
    let b_device_ref = _malloc[Float32](K * N)
    let c_device_ref = _malloc[Float32](M * N)

    _copy_host_to_device(a_device, a_host, M * K)
    _copy_host_to_device(b_device, b_host, K * N)

    let func_mma = Function[
        fn (
            DTypePointer[DType.float32],
            DTypePointer[DType.float16],
            DTypePointer[DType.float16],
            Int,
            Int,
            Int,
        ) -> None, mma_kernel_fp32_fp16
    ](dump_ptx=False)

    alias WARP_PER_BLOCK = 1
    alias MMA_M = 16
    alias MMA_N = 8
    alias MMA_K = 8

    @always_inline
    @parameter
    fn run_func_mma(stream: Stream) raises:
        func_mma(
            (div_ceil(M, MMA_M), div_ceil(N, MMA_N)),
            WARP_PER_BLOCK * WARP_SIZE,
            c_device,
            a_device,
            b_device,
            M,
            N,
            K,
            stream=stream,
        )

    var nstime = 0.0
    for i in range(iterations):
        nstime += time_function[run_func_mma](stream)
    let flops = 2 * M * N * K
    let sectime = ((nstime / iterations) / 1000000000)
    print("Basic Tensor core kernel:")
    print(sectime, "sec")
    print(flops * 1e-9 / sectime, " GFLOPS")
    print()

    _copy_device_to_host(c_host, c_device, M * N)

    # Run naive matmul.
    _copy_host_to_device(a_device_ref, a_host_ref, M * K)
    _copy_host_to_device(b_device_ref, b_host_ref, K * N)

    alias BLOCK_DIM = 16
    let func_naive = Function[
        fn (
            DTypePointer[DType.float32],
            DTypePointer[DType.float32],
            DTypePointer[DType.float32],
            Int,
            Int,
            Int,
        ) capturing -> None, matmul_kernel_naive[
            DType.float32, DType.float32, DType.float32, BLOCK_DIM
        ]
    ]()

    @always_inline
    @parameter
    fn run_func_naive(stream: Stream) raises:
        func_naive(
            (div_ceil(M, BLOCK_DIM), div_ceil(N, BLOCK_DIM)),
            (BLOCK_DIM, BLOCK_DIM),
            c_device_ref,
            a_device_ref,
            b_device_ref,
            M,
            N,
            K,
            stream=stream,
        )

    nstime = 0.0
    for i in range(iterations):
        nstime += time_function[run_func_naive](stream)
    let sectime2 = ((nstime / iterations) / 1000000000)
    print("Naive matmul kernel:")
    print(sectime2, "sec")
    print(flops * 1e-9 / sectime2, " GFLOPS")
    print()

    _copy_device_to_host(c_host_ref, c_device_ref, M * N)

    # Check correctness.
    var failed = False
    for i in range(M * N):
        let outVal = c_host.load(i)
        let outRef = c_host_ref.load(i)
        let relDiff = (max(outVal, outRef) / min(outVal, outRef)) - 1.0
        if (
            (relDiff > errorTolerance)
            or math.isnan(outVal)
            or math.isnan(outRef)
        ):
            failed = True
            print(i, outVal, outRef)

    # CHECK: Success
    if not failed:
        print("Success 🎉: Results match.")
        print(
            "Performance basic tensor core matmul vs. naive matmul: ",
            sectime2 / sectime,
            "x",
        )
    else:
        print("Failed ❌: results mismatch.")

    _free(a_device)
    _free(b_device)
    _free(c_device)
    _free(a_device_ref)
    _free(b_device_ref)
    _free(c_device_ref)

    _ = a_host
    _ = b_host
    _ = c_host
    _ = a_host_ref
    _ = a_host_ref
    _ = c_host_ref

    _ = func_mma ^
    _ = func_naive ^
    _ = stream ^


fn run_mma_fp16_fp16(
    M: Int,
    N: Int,
    K: Int,
    rand_min: Int64,
    rand_max: Int64,
    iterations: Int,
    errorTolerance: Float32,
) raises:
    print("== run_matmul fp16.fp16 tensor core kernel")

    let stream = Stream()
    let a_host = Pointer[Float16].alloc(M * K)
    let b_host = Pointer[Float16].alloc(K * N)
    let c_host = Pointer[Float16].alloc(M * N)
    let a_host_ref = Pointer[Float32].alloc(M * K)
    let b_host_ref = Pointer[Float32].alloc(K * N)
    let c_host_ref = Pointer[Float32].alloc(M * N)

    for i in range(M * K):
        let val = random_si64(rand_min, rand_max)
        a_host.store(i, val.cast[DType.float16]())
        a_host_ref.store(i, val.cast[DType.float32]())

    for i in range(K * N):
        let val = random_si64(rand_min, rand_max)
        b_host.store(i, val.cast[DType.float16]())
        b_host_ref.store(i, val.cast[DType.float32]())

    for i in range(M * N):
        c_host.store(i, 0)
        c_host_ref.store(i, 0)

    let a_device = _malloc[Float16](M * K)
    let b_device = _malloc[Float16](K * N)
    let c_device = _malloc[Float16](M * N)
    let a_device_ref = _malloc[Float32](M * K)
    let b_device_ref = _malloc[Float32](K * N)
    let c_device_ref = _malloc[Float32](M * N)

    _copy_host_to_device(a_device, a_host, M * K)
    _copy_host_to_device(b_device, b_host, K * N)

    let func_mma = Function[
        fn (
            DTypePointer[DType.float16],
            DTypePointer[DType.float16],
            DTypePointer[DType.float16],
            Int,
            Int,
            Int,
        ) -> None, mma_kernel_fp16_fp16
    ](dump_ptx=False)

    alias WARP_PER_BLOCK = 1
    alias MMA_M = 16
    alias MMA_N = 8
    alias MMA_K = 8

    @always_inline
    @parameter
    fn run_func_mma(stream: Stream) raises:
        func_mma(
            (div_ceil(M, MMA_M), div_ceil(N, MMA_N)),
            WARP_PER_BLOCK * WARP_SIZE,
            c_device,
            a_device,
            b_device,
            M,
            N,
            K,
            stream=stream,
        )

    var nstime = 0.0
    for i in range(iterations):
        nstime += time_function[run_func_mma](stream)
    let flops = 2 * M * N * K
    let sectime = ((nstime / iterations) / 1000000000)
    print("Basic Tensor core kernel:")
    print(sectime, "sec")
    print(flops * 1e-9 / sectime, " GFLOPS")
    print()

    _copy_device_to_host(c_host, c_device, M * N)

    # Run naive matmul.
    _copy_host_to_device(a_device_ref, a_host_ref, M * K)
    _copy_host_to_device(b_device_ref, b_host_ref, K * N)

    alias BLOCK_DIM = 16
    let func_naive = Function[
        fn (
            DTypePointer[DType.float32],
            DTypePointer[DType.float32],
            DTypePointer[DType.float32],
            Int,
            Int,
            Int,
        ) capturing -> None, matmul_kernel_naive[
            DType.float32, DType.float32, DType.float32, BLOCK_DIM
        ]
    ]()

    @always_inline
    @parameter
    fn run_func_naive(stream: Stream) raises:
        func_naive(
            (div_ceil(M, BLOCK_DIM), div_ceil(N, BLOCK_DIM)),
            (BLOCK_DIM, BLOCK_DIM),
            c_device_ref,
            a_device_ref,
            b_device_ref,
            M,
            N,
            K,
            stream=stream,
        )

    nstime = 0.0
    for i in range(iterations):
        nstime += time_function[run_func_naive](stream)
    let sectime2 = ((nstime / iterations) / 1000000000)
    print("Naive matmul kernel:")
    print(sectime2, "sec")
    print(flops * 1e-9 / sectime2, " GFLOPS")
    print()

    _copy_device_to_host(c_host_ref, c_device_ref, M * N)

    # Check correctness.
    var failed = False
    for i in range(M * N):
        let outVal = c_host.load(i).cast[DType.float32]()
        # let outVal = c_host.load(i)
        let outRef = c_host_ref.load(i)
        let relDiff = (max(outVal, outRef) / min(outVal, outRef)) - 1.0
        if (
            (relDiff > errorTolerance)
            or math.isnan(outVal)
            or math.isnan(outRef)
        ):
            failed = True
            print(i, outVal, outRef)

    # CHECK: Success
    if not failed:
        print("Success 🎉: Results match.")
        print(
            "Performance basic tensor core matmul vs. naive matmul: ",
            sectime2 / sectime,
            "x",
        )
    else:
        print("Failed ❌: results mismatch.")

    _free(a_device)
    _free(b_device)
    _free(c_device)
    _free(a_device_ref)
    _free(b_device_ref)
    _free(c_device_ref)

    _ = a_host
    _ = b_host
    _ = c_host
    _ = a_host_ref
    _ = a_host_ref
    _ = c_host_ref

    _ = func_mma ^
    _ = func_naive ^
    _ = stream ^


# CHECK-NOT: CUDA_ERROR
def main():
    try:
        with Context() as ctx:
            # Run tensor core versions of matmul, verify correctness & compare to naive.
            run_mma_fp32_fp16(16, 8, 8, -1e2, 1e2, 10, 0.01)
            run_mma_fp32_fp16(1024, 1024, 1024, -1e2, 1e2, 10, 0.01)
            run_mma_fp32_fp16(1024, 4096, 2048, -1e2, 1e2, 10, 0.01)

            run_mma_fp32_tf32(16, 8, 8, -1e2, 1e2, 10, 0.01)
            run_mma_fp32_tf32(1024, 1024, 1024, -1e2, 1e2, 10, 0.01)
            run_mma_fp32_tf32(1024, 4096, 2048, -1e2, 1e2, 10, 0.01)

            run_mma_fp16_fp16(16, 8, 8, -1e2, 1e2, 10, 0.01)
            run_mma_fp16_fp16(512, 128, 32, -1e1, 1e1, 10, 0.01)
            run_mma_fp16_fp16(128, 256, 64, -1e1, 1e1, 10, 0.01)

    except e:
        print("CUDA_ERROR:", e)
