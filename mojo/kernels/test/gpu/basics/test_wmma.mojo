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
from random import seed
from gpu.mma import mma
from Matmul import matmul_kernel


# Calculates c = a*b
# row/col indices are inentionally verbose to illustrate the usage of mma
# instruction and how registers correspond to a and b elements.
# Uses indexing information from:
# https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
# Chapter 9.7.13.4.7. Matrix Fragments for mma.m16n8k8
fn mma_kernel_single(
    c_ptr: DTypePointer[DType.float32],
    a_ptr: DTypePointer[DType.float32],
    b_ptr: DTypePointer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
):
    let x = BlockIdx.x() * BlockDim.x() + ThreadIdx.x()

    let laneId = x & 31
    let groupId = laneId >> 2
    let threadIdInGroup = laneId % 4

    # Indices (row, col) for registers a[0], a[1], a[2], a[3].
    let row_a0_a2 = groupId
    let row_a1_a3 = groupId + 8
    let col_a0_a1 = threadIdInGroup
    let col_a2_a3 = threadIdInGroup + 4

    # Indices (row, col) for registers b[0], b[1].
    let row_b0 = threadIdInGroup
    let row_b1 = threadIdInGroup + 4
    let col_b0_b1 = groupId

    # Indices (row, col) for registers d[0], d[1], d[2], d[3].
    # Same indices for registers c[0-3] (not used in this example).
    let row_cd0_cd1 = groupId
    let row_cd2_cd3 = groupId + 8
    let col_cd0 = (threadIdInGroup * 2) + (0 & 0x1)
    let col_cd1 = (threadIdInGroup * 2) + (1 & 0x1)
    let col_cd2 = (threadIdInGroup * 2) + (2 & 0x1)
    let col_cd3 = (threadIdInGroup * 2) + (3 & 0x1)

    let a = NDBuffer[2, DimList.create_unknown[2](), DType.float32](
        a_ptr, Index(m, k)
    )
    let b = NDBuffer[2, DimList.create_unknown[2](), DType.float32](
        b_ptr, Index(k, n)
    )
    let c = NDBuffer[2, DimList.create_unknown[2](), DType.float32](
        c_ptr, Index(m, n)
    )

    # Registers to cooperate within a warp in an mma instruction.
    var a_reg = SIMD[DType.float32, 4]()
    var b_reg = SIMD[DType.float32, 2]()
    var d_reg = SIMD[DType.float32, 4]()

    # Load each threads registers with the appropriate indices from
    # multiplicands a and b.
    a_reg[0] = a[Index(row_a0_a2, col_a0_a1)]
    a_reg[1] = a[Index(row_a1_a3, col_a0_a1)]
    a_reg[2] = a[Index(row_a0_a2, col_a2_a3)]
    a_reg[3] = a[Index(row_a1_a3, col_a2_a3)]
    b_reg[0] = b[Index(row_b0, col_b0_b1)]
    b_reg[1] = b[Index(row_b1, col_b0_b1)]

    # Perform mma operation.
    mma(d_reg, a_reg, b_reg, d_reg)

    # Write back to global memory to appropriate indices.
    c[Index(row_cd0_cd1, col_cd0)] = d_reg[0]
    c[Index(row_cd0_cd1, col_cd1)] = d_reg[1]
    c[Index(row_cd2_cd3, col_cd2)] = d_reg[2]
    c[Index(row_cd2_cd3, col_cd3)] = d_reg[3]


# Doing Tensor core Matmul with shape m16n8k8
fn mma_kernel(
    c_ptr: DTypePointer[DType.float32],
    a_ptr: DTypePointer[DType.float32],
    b_ptr: DTypePointer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
):
    let mma_m = 16
    let mma_n = 8
    let mma_k = 8

    var a_reg = SIMD[DType.float32, 4]()
    var b_reg = SIMD[DType.float32, 2]()
    var d_reg = SIMD[DType.float32, 4]()
    let tile_loops = k // mma_k
    let group_id = lane_id() >> 2
    let group_lane_id = lane_id() % 4

    for i in range(tile_loops):
        let a_tile_row = BlockIdx.x() * mma_m
        let a_tile_col = i * mma_k
        let b_tile_row = i * mma_k
        let b_tile_col = BlockIdx.y() * mma_n

        let a0_row = group_id
        let a0_col = group_lane_id
        let a1_row = group_id + 8
        let a1_col = group_lane_id
        let a2_row = group_id
        let a2_col = group_lane_id + 4
        let a3_row = group_id + 8
        let a3_col = group_lane_id + 4

        let b0_row = group_lane_id
        let b0_col = group_id
        let b1_row = group_lane_id + 4
        let b1_col = group_id

        # Populate registers with input matrices
        a_reg[0] = a_ptr.load((a_tile_row + a0_row) * k + (a_tile_col + a0_col))
        a_reg[1] = a_ptr.load((a_tile_row + a1_row) * k + (a_tile_col + a1_col))
        a_reg[2] = a_ptr.load((a_tile_row + a2_row) * k + (a_tile_col + a2_col))
        a_reg[3] = a_ptr.load((a_tile_row + a3_row) * k + (a_tile_col + a3_col))
        b_reg[0] = b_ptr.load((b_tile_row + b0_row) * n + (b_tile_col + b0_col))
        b_reg[1] = b_ptr.load((b_tile_row + b1_row) * n + (b_tile_col + b1_col))

        # Perform mma (d = a * b + d)
        mma(d_reg, a_reg, b_reg, d_reg)

    let c_tile_row = BlockIdx.x() * mma_m
    let c_tile_col = BlockIdx.y() * mma_n

    let c0_row = group_id
    let c0_col = (group_lane_id * 2) + (0 & 0x1)
    let c1_row = group_id
    let c1_col = (group_lane_id * 2) + (1 & 0x1)
    let c2_row = group_id + 8
    let c2_col = (group_lane_id * 2) + (2 & 0x1)
    let c3_row = group_id + 8
    let c3_col = (group_lane_id * 2) + (3 & 0x1)

    # Write back results
    c_ptr.store((c_tile_row + c0_row) * n + (c_tile_col + c0_col), d_reg[0])
    c_ptr.store((c_tile_row + c1_row) * n + (c_tile_col + c1_col), d_reg[1])
    c_ptr.store((c_tile_row + c2_row) * n + (c_tile_col + c2_col), d_reg[2])
    c_ptr.store((c_tile_row + c3_row) * n + (c_tile_col + c3_col), d_reg[3])


fn run_mma(M: Int, N: Int, K: Int, errorTolerance: Float32) raises:
    print("== run_matmul tensor core kernel")

    let iterations = 100
    let stream = Stream()
    let a_host = Pointer[Float32].alloc(M * K)
    let b_host = Pointer[Float32].alloc(K * N)
    let c_host = Pointer[Float32].alloc(M * N)
    let c_host_naive = Pointer[Float32].alloc(M * N)

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

    let func_mma = Function[
        fn (
            DTypePointer[DType.float32],
            DTypePointer[DType.float32],
            DTypePointer[DType.float32],
            Int,
            Int,
            Int,
        ) -> None, mma_kernel
    ]()

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
            DType.float32, DType.float32, DType.float32, BLOCK_DIM
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
    print("Shmem matmul kernel:")
    print(sectime2, "sec")
    print(flops * 1e-9 / sectime2, " GFLOPS")
    print()

    _copy_device_to_host(c_host_naive, c_device, M * N)

    # Check correctness.
    var failed = False
    for i in range(M * N):
        let outVal = c_host.load(i)
        let outRef = c_host_naive.load(i)
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
            "Performance basic tensor core matmul vs. shmem matmul: ",
            sectime2 / sectime,
            "x",
        )
    else:
        print("Failed ❌: results mismatch.")

    _free(a_device)
    _free(b_device)
    _free(c_device)

    _ = a_host
    _ = b_host
    _ = c_host
    _ = c_host_naive

    _ = func_mma ^
    _ = func_naive ^
    _ = stream ^


# CHECK-NOT: CUDA_ERROR
def main():
    try:
        with Context() as ctx:
            # Run mma version of matmul, verify correctness and compare to naive.
            run_mma(4096, 4096, 4096, 0.01)
            run_mma(4096, 2048, 1024, 0.01)
            run_mma(4096, 2048, 4096, 0.01)
            run_mma(16, 32, 128, 0.01)
            run_mma(32, 64, 32, 0.01)

    except e:
        print("CUDA_ERROR:", e)
