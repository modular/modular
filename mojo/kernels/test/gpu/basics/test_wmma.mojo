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
from utils.index import Index
from utils.list import DimList
from random import seed
from gpu.mma import mma
from Matmul import matmul_kernel_naive


# Calculates c = a*b
# row/col indices are inentionally verbose to illustrate the usage of mma
# instruction and how registers correspond to a and b elements.
# Uses indexing information from:
# https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
# Chapter 9.7.13.4.7. Matrix Fragments for mma.m16n8k8
fn mma_kernel(
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


fn run_mma(M: Int, N: Int, K: Int) raises:
    print("== run_matvec kernel")

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

    func_mma(
        1,
        WARP_SIZE,
        c_device,
        a_device,
        b_device,
        M,
        N,
        K,
        stream=stream,
    )

    synchronize()

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
        ) capturing -> None, matmul_kernel_naive[
            DType.float32, DType.float32, DType.float32, BLOCK_DIM
        ]
    ]()

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

    synchronize()

    _copy_device_to_host(c_host_naive, c_device, M * N)

    # Check correctness.
    let errorTolerance = 0.0001
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

    # CHECK: Success
    if not failed:
        print("Success 🎉: Results match.")
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
            run_mma(16, 8, 8)

    except e:
        print("CUDA_ERROR:", e)
