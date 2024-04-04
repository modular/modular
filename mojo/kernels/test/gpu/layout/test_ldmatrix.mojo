# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo %s | FileCheck %s

from math import div_ceil, max, min, isclose
from random import random_si64, seed

from buffer import NDBuffer
from buffer.list import DimList
from gpu import (
    WARP_SIZE,
    BlockDim,
    BlockIdx,
    GridDim,
    ThreadIdx,
    barrier,
    lane_id,
)
from gpu.host import Context, Dim, Function, Stream, synchronize
from gpu.host.event import time_function
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
)
from gpu.mma import mma, ld_matrix
from gpu.mma_util import load_matrix_a, load_matrix_b, store_matrix_d
from gpu.sync import syncwarp
from Matmul import matmul_kernel, matmul_kernel_naive
from memory import memset_zero, stack_allocation
from memory.unsafe import DTypePointer, bitcast
from gpu.memory import AddressSpace

from utils.index import Index
from testing import *


fn test_ldmatrix(
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

    var d_reg = SIMD[DType.float32, 4](0)
    var tid = ThreadIdx.x()
    var a_shared = stack_allocation[
        mma_m * mma_k,
        DType.float32,
        alignment=32,
        address_space = AddressSpace.SHARED,
    ]()
    var b_shared = stack_allocation[
        mma_n * mma_k,
        DType.float32,
        alignment=32,
        address_space = AddressSpace.SHARED,
    ]()

    for i in range(tid, mma_m * mma_k, WARP_SIZE):
        a_shared[i] = a_ptr[i]
    for i in range(tid, mma_k * mma_n, WARP_SIZE):
        b_shared[i] = b_ptr[i]

    barrier()

    var a_reg = ld_matrix[DType.float32, 4](
        a_shared + ((lane_id() % 16) * 8 + (lane_id() // 16) * 4)
    )
    var b_reg = load_matrix_b[mma_m, mma_n, mma_k](b_shared, 0, 0, n)
    mma(d_reg, a_reg, b_reg, d_reg)
    store_matrix_d[DType.float32, mma_m, mma_n, mma_k](c_ptr, d_reg, 0, 0, n)


fn check_ldmatrix(
    M: Int,
    N: Int,
    K: Int,
    rand_min: Int64,
    rand_max: Int64,
) raises:
    print("== test ldmatrix instruction")

    var stream = Stream()
    var a_host = Pointer[Float32].alloc(M * K)
    var b_host = Pointer[Float32].alloc(K * N)
    var c_host = Pointer[Float32].alloc(M * N)
    var a_host_ref = Pointer[Float32].alloc(M * K)
    var b_host_ref = Pointer[Float32].alloc(K * N)
    var c_host_ref = Pointer[Float32].alloc(M * N)

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

    var a_device = _malloc[Float32](M * K)
    var b_device = _malloc[Float32](K * N)
    var c_device = _malloc[Float32](M * N)
    var a_device_ref = _malloc[Float32](M * K)
    var b_device_ref = _malloc[Float32](K * N)
    var c_device_ref = _malloc[Float32](M * N)

    _copy_host_to_device(a_device, a_host, M * K)
    _copy_host_to_device(b_device, b_host, K * N)

    var func_ldmatrix = Function[__type_of(test_ldmatrix), test_ldmatrix](
        dump_ptx=False
    )

    alias WARP_PER_BLOCK = 1
    alias MMA_M = 16
    alias MMA_N = 8
    alias MMA_K = 8

    @always_inline
    @__copy_capture(func_ldmatrix, c_device, a_device, b_device)
    @parameter
    fn run_ldmatrix(stream: Stream) raises:
        func_ldmatrix(
            c_device,
            a_device,
            b_device,
            M,
            N,
            K,
            grid_dim=1,
            block_dim=WARP_SIZE,
            stream=stream,
        )

    run_ldmatrix(stream)
    stream.synchronize()

    _copy_device_to_host(c_host, c_device, M * N)

    # Run naive matmul.
    _copy_host_to_device(a_device_ref, a_host_ref, M * K)
    _copy_host_to_device(b_device_ref, b_host_ref, K * N)

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
            DType.float32, DType.float32, DType.float32, BLOCK_DIM
        ]
    ]()

    @always_inline
    @__copy_capture(func_naive, c_device_ref, a_device_ref, b_device_ref)
    @parameter
    fn run_func_naive(stream: Stream) raises:
        func_naive(
            c_device_ref,
            a_device_ref,
            b_device_ref,
            M,
            N,
            K,
            grid_dim=(div_ceil(M, BLOCK_DIM), div_ceil(N, BLOCK_DIM)),
            block_dim=(BLOCK_DIM, BLOCK_DIM),
            stream=stream,
        )

    run_func_naive(stream)
    stream.synchronize()
    _copy_device_to_host(c_host_ref, c_device_ref, M * N)

    for i in range(M * N):
        var out_val = c_host.load(i)
        var out_ref = c_host_ref.load(i)
        testing.assert_true(math.isclose(out_val, out_ref))

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

    _ = func_ldmatrix^
    _ = func_naive^
    _ = stream^


# CHECK-NOT: CUDA_ERROR
def main():
    try:
        with Context() as ctx:
            check_ldmatrix(16, 8, 8, -1e2, 1e2)

    except e:
        print("CUDA_ERROR:", e)
