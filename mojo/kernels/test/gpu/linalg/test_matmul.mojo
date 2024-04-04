# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo %s | FileCheck %s

from math import div_ceil
from pathlib import Path
from random import random_float64
from buffer import NDBuffer
from buffer.list import DimList
from gpu import AddressSpace, BlockDim, BlockIdx, ThreadIdx, barrier
from gpu.host import Context, Dim, Function, Stream, synchronize
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
)
from Matmul import matmul as _matmul
from MatmulGPU import matmul_kernel_naive, matmul_kernel
from memory import memset_zero, stack_allocation
from tensor import Tensor
from math import isclose
from testing import assert_true

from utils.index import Index

alias TILE_SZ_A = 128
alias TILE_SZ_B = 16
alias TILE_SZ_RATIO = TILE_SZ_A // TILE_SZ_B


fn matmul(
    a_ptr: DTypePointer[DType.float32],
    b_ptr: DTypePointer[DType.float32],
    c_ptr: DTypePointer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
):
    var a = NDBuffer[DType.float32, 2](a_ptr, Index(m, k))
    var b = NDBuffer[DType.float32, 2](b_ptr, Index(k, n))
    var c = NDBuffer[DType.float32, 2](c_ptr, Index(m, n))

    # Compute C = A x B
    #   where A is a (m x k) matrix
    #   where B is a (k x n) matrix
    #   where C is a (m x n) matrix
    #
    # Use register and shared memory tiling and thread coarsening
    #
    # NOTE: A and C are column major, B is row major.

    # Allocate B array into shared memory for tiling.
    var b_shared = stack_allocation[
        TILE_SZ_RATIO * TILE_SZ_B,
        DType.float32,
        address_space = AddressSpace.SHARED,
    ]()

    # Thread indexing offsets.
    var row = BlockIdx.x() * BlockDim.x() + ThreadIdx.x()
    var col = BlockIdx.y() * TILE_SZ_B

    # Privatization of the C matrix.
    var c_reg = stack_allocation[TILE_SZ_B, DType.float32]()

    memset_zero(c_reg, TILE_SZ_B)

    # Loop over each input tile.
    for tile_idx in range((k - 1) // TILE_SZ_RATIO + 1):
        var i = ThreadIdx.x() // TILE_SZ_B
        var j = ThreadIdx.x() % TILE_SZ_B

        # Load the B matrix into shared memory.
        var b_val: Float32
        if tile_idx * TILE_SZ_RATIO + i < k and col + j < n:
            b_val = b[tile_idx * TILE_SZ_RATIO + i, col + j]
        else:
            b_val = 0
        b_shared[i * TILE_SZ_B + j] = b_val

        barrier()

        # Loop within the tile.
        for idx in range(TILE_SZ_RATIO):
            # Load the A tile into the register.
            var a_reg: Float32
            if row < m and tile_idx * TILE_SZ_RATIO + idx < k:
                a_reg = a[row, tile_idx * TILE_SZ_RATIO + idx]
            else:
                a_reg = 0

            # Compute the output element for each thread.
            for out_idx in range(TILE_SZ_B):
                c_reg[out_idx] += (
                    a_reg * b_shared[idx * TILE_SZ_RATIO + out_idx]
                )
        barrier()

    # Store the values into the output matrix.
    for out_idx in range(TILE_SZ_B):
        if row < m and col + out_idx < n:
            c[Index(row, col + out_idx)] = c_reg[out_idx]


# CHECK-LABEL: run_matmul
fn run_matmul() raises:
    print("== run_matmul")

    alias m = 512
    alias n = 512
    alias k = 512

    var stream = Stream()

    var a_host = Tensor[DType.float32](m, k)
    var b_host = Tensor[DType.float32](k, n)
    var c_host = Tensor[DType.float32](m, n)

    for i in range(m):
        for j in range(k):
            a_host[Index(i, j)] = 1

    for i in range(k):
        for j in range(n):
            b_host[Index(i, j)] = 1

    for i in range(m):
        for j in range(n):
            c_host[Index(i, j)] = 0

    var a_device = _malloc[Float32](m * k)
    var b_device = _malloc[Float32](k * n)
    var c_device = _malloc[Float32](m * n)

    _copy_host_to_device(a_device, a_host.data(), m * k)
    _copy_host_to_device(b_device, b_host.data(), k * n)

    var func = Function[__type_of(matmul), matmul]()

    func(
        a_device,
        b_device,
        c_device,
        m,
        n,
        k,
        grid_dim=(div_ceil(m, TILE_SZ_A), div_ceil(n, TILE_SZ_B)),
        block_dim=(TILE_SZ_A, 1),
        stream=stream,
    )
    synchronize()

    _copy_device_to_host(c_host.data(), c_device, m * n)

    for i in range(10):
        for j in range(10):
            print("at index = [", i, ",", j, "] the value is", c_host[i, j])

    _free(a_device)
    _free(b_device)
    _free(c_device)

    _ = a_host
    _ = b_host
    _ = c_host

    _ = func^
    _ = stream^


fn run_matmul_from_mogg_interface[M: Int, K: Int, N: Int, type: DType]() raises:
    var stream = Stream()

    var a_host = Tensor[type](M, K)
    var b_host = Tensor[type](K, N)
    var c_host = Tensor[type](M, N)
    var c_host_ref = Tensor[type](M, N)

    for i in range(M):
        for j in range(K):
            a_host[Index(i, j)] = 1

    for i in range(K):
        for j in range(N):
            b_host[Index(i, j)] = 1

    for i in range(M):
        for j in range(N):
            c_host[Index(i, j)] = 0
            c_host_ref[Index(i, j)] = 0

    var a_device = _malloc[Scalar[type]](M * K)
    var b_device = _malloc[Scalar[type]](K * N)
    var c_device = _malloc[Scalar[type]](M * N)
    var c_device_ref = _malloc[Scalar[type]](M * N)

    alias a_shape = DimList(M, K)
    var a_device_nd = NDBuffer[type, 2, a_shape](a_device, Index(M, K))
    alias b_shape = DimList(K, N)
    var b_device_nd = NDBuffer[type, 2, b_shape](b_device, Index(K, N))
    alias c_shape = DimList(M, N)
    var c_device_nd = NDBuffer[type, 2, c_shape](c_device, Index(M, N))
    _copy_host_to_device(a_device, a_host.data(), M * K)
    _copy_host_to_device(b_device, b_host.data(), K * N)
    _copy_host_to_device(c_device, c_host.data(), M * N)
    _copy_host_to_device(c_device_ref, c_host_ref.data(), M * N)

    _matmul[
        type,
        a_shape,
        type,
        b_shape,
        type,
        c_shape,
        target="cuda",
    ](c_device_nd, a_device_nd, b_device_nd)
    synchronize()
    _copy_device_to_host(c_host.data(), c_device, M * N)

    alias BLOCK_DIM = 16
    var func_naive = Function[
        fn (
            DTypePointer[type],
            DTypePointer[type],
            DTypePointer[type],
            Int,
            Int,
            Int,
        ) capturing -> None, matmul_kernel_naive[type, type, type, BLOCK_DIM,]
    ]()

    func_naive(
        c_device_ref,
        a_device,
        b_device,
        M,
        N,
        K,
        grid_dim=(div_ceil(M, BLOCK_DIM), div_ceil(N, BLOCK_DIM)),
        block_dim=(BLOCK_DIM, BLOCK_DIM),
        stream=stream,
    )

    synchronize()
    _copy_device_to_host(c_host_ref.data(), c_device_ref, M * N)

    for i in range(M):
        for j in range(N):
            assert_true(isclose(c_host_ref[i, j], c_host[i, j]))

    _free(a_device)
    _free(b_device)
    _free(c_device)
    _free(c_device_ref)

    _ = a_host
    _ = b_host
    _ = c_host
    _ = c_host_ref

    _ = func_naive^
    _ = stream^


fn run_matmul_from_mogg_interface_with_epilogue[
    M: Int, K: Int, N: Int, type: DType
]() raises:
    var stream = Stream()

    var a_host = Tensor[type](M, K)
    var b_host = Tensor[type](K, N)
    var c_host = Tensor[type](M, N)
    var c_host_ref = Tensor[type](M, N)

    for i in range(M):
        for j in range(K):
            a_host[Index(i, j)] = random_float64(-10, 10)

    for i in range(K):
        for j in range(N):
            b_host[Index(i, j)] = random_float64(-10, 10)

    for i in range(M):
        for j in range(N):
            c_host[Index(i, j)] = 0
            c_host_ref[Index(i, j)] = 0

    var a_device = _malloc[Scalar[type]](M * K)
    var b_device = _malloc[Scalar[type]](K * N)
    var c_device = _malloc[Scalar[type]](M * N)
    var c_device_ref = _malloc[Scalar[type]](M * N)

    alias a_shape = DimList(M, K)
    var a_device_nd = NDBuffer[type, 2, a_shape](a_device, Index(M, K))
    alias b_shape = DimList(K, N)
    var b_device_nd = NDBuffer[type, 2, b_shape](b_device, Index(K, N))
    alias c_shape = DimList(M, N)
    var c_device_nd = NDBuffer[type, 2, c_shape](c_device, Index(M, N))
    var c_device_ref_nd = NDBuffer[type, 2, c_shape](c_device_ref, Index(M, N))
    _copy_host_to_device(a_device, a_host.data(), M * K)
    _copy_host_to_device(b_device, b_host.data(), K * N)
    _copy_host_to_device(c_device, c_host.data(), M * N)
    _copy_host_to_device(c_device_ref, c_host_ref.data(), M * N)

    alias some_constant = 20

    @parameter
    @always_inline
    @__copy_capture(c_device_nd)
    fn epilogue_fn[
        _type: DType, width: Int
    ](idx: StaticIntTuple[2], val: SIMD[_type, width]) capturing -> None:
        c_device_nd.store(idx, rebind[SIMD[type, width]](val + some_constant))

    @parameter
    @always_inline
    @__copy_capture(c_device_ref_nd)
    fn naive_epilogue_fn[
        _type: DType, width: Int
    ](idx: StaticIntTuple[2], val: SIMD[_type, width]) capturing -> None:
        c_device_ref_nd.store(
            idx, rebind[SIMD[type, width]](val + some_constant)
        )

    _matmul[
        type,
        a_shape,
        type,
        b_shape,
        type,
        c_shape,
        target="cuda",
        elementwise_lambda_fn=epilogue_fn,
    ](c_device_nd, a_device_nd, b_device_nd)
    synchronize()
    _copy_device_to_host(c_host.data(), c_device, M * N)

    alias BLOCK_DIM = 16
    var func_naive = Function[
        fn (
            DTypePointer[type],
            DTypePointer[type],
            DTypePointer[type],
            Int,
            Int,
            Int,
        ) capturing -> None, matmul_kernel_naive[
            type,
            type,
            type,
            BLOCK_DIM,
            DType.float32,
            elementwise_lambda_fn=naive_epilogue_fn,
        ]
    ]()

    func_naive(
        c_device_ref,
        a_device,
        b_device,
        M,
        N,
        K,
        grid_dim=(div_ceil(M, BLOCK_DIM), div_ceil(N, BLOCK_DIM)),
        block_dim=(BLOCK_DIM, BLOCK_DIM),
        stream=stream,
    )

    synchronize()
    _copy_device_to_host(c_host_ref.data(), c_device_ref, M * N)

    for i in range(M):
        for j in range(N):
            assert_true(isclose(c_host_ref[i, j], c_host[i, j]))

    _free(a_device)
    _free(b_device)
    _free(c_device)
    _free(c_device_ref)

    _ = a_host
    _ = b_host
    _ = c_host
    _ = c_host_ref

    _ = func_naive^
    _ = stream^


# CHECK-NOT: CUDA_ERROR
def main():
    try:
        with Context() as ctx:
            run_matmul()
            run_matmul_from_mogg_interface[1024, 3072, 5120, DType.float32]()
            run_matmul_from_mogg_interface[1024, 12288, 3072, DType.float32]()
            run_matmul_from_mogg_interface_with_epilogue[
                1024, 3072, 5120, DType.float32
            ]()
            run_matmul_from_mogg_interface_with_epilogue[
                1024, 3072, 5120, DType.bfloat16
            ]()
    except e:
        print("CUDA_ERROR:", e)
