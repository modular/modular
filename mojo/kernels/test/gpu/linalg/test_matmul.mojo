# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s | FileCheck %s

from math import ceildiv, isclose
from random import random_float64

from buffer import NDBuffer
from buffer.list import DimList
from gpu import WARP_SIZE, AddressSpace, BlockDim, BlockIdx, ThreadIdx, barrier
from gpu.host import Context, CUDADeviceStream, Function, Stream, synchronize
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
)
from LinAlg.MatmulGPU import _matmul_gpu, matmul_kernel_naive
from memory import memset_zero, stack_allocation
from testing import assert_almost_equal, assert_true

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

    var a_host_mem = DTypePointer[DType.float32].alloc(m * k)
    var b_host_mem = DTypePointer[DType.float32].alloc(k * m)
    var c_host_mem = DTypePointer[DType.float32].alloc(m * n)

    var a_host = NDBuffer[DType.float32, 2, DimList(m, k)](a_host_mem)
    var b_host = NDBuffer[DType.float32, 2, DimList(k, m)](b_host_mem)
    var c_host = NDBuffer[DType.float32, 2, DimList(m, n)](c_host_mem)

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

    _copy_host_to_device(a_device, a_host.data, m * k)
    _copy_host_to_device(b_device, b_host.data, k * n)

    var func = Function[matmul]()

    func(
        a_device,
        b_device,
        c_device,
        m,
        n,
        k,
        grid_dim=(ceildiv(m, TILE_SZ_A), ceildiv(n, TILE_SZ_B)),
        block_dim=(TILE_SZ_A, 1),
        stream=stream,
    )
    synchronize()

    _copy_device_to_host(c_host.data, c_device, m * n)

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


fn test_gemm_transpose_b[type: DType, M: Int, N: Int, K: Int]() raises:
    var stream = Stream()

    var a_host_mem = DTypePointer[type].alloc(M * K)
    var b_host_mem = DTypePointer[type].alloc(K * N)
    var c_host_mem = DTypePointer[type].alloc(M * N)
    var c_host_ref_mem = DTypePointer[type].alloc(M * N)

    var a_host = NDBuffer[type, 2, DimList(M, K)](a_host_mem)
    var b_host = NDBuffer[type, 2, DimList(K, N)](b_host_mem)
    var c_host = NDBuffer[type, 2, DimList(M, N)](c_host_mem)
    var c_host_ref = NDBuffer[type, 2, DimList(M, N)]((c_host_ref_mem))

    a_host.fill(1)
    b_host.fill(1)
    c_host.fill(0)

    var a_device = _malloc[Scalar[type]](M * K)
    var b_device = _malloc[Scalar[type]](N * K)
    var c_device = _malloc[Scalar[type]](M * N)
    var c_device_ref = _malloc[Scalar[type]](M * N)

    alias a_shape = DimList(M, K)
    var a_device_nd = NDBuffer[type, 2, a_shape](a_device, Index(M, K))
    alias b_shape = DimList(K, N)
    var b_device_nd = NDBuffer[type, 2, b_shape](b_device, Index(N, K))
    alias c_shape = DimList(M, N)
    var c_device_nd = NDBuffer[type, 2, c_shape](c_device, Index(M, N))
    _copy_host_to_device(a_device, a_host.data, M * K)
    _copy_host_to_device(b_device, b_host.data, K * N)
    _copy_host_to_device(c_device, c_host.data, M * N)
    _copy_host_to_device(c_device_ref, c_host_ref.data, M * N)

    _matmul_gpu[use_tensor_core=False](
        c_device_nd, a_device_nd, b_device_nd, CUDADeviceStream(stream)
    )
    synchronize()
    _copy_device_to_host(c_host.data, c_device, M * N)

    alias BLOCK_DIM = 16
    var func_naive = Function[
        matmul_kernel_naive[type, type, type, BLOCK_DIM, True]
    ]()

    func_naive(
        c_device_ref,
        a_device,
        b_device,
        M,
        N,
        K,
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
        block_dim=(BLOCK_DIM, BLOCK_DIM),
        stream=stream,
    )

    synchronize()
    _copy_device_to_host(c_host_ref.data, c_device_ref, M * N)

    for i in range(M):
        for j in range(N):
            assert_true(isclose(c_host_ref[i, j], c_host[i, j]))

    _free(a_device)
    _free(b_device)
    _free(c_device)
    _free(c_device_ref)

    a_host_mem.free()
    b_host_mem.free()
    c_host_mem.free()
    c_host_ref_mem.free()

    _ = func_naive^
    _ = stream^


fn run_matmul_from_mogg_interface[M: Int, K: Int, N: Int, type: DType]() raises:
    var stream = Stream()

    var a_host_mem = DTypePointer[type].alloc(M * K)
    var b_host_mem = DTypePointer[type].alloc(K * N)
    var c_host_mem = DTypePointer[type].alloc(M * N)
    var c_host_ref_mem = DTypePointer[type].alloc(M * N)

    var a_host = NDBuffer[type, 2, DimList(M, K)](a_host_mem)
    var b_host = NDBuffer[type, 2, DimList(K, N)](b_host_mem)
    var c_host = NDBuffer[type, 2, DimList(M, N)](c_host_mem)
    var c_host_ref = NDBuffer[type, 2, DimList(M, N)]((c_host_ref_mem))

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
    _copy_host_to_device(a_device, a_host.data, M * K)
    _copy_host_to_device(b_device, b_host.data, K * N)
    _copy_host_to_device(c_device, c_host.data, M * N)
    _copy_host_to_device(c_device_ref, c_host_ref.data, M * N)

    _matmul_gpu[use_tensor_core=False](
        c_device_nd, a_device_nd, b_device_nd, CUDADeviceStream(stream)
    )
    synchronize()
    _copy_device_to_host(c_host.data, c_device, M * N)

    alias BLOCK_DIM = 16
    var func_naive = Function[
        matmul_kernel_naive[type, type, type, BLOCK_DIM]
    ]()

    func_naive(
        c_device_ref,
        a_device,
        b_device,
        M,
        N,
        K,
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
        block_dim=(BLOCK_DIM, BLOCK_DIM),
        stream=stream,
    )

    synchronize()
    _copy_device_to_host(c_host_ref.data, c_device_ref, M * N)

    for i in range(M):
        for j in range(N):
            assert_true(isclose(c_host_ref[i, j], c_host[i, j]))

    _free(a_device)
    _free(b_device)
    _free(c_device)
    _free(c_device_ref)

    a_host_mem.free()
    b_host_mem.free()
    c_host_mem.free()
    c_host_ref_mem.free()

    _ = func_naive^
    _ = stream^


fn run_matmul_from_mogg_interface_with_epilogue[
    M: Int, K: Int, N: Int, type: DType
]() raises:
    var stream = Stream()

    var a_host_mem = DTypePointer[type].alloc(M * K)
    var b_host_mem = DTypePointer[type].alloc(K * N)
    var c_host_mem = DTypePointer[type].alloc(M * N)
    var c_host_ref_mem = DTypePointer[type].alloc(M * N)

    var a_host = NDBuffer[type, 2, DimList(M, K)](a_host_mem)
    var b_host = NDBuffer[type, 2, DimList(K, N)](b_host_mem)
    var c_host = NDBuffer[type, 2, DimList(M, N)](c_host_mem)
    var c_host_ref = NDBuffer[type, 2, DimList(M, N)]((c_host_ref_mem))

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
    _copy_host_to_device(a_device, a_host.data, M * K)
    _copy_host_to_device(b_device, b_host.data, K * N)
    _copy_host_to_device(c_device, c_host.data, M * N)
    _copy_host_to_device(c_device_ref, c_host_ref.data, M * N)

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

    _matmul_gpu[
        use_tensor_core=False,
        transpose_b=False,
        elementwise_lambda_fn=epilogue_fn,
    ](c_device_nd, a_device_nd, b_device_nd, CUDADeviceStream(stream))
    synchronize()
    _copy_device_to_host(c_host.data, c_device, M * N)

    alias BLOCK_DIM = 16
    var func_naive = Function[
        matmul_kernel_naive[
            type,
            type,
            type,
            BLOCK_DIM,
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
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
        block_dim=(BLOCK_DIM, BLOCK_DIM),
        stream=stream,
    )

    synchronize()
    _copy_device_to_host(c_host_ref.data, c_device_ref, M * N)

    for i in range(M):
        for j in range(N):
            assert_true(isclose(c_host_ref[i, j], c_host[i, j]))

    _free(a_device)
    _free(b_device)
    _free(c_device)
    _free(c_device_ref)

    a_host_mem.free()
    b_host_mem.free()
    c_host_mem.free()
    c_host_ref_mem.free()

    _ = func_naive^
    _ = stream^


fn run_low_precision_test[
    M: Int,
    N: Int,
    K: Int,
    type: DType,
    accum_type: DType,
    transpose_b: Bool = False,
]() raises:
    var stream = Stream()

    var a_host = DTypePointer[type].alloc(M * K)
    var b_host = DTypePointer[type].alloc(K * N)
    var b_trans_host = DTypePointer[type].alloc(K * N)
    var c_host = DTypePointer[accum_type].alloc(M * N)
    var c_host_ref = DTypePointer[accum_type].alloc(M * N)

    for m in range(M):
        for k in range(K):
            a_host[m * K + k] = m * K + k

    for k in range(K):
        for n in range(N):
            b_host[k * N + n] = k * N + n

            @parameter
            if transpose_b:
                b_trans_host[n * K + k] = k * N + n
            else:
                b_trans_host[k * N + n] = k * N + n

    var a_device = _malloc[type](M * K)
    var b_device = _malloc[type](K * N)
    var c_device = _malloc[accum_type](M * N)
    var c_device_ref = _malloc[accum_type](M * N)

    _copy_host_to_device(a_device, a_host, M * K)
    _copy_host_to_device(b_device, b_trans_host, K * N)

    var a_buffer = NDBuffer[type, 2, DimList(M, K)](a_device)
    var b_buffer = NDBuffer[type, 2, DimList(K, N)](b_device)
    var c_buffer = NDBuffer[accum_type, 2, DimList(M, N)](c_device)

    _matmul_gpu[
        use_tensor_core=True,
        transpose_b=False,
    ](c_buffer, a_buffer, b_buffer, CUDADeviceStream(stream))

    synchronize()

    _copy_device_to_host(c_host, c_device, M * N)
    _copy_host_to_device(b_device, b_host, K * N)

    # Naive gemm.
    alias BLOCK_DIM = 16
    alias gemm_naive = matmul_kernel_naive[accum_type, type, type, BLOCK_DIM]
    var func_naive = Function[gemm_naive](threads_per_block=256)
    var c_buffer_ref = NDBuffer[accum_type, 2, DimList(M, N)](c_device_ref)
    func_naive(
        c_buffer_ref,
        a_buffer,
        b_buffer,
        M,
        N,
        K,
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM), 1),
        block_dim=(BLOCK_DIM, BLOCK_DIM, 1),
    )

    synchronize()
    _copy_device_to_host(c_host_ref, c_device_ref, M * N)

    for i in range(M * N):
        if not isclose(c_host[i], c_host_ref[i], rtol=0.01):
            print(i, c_host[i], c_host_ref[i])
        assert_almost_equal(c_host[i], c_host_ref[i], rtol=0.01)

    _free(c_device)
    _free(c_device_ref)
    _free(a_device)
    _free(b_device)

    c_host.free()
    c_host_ref.free()
    a_host.free()
    b_host.free()
    b_trans_host.free()

    _ = func_naive^
    _ = stream^


fn run_low_precision_test_with_epilogue[
    M: Int,
    N: Int,
    K: Int,
    type: DType,
    accum_type: DType,
    transpose_b: Bool = False,
]() raises:
    var stream = Stream()

    var a_host = DTypePointer[type].alloc(M * K)
    var b_host = DTypePointer[type].alloc(K * N)
    var b_trans_host = DTypePointer[type].alloc(K * N)
    var c_host = DTypePointer[accum_type].alloc(M * N)
    var c_host_ref = DTypePointer[accum_type].alloc(M * N)

    for m in range(M):
        for k in range(K):
            a_host[m * K + k] = m * K + k

    for k in range(K):
        for n in range(N):
            b_host[k * N + n] = k * N + n

            @parameter
            if transpose_b:
                b_trans_host[n * K + k] = k * N + n
            else:
                b_trans_host[k * N + n] = k * N + n

    var a_device = _malloc[type](M * K)
    var b_device = _malloc[type](K * N)
    var c_device = _malloc[accum_type](M * N)
    var c_device_ref = _malloc[accum_type](M * N)

    _copy_host_to_device(a_device, a_host, M * K)
    _copy_host_to_device(b_device, b_trans_host, K * N)

    var a_buffer = NDBuffer[type, 2, DimList(M, K)](a_device)
    var b_buffer = NDBuffer[type, 2, DimList(K, N)](b_device)
    var c_buffer = NDBuffer[accum_type, 2, DimList(M, N)](c_device)
    var c_buffer_ref = NDBuffer[accum_type, 2, DimList(M, N)](c_device_ref)

    alias some_constant = 20

    @parameter
    @always_inline
    @__copy_capture(c_buffer)
    fn epilogue_fn[
        _type: DType, width: Int
    ](idx: StaticIntTuple[2], val: SIMD[_type, width]) capturing -> None:
        c_buffer.store(
            idx, rebind[SIMD[accum_type, width]](val + some_constant)
        )

    @parameter
    @always_inline
    @__copy_capture(c_buffer_ref)
    fn naive_epilogue_fn[
        _type: DType, width: Int
    ](idx: StaticIntTuple[2], val: SIMD[_type, width]) capturing -> None:
        c_buffer_ref.store(
            idx, rebind[SIMD[accum_type, width]](val + some_constant)
        )

    _matmul_gpu[
        use_tensor_core=True,
        transpose_b=False,
        elementwise_lambda_fn=epilogue_fn,
    ](c_buffer, a_buffer, b_buffer, CUDADeviceStream(stream))

    synchronize()

    _copy_device_to_host(c_host, c_device, M * N)
    _copy_host_to_device(b_device, b_host, K * N)

    # Naive gemm.
    alias BLOCK_DIM = 16
    alias gemm_naive = matmul_kernel_naive[
        accum_type,
        type,
        type,
        BLOCK_DIM,
        elementwise_lambda_fn=naive_epilogue_fn,
    ]
    var func_naive = Function[gemm_naive](threads_per_block=256)
    func_naive(
        c_buffer_ref,
        a_buffer,
        b_buffer,
        M,
        N,
        K,
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM), 1),
        block_dim=(BLOCK_DIM, BLOCK_DIM, 1),
    )

    synchronize()
    _copy_device_to_host(c_host_ref, c_device_ref, M * N)

    for i in range(M * N):
        if not isclose(c_host[i], c_host_ref[i], rtol=0.01):
            print(i, c_host[i], c_host_ref[i])
        assert_almost_equal(c_host[i], c_host_ref[i], rtol=0.01)

    _free(c_device)
    _free(c_device_ref)
    _free(a_device)
    _free(b_device)

    c_host.free()
    c_host_ref.free()
    a_host.free()
    b_host.free()
    b_trans_host.free()

    _ = func_naive^


# CHECK-NOT: CUDA_ERROR
def main():
    try:
        with Context() as ctx:
            run_matmul()
            test_gemm_transpose_b[DType.float32, 512, 512, 512]()
            test_gemm_transpose_b[DType.float32, 512, 1024, 3072]()
            run_matmul_from_mogg_interface[1024, 3072, 5120, DType.float32]()
            run_matmul_from_mogg_interface[1024, 12288, 3072, DType.float32]()
            run_matmul_from_mogg_interface_with_epilogue[
                1024, 3072, 5120, DType.float32
            ]()
            run_matmul_from_mogg_interface_with_epilogue[
                1024, 3072, 5120, DType.bfloat16
            ]()
            run_low_precision_test[
                1024, 3072, 5120, DType.float32, DType.float32
            ]()
            run_low_precision_test_with_epilogue[
                1024, 3072, 5120, DType.float32, DType.float32
            ]()
    except e:
        print("CUDA_ERROR:", e)
