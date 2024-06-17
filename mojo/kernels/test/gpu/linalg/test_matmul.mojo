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
from gpu import WARP_SIZE, BlockDim, BlockIdx, ThreadIdx, barrier
from gpu.host.device_context import DeviceBuffer, DeviceContext
from linalg.matmul_gpu import _matmul_gpu, matmul_kernel_naive
from memory import memset_zero, stack_allocation
from memory.reference import _GPUAddressSpace as AddressSpace
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
fn run_matmul(ctx: DeviceContext) raises:
    print("== run_matmul")

    alias m = 512
    alias n = 512
    alias k = 512

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

    var a_device = ctx.create_buffer[DType.float32](m * k)
    var b_device = ctx.create_buffer[DType.float32](k * n)
    var c_device = ctx.create_buffer[DType.float32](m * n)

    ctx.enqueue_copy_to_device(a_device, a_host.data)
    ctx.enqueue_copy_to_device(b_device, b_host.data)

    var func_matmul = ctx.compile_function[matmul]()

    ctx.enqueue_function(
        func_matmul,
        a_device,
        b_device,
        c_device,
        m,
        n,
        k,
        grid_dim=(ceildiv(m, TILE_SZ_A), ceildiv(n, TILE_SZ_B)),
        block_dim=(TILE_SZ_A, 1),
    )
    ctx.enqueue_copy_from_device(c_host.data, c_device)

    ctx.synchronize()

    for i in range(10):
        for j in range(10):
            print("at index = [", i, ",", j, "] the value is", c_host[i, j])

    _ = a_device
    _ = b_device
    _ = c_device

    _ = a_host
    _ = b_host
    _ = c_host

    _ = func_matmul^


fn test_gemm_transpose_b[
    type: DType, M: Int, N: Int, K: Int
](ctx: DeviceContext) raises:
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

    var a_device = ctx.create_buffer[type](M * K)
    var b_device = ctx.create_buffer[type](N * K)
    var c_device = ctx.create_buffer[type](M * N)
    var c_device_ref = ctx.create_buffer[type](M * N)

    alias a_shape = DimList(M, K)
    var a_device_nd = NDBuffer[type, 2, a_shape](a_device.ptr, Index(M, K))
    alias b_shape = DimList(K, N)
    var b_device_nd = NDBuffer[type, 2, b_shape](b_device.ptr, Index(N, K))
    alias c_shape = DimList(M, N)
    var c_device_nd = NDBuffer[type, 2, c_shape](c_device.ptr, Index(M, N))
    ctx.enqueue_copy_to_device(a_device, a_host.data)
    ctx.enqueue_copy_to_device(b_device, b_host.data)
    ctx.enqueue_copy_to_device(c_device, c_host.data)
    ctx.enqueue_copy_to_device(c_device_ref, c_host_ref.data)

    _matmul_gpu[use_tensor_core=False](
        c_device_nd, a_device_nd, b_device_nd, ctx
    )
    ctx.enqueue_copy_from_device(c_host.data, c_device)

    alias BLOCK_DIM = 16
    var func_naive = ctx.compile_function[
        matmul_kernel_naive[type, type, type, BLOCK_DIM, True]
    ]()

    ctx.enqueue_function(
        func_naive,
        c_device_ref,
        a_device,
        b_device,
        M,
        N,
        K,
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
        block_dim=(BLOCK_DIM, BLOCK_DIM),
    )

    ctx.enqueue_copy_from_device(c_host_ref.data, c_device_ref)
    ctx.synchronize()

    for i in range(M):
        for j in range(N):
            assert_true(isclose(c_host_ref[i, j], c_host[i, j]))

    _ = a_device
    _ = b_device
    _ = c_device
    _ = c_device_ref

    a_host_mem.free()
    b_host_mem.free()
    c_host_mem.free()
    c_host_ref_mem.free()

    _ = func_naive^


fn run_matmul_from_mogg_interface[
    M: Int, K: Int, N: Int, type: DType
](ctx: DeviceContext) raises:
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

    var a_device = ctx.create_buffer[type](M * K)
    var b_device = ctx.create_buffer[type](K * N)
    var c_device = ctx.create_buffer[type](M * N)
    var c_device_ref = ctx.create_buffer[type](M * N)

    alias a_shape = DimList(M, K)
    var a_device_nd = NDBuffer[type, 2, a_shape](a_device.ptr, Index(M, K))
    alias b_shape = DimList(K, N)
    var b_device_nd = NDBuffer[type, 2, b_shape](b_device.ptr, Index(K, N))
    alias c_shape = DimList(M, N)
    var c_device_nd = NDBuffer[type, 2, c_shape](c_device.ptr, Index(M, N))
    ctx.enqueue_copy_to_device(a_device, a_host.data)
    ctx.enqueue_copy_to_device(b_device, b_host.data)
    ctx.enqueue_copy_to_device(c_device, c_host.data)
    ctx.enqueue_copy_to_device(c_device_ref, c_host_ref.data)

    _matmul_gpu[use_tensor_core=False](
        c_device_nd, a_device_nd, b_device_nd, ctx
    )
    ctx.enqueue_copy_from_device(c_host.data, c_device)

    alias BLOCK_DIM = 16
    var func_naive = ctx.compile_function[
        matmul_kernel_naive[type, type, type, BLOCK_DIM]
    ]()

    ctx.enqueue_function(
        func_naive,
        c_device_ref,
        a_device,
        b_device,
        M,
        N,
        K,
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
        block_dim=(BLOCK_DIM, BLOCK_DIM),
    )

    ctx.enqueue_copy_from_device(c_host_ref.data, c_device_ref)
    ctx.synchronize()

    for i in range(M):
        for j in range(N):
            assert_true(isclose(c_host_ref[i, j], c_host[i, j]))

    _ = a_device
    _ = b_device
    _ = c_device
    _ = c_device_ref

    a_host_mem.free()
    b_host_mem.free()
    c_host_mem.free()
    c_host_ref_mem.free()

    _ = func_naive^


fn run_matmul_from_mogg_interface_with_epilogue[
    M: Int, K: Int, N: Int, type: DType
](ctx: DeviceContext) raises:
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

    var a_device = ctx.create_buffer[type](M * K)
    var b_device = ctx.create_buffer[type](K * N)
    var c_device = ctx.create_buffer[type](M * N)
    var c_device_ref = ctx.create_buffer[type](M * N)

    alias a_shape = DimList(M, K)
    var a_device_nd = NDBuffer[type, 2, a_shape](a_device.ptr, Index(M, K))
    alias b_shape = DimList(K, N)
    var b_device_nd = NDBuffer[type, 2, b_shape](b_device.ptr, Index(K, N))
    alias c_shape = DimList(M, N)
    var c_device_nd = NDBuffer[type, 2, c_shape](c_device.ptr, Index(M, N))
    var c_device_ref_nd = NDBuffer[type, 2, c_shape](
        c_device_ref.ptr, Index(M, N)
    )
    ctx.enqueue_copy_to_device(a_device, a_host.data)
    ctx.enqueue_copy_to_device(b_device, b_host.data)
    ctx.enqueue_copy_to_device(c_device, c_host.data)
    ctx.enqueue_copy_to_device(c_device_ref, c_host_ref.data)

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
    ](c_device_nd, a_device_nd, b_device_nd, ctx)
    ctx.enqueue_copy_from_device(c_host.data, c_device)

    alias BLOCK_DIM = 16
    var func_naive = ctx.compile_function[
        matmul_kernel_naive[
            type,
            type,
            type,
            BLOCK_DIM,
            elementwise_lambda_fn=naive_epilogue_fn,
        ]
    ]()

    ctx.enqueue_function(
        func_naive,
        c_device_ref,
        a_device,
        b_device,
        M,
        N,
        K,
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
        block_dim=(BLOCK_DIM, BLOCK_DIM),
    )

    ctx.enqueue_copy_from_device(c_host_ref.data, c_device_ref)
    ctx.synchronize()

    for i in range(M):
        for j in range(N):
            assert_true(isclose(c_host_ref[i, j], c_host[i, j]))

    _ = a_device
    _ = b_device
    _ = c_device
    _ = c_device_ref

    a_host_mem.free()
    b_host_mem.free()
    c_host_mem.free()
    c_host_ref_mem.free()

    _ = func_naive^


fn run_low_precision_test[
    M: Int,
    N: Int,
    K: Int,
    type: DType,
    accum_type: DType,
    transpose_b: Bool = False,
](ctx: DeviceContext) raises:
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

    var a_device = ctx.create_buffer[type](M * K)
    var b_device = ctx.create_buffer[type](K * N)
    var c_device = ctx.create_buffer[accum_type](M * N)
    var c_device_ref = ctx.create_buffer[accum_type](M * N)

    ctx.enqueue_copy_to_device(a_device, a_host)
    ctx.enqueue_copy_to_device(b_device, b_trans_host)

    var a_buffer = NDBuffer[type, 2, DimList(M, K)](a_device.ptr)
    var b_buffer = NDBuffer[type, 2, DimList(K, N)](b_device.ptr)
    var c_buffer = NDBuffer[accum_type, 2, DimList(M, N)](c_device.ptr)

    _matmul_gpu[
        use_tensor_core=True,
        transpose_b=False,
    ](c_buffer, a_buffer, b_buffer, ctx)

    ctx.enqueue_copy_from_device(c_host, c_device)
    ctx.enqueue_copy_to_device(b_device, b_host)

    # Naive gemm.
    alias BLOCK_DIM = 16
    alias gemm_naive = matmul_kernel_naive[accum_type, type, type, BLOCK_DIM]
    var func_naive = ctx.compile_function[gemm_naive](threads_per_block=256)
    var c_buffer_ref = NDBuffer[accum_type, 2, DimList(M, N)](c_device_ref.ptr)
    ctx.enqueue_function(
        func_naive,
        c_buffer_ref,
        a_buffer,
        b_buffer,
        M,
        N,
        K,
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM), 1),
        block_dim=(BLOCK_DIM, BLOCK_DIM, 1),
    )

    ctx.enqueue_copy_from_device(c_host_ref, c_device_ref)
    ctx.synchronize()

    for i in range(M * N):
        if not isclose(c_host[i], c_host_ref[i], rtol=0.01):
            print(i, c_host[i], c_host_ref[i])
        assert_almost_equal(c_host[i], c_host_ref[i], rtol=0.01)

    _ = c_device
    _ = c_device_ref
    _ = a_device
    _ = b_device

    c_host.free()
    c_host_ref.free()
    a_host.free()
    b_host.free()
    b_trans_host.free()

    _ = func_naive^


fn run_low_precision_test_with_epilogue[
    M: Int,
    N: Int,
    K: Int,
    type: DType,
    accum_type: DType,
    transpose_b: Bool = False,
](ctx: DeviceContext) raises:
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

    var a_device = ctx.create_buffer[type](M * K)
    var b_device = ctx.create_buffer[type](K * N)
    var c_device = ctx.create_buffer[accum_type](M * N)
    var c_device_ref = ctx.create_buffer[accum_type](M * N)

    ctx.enqueue_copy_to_device(a_device, a_host)
    ctx.enqueue_copy_to_device(b_device, b_trans_host)

    var a_buffer = NDBuffer[type, 2, DimList(M, K)](a_device.ptr)
    var b_buffer = NDBuffer[type, 2, DimList(K, N)](b_device.ptr)
    var c_buffer = NDBuffer[accum_type, 2, DimList(M, N)](c_device.ptr)
    var c_buffer_ref = NDBuffer[accum_type, 2, DimList(M, N)](c_device_ref.ptr)

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
    ](c_buffer, a_buffer, b_buffer, ctx)

    ctx.enqueue_copy_from_device(c_host, c_device)
    ctx.enqueue_copy_to_device(b_device, b_host)

    # Naive gemm.
    alias BLOCK_DIM = 16
    alias gemm_naive = matmul_kernel_naive[
        accum_type,
        type,
        type,
        BLOCK_DIM,
        elementwise_lambda_fn=naive_epilogue_fn,
    ]
    var func_naive = ctx.compile_function[gemm_naive](threads_per_block=256)
    ctx.enqueue_function(
        func_naive,
        c_buffer_ref,
        a_buffer,
        b_buffer,
        M,
        N,
        K,
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM), 1),
        block_dim=(BLOCK_DIM, BLOCK_DIM, 1),
    )

    ctx.enqueue_copy_from_device(c_host_ref, c_device_ref)
    ctx.synchronize()

    for i in range(M * N):
        if not isclose(c_host[i], c_host_ref[i], rtol=0.01):
            print(i, c_host[i], c_host_ref[i])
        assert_almost_equal(c_host[i], c_host_ref[i], rtol=0.01)

    _ = c_device
    _ = c_device_ref
    _ = a_device
    _ = b_device

    c_host.free()
    c_host_ref.free()
    a_host.free()
    b_host.free()
    b_trans_host.free()

    _ = func_naive^


# CHECK-NOT: CUDA_ERROR
def main():
    try:
        var ctx = DeviceContext()

        run_matmul(ctx)
        test_gemm_transpose_b[DType.float32, 512, 512, 512](ctx)
        test_gemm_transpose_b[DType.float32, 512, 1024, 3072](ctx)
        run_matmul_from_mogg_interface[1024, 3072, 5120, DType.float32](ctx)
        run_matmul_from_mogg_interface[1024, 12288, 3072, DType.float32](ctx)
        run_matmul_from_mogg_interface_with_epilogue[
            1024, 3072, 5120, DType.float32
        ](ctx)
        run_matmul_from_mogg_interface_with_epilogue[
            1024, 3072, 5120, DType.bfloat16
        ](ctx)
        run_low_precision_test[1024, 3072, 5120, DType.float32, DType.float32](
            ctx
        )
        run_low_precision_test_with_epilogue[
            1024, 3072, 5120, DType.float32, DType.float32
        ](ctx)
        run_low_precision_test[
            1024, 3072, 5120, DType.bfloat16, DType.bfloat16
        ](ctx)
        run_low_precision_test_with_epilogue[
            1024, 3072, 5120, DType.bfloat16, DType.bfloat16
        ](ctx)

        _ = ctx
    except e:
        print("CUDA_ERROR:", e)
