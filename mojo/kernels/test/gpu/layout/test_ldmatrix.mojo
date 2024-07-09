# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s | FileCheck %s

from math import ceildiv
from random import random_si64

from gpu import WARP_SIZE, ThreadIdx, barrier, lane_id
from gpu.host import Context, Function, Stream, synchronize
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
)
from gpu.memory import AddressSpace
from gpu.mma import ld_matrix, mma
from gpu.mma_util import store_matrix_d
from layout.tensor_core import get_accum_type, get_fragment_size, get_mma_shape
from linalg.matmul_gpu import matmul_kernel_naive
from memory import stack_allocation
from memory.unsafe import DTypePointer
from testing import assert_almost_equal


fn test_ldmatrix_fp32(
    c_ptr: DTypePointer[DType.float32],
    a_ptr: DTypePointer[DType.float32],
    b_ptr: DTypePointer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
):
    alias mma_m: UInt = 16
    alias mma_n: UInt = 8
    alias mma_k: UInt = 8

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

    # Transpose B to fit ld_matrix layout.
    for i in range(tid, mma_k * mma_n, WARP_SIZE):
        var x = i % mma_n
        var y = i // mma_n
        b_shared[x * mma_k + y] = b_ptr[i]

    barrier()

    var a_reg = ld_matrix[DType.float32, 4](
        a_shared + int((lane_id() % 16) * 8 + (lane_id() // 16) * 4)
    )
    var b_reg = ld_matrix[DType.float32, 2](
        b_shared + int((lane_id() % 8) * 8 + (lane_id() // 8) * 4)
    )

    mma(d_reg, a_reg, b_reg, d_reg)
    store_matrix_d[DType.float32, mma_m, mma_n, mma_k](c_ptr, d_reg, 0, 0, n)


fn test_ldmatrix_transposed[
    input_type: DType, output_type: DType
](
    c_ptr: DTypePointer[output_type],
    a_ptr: DTypePointer[input_type],
    b_ptr: DTypePointer[input_type],
):
    alias accum_type = get_accum_type[input_type]()
    alias mma_shape = get_mma_shape[input_type, accum_type]()
    alias M = mma_shape[0]
    alias N = mma_shape[1]
    alias K = mma_shape[2]
    alias frag_size = get_fragment_size[mma_shape]()
    alias a_frag_size = frag_size[0]
    alias b_frag_size = frag_size[1]
    alias c_frag_size = frag_size[2]

    var lane = lane_id()
    var d = SIMD[accum_type, c_frag_size](0)

    var a_shared = stack_allocation[
        M * K, input_type, alignment=32, address_space = AddressSpace.SHARED
    ]()
    var b_shared = stack_allocation[
        N * K, input_type, alignment=32, address_space = AddressSpace.SHARED
    ]()

    for i in range(lane, M * K, WARP_SIZE):
        a_shared[i] = a_ptr[i]

    # Transpose B to fit ld_matrix layout.
    for i in range(lane, N * K, WARP_SIZE):
        b_shared[i] = b_ptr[i]

    barrier()

    var a_reg = ld_matrix[input_type, a_frag_size](
        a_shared + int((lane % M) * K + (lane // M) * K // 2)
    )
    var b_reg = ld_matrix[input_type, b_frag_size, transpose=True](
        b_shared + int((lane % K) * N + (lane // K) * N // 2)
    )

    mma(d, a_reg, b_reg, d)
    store_matrix_d[output_type, M, N, K](
        c_ptr,
        # Store matrix is hardcoded to store 4 elements.
        rebind[SIMD[output_type, 4]](d.cast[output_type]()),
        0,
        0,
        mma_shape[1],
    )


fn check_ldmatrix_transposed_bf16[
    input_type: DType,
    output_type: DType,
]() raises:
    print("== test ldmatrix transposed bf16")

    # Shape for a single mma.
    alias accum_type = get_accum_type[input_type]()
    alias mma_shape = get_mma_shape[input_type, accum_type]()
    alias M = mma_shape[0]
    alias N = mma_shape[1]
    alias K = mma_shape[2]

    var stream = Stream()
    var a_host = DTypePointer[input_type].alloc(M * K)
    var b_host = DTypePointer[input_type].alloc(K * N)
    var c_host = DTypePointer[output_type].alloc(M * N)
    var c_host_ref = DTypePointer[output_type].alloc(M * N)

    for m in range(M):
        for k in range(K):
            a_host[m * K + k] = m * K + k

    for k in range(K):
        for n in range(N):
            b_host[k * N + n] = k * N + n

    for i in range(M * N):
        c_host[i] = 0
        c_host_ref[i] = 0

    var a_device = _malloc[input_type](M * K)
    var b_device = _malloc[input_type](K * N)
    var c_device = _malloc[output_type](M * N)
    var c_device_ref = _malloc[output_type](M * N)

    _copy_host_to_device(a_device, a_host, M * K)
    _copy_host_to_device(b_device, b_host, K * N)

    var func = Function[test_ldmatrix_transposed[input_type, output_type]](
        dump_ptx=False
    )

    func(
        c_device,
        a_device,
        b_device,
        grid_dim=1,
        block_dim=WARP_SIZE,
        stream=stream,
    )

    _copy_device_to_host(c_host, c_device, M * N)

    # Run naive matmul.
    alias BLOCK_DIM = 16
    var func_naive = Function[
        matmul_kernel_naive[output_type, input_type, input_type, BLOCK_DIM]
    ]()
    func_naive(
        c_device_ref,
        a_device,
        b_device,
        M,
        N,
        K,
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM), 1),
        block_dim=(BLOCK_DIM, BLOCK_DIM, 1),
    )

    _copy_device_to_host(c_host_ref, c_device_ref, M * N)

    for i in range(M * N):
        var out_val = Scalar.load(c_host, i)
        var out_ref = Scalar.load(c_host_ref, i)
        assert_almost_equal(out_val, out_ref)

    _free(a_device)
    _free(b_device)
    _free(c_device)
    _free(c_device_ref)

    _ = a_host
    _ = b_host
    _ = c_host
    _ = c_host_ref

    _ = func^
    _ = func_naive^
    _ = stream^


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
    var c_host_ref = Pointer[Float32].alloc(M * N)

    for i in range(M * K):
        var val = random_si64(rand_min, rand_max)
        a_host[i] = val.cast[DType.float32]()

    for i in range(K * N):
        var val = random_si64(rand_min, rand_max)
        b_host[i] = val.cast[DType.float32]()

    for i in range(M * N):
        c_host[i] = 0
        c_host_ref[i] = 0

    var a_device = _malloc[Float32](M * K)
    var b_device = _malloc[Float32](K * N)
    var c_device = _malloc[Float32](M * N)
    var c_device_ref = _malloc[Float32](M * N)

    _copy_host_to_device(a_device, a_host, M * K)
    _copy_host_to_device(b_device, b_host, K * N)

    var func_ldmatrix = Function[test_ldmatrix_fp32](dump_ptx=False)

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
    alias BLOCK_DIM = 16
    var func_naive = Function[
        matmul_kernel_naive[
            DType.float32, DType.float32, DType.float32, BLOCK_DIM
        ]
    ]()

    @always_inline
    @parameter
    fn run_func_naive(stream: Stream) raises:
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

    run_func_naive(stream)
    stream.synchronize()
    _copy_device_to_host(c_host_ref, c_device_ref, M * N)

    for i in range(M * N):
        var out_val = c_host[i]
        var out_ref = c_host_ref[i]
        assert_almost_equal(out_val, out_ref)

    _free(a_device)
    _free(b_device)
    _free(c_device)
    _free(c_device_ref)

    _ = a_host
    _ = b_host
    _ = c_host
    _ = c_host_ref

    _ = func_ldmatrix^
    _ = func_naive^
    _ = stream^


# CHECK-NOT: CUDA_ERROR
def main():
    try:
        with Context() as ctx:
            check_ldmatrix(16, 8, 8, -100, 100)
            check_ldmatrix_transposed_bf16[DType.bfloat16, DType.bfloat16]()

    except e:
        print("CUDA_ERROR:", e)
