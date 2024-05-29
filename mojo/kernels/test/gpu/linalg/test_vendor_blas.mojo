# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# TODO(#31429): Restore `--debug-level full` here
# RUN: %mojo %s

from buffer import NDBuffer, DimList
from gpu import BlockDim, BlockIdx, ThreadIdx
from gpu.host import Context, Function, Stream, synchronize
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
)
from gpu.cublas.cublas import *
from math import ceildiv, isclose
from memory.unsafe import DTypePointer
from random import random_float64
from testing import assert_almost_equal
from utils.index import Index
from LinAlg.MatmulGPU import matmul_kernel_naive
from LinAlg.MatmulCublas import cublas_matmul


fn test_cublas() raises:
    print("== test_cublas")

    alias M = 63
    alias N = 65
    alias K = 66
    alias type = DType.float32

    var stream = Stream()

    var a_host = DTypePointer[type].alloc(M * K)
    var b_host = DTypePointer[type].alloc(K * N)
    var c_host = DTypePointer[type].alloc(M * N)
    var c_host_ref = DTypePointer[type].alloc(M * N)

    for m in range(M):
        for k in range(K):
            a_host[m * K + k] = random_float64(-10, 10)

    for k in range(K):
        for n in range(N):
            b_host[k * N + n] = random_float64(-10, 10)

    var a_device = _malloc[type](M * K)
    var b_device = _malloc[type](K * N)
    var c_device = _malloc[type](M * N)
    var c_device_ref = _malloc[type](M * N)

    _copy_host_to_device(a_device, a_host, M * K)
    _copy_host_to_device(b_device, b_host, K * N)

    synchronize()

    var a = NDBuffer[type, 2, DimList(M, K)](a_device)
    var b = NDBuffer[type, 2, DimList(K, N)](b_device)
    var c = NDBuffer[type, 2, DimList(M, N)](c_device)
    var c_ref = NDBuffer[type, 2, DimList(M, N)](c_device_ref)

    var handle = Pointer[cublasContext]()
    check_cublas_error(cublasCreate(Pointer.address_of(handle)))
    check_cublas_error(cublas_matmul(handle, c, a, b, c_row_major=True))
    check_cublas_error(cublasDestroy(handle))

    _copy_device_to_host(c_host, c_device, M * N)

    alias BLOCK_DIM = 16
    alias gemm_naive = matmul_kernel_naive[type, type, type, BLOCK_DIM]
    var func_naive = Function[gemm_naive](threads_per_block=256)
    func_naive(
        c_ref,
        a,
        b,
        M,
        N,
        K,
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM), 1),
        block_dim=(BLOCK_DIM, BLOCK_DIM, 1),
    )

    _copy_device_to_host(c_host_ref, c_device_ref, M * N)

    for i in range(M * N):
        assert_almost_equal(c_host[i], c_host_ref[i], atol=1e-4, rtol=1e-4)

    _free(a_device)
    _free(b_device)
    _free(c_device)
    _free(c_device_ref)

    a_host.free()
    b_host.free()
    c_host.free()
    c_host_ref.free()

    _ = stream^


# CHECK-NOT: CUDA_ERROR
def main():
    try:
        with Context() as ctx:
            test_cublas()
    except e:
        print("CUDA_ERROR:", e)
