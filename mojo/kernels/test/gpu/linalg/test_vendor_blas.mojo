# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# TODO(#31429): Restore `--debug-level full` here
# REQUIRES: NVIDIA-GPU
# UNSUPPORTED: asan
# RUN: %mojo-no-debug-no-assert %s

from math import ceildiv
from random import random_float64

import linalg.vendor_blas
from buffer import DimList, NDBuffer
from gpu import block_dim, block_idx, thread_idx
from gpu._cublas.cublas import *
from gpu.host import DeviceContext
from linalg.matmul_gpu import matmul_kernel_naive
from memory import UnsafePointer
from testing import assert_almost_equal, assert_equal

from utils.index import Index


fn test_cublas(ctx: DeviceContext) raises:
    print("== test_cublas")

    alias M = 63
    alias N = 65
    alias K = 66
    alias type = DType.float32

    var a_host = UnsafePointer[Scalar[type]].alloc(M * K)
    var b_host = UnsafePointer[Scalar[type]].alloc(K * N)
    var c_host = UnsafePointer[Scalar[type]].alloc(M * N)
    var c_host_ref = UnsafePointer[Scalar[type]].alloc(M * N)

    for m in range(M):
        for k in range(K):
            a_host[m * K + k] = random_float64(-0.1, 0.1).cast[type]()

    for k in range(K):
        for n in range(N):
            b_host[k * N + n] = random_float64(-0.1, 0.1).cast[type]()

    var a_device = ctx.enqueue_create_buffer[type](M * K)
    var b_device = ctx.enqueue_create_buffer[type](K * N)
    var c_device = ctx.enqueue_create_buffer[type](M * N)
    var c_device_ref = ctx.enqueue_create_buffer[type](M * N)

    ctx.enqueue_copy_to_device(a_device, a_host)
    ctx.enqueue_copy_to_device(b_device, b_host)

    var a = NDBuffer[type, 2, DimList(M, K)](a_device.unsafe_ptr())
    var b = NDBuffer[type, 2, DimList(K, N)](b_device.unsafe_ptr())
    var c = NDBuffer[type, 2, DimList(M, N)](c_device.unsafe_ptr())
    var c_ref = NDBuffer[type, 2, DimList(M, N)](c_device_ref.unsafe_ptr())

    with vendor_blas.Handle() as handle:
        vendor_blas.matmul(ctx, handle, c, a, b, c_row_major=True)

    ctx.enqueue_copy_from_device(c_host, c_device)

    alias BLOCK_DIM = 16
    alias gemm_naive = matmul_kernel_naive[type, type, type, BLOCK_DIM]
    ctx.enqueue_function[gemm_naive](
        c_ref,
        a,
        b,
        M,
        N,
        K,
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM), 1),
        block_dim=(BLOCK_DIM, BLOCK_DIM, 1),
    )

    ctx.enqueue_copy_from_device(c_host_ref, c_device_ref)

    ctx.synchronize()

    for i in range(M * N):
        assert_almost_equal(c_host[i], c_host_ref[i], atol=1e-3, rtol=1e-3)

    _ = a_device
    _ = b_device
    _ = c_device
    _ = c_device_ref

    a_host.free()
    b_host.free()
    c_host.free()
    c_host_ref.free()


def test_cublas_result_format():
    assert_equal(String(Result.SUCCESS), "SUCCESS")
    assert_equal(String(Result.LICENSE_ERROR), "LICENSE_ERROR")


def main():
    test_cublas_result_format()

    with DeviceContext() as ctx:
        test_cublas(ctx)
