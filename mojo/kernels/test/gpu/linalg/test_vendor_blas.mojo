# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# TODO(#31429): Restore `--debug-level full` here
# REQUIRES: NVIDIA-GPU
# UNSUPPORTED: asan
# RUN: %mojo-no-debug %s

from math import ceildiv
from random import random_float64

import linalg.vendor_blas
from buffer import DimList, NDBuffer
from gpu import block_dim, block_idx, thread_idx
from gpu.host import DeviceContext
from linalg.matmul_gpu import matmul_kernel_naive
from memory import UnsafePointer
from testing import assert_almost_equal, assert_equal

from utils.index import Index


def test_vendor_blas[
    type: DType
](*, M: Int, N: Int, K: Int, ctx: DeviceContext):
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

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    var a = NDBuffer[type, 2](a_device._unsafe_ptr(), (M, K))
    var b = NDBuffer[type, 2](b_device._unsafe_ptr(), (K, N))
    var c = NDBuffer[type, 2](c_device._unsafe_ptr(), (M, N))
    var c_ref = NDBuffer[type, 2](c_device_ref._unsafe_ptr(), (M, N))

    vendor_blas.matmul(ctx, c, a, b, c_row_major=True)

    ctx.enqueue_copy(c_host, c_device)

    alias BLOCK_DIM = 16
    ctx.enqueue_function[matmul_kernel_naive[type, type, type, BLOCK_DIM]](
        c_ref,
        a,
        b,
        M,
        N,
        K,
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM), 1),
        block_dim=(BLOCK_DIM, BLOCK_DIM, 1),
    )

    ctx.enqueue_copy(c_host_ref, c_device_ref)

    ctx.synchronize()

    for i in range(M * N):
        assert_almost_equal(
            c_host[i],
            c_host_ref[i],
            atol=1e-2 if type.is_half_float() else 1e-3,
            rtol=1e-2 if type.is_half_float() else 1e-3,
        )

    _ = a_device
    _ = b_device
    _ = c_device
    _ = c_device_ref

    a_host.free()
    b_host.free()
    c_host.free()
    c_host_ref.free()


def dispatch_test_vendor_blas(*, M: Int, N: Int, K: Int, ctx: DeviceContext):
    test_vendor_blas[type = DType.bfloat16](M=M, N=N, K=K, ctx=ctx)
    test_vendor_blas[type = DType.float32](M=M, N=N, K=K, ctx=ctx)


def main():
    with DeviceContext() as ctx:
        dispatch_test_vendor_blas(M=63, N=65, K=66, ctx=ctx)
        dispatch_test_vendor_blas(M=7, N=6144, K=4096, ctx=ctx)
        dispatch_test_vendor_blas(M=1024, N=1024, K=1024, ctx=ctx)
        dispatch_test_vendor_blas(M=1, N=1024, K=1024, ctx=ctx)
