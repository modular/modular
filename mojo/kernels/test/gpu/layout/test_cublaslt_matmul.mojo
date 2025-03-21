# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# FIXME: KERN-1480
# UNSUPPORTED: NVIDIA-GPU
# UNSUPPORTED: AMD-GPU
# RUN: %mojo-no-debug-no-assert %s

from math import ceildiv

from buffer import DimList, NDBuffer
from gpu.host import DeviceContext
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    assert_almost_equal,
    zero,
)
from linalg.matmul_gpu import matmul_kernel_naive
from linalg.vendor_blas import Backend, Handle, matmul


fn test_cublaslt_64x16x32[input_type: DType](ctx: DeviceContext) raises:
    print("== test_cublaslt_64x16x32")

    alias M = 64
    alias N = 16
    alias K = 32
    alias transpose_b = True
    alias static_a_shape = DimList(M, K)
    alias static_b_shape = DimList(N, K) if transpose_b else DimList(K, N)
    alias static_c_shape = DimList(M, N)

    var a_host = HostNDBuffer[input_type, 2, static_a_shape]()
    var b_host = HostNDBuffer[input_type, 2, static_b_shape]()
    var c_host = HostNDBuffer[DType.float32, 2, static_c_shape]()
    var c_host_ref = HostNDBuffer[DType.float32, 2, static_c_shape]()

    @parameter
    for i in range(M):

        @parameter
        for j in range(K):
            a_host.tensor[i, j] = i

    @parameter
    for i in range(N):

        @parameter
        for j in range(K):
            b_host.tensor[i, j] = j

    zero(c_host.tensor)
    zero(c_host_ref.tensor)

    var a_device = DeviceNDBuffer[input_type, 2, static_a_shape](ctx=ctx)
    var b_device = DeviceNDBuffer[input_type, 2, static_b_shape](ctx=ctx)
    var c_device = DeviceNDBuffer[DType.float32, 2, static_c_shape](ctx=ctx)
    var c_device_ref = DeviceNDBuffer[DType.float32, 2, static_c_shape](ctx=ctx)

    ctx.enqueue_copy(a_device.buffer, a_host.tensor.data)
    ctx.enqueue_copy(b_device.buffer, b_host.tensor.data)

    with Handle[Backend.CUBLASLT]() as handle:
        matmul(
            ctx,
            handle,
            c_device.tensor,
            a_device.tensor,
            b_device.tensor,
            c_row_major=True,
        )

    ctx.enqueue_copy(c_host.tensor.data, c_device.buffer)

    # Run naive matmul.
    alias BLOCK_DIM = 16
    ctx.enqueue_function[
        matmul_kernel_naive[
            DType.float32,
            input_type,
            input_type,
            BLOCK_DIM,
            transpose_b=True,
        ]
    ](
        c_device_ref.buffer,
        a_device.buffer,
        b_device.buffer,
        M,
        N,
        K,
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM), 1),
        block_dim=(BLOCK_DIM, BLOCK_DIM, 1),
    )

    ctx.enqueue_copy(c_host_ref.tensor.data, c_device_ref.buffer)

    ctx.synchronize()

    assert_almost_equal(
        c_host.tensor,
        c_host_ref.tensor,
        atol=0.0001,
        rtol=0.01,
    )

    _ = a_device
    _ = b_device
    _ = c_device
    _ = c_device_ref

    _ = a_host
    _ = b_host
    _ = c_host
    _ = c_host_ref


fn main() raises:
    with DeviceContext() as ctx:
        test_cublaslt_64x16x32[DType.float8_e4m3fn](ctx)
