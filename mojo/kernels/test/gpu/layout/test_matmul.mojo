# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# FIXME: KERN-1377, MSTDL-1155
# UNSUPPORTED: AMD-GPU, asan
# RUN: %mojo-no-debug-no-assert %s

import time
from sys import has_nvidia_gpu_accelerator

from benchmark import Bench
from buffer.dimlist import DimList
from gpu.host import DeviceBuffer, DeviceContext
from internal_utils import (
    HostNDBuffer,
    assert_almost_equal,
    assert_equal,
    random,
    zero,
)
from layout.layout_tensor import (
    Layout,
    LayoutTensor,
    RuntimeLayout,
    RuntimeTuple,
)
from matmul_kernels import (
    run_cublas,
    run_gemm_kernel_1,
    run_gemm_kernel_2,
    run_gemm_kernel_3,
    run_gemm_kernel_4,
    run_gemm_kernel_5,
    run_gemm_kernel_6,
    run_gemm_kernel_tc,
)
from memory import UnsafePointer

alias run_gemm_kernel_type = fn (
    mut m: Bench,
    ctx: DeviceContext,
    a: LayoutTensor,
    b: LayoutTensor,
    c: LayoutTensor,
) raises -> None


struct test_matmul[
    dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    enable_tc: Bool,
]:
    var ctx: DeviceContext
    var M: Int
    var N: Int
    var K: Int

    var a_host: HostNDBuffer[dtype, 2]
    var b_host: HostNDBuffer[dtype, 2]
    var c_host: HostNDBuffer[dtype, 2]
    var c_host_ref: HostNDBuffer[dtype, 2]

    var a_device_buffer: DeviceBuffer[dtype]
    var b_device_buffer: DeviceBuffer[dtype]
    var c_device_buffer: DeviceBuffer[dtype]
    var c_device_buffer_ref: DeviceBuffer[dtype]

    fn __init__(out self, mut m: Bench, ctx: DeviceContext) raises:
        self.ctx = ctx
        self.M = a_layout.shape[0].value()
        self.N = b_layout.shape[1].value()
        self.K = b_layout.shape[0].value()
        var a_shape = DimList(self.M, self.K)
        var b_shape = DimList(self.K, self.N)
        var c_shape = DimList(self.M, self.N)

        self.a_host = HostNDBuffer[dtype, 2](a_shape)
        self.b_host = HostNDBuffer[dtype, 2](b_shape)
        self.c_host = HostNDBuffer[dtype, 2](c_shape)
        self.c_host_ref = HostNDBuffer[dtype, 2](c_shape)

        self.a_device_buffer = ctx.enqueue_create_buffer[dtype](
            a_shape.product().get()
        )
        self.b_device_buffer = ctx.enqueue_create_buffer[dtype](
            b_shape.product().get()
        )
        self.c_device_buffer = ctx.enqueue_create_buffer[dtype](
            c_shape.product().get()
        )
        self.c_device_buffer_ref = ctx.enqueue_create_buffer[dtype](
            c_shape.product().get()
        )

        random(self.a_host.tensor)
        random(self.b_host.tensor)
        zero(self.c_host.tensor)
        zero(self.c_host_ref.tensor)

        ctx.enqueue_copy_to_device(
            self.a_device_buffer, self.a_host.tensor.data
        )
        ctx.enqueue_copy_to_device(
            self.b_device_buffer, self.b_host.tensor.data
        )
        ctx.memset(self.c_device_buffer_ref, 0)

        run_cublas[dtype, enable_tc](
            m,
            ctx,
            self.M,
            self.N,
            self.K,
            self.a_device_buffer.unsafe_ptr(),
            self.b_device_buffer.unsafe_ptr(),
            self.c_device_buffer_ref.unsafe_ptr(),
        )

        ctx.enqueue_copy_from_device(
            self.c_host_ref.tensor.data, self.c_device_buffer_ref
        )

    fn run_test[gemm: run_gemm_kernel_type](self, mut m: Bench) raises:
        print("=== test_matmul")

        var ctx = self.ctx
        ctx.memset(self.c_device_buffer_ref, 0)

        fn create_tensor[
            layout: Layout
        ](m: Int, n: Int, ptr: UnsafePointer[Scalar[dtype]]) -> LayoutTensor[
            dtype, layout
        ]:
            var dynamic_layout = RuntimeLayout[layout](
                RuntimeTuple[layout.shape, unsigned=True](m, n),
                RuntimeTuple[layout.stride, unsigned=True](n, 1),
            )
            return LayoutTensor[dtype, layout](ptr, dynamic_layout)

        var a = create_tensor[a_layout](
            self.M, self.K, self.a_device_buffer.unsafe_ptr()
        )
        var b = create_tensor[b_layout](
            self.K, self.N, self.b_device_buffer.unsafe_ptr()
        )
        var c = create_tensor[c_layout](
            self.M, self.N, self.c_device_buffer.unsafe_ptr()
        )

        gemm(m, ctx, a, b, c)

        ctx.enqueue_copy_from_device(
            self.c_host.tensor.data, self.c_device_buffer
        )
        assert_almost_equal(
            self.c_host_ref.tensor,
            self.c_host.tensor,
            atol=0.0001,
            rtol=0.01,
        )


def main():
    alias N = 4096
    alias M = N
    alias K = M

    var m = Bench()
    with DeviceContext() as ctx:
        alias a_layout = Layout.row_major(M, K)
        alias b_layout = Layout.row_major(K, N)
        alias c_layout = Layout.row_major(M, N)

        var test = test_matmul[
            DType.float32, a_layout, b_layout, c_layout, False
        ](m, ctx)

        var test_tc = test_matmul[
            DType.float32, a_layout, b_layout, c_layout, True
        ](m, ctx)

        alias k1 = run_gemm_kernel_1[
            DType.float32, a_layout, b_layout, c_layout, 32, 32
        ]

        alias k2 = run_gemm_kernel_2[
            DType.float32, a_layout, b_layout, c_layout, 32, 32
        ]

        alias k3 = run_gemm_kernel_3[
            DType.float32, a_layout, b_layout, c_layout, 32, 32, 32
        ]

        alias k4 = run_gemm_kernel_4[
            DType.float32, a_layout, b_layout, c_layout, 64, 64, 8, 8
        ]

        alias k5 = run_gemm_kernel_5[
            DType.float32, a_layout, b_layout, c_layout, 128, 128, 8, 8, 8
        ]

        alias k6 = run_gemm_kernel_6[
            DType.float32, a_layout, b_layout, c_layout, 128, 128, 8, 8, 8
        ]

        alias MMA_M = 16
        alias MMA_N = 8 if has_nvidia_gpu_accelerator() else 16
        alias MMA_K = 8 if has_nvidia_gpu_accelerator() else 4

        alias k_tc = run_gemm_kernel_tc[
            DType.float32,
            a_layout,
            b_layout,
            c_layout,
            64,  # BM: The block size in the M dimension
            64,  # BN: The block size in the N dimension
            32,  # BK: The block size in the K dimension
            32,  # WM: The warp tile size in the M dimension
            32,  # WN: The warp tile size in the N dimension
            MMA_M,  # MMA_M: Tensor core instruction shape in M dimension
            MMA_N,  # MMA_N: Tensor core instruction shape in N dimension
            MMA_K,  # MMA_K: Tensor core instruction shape in K dimension
        ]

        test.run_test[k1](m)
        test.run_test[k2](m)
        test.run_test[k3](m)
        test.run_test[k4](m)
        test.run_test[k5](m)
        test.run_test[k6](m)
        test_tc.run_test[k_tc](m)

    m.dump_report()
