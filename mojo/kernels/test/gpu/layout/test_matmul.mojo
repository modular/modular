# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s

from math import ceildiv
from collections.optional import Optional
from utils.numerics import FlushDenormals
from gpu import BlockDim, BlockIdx, ThreadIdx, barrier
from gpu.host.memory import _memset, _memset_async
from gpu.host._compile import _get_nvptx_target
from algorithm.functional import _elementwise_impl_gpu
from gpu.host.device_context import DeviceContext, DeviceBuffer
from linalg.matmul_gpu import _matmul_gpu, matmul_kernel_naive
from memory import memset_zero, stack_allocation
from memory.reference import _GPUAddressSpace as GPUAddressSpace
from gpu.cublas.cublas import (
    check_cublas_error,
    cublasContext,
    cublasCreate,
    cublasDestroy,
)
from buffer.dimlist import DimList

import time
from linalg.cublas import cublas_matmul
from utils import StaticIntTuple
from utils.index import Index
from internal_utils import (
    HostNDBuffer,
    DeviceNDBuffer,
    fill,
    zero,
    linspace,
    random,
    assert_equal,
    assert_almost_equal,
)
from layout.int_tuple import IntTuple, UNKNOWN_VALUE

from layout.layout_tensor import (
    LayoutTensor,
    Layout,
    RuntimeTuple,
    RuntimeLayout,
)


from buffer.dimlist import _make_tuple
from testing import assert_equal as assert_equal_val

from matmul_kernels import (
    run_gemm_kernel_1,
    run_gemm_kernel_2,
    run_gemm_kernel_3,
    run_gemm_kernel_4,
    run_gemm_kernel_5,
    run_cublas,
)

from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure

alias run_gemm_kernel_type = fn (
    inout m: Bench,
    ctx: DeviceContext,
    a: LayoutTensor,
    b: LayoutTensor,
    c: LayoutTensor,
) raises -> None


struct test_matmul[
    dtype: DType, a_layout: Layout, b_layout: Layout, c_layout: Layout
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

    fn __init__(inout self, inout m: Bench, ctx: DeviceContext) raises:
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

        self.a_device_buffer = ctx.create_buffer[dtype](a_shape.product().get())
        self.b_device_buffer = ctx.create_buffer[dtype](b_shape.product().get())
        self.c_device_buffer = ctx.create_buffer[dtype](c_shape.product().get())
        self.c_device_buffer_ref = ctx.create_buffer[dtype](
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

        _memset_async(
            self.c_device_buffer_ref.ptr, 0, self.M * self.N, ctx.cuda_stream
        )

        run_cublas[dtype](
            m,
            ctx,
            self.M,
            self.N,
            self.K,
            self.a_device_buffer.ptr,
            self.b_device_buffer.ptr,
            self.c_device_buffer_ref.ptr,
        )

        ctx.enqueue_copy_from_device(
            self.c_host_ref.tensor.data, self.c_device_buffer_ref
        )

    fn run_test[gemm: run_gemm_kernel_type](self, inout m: Bench) raises:
        print("=== test_matmul")

        var ctx = self.ctx

        _memset_async(
            self.c_device_buffer.ptr, 0, self.M * self.N, ctx.cuda_stream
        )

        fn create_tensor[
            layout: Layout
        ](m: Int, n: Int, ptr: UnsafePointer[Scalar[dtype]]) -> LayoutTensor[
            dtype, layout
        ]:
            var dynamic_layout = RuntimeLayout[layout](
                RuntimeTuple[layout.shape](m, n),
                RuntimeTuple[layout.stride](n, 1),
            )
            return LayoutTensor[dtype, layout](ptr, dynamic_layout)

        var a = create_tensor[a_layout](
            self.M, self.K, self.a_device_buffer.ptr
        )
        var b = create_tensor[b_layout](
            self.K, self.N, self.b_device_buffer.ptr
        )
        var c = create_tensor[c_layout](
            self.M, self.N, self.c_device_buffer.ptr
        )

        gemm(m, ctx, a, b, c)

        ctx.enqueue_copy_from_device(
            self.c_host.tensor.data, self.c_device_buffer
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

        var test = test_matmul[DType.float32, a_layout, b_layout, c_layout](
            m, ctx
        )

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

        test.run_test[k1](m)
        test.run_test[k2](m)
        test.run_test[k3](m)
        test.run_test[k4](m)
        test.run_test[k5](m)

    m.dump_report()
