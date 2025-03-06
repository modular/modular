# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug-no-assert %s
# COM: use
# mojo build --debug-level=full --mcmodel=medium --large-data-threshold=1048576
# to build this file if running into linking issues with large PTX kernels.

from math import ceildiv
from random import random_si64

import linalg.vendor_blas
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from gpu.host import DeviceContext
from internal_utils import DeviceNDBuffer, HostNDBuffer
from internal_utils._utils import ValOrDim, dynamic, static
from linalg.matmul_gpu import _matmul_gpu
from memory import UnsafePointer

from utils import IndexList
from utils.index import Index


alias epilogue_func_type = fn[type: DType, width: Int, *, alignment: Int = 1] (
    IndexList[2], IndexList[2], SIMD[type, width]
) capturing -> SIMD[type, width]


@parameter
@always_inline
fn epilogue_test_fn[
    type: DType, width: Int, *, alignment: Int = 1
](
    idx: IndexList[2],
    dim_space: IndexList[2],
    val: SIMD[type, width],
) -> SIMD[
    type, width
]:
    var bias = SIMD[type, width](0)

    @parameter
    for i in range(width):
        bias[i] = (
            0.5
            + ((idx[0] + idx[1] + i) / (dim_space[0] + dim_space[1])).cast[
                type
            ]()
        )

    return val + bias


fn test[
    in_type: DType,
    out_type: DType,
    transpose_b: Bool,
](
    mut bench: Bench,
    ctx: DeviceContext,
    m: ValOrDim,
    n: ValOrDim,
    k: ValOrDim,
) raises:
    constrained[
        Int(n.dim) > 0 and Int(k.dim) > 0,
        "This test currently requires static N and K.",
    ]()

    var M = m.value
    var N = n.value
    var K = k.value
    print(M, "x", N, "x", K, "transpose_b", transpose_b)

    alias static_a_shape = DimList(m.dim, k.dim)
    alias static_b_shape = DimList(n.dim, k.dim) if transpose_b else DimList(
        k.dim, n.dim
    )
    alias static_c_shape = DimList(m.dim, n.dim)

    var dynamic_a_shape = DimList(m.value, k.value)
    var dynamic_b_shape = DimList(n.value, k.value) if transpose_b else DimList(
        k.value, n.value
    )

    var dynamic_c_shape = DimList(m.value, n.value)

    var a_host = HostNDBuffer[in_type, 2, static_a_shape](dynamic_a_shape)
    var b_host = HostNDBuffer[in_type, 2, static_b_shape](dynamic_b_shape)
    var c_host = HostNDBuffer[out_type, 2, static_c_shape](dynamic_c_shape)
    var c_host_ref = HostNDBuffer[out_type, 2, static_c_shape](dynamic_c_shape)

    var a_device = DeviceNDBuffer[in_type, 2, static_a_shape](
        dynamic_a_shape, ctx=ctx
    )
    var b_device = DeviceNDBuffer[in_type, 2, static_b_shape](
        dynamic_b_shape, ctx=ctx
    )
    var c_device = DeviceNDBuffer[out_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )
    var c_device_ref = DeviceNDBuffer[out_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )

    alias rand_min = -100
    alias rand_max = 100

    for i in range(M * K):
        var val = random_si64(rand_min, rand_max)
        a_host.tensor.data[i] = val.cast[in_type]()

    for i in range(K * N):
        var val = random_si64(rand_min, rand_max)
        b_host.tensor.data[i] = val.cast[in_type]()

    for i in range(M * N):
        c_host.tensor.data[i] = 0
        c_host_ref.tensor.data[i] = 0

    # Move operands to the Device

    ctx.enqueue_copy(a_device.buffer, a_host.tensor.data)
    ctx.enqueue_copy(b_device.buffer, b_host.tensor.data)
    ctx.enqueue_copy(c_device.buffer, c_host.tensor.data)

    _matmul_gpu[use_tensor_core=True, transpose_b=transpose_b](
        c_device.tensor,
        a_device.tensor,
        b_device.tensor,
        ctx,
    )

    ctx.synchronize()

    ctx.enqueue_copy(c_host.tensor.data, c_device.buffer)

    var handle = vendor_blas.Handle()

    vendor_blas.matmul(
        ctx,
        handle,
        c_device_ref.tensor,
        a_device.tensor,
        b_device.tensor,
        c_row_major=True,
        transpose_b=transpose_b,
    )

    ctx.enqueue_copy(c_host_ref.tensor.data, c_device_ref.buffer)

    ctx.synchronize()
    var errors = 0
    for i in range(M * N):
        # print(i // N, i % N, c_host.tensor.data[i], c_host_ref.tensor.data[i])
        if c_host.tensor.data[i] != c_host_ref.tensor.data[i]:
            # print(i//N, i%N, c_host.tensor.data[i], c_host_ref.tensor.data[i])
            errors += 1

    print("errors", errors)

    @parameter
    fn bench_func(mut m: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            _matmul_gpu[use_tensor_core=True, transpose_b=transpose_b](
                c_device.tensor,
                a_device.tensor,
                b_device.tensor,
                ctx,
            )

        m.iter_custom[kernel_launch](ctx)

    bench.bench_function[bench_func](
        BenchId("mojo matmul"),
        ThroughputMeasure(BenchMetric.elements, 2 * M * N * K),
    )

    @parameter
    fn bench_func_vendor_blas(mut m: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            vendor_blas.matmul(
                ctx,
                handle,
                c_device_ref.tensor,
                a_device.tensor,
                b_device.tensor,
                c_row_major=True,
                transpose_b=transpose_b,
            )

        m.iter_custom[kernel_launch](ctx)

    bench.bench_function[bench_func_vendor_blas](
        BenchId("vendor_blas matmul"),
        ThroughputMeasure(BenchMetric.elements, 2 * M * N * K),
    )

    _ = c_device
    _ = c_device_ref
    _ = a_host
    _ = b_host
    _ = c_host_ref
    _ = c_host
    _ = a_device
    _ = b_device


def main():
    var bench = Bench()

    with DeviceContext() as ctx:
        # GEMV_SPLIT_K
        # M = 1, K % simd_width == 0, transpose_b = True

        test[
            in_type = DType.bfloat16,
            out_type = DType.float32,
            transpose_b=True,
        ](bench, ctx, dynamic(1), static[4096](), static[4096]())

        # GEMV_KERNEL_VECTOR

        # N = 1, K % simd_width == 0, transpose_b = False
        test[
            in_type = DType.bfloat16,
            out_type = DType.float32,
            transpose_b=False,
        ](bench, ctx, dynamic(4096), static[1](), static[4096]())

        # GEMV_KERNEL

        # M = 1, K % simd_width !=0, transpose_b = True
        test[
            in_type = DType.bfloat16,
            out_type = DType.float32,
            transpose_b=True,
        ](bench, ctx, dynamic(1), static[4096](), static[4095]())

        # N = 1, K % simd_width !=0, transpose_b = False
        test[
            in_type = DType.bfloat16,
            out_type = DType.float32,
            transpose_b=False,
        ](bench, ctx, dynamic(4096), static[1](), static[4095]())

        # gevm_tc_kernel_vector_8x with Nvidia multistage_gemm with AMDGPU
        # M = 1, K % WARP_SIZE == 0, transpose_b = False
        # Need to add tolerance
        # test[
        #    in_type = DType.bfloat16,
        #    out_type = DType.float32,
        #    transpose_b=False,
        # ](bench, ctx, dynamic(1), static[4096](), static[4096]())

        # matmaul_naive
        # M = 1, K % WARP_SIZE != 0, transpose_b = False
        test[
            in_type = DType.bfloat16,
            out_type = DType.float32,
            transpose_b=False,
        ](bench, ctx, dynamic(1), static[4096](), static[4095]())

    bench.dump_report()
