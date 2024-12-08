# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug-no-assert %s
# COM: use
# mojo build --debug-level=full --mcmodel=medium --large-data-threshold=1048576
# to build this file if running into linking issues with large PTX kernels.

from collections.optional import Optional, OptionalReg
from math import ceildiv
from random import random_si64
from sys import simdwidthof, alignof

from algorithm.functional import elementwise
from buffer import NDBuffer
from buffer.dimlist import Dim, DimList, _make_tuple
from gpu import BlockDim, BlockIdx, ThreadIdx, barrier

# from gpu.cublas.cublas import (
#    check_cublas_error,
#    cublasContext,
#    cublasCreate,
#    cublasDestroy,
# )
from gpu.host._compile import _get_gpu_target
from gpu.host import DeviceBuffer, DeviceContext
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    assert_with_measure,
    fill,
    linspace,
    random,
    zero,
)
from internal_utils._utils import ValOrDim, dynamic, static

# from linalg.cublas import cublas_matmul
from linalg.matmul_gpu import _matmul_gpu, matmul_kernel_naive
from linalg.utils import elementwise_epilogue_type
from linalg.utils_gpu import MatmulConfig, MatmulKernels
from memory import UnsafePointer, memset_zero, stack_allocation
from memory.pointer import _GPUAddressSpace as GPUAddressSpace

from utils import IndexList
from utils.index import Index

from builtin._location import __source_location
from internal_utils._measure import cosine

from math import exp2
from utils.numerics import FPUtils
from testing import assert_equal
from gpu.host.info import DEFAULT_GPU_ARCH

alias init_fn_type = fn (buff: NDBuffer) capturing -> None

alias epilogue_func_type = fn[type: DType, width: Int, *, alignment: Int = 1] (
    IndexList[2], IndexList[2], SIMD[type, width]
) capturing -> SIMD[type, width]


fn matmul_naive[
    a_type: DType, b_type: DType, c_type: DType, transpose_b: Bool
](
    a: UnsafePointer[Scalar[a_type]],
    b: UnsafePointer[Scalar[b_type]],
    c: UnsafePointer[Scalar[c_type]],
    m: Int,
    n: Int,
    k: Int,
):
    @parameter
    if transpose_b:
        for i in range(m):
            for l in range(k):
                for j in range(n):
                    var av = a[k * i + l].cast[c_type]()
                    var bv = b[k * j + l].cast[c_type]()
                    c[n * i + j] += av * bv
    else:
        for i in range(m):
            for l in range(k):
                for j in range(n):
                    var av = a[k * i + l].cast[c_type]()
                    var bv = b[n * l + j].cast[c_type]()
                    c[n * i + j] += av * bv


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
](ctx: DeviceContext, m: ValOrDim, n: ValOrDim, k: ValOrDim,) raises:
    constrained[
        int(n.dim) > 0 and int(k.dim) > 0,
        "This test currently requires static N and K.",
    ]()

    var M = m.value
    var N = n.value
    var K = k.value
    print(M, "x", N, "x", K)

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
        val = 1
        a_host.tensor.data[i] = val.cast[in_type]()

    for i in range(K * N):
        var val = random_si64(rand_min, rand_max)
        val = 1
        b_host.tensor.data[i] = val.cast[in_type]()

    for i in range(M * N):
        c_host.tensor.data[i] = 0
        c_host_ref.tensor.data[i] = 0

    # Move operands to the Device

    ctx.enqueue_copy_to_device(a_device.buffer, a_host.tensor.data)
    ctx.enqueue_copy_to_device(b_device.buffer, b_host.tensor.data)
    ctx.enqueue_copy_to_device(c_device.buffer, c_host.tensor.data)

    _matmul_gpu[use_tensor_core=True, transpose_b=transpose_b,](
        c_device.tensor,
        a_device.tensor,
        b_device.tensor,
        ctx,
    )

    ctx.synchronize()

    ctx.enqueue_copy_from_device(c_host.tensor.data, c_device.buffer)

    matmul_naive[transpose_b=transpose_b](
        a_host.tensor.data, b_host.tensor.data, c_host_ref.tensor.data, M, N, K
    )

    var errors = 0
    for i in range(M * N):
        # print(i // N, i % N, c_host.tensor.data[i], c_host_ref.tensor.data[i])
        if c_host.tensor.data[i] != c_host_ref.tensor.data[i]:
            # print(i//N, i%N, c_host.tensor.data[i], c_host_ref.tensor.data[i])
            errors += 1

    print("errors", errors)

    _ = c_device
    _ = c_device_ref
    _ = a_host
    _ = b_host
    _ = c_host_ref
    _ = c_host
    _ = a_device
    _ = b_device


def main():
    with DeviceContext() as ctx:
        test[
            in_type = DType.bfloat16,
            out_type = DType.float32,
            transpose_b=False,
        ](ctx, dynamic(256), static[256](), static[128]())
        test[
            in_type = DType.bfloat16,
            out_type = DType.float32,
            transpose_b=True,
        ](ctx, dynamic(256), static[256](), static[128]())
        test[
            in_type = DType.bfloat16,
            out_type = DType.bfloat16,
            transpose_b=False,
        ](ctx, dynamic(256), static[256](), static[128]())
        test[
            in_type = DType.bfloat16,
            out_type = DType.bfloat16,
            transpose_b=True,
        ](ctx, dynamic(256), static[256](), static[128]())

        test[
            in_type = DType.bfloat16,
            out_type = DType.bfloat16,
            transpose_b=False,
        ](ctx, dynamic(1024), static[256](), static[128]())

        test[
            in_type = DType.bfloat16,
            out_type = DType.bfloat16,
            transpose_b=False,
        ](ctx, dynamic(1024), static[256](), static[256]())

        test[
            in_type = DType.bfloat16,
            out_type = DType.float32,
            transpose_b=True,
        ](ctx, dynamic(1024), static[256](), static[1024]())
