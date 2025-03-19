# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: asan
# RUN: %mojo-no-debug-no-assert %s

from random import rand

import linalg.vendor_blas
from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from gpu._cublas.cublas import check_cublas_error, cublasContext
from gpu.host import DeviceBuffer, DeviceContext, FuncAttribute
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    assert_almost_equal,
    assert_equal,
    fill,
    arange,
    random,
    zero,
)
from layout import IntTuple, Layout, LayoutTensor, RuntimeLayout, RuntimeTuple
from layout.layout import UNKNOWN_VALUE
from linalg._multistage_gemm_gpu import multistage_gemm_kernel
from linalg.utils_gpu import MatmulKernels
from memory import UnsafePointer

from utils import IndexList


# TODO: try not to copy into submatrices.
fn cublas_matrix_list[
    type: DType,
    mat_dim: DimList,
    /,
    axis: Int = 0,
](
    host: HostNDBuffer[type, 2, mat_dim],
    partition: Int,
    sub_dim: DimList,
    ctx: DeviceContext,
    out result: List[DeviceNDBuffer[type, 2]],
) raises:
    var dev_list = List[DeviceNDBuffer[type, 2]]()

    var dim = sub_dim.get[0]()
    var part = sub_dim.get[1]()

    for p in range(partition):
        var a_part_ptr = UnsafePointer[Scalar[type]].alloc(dim * part)
        var host_part = NDBuffer[type, 2](a_part_ptr, sub_dim)
        var dev_part = DeviceNDBuffer[type, 2](
            dynamic_shape=DimList(dim, part), ctx=ctx
        )
        for i in range(dim):
            for j in range(part):
                host_part[i, j] = (
                    host.tensor[i, p * part + j] if axis
                    == 1 else host.tensor[p * part + i, j]
                )
        # copy to device buffer
        ctx.enqueue_copy(dev_part.buffer, host_part.data)
        a_part_ptr.free()
        dev_list.append(dev_part)

    return dev_list^


fn test_split_k_multistage_gemm[
    type: DType,
    /,
    *,
    dim: Tuple[Int, Int, Int],
    transpose_b: Bool = False,
](ctx: DeviceContext, k_partition: Int) raises:
    alias M = dim[0]
    alias N = dim[1]
    alias K = dim[2]
    var K_part = K // k_partition

    alias a_layout = Layout(
        IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE), IntTuple(K, 1)
    )
    alias b_layout = Layout(
        IntTuple(N, UNKNOWN_VALUE), IntTuple(K, 1)
    ) if transpose_b else Layout(IntTuple(UNKNOWN_VALUE, N), IntTuple(N, 1))
    alias c_layout = Layout(IntTuple(UNKNOWN_VALUE, N), IntTuple(N, 1))

    var a_runtime_layout = RuntimeLayout(
        RuntimeTuple[a_layout.shape, unsigned=True](M, K_part),
        RuntimeTuple[a_layout.stride, unsigned=True](K, 1),
    )

    var b_runtime_layout = RuntimeLayout(
        RuntimeTuple[b_layout.shape, unsigned=True](N, K_part),
        RuntimeTuple[b_layout.stride, unsigned=True](K, 1),
    ) if transpose_b else RuntimeLayout(
        RuntimeTuple[b_layout.shape, unsigned=True](K_part, N),
        RuntimeTuple[b_layout.stride, unsigned=True](N, 1),
    )

    var c_runtime_layout = RuntimeLayout(
        RuntimeTuple[c_layout.shape, unsigned=True](M, N),
        RuntimeTuple[c_layout.stride, unsigned=True](N, 1),
    )

    alias a_dim = DimList(M, K)
    alias b_dim = DimList(N, K) if transpose_b else DimList(K, N)
    alias c_dim = DimList(M, N)

    var a_host = HostNDBuffer[type, 2, a_dim]()
    var b_host = HostNDBuffer[type, 2, b_dim]()
    var c_host = HostNDBuffer[type, 2, c_dim]()
    var c_host_ref = HostNDBuffer[type, 2, c_dim]()

    random(a_host.tensor)
    random(b_host.tensor)

    var a_submat = cublas_matrix_list[type, a_dim, axis=1](
        a_host, k_partition, DimList(M, K_part), ctx
    )

    var b_submat = cublas_matrix_list[type, b_dim, axis=1](
        b_host, k_partition, DimList(N, K_part), ctx
    )

    # create empty c submatrix
    var c_dev_list = List[DeviceNDBuffer[type, 2, c_dim]]()
    for _ in range(k_partition):
        c_dev_list.append(DeviceNDBuffer[type, 2, c_dim](ctx=ctx))

    var a_device = ctx.enqueue_create_buffer[type](M * K)
    var b_device = ctx.enqueue_create_buffer[type](K * N)
    var c_device = ctx.enqueue_create_buffer[type](M * N)
    var c_device_ref = ctx.enqueue_create_buffer[type](M * N)

    var a_tensor = LayoutTensor[type, a_layout](a_device, a_runtime_layout)
    var b_tensor = LayoutTensor[type, b_layout](b_device, b_runtime_layout)
    var c_tensor = LayoutTensor[type, c_layout](c_device, c_runtime_layout)

    alias kernels = MatmulKernels[type, type, type, transpose_b]()
    alias config = kernels.ampere_128x128_4
    alias mgemm = multistage_gemm_kernel[
        type,
        c_layout,
        type,
        a_layout,
        type,
        b_layout,
        transpose_b,
        config,
    ]

    ctx.enqueue_copy(a_device, a_host.tensor.data)
    ctx.enqueue_copy(b_device, b_host.tensor.data)

    print("copied to device")

    ctx.enqueue_function[mgemm](
        c_tensor,
        a_tensor,
        b_tensor,
        UnsafePointer[Int32](),
        grid_dim=config.grid_dim(M, N),
        block_dim=config.block_dim(),
        shared_mem_bytes=config.shared_mem_usage(),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            config.shared_mem_usage()
        ),
    )

    ctx.enqueue_copy(c_host.tensor.data, c_device)

    print("copied from device")

    with vendor_blas.Handle() as handle:
        vendor_blas.matmul(
            ctx,
            handle,
            c_dev_list[0].tensor,
            a_submat[0].tensor,
            b_submat[0].tensor,
            c_row_major=True,
            transpose_b=transpose_b,
        )

    print("cublas'd")

    ctx.enqueue_copy(c_host_ref.tensor.data, c_dev_list[0].buffer)

    print("copy from")

    ctx.synchronize()

    print("synchronized")

    assert_almost_equal(
        c_host.tensor,
        c_host_ref.tensor,
        atol=0.0001,
        rtol=0.01,
    )

    _ = a_submat^
    _ = b_submat^

    _ = c_device^

    _ = c_device_ref^
    _ = a_device^
    _ = b_device^

    _ = a_host^

    _ = b_host^

    _ = c_host^
    _ = c_host_ref^


def main():
    with DeviceContext() as ctx:
        test_split_k_multistage_gemm[
            DType.bfloat16, dim= (128, 128, 1024), transpose_b=True
        ](ctx, 8)
        print("-----")
        test_split_k_multistage_gemm[
            DType.float32, dim= (128, 128, 1024), transpose_b=True
        ](ctx, 8)
        print("-----")
