# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: H100-GPU
# RUN: %mojo-no-debug %s
# ALLOW_RETRIES: 5

from math import ceildiv
from sys import sizeof

import linalg.vendor_blas
from buffer import NDBuffer
from buffer.dimlist import Dim, DimList, _make_tuple
from gpu import WARP_SIZE, barrier
from gpu.host import DeviceContext
from gpu.host._compile import _compile_code_asm, _get_gpu_target
from gpu.id import block_dim, block_idx, thread_idx
from gpu.memory import AddressSpace
from gpu.mma import (
    WGMMADescriptor,
    wgmma_async,
    wgmma_commit_group_sync,
    wgmma_fence_aligned,
    wgmma_wait_group_sync,
)
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    arange,
    assert_almost_equal,
    assert_with_measure,
    fill,
    random,
    random_float8,
    zero,
)
from internal_utils._measure import cosine, relative_difference
from internal_utils._utils import ValOrDim, dynamic, static
from layout import IntTuple, Layout, LayoutTensor
from layout._ndbuffer_stub import from_ndbuffer_row_major
from layout._utils import ManagedLayoutTensor
from layout.layout_tensor import copy_local_to_dram
from layout.tensor_core_async import (
    TensorCoreAsync,
    _lhs_descriptor,
    _rhs_descriptor,
    tile_layout_k_major,
)
from linalg.matmul_sm90 import hopper_matmul_tma_wgmma
from testing import assert_true

from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple


fn test_hopper_fp8_matmul0_tma_wgmma[
    wgmma_n: Int,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool = True,
](
    ctx: DeviceContext,
    handle: vendor_blas.Handle,
    m: ValOrDim,
    n: ValOrDim,
    k: ValOrDim,
) raises:
    var M = m.value
    var N = n.value
    var K = k.value

    print(
        "wgmma_n",
        wgmma_n,
        " : ",
        M,
        "x",
        N,
        "x",
        K,
        " : ",
        a_type,
        "x",
        b_type,
        "x",
        c_type,
    )

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

    var a_host = HostNDBuffer[a_type, 2, static_a_shape](dynamic_a_shape)
    var b_host = HostNDBuffer[b_type, 2, static_b_shape](dynamic_b_shape)
    var c_host = HostNDBuffer[c_type, 2, static_c_shape](dynamic_c_shape)
    var c_host_ref = HostNDBuffer[c_type, 2, static_c_shape](dynamic_c_shape)

    var a_device = DeviceNDBuffer[a_type, 2, static_a_shape](
        dynamic_a_shape, ctx=ctx
    )
    var b_device = DeviceNDBuffer[b_type, 2, static_b_shape](
        dynamic_b_shape, ctx=ctx
    )
    var c_device = DeviceNDBuffer[c_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )
    var c_device_ref = DeviceNDBuffer[c_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )

    # Initialize matmul operands
    random_float8(a_host.tensor)
    random_float8(b_host.tensor)
    zero(c_host.tensor)
    zero(c_host_ref.tensor)

    # Move operands to the Device

    ctx.enqueue_copy(a_device.buffer, a_host.tensor.data)
    ctx.enqueue_copy(b_device.buffer, b_host.tensor.data)

    ctx.enqueue_copy(c_device.buffer, c_host.tensor.data)
    ctx.enqueue_copy(c_device_ref.buffer, c_host_ref.tensor.data)

    hopper_matmul_tma_wgmma[
        transpose_b=transpose_b,
        wgmma_shape = Index(64, wgmma_n, 32),
        block_tile_shape = Index(64, wgmma_n, 128),
    ](
        c_device.tensor,
        a_device.tensor,
        b_device.tensor,
        M,
        N,
        K,
        ctx,
    )

    ctx.synchronize()

    if transpose_b:
        vendor_blas.matmul(
            ctx,
            handle,
            c_device_ref.tensor,
            a_device.tensor,
            b_device.tensor,
            c_row_major=True,
            transpose_b=True,
        )

    else:
        # TODO: Matrix B should always be in col-major layout for cublasLt to work
        var b_host_col_major = HostNDBuffer[b_type, 2, static_b_shape](
            dynamic_b_shape
        )

        for i in range(N):
            for j in range(K):
                b_host_col_major.tensor[i, j] = b_host.tensor[j, i]

        var b_device_col_major = DeviceNDBuffer[b_type, 2, static_b_shape](
            dynamic_b_shape, ctx=ctx
        )
        ctx.enqueue_copy(
            b_device_col_major.buffer, b_host_col_major.tensor.data
        )

        vendor_blas.matmul(
            ctx,
            handle,
            c_device_ref.tensor,
            a_device.tensor,
            b_device_col_major.tensor,
            c_row_major=True,
            transpose_b=True,
        )

    ctx.synchronize()

    ctx.enqueue_copy(c_host.tensor.data, c_device.buffer)
    ctx.enqueue_copy(c_host_ref.tensor.data, c_device_ref.buffer)
    ctx.synchronize()

    # Both cutlass and cuBLAS promote output every 4 WGMMA instructions.
    # The threshold is set to a very low value to find potential changes to cutlass/cublas strategies.
    # If these tests fail, then it means that the cublas is doing something different.
    assert_with_measure[relative_difference](
        c_host.tensor, c_host_ref.tensor, threshold=0.00001
    )

    alias rtol = 0.0001
    assert_almost_equal(
        c_host.tensor,
        c_host_ref.tensor,
        atol=0.0001,
        rtol=rtol,
    )

    _ = c_device
    _ = c_device_ref
    _ = a_host
    _ = b_host
    _ = c_host_ref
    _ = c_host
    _ = a_device
    _ = b_device


fn main() raises:
    with DeviceContext() as ctx:
        with vendor_blas.Handle[vendor_blas.Backend.CUBLASLT]() as handle:
            test_hopper_fp8_matmul0_tma_wgmma[
                80, DType.float8_e4m3fn, DType.float8_e4m3fn, DType.float32
            ](ctx, handle, static[512](), static[2560](), static[8192]())

            test_hopper_fp8_matmul0_tma_wgmma[
                256, DType.float8_e4m3fn, DType.float8_e4m3fn, DType.float32
            ](ctx, handle, static[8192](), static[2560](), static[8192]())

            test_hopper_fp8_matmul0_tma_wgmma[
                256, DType.float8_e4m3fn, DType.float8_e4m3fn, DType.float32
            ](ctx, handle, static[4096](), static[2560](), static[8192]())

            test_hopper_fp8_matmul0_tma_wgmma[
                256, DType.float8_e4m3fn, DType.float8_e4m3fn, DType.float32
            ](ctx, handle, static[8192](), static[8192](), static[2048]())

            test_hopper_fp8_matmul0_tma_wgmma[
                256, DType.float8_e4m3fn, DType.float8_e4m3fn, DType.float32
            ](ctx, handle, static[4096](), static[8192](), static[2048]())

            test_hopper_fp8_matmul0_tma_wgmma[
                256, DType.float8_e4m3fn, DType.float8_e4m3fn, DType.float32
            ](ctx, handle, static[8192](), static[14336](), static[8192]())

            test_hopper_fp8_matmul0_tma_wgmma[
                256, DType.float8_e4m3fn, DType.float8_e4m3fn, DType.float32
            ](ctx, handle, static[4096](), static[14336](), static[8192]())

            test_hopper_fp8_matmul0_tma_wgmma[
                256, DType.float8_e4m3fn, DType.float8_e4m3fn, DType.float32
            ](ctx, handle, static[8192](), static[8192](), static[7168]())

            test_hopper_fp8_matmul0_tma_wgmma[
                256, DType.float8_e4m3fn, DType.float8_e4m3fn, DType.float32
            ](ctx, handle, static[4096](), static[8192](), static[7168]())

            test_hopper_fp8_matmul0_tma_wgmma[
                80, DType.float8_e4m3fn, DType.float8_e4m3fn, DType.bfloat16
            ](ctx, handle, static[512](), static[2560](), static[8192]())

            test_hopper_fp8_matmul0_tma_wgmma[
                256, DType.float8_e4m3fn, DType.float8_e4m3fn, DType.bfloat16
            ](ctx, handle, static[8192](), static[2560](), static[8192]())

            test_hopper_fp8_matmul0_tma_wgmma[
                256, DType.float8_e4m3fn, DType.float8_e4m3fn, DType.bfloat16
            ](ctx, handle, static[4096](), static[2560](), static[8192]())

            test_hopper_fp8_matmul0_tma_wgmma[
                256, DType.float8_e4m3fn, DType.float8_e4m3fn, DType.bfloat16
            ](ctx, handle, static[8192](), static[8192](), static[2048]())

            test_hopper_fp8_matmul0_tma_wgmma[
                256, DType.float8_e4m3fn, DType.float8_e4m3fn, DType.bfloat16
            ](ctx, handle, static[4096](), static[8192](), static[2048]())

            test_hopper_fp8_matmul0_tma_wgmma[
                256, DType.float8_e4m3fn, DType.float8_e4m3fn, DType.bfloat16
            ](ctx, handle, static[8192](), static[14336](), static[8192]())

            test_hopper_fp8_matmul0_tma_wgmma[
                256, DType.float8_e4m3fn, DType.float8_e4m3fn, DType.bfloat16
            ](ctx, handle, static[4096](), static[14336](), static[8192]())

            test_hopper_fp8_matmul0_tma_wgmma[
                256, DType.float8_e4m3fn, DType.float8_e4m3fn, DType.bfloat16
            ](ctx, handle, static[8192](), static[8192](), static[7168]())

            test_hopper_fp8_matmul0_tma_wgmma[
                256, DType.float8_e4m3fn, DType.float8_e4m3fn, DType.bfloat16
            ](ctx, handle, static[4096](), static[8192](), static[7168]())
