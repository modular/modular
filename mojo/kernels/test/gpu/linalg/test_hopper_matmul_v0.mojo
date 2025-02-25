# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: H100-GPU
# RUN: %mojo-no-debug-no-assert %s

from math import ceildiv
from sys import sizeof

import linalg.vendor_blas
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
    assert_almost_equal,
    fill,
    arange,
    random,
    zero,
)
from internal_utils._measure import cosine
from internal_utils._utils import ValOrDim, dynamic, static
from layout import IntTuple, Layout, LayoutTensor
from layout._utils import ManagedLayoutTensor
from layout.layout_tensor import copy_local_to_dram
from layout.nd_buffer_stub import from_ndbuffer_row_major
from utils.numerics import get_accum_type
from layout.tensor_core_async import (
    TensorCoreAsync,
    _lhs_descriptor,
    _rhs_descriptor,
    tile_layout_k_major,
)
from layout.tma_async import (
    PipelineState,
    TMABarrier,
    TMATensorTile,
    create_tma_tile,
)
from linalg.matmul_sm90 import hopper_matmul_tma_wgmma
from utils.index import Index, IndexList
from utils.static_tuple import StaticTuple


def test_hopper_matmul0_tma_wgmma[
    wgmma_n: Int,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool = True,
](ctx: DeviceContext, m: ValOrDim, n: ValOrDim, k: ValOrDim,):
    var M = m.value
    var N = n.value
    var K = k.value

    print("wgmma_n", wgmma_n, " : ", M, "x", N, "x", K)

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
    random(a_host.tensor)
    random(b_host.tensor)
    zero(c_host.tensor)
    zero(c_host_ref.tensor)

    # Move operands to the Device

    ctx.enqueue_copy_to_device(a_device.buffer, a_host.tensor.data)
    ctx.enqueue_copy_to_device(b_device.buffer, b_host.tensor.data)

    ctx.enqueue_copy_to_device(c_device.buffer, c_host.tensor.data)
    ctx.enqueue_copy_to_device(c_device_ref.buffer, c_host_ref.tensor.data)

    hopper_matmul_tma_wgmma[transpose_b=transpose_b, wgmma_n=wgmma_n](
        c_device.tensor,
        a_device.tensor,
        b_device.tensor,
        M,
        N,
        K,
        ctx,
    )

    ctx.synchronize()

    with vendor_blas.Handle() as handle:
        vendor_blas.matmul(
            ctx,
            handle,
            c_device_ref.tensor,
            a_device.tensor,
            b_device.tensor,
            c_row_major=True,
            transpose_b=transpose_b,
        )

    ctx.synchronize()

    ctx.enqueue_copy_from_device(c_host.tensor.data, c_device.buffer)
    ctx.enqueue_copy_from_device(c_host_ref.tensor.data, c_device_ref.buffer)
    ctx.synchronize()
    alias rtol = 1e-2
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


def main():
    with DeviceContext() as ctx:
        test_hopper_matmul0_tma_wgmma[
            128, DType.bfloat16, DType.bfloat16, DType.bfloat16
        ](ctx, static[128](), static[128](), static[128]())

        test_hopper_matmul0_tma_wgmma[
            64, DType.bfloat16, DType.bfloat16, DType.bfloat16
        ](ctx, static[128](), static[64](), static[64]())

        alias wgmma_n = List[Int](8, 32, 64, 128, 256)
        alias num_ins = 5

        @parameter
        for i in range(num_ins):
            test_hopper_matmul0_tma_wgmma[
                wgmma_n[i], DType.bfloat16, DType.bfloat16, DType.bfloat16
            ](ctx, static[1024](), static[512](), static[128]())

            test_hopper_matmul0_tma_wgmma[
                wgmma_n[i], DType.bfloat16, DType.bfloat16, DType.bfloat16
            ](ctx, dynamic(1024), static[512](), static[128]())

            test_hopper_matmul0_tma_wgmma[
                wgmma_n[i], DType.bfloat16, DType.bfloat16, DType.bfloat16
            ](ctx, dynamic(99), static[1024](), static[1024]())

            test_hopper_matmul0_tma_wgmma[
                wgmma_n[i], DType.bfloat16, DType.bfloat16, DType.bfloat16
            ](ctx, dynamic(100), static[512](), static[256]())

            test_hopper_matmul0_tma_wgmma[
                wgmma_n[i], DType.bfloat16, DType.bfloat16, DType.bfloat16
            ](ctx, dynamic(201), static[2048](), static[256]())
