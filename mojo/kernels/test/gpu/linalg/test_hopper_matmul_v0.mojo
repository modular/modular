# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: H100-GPU
# RUN: %mojo-no-debug-no-assert %s

from gpu import barrier, WARP_SIZE
from gpu.host import DeviceContext
from gpu.host._compile import _compile_code_asm, _get_gpu_target
from gpu.id import block_idx, thread_idx, block_dim
from gpu.memory import AddressSpace
from gpu.mma import (
    WGMMADescriptor,
    wgmma_async,
    wgmma_commit_group_sync,
    wgmma_fence_aligned,
    wgmma_wait_group_sync,
)
from layout import IntTuple, Layout, LayoutTensor
from layout._utils import ManagedLayoutTensor
from layout.nd_buffer_stub import from_ndbuffer_row_major
from layout.layout_tensor import copy_local_to_dram
from layout.tensor_core import get_accum_type
from layout.tensor_core_async import (
    tile_layout_k_major,
    TensorCoreAsync,
    _lhs_descriptor,
    _rhs_descriptor,
)
from layout.tma_async import (
    TMATensorTile,
    create_tma_tile,
    TMABarrier,
    PipelineState,
)
from buffer.dimlist import Dim, DimList, _make_tuple
from internal_utils._utils import ValOrDim, dynamic, static
from internal_utils._measure import cosine
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    assert_almost_equal,
    fill,
    linspace,
    random,
    zero,
)
import linalg.vendor_blas
from utils.index import Index, IndexList
from utils.static_tuple import StaticTuple
from math import ceildiv
from sys import sizeof


@__llvm_metadata(`nvvm.grid_constant`=StaticTuple[Int, 2](0, 1))
fn test_hopper_matmul0_tma_wgmma_kernel[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    a_tile_layout: Layout,
    b_tile_layout: Layout,
    c_layout: Layout,
    block_tile_shape: IndexList[3],
    wgmma_shape: IndexList[3],
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    a_smem_layout: Layout,
    b_smem_layout: Layout,
    transpose_b: Bool = True,
](
    a_tma_op: TMATensorTile[a_type, a_tile_layout, a_desc_layout],
    b_tma_op: TMATensorTile[b_type, b_tile_layout, b_desc_layout],
    c: LayoutTensor[c_type, c_layout],
):
    constrained[transpose_b, "Only support transposed B in layout"]()

    alias K = b_layout.shape[1].value()

    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]

    alias num_m_mmas = BM // wgmma_shape[0]
    alias num_n_mmas = BN // wgmma_shape[1]

    alias accum_type = get_accum_type[a_type]()
    alias c_frag_size = wgmma_shape[0] * wgmma_shape[1] // 128

    alias num_k_iters = K // BK

    var a_smem_tile = LayoutTensor[
        a_type,
        a_smem_layout,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ].stack_allocation()

    var b_smem_tile = LayoutTensor[
        b_type,
        b_smem_layout,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ].stack_allocation()

    var c_reg_tile = LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas * num_n_mmas, c_frag_size),
        address_space = AddressSpace.LOCAL,
    ].stack_allocation()

    _ = c_reg_tile.fill(0.0)

    wgmma_op = TensorCoreAsync[
        accum_type, a_type, b_type, wgmma_shape, transpose_b=transpose_b
    ]()

    alias a_expected_bytes = a_smem_layout.size() * sizeof[a_type]()
    alias b_expected_bytes = b_smem_layout.size() * sizeof[b_type]()
    alias expected_bytes = a_expected_bytes + b_expected_bytes

    var mbar = TMABarrier()
    var phase = PipelineState[1]()

    if thread_idx.x == 0:
        mbar.init()

    barrier()

    for i in range(num_k_iters):
        if thread_idx.x == 0:
            mbar.expect_bytes(expected_bytes)
            a_tma_op.async_copy(
                a_smem_tile,
                mbar,
                (UInt(i) * BK, block_idx.y * BM),
            )

            b_tma_op.async_copy(
                b_smem_tile,
                mbar,
                (UInt(i) * BK, block_idx.x * BN),
            )
        barrier()

        mbar.wait(phase.phase())
        phase.step()

        wgmma_op.arrive()
        wgmma_op.wgmma(a_smem_tile, b_smem_tile, c_reg_tile)
        wgmma_op.commit_group()
        wgmma_op.wait_for_all()

        barrier()

    c_gmem_tile = c.tile[BM, BN](block_idx.y, block_idx.x)
    warp_id = thread_idx.x // WARP_SIZE

    @parameter
    for m_mma in range(num_m_mmas):

        @parameter
        for n_mma in range(num_n_mmas):
            alias mma_id = n_mma * num_m_mmas + m_mma

            # (m_mma, n_mma) is coordinates for a warp group's tile.
            # A warp group is 4x1 warps.
            warp_tile = c_gmem_tile.tile[wgmma_shape[0] // 4, wgmma_shape[1]](
                m_mma * 4 + warp_id, n_mma
            )

            # Tile at (mma_id, 0) is a long vector containing all fragments
            # for this warp.
            c_frag = c_reg_tile.tile[1, c_frag_size](mma_id, 0)

            # A warp is organized as row_major(8, 4) and each thread owns 2 contiguous
            # elementwise. This pattern repeates to fill the warp tile.
            copy_local_to_dram[Layout.row_major(8, 4)](
                warp_tile.vectorize[1, 2](), c_frag.vectorize[1, 2]()
            )


def test_hopper_matmul0_tma_wgmma[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool = True,
](ctx: DeviceContext, m: ValOrDim, n: ValOrDim, k: ValOrDim,):
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

    var a = from_ndbuffer_row_major(a_device.tensor)
    var b = from_ndbuffer_row_major(b_device.tensor)
    var c = from_ndbuffer_row_major(c_device.tensor)

    alias block_tile_shape = Index(64, 8, 32)
    alias wgmma_shape = Index(64, 8, 16)

    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]

    alias a_smem_layout = tile_layout_k_major[a_type, BM, BK]()
    alias b_smem_layout = tile_layout_k_major[b_type, BN, BK]()

    a_tma_op = create_tma_tile[a_type, 2, Index(BM, BK)](ctx, a)
    b_tma_op = create_tma_tile[b_type, 2, Index(BN, BK)](ctx, b)

    alias kernel = test_hopper_matmul0_tma_wgmma_kernel[
        a_type,
        b_type,
        c_type,
        __type_of(a).layout,
        __type_of(b).layout,
        Layout.row_major(BM, BK),
        Layout.row_major(BN, BK),
        __type_of(c).layout,
        block_tile_shape,
        wgmma_shape,
        __type_of(a_tma_op).desc_layout,
        __type_of(b_tma_op).desc_layout,
        a_smem_layout,
        b_smem_layout,
        transpose_b=True,
    ]

    var func = ctx.compile_function[
        kernel,
        _target = _get_gpu_target["sm_90"](),
    ]()

    ctx.enqueue_function(
        func,
        a_tma_op,
        b_tma_op,
        c,
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(128),
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

    _ = a
    _ = b
    _ = c


def main():
    with DeviceContext() as ctx:
        test_hopper_matmul0_tma_wgmma[
            DType.bfloat16, DType.bfloat16, DType.bfloat16
        ](ctx, static[128](), static[128](), static[128]())

        test_hopper_matmul0_tma_wgmma[
            DType.bfloat16, DType.bfloat16, DType.bfloat16
        ](ctx, static[128](), static[64](), static[64]())

        test_hopper_matmul0_tma_wgmma[
            DType.bfloat16, DType.bfloat16, DType.bfloat16
        ](ctx, static[1024](), static[512](), static[128]())

        test_hopper_matmul0_tma_wgmma[
            DType.bfloat16, DType.bfloat16, DType.bfloat16
        ](ctx, dynamic(1024), static[512](), static[128]())

        test_hopper_matmul0_tma_wgmma[
            DType.bfloat16, DType.bfloat16, DType.bfloat16
        ](ctx, dynamic(99), static[1024](), static[1024]())

        test_hopper_matmul0_tma_wgmma[
            DType.bfloat16, DType.bfloat16, DType.bfloat16
        ](ctx, dynamic(100), static[512](), static[256]())

        test_hopper_matmul0_tma_wgmma[
            DType.bfloat16, DType.bfloat16, DType.bfloat16
        ](ctx, dynamic(201), static[2048](), static[256]())
