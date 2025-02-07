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
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.id import block_idx, thread_idx
from gpu.intrinsics import threadfence
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
from layout.fillers import arange
from layout.int_tuple import to_int
from layout.layout import print_layout
from layout.layout_tensor import copy_local_to_dram
from layout.tensor_core_async import (
    tile_layout_k_major,
    TensorCoreAsync,
    _lhs_descriptor,
    _rhs_descriptor,
)
from layout.tensor_core import get_accum_type
from layout.tma_async import TMATensorTile, create_tma_tile, TMABarrier
from linalg import vendor_blas
from memory import bitcast
from utils.index import Index, IndexList
from utils.static_tuple import StaticTuple
from sys import sizeof
from testing import assert_almost_equal


@__llvm_metadata(`nvvm.grid_constant`=StaticTuple[Int, 2](0, 1))
fn tma_wgmma_kernel[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    block_tile_shape: IndexList[3],
    wgmma_shape: IndexList[3],
    a_smem_layout: Layout,
    b_smem_layout: Layout,
    transpose_b: Bool = True,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
](
    a_tma_op: TMATensorTile[a_type, a_layout, a_desc_layout],
    b_tma_op: TMATensorTile[b_type, b_layout, b_desc_layout],
    c: LayoutTensor[c_type, c_layout],
    num_iters: UInt,
):
    constrained[transpose_b, "Only support transposed B in layout"]()

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

    alias accum_type = get_accum_type[a_type]()
    wgmma_op = TensorCoreAsync[
        accum_type,
        a_type,
        b_type,
        wgmma_shape,
        a_swizzle=swizzle_mode,
        b_swizzle=swizzle_mode,
        transpose_b=transpose_b,
    ]()

    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[1]
    alias num_m_mmas = BM // wgmma_shape[0]
    alias num_n_mmas = BN // wgmma_shape[1]

    alias c_frag_size = wgmma_shape[0] * wgmma_shape[1] // 128
    var c_reg_tile = LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas * num_n_mmas, c_frag_size),
        address_space = AddressSpace.LOCAL,
    ].stack_allocation()

    _ = c_reg_tile.fill(0.0)

    alias a_expected_bytes = a_smem_layout.size() * sizeof[a_type]()
    alias b_expected_bytes = b_smem_layout.size() * sizeof[b_type]()
    alias expected_bytes = a_expected_bytes + b_expected_bytes

    mbar = TMABarrier()
    if thread_idx.x == 0:
        mbar.init()

    var phase: UInt32 = 0

    for i in range(num_iters):
        if thread_idx.x == 0:
            mbar.expect_bytes(expected_bytes)
            a_tma_op.async_copy(
                a_smem_tile, mbar, (UInt(i) * BK, block_idx.y * BM)
            )
            b_tma_op.async_copy(
                b_smem_tile, mbar, (UInt(i) * BK, block_idx.x * BN)
            )

        # Ensure all threads sees initialized mbarrier
        barrier()

        mbar.wait(phase)
        phase ^= 1

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


def test_tma_wgmma[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    prob_shape: IndexList[3],
    block_tile_shape: IndexList[3],
    wgmma_shape: IndexList[3],
    transpose_b: Bool = True,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
](ctx: DeviceContext):
    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]

    alias WGMMA_M = wgmma_shape[0]
    alias WGMMA_N = wgmma_shape[1]
    alias WGMMA_K = wgmma_shape[2]

    print(
        "wgmma_bf16_bf16_f32 block tile "
        + String(block_tile_shape)
        + " transb inst shape "
        + String(wgmma_shape)
        + " "
        + String(swizzle_mode)
    )

    constrained[transpose_b, "Only support transpose_b for now"]()

    alias M = prob_shape[0]
    alias N = prob_shape[1]
    alias K = prob_shape[2]

    var a = ManagedLayoutTensor[
        a_type,
        Layout.row_major(M, K),
    ](ctx)
    arange(a.tensor())

    var b = ManagedLayoutTensor[
        b_type,
        Layout.row_major(N, K),
    ](ctx)
    arange(b.tensor())

    var c = ManagedLayoutTensor[
        c_type,
        Layout.row_major(M, N),
    ](ctx)

    var c_ref = ManagedLayoutTensor[
        c_type,
        Layout.row_major(M, N),
    ](ctx)

    # Shared memory tile layouts
    alias a_smem_layout = tile_layout_k_major[
        a_type, BM, BK, swizzle_mode=swizzle_mode
    ]()
    alias b_smem_layout = tile_layout_k_major[
        b_type, BN, BK, swizzle_mode=swizzle_mode
    ]()

    a_tma_op = create_tma_tile[
        a_type, 2, Index(BM, BK), swizzle_mode=swizzle_mode
    ](ctx, a.device_tensor())
    b_tma_op = create_tma_tile[
        b_type, 2, Index(BN, BK), swizzle_mode=swizzle_mode
    ](ctx, b.device_tensor())

    alias kernel = tma_wgmma_kernel[
        a_type,
        b_type,
        c_type,
        Layout.row_major(BM, BK),
        Layout.row_major(BN, BK),
        Layout.row_major(M, N),
        __type_of(a_tma_op).desc_layout,
        __type_of(b_tma_op).desc_layout,
        block_tile_shape,
        wgmma_shape,
        a_smem_layout,
        b_smem_layout,
        transpose_b=True,
        swizzle_mode=swizzle_mode,
    ]
    var func = ctx.compile_function[
        kernel,
        _target = _get_gpu_target["sm_90"](),
    ]()

    ctx.enqueue_function(
        func,
        a_tma_op,
        b_tma_op,
        c.device_tensor(),
        K // BK,
        grid_dim=(1, 1),
        block_dim=(128),
    )

    with vendor_blas.Handle() as handle:
        vendor_blas.matmul(
            ctx,
            handle,
            c_ref.device_buffer(),
            a.device_buffer[update=False](),
            b.device_buffer[update=False](),
            c_row_major=True,
            transpose_b=transpose_b,
        )

    ctx.synchronize()

    c_host = c.tensor()
    c_host_ref = c_ref.tensor()

    for m in range(M):
        for n in range(N):
            assert_almost_equal(
                c_host[m, n], c_host_ref[m, n], atol=1e-3, rtol=1e-4
            )

    # print(c.tensor())
    _ = a^
    _ = b^
    _ = c^
    _ = c_ref^


def main():
    with DeviceContext() as ctx:
        test_tma_wgmma[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(128, 16, 32),
            Index(128, 16, 32),
            Index(64, 8, 16),
        ](ctx)

        test_tma_wgmma[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(64, 8, 64),
            Index(64, 8, 64),
            Index(64, 8, 16),
        ](ctx)

        test_tma_wgmma[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(64, 8, 64),
            Index(64, 8, 64),
            Index(64, 8, 16),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)
