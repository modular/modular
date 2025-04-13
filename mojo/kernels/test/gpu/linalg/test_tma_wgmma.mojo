# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: H100-GPU
# RUN: %mojo-no-debug %s

from math import ceildiv
from sys import sizeof

from gpu import WARP_SIZE, barrier, warp_id as get_warp_id
from gpu.host import DeviceContext
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.id import block_idx, thread_idx
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from layout._fillers import arange
from layout._utils import ManagedLayoutTensor
from layout.int_tuple import product
from layout.layout_tensor import copy_local_to_dram
from layout.tensor_core_async import (
    TensorCoreAsync,
    tile_layout_k_major,
    tile_layout_mn_major,
)
from layout.tma_async import SharedMemBarrier, TMATensorTile, create_tma_tile
from linalg import vendor_blas
from memory import stack_allocation
from memory.pointer import _GPUAddressSpace
from testing import assert_almost_equal

from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple


fn _compute_reg_tile_layout(layout: Layout, frag_size: Int) -> Layout:
    var local_size = layout.size() // 128
    return Layout.row_major(local_size // frag_size, frag_size)


@always_inline
fn _load_a_reg_tile[
    dtype: DType,
    layout: Layout, //,
    wgmma_shape: IndexList[3],
](
    out ret: LayoutTensor[
        dtype,
        _compute_reg_tile_layout(layout, 16 // sizeof[dtype]()),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ],
    smem_tile: LayoutTensor[
        dtype,
        layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        *_, **_,
    ],
):
    constrained[ret.layout[0].shape[0].value() > 0]()
    ret = __type_of(ret).stack_allocation()
    var tid = thread_idx.x
    var lane = tid % 32
    var wgid = tid // 32
    alias WGMMA_M = wgmma_shape[0]
    alias WGMMA_K = wgmma_shape[2]

    alias rows = product(layout[0].shape)
    alias cols = product(layout[1].shape)

    alias num_wgmma_m = ceildiv(rows, WGMMA_M)
    alias num_wgmma_k = ceildiv(cols, WGMMA_K)
    constrained[num_wgmma_m * num_wgmma_k == ret.layout[0].shape[0].value()]()

    alias simd_size = 4 // sizeof[dtype]()
    var vret = ret.vectorize[1, simd_size]()

    @parameter
    for m_mma in range(num_wgmma_m):

        @parameter
        for k_mma in range(num_wgmma_k):
            alias r_id = m_mma + k_mma * num_wgmma_m
            var smem_wg = smem_tile.tile[WGMMA_M, WGMMA_K](m_mma, k_mma).tile[
                WGMMA_M // 4, WGMMA_K
            ](wgid, 0).vectorize[1, simd_size]().distribute[
                Layout.row_major(8, 4)
            ](
                lane
            )
            vret.tile[1, 4](r_id, 0).copy_from(smem_wg)


@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
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
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    a_smem: Bool = True,
](
    a_tma_op: TMATensorTile[a_type, a_layout, a_desc_layout],
    b_tma_op: TMATensorTile[b_type, b_layout, b_desc_layout],
    c: LayoutTensor[c_type, c_layout, MutableAnyOrigin],
    num_iters: UInt,
):
    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]
    alias num_m_mmas = BM // wgmma_shape[0]
    alias num_n_mmas = BN // wgmma_shape[1]

    alias a_smem_layout = tile_layout_k_major[
        a_type, BM, BK, swizzle_mode=a_swizzle
    ]()
    alias b_smem_layout = tile_layout_k_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]() if transpose_b else tile_layout_mn_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]()

    var a_smem_tile = LayoutTensor[
        a_type,
        a_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ].stack_allocation()

    var b_smem_tile = LayoutTensor[
        b_type,
        b_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ].stack_allocation()

    alias accum_type = get_accum_type[a_type]()
    wgmma_op = TensorCoreAsync[
        accum_type,
        a_type,
        b_type,
        wgmma_shape,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        transpose_b=transpose_b,
    ]()

    alias c_frag_size = wgmma_shape[0] * wgmma_shape[1] // 128
    var c_reg_tile = LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas * num_n_mmas, c_frag_size),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ].stack_allocation()

    _ = c_reg_tile.fill(0.0)

    alias a_expected_bytes = a_smem_layout.size() * sizeof[a_type]()
    alias b_expected_bytes = b_smem_layout.size() * sizeof[b_type]()
    alias expected_bytes = a_expected_bytes + b_expected_bytes

    mbar = stack_allocation[
        1,
        SharedMemBarrier,
        address_space = _GPUAddressSpace.SHARED,
        alignment=8,
    ]()
    if thread_idx.x == 0:
        mbar[0].init()

    var phase: UInt32 = 0

    for i in range(num_iters):
        if thread_idx.x == 0:
            mbar[0].expect_bytes(expected_bytes)
            a_tma_op.async_copy(
                a_smem_tile, mbar[0], (UInt(i) * BK, block_idx.y * BM)
            )
            b_tma_op.async_copy(
                b_smem_tile,
                mbar[0],
                (UInt(i) * BK, block_idx.x * BN) if transpose_b else (
                    block_idx.x * BN,
                    UInt(i) * BK,
                ),
            )

        # Ensure all threads sees initialized mbarrier
        barrier()

        mbar[0].wait(phase)
        phase ^= 1

        wgmma_op.arrive()

        @parameter
        if a_smem:
            wgmma_op.wgmma(a_smem_tile, b_smem_tile, c_reg_tile)
        else:
            var a_reg_tile = _load_a_reg_tile[wgmma_shape](a_smem_tile)
            wgmma_op.wgmma(a_reg_tile, b_smem_tile, c_reg_tile)
        wgmma_op.commit_group()
        wgmma_op.wait_group()

        barrier()

    c_gmem_tile = c.tile[BM, BN](block_idx.y, block_idx.x)
    warp_id = get_warp_id()

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
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    a_smem: Bool = True,
](ctx: DeviceContext):
    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]

    alias WGMMA_M = wgmma_shape[0]
    alias WGMMA_N = wgmma_shape[1]
    alias WGMMA_K = wgmma_shape[2]

    print(
        "wgmma_"
        + ("s" if a_smem else StaticString("r"))
        + "s_bf16_bf16_f32 block tile "
        + String(block_tile_shape)
        + " transb="
        + String(transpose_b)
        + "; inst shape "
        + String(wgmma_shape)
        + " A "
        + String(a_swizzle)
        + " B "
        + String(b_swizzle)
    )

    alias M = prob_shape[0]
    alias N = prob_shape[1]
    alias K = prob_shape[2]

    var a = ManagedLayoutTensor[
        a_type,
        Layout.row_major(M, K),
    ](ctx)
    arange(a.tensor[update=False]())

    alias b_layout = Layout.row_major(
        N, K
    ) if transpose_b else Layout.row_major(K, N)
    var b = ManagedLayoutTensor[b_type, b_layout](ctx)
    arange(b.tensor[update=False]())

    var c = ManagedLayoutTensor[
        c_type,
        Layout.row_major(M, N),
    ](ctx)

    var c_ref = ManagedLayoutTensor[
        c_type,
        Layout.row_major(M, N),
    ](ctx)

    a_tma_op = create_tma_tile[
        a_type, 2, Index(BM, BK), swizzle_mode=a_swizzle
    ](ctx, a.device_tensor())
    b_tma_op = create_tma_tile[
        b_type,
        2,
        Index(BN, BK) if transpose_b else Index(BK, BN),
        is_k_major=transpose_b,
        swizzle_mode=b_swizzle,
    ](ctx, b.device_tensor())

    alias kernel = tma_wgmma_kernel[
        a_type,
        b_type,
        c_type,
        __type_of(a_tma_op).layout,
        __type_of(b_tma_op).layout,
        Layout.row_major(M, N),
        __type_of(a_tma_op).desc_layout,
        __type_of(b_tma_op).desc_layout,
        block_tile_shape,
        wgmma_shape,
        transpose_b=transpose_b,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        a_smem=a_smem,
    ]

    ctx.enqueue_function[kernel](
        # ctx.enqueue_function[kernel, dump_llvm=Path("invalid.ll")](
        a_tma_op,
        b_tma_op,
        c.device_tensor(),
        K // BK,
        grid_dim=(1, 1),
        block_dim=(128),
    )

    vendor_blas.matmul(
        ctx,
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
            a_swizzle = TensorMapSwizzle.SWIZZLE_128B,
            b_swizzle = TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)

        test_tma_wgmma[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(128, 16, 32),
            Index(128, 16, 32),
            Index(64, 8, 16),
            a_swizzle = TensorMapSwizzle.SWIZZLE_64B,
            b_swizzle = TensorMapSwizzle.SWIZZLE_64B,
        ](ctx)

        test_tma_wgmma[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(128, 16, 16),
            Index(128, 16, 16),
            Index(64, 8, 16),
            a_swizzle = TensorMapSwizzle.SWIZZLE_32B,
            b_swizzle = TensorMapSwizzle.SWIZZLE_32B,
        ](ctx)

        test_tma_wgmma[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(64, 128, 16),
            Index(64, 128, 16),
            Index(64, 64, 16),
            a_swizzle = TensorMapSwizzle.SWIZZLE_NONE,
            b_swizzle = TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=False,
        ](ctx)

        test_tma_wgmma[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(64, 128, 16),
            Index(64, 128, 16),
            Index(64, 64, 16),
            a_swizzle = TensorMapSwizzle.SWIZZLE_NONE,
            b_swizzle = TensorMapSwizzle.SWIZZLE_128B,
            a_smem=False,
            transpose_b=False,
        ](ctx)

        test_tma_wgmma[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            Index(64, 8, 64),
            Index(64, 8, 64),
            Index(64, 8, 16),
            a_swizzle = TensorMapSwizzle.SWIZZLE_NONE,
            b_swizzle = TensorMapSwizzle.SWIZZLE_128B,
            a_smem=False,
            transpose_b=True,
        ](ctx)
