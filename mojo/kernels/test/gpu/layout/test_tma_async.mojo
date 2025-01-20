# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: H100-GPU
# RUN: %mojo-no-debug-no-assert %s

from builtin.io import _printf
from gpu.host import DeviceContext
from gpu.host._compile import _get_gpu_target
from gpu.id import block_idx, thread_idx
from gpu import barrier
from layout import Layout, LayoutTensor
from layout.tma_async import TMATensorTile, create_tma_tile, TMABarrier
from layout._utils import ManagedLayoutTensor
from layout.fillers import arange
from layout.layout_tensor import copy_sram_to_dram
from memory.pointer import _GPUAddressSpace

from math import align_up, ceildiv
from sys import sizeof
from testing import assert_equal
from utils.static_tuple import StaticTuple


@__llvm_metadata(`nvvm.grid_constant`=StaticTuple[Int, 1](1))
fn test_tma_load_kernel[
    dtype: DType,
    layout: Layout,
    tile_layout: Layout,
    thread_layout: Layout,
](
    dst: LayoutTensor[dtype, layout],
    tma_tile: TMATensorTile[dtype, tile_layout],
):
    alias tileM = tile_layout.shape[0].value()
    alias tileN = tile_layout.shape[1].value()
    alias expected_bytes = tile_layout.size() * sizeof[dtype]()

    tile = LayoutTensor[
        dtype,
        tile_layout,
        address_space = _GPUAddressSpace.SHARED,
        alignment=128,
    ].stack_allocation()

    mbar = TMABarrier()

    if thread_idx.x == 0:
        mbar.init()
        mbar.expect_bytes(expected_bytes)
        tma_tile.async_copy(
            tile, mbar, (block_idx.x * tileN, block_idx.y * tileM)
        )
    # Ensure all threads sees initialized mbarrier
    barrier()
    mbar.wait()

    dst_tile = dst.tile[tileM, tileN](block_idx.y, block_idx.x)
    copy_sram_to_dram[thread_layout](dst_tile, tile)


def test_tma_load_row_major[
    src_layout: Layout, tile_layout: Layout
](ctx: DeviceContext):
    alias M = src_layout.shape[0].value()
    alias N = src_layout.shape[1].value()
    alias tileM = tile_layout.shape[0].value()
    alias tileN = tile_layout.shape[1].value()
    alias M_roundup = align_up(M, tileM)
    alias N_roundup = align_up(N, tileN)

    var src = ManagedLayoutTensor[DType.float32, src_layout](ctx)
    var dst = ManagedLayoutTensor[
        DType.float32, Layout.row_major(M_roundup, N_roundup)
    ](ctx)

    arange(src.tensor(), 1)
    var tma_tensor = create_tma_tile[tileM, tileN](ctx, src.device_tensor())
    ctx.synchronize()

    var kernel = ctx.compile_function[
        test_tma_load_kernel[
            __type_of(tma_tensor).dtype,
            Layout.row_major(M_roundup, N_roundup),  # dst layout
            __type_of(tma_tensor).layout,  # smem layout
            __type_of(tma_tensor).layout,  # thread layout
        ],
        _target = _get_gpu_target["sm_90"](),
    ]()
    ctx.enqueue_function(
        kernel,
        dst.device_tensor(),
        tma_tensor,
        grid_dim=(N_roundup // tileN, M_roundup // tileM),
        block_dim=(tileM * tileN),
    )

    src_host = dst.tensor()
    dst_host = dst.tensor()

    # Check M x N keep the same value and others in M_roundup x N_roundup
    # are set to zeros.
    for m in range(M_roundup):
        for n in range(N_roundup):
            if m < M and n < N:
                assert_equal(
                    src_host[m, n].cast[DType.float32](),
                    dst_host[m, n].cast[DType.float32](),
                )
            else:
                assert_equal(dst_host[m, n].cast[DType.float32](), 0.0)

    ctx.synchronize()
    _ = kernel^
    _ = src^
    _ = dst^


def main():
    with DeviceContext() as ctx:
        # fmt: off
        print("test_tma_load")
        test_tma_load_row_major[Layout.row_major(8, 8), Layout.row_major(4, 4)](ctx)
        test_tma_load_row_major[Layout.row_major(9, 24), Layout.row_major(3, 8)](ctx)
        print("test_tma_load_oob_file")
        test_tma_load_row_major[Layout.row_major(7, 8), Layout.row_major(4, 4)](ctx)
        test_tma_load_row_major[Layout.row_major(10, 12), Layout.row_major(4, 8)](ctx)
        # fmt: on
