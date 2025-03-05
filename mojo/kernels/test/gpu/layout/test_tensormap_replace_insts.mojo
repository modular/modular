# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: H100-GPU
# RUN: %mojo-no-debug-no-assert %s

from math import align_up, ceildiv
from sys import sizeof

from builtin.io import _printf
from gpu import barrier
from gpu.host import DeviceContext, DeviceBuffer
from gpu.host._compile import _get_gpu_target
from gpu.id import block_idx, thread_idx
from layout import Layout, LayoutTensor
from layout._utils import ManagedLayoutTensor
from layout.fillers import arange
from layout.layout_tensor import copy_dram_to_sram, copy_sram_to_dram
from layout.tma_async import (
    TMABarrier,
    TMATensorTile,
    create_tma_tile,
    TMATensorTileArray,
)
from memory.pointer import _GPUAddressSpace
from memory import stack_allocation, UnsafePointer
from testing import assert_equal, assert_not_equal
from gpu.host._nvidia_cuda import TMADescriptor
from utils.static_tuple import StaticTuple
from utils.index import IndexList, Index
from gpu.sync import syncwarp
from gpu.memory import (
    async_copy,
    async_copy_commit_group,
    async_copy_wait_group,
)


@__llvm_metadata(`nvvm.grid_constant`=StaticTuple[Int, 1](2))
fn test_tma_replace_in_gmem_descriptor_kernel[
    dtype: DType,
    num_of_tensormaps: Int,
    src_layout: Layout,
    dst_layout: Layout,
    cta_tile_layout: Layout,
    desc_layout: Layout,
    thread_layout: Layout,
](
    dst: LayoutTensor[dtype, dst_layout, MutableAnyOrigin],
    new_src: LayoutTensor[dtype, src_layout, MutableAnyOrigin],
    template_tma_tensormap: TMATensorTile[dtype, cta_tile_layout, desc_layout],
    device_tma_tile: TMATensorTileArray[
        num_of_tensormaps, dtype, cta_tile_layout, desc_layout
    ],
):
    alias M = cta_tile_layout.shape[0].value()
    alias N = cta_tile_layout.shape[1].value()
    alias expected_bytes = cta_tile_layout.size() * sizeof[dtype]()

    tile = LayoutTensor[
        dtype,
        cta_tile_layout,
        MutableAnyOrigin,
        address_space = _GPUAddressSpace.SHARED,
        alignment=128,
    ].stack_allocation()

    device_tma_tile[block_idx.x][].tensormap_fence_acquire()
    device_tma_tile[block_idx.x][].replace_tensormap_global_address_in_gmem[
        dtype, src_layout
    ](new_src)
    device_tma_tile[block_idx.x][].tensormap_fence_release()
    mbar = TMABarrier()

    if thread_idx.x == 0:
        mbar.init()
        mbar.expect_bytes(expected_bytes)
        device_tma_tile[block_idx.x][].async_copy(
            tile, mbar, (UInt(0), UInt(0))
        )

    # Ensure all threads sees initialized mbarrier
    barrier()
    mbar.wait()

    dst_tile = dst.tile[M, N](block_idx.x, 0)
    copy_sram_to_dram[thread_layout](dst_tile, tile)


def test_tma_replace_in_gmem_descriptor[
    src_layout: Layout,
](ctx: DeviceContext):
    alias M = src_layout.shape[0].value()
    alias N = src_layout.shape[1].value()

    alias num_of_tensormaps = 4

    alias dst_layout = Layout.row_major(num_of_tensormaps * M, N)

    var old_src = ManagedLayoutTensor[DType.bfloat16, src_layout](ctx)
    var new_src = ManagedLayoutTensor[DType.bfloat16, src_layout](ctx)
    var dst = ManagedLayoutTensor[DType.bfloat16, dst_layout](ctx)

    arange(old_src.tensor(), 1)
    arange(new_src.tensor(), 1001)

    var template_tma_tensormap = create_tma_tile[
        DType.bfloat16, 2, Index(M, N)
    ](ctx, old_src.device_tensor())

    var device_tensormaps = ctx.enqueue_create_buffer[DType.uint8](
        128 * num_of_tensormaps
    )
    var tensormaps = TMATensorTileArray[
        num_of_tensormaps,
        __type_of(template_tma_tensormap).dtype,
        __type_of(template_tma_tensormap).layout,
        __type_of(template_tma_tensormap).desc_layout,
    ](ctx, device_tensormaps, template_tma_tensormap)

    ctx.synchronize()

    alias kernel = test_tma_replace_in_gmem_descriptor_kernel[
        __type_of(template_tma_tensormap).dtype,
        num_of_tensormaps,
        src_layout,  # src layout
        dst_layout,  # dst layout
        __type_of(template_tma_tensormap).layout,  # smem layout
        __type_of(template_tma_tensormap).desc_layout,  # desc layout
        __type_of(template_tma_tensormap).layout,  # thread layout
    ]

    ctx.enqueue_function[kernel](
        dst.device_tensor(),
        new_src.device_buffer(),
        template_tma_tensormap,
        tensormaps,
        grid_dim=(num_of_tensormaps),
        block_dim=(M * N),
    )

    new_src_host = new_src.tensor()
    dst_host = dst.tensor()

    for m in range(num_of_tensormaps * M):
        for n in range(N):
            if m < M and n < N:
                assert_equal(
                    new_src_host[m % M, n].cast[DType.float32](),
                    dst_host[m, n].cast[DType.float32](),
                )

    ctx.synchronize()
    _ = old_src^
    _ = new_src^
    _ = dst^


# Test loading a single 2d tile.
@__llvm_metadata(`nvvm.grid_constant`=StaticTuple[Int, 1](2))
fn test_tma_replace_in_smem_descriptor_kernel[
    dtype: DType,
    num_of_tensormaps: Int,
    src_layout: Layout,
    dst_layout: Layout,
    cta_tile_layout: Layout,
    desc_layout: Layout,
    thread_layout: Layout,
](
    dst: LayoutTensor[dtype, dst_layout, MutableAnyOrigin],
    new_src: LayoutTensor[dtype, src_layout, MutableAnyOrigin],
    template_tma_tensormap: TMATensorTile[dtype, cta_tile_layout, desc_layout],
    device_tma_tile: TMATensorTileArray[
        num_of_tensormaps, dtype, cta_tile_layout, desc_layout
    ],
):
    alias M = cta_tile_layout.shape[0].value()
    alias N = cta_tile_layout.shape[1].value()
    alias expected_bytes = cta_tile_layout.size() * sizeof[dtype]()

    tile = LayoutTensor[
        dtype,
        cta_tile_layout,
        MutableAnyOrigin,
        address_space = _GPUAddressSpace.SHARED,
        alignment=128,
    ].stack_allocation()

    var smem_desc = stack_allocation[
        1, TMADescriptor, alignment=128, address_space = _GPUAddressSpace.SHARED
    ]()

    # load the tensormap from gmem into smem. Only the one elected thread should call this
    if thread_idx.x == 0:
        template_tma_tensormap.smem_tensormap_init(smem_desc)
    async_copy_commit_group()
    async_copy_wait_group(0)

    barrier()

    device_tma_tile[block_idx.x][].tensormap_fence_acquire()

    # update the smem tensor map global addr. Only the one elected thread should call this
    if thread_idx.x == 0:
        device_tma_tile[
            block_idx.x
        ][].replace_tensormap_global_address_in_shared_mem[dtype, src_layout](
            smem_desc, new_src
        )

    # Ensure warp is converged before issuing tensormap fence release
    syncwarp()

    # Entire warp should call this as it's an aligned instruction
    device_tma_tile[block_idx.x][].tensormap_cp_fence_release(smem_desc)

    mbar = TMABarrier()

    if thread_idx.x == 0:
        mbar.init()
        mbar.expect_bytes(expected_bytes)
        device_tma_tile[block_idx.x][].async_copy(
            tile, mbar, (UInt(0), UInt(0))
        )

    # Ensure all threads sees initialized mbarrier
    barrier()
    mbar.wait()

    dst_tile = dst.tile[M, N](0, 0)
    copy_sram_to_dram[thread_layout](dst_tile, tile)


def test_tma_replace_in_smem_descriptor[
    src_layout: Layout,
](ctx: DeviceContext):
    alias M = src_layout.shape[0].value()
    alias N = src_layout.shape[1].value()

    alias num_of_tensormaps = 4
    alias dst_layout = Layout.row_major(num_of_tensormaps * M, N)

    var old_src = ManagedLayoutTensor[DType.bfloat16, src_layout](ctx)
    var new_src = ManagedLayoutTensor[DType.bfloat16, src_layout](ctx)
    var dst = ManagedLayoutTensor[DType.bfloat16, dst_layout](ctx)

    arange(old_src.tensor(), 1)
    arange(new_src.tensor(), 1001)

    var template_tma_tensormap = create_tma_tile[
        DType.bfloat16, 2, Index(M, N)
    ](ctx, old_src.device_tensor())

    var device_tensormaps = ctx.enqueue_create_buffer[DType.uint8](
        128 * num_of_tensormaps
    )
    var tensormaps = TMATensorTileArray[
        num_of_tensormaps,
        __type_of(template_tma_tensormap).dtype,
        __type_of(template_tma_tensormap).layout,
        __type_of(template_tma_tensormap).desc_layout,
    ](ctx, device_tensormaps, template_tma_tensormap)

    ctx.synchronize()

    alias kernel = test_tma_replace_in_gmem_descriptor_kernel[
        __type_of(template_tma_tensormap).dtype,
        num_of_tensormaps,
        src_layout,  # src layout
        dst_layout,  # dst layout
        __type_of(template_tma_tensormap).layout,  # smem layout
        __type_of(template_tma_tensormap).desc_layout,  # desc layout
        __type_of(template_tma_tensormap).layout,  # thread layout
    ]

    ctx.enqueue_function[kernel](
        dst.device_tensor(),
        new_src.device_buffer(),
        template_tma_tensormap,
        tensormaps,
        grid_dim=(num_of_tensormaps),
        block_dim=(M * N),
    )

    new_src_host = new_src.tensor()
    dst_host = dst.tensor()

    for m in range(num_of_tensormaps * M):
        for n in range(N):
            if m < M and n < N:
                assert_equal(
                    new_src_host[m % M, n].cast[DType.float32](),
                    dst_host[m, n].cast[DType.float32](),
                )

    ctx.synchronize()
    _ = old_src^
    _ = new_src^
    _ = dst^


def main():
    with DeviceContext() as ctx:
        print("test_tma_replace_in_gmem_descriptor")
        test_tma_replace_in_gmem_descriptor[
            src_layout = Layout.row_major(8, 8),
        ](ctx)

        print("test_tma_replace_in_smem_descriptor")
        test_tma_replace_in_smem_descriptor[
            src_layout = Layout.row_major(8, 8),
        ](ctx)
