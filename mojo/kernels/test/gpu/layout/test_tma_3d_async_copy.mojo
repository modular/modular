# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: H100-GPU
# RUN: %mojo-no-debug-no-assert %s

from math import align_up, ceildiv
from sys import sizeof
from layout.swizzle import make_swizzle
from builtin.io import _printf
from gpu import barrier
from gpu.host import DeviceContext
from gpu.host._compile import _get_gpu_target
from gpu.id import block_idx, thread_idx, grid_dim
from gpu.memory import tma_store_fence
from gpu.sync import cp_async_bulk_commit_group, cp_async_bulk_wait_group
from layout import Layout, LayoutTensor, IntTuple
from layout._utils import ManagedLayoutTensor
from layout._fillers import arange
from layout.layout_tensor import copy_dram_to_sram, copy_sram_to_dram
from layout.tma_async import TMABarrier, TMATensorTile, create_tma_tile
from memory.pointer import _GPUAddressSpace
from testing import assert_equal, assert_not_equal
from layout.runtime_layout import RuntimeLayout
from utils.index import Index, IndexList
from utils.static_tuple import StaticTuple
from gpu.host._nvidia_cuda import TensorMapSwizzle


# Test loading a single 2d tile.
@__llvm_arg_metadata(tma_tile, `nvvm.grid_constant`)
fn test_tma_3d_load_kernel[
    dtype: DType,
    dst_layout: Layout,
    cta_tile_layout: Layout,
    desc_layout: Layout,
    smem_layout: Layout,
](
    dst: LayoutTensor[dtype, dst_layout, MutableAnyOrigin],
    tma_tile: TMATensorTile[dtype, cta_tile_layout, desc_layout],
):
    constrained[
        cta_tile_layout.size() == smem_layout.size(),
        "CTA Tile and SMEM tile should be the same size",
    ]()

    constrained[
        cta_tile_layout == smem_layout,
        "for these test cases cta and smem should have the same size",
    ]()

    alias dst_dim0 = dst_layout.shape[0].value()
    alias dst_dim1 = dst_layout.shape[1].value()

    alias cta_tile_dim0 = cta_tile_layout.shape[0].value()
    alias cta_tile_dim1 = cta_tile_layout.shape[1].value()
    alias cta_tile_dim2 = cta_tile_layout.shape[2].value()

    constrained[
        dst_dim1 == cta_tile_dim2,
        "dst and cta should have the same last dimension for these test cases",
    ]()

    smem_tile = LayoutTensor[
        dtype,
        smem_layout,
        MutableAnyOrigin,
        address_space = _GPUAddressSpace.SHARED,
        alignment=128,
    ].stack_allocation()

    alias expected_bytes = cta_tile_layout.size() * sizeof[dtype]()

    mbar = TMABarrier()

    if thread_idx.x == 0:
        mbar.init()
        mbar.expect_bytes(expected_bytes)
        tma_tile.async_copy_3d(
            smem_tile,
            mbar,
            (
                block_idx.x * cta_tile_dim2,
                block_idx.y * cta_tile_dim1,
                block_idx.z * cta_tile_dim0,
            ),
        )
    # Ensure all threads sees initialized mbarrier
    barrier()
    mbar.wait()

    alias smem_dim0 = smem_layout.shape[0].value()
    alias smem_dim1 = smem_layout.shape[1].value()
    alias smem_dim2 = smem_layout.shape[2].value()

    var idx = block_idx.z * grid_dim.x * grid_dim.y + block_idx.y * grid_dim.x + block_idx.x
    for i in range(cta_tile_dim0):
        smem_tile_i = smem_tile.tile[1, cta_tile_dim1, cta_tile_dim2](i)

        dst_tile = dst.tile[cta_tile_dim1, cta_tile_dim2](
            idx * cta_tile_dim0 + i, 0
        )
        if thread_idx.x == 0:
            dst_tile.copy_from(smem_tile_i)


def test_tma_3d_load_row_major[
    dtype: DType,
    src_layout: Layout,
    cta_tile_layout: Layout,
    smem_tile_layout: Layout,
    swizzle_mode: TensorMapSwizzle,
](ctx: DeviceContext):
    print("test_tma_3d_load")

    alias src_dim0 = src_layout.shape[0].value()
    alias src_dim1 = src_layout.shape[1].value()
    alias src_dim2 = src_layout.shape[2].value()

    alias cta_tile_dim0 = cta_tile_layout.shape[0].value()
    alias cta_tile_dim1 = cta_tile_layout.shape[1].value()
    alias cta_tile_dim2 = cta_tile_layout.shape[2].value()

    alias dst_layout = Layout.row_major(
        src_dim0 * src_dim1 * src_dim2 // cta_tile_dim2, cta_tile_dim2
    )

    var src = ManagedLayoutTensor[dtype, src_layout](ctx)
    var dst = ManagedLayoutTensor[dtype, dst_layout](ctx)

    arange(src.tensor(), 1)

    tma_tensor = create_tma_tile[
        dtype,
        3,
        Index(cta_tile_dim0, cta_tile_dim1, cta_tile_dim2),
        swizzle_mode=swizzle_mode,
        __tile_layout=cta_tile_layout,
    ](ctx, src.device_tensor())

    ctx.synchronize()

    print("src layout:", src_layout)
    print("cta tile layout:", cta_tile_layout)
    print("desc layout:", __type_of(tma_tensor).desc_layout)

    alias kernel = test_tma_3d_load_kernel[
        __type_of(tma_tensor).dtype,
        dst_layout,  # dst layout
        __type_of(tma_tensor).layout,  # cta_tile
        __type_of(tma_tensor).desc_layout,  # desc_tile
        smem_tile_layout,  # smem layout
    ]
    ctx.enqueue_function[kernel](
        dst.device_tensor(),
        tma_tensor,
        grid_dim=(
            src_dim2 // cta_tile_dim2,
            src_dim1 // cta_tile_dim1,
            src_dim0 // cta_tile_dim0,
        ),
        block_dim=(1),
    )

    src_host = src.tensor()
    dst_host = dst.tensor()

    alias swizzle = make_swizzle[dtype, swizzle_mode]()

    alias cta_tile_size = cta_tile_layout.size()

    alias desc_tile_dim0 = __type_of(tma_tensor).desc_layout.shape[0].value()
    alias desc_tile_dim1 = __type_of(tma_tensor).desc_layout.shape[1].value()
    alias desc_tile_dim2 = __type_of(tma_tensor).desc_layout.shape[2].value()

    alias desc_tile_size = desc_tile_dim1 * desc_tile_dim2

    desc_tile = LayoutTensor[
        dtype,
        Layout.row_major(desc_tile_dim1, desc_tile_dim2),
        MutableAnyOrigin,
    ].stack_allocation()

    var dest_ptr = dst_host.ptr
    for dest_tile_z in range(src_dim0 // cta_tile_dim0):
        for dest_tile_y in range(src_dim1 // cta_tile_dim1):
            for dest_tile_x in range(src_dim2 // cta_tile_dim2):
                for x in range(cta_tile_dim2 // desc_tile_dim2):
                    for y in range(cta_tile_dim1 // desc_tile_dim1):
                        for z in range(cta_tile_dim0):
                            var src_tile = src_host.tile[
                                1, desc_tile_dim1, desc_tile_dim2
                            ](
                                dest_tile_z * cta_tile_dim0 + z,
                                dest_tile_y + y,
                                dest_tile_x + x,
                            )

                            desc_tile.copy_from(src_tile)

                            for i in range(desc_tile_size):
                                desc_idx = swizzle(i)
                                assert_equal(
                                    desc_tile.ptr[desc_idx], dest_ptr[i]
                                )

                            dest_ptr += desc_tile_size

    _ = src^
    _ = dst^


def main():
    with DeviceContext() as ctx:
        test_tma_3d_load_row_major[
            DType.bfloat16,
            src_layout = Layout(
                IntTuple(4, 8, 8),
                IntTuple(64, 8, 1),
            ),
            cta_tile_layout = Layout(
                IntTuple(1, 8, 8),
                IntTuple(64, 8, 1),
            ),
            smem_tile_layout = Layout(
                IntTuple(1, 8, 8),
                IntTuple(64, 8, 1),
            ),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE,
        ](ctx)

        test_tma_3d_load_row_major[
            DType.bfloat16,
            src_layout = Layout(
                IntTuple(4, 16, 8),
                IntTuple(128, 8, 1),
            ),
            cta_tile_layout = Layout(
                IntTuple(2, 8, 8),
                IntTuple(64, 8, 1),
            ),
            smem_tile_layout = Layout(
                IntTuple(2, 8, 8),
                IntTuple(64, 8, 1),
            ),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE,
        ](ctx)

        test_tma_3d_load_row_major[
            DType.bfloat16,
            src_layout = Layout(
                IntTuple(4, 32, 8),
                IntTuple(256, 8, 1),
            ),
            cta_tile_layout = Layout(
                IntTuple(2, 32, 8),
                IntTuple(256, 8, 1),
            ),
            smem_tile_layout = Layout(
                IntTuple(2, 32, 8),
                IntTuple(256, 8, 1),
            ),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE,
        ](ctx)

        test_tma_3d_load_row_major[
            DType.bfloat16,
            src_layout = Layout(
                IntTuple(4, 16, 16),
                IntTuple(256, 16, 1),
            ),
            cta_tile_layout = Layout(
                IntTuple(2, 8, 8),
                IntTuple(64, 8, 1),
            ),
            smem_tile_layout = Layout(
                IntTuple(2, 8, 8),
                IntTuple(64, 8, 1),
            ),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE,
        ](ctx)

        test_tma_3d_load_row_major[
            DType.bfloat16,
            src_layout = Layout(
                IntTuple(4, 16, 16),
                IntTuple(256, 16, 1),
            ),
            cta_tile_layout = Layout(
                IntTuple(2, 8, 16),
                IntTuple(128, 16, 1),
            ),
            smem_tile_layout = Layout(
                IntTuple(2, 8, 16),
                IntTuple(128, 16, 1),
            ),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE,
        ](ctx)

        test_tma_3d_load_row_major[
            DType.bfloat16,
            src_layout = Layout(
                IntTuple(4, 8, 16),
                IntTuple(128, 16, 1),
            ),
            cta_tile_layout = Layout(
                IntTuple(2, 8, 16),
                IntTuple(128, 16, 1),
            ),
            smem_tile_layout = Layout(
                IntTuple(2, 8, 16),
                IntTuple(128, 16, 1),
            ),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE,
        ](ctx)

        test_tma_3d_load_row_major[
            DType.bfloat16,
            src_layout = Layout(
                IntTuple(4, 16, 64),
                IntTuple(1024, 64, 1),
            ),
            cta_tile_layout = Layout(
                IntTuple(2, 16, 64),
                IntTuple(1024, 64, 1),
            ),
            smem_tile_layout = Layout(
                IntTuple(2, 16, 64),
                IntTuple(1024, 64, 1),
            ),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)

        test_tma_3d_load_row_major[
            DType.bfloat16,
            src_layout = Layout(
                IntTuple(4, 16, 64),
                IntTuple(1024, 64, 1),
            ),
            cta_tile_layout = Layout(
                IntTuple(2, 8, 64),
                IntTuple(512, 64, 1),
            ),
            smem_tile_layout = Layout(
                IntTuple(2, 8, 64),
                IntTuple(512, 64, 1),
            ),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)

        test_tma_3d_load_row_major[
            DType.bfloat16,
            src_layout = Layout(
                IntTuple(4, 32, 64),
                IntTuple(2048, 64, 1),
            ),
            cta_tile_layout = Layout(
                IntTuple(2, 32, 64),
                IntTuple(2048, 64, 1),
            ),
            smem_tile_layout = Layout(
                IntTuple(2, 32, 64),
                IntTuple(2048, 64, 1),
            ),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)

        test_tma_3d_load_row_major[
            DType.bfloat16,
            src_layout = Layout(
                IntTuple(4, 8, 128),
                IntTuple(1024, 128, 1),
            ),
            cta_tile_layout = Layout(
                IntTuple(2, 8, 128),
                IntTuple(1024, 128, 1),
            ),
            smem_tile_layout = Layout(
                IntTuple(2, 8, 128),
                IntTuple(1024, 128, 1),
            ),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)

        test_tma_3d_load_row_major[
            DType.bfloat16,
            src_layout = Layout(
                IntTuple(4, 16, 32),
                IntTuple(512, 32, 1),
            ),
            cta_tile_layout = Layout(
                IntTuple(2, 16, 32),
                IntTuple(512, 32, 1),
            ),
            smem_tile_layout = Layout(
                IntTuple(2, 16, 32),
                IntTuple(512, 32, 1),
            ),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_64B,
        ](ctx)

        test_tma_3d_load_row_major[
            DType.bfloat16,
            src_layout = Layout(
                IntTuple(4, 16, 32),
                IntTuple(512, 32, 1),
            ),
            cta_tile_layout = Layout(
                IntTuple(2, 8, 32),
                IntTuple(256, 32, 1),
            ),
            smem_tile_layout = Layout(
                IntTuple(2, 8, 32),
                IntTuple(256, 32, 1),
            ),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_64B,
        ](ctx)

        test_tma_3d_load_row_major[
            DType.bfloat16,
            src_layout = Layout(
                IntTuple(4, 32, 32),
                IntTuple(1024, 32, 1),
            ),
            cta_tile_layout = Layout(
                IntTuple(2, 32, 32),
                IntTuple(1024, 32, 1),
            ),
            smem_tile_layout = Layout(
                IntTuple(2, 32, 32),
                IntTuple(1024, 32, 1),
            ),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_64B,
        ](ctx)

        test_tma_3d_load_row_major[
            DType.bfloat16,
            src_layout = Layout(
                IntTuple(4, 8, 64),
                IntTuple(512, 64, 1),
            ),
            cta_tile_layout = Layout(
                IntTuple(2, 8, 64),
                IntTuple(512, 64, 1),
            ),
            smem_tile_layout = Layout(
                IntTuple(2, 8, 64),
                IntTuple(512, 64, 1),
            ),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_64B,
        ](ctx)

        test_tma_3d_load_row_major[
            DType.bfloat16,
            src_layout = Layout(
                IntTuple(4, 16, 16),
                IntTuple(256, 16, 1),
            ),
            cta_tile_layout = Layout(
                IntTuple(2, 16, 16),
                IntTuple(256, 16, 1),
            ),
            smem_tile_layout = Layout(
                IntTuple(2, 16, 16),
                IntTuple(256, 16, 1),
            ),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_32B,
        ](ctx)

        test_tma_3d_load_row_major[
            DType.bfloat16,
            src_layout = Layout(
                IntTuple(4, 16, 16),
                IntTuple(256, 16, 1),
            ),
            cta_tile_layout = Layout(
                IntTuple(2, 8, 16),
                IntTuple(128, 16, 1),
            ),
            smem_tile_layout = Layout(
                IntTuple(2, 8, 16),
                IntTuple(128, 16, 1),
            ),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_32B,
        ](ctx)

        test_tma_3d_load_row_major[
            DType.bfloat16,
            src_layout = Layout(
                IntTuple(4, 32, 16),
                IntTuple(512, 16, 1),
            ),
            cta_tile_layout = Layout(
                IntTuple(2, 32, 16),
                IntTuple(512, 16, 1),
            ),
            smem_tile_layout = Layout(
                IntTuple(2, 32, 16),
                IntTuple(512, 16, 1),
            ),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_32B,
        ](ctx)

        test_tma_3d_load_row_major[
            DType.bfloat16,
            src_layout = Layout(
                IntTuple(4, 8, 32),
                IntTuple(256, 32, 1),
            ),
            cta_tile_layout = Layout(
                IntTuple(2, 8, 32),
                IntTuple(256, 32, 1),
            ),
            smem_tile_layout = Layout(
                IntTuple(2, 8, 32),
                IntTuple(256, 32, 1),
            ),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_32B,
        ](ctx)
