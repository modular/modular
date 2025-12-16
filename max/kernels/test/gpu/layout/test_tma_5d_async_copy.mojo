# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from sys import size_of

from gpu import barrier
from gpu.host import DeviceContext
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu import block_idx, grid_dim, thread_idx
from layout import IntTuple, Layout, LayoutTensor
from layout._fillers import arange
from layout._utils import ManagedLayoutTensor
from layout.swizzle import make_swizzle
from layout.tma_async import SharedMemBarrier, TMATensorTile, create_tma_tile
from memory import stack_allocation
from testing import assert_equal

from utils.index import Index


# Test loading a single 5d tile.
@__llvm_arg_metadata(tma_tile, `nvvm.grid_constant`)
fn test_tma_5d_load_kernel[
    dtype: DType,
    dst_layout: Layout,
    cta_tile_layout: Layout,
    desc_layout: Layout,
    smem_layout: Layout,
    grid_dim1: Int,
    grid_dim2: Int,
](
    dst: LayoutTensor[dtype, dst_layout, MutAnyOrigin],
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

    comptime dst_dim0 = dst_layout.shape[0].value()
    comptime dst_dim1 = dst_layout.shape[1].value()

    comptime cta_tile_dim0 = cta_tile_layout.shape[0].value()
    comptime cta_tile_dim1 = cta_tile_layout.shape[1].value()
    comptime cta_tile_dim2 = cta_tile_layout.shape[2].value()
    comptime cta_tile_dim3 = cta_tile_layout.shape[3].value()
    comptime cta_tile_dim4 = cta_tile_layout.shape[4].value()

    constrained[
        dst_dim1 == cta_tile_dim4,
        "dst and cta should have the same last dimension for these test cases",
    ]()

    smem_tile = LayoutTensor[
        dtype,
        smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ].stack_allocation()

    comptime expected_bytes = cta_tile_layout.size() * size_of[dtype]()

    mbar = stack_allocation[
        1,
        SharedMemBarrier,
        address_space = AddressSpace.SHARED,
        alignment=8,
    ]()
    idx0 = Int(block_idx.z) // (grid_dim1 * grid_dim2)
    idx1 = (Int(block_idx.z) // grid_dim2) % grid_dim1
    idx2 = Int(block_idx.z) % grid_dim2
    idx3 = Int(block_idx.y)
    idx4 = Int(block_idx.x)

    if thread_idx.x == 0:
        mbar[0].init()
        mbar[0].expect_bytes(expected_bytes)
        tma_tile.async_copy_5d(
            smem_tile,
            mbar[0],
            (
                UInt(idx4 * cta_tile_dim4),
                UInt(idx3 * cta_tile_dim3),
                UInt(idx2 * cta_tile_dim2),
                UInt(idx1 * cta_tile_dim1),
                UInt(idx0 * cta_tile_dim0),
            ),
        )
    # Ensure all threads see initialized mbarrier
    barrier()
    mbar[0].wait()

    comptime smem_dim0 = smem_layout.shape[0].value()
    comptime smem_dim1 = smem_layout.shape[1].value()
    comptime smem_dim2 = smem_layout.shape[2].value()
    comptime smem_dim3 = smem_layout.shape[3].value()
    comptime smem_dim4 = smem_layout.shape[4].value()

    var idx = (
        block_idx.z * grid_dim.y + block_idx.y
    ) * grid_dim.x + block_idx.x
    comptime dst_tile_layout = Layout.row_major(
        cta_tile_dim1, cta_tile_dim2, cta_tile_dim3, cta_tile_dim4
    )
    comptime dst_tile_size = dst_tile_layout.size()
    comptime DstTileType = LayoutTensor[dtype, dst_tile_layout, MutAnyOrigin]
    comptime cta_tile_size = cta_tile_layout.size()

    local_dst_ptr = dst.ptr + idx * UInt(cta_tile_size)

    for i in range(cta_tile_dim0):
        smem_tile_i = smem_tile.tile[
            1, cta_tile_dim1, cta_tile_dim2, cta_tile_dim3, cta_tile_dim4
        ](i)

        dst_tile = DstTileType(local_dst_ptr + i * dst_tile_size)
        if thread_idx.x == 0:
            dst_tile.copy_from(smem_tile_i)


def test_tma_5d_load_row_major[
    dtype: DType,
    src_layout: Layout,
    cta_tile_layout: Layout,
    smem_tile_layout: Layout,
    swizzle_mode: TensorMapSwizzle,
](ctx: DeviceContext):
    print("test_tma_5d_load")

    comptime src_dim0 = src_layout.shape[0].value()
    comptime src_dim1 = src_layout.shape[1].value()
    comptime src_dim2 = src_layout.shape[2].value()
    comptime src_dim3 = src_layout.shape[3].value()
    comptime src_dim4 = src_layout.shape[4].value()

    comptime cta_tile_dim0 = cta_tile_layout.shape[0].value()
    comptime cta_tile_dim1 = cta_tile_layout.shape[1].value()
    comptime cta_tile_dim2 = cta_tile_layout.shape[2].value()
    comptime cta_tile_dim3 = cta_tile_layout.shape[3].value()
    comptime cta_tile_dim4 = cta_tile_layout.shape[4].value()

    comptime dst_layout = Layout.row_major(
        src_dim0 * src_dim1 * src_dim2 * src_dim3 * src_dim4 // cta_tile_dim4,
        cta_tile_dim4,
    )

    var src = ManagedLayoutTensor[dtype, src_layout](ctx)
    var dst = ManagedLayoutTensor[dtype, dst_layout](ctx)

    arange(src.tensor(), start=0, step=0.015625)

    tma_tensor = create_tma_tile[
        Index(
            cta_tile_dim0,
            cta_tile_dim1,
            cta_tile_dim2,
            cta_tile_dim3,
            cta_tile_dim4,
        ),
        swizzle_mode=swizzle_mode,
        __tile_layout=cta_tile_layout,
    ](ctx, src.device_tensor())

    ctx.synchronize()

    print("src layout:", src_layout)
    print("cta tile layout:", cta_tile_layout)
    print("desc layout:", type_of(tma_tensor).desc_layout)

    comptime kernel = test_tma_5d_load_kernel[
        type_of(tma_tensor).dtype,
        dst_layout,  # dst layout
        type_of(tma_tensor).layout,  # cta_tile
        type_of(tma_tensor).desc_layout,  # desc_tile
        smem_tile_layout,  # smem layout
        grid_dim1 = src_dim1 // cta_tile_dim1,
        grid_dim2 = src_dim2 // cta_tile_dim2,
    ]
    ctx.enqueue_function_checked[kernel, kernel](
        dst.device_tensor(),
        tma_tensor,
        grid_dim=(
            src_dim4 // cta_tile_dim4,
            src_dim3 // cta_tile_dim3,
            (src_dim2 // cta_tile_dim2)
            * (src_dim1 // cta_tile_dim1)
            * (src_dim0 // cta_tile_dim0),
        ),
        block_dim=(1),
    )

    src_host = src.tensor()
    dst_host = dst.tensor()

    comptime swizzle = make_swizzle[dtype, swizzle_mode]()

    comptime cta_tile_size = cta_tile_layout.size()

    comptime desc_tile_dim0 = type_of(tma_tensor).desc_layout.shape[0].value()
    comptime desc_tile_dim1 = type_of(tma_tensor).desc_layout.shape[1].value()
    comptime desc_tile_dim2 = type_of(tma_tensor).desc_layout.shape[2].value()
    comptime desc_tile_dim3 = type_of(tma_tensor).desc_layout.shape[3].value()
    comptime desc_tile_dim4 = type_of(tma_tensor).desc_layout.shape[4].value()

    comptime desc_tile_size = (
        desc_tile_dim1 * desc_tile_dim2 * desc_tile_dim3 * desc_tile_dim4
    )

    desc_tile = LayoutTensor[
        dtype,
        Layout.row_major(
            desc_tile_dim1, desc_tile_dim2, desc_tile_dim3, desc_tile_dim4
        ),
        MutAnyOrigin,
    ].stack_allocation()

    var dest_ptr = dst_host.ptr
    for dest_tile_v in range(src_dim0 // cta_tile_dim0):
        for dest_tile_w in range(src_dim1 // cta_tile_dim1):
            for dest_tile_z in range(src_dim2 // cta_tile_dim2):
                for dest_tile_y in range(src_dim3 // cta_tile_dim3):
                    for dest_tile_x in range(src_dim4 // cta_tile_dim4):
                        for x in range(cta_tile_dim4 // desc_tile_dim4):
                            for y in range(cta_tile_dim3 // desc_tile_dim3):
                                for z in range(cta_tile_dim2 // desc_tile_dim2):
                                    for w in range(
                                        cta_tile_dim1 // desc_tile_dim1
                                    ):
                                        for v in range(cta_tile_dim0):
                                            var src_tile = src_host.tile[
                                                1,
                                                1,
                                                desc_tile_dim2,
                                                desc_tile_dim3,
                                                desc_tile_dim4,
                                            ](
                                                dest_tile_v * cta_tile_dim0 + v,
                                                dest_tile_w + w,
                                                dest_tile_z + z,
                                                dest_tile_y + y,
                                                dest_tile_x + x,
                                            )

                                            desc_tile.copy_from(src_tile)

                                            for i in range(desc_tile_size):
                                                desc_idx = swizzle(i)
                                                assert_equal(
                                                    desc_tile.ptr[desc_idx],
                                                    dest_ptr[i],
                                                )

                                            dest_ptr += desc_tile_size

    _ = src^
    _ = dst^


def main():
    with DeviceContext() as ctx:
        # Basic 5D test with no swizzling
        test_tma_5d_load_row_major[
            DType.bfloat16,
            src_layout = Layout(
                IntTuple(2, 2, 4, 8, 8),
                IntTuple(512, 256, 64, 8, 1),
            ),
            cta_tile_layout = Layout(
                IntTuple(1, 1, 4, 8, 8),
                IntTuple(256, 256, 64, 8, 1),
            ),
            smem_tile_layout = Layout(
                IntTuple(1, 1, 4, 8, 8),
                IntTuple(256, 256, 64, 8, 1),
            ),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE,
        ](ctx)

        # 5D with larger dimensions
        test_tma_5d_load_row_major[
            DType.bfloat16,
            src_layout = Layout(
                IntTuple(2, 2, 4, 8, 16),
                IntTuple(1024, 512, 128, 16, 1),
            ),
            cta_tile_layout = Layout(
                IntTuple(1, 1, 2, 8, 16),
                IntTuple(256, 256, 128, 16, 1),
            ),
            smem_tile_layout = Layout(
                IntTuple(1, 1, 2, 8, 16),
                IntTuple(256, 256, 128, 16, 1),
            ),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE,
        ](ctx)

        # 5D with 128B swizzling
        test_tma_5d_load_row_major[
            DType.bfloat16,
            src_layout = Layout(
                IntTuple(2, 2, 4, 8, 64),
                IntTuple(4096, 2048, 512, 64, 1),
            ),
            cta_tile_layout = Layout(
                IntTuple(1, 1, 2, 8, 64),
                IntTuple(1024, 1024, 512, 64, 1),
            ),
            smem_tile_layout = Layout(
                IntTuple(1, 1, 2, 8, 64),
                IntTuple(1024, 1024, 512, 64, 1),
            ),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)

        # 5D with 64B swizzling
        test_tma_5d_load_row_major[
            DType.bfloat16,
            src_layout = Layout(
                IntTuple(2, 2, 4, 8, 32),
                IntTuple(2048, 1024, 256, 32, 1),
            ),
            cta_tile_layout = Layout(
                IntTuple(1, 1, 2, 8, 32),
                IntTuple(512, 512, 256, 32, 1),
            ),
            smem_tile_layout = Layout(
                IntTuple(1, 1, 2, 8, 32),
                IntTuple(512, 512, 256, 32, 1),
            ),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_64B,
        ](ctx)

        # 5D with 32B swizzling
        test_tma_5d_load_row_major[
            DType.bfloat16,
            src_layout = Layout(
                IntTuple(2, 2, 4, 8, 16),
                IntTuple(1024, 512, 128, 16, 1),
            ),
            cta_tile_layout = Layout(
                IntTuple(1, 1, 2, 8, 16),
                IntTuple(256, 256, 128, 16, 1),
            ),
            smem_tile_layout = Layout(
                IntTuple(1, 1, 2, 8, 16),
                IntTuple(256, 256, 128, 16, 1),
            ),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_32B,
        ](ctx)
