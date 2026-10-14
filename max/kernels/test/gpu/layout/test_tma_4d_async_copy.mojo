# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

from std.sys import align_of, size_of

from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu import block_idx, grid_dim, thread_idx
from layout import Coord, Idx, IntTuple, Layout, TileTensor, coord, row_major
from layout.tile_tensor import stack_allocation
from layout._fillers import arange
from layout._host_device_tile_tensor import HostDeviceTileTensor
from layout.swizzle import make_swizzle
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    _idx_product,
    create_tensor_tile,
)
from std.memory import unsafe_stack_allocation
from std.testing import assert_equal


# Test loading a single 4d tile.
@__llvm_arg_metadata(tma_tile, `nvvm.grid_constant`)
def test_tma_4d_load_kernel[
    dtype: DType,
    dst_layout: Layout,
    cta_tile_shape: Coord,
    desc_shape: Coord,
    smem_layout: Layout,
    grid_dim1: Int,
](
    dst: TileTensor[
        dtype,
        type_of(
            row_major[
                dst_layout.shape[0].value(), dst_layout.shape[1].value()
            ]()
        ),
        MutAnyOrigin,
    ],
    tma_tile: TMATensorTile[dtype, cta_tile_shape, desc_shape],
):
    comptime assert (
        _idx_product[cta_tile_shape]() == smem_layout.size()
    ), "CTA Tile and SMEM tile should be the same size"

    comptime dst_dim0 = dst_layout.shape[0].value()
    comptime dst_dim1 = dst_layout.shape[1].value()

    comptime cta_tile_dim0 = Int(cta_tile_shape[0].value())
    comptime cta_tile_dim1 = Int(cta_tile_shape[1].value())
    comptime cta_tile_dim2 = Int(cta_tile_shape[2].value())
    comptime cta_tile_dim3 = Int(cta_tile_shape[3].value())

    comptime assert (
        dst_dim1 == cta_tile_dim3
    ), "dst and cta should have the same last dimension for these test cases"

    comptime assert smem_layout == Layout.row_major(smem_layout.shape)
    var smem_tile = stack_allocation[
        dtype=dtype, address_space=.SHARED, alignment=128
    ](
        row_major[
            smem_layout.shape[0].value(),
            smem_layout.shape[1].value(),
            smem_layout.shape[2].value(),
            smem_layout.shape[3].value(),
        ]()
    )

    comptime cta_tile_size = _idx_product[cta_tile_shape]()
    comptime expected_bytes = cta_tile_size * size_of[dtype]()

    var mbar = unsafe_stack_allocation[
        1,
        SharedMemBarrier,
        address_space=.SHARED,
        alignment=8,
    ]()
    var idx0, idx1 = divmod(block_idx.z, grid_dim1)
    var idx2 = block_idx.y
    var idx3 = block_idx.x

    if thread_idx.x == 0:
        mbar[0].init()
        mbar[0].expect_bytes(Int32(expected_bytes))
        tma_tile.async_copy_4d(
            smem_tile,
            mbar[0],
            (
                idx3 * cta_tile_dim3,
                idx2 * cta_tile_dim2,
                idx1 * cta_tile_dim1,
                idx0 * cta_tile_dim0,
            ),
        )
    # Ensure all threads see initialized mbarrier
    barrier()
    mbar[0].wait()

    var idx = (
        block_idx.z * grid_dim.y + block_idx.y
    ) * grid_dim.x + block_idx.x
    comptime dst_tile_layout = Layout.row_major(
        cta_tile_dim1, cta_tile_dim2, cta_tile_dim3
    )
    comptime dst_tile_size = dst_tile_layout.size()

    for i in range(cta_tile_dim0):
        var smem_tile_i = smem_tile.tile[
            1, cta_tile_dim1, cta_tile_dim2, cta_tile_dim3
        ](Coord(i, 0, 0, 0))

        var dst_tile = dst.tile[dst_tile_size // dst_dim1, dst_dim1](
            Coord(idx * cta_tile_dim0 + i, 0)
        ).reshape(
            Coord(Idx[cta_tile_dim1], Idx[cta_tile_dim2], Idx[cta_tile_dim3])
        )
        if thread_idx.x == 0:
            dst_tile.copy_from(smem_tile_i)


def test_tma_4d_load_row_major[
    dtype: DType,
    src_layout: Layout,
    cta_tile_layout: Layout,
    smem_tile_layout: Layout,
    swizzle_mode: TensorMapSwizzle,
](ctx: DeviceContext) raises:
    print("test_tma_4d_load")

    comptime src_dim0 = src_layout.shape[0].value()
    comptime src_dim1 = src_layout.shape[1].value()
    comptime src_dim2 = src_layout.shape[2].value()
    comptime src_dim3 = src_layout.shape[3].value()

    comptime cta_tile_dim0 = cta_tile_layout.shape[0].value()
    comptime cta_tile_dim1 = cta_tile_layout.shape[1].value()
    comptime cta_tile_dim2 = cta_tile_layout.shape[2].value()
    comptime cta_tile_dim3 = cta_tile_layout.shape[3].value()

    comptime dst_layout = Layout.row_major(
        src_dim0 * src_dim1 * src_dim2 * src_dim3 // cta_tile_dim3,
        cta_tile_dim3,
    )

    comptime assert src_layout == Layout.row_major(src_layout.shape)
    var src = HostDeviceTileTensor[dtype](
        row_major[src_dim0, src_dim1, src_dim2, src_dim3](), ctx
    )
    var dst = HostDeviceTileTensor[dtype](
        row_major[dst_layout.shape[0].value(), dst_layout.shape[1].value()](),
        ctx,
    )

    arange(src.host_tensor(), start=0, step=0.015625)
    src.to_device()

    var tma_tensor = create_tensor_tile[
        coord[cta_tile_dim0, cta_tile_dim1, cta_tile_dim2, cta_tile_dim3],
        swizzle_mode=swizzle_mode,
    ](ctx, src.device_tensor())

    ctx.synchronize()

    print("src layout:", materialize[src_layout]())
    print("cta tile layout:", materialize[cta_tile_layout]())
    print("desc shape:", type_of(tma_tensor).desc_shape)

    comptime kernel = test_tma_4d_load_kernel[
        type_of(tma_tensor).dtype,
        dst_layout,  # dst layout
        type_of(tma_tensor).tile_shape,  # cta_tile
        type_of(tma_tensor).desc_shape,  # desc_tile
        smem_tile_layout,  # smem layout
        grid_dim1=src_dim1 // cta_tile_dim1,
    ]
    ctx.enqueue_function[kernel](
        dst.device_tensor().as_unsafe_any_origin(),
        tma_tensor,
        grid_dim=(
            src_dim3 // cta_tile_dim3,
            src_dim2 // cta_tile_dim2,
            (src_dim1 // cta_tile_dim1) * (src_dim0 // cta_tile_dim0),
        ),
        block_dim=(1),
    )

    dst.to_host()
    var src_host = src.host_tensor()
    var dst_host = dst.host_tensor()

    comptime swizzle = make_swizzle[dtype, swizzle_mode]()

    comptime cta_tile_size = cta_tile_layout.size()

    comptime desc_tile_dim0 = Int(
        type_of(tma_tensor).desc_shape.element_types[0].static_value.value()
    )
    comptime desc_tile_dim1 = Int(
        type_of(tma_tensor).desc_shape.element_types[1].static_value.value()
    )
    comptime desc_tile_dim2 = Int(
        type_of(tma_tensor).desc_shape.element_types[2].static_value.value()
    )
    comptime desc_tile_dim3 = Int(
        type_of(tma_tensor).desc_shape.element_types[3].static_value.value()
    )

    comptime desc_tile_size = desc_tile_dim1 * desc_tile_dim2 * desc_tile_dim3

    var desc_tile = stack_allocation[dtype=dtype, alignment=align_of[dtype]()](
        row_major[desc_tile_dim1, desc_tile_dim2, desc_tile_dim3]()
    )

    var dest_ptr = dst_host.ptr
    for dest_tile_w in range(src_dim0 // cta_tile_dim0):
        for dest_tile_z in range(src_dim1 // cta_tile_dim1):
            for dest_tile_y in range(src_dim2 // cta_tile_dim2):
                for dest_tile_x in range(src_dim3 // cta_tile_dim3):
                    for x in range(cta_tile_dim3 // desc_tile_dim3):
                        for y in range(cta_tile_dim2 // desc_tile_dim2):
                            for z in range(cta_tile_dim1 // desc_tile_dim1):
                                for w in range(cta_tile_dim0):
                                    var src_tile = src_host.tile[
                                        1,
                                        desc_tile_dim1,
                                        desc_tile_dim2,
                                        desc_tile_dim3,
                                    ](
                                        Coord(
                                            dest_tile_w * cta_tile_dim0 + w,
                                            dest_tile_z + z,
                                            dest_tile_y + y,
                                            dest_tile_x + x,
                                        ),
                                    )

                                    desc_tile.copy_from(src_tile)

                                    for i in range(desc_tile_size):
                                        var desc_idx = swizzle(i)
                                        assert_equal(
                                            desc_tile.ptr[desc_idx], dest_ptr[i]
                                        )

                                    dest_ptr += desc_tile_size

    _ = src^
    _ = dst^


def main() raises:
    with DeviceContext() as ctx:
        # Basic 4D test with no swizzling
        test_tma_4d_load_row_major[
            DType.bfloat16,
            src_layout=Layout(
                IntTuple(2, 4, 8, 8),
                IntTuple(256, 64, 8, 1),
            ),
            cta_tile_layout=Layout(
                IntTuple(1, 4, 8, 8),
                IntTuple(256, 64, 8, 1),
            ),
            smem_tile_layout=Layout(
                IntTuple(1, 4, 8, 8),
                IntTuple(256, 64, 8, 1),
            ),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
        ](ctx)

        # 4D with larger dimensions
        test_tma_4d_load_row_major[
            DType.bfloat16,
            src_layout=Layout(
                IntTuple(2, 4, 16, 16),
                IntTuple(1024, 256, 16, 1),
            ),
            cta_tile_layout=Layout(
                IntTuple(1, 2, 8, 16),
                IntTuple(256, 128, 16, 1),
            ),
            smem_tile_layout=Layout(
                IntTuple(1, 2, 8, 16),
                IntTuple(256, 128, 16, 1),
            ),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
        ](ctx)

        # 4D with 128B swizzling
        test_tma_4d_load_row_major[
            DType.bfloat16,
            src_layout=Layout(
                IntTuple(2, 4, 16, 64),
                IntTuple(4096, 1024, 64, 1),
            ),
            cta_tile_layout=Layout(
                IntTuple(1, 2, 16, 64),
                IntTuple(2048, 1024, 64, 1),
            ),
            smem_tile_layout=Layout(
                IntTuple(1, 2, 16, 64),
                IntTuple(2048, 1024, 64, 1),
            ),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)

        # 4D with 64B swizzling
        test_tma_4d_load_row_major[
            DType.bfloat16,
            src_layout=Layout(
                IntTuple(2, 4, 16, 32),
                IntTuple(2048, 512, 32, 1),
            ),
            cta_tile_layout=Layout(
                IntTuple(1, 2, 16, 32),
                IntTuple(1024, 512, 32, 1),
            ),
            smem_tile_layout=Layout(
                IntTuple(1, 2, 16, 32),
                IntTuple(1024, 512, 32, 1),
            ),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_64B,
        ](ctx)

        # 4D with 32B swizzling
        test_tma_4d_load_row_major[
            DType.bfloat16,
            src_layout=Layout(
                IntTuple(2, 4, 16, 16),
                IntTuple(1024, 256, 16, 1),
            ),
            cta_tile_layout=Layout(
                IntTuple(1, 2, 16, 16),
                IntTuple(512, 256, 16, 1),
            ),
            smem_tile_layout=Layout(
                IntTuple(1, 2, 16, 16),
                IntTuple(512, 256, 16, 1),
            ),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_32B,
        ](ctx)
