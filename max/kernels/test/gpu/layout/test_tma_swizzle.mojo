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

from std.sys import size_of

from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu import block_idx, thread_idx
from layout import MixedLayout, TileTensor, row_major, stack_allocation
from layout._fillers import arange, random
from layout._host_device_tile_tensor import HostDeviceTileTensor
from layout.swizzle import make_swizzle
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    _idx_product,
    create_tensor_tile,
)
from std.memory import unsafe_stack_allocation

from std.utils.index import Index, IndexList


# Test loading a single 2d tile.
@__llvm_arg_metadata(tma_tile, `nvvm.grid_constant`)
def tma_swizzle_load_kernel[
    dtype: DType,
    layout: MixedLayout,
    tile_rank: Int,
    tile_shape: IndexList[tile_rank],
    desc_shape: IndexList[tile_rank],
](
    dst: TileTensor[dtype, type_of(layout), MutAnyOrigin],
    tma_tile: TMATensorTile[dtype, tile_rank, tile_shape, desc_shape],
):
    comptime tileM = tile_shape[0]
    comptime tileN = tile_shape[1]
    comptime expected_bytes = _idx_product[tile_rank, tile_shape]() * size_of[
        dtype
    ]()

    comptime __tile_layout = row_major[tileM, tileN]()
    var tile = stack_allocation[dtype, address_space=.SHARED, alignment=128](
        __tile_layout
    )

    var mbar = unsafe_stack_allocation[
        1,
        SharedMemBarrier,
        address_space=.SHARED,
        alignment=8,
    ]()

    if thread_idx.x == 0:
        mbar[0].init()
        mbar[0].expect_bytes(Int32(expected_bytes))
        tma_tile.async_copy(
            tile,
            mbar[0],
            (block_idx.x * tileN, block_idx.y * tileM),
        )
    # Ensure all threads sees initialized mbarrier
    barrier()
    mbar[0].wait()

    var dst_tile = dst.tile[tileM, tileN](block_idx.y, block_idx.x)

    if thread_idx.x == 0:
        dst_tile.copy_from(tile)


def test_tma_swizzle[
    dtype: DType,
    shape: IndexList[2],
    tile_shape: IndexList[2],
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    is_k_major: Bool = True,
](ctx: DeviceContext) raises:
    comptime assert (
        shape == tile_shape
    ), "Only support same shape and tile shape."

    comptime layout = row_major[shape[0], shape[1]]()
    var src = HostDeviceTileTensor[dtype](layout, ctx)
    var dst = HostDeviceTileTensor[dtype](layout, ctx)

    comptime if dtype == .float8_e4m3fn:
        random(src.host_tensor())
        random(dst.host_tensor())
    else:
        arange(src.host_tensor(), 0)
        arange(dst.host_tensor(), 0)

    src.to_device()
    dst.to_device()

    var tma_tensor = create_tensor_tile[
        tile_shape,
        swizzle_mode=swizzle_mode,
    ](ctx, src.device_tensor())

    # print test info
    comptime use_multiple_loads = (
        _idx_product[type_of(tma_tensor).rank, type_of(tma_tensor).tile_shape]()
        > _idx_product[
            type_of(tma_tensor).rank, type_of(tma_tensor).desc_shape
        ]()
    )
    comptime test_name = "test " + String(dtype) + (
        " multiple " if use_multiple_loads else " single "
    ) + "tma w/ " + String(swizzle_mode) + " k-major " + String(is_k_major)
    print(test_name)

    # Descriptor tile is the copy per tma instruction. One load could have multiple tma copies.
    comptime descM = type_of(tma_tensor).desc_shape[0]
    comptime descN = type_of(tma_tensor).desc_shape[1]
    comptime desc_tile_size = descM * descN
    comptime __desc_layout = row_major[descM, descN]()
    var desc_tile = stack_allocation[dtype](__desc_layout)

    comptime kernel = tma_swizzle_load_kernel[
        type_of(tma_tensor).dtype,
        layout,
        type_of(tma_tensor).rank,
        type_of(tma_tensor).tile_shape,
        type_of(tma_tensor).desc_shape,
    ]
    ctx.enqueue_function[kernel](
        dst.device_tensor(),
        tma_tensor,
        grid_dim=(shape[1] // tile_shape[1], shape[0] // tile_shape[0]),
        block_dim=(1),
    )

    dst.to_host()
    var desc_flat = desc_tile.reshape(row_major[desc_tile_size]())
    comptime assert desc_flat.flat_rank == 1

    var src_host = src.host_tensor()
    var dst_host = dst.host_tensor()

    comptime swizzle = make_swizzle[dtype, swizzle_mode]()

    var dst_flat = dst_host.reshape(row_major[shape[0] * shape[1]]())
    comptime assert dst_flat.flat_rank == 1
    var dst_tile_offset = 0
    for desc_tile_m in range(shape[0] // descM):
        for desc_tile_n in range(shape[1] // descN):
            desc_tile.copy_from(
                src_host.tile[descM, descN](desc_tile_m, desc_tile_n)
            )
            for i in range(desc_tile_size):
                var desc_idx = swizzle(i)
                if (
                    desc_flat[desc_idx].cast[.float64]()
                    != dst_flat[dst_tile_offset + i].cast[.float64]()
                ):
                    print(
                        desc_tile_m,
                        desc_tile_n,
                        desc_flat[desc_idx],
                        dst_flat[dst_tile_offset + i],
                    )
                    break
            dst_tile_offset += desc_tile_size

    _ = src^
    _ = dst^


def main() raises:
    with DeviceContext() as ctx:
        print("test_tma_swizzle_bf16")
        test_tma_swizzle[
            .bfloat16,
            shape=Index(8, 64),
            tile_shape=Index(8, 64),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)

        test_tma_swizzle[
            .bfloat16,
            shape=Index(8, 128),
            tile_shape=Index(8, 128),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)

        test_tma_swizzle[
            .bfloat16,
            shape=Index(8, 32),
            tile_shape=Index(8, 32),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_64B,
        ](ctx)

        test_tma_swizzle[
            .bfloat16,
            shape=Index(8, 64),
            tile_shape=Index(8, 64),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_64B,
        ](ctx)

        test_tma_swizzle[
            .bfloat16,
            shape=Index(8, 16),
            tile_shape=Index(8, 16),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_32B,
        ](ctx)

        test_tma_swizzle[
            .bfloat16,
            shape=Index(8, 32),
            tile_shape=Index(8, 32),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_32B,
        ](ctx)

        test_tma_swizzle[
            .bfloat16,
            shape=Index(8, 16),
            tile_shape=Index(8, 16),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
        ](ctx)

        test_tma_swizzle[
            .bfloat16,
            shape=Index(8, 32),
            tile_shape=Index(8, 32),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
        ](ctx)

        test_tma_swizzle[
            .bfloat16,
            shape=Index(16, 64),
            tile_shape=Index(16, 64),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
            is_k_major=False,
        ](ctx)

        test_tma_swizzle[
            .bfloat16,
            shape=Index(16, 128),
            tile_shape=Index(16, 128),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
            is_k_major=False,
        ](ctx)

        print("test_tma_swizzle_f8e4m3fn")
        test_tma_swizzle[
            .float8_e4m3fn,
            shape=Index(8, 128),
            tile_shape=Index(8, 128),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)

        test_tma_swizzle[
            .float8_e4m3fn,
            shape=Index(8, 256),
            tile_shape=Index(8, 256),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)

        test_tma_swizzle[
            .float8_e4m3fn,
            shape=Index(8, 64),
            tile_shape=Index(8, 64),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_64B,
        ](ctx)

        test_tma_swizzle[
            .float8_e4m3fn,
            shape=Index(8, 128),
            tile_shape=Index(8, 128),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_64B,
        ](ctx)

        test_tma_swizzle[
            .float8_e4m3fn,
            shape=Index(8, 32),
            tile_shape=Index(8, 32),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_32B,
        ](ctx)

        test_tma_swizzle[
            .float8_e4m3fn,
            shape=Index(8, 64),
            tile_shape=Index(8, 64),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_32B,
        ](ctx)

        test_tma_swizzle[
            .float8_e4m3fn,
            shape=Index(8, 16),
            tile_shape=Index(8, 16),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
        ](ctx)

        test_tma_swizzle[
            .float8_e4m3fn,
            shape=Index(8, 32),
            tile_shape=Index(8, 32),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
        ](ctx)

        test_tma_swizzle[
            .float8_e4m3fn,
            shape=Index(16, 128),
            tile_shape=Index(16, 128),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
            is_k_major=False,
        ](ctx)

        test_tma_swizzle[
            .float8_e4m3fn,
            shape=Index(16, 256),
            tile_shape=Index(16, 256),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
            is_k_major=False,
        ](ctx)
