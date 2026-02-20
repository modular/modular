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

from sys import size_of

from gpu import barrier
from gpu.host import DeviceContext
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu import block_idx, thread_idx
from layout import Layout, LayoutTensor
from layout._fillers import arange, random
from layout._utils import ManagedLayoutTensor
from layout.swizzle import make_swizzle
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    create_tensor_tile,
    create_tma_tile,
)
from memory import stack_allocation

from utils.index import Index, IndexList


# Test loading a single 2d tile.
@__llvm_arg_metadata(tma_tile, `nvvm.grid_constant`)
fn tma_swizzle_load_kernel[
    dtype: DType,
    layout: Layout,
    tile_layout: Layout,
    desc_layout: Layout,
](
    dst: LayoutTensor[dtype, layout, MutAnyOrigin],
    tma_tile: TMATensorTile[dtype, tile_layout, desc_layout],
):
    comptime tileM = tile_layout.shape[0].value()
    comptime tileN = tile_layout.shape[1].value()
    comptime expected_bytes = tile_layout.size() * size_of[dtype]()

    tile = LayoutTensor[
        dtype,
        tile_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ].stack_allocation()

    mbar = stack_allocation[
        1,
        SharedMemBarrier,
        address_space = AddressSpace.SHARED,
        alignment=8,
    ]()

    if thread_idx.x == 0:
        mbar[0].init()
        mbar[0].expect_bytes(Int32(expected_bytes))
        tma_tile.async_copy(
            tile,
            mbar[0],
            (Int(block_idx.x) * tileN, Int(block_idx.y) * tileM),
        )
    # Ensure all threads sees initialized mbarrier
    barrier()
    mbar[0].wait()

    dst_tile = dst.tile[tileM, tileN](Int(block_idx.y), Int(block_idx.x))

    if thread_idx.x == 0:
        dst_tile.copy_from(tile)


def test_tma_swizzle[
    dtype: DType,
    shape: IndexList[2],
    tile_shape: IndexList[2],
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    is_k_major: Bool = True,
](ctx: DeviceContext):
    comptime assert (
        shape == tile_shape
    ), "Only support same shape and tile shape."

    comptime layout = Layout.row_major(shape[0], shape[1])
    var src = ManagedLayoutTensor[dtype, layout](ctx)
    var dst = ManagedLayoutTensor[dtype, layout](ctx)

    comptime if dtype == DType.float8_e4m3fn:
        random(src.tensor[update=False]())
        random(dst.tensor[update=False]())
    else:
        arange(src.tensor[update=False](), 0)
        arange(dst.tensor[update=False](), 0)

    var tma_tensor = create_tensor_tile[
        tile_shape,
        swizzle_mode=swizzle_mode,
    ](ctx, src.device_tensor())

    # print test info
    comptime use_multiple_loads = (
        tma_tensor.layout.size() > tma_tensor.desc_layout.size()
    )
    comptime test_name = "test " + String(dtype) + (
        " multiple " if use_multiple_loads else " single "
    ) + "tma w/ " + String(swizzle_mode) + " k-major " + String(is_k_major)
    print(test_name)

    # Descriptor tile is the copy per tma instruction. One load could have multiple tma copies.
    comptime descM = type_of(tma_tensor).desc_layout.shape[0].value()
    comptime descN = type_of(tma_tensor).desc_layout.shape[1].value()
    comptime desc_tile_size = descM * descN
    desc_tile = LayoutTensor[
        dtype, type_of(tma_tensor).desc_layout, MutAnyOrigin
    ].stack_allocation()

    comptime kernel = tma_swizzle_load_kernel[
        type_of(tma_tensor).dtype,
        layout,
        type_of(tma_tensor).layout,
        type_of(tma_tensor).desc_layout,
    ]
    ctx.enqueue_function[kernel, kernel](
        dst.device_tensor(),
        tma_tensor,
        grid_dim=(shape[1] // tile_shape[1], shape[0] // tile_shape[0]),
        block_dim=(1),
    )

    src_host = src.tensor()
    dst_host = dst.tensor()

    comptime swizzle = make_swizzle[dtype, swizzle_mode]()

    dst_tile_ptr = dst_host.ptr
    for desc_tile_m in range(shape[0] // descM):
        for desc_tile_n in range(shape[1] // descN):
            desc_tile.copy_from(
                src_host.tile[descM, descN](desc_tile_m, desc_tile_n)
            )
            for i in range(desc_tile_size):
                desc_idx = swizzle(i)
                if (
                    desc_tile.ptr[desc_idx].cast[DType.float64]()
                    != dst_tile_ptr[i].cast[DType.float64]()
                ):
                    print(
                        desc_tile_m,
                        desc_tile_n,
                        desc_tile.ptr[desc_idx],
                        dst_tile_ptr[i],
                    )
                    break
            dst_tile_ptr += desc_tile_size

    _ = src^
    _ = dst^


def main():
    with DeviceContext() as ctx:
        print("test_tma_swizzle_bf16")
        test_tma_swizzle[
            DType.bfloat16,
            shape = Index(8, 64),
            tile_shape = Index(8, 64),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)

        test_tma_swizzle[
            DType.bfloat16,
            shape = Index(8, 128),
            tile_shape = Index(8, 128),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)

        test_tma_swizzle[
            DType.bfloat16,
            shape = Index(8, 32),
            tile_shape = Index(8, 32),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_64B,
        ](ctx)

        test_tma_swizzle[
            DType.bfloat16,
            shape = Index(8, 64),
            tile_shape = Index(8, 64),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_64B,
        ](ctx)

        test_tma_swizzle[
            DType.bfloat16,
            shape = Index(8, 16),
            tile_shape = Index(8, 16),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_32B,
        ](ctx)

        test_tma_swizzle[
            DType.bfloat16,
            shape = Index(8, 32),
            tile_shape = Index(8, 32),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_32B,
        ](ctx)

        test_tma_swizzle[
            DType.bfloat16,
            shape = Index(8, 16),
            tile_shape = Index(8, 16),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE,
        ](ctx)

        test_tma_swizzle[
            DType.bfloat16,
            shape = Index(8, 32),
            tile_shape = Index(8, 32),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE,
        ](ctx)

        test_tma_swizzle[
            DType.bfloat16,
            shape = Index(16, 64),
            tile_shape = Index(16, 64),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_128B,
            is_k_major=False,
        ](ctx)

        test_tma_swizzle[
            DType.bfloat16,
            shape = Index(16, 128),
            tile_shape = Index(16, 128),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_128B,
            is_k_major=False,
        ](ctx)

        print("test_tma_swizzle_f8e4m3fn")
        test_tma_swizzle[
            DType.float8_e4m3fn,
            shape = Index(8, 128),
            tile_shape = Index(8, 128),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)

        test_tma_swizzle[
            DType.float8_e4m3fn,
            shape = Index(8, 256),
            tile_shape = Index(8, 256),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)

        test_tma_swizzle[
            DType.float8_e4m3fn,
            shape = Index(8, 64),
            tile_shape = Index(8, 64),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_64B,
        ](ctx)

        test_tma_swizzle[
            DType.float8_e4m3fn,
            shape = Index(8, 128),
            tile_shape = Index(8, 128),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_64B,
        ](ctx)

        test_tma_swizzle[
            DType.float8_e4m3fn,
            shape = Index(8, 32),
            tile_shape = Index(8, 32),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_32B,
        ](ctx)

        test_tma_swizzle[
            DType.float8_e4m3fn,
            shape = Index(8, 64),
            tile_shape = Index(8, 64),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_32B,
        ](ctx)

        test_tma_swizzle[
            DType.float8_e4m3fn,
            shape = Index(8, 16),
            tile_shape = Index(8, 16),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE,
        ](ctx)

        test_tma_swizzle[
            DType.float8_e4m3fn,
            shape = Index(8, 32),
            tile_shape = Index(8, 32),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE,
        ](ctx)

        test_tma_swizzle[
            DType.float8_e4m3fn,
            shape = Index(16, 128),
            tile_shape = Index(16, 128),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_128B,
            is_k_major=False,
        ](ctx)

        test_tma_swizzle[
            DType.float8_e4m3fn,
            shape = Index(16, 256),
            tile_shape = Index(16, 256),
            swizzle_mode = TensorMapSwizzle.SWIZZLE_128B,
            is_k_major=False,
        ](ctx)
