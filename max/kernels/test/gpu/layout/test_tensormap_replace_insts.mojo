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
from max.gpu.host.nvidia.tma import TensorMapSwizzle, TMADescriptor
from max.gpu import block_idx, thread_idx
from max.gpu.sync import syncwarp
from layout import MixedLayout, TileTensor, row_major, stack_allocation
from layout._fillers import arange
from layout._host_device_tile_tensor import HostDeviceTileTensor
from layout.tile_io import copy_sram_to_dram
from layout.swizzle import make_swizzle
from layout.tma_async import (
    create_tensor_tile,
    SharedMemBarrier,
    TMATensorTile,
    TMATensorTileArray,
    _idx_product,
)
from std.memory import unsafe_stack_allocation
from std.testing import assert_equal

from std.utils.index import Index, IndexList


@__llvm_arg_metadata(template_tma_tensormap, `nvvm.grid_constant`)
def test_tma_replace_global_addr_in_gmem_descriptor_kernel[
    dtype: DType,
    num_of_tensormaps: Int,
    src_layout: MixedLayout,
    dst_layout: MixedLayout,
    tile_rank: Int,
    cta_tile_shape: IndexList[tile_rank],
    desc_shape: IndexList[tile_rank],
    thread_layout: MixedLayout,
](
    dst: TileTensor[dtype, type_of(dst_layout), MutAnyOrigin],
    new_src: TileTensor[dtype, type_of(src_layout), ImmutAnyOrigin],
    template_tma_tensormap: TMATensorTile[
        dtype, tile_rank, cta_tile_shape, desc_shape
    ],
    device_tma_tile: TMATensorTileArray[
        num_of_tensormaps, dtype, tile_rank, cta_tile_shape, desc_shape
    ],
):
    comptime M = cta_tile_shape[0]
    comptime N = cta_tile_shape[1]
    comptime expected_bytes = _idx_product[
        tile_rank, cta_tile_shape
    ]() * size_of[dtype]()

    comptime __cta_tile_layout = row_major[M, N]()
    var tile = stack_allocation[dtype, address_space=.SHARED, alignment=128](
        __cta_tile_layout
    )

    device_tma_tile[block_idx.x][].tensormap_fence_acquire()
    device_tma_tile[block_idx.x][].replace_tensormap_global_address_in_gmem(
        new_src.unsafe_ptr()
    )
    device_tma_tile[block_idx.x][].tensormap_fence_release()

    var mbar = unsafe_stack_allocation[
        1,
        SharedMemBarrier,
        address_space=.SHARED,
        alignment=8,
    ]()

    if thread_idx.x == 0:
        mbar[0].init()
        mbar[0].expect_bytes(Int32(expected_bytes))
        device_tma_tile[block_idx.x][].async_copy(tile, mbar[0], (0, 0))

    # Ensure all threads sees initialized mbarrier
    barrier()
    mbar[0].wait()

    var dst_tile = dst.tile[M, N](block_idx.x, 0)
    copy_sram_to_dram[thread_layout](dst_tile, tile)


def test_tma_replace_global_addr_in_gmem_descriptor[
    src_layout: MixedLayout,
](ctx: DeviceContext) raises:
    comptime M = type_of(src_layout).static_shape[0]
    comptime N = type_of(src_layout).static_shape[1]

    comptime num_of_tensormaps = 4

    comptime dst_layout = row_major[num_of_tensormaps * M, N]()

    var old_src = HostDeviceTileTensor[.bfloat16](src_layout, ctx)
    var new_src = HostDeviceTileTensor[.bfloat16](src_layout, ctx)
    var dst = HostDeviceTileTensor[.bfloat16](dst_layout, ctx)

    arange(old_src.host_tensor(), 1)
    old_src.to_device()
    arange(new_src.host_tensor(), 1001)
    new_src.to_device()

    var template_tma_tensormap = create_tensor_tile[Index(M, N)](
        ctx, old_src.device_tensor()
    )

    var device_tensormaps = ctx.enqueue_create_buffer[.uint8](
        128 * num_of_tensormaps
    )
    var tensormaps = TMATensorTileArray[
        num_of_tensormaps,
        type_of(template_tma_tensormap).dtype,
        type_of(template_tma_tensormap).rank,
        type_of(template_tma_tensormap).tile_shape,
        type_of(template_tma_tensormap).desc_shape,
    ](device_tensormaps)

    var tensormaps_host_ptr = unsafe_stack_allocation[
        num_of_tensormaps * 128, UInt8
    ]()

    comptime for i in range(num_of_tensormaps):
        for j in range(128):
            tensormaps_host_ptr[
                i * 128 + j
            ] = template_tma_tensormap.descriptor.data[j]
    ctx.enqueue_copy(device_tensormaps, tensormaps_host_ptr)

    ctx.synchronize()

    comptime __smem_M = type_of(template_tma_tensormap).tile_shape[0]
    comptime __smem_N = type_of(template_tma_tensormap).tile_shape[1]
    comptime __thread_layout = row_major[__smem_M, __smem_N]()

    comptime kernel = test_tma_replace_global_addr_in_gmem_descriptor_kernel[
        type_of(template_tma_tensormap).dtype,
        num_of_tensormaps,
        src_layout,  # src layout
        dst_layout,  # dst layout
        type_of(template_tma_tensormap).rank,  # tile rank
        type_of(template_tma_tensormap).tile_shape,  # cta tile shape
        type_of(template_tma_tensormap).desc_shape,  # desc shape
        __thread_layout,  # thread layout
    ]

    ctx.enqueue_function[kernel](
        dst.device_tensor(),
        new_src.device_tensor().as_imm(),
        template_tma_tensormap,
        tensormaps,
        grid_dim=(num_of_tensormaps),
        block_dim=(M * N),
    )

    dst.to_host()
    var new_src_host = new_src.host_tensor()
    var dst_host = dst.host_tensor()
    comptime assert new_src_host.flat_rank == 2 and dst_host.flat_rank == 2

    for m in range(num_of_tensormaps * M):
        for n in range(N):
            if m < M and n < N:
                assert_equal(
                    new_src_host[m % M, n].cast[.float32](),
                    dst_host[m, n].cast[.float32](),
                )

    ctx.synchronize()
    _ = old_src^
    _ = new_src^
    _ = dst^


# Test loading a single 2d tile.
@__llvm_arg_metadata(template_tma_tensormap, `nvvm.grid_constant`)
def test_tma_replace_global_addr_in_smem_descriptor_kernel[
    dtype: DType,
    num_of_tensormaps: Int,
    src_layout: MixedLayout,
    dst_layout: MixedLayout,
    tile_rank: Int,
    cta_tile_shape: IndexList[tile_rank],
    desc_shape: IndexList[tile_rank],
    thread_layout: MixedLayout,
](
    dst: TileTensor[dtype, type_of(dst_layout), MutAnyOrigin],
    new_src: TileTensor[dtype, type_of(src_layout), ImmutAnyOrigin],
    template_tma_tensormap: TMATensorTile[
        dtype, tile_rank, cta_tile_shape, desc_shape
    ],
    device_tma_tile: TMATensorTileArray[
        num_of_tensormaps, dtype, tile_rank, cta_tile_shape, desc_shape
    ],
):
    comptime M = cta_tile_shape[0]
    comptime N = cta_tile_shape[1]
    comptime expected_bytes = _idx_product[
        tile_rank, cta_tile_shape
    ]() * size_of[dtype]()

    comptime __cta_tile_layout = row_major[M, N]()
    var tile = stack_allocation[dtype, address_space=.SHARED, alignment=128](
        __cta_tile_layout
    )

    var smem_desc = unsafe_stack_allocation[
        1, TMADescriptor, alignment=128, address_space=.SHARED
    ]()

    # load the tensormap from gmem into smem. Only the one elected thread should call this
    if thread_idx.x == 0:
        template_tma_tensormap.smem_tensormap_init(smem_desc)

    barrier()

    device_tma_tile[block_idx.x][].tensormap_fence_acquire()

    # update the smem tensor map global addr. Only the one elected thread should call this
    if thread_idx.x == 0:
        device_tma_tile[
            block_idx.x
        ][].replace_tensormap_global_address_in_shared_mem(
            smem_desc, new_src.unsafe_ptr()
        )

    # Ensure warp is converged before issuing tensormap fence release
    syncwarp()

    # Entire warp should call this as it's an aligned instruction
    device_tma_tile[block_idx.x][].tensormap_cp_fence_release(smem_desc)

    var mbar = unsafe_stack_allocation[
        1,
        SharedMemBarrier,
        address_space=.SHARED,
        alignment=8,
    ]()

    if thread_idx.x == 0:
        mbar[0].init()
        mbar[0].expect_bytes(Int32(expected_bytes))
        device_tma_tile[block_idx.x][].async_copy(tile, mbar[0], (0, 0))

    # Ensure all threads sees initialized mbarrier
    barrier()
    mbar[0].wait()

    var dst_tile = dst.tile[M, N](0, 0)
    copy_sram_to_dram[thread_layout](dst_tile, tile)


def test_tma_replace_global_addr_in_smem_descriptor[
    src_layout: MixedLayout,
](ctx: DeviceContext) raises:
    comptime M = type_of(src_layout).static_shape[0]
    comptime N = type_of(src_layout).static_shape[1]

    comptime num_of_tensormaps = 4
    comptime dst_layout = row_major[num_of_tensormaps * M, N]()

    var old_src = HostDeviceTileTensor[.bfloat16](src_layout, ctx)
    var new_src = HostDeviceTileTensor[.bfloat16](src_layout, ctx)
    var dst = HostDeviceTileTensor[.bfloat16](dst_layout, ctx)

    arange(old_src.host_tensor(), 1)
    old_src.to_device()
    arange(new_src.host_tensor(), 1001)
    new_src.to_device()

    var template_tma_tensormap = create_tensor_tile[Index(M, N)](
        ctx, old_src.device_tensor()
    )

    var device_tensormaps = ctx.enqueue_create_buffer[.uint8](
        128 * num_of_tensormaps
    )
    var tensormaps = TMATensorTileArray[
        num_of_tensormaps,
        type_of(template_tma_tensormap).dtype,
        type_of(template_tma_tensormap).rank,
        type_of(template_tma_tensormap).tile_shape,
        type_of(template_tma_tensormap).desc_shape,
    ](device_tensormaps)

    var tensormaps_host_ptr = unsafe_stack_allocation[
        num_of_tensormaps * 128, UInt8
    ]()

    comptime for i in range(num_of_tensormaps):
        for j in range(128):
            tensormaps_host_ptr[
                i * 128 + j
            ] = template_tma_tensormap.descriptor.data[j]
    ctx.enqueue_copy(device_tensormaps, tensormaps_host_ptr)

    ctx.synchronize()

    comptime __smem_M = type_of(template_tma_tensormap).tile_shape[0]
    comptime __smem_N = type_of(template_tma_tensormap).tile_shape[1]
    comptime __thread_layout = row_major[__smem_M, __smem_N]()

    comptime kernel = test_tma_replace_global_addr_in_gmem_descriptor_kernel[
        type_of(template_tma_tensormap).dtype,
        num_of_tensormaps,
        src_layout,  # src layout
        dst_layout,  # dst layout
        type_of(template_tma_tensormap).rank,  # tile rank
        type_of(template_tma_tensormap).tile_shape,  # cta tile shape
        type_of(template_tma_tensormap).desc_shape,  # desc shape
        __thread_layout,  # thread layout
    ]

    ctx.enqueue_function[kernel](
        dst.device_tensor(),
        new_src.device_tensor().as_imm(),
        template_tma_tensormap,
        tensormaps,
        grid_dim=(num_of_tensormaps),
        block_dim=(M * N),
    )

    dst.to_host()
    var new_src_host = new_src.host_tensor()
    var dst_host = dst.host_tensor()
    comptime assert new_src_host.flat_rank == 2 and dst_host.flat_rank == 2

    for m in range(num_of_tensormaps * M):
        for n in range(N):
            if m < M and n < N:
                assert_equal(
                    new_src_host[m % M, n].cast[.float32](),
                    dst_host[m, n].cast[.float32](),
                )

    ctx.synchronize()
    _ = old_src^
    _ = new_src^
    _ = dst^


@__llvm_arg_metadata(template_tma_tensormap, `nvvm.grid_constant`)
def test_tma_replace_global_dim_in_smem_descriptor_kernel[
    dtype: DType,
    num_of_subtensors: Int,
    src_layout: MixedLayout,
    dst_layout: MixedLayout,
    tile_rank: Int,
    cta_tile_shape: IndexList[tile_rank],
    desc_shape: IndexList[tile_rank],
](
    dst: TileTensor[dtype, type_of(dst_layout), MutAnyOrigin],
    src: TileTensor[dtype, type_of(src_layout), ImmutAnyOrigin],
    template_tma_tensormap: TMATensorTile[
        dtype, tile_rank, cta_tile_shape, desc_shape
    ],
    subtensors_m: IndexList[num_of_subtensors + 1],
    device_tma_tile: TMATensorTileArray[
        num_of_subtensors, dtype, tile_rank, cta_tile_shape, desc_shape
    ],
):
    comptime tile_M = cta_tile_shape[0]
    comptime tile_N = cta_tile_shape[1]
    comptime expected_bytes = _idx_product[
        tile_rank, cta_tile_shape
    ]() * size_of[dtype]()

    comptime __cta_tile_layout = row_major[tile_M, tile_N]()
    var tile = stack_allocation[dtype, address_space=.SHARED, alignment=128](
        __cta_tile_layout
    )

    var smem_desc = unsafe_stack_allocation[
        1, TMADescriptor, alignment=128, address_space=.SHARED
    ]()

    # load the tensormap from gmem into smem. Only the one elected thread should call this
    if thread_idx.x == 0:
        template_tma_tensormap.smem_tensormap_init(smem_desc)

    barrier()

    device_tma_tile[block_idx.x][].tensormap_fence_acquire()

    # update the smem tensor map global addr, dims, and strides. Only the one elected thread should call this
    if thread_idx.x == 0:
        var src_tile = src.tile[1, tile_N](subtensors_m[block_idx.x], 0)
        var global_addr = src_tile.unsafe_ptr()

        device_tma_tile[
            block_idx.x
        ][].replace_tensormap_global_address_in_shared_mem(
            smem_desc,
            global_addr,
        )

        var block_size = (
            subtensors_m[block_idx.x + 1] - subtensors_m[block_idx.x]
        )

        device_tma_tile[
            block_idx.x
        ][].replace_tensormap_global_dim_strides_in_shared_mem[
            dtype,
            2,
            0,
        ](
            smem_desc, UInt32(block_size)
        )

    # Ensure warp is converged before issuing tensormap fence release
    syncwarp()

    # Entire warp should call this as it's an aligned instruction
    device_tma_tile[block_idx.x][].tensormap_cp_fence_release(smem_desc)

    var mbar = unsafe_stack_allocation[
        1,
        SharedMemBarrier,
        address_space=.SHARED,
        alignment=8,
    ]()

    if thread_idx.x == 0:
        mbar[0].init()
        mbar[0].expect_bytes(Int32(expected_bytes))
        device_tma_tile[block_idx.x][].async_copy(tile, mbar[0], (0, 0))

    # Ensure all threads sees initialized mbarrier
    barrier()
    mbar[0].wait()

    var dst_tile = dst.tile[tile_M, tile_N](block_idx.x, 0)
    copy_sram_to_dram[row_major[tile_M, tile_N]()](dst_tile, tile)


def test_tma_replace_global_dim_in_smem_descriptor[
    dtype: DType,
    src_layout: MixedLayout,
    cta_tile_layout: MixedLayout,
    size_of_subtensors: Int,
    swizzle_mode: TensorMapSwizzle,
](ctx: DeviceContext, subtensors_m: IndexList[size_of_subtensors]) raises:
    comptime M = type_of(src_layout).static_shape[0]
    comptime N = type_of(src_layout).static_shape[1]

    comptime cta_tile_M = type_of(cta_tile_layout).static_shape[0]
    comptime cta_tile_N = type_of(cta_tile_layout).static_shape[1]

    comptime assert N == cta_tile_N, (
        "for this test number of columns in src layout should be equal to"
        " number of columns in cta tile layout"
    )

    assert ctx.get_api_version() >= 12050, (
        "CUDA version must be >= 12.5. Current implementation of"
        " `replace_tensormap_global_dim_strides_in_shared_mem` dose not"
        " support CUDA versions < 12.5"
    )
    comptime num_of_subtensors = size_of_subtensors - 1

    var old_src = HostDeviceTileTensor[dtype](
        row_major[cta_tile_M, cta_tile_N](), ctx
    )
    arange(old_src.host_tensor(), 1)
    old_src.to_device()

    var template_tma_tensormap = create_tensor_tile[
        Index(cta_tile_M, cta_tile_N), swizzle_mode=swizzle_mode
    ](ctx, old_src.device_tensor())

    comptime dst_layout = row_major[
        num_of_subtensors * cta_tile_M, cta_tile_N
    ]()

    var new_src = HostDeviceTileTensor[dtype](src_layout, ctx)
    var dst = HostDeviceTileTensor[dtype](dst_layout, ctx)
    arange(new_src.host_tensor(), 1001)
    new_src.to_device()

    var device_tensormaps = ctx.enqueue_create_buffer[.uint8](
        128 * num_of_subtensors
    )
    var tensormaps_host_ptr = unsafe_stack_allocation[
        num_of_subtensors * 128, UInt8
    ]()

    comptime for i in range(num_of_subtensors):
        for j in range(128):
            tensormaps_host_ptr[
                i * 128 + j
            ] = template_tma_tensormap.descriptor.data[j]
    ctx.enqueue_copy(device_tensormaps, tensormaps_host_ptr)

    var tensormaps = TMATensorTileArray[
        num_of_subtensors,
        dtype,
        type_of(template_tma_tensormap).rank,
        type_of(template_tma_tensormap).tile_shape,
        type_of(template_tma_tensormap).desc_shape,
    ](device_tensormaps)

    ctx.synchronize()

    comptime kernel = test_tma_replace_global_dim_in_smem_descriptor_kernel[
        dtype,
        num_of_subtensors,
        src_layout,  # new src layout
        dst_layout,  # dst layout
        type_of(template_tma_tensormap).rank,  # tile rank
        type_of(template_tma_tensormap).tile_shape,  # cta tile shape
        type_of(template_tma_tensormap).desc_shape,  # desc shape
    ]

    ctx.enqueue_function[kernel](
        dst.device_tensor(),
        new_src.device_tensor().as_imm(),
        template_tma_tensormap,
        subtensors_m,
        tensormaps,
        grid_dim=(num_of_subtensors),
        block_dim=(cta_tile_M * cta_tile_N),
    )

    comptime swizzle = make_swizzle[dtype, swizzle_mode]()

    var dest_tile = stack_allocation[dtype](row_major[cta_tile_M, cta_tile_N]())
    var dest_flat = dest_tile.reshape(row_major[cta_tile_M * cta_tile_N]())
    comptime assert dest_flat.flat_rank == 1

    dst.to_host()
    var new_src_host = new_src.host_tensor()
    var dst_host = dst.host_tensor()
    comptime assert new_src_host.flat_rank == 2 and dst_host.flat_rank == 2

    for i in range(num_of_subtensors):
        dest_tile.copy_from(dst_host.tile[cta_tile_M, cta_tile_N](i, 0))

        var src_tile = new_src_host[subtensors_m[i] : subtensors_m[i + 1], :]
        comptime assert src_tile.flat_rank == 2
        var src_M = subtensors_m[i + 1] - subtensors_m[i]
        var src_N = cta_tile_N

        for dest_idx in range(cta_tile_M * cta_tile_N):
            if dest_idx < src_M * src_N:
                var swizzled_dest_idx = swizzle(dest_idx)
                assert_equal(
                    dest_flat[swizzled_dest_idx],
                    src_tile[dest_idx // src_N, dest_idx % src_N],
                )
            else:
                assert_equal(dest_flat[dest_idx], 0)

    ctx.synchronize()
    _ = old_src^
    _ = new_src^
    _ = dst^


def main() raises:
    with DeviceContext() as ctx:
        print("test_tma_replace_global_addr_in_gmem_descriptor")
        test_tma_replace_global_addr_in_gmem_descriptor[
            src_layout=row_major[8, 8](),
        ](ctx)

        print("test_tma_replace_global_addr_in_smem_descriptor")
        test_tma_replace_global_addr_in_smem_descriptor[
            src_layout=row_major[8, 8](),
        ](ctx)

        print("test_tma_replace_global_dim_in_smem_descriptor")
        print(" - SWIZZLE_NONE")
        test_tma_replace_global_dim_in_smem_descriptor[
            DType.bfloat16,
            src_layout=row_major[16, 8](),
            cta_tile_layout=row_major[32, 8](),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
        ](
            ctx,
            Index(0, 9, 16),
        )
        test_tma_replace_global_dim_in_smem_descriptor[
            DType.bfloat16,
            src_layout=row_major[29, 8](),
            cta_tile_layout=row_major[32, 8](),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
        ](
            ctx,
            Index(0, 9, 16, 25, 29),
        )
        print(" - SWIZZLE_32B")
        test_tma_replace_global_dim_in_smem_descriptor[
            DType.bfloat16,
            src_layout=row_major[29, 16](),
            cta_tile_layout=row_major[32, 16](),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_32B,
        ](
            ctx,
            Index(0, 9, 16, 25, 29),
        )
        print(" - SWIZZLE_64B")
        test_tma_replace_global_dim_in_smem_descriptor[
            DType.bfloat16,
            src_layout=row_major[29, 32](),
            cta_tile_layout=row_major[32, 32](),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_64B,
        ](
            ctx,
            Index(0, 9, 16, 25, 29),
        )
        print(" - SWIZZLE_128B")
        test_tma_replace_global_dim_in_smem_descriptor[
            DType.bfloat16,
            src_layout=row_major[15, 64](),
            cta_tile_layout=row_major[16, 64](),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
        ](
            ctx,
            Index(0, 3, 7, 11, 15),
        )
