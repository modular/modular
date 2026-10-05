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
from max.gpu.primitives.cluster import block_rank_in_cluster, cluster_sync
from max.gpu.host import DeviceContext, Dim
from max.gpu import block_idx, thread_idx
from max.gpu.memory import fence_mbarrier_init
from layout import MixedLayout, TileTensor, row_major, stack_allocation
from layout._fillers import arange
from layout._host_device_tile_tensor import HostDeviceTileTensor
from layout.tile_io import copy_sram_to_dram
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    _idx_product,
    create_tma_tile,
)
from std.memory import unsafe_stack_allocation
from std.testing import assert_equal
from std.utils.index import IndexList


# Test loading a single 2d tile.
@__llvm_arg_metadata(tma_tile, `nvvm.grid_constant`)
def test_tma_mcast_load_kernel[
    dtype: DType,
    layout: MixedLayout,
    tile_rank: Int,
    tile_shape: IndexList[tile_rank],
    thread_layout: MixedLayout,
    CLUSTER_M: UInt32,
    CLUSTER_N: UInt32,
](
    dst: TileTensor[dtype, type_of(layout), MutAnyOrigin],
    tma_tile: TMATensorTile[dtype, tile_rank, tile_shape],
):
    comptime tileM = tile_shape[0]
    comptime tileN = tile_shape[1]
    comptime expected_bytes = _idx_product[tile_rank, tile_shape]() * size_of[
        dtype
    ]()

    var block_rank = block_rank_in_cluster()
    comptime CLUSTER_SIZE = CLUSTER_M * CLUSTER_N

    var rank_m, rank_n = divmod(block_rank, CLUSTER_N)

    var tma_multicast_mask = (1 << CLUSTER_N) - 1

    comptime __tile_layout = row_major[tileM, tileN]()
    var tile = stack_allocation[dtype, address_space=.SHARED, alignment=128](
        __tile_layout
    ).fill(10)

    barrier()

    var mbar = unsafe_stack_allocation[
        1,
        SharedMemBarrier,
        address_space=.SHARED,
        alignment=8,
    ]()
    if thread_idx.x == 0:
        mbar[0].init()

    barrier()

    # Make the initialized barrier visible across the cluster.
    cluster_sync()
    fence_mbarrier_init()

    if thread_idx.x == 0:
        mbar[0].expect_bytes(Int32(expected_bytes))
        if rank_n == 0:
            var multicast_mask = tma_multicast_mask << (rank_m * CLUSTER_N)
            tma_tile.async_multicast_load(
                tile,
                mbar[0],
                (
                    block_idx.x * tileN,
                    block_idx.y * tileM,
                ),
                multicast_mask.cast[.uint16](),
            )

    barrier()

    # Wait for this CTA's multicast transfer to finish.
    mbar[0].wait()

    # Keep every CTA alive until all multicast recipients have finished.
    cluster_sync()

    var dst_tile = dst.tile[tileM, tileN](block_idx.y, block_idx.x)
    copy_sram_to_dram[thread_layout](dst_tile, tile)


def test_tma_multicast_load_row_major[
    src_layout: MixedLayout,
    tile_layout: MixedLayout,
    dst_layout: MixedLayout,
    CLUSTER_M: Int,
    CLUSTER_N: Int,
](ctx: DeviceContext) raises:
    comptime src_M = type_of(src_layout).static_shape[0]
    comptime src_N = type_of(src_layout).static_shape[1]
    comptime tileM = type_of(tile_layout).static_shape[0]
    comptime tileN = type_of(tile_layout).static_shape[1]
    comptime dst_M = type_of(dst_layout).static_shape[0]
    comptime dst_N = type_of(dst_layout).static_shape[1]

    var src = HostDeviceTileTensor[.float32](src_layout, ctx)
    var dst = HostDeviceTileTensor[.float32](dst_layout, ctx)

    arange(src.host_tensor(), 1)
    arange(dst.host_tensor(), 100001)
    src.to_device()
    dst.to_device()
    var tma_tensor = create_tma_tile[tileM, tileN](ctx, src.device_tensor())
    ctx.synchronize()

    comptime __tileM = type_of(tma_tensor).tile_shape[0]
    comptime __tileN = type_of(tma_tensor).tile_shape[1]
    comptime __thread_layout = row_major[__tileM, __tileN]()
    comptime kernel = test_tma_mcast_load_kernel[
        type_of(tma_tensor).dtype,
        dst_layout,  # dst layout
        type_of(tma_tensor).rank,  # tile rank
        type_of(tma_tensor).tile_shape,  # tile shape
        __thread_layout,  # thread layout
        UInt32(CLUSTER_M),
        UInt32(CLUSTER_N),
    ]

    ctx.enqueue_function[kernel](
        dst.device_tensor(),
        tma_tensor,
        grid_dim=(dst_N // tileN, dst_M // tileM),
        block_dim=(tileN * tileM),
        cluster_dim=Dim(CLUSTER_N, CLUSTER_M, 1),
    )

    dst.to_host()
    var src_host = src.host_tensor()
    var dst_host = dst.host_tensor()
    comptime assert src_host.flat_rank == dst_host.flat_rank == 2

    for m in range(dst_M):
        for n in range(dst_N):
            assert_equal(
                dst_host[m, n].cast[.float32](),
                src_host[m, n % src_N].cast[.float32](),
            )

    ctx.synchronize()
    _ = src^
    _ = dst^


# Test loading a single 2d tile.
@__llvm_arg_metadata(tma_tile, `nvvm.grid_constant`)
def test_tma_sliced_multicast_load_kernel[
    dtype: DType,
    layout: MixedLayout,
    tile_layout: MixedLayout,
    thread_layout: MixedLayout,
    CLUSTER_M: UInt32,
    CLUSTER_N: UInt32,
    tma_rank: Int,
    tma_tile_shape: IndexList[tma_rank],
](
    dst: TileTensor[dtype, type_of(layout), MutAnyOrigin],
    tma_tile: TMATensorTile[dtype, tma_rank, tma_tile_shape],
):
    comptime tileM = type_of(tile_layout).static_shape[0]
    comptime tileN = type_of(tile_layout).static_shape[1]
    comptime expected_bytes = Int(tile_layout.product()) * size_of[dtype]()

    var block_rank = block_rank_in_cluster()
    comptime CLUSTER_SIZE = CLUSTER_M * CLUSTER_N

    var rank_m, rank_n = divmod(block_rank, CLUSTER_N)

    var tma_multicast_mask = (1 << CLUSTER_N) - 1

    var tile = stack_allocation[dtype, address_space=.SHARED, alignment=128](
        row_major[tileM, tileN]()
    ).fill(10)

    barrier()

    var mbar = unsafe_stack_allocation[
        1,
        SharedMemBarrier,
        address_space=.SHARED,
        alignment=8,
    ]()
    if thread_idx.x == 0:
        mbar[0].init()

    barrier()

    # Make the initialized barrier visible across the cluster.
    cluster_sync()
    fence_mbarrier_init()

    if thread_idx.x == 0:
        mbar[0].expect_bytes(Int32(expected_bytes))
        var slice_cord = Int(
            UInt32(block_idx.y) * UInt32(tileM)
            + UInt32(Int(block_rank % CLUSTER_N) * tileM) // CLUSTER_N
        )
        var multicast_mask = tma_multicast_mask << (rank_m * CLUSTER_N)
        tma_tile.async_multicast_load(
            tile.tile[tileM // Int(CLUSTER_N), tileN](
                Int(block_rank % CLUSTER_N), 0
            ),
            mbar[0],
            (0, slice_cord),
            multicast_mask.cast[.uint16](),
        )

    barrier()

    # Wait for this CTA's multicast transfer to finish.
    mbar[0].wait()

    # Keep every CTA alive until all multicast recipients have finished.
    cluster_sync()

    var dst_tile = dst.tile[tileM, tileN](block_idx.y, block_idx.x)
    copy_sram_to_dram[thread_layout](dst_tile, tile)


def test_tma_sliced_multicast_load_row_major[
    src_layout: MixedLayout,
    tile_layout: MixedLayout,
    dst_layout: MixedLayout,
    CLUSTER_M: Int,
    CLUSTER_N: Int,
](ctx: DeviceContext) raises:
    comptime src_M = type_of(src_layout).static_shape[0]
    comptime src_N = type_of(src_layout).static_shape[1]
    comptime tileM = type_of(tile_layout).static_shape[0]
    comptime tileN = type_of(tile_layout).static_shape[1]
    comptime dst_M = type_of(dst_layout).static_shape[0]
    comptime dst_N = type_of(dst_layout).static_shape[1]

    var src = HostDeviceTileTensor[.float32](src_layout, ctx)
    var dst = HostDeviceTileTensor[.float32](dst_layout, ctx)

    arange(src.host_tensor(), 1)
    arange(dst.host_tensor(), 100001)
    src.to_device()
    dst.to_device()
    var tma_tensor = create_tma_tile[tileM // CLUSTER_N, tileN](
        ctx, src.device_tensor()
    )
    ctx.synchronize()

    comptime kernel = test_tma_sliced_multicast_load_kernel[
        type_of(tma_tensor).dtype,
        dst_layout,  # dst layout
        row_major[tileM, tileN](),
        row_major[tileM, tileN](),
        UInt32(CLUSTER_M),
        UInt32(CLUSTER_N),
        type_of(tma_tensor).rank,  # tma rank
        type_of(tma_tensor).tile_shape,  # tma tile shape
    ]

    ctx.enqueue_function[kernel](
        dst.device_tensor(),
        tma_tensor,
        grid_dim=(dst_N // tileN, dst_M // tileM),
        block_dim=(tileN * tileM),
        cluster_dim=Dim(CLUSTER_N, CLUSTER_M, 1),
    )

    dst.to_host()
    var src_host = src.host_tensor()
    var dst_host = dst.host_tensor()
    comptime assert src_host.flat_rank == dst_host.flat_rank == 2

    for m in range(dst_M):
        for n in range(dst_N):
            assert_equal(
                dst_host[m, n].cast[.float32](),
                src_host[m, n % src_N].cast[.float32](),
            )

    ctx.synchronize()
    _ = src^
    _ = dst^


def main() raises:
    with DeviceContext() as ctx:
        print("test_tma_multicast_load_row_major")
        test_tma_multicast_load_row_major[
            src_layout=row_major[8, 8](),
            tile_layout=row_major[4, 8](),
            dst_layout=row_major[8, 16](),
            CLUSTER_M=1,
            CLUSTER_N=2,
        ](ctx)
        test_tma_multicast_load_row_major[
            src_layout=row_major[16, 8](),
            tile_layout=row_major[4, 8](),
            dst_layout=row_major[16, 16](),
            CLUSTER_M=2,
            CLUSTER_N=2,
        ](ctx)

        print("test_tma_sliced_multicast_load_row_major")
        test_tma_sliced_multicast_load_row_major[
            src_layout=row_major[8, 16](),
            tile_layout=row_major[4, 16](),
            dst_layout=row_major[8, 32](),
            CLUSTER_M=1,
            CLUSTER_N=2,
        ](ctx)
        test_tma_sliced_multicast_load_row_major[
            src_layout=row_major[16, 16](),
            tile_layout=row_major[4, 16](),
            dst_layout=row_major[16, 32](),
            CLUSTER_M=2,
            CLUSTER_N=2,
        ](ctx)
        test_tma_sliced_multicast_load_row_major[
            src_layout=row_major[32, 16](),
            tile_layout=row_major[4, 16](),
            dst_layout=row_major[32, 32](),
            CLUSTER_M=4,
            CLUSTER_N=2,
        ](ctx)
        test_tma_sliced_multicast_load_row_major[
            src_layout=row_major[32, 16](),
            tile_layout=row_major[16, 16](),
            dst_layout=row_major[32, 64](),
            CLUSTER_M=2,
            CLUSTER_N=4,
        ](ctx)
