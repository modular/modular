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

"""TMA load and store tests for TileTensor descriptors and shared sources.

The descriptor's global shape and strides must come from the TileTensor's
layout. Each test builds the source as a DeviceBuffer-backed TileTensor and
TMA-loads every tile of it into shared memory, then verifies the device
result against the host values:

- contiguous row-major source with exact tiling
- ragged global shape, so TMA must zero-fill the out-of-bounds region
- padded row stride (`row_stride > N`): the buffer holds sentinel values in
  the padding, which leak into the result if the descriptor ignores the
  tensor's strides
- bfloat16 with a padded row stride

Store tests cover a flat shared-memory source across ranks 2–5, direct and
rank-dispatched APIs, both descriptor box orders, and nonzero global
coordinates. Rank-4/5 tiles contain multiple descriptor boxes.

Prefetch tests issue the rank-3 and rank-4 L2 prefetch of a multi-box tile
at nonzero coordinates, then load the same tile and check every element.
"""

from std.math import align_up
from std.sys import size_of

from max.gpu.sync import (
    barrier,
    cp_async_bulk_commit_group,
    cp_async_bulk_wait_group,
)
from max.gpu.host import DeviceContext
from max.gpu.memory import fence_async_view_proxy
from max.gpu import block_idx, thread_idx
from layout import (
    ComptimeInt,
    Coord,
    Idx,
    MixedLayout,
    coord,
    RowMajorLayout,
    TileTensor,
    row_major,
)
from layout._host_device_tile_tensor import HostDeviceTileTensor
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    _idx_product,
    create_tma_tile,
    create_tensor_tile,
)
from std.memory import unsafe_stack_allocation
from std.testing import assert_equal
from std.utils.static_tuple import StaticTuple


@__llvm_arg_metadata(tma_tile, `nvvm.grid_constant`)
def tma_tile_tensor_load_kernel[
    dtype: DType,
    dst_rows: Int,
    dst_cols: Int,
    tile_shape: Coord,
](
    dst: TileTensor[
        dtype,
        RowMajorLayout[ComptimeInt[dst_rows], ComptimeInt[dst_cols]],
        MutAnyOrigin,
    ],
    tma_tile: TMATensorTile[dtype, tile_shape],
):
    comptime tileM = Int(tile_shape.element_types[0].static_value)
    comptime tileN = Int(tile_shape.element_types[1].static_value)
    comptime expected_bytes = _idx_product[tile_shape]() * size_of[dtype]()

    var tile = TileTensor(
        unsafe_stack_allocation[
            tileM * tileN, Scalar[dtype], address_space=.SHARED, alignment=128
        ](),
        row_major[tileM, tileN](),
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

    # One thread per tile element, so each thread copies its own element.
    var row, col = divmod(thread_idx.x, tileN)
    var dst_tile = dst.tile[tileM, tileN](block_idx.y, block_idx.x)
    dst_tile[row, col] = tile[row, col]


def test_tma_load_tile_tensor[
    dtype: DType,
    M: Int,
    N: Int,
    tileM: Int,
    tileN: Int,
    row_stride: Int,
](ctx: DeviceContext) raises:
    comptime M_roundup = align_up(M, tileM)
    comptime N_roundup = align_up(N, tileN)

    # The backing buffer holds `row_stride` elements per row; the TileTensor
    # views only the first `N` columns of each row. Logical element (m, n)
    # is filled with `m * N + n` and the row padding with a sentinel that
    # fails verification if the descriptor reads with the wrong stride.
    var src_host = ctx.enqueue_create_host_buffer[dtype](M * row_stride)
    for m in range(M):
        for n in range(N):
            src_host[m * row_stride + n] = Scalar[dtype](m * N + n)
        for n in range(N, row_stride):
            src_host[m * row_stride + n] = Scalar[dtype](-1)

    var src_device = ctx.enqueue_create_buffer[dtype](M * row_stride)
    ctx.enqueue_copy(src_device, src_host)

    var src_tensor = TileTensor(
        src_device,
        MixedLayout(Coord(Idx[M], Idx[N]), Coord(Idx[row_stride], Idx[1])),
    )
    var tma_tensor = create_tma_tile[tileM, tileN](ctx, src_tensor)
    ctx.synchronize()

    var dst = HostDeviceTileTensor[dtype](
        row_major[M_roundup, N_roundup](), ctx
    )

    comptime kernel = tma_tile_tensor_load_kernel[
        type_of(tma_tensor).dtype,
        M_roundup,
        N_roundup,
        type_of(tma_tensor).tile_shape,  # tile shape
    ]
    ctx.enqueue_function[kernel](
        dst.device_tensor(),
        tma_tensor,
        grid_dim=(N_roundup // tileN, M_roundup // tileM),
        block_dim=(tileM * tileN),
    )

    dst.to_host()
    # The descriptor holds a raw pointer, so the source must outlive the
    # kernel rather than end at its last use above.
    _ = src_device^
    var dst_host = dst.host_tensor()

    # In-bounds elements keep their values, and the region rounded up to
    # the tile shape is zero-filled by TMA.
    for m in range(M_roundup):
        for n in range(N_roundup):
            if m < M and n < N:
                assert_equal(
                    dst_host[m, n].cast[.float32](),
                    Float32(m * N + n),
                )
            else:
                assert_equal(dst_host[m, n].cast[.float32](), 0.0)


@__llvm_arg_metadata(tma_tile, `nvvm.grid_constant`)
def tma_store_rank_kernel[
    tile_shape: Coord,
    desc_shape: Coord,
    k_major: Bool,
    generic: Bool,
](tma_tile: TMATensorTile[.float32, tile_shape, desc_shape, k_major]):
    comptime rank = tile_shape.rank
    comptime size = _idx_product[tile_shape]()
    var ptr = unsafe_stack_allocation[
        size, Float32, address_space=.SHARED, alignment=128
    ]()
    for i in range(thread_idx.x, size, 32):
        ptr[i] = Float32(i + 1)
    barrier()
    fence_async_view_proxy()

    if thread_idx.x == 0:
        var src = TileTensor(ptr.as_imm(), row_major[size]())
        comptime if generic:
            var coords = StaticTuple[UInt32, rank]()
            comptime for i in range(rank):
                coords[i] = UInt32(4 if i == 0 else 1)
            tma_tile.async_store(src, coords)
        elif rank == 2:
            tma_tile.async_store(src, (4, 1))
        elif rank == 3:
            tma_tile.async_store_3d(src, (4, 1, 1))
        elif rank == 4:
            tma_tile.async_store_4d(src, (4, 1, 1, 1))
        else:
            tma_tile.async_store_5d(src, (4, 1, 1, 1, 1))

        cp_async_bulk_commit_group()
        cp_async_bulk_wait_group[0]()


def test_tma_store_rank[
    tile_shape: Coord,
    desc_shape: Coord,
    global_shape: Coord,
    k_major: Bool,
    generic: Bool,
](ctx: DeviceContext) raises:
    comptime rank = tile_shape.rank
    var dst = HostDeviceTileTensor[.float32](
        row_major(Coord[*global_shape.element_types]()), ctx
    )
    var initial_host = dst.host_tensor()
    _ = initial_host.fill(-1)
    dst.to_device()
    var tma = create_tensor_tile[
        tile_shape,
        k_major_tma=k_major,
        __desc_shape=desc_shape,
    ](ctx, dst.device_tensor())
    ctx.enqueue_function[
        tma_store_rank_kernel[tile_shape, desc_shape, k_major, generic]
    ](tma, grid_dim=1, block_dim=32)
    dst.to_host()
    var host = dst.host_tensor().reshape(
        row_major[_idx_product[global_shape]()]()
    )

    # Decode destination coordinates independently of the device offset helper.
    # Values outside the translated tile must retain the sentinel.
    for linear in range(_idx_product[global_shape]()):
        var remaining = linear
        var local = StaticTuple[Int, rank]()
        var inside = True
        comptime for reverse in range(rank):
            comptime dim = rank - reverse - 1
            var value = remaining % Int(global_shape[dim])
            remaining //= Int(global_shape[dim])
            local[dim] = value - (4 if dim == rank - 1 else 1)
            inside = inside and (0 <= local[dim] < Int(tile_shape[dim]))
        if not inside:
            assert_equal(host[linear], Float32(-1))
            continue

        var box = 0
        var inner = 0
        comptime for i in range(rank):
            comptime dim = rank - i - 1 if k_major else i
            comptime copies = Int(tile_shape[dim]) // Int(desc_shape[dim])
            box = box * copies + local[dim] // Int(desc_shape[dim])
            inner = inner * Int(desc_shape[i]) + local[i] % Int(desc_shape[i])
        assert_equal(
            host[linear],
            Float32(box * _idx_product[desc_shape]() + inner + 1),
        )

    print(
        "TMA store passed: rank=",
        rank,
        " k_major=",
        k_major,
        " generic=",
        generic,
    )


def test_tma_store_ranks(ctx: DeviceContext) raises:
    comptime for api in range(2):
        comptime generic = api == 1
        comptime for major in range(2):
            comptime k_major = major == 1
            test_tma_store_rank[
                coord[8, 8],
                coord[8, 8],
                coord[12, 16],
                k_major,
                generic,
            ](ctx)
            test_tma_store_rank[
                coord[2, 2, 8],
                coord[2, 2, 8],
                coord[4, 4, 16],
                k_major,
                generic,
            ](ctx)
            test_tma_store_rank[
                coord[2, 2, 2, 8],
                coord[1, 2, 2, 8],
                coord[4, 4, 4, 16],
                k_major,
                generic,
            ](ctx)
            test_tma_store_rank[
                coord[2, 2, 2, 2, 8],
                coord[1, 1, 2, 2, 8],
                coord[4, 4, 4, 4, 16],
                k_major,
                generic,
            ](ctx)


@__llvm_arg_metadata(tma_tile, `nvvm.grid_constant`)
def tma_prefetch_load_kernel[
    tile_shape: Coord,
    desc_shape: Coord,
](
    dst: TileTensor[
        .float32,
        RowMajorLayout[ComptimeInt[_idx_product[tile_shape]()]],
        MutAnyOrigin,
    ],
    tma_tile: TMATensorTile[.float32, tile_shape, desc_shape],
):
    comptime rank = tile_shape.rank
    comptime size = _idx_product[tile_shape]()
    var tile = TileTensor(
        unsafe_stack_allocation[
            size, Float32, address_space=.SHARED, alignment=128
        ](),
        row_major[size](),
    )
    var mbar = unsafe_stack_allocation[
        1,
        SharedMemBarrier,
        address_space=.SHARED,
        alignment=8,
    ]()

    if thread_idx.x == 0:
        mbar[0].init()
        mbar[0].expect_bytes(Int32(size * size_of[Float32]()))
        comptime if rank == 3:
            tma_tile.prefetch_l2_3d((4, 1, 1))
            tma_tile.async_copy_3d(tile, mbar[0], (4, 1, 1))
        else:
            tma_tile.prefetch_l2_4d((4, 1, 1, 1))
            tma_tile.async_copy_4d(tile, mbar[0], (4, 1, 1, 1))
    barrier()
    mbar[0].wait()

    for i in range(thread_idx.x, size, 32):
        dst[i] = tile[i]


def test_tma_prefetch_load[
    tile_shape: Coord,
    desc_shape: Coord,
    global_shape: Coord,
](ctx: DeviceContext) raises:
    comptime rank = tile_shape.rank
    comptime size = _idx_product[tile_shape]()
    comptime global_size = _idx_product[global_shape]()
    var src = HostDeviceTileTensor[.float32](
        row_major(Coord[*global_shape.element_types]()), ctx
    )
    var src_host = src.host_tensor().reshape(row_major[global_size]())
    for i in range(global_size):
        src_host[i] = Float32(i + 1)
    src.to_device()
    var tma = create_tensor_tile[tile_shape, __desc_shape=desc_shape](
        ctx, src.device_tensor()
    )
    var dst = HostDeviceTileTensor[.float32](row_major[size](), ctx)
    ctx.enqueue_function[tma_prefetch_load_kernel[tile_shape, desc_shape]](
        dst.device_tensor(), tma, grid_dim=1, block_dim=32
    )
    dst.to_host()
    _ = src^
    var host = dst.host_tensor()

    # Shared memory holds the boxes in row-major box order, each box
    # row-major; the tile starts at 4 on the innermost dim and 1 elsewhere.
    comptime box_size = _idx_product[desc_shape]()
    for s in range(size):
        var box, inner = divmod(s, box_size)
        var global_coord = StaticTuple[Int, rank]()
        comptime for r in range(rank):
            comptime dim = rank - 1 - r
            comptime copies = Int(tile_shape[dim]) // Int(desc_shape[dim])
            var box_rest, box_dim = divmod(box, copies)
            var inner_rest, inner_dim = divmod(inner, Int(desc_shape[dim]))
            box = box_rest
            inner = inner_rest
            global_coord[dim] = (
                box_dim * Int(desc_shape[dim])
                + inner_dim
                + (4 if dim == rank - 1 else 1)
            )
        var linear = 0
        comptime for dim in range(rank):
            linear = linear * Int(global_shape[dim]) + global_coord[dim]
        assert_equal(host[s], Float32(linear + 1))

    print("TMA prefetch + load passed: rank=", rank)


def main() raises:
    with DeviceContext() as ctx:
        test_tma_store_ranks(ctx)
        print("test_tma_load_tile_tensor_f32")
        test_tma_load_tile_tensor[
            dtype=DType.float32, M=8, N=8, tileM=4, tileN=4, row_stride=8
        ](ctx)

        print("test_tma_load_tile_tensor_oob_fill_f32")
        test_tma_load_tile_tensor[
            dtype=DType.float32, M=6, N=20, tileM=4, tileN=8, row_stride=20
        ](ctx)

        print("test_tma_load_tile_tensor_padded_stride_f32")
        test_tma_load_tile_tensor[
            dtype=DType.float32, M=5, N=16, tileM=4, tileN=8, row_stride=24
        ](ctx)

        print("test_tma_load_tile_tensor_padded_stride_bf16")
        test_tma_load_tile_tensor[
            dtype=DType.bfloat16, M=4, N=16, tileM=4, tileN=8, row_stride=24
        ](ctx)

        test_tma_prefetch_load[
            coord[2, 2, 16], coord[1, 2, 16], coord[4, 4, 32]
        ](ctx)
        test_tma_prefetch_load[
            coord[2, 2, 2, 8], coord[1, 2, 2, 8], coord[4, 4, 4, 16]
        ](ctx)
