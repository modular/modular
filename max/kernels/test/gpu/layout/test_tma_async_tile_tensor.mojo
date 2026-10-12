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

Store tests compare native immutable and legacy flat/nested shared views across
ranks 2–5, direct and rank-dispatched APIs, both descriptor box orders, and
nonzero global coordinates. Rank-4/5 tiles contain multiple descriptor boxes.
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
    Layout,
    LayoutTensor,
    IntTuple,
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
    comptime tileM = Int(tile_shape.element_types[0].static_value.value())
    comptime tileN = Int(tile_shape.element_types[1].static_value.value())
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
    var row, col = divmod(Int(thread_idx.x), tileN)
    var dst_tile = dst.tile[tileM, tileN](Int(block_idx.y), Int(block_idx.x))
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
    legacy: Int,
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
        comptime if legacy == 1:
            # A rank-one source verifies that TMA does not use logical strides
            # to address the descriptor's higher-rank physical boxes.
            var src = LayoutTensor[
                .float32,
                Layout.row_major(size),
                address_space=.SHARED,
                alignment=128,
            ](ptr)
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
        elif legacy == 2:
            var src = LayoutTensor[
                .float32,
                Layout.row_major(IntTuple(IntTuple(2, size // 4), 2)),
                address_space=.SHARED,
                alignment=128,
            ](ptr)
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
        else:
            var src = TileTensor(
                ptr.unsafe_mut_cast[False](), row_major[size]()
            )
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
    legacy: Int,
    generic: Bool,
](ctx: DeviceContext) raises:
    comptime rank = tile_shape.rank
    var dst = HostDeviceTileTensor[.float32](
        row_major(Coord[*global_shape.element_types]()), ctx
    )
    var initial_host = dst.host_tensor()
    for i in range(_idx_product[global_shape]()):
        initial_host.ptr[i] = -1
    dst.to_device()
    var tma = create_tensor_tile[
        tile_shape,
        k_major_tma=k_major,
        __desc_shape=desc_shape,
    ](ctx, dst.device_tensor())
    ctx.enqueue_function[
        tma_store_rank_kernel[tile_shape, desc_shape, k_major, legacy, generic]
    ](tma, grid_dim=1, block_dim=32)
    dst.to_host()
    var host = dst.host_tensor()

    # Decode destination coordinates independently of the device offset helper.
    # Values outside the translated tile must retain the sentinel.
    for linear in range(_idx_product[global_shape]()):
        var remaining = linear
        var local = StaticTuple[Int, rank]()
        var inside = True
        comptime for reverse in range(rank):
            comptime dim = rank - reverse - 1
            var value = remaining % Int(global_shape[dim].value())
            remaining //= Int(global_shape[dim].value())
            local[dim] = value - (4 if dim == rank - 1 else 1)
            inside = inside and (0 <= local[dim] < Int(tile_shape[dim].value()))
        if not inside:
            assert_equal(host.ptr[linear], Float32(-1))
            continue

        var box = 0
        var inner = 0
        comptime for i in range(rank):
            comptime dim = rank - i - 1 if k_major else i
            comptime copies = Int(tile_shape[dim].value()) // Int(
                desc_shape[dim].value()
            )
            box = box * copies + local[dim] // Int(desc_shape[dim].value())
            inner = inner * Int(desc_shape[i].value()) + local[i] % Int(
                desc_shape[i].value()
            )
        assert_equal(
            host.ptr[linear],
            Float32(box * _idx_product[desc_shape]() + inner + 1),
        )

    print(
        "TMA store passed: rank=",
        rank,
        " k_major=",
        k_major,
        " source_mode=",
        legacy,
        " generic=",
        generic,
    )


def test_tma_store_ranks(ctx: DeviceContext) raises:
    comptime for legacy in range(3):
        comptime for api in range(2):
            comptime generic = api == 1
            comptime for major in range(2):
                comptime k_major = major == 1
                test_tma_store_rank[
                    coord[8, 8],
                    coord[8, 8],
                    coord[12, 16],
                    k_major,
                    legacy,
                    generic,
                ](ctx)
                test_tma_store_rank[
                    coord[2, 2, 8],
                    coord[2, 2, 8],
                    coord[4, 4, 16],
                    k_major,
                    legacy,
                    generic,
                ](ctx)
                test_tma_store_rank[
                    coord[2, 2, 2, 8],
                    coord[1, 2, 2, 8],
                    coord[4, 4, 4, 16],
                    k_major,
                    legacy,
                    generic,
                ](ctx)
                test_tma_store_rank[
                    coord[2, 2, 2, 2, 8],
                    coord[1, 1, 2, 2, 8],
                    coord[4, 4, 4, 4, 16],
                    k_major,
                    legacy,
                    generic,
                ](ctx)


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
