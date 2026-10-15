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

from std.math import align_up, ceildiv
from std.sys import size_of

from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu import block_idx, thread_idx
from max.gpu.memory import fence_async_view_proxy
from max.gpu.sync import cp_async_bulk_commit_group, cp_async_bulk_wait_group
from layout import Coord, Layout, TileTensor, coord, row_major
from layout.tile_layout import Layout as TileLayout
from layout.tile_tensor import stack_allocation
from layout._fillers import arange, random
from layout._host_device_tile_tensor import HostDeviceTileTensor
from layout.tile_io import copy_dram_to_sram, copy_sram_to_dram
from layout.tma_async import (
    create_tensor_tile,
    SharedMemBarrier,
    TMATensorTile,
    _idx_product,
    create_tma_tile,
)
from std.memory import unsafe_stack_allocation
from std.testing import assert_equal


# Test loading a single 2d tile.
@__llvm_arg_metadata(tma_tile, `nvvm.grid_constant`)
def test_tma_load_kernel[
    dtype: DType,
    layout: Layout,
    tile_shape: Coord,
    thread_layout: TileLayout,
](
    dst: TileTensor[
        dtype,
        type_of(row_major[layout.shape[0].value(), layout.shape[1].value()]()),
        MutAnyOrigin,
    ],
    tma_tile: TMATensorTile[dtype, tile_shape],
):
    comptime tileM = Int(tile_shape.element_types[0].static_value.value())
    comptime tileN = Int(tile_shape.element_types[1].static_value.value())
    comptime expected_bytes = _idx_product[tile_shape]() * size_of[dtype]()

    comptime __tile_layout = Layout.row_major(tileM, tileN)
    var tile = stack_allocation[
        dtype=dtype, address_space=.SHARED, alignment=128
    ](
        row_major[
            __tile_layout.shape[0].value(), __tile_layout.shape[1].value()
        ]()
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

    var dst_tile = dst.tile[tileM, tileN](Coord(block_idx.y, block_idx.x))
    copy_sram_to_dram[thread_layout](dst_tile, tile)


# Test loading tiles along the last axis.
@__llvm_arg_metadata(tma_tile, `nvvm.grid_constant`)
def test_tma_multiple_loads_kernel[
    dtype: DType,
    layout: Layout,
    tile_shape: Coord,
    thread_layout: TileLayout,
](
    dst: TileTensor[
        dtype,
        type_of(row_major[layout.shape[0].value(), layout.shape[1].value()]()),
        MutAnyOrigin,
    ],
    tma_tile: TMATensorTile[dtype, tile_shape],
):
    comptime tileM = Int(tile_shape.element_types[0].static_value.value())
    comptime tileN = Int(tile_shape.element_types[1].static_value.value())
    comptime expected_bytes = _idx_product[tile_shape]() * size_of[dtype]()

    comptime N = layout.shape[1].value()
    comptime num_iters = ceildiv(N, tileN)

    comptime __tile_layout = Layout.row_major(tileM, tileN)
    var tile = stack_allocation[
        dtype=dtype, address_space=.SHARED, alignment=128
    ](
        row_major[
            __tile_layout.shape[0].value(), __tile_layout.shape[1].value()
        ]()
    )

    var mbar = unsafe_stack_allocation[
        1,
        SharedMemBarrier,
        address_space=.SHARED,
        alignment=8,
    ]()

    if thread_idx.x == 0:
        mbar[0].init()

    var phase: UInt32 = 0

    for i in range(num_iters):
        if thread_idx.x == 0:
            mbar[0].expect_bytes(Int32(expected_bytes))
            tma_tile.async_copy(
                tile,
                mbar[0],
                (i * tileN, block_idx.y * tileM),
            )
        # Ensure all threads sees initialized mbarrier
        barrier()
        mbar[0].wait(phase)
        phase ^= 1

        var dst_tile = dst.tile[tileM, tileN](Coord(block_idx.y, i))
        copy_sram_to_dram[thread_layout](dst_tile, tile)


def test_tma_load_row_major[
    dtype: DType,
    src_layout: Layout,
    tile_layout: Layout,
    load_along_last_dim: Bool = False,
](ctx: DeviceContext) raises:
    comptime M = src_layout.shape[0].value()
    comptime N = src_layout.shape[1].value()
    comptime tileM = tile_layout.shape[0].value()
    comptime tileN = tile_layout.shape[1].value()
    comptime M_roundup = align_up(M, tileM)
    comptime N_roundup = align_up(N, tileN)

    var src = HostDeviceTileTensor[dtype](
        row_major[src_layout.shape[0].value(), src_layout.shape[1].value()](),
        ctx,
    )
    var dst = HostDeviceTileTensor[dtype](
        row_major[M_roundup, N_roundup](), ctx
    )

    comptime if dtype == .float8_e4m3fn:
        random(src.host_tensor())
    else:
        arange(src.host_tensor(), 0)

    src.to_device()

    var tma_tensor = create_tma_tile[tileM, tileN](ctx, src.device_tensor())
    ctx.synchronize()

    comptime __tileM = Int(
        type_of(tma_tensor).tile_shape.element_types[0].static_value.value()
    )
    comptime __tileN = Int(
        type_of(tma_tensor).tile_shape.element_types[1].static_value.value()
    )
    comptime __thread_layout = row_major[__tileM, __tileN]()
    comptime if load_along_last_dim:
        comptime kernel = test_tma_multiple_loads_kernel[
            type_of(tma_tensor).dtype,
            Layout.row_major(M_roundup, N_roundup),  # dst layout
            type_of(tma_tensor).tile_shape,  # tile shape
            __thread_layout,  # thread layout
        ]
        ctx.enqueue_function[kernel](
            dst.device_tensor().as_unsafe_any_origin(),
            tma_tensor,
            grid_dim=(1, M_roundup // tileM),
            block_dim=(tileM * tileN),
        )
    else:
        comptime kernel = test_tma_load_kernel[
            type_of(tma_tensor).dtype,
            Layout.row_major(M_roundup, N_roundup),  # dst layout
            type_of(tma_tensor).tile_shape,  # tile shape
            __thread_layout,  # thread layout
        ]
        ctx.enqueue_function[kernel](
            dst.device_tensor().as_unsafe_any_origin(),
            tma_tensor,
            grid_dim=(N_roundup // tileN, M_roundup // tileM),
            block_dim=(tileM * tileN),
        )

    dst.to_host()
    var src_host = src.host_tensor()
    var dst_host = dst.host_tensor()

    # Check M x N keep the same value and others in M_roundup x N_roundup
    # are set to zeros.
    for m in range(M_roundup):
        for n in range(N_roundup):
            if m < M and n < N:
                assert_equal(
                    src_host[m, n].cast[.float32](),
                    dst_host[m, n].cast[.float32](),
                )
            else:
                assert_equal(dst_host[m, n].cast[.float32](), 0.0)
    ctx.synchronize()
    _ = src^
    _ = dst^


@__llvm_arg_metadata(tma_tile, `nvvm.grid_constant`)
def test_tma_async_store_kernel[
    dtype: DType,
    tile_shape_param: Coord,
    desc_shape_param: Coord,
    thread_layout: TileLayout,
    layout: Layout,
](
    tma_tile: TMATensorTile[dtype, tile_shape_param, desc_shape_param],
    src: TileTensor[
        dtype,
        type_of(row_major[layout.shape[0].value(), layout.shape[1].value()]()),
        MutAnyOrigin,
    ],
):
    comptime tileM = Int(tile_shape_param[0].value())
    comptime tileN = Int(tile_shape_param[1].value())
    comptime __tile_layout = Layout.row_major(tileM, tileN)
    var tile = stack_allocation[
        dtype=dtype, address_space=.SHARED, alignment=128
    ](
        row_major[
            __tile_layout.shape[0].value(), __tile_layout.shape[1].value()
        ]()
    )

    var src_tile = src.tile[tileM, tileN](Coord(block_idx.y, block_idx.x))
    copy_dram_to_sram[thread_layout](tile, src_tile)

    barrier()
    fence_async_view_proxy()

    if thread_idx.x == 0:
        tma_tile.async_store(tile, (block_idx.x * tileN, block_idx.y * tileM))
        cp_async_bulk_commit_group()

    cp_async_bulk_wait_group[0]()


@__llvm_arg_metadata(tma_tile, `nvvm.grid_constant`)
def test_tma_async_multiple_store_kernel[
    dtype: DType,
    tile_shape_param: Coord,
    thread_layout: TileLayout,
    layout: Layout,
](
    tma_tile: TMATensorTile[dtype, tile_shape_param],
    src: TileTensor[
        dtype,
        type_of(row_major[layout.shape[0].value(), layout.shape[1].value()]()),
        MutAnyOrigin,
    ],
):
    comptime tileM = Int(tile_shape_param[0].value())
    comptime tileN = Int(tile_shape_param[1].value())
    comptime __tile_layout = Layout.row_major(tileM, tileN)
    var tile = stack_allocation[
        dtype=dtype, address_space=.SHARED, alignment=128
    ](
        row_major[
            __tile_layout.shape[0].value(), __tile_layout.shape[1].value()
        ]()
    )

    comptime N = layout.shape[1].value()
    comptime num_iters = ceildiv(N, tileN)

    for i in range(num_iters):
        var src_tile = src.tile[tileM, tileN](Coord(block_idx.y, i))
        copy_dram_to_sram[thread_layout](tile, src_tile)

        barrier()
        fence_async_view_proxy()

        if thread_idx.x == 0:
            tma_tile.async_store(tile, (i * tileN, block_idx.y * tileM))
            cp_async_bulk_commit_group()
            # Wait for the TMA store to finish reading from shared memory
            # before the next iteration overwrites it.
            cp_async_bulk_wait_group[0]()

        barrier()


def test_tma_async_store[
    src_layout: Layout,
    tile_layout: Layout,
    dst_layout: Layout,
    load_along_last_dim: Bool = False,
](ctx: DeviceContext) raises:
    comptime src_M = src_layout.shape[0].value()
    comptime src_N = src_layout.shape[1].value()
    comptime tileM = tile_layout.shape[0].value()
    comptime tileN = tile_layout.shape[1].value()
    comptime dst_M = dst_layout.shape[0].value()
    comptime dst_N = dst_layout.shape[1].value()

    var src = HostDeviceTileTensor[.float32](
        row_major[src_layout.shape[0].value(), src_layout.shape[1].value()](),
        ctx,
    )
    var dst = HostDeviceTileTensor[.float32](
        row_major[dst_layout.shape[0].value(), dst_layout.shape[1].value()](),
        ctx,
    )
    arange(src.host_tensor(), 1)
    arange(dst.host_tensor(), 100001)
    src.to_device()
    dst.to_device()

    var tma_tensor = create_tma_tile[tileM, tileN](ctx, dst.device_tensor())

    ctx.synchronize()

    comptime __tileM = Int(
        type_of(tma_tensor).tile_shape.element_types[0].static_value.value()
    )
    comptime __tileN = Int(
        type_of(tma_tensor).tile_shape.element_types[1].static_value.value()
    )
    comptime __thread_layout = row_major[__tileM, __tileN]()
    comptime if load_along_last_dim:
        comptime kernel = test_tma_async_multiple_store_kernel[
            type_of(tma_tensor).dtype,
            type_of(tma_tensor).tile_shape,
            __thread_layout,
            src_layout,
        ]
        ctx.enqueue_function[kernel](
            tma_tensor,
            src.device_tensor().as_unsafe_any_origin(),
            grid_dim=(1, src_M // tileM),
            block_dim=(tileM * tileN),
        )
    else:
        comptime kernel = test_tma_async_store_kernel[
            type_of(tma_tensor).dtype,
            type_of(tma_tensor).tile_shape,
            type_of(tma_tensor).desc_shape,
            __thread_layout,
            src_layout,
        ]
        ctx.enqueue_function[kernel](
            tma_tensor,
            src.device_tensor().as_unsafe_any_origin(),
            grid_dim=(src_N // tileN, src_M // tileM),
            block_dim=(tileM * tileN),
        )
    ctx.synchronize()

    dst.to_host()
    var src_host = src.host_tensor()
    var dst_host = dst.host_tensor()

    # Check M x N keep the same value
    for m in range(dst_M):
        for n in range(dst_N):
            assert_equal(
                src_host[m, n].cast[.float32](),
                dst_host[m, n].cast[.float32](),
            )

    ctx.synchronize()
    _ = src^
    _ = dst^


# Test loading tiles along the last axis.
@__llvm_arg_metadata(a_tma_tile, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_tile, `nvvm.grid_constant`)
def test_tma_loads_two_buffers_kernel[
    dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    a_tile_shape: Coord,
    b_tile_shape: Coord,
    a_thread_layout: TileLayout,
    b_thread_layout: TileLayout,
](
    a_dst: TileTensor[
        dtype,
        type_of(
            row_major[a_layout.shape[0].value(), a_layout.shape[1].value()]()
        ),
        MutAnyOrigin,
    ],
    b_dst: TileTensor[
        dtype,
        type_of(
            row_major[b_layout.shape[0].value(), b_layout.shape[1].value()]()
        ),
        MutAnyOrigin,
    ],
    a_tma_tile: TMATensorTile[dtype, a_tile_shape],
    b_tma_tile: TMATensorTile[dtype, b_tile_shape],
):
    comptime tileM = Int(a_tile_shape[0].value())
    comptime tileN = Int(a_tile_shape[1].value())
    comptime expected_bytes = _idx_product[a_tile_shape]() * size_of[dtype]()

    comptime N = a_layout.shape[1].value()
    comptime num_iters = ceildiv(N, tileN)

    comptime __a_tile_layout = Layout.row_major(tileM, tileN)
    var a_tile = stack_allocation[
        dtype=dtype, address_space=.SHARED, alignment=128
    ](
        row_major[
            __a_tile_layout.shape[0].value(), __a_tile_layout.shape[1].value()
        ]()
    )

    comptime __b_tile_layout = Layout.row_major(
        Int(b_tile_shape[0].value()), Int(b_tile_shape[1].value())
    )
    var b_tile = stack_allocation[
        dtype=dtype, address_space=.SHARED, alignment=128
    ](
        row_major[
            __b_tile_layout.shape[0].value(), __b_tile_layout.shape[1].value()
        ]()
    )

    var mbar = unsafe_stack_allocation[
        1,
        SharedMemBarrier,
        address_space=.SHARED,
        alignment=8,
    ]()

    if thread_idx.x == 0:
        mbar[0].init()

    var phase: UInt32 = 0

    for i in range(num_iters):
        if thread_idx.x == 0:
            mbar[0].expect_bytes(Int32(expected_bytes * 2))
            a_tma_tile.async_copy(
                a_tile,
                mbar[0],
                (i * tileN, block_idx.y * tileM),
            )
            b_tma_tile.async_copy(
                b_tile,
                mbar[0],
                (i * tileN, block_idx.y * tileM),
            )

        # Ensure all threads sees initialized mbarrier
        barrier()

        mbar[0].wait(phase)
        phase ^= 1

        var a_dst_tile = a_dst.tile[tileM, tileN](Coord(block_idx.y, i))
        var b_dst_tile = b_dst.tile[tileM, tileN](Coord(block_idx.y, i))
        copy_sram_to_dram[a_thread_layout](a_dst_tile, a_tile)
        copy_sram_to_dram[b_thread_layout](b_dst_tile, b_tile)


def test_tma_load_two_buffers_row_major[
    src_layout: Layout, tile_layout: Layout, load_along_last_dim: Bool = False
](ctx: DeviceContext) raises:
    comptime M = src_layout.shape[0].value()
    comptime N = src_layout.shape[1].value()
    comptime tileM = tile_layout.shape[0].value()
    comptime tileN = tile_layout.shape[1].value()
    comptime M_roundup = align_up(M, tileM)
    comptime N_roundup = align_up(N, tileN)

    var a_src = HostDeviceTileTensor[.float32](
        row_major[src_layout.shape[0].value(), src_layout.shape[1].value()](),
        ctx,
    )
    var b_src = HostDeviceTileTensor[.float32](
        row_major[src_layout.shape[0].value(), src_layout.shape[1].value()](),
        ctx,
    )

    var a_dst = HostDeviceTileTensor[.float32](
        row_major[M_roundup, N_roundup](), ctx
    )

    var b_dst = HostDeviceTileTensor[.float32](
        row_major[M_roundup, N_roundup](), ctx
    )

    arange(a_src.host_tensor(), 1)
    arange(b_src.host_tensor(), 1)
    a_src.to_device()
    b_src.to_device()

    var a_tma_tensor = create_tma_tile[tileM, tileN](ctx, a_src.device_tensor())
    var b_tma_tensor = create_tma_tile[tileM, tileN](ctx, b_src.device_tensor())
    ctx.synchronize()

    comptime __a_tileM = Int(
        type_of(a_tma_tensor).tile_shape.element_types[0].static_value.value()
    )
    comptime __a_tileN = Int(
        type_of(a_tma_tensor).tile_shape.element_types[1].static_value.value()
    )
    comptime __b_tileM = Int(
        type_of(b_tma_tensor).tile_shape.element_types[0].static_value.value()
    )
    comptime __b_tileN = Int(
        type_of(b_tma_tensor).tile_shape.element_types[1].static_value.value()
    )
    comptime kernel = test_tma_loads_two_buffers_kernel[
        type_of(a_tma_tensor).dtype,
        Layout.row_major(M_roundup, N_roundup),  # dst layout
        Layout.row_major(M_roundup, N_roundup),  # dst layout
        type_of(a_tma_tensor).tile_shape,
        type_of(b_tma_tensor).tile_shape,
        row_major[__a_tileM, __a_tileN](),  # thread layout
        row_major[__b_tileM, __b_tileN](),  # thread layout
    ]
    ctx.enqueue_function[kernel](
        a_dst.device_tensor().as_unsafe_any_origin(),
        b_dst.device_tensor().as_unsafe_any_origin(),
        a_tma_tensor,
        b_tma_tensor,
        grid_dim=(1, M_roundup // tileM),
        block_dim=(tileM * tileN),
    )

    a_dst.to_host()
    b_dst.to_host()
    var a_src_host = a_src.host_tensor()
    var a_dst_host = a_dst.host_tensor()

    var b_src_host = b_src.host_tensor()
    var b_dst_host = b_dst.host_tensor()

    # Check M x N keep the same value and others in M_roundup x N_roundup
    # are set to zeros.
    for m in range(M_roundup):
        for n in range(N_roundup):
            if m < M and n < N:
                assert_equal(
                    a_src_host[m, n].cast[.float32](),
                    a_dst_host[m, n].cast[.float32](),
                )

                assert_equal(
                    b_src_host[m, n].cast[.float32](),
                    b_dst_host[m, n].cast[.float32](),
                )

            else:
                assert_equal(a_dst_host[m, n].cast[.float32](), 0.0)
                assert_equal(b_dst_host[m, n].cast[.float32](), 0.0)
    ctx.synchronize()
    _ = a_src^
    _ = a_dst^

    _ = b_src^
    _ = b_dst^


# Test loading tiles along the last axis.
@__llvm_arg_metadata(a_tma_dst_tile, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_dst_tile, `nvvm.grid_constant`)
@__llvm_arg_metadata(a_tma_src_tile, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_src_tile, `nvvm.grid_constant`)
def test_tma_loads_and_store_two_buffers_kernel[
    dtype: DType,
    a_tile_shape: Coord,
    b_tile_shape: Coord,
    a_desc_shape: Coord,
    b_desc_shape: Coord,
    /,
    *,
    a_layout: Layout,
    b_layout: Layout,
](
    a_tma_dst_tile: TMATensorTile[dtype, a_tile_shape, a_desc_shape],
    b_tma_dst_tile: TMATensorTile[dtype, b_tile_shape, b_desc_shape],
    a_tma_src_tile: TMATensorTile[dtype, a_tile_shape, a_desc_shape],
    b_tma_src_tile: TMATensorTile[dtype, b_tile_shape, b_desc_shape],
):
    comptime tileM = Int(a_tile_shape[0].value())
    comptime tileN = Int(a_tile_shape[1].value())
    comptime expected_bytes = _idx_product[a_tile_shape]() * size_of[dtype]()

    comptime N = a_layout.shape[1].value()
    comptime num_iters = ceildiv(N, tileN)

    comptime __a_tile_layout = Layout.row_major(tileM, tileN)
    var a_tile = stack_allocation[
        dtype=dtype, address_space=.SHARED, alignment=128
    ](
        row_major[
            __a_tile_layout.shape[0].value(), __a_tile_layout.shape[1].value()
        ]()
    )

    comptime __b_tile_layout = Layout.row_major(
        Int(b_tile_shape[0].value()), Int(b_tile_shape[1].value())
    )
    var b_tile = stack_allocation[
        dtype=dtype, address_space=.SHARED, alignment=128
    ](
        row_major[
            __b_tile_layout.shape[0].value(), __b_tile_layout.shape[1].value()
        ]()
    )

    var mbar = unsafe_stack_allocation[
        1,
        SharedMemBarrier,
        address_space=.SHARED,
        alignment=8,
    ]()

    if thread_idx.x == 0:
        mbar[0].init()

    var phase: UInt32 = 0

    for i in range(num_iters):
        if thread_idx.x == 0:
            mbar[0].expect_bytes(Int32(expected_bytes * 2))
            a_tma_src_tile.async_copy(
                a_tile,
                mbar[0],
                (i * tileN, block_idx.y * tileM),
            )
            b_tma_src_tile.async_copy(
                b_tile,
                mbar[0],
                (i * tileN, block_idx.y * tileM),
            )

        # Ensure all threads sees initialized mbarrier
        barrier()

        mbar[0].wait(phase)
        phase ^= 1

        fence_async_view_proxy()

        if thread_idx.x == 0:
            a_tma_dst_tile.async_store(a_tile, (i * tileN, block_idx.y * tileM))
            b_tma_dst_tile.async_store(b_tile, (i * tileN, block_idx.y * tileM))
            cp_async_bulk_commit_group()

        cp_async_bulk_wait_group[0]()


def test_tma_load_and_store_two_buffers_row_major[
    src_layout: Layout, tile_layout: Layout, dst_layout: Layout
](ctx: DeviceContext) raises:
    comptime M = src_layout.shape[0].value()
    comptime N = src_layout.shape[1].value()
    comptime tileM = tile_layout.shape[0].value()
    comptime tileN = tile_layout.shape[1].value()
    comptime dst_M = dst_layout.shape[0].value()
    comptime dst_N = dst_layout.shape[1].value()

    var a_src = HostDeviceTileTensor[.float32](
        row_major[src_layout.shape[0].value(), src_layout.shape[1].value()](),
        ctx,
    )
    var b_src = HostDeviceTileTensor[.float32](
        row_major[src_layout.shape[0].value(), src_layout.shape[1].value()](),
        ctx,
    )
    var a_dst = HostDeviceTileTensor[.float32](
        row_major[dst_layout.shape[0].value(), dst_layout.shape[1].value()](),
        ctx,
    )
    var b_dst = HostDeviceTileTensor[.float32](
        row_major[dst_layout.shape[0].value(), dst_layout.shape[1].value()](),
        ctx,
    )

    # Initialize destinations to known values.
    comptime a_dst_value = 1.5
    comptime b_dst_value = 1.25

    var a_dst_host = a_dst.host_tensor()
    var b_dst_host = b_dst.host_tensor()
    for m in range(dst_M):
        for n in range(dst_N):
            a_dst_host[m, n] = a_dst_value
            b_dst_host[m, n] = b_dst_value

    arange(a_src.host_tensor(), 1)
    arange(b_src.host_tensor(), 1)
    a_src.to_device()
    b_src.to_device()
    a_dst.to_device()
    b_dst.to_device()

    var a_tma_src_tensor = create_tensor_tile[coord[tileM, tileN]](
        ctx, a_src.device_tensor()
    )
    var b_tma_src_tensor = create_tensor_tile[coord[tileM, tileN]](
        ctx, b_src.device_tensor()
    )
    var a_tma_dst_tensor = create_tensor_tile[coord[tileM, tileN]](
        ctx, a_dst.device_tensor()
    )
    var b_tma_dst_tensor = create_tensor_tile[coord[tileM, tileN]](
        ctx, b_dst.device_tensor()
    )
    ctx.synchronize()

    comptime kernel = test_tma_loads_and_store_two_buffers_kernel[
        type_of(a_tma_src_tensor).dtype,
        type_of(a_tma_src_tensor).tile_shape,
        type_of(b_tma_src_tensor).tile_shape,
        type_of(a_tma_src_tensor).desc_shape,
        type_of(b_tma_src_tensor).desc_shape,
        a_layout=dst_layout,  # dst layout
        b_layout=dst_layout,  # dst layout
    ]
    ctx.enqueue_function[kernel](
        a_tma_dst_tensor,
        b_tma_dst_tensor,
        a_tma_src_tensor,
        b_tma_src_tensor,
        grid_dim=(1, dst_M // tileM),
        block_dim=(tileM * tileN),
    )

    a_dst.to_host()
    b_dst.to_host()
    var a_src_host = a_src.host_tensor()
    a_dst_host = a_dst.host_tensor()

    var b_src_host = b_src.host_tensor()
    b_dst_host = b_dst.host_tensor()

    for m in range(dst_M):
        for n in range(dst_N):
            if m < M and n < N:
                assert_equal(
                    a_src_host[m, n].cast[.float32](),
                    a_dst_host[m, n].cast[.float32](),
                )

                assert_equal(
                    b_src_host[m, n].cast[.float32](),
                    b_dst_host[m, n].cast[.float32](),
                )

            else:
                assert_equal(a_dst_host[m, n].cast[.float32](), a_dst_value)
                assert_equal(b_dst_host[m, n].cast[.float32](), b_dst_value)

    ctx.synchronize()
    _ = a_src^
    _ = a_dst^

    _ = b_src^
    _ = b_dst^


def main() raises:
    with DeviceContext() as ctx:
        print("test_tma_load_f32")
        test_tma_load_row_major[
            dtype=DType.float32,
            src_layout=Layout.row_major(8, 8),
            tile_layout=Layout.row_major(4, 4),
        ](ctx)
        test_tma_load_row_major[
            dtype=DType.float32,
            src_layout=Layout.row_major(9, 24),
            tile_layout=Layout.row_major(3, 8),
        ](ctx)
        print("test_tma_load_oob_fill_f32")
        test_tma_load_row_major[
            dtype=DType.float32,
            src_layout=Layout.row_major(7, 8),
            tile_layout=Layout.row_major(4, 4),
        ](ctx)
        test_tma_load_row_major[
            dtype=DType.float32,
            src_layout=Layout.row_major(10, 12),
            tile_layout=Layout.row_major(4, 8),
        ](ctx)

        print("test_tma_multiple_loads_f32")
        test_tma_load_row_major[
            dtype=DType.float32,
            src_layout=Layout.row_major(12, 16),
            tile_layout=Layout.row_major(4, 4),
            load_along_last_dim=True,
        ](ctx)
        test_tma_load_row_major[
            dtype=DType.float32,
            src_layout=Layout.row_major(24, 80),
            tile_layout=Layout.row_major(3, 16),
            load_along_last_dim=True,
        ](ctx)

        print("test_tma_multiple_loads_oob_fill_f32")
        test_tma_load_row_major[
            dtype=DType.float32,
            src_layout=Layout.row_major(6, 20),
            tile_layout=Layout.row_major(4, 8),
            load_along_last_dim=True,
        ](ctx)
        test_tma_load_row_major[
            dtype=DType.float32,
            src_layout=Layout.row_major(9, 60),
            tile_layout=Layout.row_major(8, 16),
            load_along_last_dim=True,
        ](ctx)

        print("test_tma_load_f8e4m3fn")
        test_tma_load_row_major[
            dtype=DType.float8_e4m3fn,
            src_layout=Layout.row_major(8, 32),
            tile_layout=Layout.row_major(4, 16),
        ](ctx)
        test_tma_load_row_major[
            dtype=DType.float8_e4m3fn,
            src_layout=Layout.row_major(9, 48),
            tile_layout=Layout.row_major(3, 16),
        ](ctx)
        print("test_tma_load_oob_fill_f8e4m3fn")
        test_tma_load_row_major[
            dtype=DType.float8_e4m3fn,
            src_layout=Layout.row_major(7, 32),
            tile_layout=Layout.row_major(4, 16),
        ](ctx)
        test_tma_load_row_major[
            dtype=DType.float8_e4m3fn,
            src_layout=Layout.row_major(10, 48),
            tile_layout=Layout.row_major(4, 32),
        ](ctx)

        print("test_tma_multiple_loads_f8e4m3fn")
        test_tma_load_row_major[
            dtype=DType.float8_e4m3fn,
            src_layout=Layout.row_major(12, 64),
            tile_layout=Layout.row_major(4, 16),
            load_along_last_dim=True,
        ](ctx)
        test_tma_load_row_major[
            dtype=DType.float8_e4m3fn,
            src_layout=Layout.row_major(24, 160),
            tile_layout=Layout.row_major(3, 64),
            load_along_last_dim=True,
        ](ctx)

        print("test_tma_multiple_loads_oob_fill_f8e4m3fn")
        test_tma_load_row_major[
            dtype=DType.float8_e4m3fn,
            src_layout=Layout.row_major(6, 80),
            tile_layout=Layout.row_major(4, 16),
            load_along_last_dim=True,
        ](ctx)
        test_tma_load_row_major[
            dtype=DType.float8_e4m3fn,
            src_layout=Layout.row_major(9, 240),
            tile_layout=Layout.row_major(8, 64),
            load_along_last_dim=True,
        ](ctx)

        print("test_tma_async_store")
        test_tma_async_store[
            src_layout=Layout.row_major(8, 8),
            tile_layout=Layout.row_major(4, 4),
            dst_layout=Layout.row_major(8, 8),
        ](ctx)
        test_tma_async_store[
            src_layout=Layout.row_major(32, 24),
            tile_layout=Layout.row_major(16, 8),
            dst_layout=Layout.row_major(32, 24),
        ](ctx)

        print("test_tma_multiple_async_store")
        test_tma_async_store[
            src_layout=Layout.row_major(8, 8),
            tile_layout=Layout.row_major(4, 4),
            dst_layout=Layout.row_major(8, 8),
            load_along_last_dim=True,
        ](ctx)
        test_tma_async_store[
            src_layout=Layout.row_major(9, 24),
            tile_layout=Layout.row_major(3, 8),
            dst_layout=Layout.row_major(9, 24),
            load_along_last_dim=True,
        ](ctx)

        print("test_tma_async_store_oob")
        test_tma_async_store[
            src_layout=Layout.row_major(8, 8),
            tile_layout=Layout.row_major(8, 8),
            dst_layout=Layout.row_major(6, 8),
        ](ctx)
        test_tma_async_store[
            src_layout=Layout.row_major(32, 8),
            tile_layout=Layout.row_major(8, 8),
            dst_layout=Layout.row_major(26, 8),
        ](ctx)
        test_tma_async_store[
            src_layout=Layout.row_major(8, 8),
            tile_layout=Layout.row_major(8, 8),
            dst_layout=Layout.row_major(6, 4),
        ](ctx)
        test_tma_async_store[
            src_layout=Layout.row_major(32, 16),
            tile_layout=Layout.row_major(8, 8),
            dst_layout=Layout.row_major(26, 12),
        ](ctx)

        print("test_tma_load_two_buffer_row_major")
        test_tma_load_two_buffers_row_major[
            src_layout=Layout.row_major(32, 64),
            tile_layout=Layout.row_major(8, 16),
            load_along_last_dim=True,
        ](ctx)
        test_tma_load_two_buffers_row_major[
            src_layout=Layout.row_major(9, 60),
            tile_layout=Layout.row_major(8, 16),
        ](ctx)
        print("test_tma_load_and_store_two_buffer")
        test_tma_load_and_store_two_buffers_row_major[
            src_layout=Layout.row_major(32, 64),
            tile_layout=Layout.row_major(8, 16),
            dst_layout=Layout.row_major(32, 64),
        ](ctx)
        test_tma_load_and_store_two_buffers_row_major[
            src_layout=Layout.row_major(32, 64),
            tile_layout=Layout.row_major(16, 16),
            dst_layout=Layout.row_major(40, 64),
        ](ctx)
