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

from math import align_up, ceildiv
from sys import size_of

from gpu import barrier
from gpu.host import DeviceContext
from gpu.id import block_idx, thread_idx
from gpu.memory import ReduceOp, fence_async_view_proxy
from gpu.sync import cp_async_bulk_commit_group, cp_async_bulk_wait_group
from layout import Layout, LayoutTensor
from layout._fillers import arange, random
from layout._utils import ManagedLayoutTensor
from layout.layout_tensor import copy_dram_to_sram, copy_sram_to_dram
from layout.tma_async import SharedMemBarrier, TMATensorTile, create_tma_tile
from memory import stack_allocation
from memory.pointer import _GPUAddressSpace
from testing import assert_equal

from utils.index import Index


# Test loading a single 2d tile.
@__llvm_arg_metadata(tma_tile, `nvvm.grid_constant`)
fn test_tma_load_kernel[
    dtype: DType,
    layout: Layout,
    tile_layout: Layout,
    thread_layout: Layout,
](
    dst: LayoutTensor[dtype, layout, MutableAnyOrigin],
    tma_tile: TMATensorTile[dtype, tile_layout],
):
    alias tileM = tile_layout.shape[0].value()
    alias tileN = tile_layout.shape[1].value()
    alias expected_bytes = tile_layout.size() * size_of[dtype]()

    tile = LayoutTensor[
        dtype,
        tile_layout,
        MutableAnyOrigin,
        address_space = _GPUAddressSpace.SHARED,
        alignment=128,
    ].stack_allocation()

    mbar = stack_allocation[
        1,
        SharedMemBarrier,
        address_space = _GPUAddressSpace.SHARED,
        alignment=8,
    ]()

    if thread_idx.x == 0:
        mbar[0].init()
        mbar[0].expect_bytes(expected_bytes)
        tma_tile.async_copy(
            tile,
            mbar[0],
            (block_idx.x * UInt(tileN), block_idx.y * UInt(tileM)),
        )
    # Ensure all threads sees initialized mbarrier
    barrier()
    mbar[0].wait()

    dst_tile = dst.tile[tileM, tileN](block_idx.y, block_idx.x)
    copy_sram_to_dram[thread_layout](dst_tile, tile)


# Test loading tiles along the last axis.
@__llvm_arg_metadata(tma_tile, `nvvm.grid_constant`)
fn test_tma_multiple_loads_kernel[
    dtype: DType,
    layout: Layout,
    tile_layout: Layout,
    thread_layout: Layout,
](
    dst: LayoutTensor[dtype, layout, MutableAnyOrigin],
    tma_tile: TMATensorTile[dtype, tile_layout],
):
    alias tileM = tile_layout.shape[0].value()
    alias tileN = tile_layout.shape[1].value()
    alias expected_bytes = tile_layout.size() * size_of[dtype]()

    alias N = layout.shape[1].value()
    alias num_iters = ceildiv(N, tileN)

    tile = LayoutTensor[
        dtype,
        tile_layout,
        MutableAnyOrigin,
        address_space = _GPUAddressSpace.SHARED,
        alignment=128,
    ].stack_allocation()

    mbar = stack_allocation[
        1,
        SharedMemBarrier,
        address_space = _GPUAddressSpace.SHARED,
        alignment=8,
    ]()

    if thread_idx.x == 0:
        mbar[0].init()

    var phase: UInt32 = 0

    for i in range(num_iters):
        if thread_idx.x == 0:
            mbar[0].expect_bytes(expected_bytes)
            tma_tile.async_copy(
                tile,
                mbar[0],
                (UInt(i) * UInt(tileN), block_idx.y * UInt(tileM)),
            )
        # Ensure all threads sees initialized mbarrier
        barrier()
        mbar[0].wait(phase)
        phase ^= 1

        dst_tile = dst.tile[tileM, tileN](block_idx.y, i)
        copy_sram_to_dram[thread_layout](dst_tile, tile)


def test_tma_load_row_major[
    dtype: DType,
    src_layout: Layout,
    tile_layout: Layout,
    load_along_last_dim: Bool = False,
](ctx: DeviceContext):
    alias M = src_layout.shape[0].value()
    alias N = src_layout.shape[1].value()
    alias tileM = tile_layout.shape[0].value()
    alias tileN = tile_layout.shape[1].value()
    alias M_roundup = align_up(M, tileM)
    alias N_roundup = align_up(N, tileN)

    var src = ManagedLayoutTensor[dtype, src_layout](ctx)
    var dst = ManagedLayoutTensor[
        dtype, Layout.row_major(M_roundup, N_roundup)
    ](ctx)

    @parameter
    if dtype is DType.float8_e4m3fn:
        random(src.tensor())
    else:
        arange(src.tensor(), 0)

    var tma_tensor = create_tma_tile[tileM, tileN](ctx, src.device_tensor())
    ctx.synchronize()

    @parameter
    if load_along_last_dim:
        alias kernel = test_tma_multiple_loads_kernel[
            __type_of(tma_tensor).dtype,
            Layout.row_major(M_roundup, N_roundup),  # dst layout
            __type_of(tma_tensor).layout,  # smem layout
            __type_of(tma_tensor).layout,  # thread layout
        ]
        ctx.enqueue_function[kernel](
            dst.device_tensor(),
            tma_tensor,
            grid_dim=(1, M_roundup // tileM),
            block_dim=(tileM * tileN),
        )
    else:
        alias kernel = test_tma_load_kernel[
            __type_of(tma_tensor).dtype,
            Layout.row_major(M_roundup, N_roundup),  # dst layout
            __type_of(tma_tensor).layout,  # smem layout
            __type_of(tma_tensor).layout,  # thread layout
        ]
        ctx.enqueue_function[kernel](
            dst.device_tensor(),
            tma_tensor,
            grid_dim=(N_roundup // tileN, M_roundup // tileM),
            block_dim=(tileM * tileN),
        )

    src_host = src.tensor()
    dst_host = dst.tensor()

    # Check M x N keep the same value and others in M_roundup x N_roundup
    # are set to zeros.
    for m in range(M_roundup):
        for n in range(N_roundup):
            if m < M and n < N:
                assert_equal(
                    src_host[m, n].cast[DType.float32](),
                    dst_host[m, n].cast[DType.float32](),
                )
            else:
                assert_equal(dst_host[m, n].cast[DType.float32](), 0.0)
    ctx.synchronize()
    _ = src^
    _ = dst^


@__llvm_arg_metadata(tma_tile, `nvvm.grid_constant`)
fn test_tma_async_store_kernel[
    dtype: DType,
    tile_layout: Layout,
    desc_layout: Layout,
    thread_layout: Layout,
    layout: Layout,
](
    tma_tile: TMATensorTile[dtype, tile_layout, desc_layout],
    src: LayoutTensor[dtype, layout, MutableAnyOrigin],
):
    alias tileM = tile_layout.shape[0].value()
    alias tileN = tile_layout.shape[1].value()
    tile = LayoutTensor[
        dtype,
        tile_layout,
        MutableAnyOrigin,
        address_space = _GPUAddressSpace.SHARED,
        alignment=128,
    ].stack_allocation[]()

    src_tile = src.tile[tileM, tileN](block_idx.y, block_idx.x)
    copy_dram_to_sram[thread_layout](tile, src_tile)

    barrier()
    fence_async_view_proxy()

    if thread_idx.x == 0:
        tma_tile.async_store(
            tile, (block_idx.x * UInt(tileN), block_idx.y * UInt(tileM))
        )
        cp_async_bulk_commit_group()

    cp_async_bulk_wait_group[0]()


@__llvm_arg_metadata(tma_tile, `nvvm.grid_constant`)
fn test_tma_async_multiple_store_kernel[
    dtype: DType, tile_layout: Layout, thread_layout: Layout, layout: Layout
](
    tma_tile: TMATensorTile[dtype, tile_layout],
    src: LayoutTensor[dtype, layout, MutableAnyOrigin],
):
    alias tileM = tile_layout.shape[0].value()
    alias tileN = tile_layout.shape[1].value()
    tile = LayoutTensor[
        dtype,
        tile_layout,
        MutableAnyOrigin,
        address_space = _GPUAddressSpace.SHARED,
        alignment=128,
    ].stack_allocation[]()

    alias N = layout.shape[1].value()
    alias num_iters = ceildiv(N, tileN)

    for i in range(num_iters):
        src_tile = src.tile[tileM, tileN](block_idx.y, i)
        copy_dram_to_sram[thread_layout](tile, src_tile)

        barrier()
        fence_async_view_proxy()

        if thread_idx.x == 0:
            tma_tile.async_store(
                tile, (UInt(i) * UInt(tileN), block_idx.y * UInt(tileM))
            )
            cp_async_bulk_commit_group()

    cp_async_bulk_wait_group[0]()


def test_tma_async_store[
    src_layout: Layout,
    tile_layout: Layout,
    dst_layout: Layout,
    load_along_last_dim: Bool = False,
](ctx: DeviceContext):
    alias src_M = src_layout.shape[0].value()
    alias src_N = src_layout.shape[1].value()
    alias tileM = tile_layout.shape[0].value()
    alias tileN = tile_layout.shape[1].value()
    alias dst_M = dst_layout.shape[0].value()
    alias dst_N = dst_layout.shape[1].value()

    var src = ManagedLayoutTensor[DType.float32, src_layout](ctx)
    var dst = ManagedLayoutTensor[DType.float32, dst_layout](ctx)
    arange(src.tensor(), 1)
    arange(dst.tensor(), 100001)
    var tma_tensor = create_tma_tile[tileM, tileN](ctx, dst.device_tensor())

    ctx.synchronize()

    @parameter
    if load_along_last_dim:
        alias kernel = test_tma_async_multiple_store_kernel[
            __type_of(tma_tensor).dtype,
            __type_of(tma_tensor).layout,
            __type_of(tma_tensor).layout,
            src_layout,
        ]
        ctx.enqueue_function[kernel](
            tma_tensor,
            src.device_tensor(),
            grid_dim=(1, src_M // tileM),
            block_dim=(tileM * tileN),
        )
    else:
        alias kernel = test_tma_async_store_kernel[
            __type_of(tma_tensor).dtype,
            __type_of(tma_tensor).layout,
            __type_of(tma_tensor).desc_layout,
            __type_of(tma_tensor).layout,
            src_layout,
        ]
        ctx.enqueue_function[kernel](
            tma_tensor,
            src.device_tensor(),
            grid_dim=(src_N // tileN, src_M // tileM),
            block_dim=(tileM * tileN),
        )
    ctx.synchronize()

    src_host = src.tensor()
    dst_host = dst.tensor()

    # Check M x N keep the same value
    for m in range(dst_M):
        for n in range(dst_N):
            assert_equal(
                src_host[m, n].cast[DType.float32](),
                dst_host[m, n].cast[DType.float32](),
            )

    ctx.synchronize()
    _ = src^
    _ = dst^


@__llvm_arg_metadata(tma_tile, `nvvm.grid_constant`)
fn test_tma_async_reduce_kernel[
    dtype: DType, tile_layout: Layout, thread_layout: Layout, layout: Layout
](
    tma_tile: TMATensorTile[dtype, tile_layout],
    src: LayoutTensor[dtype, layout, MutableAnyOrigin],
):
    alias tileM = tile_layout.shape[0].value()
    alias tileN = tile_layout.shape[1].value()
    tile = LayoutTensor[
        dtype,
        tile_layout,
        MutableAnyOrigin,
        address_space = _GPUAddressSpace.SHARED,
        alignment=128,
    ].stack_allocation[]()

    src_tile = src.tile[tileM, tileN](block_idx.y, block_idx.x)
    copy_dram_to_sram[thread_layout](tile, src_tile)

    barrier()
    fence_async_view_proxy()

    if thread_idx.x == 0:
        tma_tile.async_reduce[reduction_kind = ReduceOp.ADD](
            tile, (block_idx.x * UInt(tileN), block_idx.y * UInt(tileM))
        )
        cp_async_bulk_commit_group()

    cp_async_bulk_wait_group[0]()


@__llvm_arg_metadata(tma_tile, `nvvm.grid_constant`)
fn test_tma_async_multiple_reduce_kernel[
    dtype: DType, tile_layout: Layout, thread_layout: Layout, layout: Layout
](
    tma_tile: TMATensorTile[dtype, tile_layout],
    src: LayoutTensor[dtype, layout, MutableAnyOrigin],
):
    alias tileM = tile_layout.shape[0].value()
    alias tileN = tile_layout.shape[1].value()
    tile = LayoutTensor[
        dtype,
        tile_layout,
        MutableAnyOrigin,
        address_space = _GPUAddressSpace.SHARED,
        alignment=128,
    ].stack_allocation[]()

    alias N = layout.shape[1].value()
    alias num_iters = ceildiv(N, tileN)

    for i in range(num_iters):
        src_tile = src.tile[tileM, tileN](block_idx.y, i)
        copy_dram_to_sram[thread_layout](tile, src_tile)

        barrier()
        fence_async_view_proxy()

        if thread_idx.x == 0:
            tma_tile.async_reduce[reduction_kind = ReduceOp.ADD](
                tile, (UInt(i) * UInt(tileN), block_idx.y * UInt(tileM))
            )
            cp_async_bulk_commit_group()

    cp_async_bulk_wait_group[0]()


def test_tma_async_reduce[
    src_layout: Layout,
    tile_layout: Layout,
    dst_layout: Layout,
    load_along_last_dim: Bool = False,
](ctx: DeviceContext):
    alias src_M = src_layout.shape[0].value()
    alias src_N = src_layout.shape[1].value()
    alias tileM = tile_layout.shape[0].value()
    alias tileN = tile_layout.shape[1].value()
    alias dst_M = dst_layout.shape[0].value()
    alias dst_N = dst_layout.shape[1].value()

    var src = ManagedLayoutTensor[DType.float32, src_layout](ctx)
    var dst = ManagedLayoutTensor[DType.float32, dst_layout](ctx)
    arange(src.tensor(), 1)
    arange(dst.tensor(), 3546)
    var tma_tensor = create_tma_tile[tileM, tileN](ctx, dst.device_tensor())

    ctx.synchronize()

    @parameter
    if load_along_last_dim:
        alias kernel = test_tma_async_multiple_reduce_kernel[
            __type_of(tma_tensor).dtype,
            __type_of(tma_tensor).layout,
            __type_of(tma_tensor).layout,
            src_layout,
        ]
        ctx.enqueue_function[kernel](
            tma_tensor,
            src.device_tensor(),
            grid_dim=(1, src_M // tileM),
            block_dim=(tileM * tileN),
        )
    else:
        alias kernel = test_tma_async_reduce_kernel[
            __type_of(tma_tensor).dtype,
            __type_of(tma_tensor).layout,
            __type_of(tma_tensor).layout,
            src_layout,
        ]
        ctx.enqueue_function[kernel](
            tma_tensor,
            src.device_tensor(),
            grid_dim=(src_N // tileN, src_M // tileM),
            block_dim=(tileM * tileN),
        )
    ctx.synchronize()

    src_host = src.tensor()
    dst_host = dst.tensor()

    # Check M x N keep the same value and others in M_roundup x N_roundup
    for m in range(dst_M):
        for n in range(dst_N):
            assert_equal(
                src_host[m, n].cast[DType.float32]() + 3546 + m * dst_N + n,
                dst_host[m, n].cast[DType.float32](),
            )

    ctx.synchronize()
    _ = src^
    _ = dst^


# Test loading tiles along the last axis.
@__llvm_arg_metadata(a_tma_tile, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_tile, `nvvm.grid_constant`)
fn test_tma_loads_two_buffers_kernel[
    dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    a_tile_layout: Layout,
    b_tile_layout: Layout,
    a_thread_layout: Layout,
    b_thread_layout: Layout,
](
    a_dst: LayoutTensor[dtype, a_layout, MutableAnyOrigin],
    b_dst: LayoutTensor[dtype, b_layout, MutableAnyOrigin],
    a_tma_tile: TMATensorTile[dtype, a_tile_layout],
    b_tma_tile: TMATensorTile[dtype, b_tile_layout],
):
    alias tileM = a_tile_layout.shape[0].value()
    alias tileN = a_tile_layout.shape[1].value()
    alias expected_bytes = a_tile_layout.size() * size_of[dtype]()

    alias N = a_layout.shape[1].value()
    alias num_iters = ceildiv(N, tileN)

    a_tile = LayoutTensor[
        dtype,
        a_tile_layout,
        MutableAnyOrigin,
        address_space = _GPUAddressSpace.SHARED,
        alignment=128,
    ].stack_allocation()

    b_tile = LayoutTensor[
        dtype,
        b_tile_layout,
        MutableAnyOrigin,
        address_space = _GPUAddressSpace.SHARED,
        alignment=128,
    ].stack_allocation()

    mbar = stack_allocation[
        1,
        SharedMemBarrier,
        address_space = _GPUAddressSpace.SHARED,
        alignment=8,
    ]()

    if thread_idx.x == 0:
        mbar[0].init()

    var phase: UInt32 = 0

    for i in range(num_iters):
        if thread_idx.x == 0:
            mbar[0].expect_bytes(expected_bytes * 2)
            a_tma_tile.async_copy(
                a_tile,
                mbar[0],
                (UInt(i) * UInt(tileN), block_idx.y * UInt(tileM)),
            )
            b_tma_tile.async_copy(
                b_tile,
                mbar[0],
                (UInt(i) * UInt(tileN), block_idx.y * UInt(tileM)),
            )

        # Ensure all threads sees initialized mbarrier
        barrier()

        mbar[0].wait(phase)
        phase ^= 1

        a_dst_tile = a_dst.tile[tileM, tileN](block_idx.y, i)
        b_dst_tile = b_dst.tile[tileM, tileN](block_idx.y, i)
        copy_sram_to_dram[a_thread_layout](a_dst_tile, a_tile)
        copy_sram_to_dram[b_thread_layout](b_dst_tile, b_tile)


def test_tma_load_two_buffers_row_major[
    src_layout: Layout, tile_layout: Layout, load_along_last_dim: Bool = False
](ctx: DeviceContext):
    alias M = src_layout.shape[0].value()
    alias N = src_layout.shape[1].value()
    alias tileM = tile_layout.shape[0].value()
    alias tileN = tile_layout.shape[1].value()
    alias M_roundup = align_up(M, tileM)
    alias N_roundup = align_up(N, tileN)

    var a_src = ManagedLayoutTensor[DType.float32, src_layout](ctx)
    var b_src = ManagedLayoutTensor[DType.float32, src_layout](ctx)

    var a_dst = ManagedLayoutTensor[
        DType.float32, Layout.row_major(M_roundup, N_roundup)
    ](ctx)

    var b_dst = ManagedLayoutTensor[
        DType.float32, Layout.row_major(M_roundup, N_roundup)
    ](ctx)

    arange(a_src.tensor(), 1)
    arange(b_src.tensor(), 1)
    var a_tma_tensor = create_tma_tile[tileM, tileN](ctx, a_src.device_tensor())
    var b_tma_tensor = create_tma_tile[tileM, tileN](ctx, b_src.device_tensor())
    ctx.synchronize()

    alias kernel = test_tma_loads_two_buffers_kernel[
        __type_of(a_tma_tensor).dtype,
        Layout.row_major(M_roundup, N_roundup),  # dst layout
        Layout.row_major(M_roundup, N_roundup),  # dst layout
        __type_of(a_tma_tensor).layout,  # smem layout
        __type_of(b_tma_tensor).layout,  # smem layout
        __type_of(a_tma_tensor).layout,  # thread layout
        __type_of(b_tma_tensor).layout,  # thread layout
    ]
    ctx.enqueue_function[kernel](
        a_dst.device_tensor(),
        b_dst.device_tensor(),
        a_tma_tensor,
        b_tma_tensor,
        grid_dim=(1, M_roundup // tileM),
        block_dim=(tileM * tileN),
    )

    a_src_host = a_src.tensor()
    a_dst_host = a_dst.tensor()

    b_src_host = b_src.tensor()
    b_dst_host = b_dst.tensor()

    # Check M x N keep the same value and others in M_roundup x N_roundup
    # are set to zeros.
    for m in range(M_roundup):
        for n in range(N_roundup):
            if m < M and n < N:
                assert_equal(
                    a_src_host[m, n].cast[DType.float32](),
                    a_dst_host[m, n].cast[DType.float32](),
                )

                assert_equal(
                    b_src_host[m, n].cast[DType.float32](),
                    b_dst_host[m, n].cast[DType.float32](),
                )

            else:
                assert_equal(a_dst_host[m, n].cast[DType.float32](), 0.0)
                assert_equal(b_dst_host[m, n].cast[DType.float32](), 0.0)
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
fn test_tma_loads_and_store_two_buffers_kernel[
    dtype: DType,
    a_tile_layout: Layout,
    b_tile_layout: Layout,
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    /,
    *,
    a_layout: Layout,
    b_layout: Layout,
](
    a_tma_dst_tile: TMATensorTile[dtype, a_tile_layout, a_desc_layout],
    b_tma_dst_tile: TMATensorTile[dtype, b_tile_layout, b_desc_layout],
    a_tma_src_tile: TMATensorTile[dtype, a_tile_layout, a_desc_layout],
    b_tma_src_tile: TMATensorTile[dtype, b_tile_layout, b_desc_layout],
):
    alias tileM = a_tile_layout.shape[0].value()
    alias tileN = a_tile_layout.shape[1].value()
    alias expected_bytes = a_tile_layout.size() * size_of[dtype]()

    alias N = a_layout.shape[1].value()
    alias num_iters = ceildiv(N, tileN)

    a_tile = LayoutTensor[
        dtype,
        a_tile_layout,
        MutableAnyOrigin,
        address_space = _GPUAddressSpace.SHARED,
        alignment=128,
    ].stack_allocation()

    b_tile = LayoutTensor[
        dtype,
        b_tile_layout,
        MutableAnyOrigin,
        address_space = _GPUAddressSpace.SHARED,
        alignment=128,
    ].stack_allocation()

    mbar = stack_allocation[
        1,
        SharedMemBarrier,
        address_space = _GPUAddressSpace.SHARED,
        alignment=8,
    ]()

    if thread_idx.x == 0:
        mbar[0].init()

    var phase: UInt32 = 0

    for i in range(num_iters):
        if thread_idx.x == 0:
            mbar[0].expect_bytes(expected_bytes * 2)
            a_tma_src_tile.async_copy(
                a_tile,
                mbar[0],
                (UInt(i) * UInt(tileN), block_idx.y * UInt(tileM)),
            )
            b_tma_src_tile.async_copy(
                b_tile,
                mbar[0],
                (UInt(i) * UInt(tileN), block_idx.y * UInt(tileM)),
            )

        # Ensure all threads sees initialized mbarrier
        barrier()

        mbar[0].wait(phase)
        phase ^= 1

        fence_async_view_proxy()

        if thread_idx.x == 0:
            a_tma_dst_tile.async_store(
                a_tile, (UInt(i) * UInt(tileN), block_idx.y * UInt(tileM))
            )
            b_tma_dst_tile.async_store(
                b_tile, (UInt(i) * UInt(tileN), block_idx.y * UInt(tileM))
            )
            cp_async_bulk_commit_group()

        cp_async_bulk_wait_group[0]()


def test_tma_load_and_store_two_buffers_row_major[
    src_layout: Layout, tile_layout: Layout, dst_layout: Layout
](ctx: DeviceContext):
    alias M = src_layout.shape[0].value()
    alias N = src_layout.shape[1].value()
    alias tileM = tile_layout.shape[0].value()
    alias tileN = tile_layout.shape[1].value()
    alias dst_M = dst_layout.shape[0].value()
    alias dst_N = dst_layout.shape[1].value()

    var a_src = ManagedLayoutTensor[DType.float32, src_layout](ctx)
    var b_src = ManagedLayoutTensor[DType.float32, src_layout](ctx)
    var a_dst = ManagedLayoutTensor[DType.float32, dst_layout](ctx)
    var b_dst = ManagedLayoutTensor[DType.float32, dst_layout](ctx)

    # Initialize destinations to known values.
    alias a_dst_value = 1.5
    alias b_dst_value = 1.25

    var a_dst_host = a_dst.tensor()
    var b_dst_host = b_dst.tensor()
    # Ensure that the buffers have been fully created before accessing their data.
    ctx.synchronize()
    for m in range(dst_M):
        for n in range(dst_N):
            a_dst_host[m, n] = a_dst_value
            b_dst_host[m, n] = b_dst_value

    arange(a_src.tensor(), 1)
    arange(b_src.tensor(), 1)
    var a_tma_src_tensor = create_tma_tile[
        DType.float32, 2, Index(tileM, tileN)
    ](ctx, a_src.device_tensor())
    var b_tma_src_tensor = create_tma_tile[
        DType.float32, 2, Index(tileM, tileN)
    ](ctx, b_src.device_tensor())
    var a_tma_dst_tensor = create_tma_tile[
        DType.float32, 2, Index(tileM, tileN)
    ](ctx, a_dst.device_tensor())
    var b_tma_dst_tensor = create_tma_tile[
        DType.float32, 2, Index(tileM, tileN)
    ](ctx, b_dst.device_tensor())
    ctx.synchronize()

    alias kernel = test_tma_loads_and_store_two_buffers_kernel[
        __type_of(a_tma_src_tensor).dtype,
        __type_of(a_tma_src_tensor).layout,  # smem layout
        __type_of(a_tma_src_tensor).layout,  # smem layout
        __type_of(a_tma_src_tensor).desc_layout,
        __type_of(b_tma_src_tensor).desc_layout,
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

    a_src_host = a_src.tensor()
    a_dst_host = a_dst.tensor()

    b_src_host = b_src.tensor()
    b_dst_host = b_dst.tensor()

    for m in range(dst_M):
        for n in range(dst_N):
            if m < M and n < N:
                assert_equal(
                    a_src_host[m, n].cast[DType.float32](),
                    a_dst_host[m, n].cast[DType.float32](),
                )

                assert_equal(
                    b_src_host[m, n].cast[DType.float32](),
                    b_dst_host[m, n].cast[DType.float32](),
                )

            else:
                assert_equal(
                    a_dst_host[m, n].cast[DType.float32](), a_dst_value
                )
                assert_equal(
                    b_dst_host[m, n].cast[DType.float32](), b_dst_value
                )

    ctx.synchronize()
    _ = a_src^
    _ = a_dst^

    _ = b_src^
    _ = b_dst^


def main():
    with DeviceContext() as ctx:
        print("test_tma_load_f32")
        test_tma_load_row_major[
            dtype = DType.float32,
            src_layout = Layout.row_major(8, 8),
            tile_layout = Layout.row_major(4, 4),
        ](ctx)
        test_tma_load_row_major[
            dtype = DType.float32,
            src_layout = Layout.row_major(9, 24),
            tile_layout = Layout.row_major(3, 8),
        ](ctx)
        print("test_tma_load_oob_fill_f32")
        test_tma_load_row_major[
            dtype = DType.float32,
            src_layout = Layout.row_major(7, 8),
            tile_layout = Layout.row_major(4, 4),
        ](ctx)
        test_tma_load_row_major[
            dtype = DType.float32,
            src_layout = Layout.row_major(10, 12),
            tile_layout = Layout.row_major(4, 8),
        ](ctx)

        print("test_tma_multiple_loads_f32")
        test_tma_load_row_major[
            dtype = DType.float32,
            src_layout = Layout.row_major(12, 16),
            tile_layout = Layout.row_major(4, 4),
            load_along_last_dim=True,
        ](ctx)
        test_tma_load_row_major[
            dtype = DType.float32,
            src_layout = Layout.row_major(24, 80),
            tile_layout = Layout.row_major(3, 16),
            load_along_last_dim=True,
        ](ctx)

        print("test_tma_multiple_loads_oob_fill_f32")
        test_tma_load_row_major[
            dtype = DType.float32,
            src_layout = Layout.row_major(6, 20),
            tile_layout = Layout.row_major(4, 8),
            load_along_last_dim=True,
        ](ctx)
        test_tma_load_row_major[
            dtype = DType.float32,
            src_layout = Layout.row_major(9, 60),
            tile_layout = Layout.row_major(8, 16),
            load_along_last_dim=True,
        ](ctx)

        print("test_tma_load_f8e4m3fn")
        test_tma_load_row_major[
            dtype = DType.float8_e4m3fn,
            src_layout = Layout.row_major(8, 32),
            tile_layout = Layout.row_major(4, 16),
        ](ctx)
        test_tma_load_row_major[
            dtype = DType.float8_e4m3fn,
            src_layout = Layout.row_major(9, 48),
            tile_layout = Layout.row_major(3, 16),
        ](ctx)
        print("test_tma_load_oob_fill_f8e4m3fn")
        test_tma_load_row_major[
            dtype = DType.float8_e4m3fn,
            src_layout = Layout.row_major(7, 32),
            tile_layout = Layout.row_major(4, 16),
        ](ctx)
        test_tma_load_row_major[
            dtype = DType.float8_e4m3fn,
            src_layout = Layout.row_major(10, 48),
            tile_layout = Layout.row_major(4, 32),
        ](ctx)

        print("test_tma_multiple_loads_f8e4m3fn")
        test_tma_load_row_major[
            dtype = DType.float8_e4m3fn,
            src_layout = Layout.row_major(12, 64),
            tile_layout = Layout.row_major(4, 16),
            load_along_last_dim=True,
        ](ctx)
        test_tma_load_row_major[
            dtype = DType.float8_e4m3fn,
            src_layout = Layout.row_major(24, 160),
            tile_layout = Layout.row_major(3, 64),
            load_along_last_dim=True,
        ](ctx)

        print("test_tma_multiple_loads_oob_fill_f8e4m3fn")
        test_tma_load_row_major[
            dtype = DType.float8_e4m3fn,
            src_layout = Layout.row_major(6, 80),
            tile_layout = Layout.row_major(4, 16),
            load_along_last_dim=True,
        ](ctx)
        test_tma_load_row_major[
            dtype = DType.float8_e4m3fn,
            src_layout = Layout.row_major(9, 240),
            tile_layout = Layout.row_major(8, 64),
            load_along_last_dim=True,
        ](ctx)

        print("test_tma_async_store")
        test_tma_async_store[
            src_layout = Layout.row_major(8, 8),
            tile_layout = Layout.row_major(4, 4),
            dst_layout = Layout.row_major(8, 8),
        ](ctx)
        test_tma_async_store[
            src_layout = Layout.row_major(32, 24),
            tile_layout = Layout.row_major(16, 8),
            dst_layout = Layout.row_major(32, 24),
        ](ctx)

        print("test_tma_multiple_async_store")
        test_tma_async_store[
            src_layout = Layout.row_major(8, 8),
            tile_layout = Layout.row_major(4, 4),
            dst_layout = Layout.row_major(8, 8),
            load_along_last_dim=True,
        ](ctx)
        test_tma_async_store[
            src_layout = Layout.row_major(9, 24),
            tile_layout = Layout.row_major(3, 8),
            dst_layout = Layout.row_major(9, 24),
            load_along_last_dim=True,
        ](ctx)

        print("test_tma_async_store_oob")
        test_tma_async_store[
            src_layout = Layout.row_major(8, 8),
            tile_layout = Layout.row_major(8, 8),
            dst_layout = Layout.row_major(6, 8),
        ](ctx)
        test_tma_async_store[
            src_layout = Layout.row_major(32, 8),
            tile_layout = Layout.row_major(8, 8),
            dst_layout = Layout.row_major(26, 8),
        ](ctx)
        test_tma_async_store[
            src_layout = Layout.row_major(8, 8),
            tile_layout = Layout.row_major(8, 8),
            dst_layout = Layout.row_major(6, 4),
        ](ctx)
        test_tma_async_store[
            src_layout = Layout.row_major(32, 16),
            tile_layout = Layout.row_major(8, 8),
            dst_layout = Layout.row_major(26, 12),
        ](ctx)

        print("test_tma_async_reduce")
        test_tma_async_reduce[
            src_layout = Layout.row_major(8, 8),
            tile_layout = Layout.row_major(4, 4),
            dst_layout = Layout.row_major(8, 8),
        ](ctx)
        test_tma_async_reduce[
            src_layout = Layout.row_major(9, 24),
            tile_layout = Layout.row_major(3, 8),
            dst_layout = Layout.row_major(9, 24),
        ](ctx)

        print("test_tma_multiple_async_reduce")
        test_tma_async_reduce[
            src_layout = Layout.row_major(8, 8),
            tile_layout = Layout.row_major(4, 4),
            dst_layout = Layout.row_major(8, 8),
            load_along_last_dim=True,
        ](ctx)
        test_tma_async_reduce[
            src_layout = Layout.row_major(9, 24),
            tile_layout = Layout.row_major(3, 8),
            dst_layout = Layout.row_major(9, 24),
            load_along_last_dim=True,
        ](ctx)

        print("test_tma_async_reduce_oob")
        test_tma_async_reduce[
            src_layout = Layout.row_major(8, 8),
            tile_layout = Layout.row_major(8, 8),
            dst_layout = Layout.row_major(6, 8),
        ](ctx)
        test_tma_async_reduce[
            src_layout = Layout.row_major(32, 8),
            tile_layout = Layout.row_major(8, 8),
            dst_layout = Layout.row_major(26, 8),
        ](ctx)
        test_tma_async_reduce[
            src_layout = Layout.row_major(8, 8),
            tile_layout = Layout.row_major(8, 8),
            dst_layout = Layout.row_major(6, 4),
        ](ctx)
        test_tma_async_reduce[
            src_layout = Layout.row_major(32, 16),
            tile_layout = Layout.row_major(8, 8),
            dst_layout = Layout.row_major(26, 12),
        ](ctx)
        print("test_tma_load_two_buffer_row_major")
        test_tma_load_two_buffers_row_major[
            src_layout = Layout.row_major(32, 64),
            tile_layout = Layout.row_major(8, 16),
            load_along_last_dim=True,
        ](ctx)
        test_tma_load_two_buffers_row_major[
            src_layout = Layout.row_major(9, 60),
            tile_layout = Layout.row_major(8, 16),
        ](ctx)
        print("test_tma_load_and_store_two_buffer")
        test_tma_load_and_store_two_buffers_row_major[
            src_layout = Layout.row_major(32, 64),
            tile_layout = Layout.row_major(8, 16),
            dst_layout = Layout.row_major(32, 64),
        ](ctx)
        test_tma_load_and_store_two_buffers_row_major[
            src_layout = Layout.row_major(32, 64),
            tile_layout = Layout.row_major(16, 16),
            dst_layout = Layout.row_major(40, 64),
        ](ctx)
