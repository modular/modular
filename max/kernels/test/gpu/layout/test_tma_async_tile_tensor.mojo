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

"""TMA load tests for the `create_tma_tile` overload taking a `TileTensor`.

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
"""

from std.math import align_up
from std.sys import size_of

from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu import block_idx, thread_idx
from layout import Coord, Idx, Layout, LayoutTensor, MixedLayout, TileTensor
from layout._utils import ManagedLayoutTensor
from layout.layout_tensor import copy_sram_to_dram
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    _idx_product,
    create_tma_tile,
)
from std.memory import unsafe_stack_allocation
from std.testing import assert_equal
from std.utils.index import IndexList


@__llvm_arg_metadata(tma_tile, `nvvm.grid_constant`)
def tma_tile_tensor_load_kernel[
    dtype: DType,
    layout: Layout,
    tile_rank: Int,
    tile_shape: IndexList[tile_rank],
    thread_layout: Layout,
](
    dst: LayoutTensor[dtype, layout, MutAnyOrigin],
    tma_tile: TMATensorTile[dtype, tile_rank, tile_shape],
):
    comptime tileM = tile_shape[0]
    comptime tileN = tile_shape[1]
    comptime expected_bytes = _idx_product[tile_rank, tile_shape]() * size_of[
        dtype
    ]()

    comptime __tile_layout = Layout.row_major(tileM, tileN)
    var tile = LayoutTensor[
        dtype,
        __tile_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ].stack_allocation()

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
    copy_sram_to_dram[thread_layout](dst_tile, tile)


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

    var dst = ManagedLayoutTensor[
        dtype, Layout.row_major(M_roundup, N_roundup)
    ](ctx)

    comptime __thread_layout = Layout.row_major(tileM, tileN)
    comptime kernel = tma_tile_tensor_load_kernel[
        type_of(tma_tensor).dtype,
        Layout.row_major(M_roundup, N_roundup),  # dst layout
        type_of(tma_tensor).rank,  # tile rank
        type_of(tma_tensor).tile_shape,  # tile shape
        __thread_layout,  # thread layout
    ]
    ctx.enqueue_function[kernel](
        dst.device_tensor(),
        tma_tensor,
        grid_dim=(N_roundup // tileN, M_roundup // tileM),
        block_dim=(tileM * tileN),
    )

    var dst_host = dst.tensor()

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
    ctx.synchronize()
    _ = dst^


def main() raises:
    with DeviceContext() as ctx:
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
