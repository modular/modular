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
from layout import (
    ComptimeInt,
    Coord,
    Idx,
    MixedLayout,
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
)
from std.memory import unsafe_stack_allocation
from std.testing import assert_equal


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
