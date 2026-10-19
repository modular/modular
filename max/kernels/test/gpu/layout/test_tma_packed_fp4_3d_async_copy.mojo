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

"""Rank-3 TMA loads of nibble-packed FP4 through `PACKED_FP4_ALIGN16B`.

The SM100 block-scaled matmul views every operand as `(batch, rows, K)` and
loads it with a rank-3 descriptor, so a W4A8 E2M1 operand needs the rank-3
form of `create_tensor_tile[..., unpack_fp4=True]`. The rank-2 form and the
padded shared-memory image are covered by `test_tma_packed_fp4_async_copy`;
this test adds what rank 3 brings:

- the innermost extent doubled while the batch and row extents and all
  strides stay in `uint8` units
- a nonzero batch coordinate selecting the right batch slice
- rows past a batch's extent zero-filled instead of reading the next batch
- an mbarrier transaction count equal to the global-memory bytes of the box

Gap bytes are never compared.
"""

from std.math import align_up

from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.sync import barrier
from layout import (
    Coord,
    Idx,
    MixedLayout,
    TensorLayout,
    TileTensor,
    coord,
    row_major,
)
from layout.tile_tensor import stack_allocation
from layout.swizzle import make_swizzle
from layout.tma_async import SharedMemBarrier, TMATensorTile, create_tensor_tile
from std.memory import unsafe_stack_allocation
from std.testing import assert_equal

comptime FP4_BOX_ELEMS = 128
"""Innermost box extent `16U4_ALIGN16B` requires, in FP4 elements."""
comptime FP4_BOX_GMEM_BYTES = FP4_BOX_ELEMS // 2
comptime FP4_BOX_SMEM_BYTES = FP4_BOX_ELEMS


@__llvm_arg_metadata(tma_tile, `nvvm.grid_constant`)
def packed_fp4_load_3d_kernel[
    DstLayout: TensorLayout,
    rows_roundup: Int,
    tile_rows: Int,
    tile_shape: Coord,
    desc_shape: Coord,
](
    dst: TileTensor[DType.uint8, DstLayout, MutAnyOrigin],
    tma_tile: TMATensorTile[DType.uint8, tile_shape, desc_shape],
):
    """Loads one `(1, tile_rows, 128)` padded FP4 tile and copies it out raw.

    Grid axes are (K tile, row tile, batch). Batch `b` lands in `dst` rows
    `[b * rows_roundup, (b + 1) * rows_roundup)`.
    """
    var smem_tile = stack_allocation[
        dtype=DType.uint8, address_space=.SHARED, alignment=1024
    ](row_major[tile_rows, FP4_BOX_SMEM_BYTES]())

    var mbar = unsafe_stack_allocation[
        1,
        SharedMemBarrier,
        address_space=.SHARED,
        alignment=8,
    ]()

    if thread_idx.x == 0:
        mbar[0].init()
        mbar[0].expect_bytes(Int32(tile_rows * FP4_BOX_GMEM_BYTES))
        tma_tile.async_copy_3d(
            smem_tile,
            mbar[0],
            (
                block_idx.x * FP4_BOX_ELEMS,
                block_idx.y * tile_rows,
                block_idx.z,
            ),
        )
    barrier()
    mbar[0].wait()

    var row0 = Int(block_idx.z) * rows_roundup + Int(block_idx.y) * tile_rows
    var col0 = Int(block_idx.x) * FP4_BOX_SMEM_BYTES
    for i in range(
        Int(thread_idx.x), tile_rows * FP4_BOX_SMEM_BYTES, Int(block_dim.x)
    ):
        var row = i // FP4_BOX_SMEM_BYTES
        var col = i % FP4_BOX_SMEM_BYTES
        dst[row0 + row, col0 + col] = smem_tile[row, col]


def _src_byte(batch: Int, row: Int, packed_col: Int) -> UInt8:
    """Returns a position-dependent byte, so a misplaced chunk is detected."""
    return UInt8((batch * 101 + row * 37 + packed_col * 11 + 5) & 0xFF)


def test_packed_fp4_load_3d[
    batch: Int,
    rows: Int,
    k_elems: Int,
    tile_rows: Int,
    swizzle_mode: TensorMapSwizzle,
](ctx: DeviceContext) raises:
    """Loads a `batch x rows x k_elems` packed FP4 tensor and checks it.

    Parameters:
        batch: Global batch count.
        rows: Rows per batch; a value that is not a multiple of `tile_rows`
            leaves each batch's last tile partly out of bounds.
        k_elems: Global innermost extent in FP4 elements.
        tile_rows: Box rows per TMA load.
        swizzle_mode: Shared-memory swizzle of the descriptor.
    """
    comptime assert (
        k_elems % FP4_BOX_ELEMS == 0
    ), "16U4_ALIGN16B requires the innermost extent to be a multiple of 128"
    comptime k_bytes = k_elems // 2
    comptime rows_roundup = align_up(rows, tile_rows)
    comptime num_k_tiles = k_elems // FP4_BOX_ELEMS
    comptime dst_layout = row_major[
        batch * rows_roundup, num_k_tiles * FP4_BOX_SMEM_BYTES
    ]()

    print(
        "test_packed_fp4_load_3d batch=",
        batch,
        " rows=",
        rows,
        " k_elems=",
        k_elems,
        " tile_rows=",
        tile_rows,
        " swizzle=",
        swizzle_mode,
        sep="",
    )

    comptime num_bytes = batch * rows * k_bytes
    var src_host = ctx.enqueue_create_host_buffer[.uint8](num_bytes)
    for b in range(batch):
        for r in range(rows):
            for c in range(k_bytes):
                src_host[(b * rows + r) * k_bytes + c] = _src_byte(b, r, c)
    var src_device = ctx.enqueue_create_buffer[.uint8](num_bytes)
    ctx.enqueue_copy(src_device, src_host)

    var src_tensor = TileTensor(
        src_device,
        MixedLayout(
            Coord(Idx[batch], Idx[rows], Idx[k_bytes]),
            Coord(Idx[rows * k_bytes], Idx[k_bytes], Idx[1]),
        ),
    )
    comptime tile_shape = coord[1, tile_rows, FP4_BOX_ELEMS]
    var tma_tensor = create_tensor_tile[
        tile_shape,
        swizzle_mode=swizzle_mode,
        __desc_shape=tile_shape,
        unpack_fp4=True,
    ](ctx, src_tensor)

    comptime dst_size = dst_layout.size()
    var dst_device = ctx.enqueue_create_buffer[.uint8](dst_size)

    comptime kernel = packed_fp4_load_3d_kernel[
        type_of(dst_layout),
        rows_roundup,
        tile_rows,
        type_of(tma_tensor).tile_shape,
        type_of(tma_tensor).desc_shape,
    ]
    ctx.enqueue_function[kernel](
        TileTensor(dst_device, dst_layout).as_unsafe_any_origin(),
        tma_tensor,
        grid_dim=(num_k_tiles, rows_roundup // tile_rows, batch),
        block_dim=(128),
    )

    var dst_host_buffer = ctx.enqueue_create_host_buffer[.uint8](dst_size)
    ctx.enqueue_copy(dst_host_buffer, dst_device)
    ctx.synchronize()
    var dst_host = TileTensor(dst_host_buffer, dst_layout)
    comptime swizzle = make_swizzle[DType.uint8, swizzle_mode]()

    for b in range(batch):
        for tile_row in range(rows_roundup // tile_rows):
            for k_tile in range(num_k_tiles):
                for local_row in range(tile_rows):
                    var row = tile_row * tile_rows + local_row
                    for smem_col in range(FP4_BOX_SMEM_BYTES):
                        if smem_col % 16 >= 8:
                            continue
                        var physical = swizzle(
                            local_row * FP4_BOX_SMEM_BYTES + smem_col
                        )
                        var got = dst_host[
                            b * rows_roundup
                            + tile_row * tile_rows
                            + physical // FP4_BOX_SMEM_BYTES,
                            k_tile * FP4_BOX_SMEM_BYTES
                            + physical % FP4_BOX_SMEM_BYTES,
                        ]
                        var packed_col = (
                            k_tile * FP4_BOX_GMEM_BYTES
                            + (smem_col // 16) * 8
                            + smem_col % 16
                        )
                        var expected = _src_byte(
                            b, row, packed_col
                        ) if row < rows else UInt8(0)
                        assert_equal(
                            got,
                            expected,
                            msg=String(
                                "batch=",
                                b,
                                " row=",
                                row,
                                " k_tile=",
                                k_tile,
                                " smem_col=",
                                smem_col,
                            ),
                        )

    _ = src_device^
    _ = dst_device^
    _ = dst_host_buffer^


def main() raises:
    with DeviceContext() as ctx:
        test_packed_fp4_load_3d[
            batch=1,
            rows=128,
            k_elems=256,
            tile_rows=64,
            swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
        ](ctx)
        test_packed_fp4_load_3d[
            batch=2,
            rows=256,
            k_elems=512,
            tile_rows=128,
            swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)
        # Each batch's last row tile is partly out of bounds; TMA must
        # zero-fill it rather than read the following batch.
        test_packed_fp4_load_3d[
            batch=2,
            rows=100,
            k_elems=384,
            tile_rows=64,
            swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)
