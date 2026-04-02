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
"""TileTensor utilities for block-aligned SMEM tiling.

The SMEM layout for AMD MHA uses blocked_product(row_major(BN, BK),
row_major(1, num_repeats)), which stores num_repeats contiguous BN×BK
blocks. TileTensor's tile[] assumes flat strides and cannot tile
hierarchical layouts directly.

These helpers compute offsets from the known block structure and create
flat TileTensor sub-views. They are correct when tile dimensions align
with block boundaries (always true for the MHA kernel's tile sizes).

Gap: TileTensor's _tile uses `stride[i]().value()` which fails on
Coord-valued strides from blocked_product. A proper fix requires
TileTensor to use zipped_divide for hierarchical layouts, which in turn
needs _Divide/_Multiply to recursively handle Coords (partially done
in coord.mojo — the type algebra extension compiles but zipped_divide
semantics need further work for floor-division across hierarchy levels).
"""

from layout import TileTensor
from layout.tile_layout import row_major as tt_row_major
from std.memory.pointer import AddressSpace


@always_inline
def smem_subtile[
    tile_rows: Int,
    tile_cols: Int,
    BN: Int,
    BK: Int,
    dtype: DType,
](
    smem_ptr: UnsafePointer[
        Scalar[dtype], MutAnyOrigin, address_space=AddressSpace.SHARED
    ],
    tile_row: Int,
    tile_col: Int,
) -> TileTensor[
    dtype,
    type_of(tt_row_major[tile_rows, tile_cols]()),
    MutAnyOrigin,
    address_space=AddressSpace.SHARED,
]:
    """Creates a flat TileTensor sub-view of a blocked SMEM layout.

    The blocked layout has num_repeats contiguous BN×BK blocks. This
    function computes the physical offset for a block-aligned tile and
    returns a row-major TileTensor view with strides (tile_cols, 1).

    Correct only when tile_cols == BK (tiles don't cross block boundaries
    in the column dimension).

    Parameters:
        tile_rows: Height of the sub-tile.
        tile_cols: Width of the sub-tile (must equal BK for block alignment).
        BN: Number of rows per block (full block height).
        BK: Number of columns per block (full block width).
        dtype: Element data type.

    Args:
        smem_ptr: Base pointer to the SMEM allocation.
        tile_row: Tile row index (0-based, in units of tile_rows).
        tile_col: Tile column index (0-based, in units of tile_cols).

    Returns:
        A TileTensor view into the specified sub-tile region.
    """
    comptime block_size = BN * BK
    var offset = tile_row * tile_rows * BK + tile_col * block_size
    return TileTensor[
        dtype,
        type_of(tt_row_major[tile_rows, tile_cols]()),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ](smem_ptr + offset, tt_row_major[tile_rows, tile_cols]())


@always_inline
def smem_mma_subtile[
    mma_rows: Int,
    mma_cols: Int,
    BN: Int,
    BK: Int,
    dtype: DType,
](
    smem_ptr: UnsafePointer[
        Scalar[dtype], MutAnyOrigin, address_space=AddressSpace.SHARED
    ],
    bk_tile: Int,
    k_sub: Int,
    mma_idx: Int,
) -> TileTensor[
    dtype,
    type_of(tt_row_major[mma_rows, mma_cols]()),
    MutAnyOrigin,
    address_space=AddressSpace.SHARED,
]:
    """Creates a flat TileTensor for an MMA-sized sub-tile in blocked SMEM.

    Used by the non-transposed (V buffer) load_from_shared path. The V
    buffer's SMEM has shape (BN, depth) with blocked layout
    (num_repeats × BN×BK blocks). Each MMA tile is mma_rows × mma_cols
    within one block.

    Parameters:
        mma_rows: MMA tile height (e.g., MMA_K=16).
        mma_cols: MMA tile width (e.g., MMA_M=32).
        BN: Block height.
        BK: Block width.
        dtype: Element data type.

    Args:
        smem_ptr: Base pointer to the SMEM allocation for this buffer stage.
        bk_tile: Which BK-tall row group (0..depth/BK-1).
        k_sub: Which MMA_K sub-row within the BK group (0..BK/MMA_K-1).
        mma_idx: Linear MMA tile index across the full depth dimension.

    Returns:
        A TileTensor view into the MMA-sized sub-tile.
    """
    comptime block_size = BN * BK
    comptime tiles_per_block = BK // mma_cols
    var block_idx = mma_idx // tiles_per_block
    var col_in_block = (mma_idx % tiles_per_block) * mma_cols
    var offset = (
        bk_tile * BK * BK
        + k_sub * mma_rows * BK
        + block_idx * block_size
        + col_in_block
    )
    return TileTensor[
        dtype,
        type_of(tt_row_major[mma_rows, mma_cols]()),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ](smem_ptr + offset, tt_row_major[mma_rows, mma_cols]())
