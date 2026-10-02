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
"""Host-side tests for `TMemEngine`.

Exercises the operations that do not touch hardware: `offset`, `distance`,
`unsafe_cast`, and the address arithmetic a `TileTensor` over the engine
performs when tiling an accumulator down to a warp's quadrant and a thread's
row. Layouts are positions on the lane-by-column grid, so these tests check
that the engine encodes them into the hardware's lane-in-the-upper-half
address format. The copies, which issue `tcgen05` instructions, need a
Blackwell GPU and live in `max/kernels/test/gpu/layout/test_tmem_engine.mojo`.
"""

from std.testing import assert_equal, TestSuite

from layout import Coord, Idx, TileTensor
from layout.tile_layout import Layout as TileLayout
from layout.tmem_engine import TMemEngine, TMemStorage, TMEM_NUM_LANES

comptime NUM_LANES = 128
comptime NUM_COLS = 32
comptime BASE_LANE = 0
comptime BASE_COL = 64


def _addr(lane: Int, col: Int) -> UInt32:
    """Encodes a TMEM cell the way the hardware does."""
    return UInt32((lane << 16) + col)


comptime BASE_ADDR = _addr(BASE_LANE, BASE_COL)

comptime ACCUM_LAYOUT = TileLayout(
    shape=Coord(Idx[NUM_LANES], Idx[NUM_COLS]),
    stride=Coord(Idx[1], Idx[TMEM_NUM_LANES]),
)
comptime AccumTile = TileTensor[
    .float32,
    type_of(ACCUM_LAYOUT),
    MutAnyOrigin,
    Engine=TMemEngine,
]
# The handle carries no `address_space`, so the engine's trait-mandated
# parameters can only be inferred inside `TileTensor`'s generic context.
# Every operation below therefore goes through a tile, as kernels do.
comptime AccumStorage = TMemStorage[.float32, MutAnyOrigin]

comptime NUM_WARPS = 4
comptime M64_LANES_PER_WARP = 16
comptime M64_LAYOUT = TileLayout(
    shape=Coord(Coord(Idx[M64_LANES_PER_WARP], Idx[NUM_WARPS]), Idx[NUM_COLS]),
    stride=Coord(Coord(Idx[1], Idx[32]), Idx[TMEM_NUM_LANES]),
)
comptime M64Tile = TileTensor[
    .float32,
    type_of(M64_LAYOUT),
    MutAnyOrigin,
    Engine=TMemEngine,
]


def test_offset_encodes_lane_and_column() raises:
    var tile = AccumTile(AccumStorage(BASE_ADDR), ACCUM_LAYOUT)
    var moved = tile._offset_storage(3 + 5 * TMEM_NUM_LANES)
    assert_equal(moved.addr, _addr(BASE_LANE + 3, BASE_COL + 5))
    var same_col = tile._offset_storage(5)
    assert_equal(same_col.addr, _addr(BASE_LANE + 5, BASE_COL))
    var same_lane = tile._offset_storage(5 * TMEM_NUM_LANES)
    assert_equal(same_lane.addr, _addr(BASE_LANE, BASE_COL + 5))


def test_distance_is_signed_grid_delta() raises:
    var a = AccumTile(AccumStorage(BASE_ADDR), ACCUM_LAYOUT)
    # Seven columns over is seven grid columns, not the raw address delta.
    var b = AccumTile(AccumStorage(BASE_ADDR + 7), ACCUM_LAYOUT)
    assert_equal(Int(b.as_imm()._distance(a.as_imm())), 7 * TMEM_NUM_LANES)
    assert_equal(Int(a.as_imm()._distance(b.as_imm())), -7 * TMEM_NUM_LANES)
    var c = AccumTile(
        AccumStorage(_addr(BASE_LANE + 2, BASE_COL)), ACCUM_LAYOUT
    )
    assert_equal(Int(c.as_imm()._distance(a.as_imm())), 2)


def test_unsafe_cast_keeps_address() raises:
    var tile = AccumTile(AccumStorage(BASE_ADDR), ACCUM_LAYOUT)
    var cast = tile._unsafe_storage_cast[to_dtype=DType.int32]()
    assert_equal(cast.addr, BASE_ADDR)


def test_layout_is_the_lane_column_grid() raises:
    var tile = AccumTile(AccumStorage(BASE_ADDR), ACCUM_LAYOUT)
    assert_equal(
        Int(tile.layout(Coord(Idx[5], Idx[7]))), 5 + 7 * TMEM_NUM_LANES
    )


def test_warp_tile_lands_on_quadrant_base() raises:
    var tile = AccumTile(AccumStorage(BASE_ADDR), ACCUM_LAYOUT)
    var warp_tile = tile.tile[32, NUM_COLS](Coord(Int(2), Idx[0]))
    assert_equal(warp_tile._storage.addr, _addr(BASE_LANE + 2 * 32, BASE_COL))
    var row = warp_tile.tile[1, NUM_COLS](Coord(Idx[0], Idx[0]))
    assert_equal(row._storage.addr, warp_tile._storage.addr)
    assert_equal(Int(row.layout(Coord(Idx[0], Idx[9]))), 9 * TMEM_NUM_LANES)


def test_m64_hierarchical_coordinate_lands_on_quadrant_base() raises:
    var tile = M64Tile(AccumStorage(BASE_ADDR), M64_LAYOUT)
    # Row (0, w) of the nested layout is warp w's quadrant base, which is
    # where a thread's row view over the M=64 placement has to start.
    var row_base = tile._offset_storage(
        Int(tile.layout(Coord(Coord(Idx[0], Int(3)), Idx[0])))
    )
    assert_equal(row_base.addr, _addr(BASE_LANE + 3 * 32, BASE_COL))
    assert_equal(
        Int(tile.layout(Coord(Coord(Idx[5], Int(2)), Idx[7]))),
        (2 * 32 + 5) + 7 * TMEM_NUM_LANES,
    )


def test_type_name() raises:
    var name = String()
    TMemEngine.write_type_name_to(name)
    assert_equal(name, "TMemEngine")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
