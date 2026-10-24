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
"""Tests `TileTensorIter` stepping, wrap-around and tile contents."""

from layout import TileTensor, row_major
from layout.tile_iterator import TileTensorIter
from layout.tile_layout import Layout
from layout.coord import ComptimeInt, Coord
from std.testing import TestSuite, assert_equal, assert_true

comptime TILE_ROWS = 2
comptime TILE_COLS = 4
comptime TILE_SIZE = TILE_ROWS * TILE_COLS
comptime NUM_TILES = 3
comptime TOTAL = TILE_SIZE * NUM_TILES


def _fill(mut data: Array[Int32, TOTAL]):
    for i in range(TOTAL):
        data[i] = Int32(i)


def test_contiguous_walk() raises:
    var data = Array[Int32, TOTAL](uninitialized=True)
    _fill(data)

    var it = TileTensorIter[.int32, _, _](
        data.unsafe_ptr(), row_major[TILE_ROWS, TILE_COLS](), bound=TOTAL
    )
    assert_equal(Int(it.stride), TILE_SIZE)

    for tile in range(NUM_TILES):
        var t = it[]
        for r in range(TILE_ROWS):
            for c in range(TILE_COLS):
                assert_equal(Int(t[r, c]), tile * TILE_SIZE + r * TILE_COLS + c)
        it._incr()
    assert_equal(Int(it.offset), TOTAL)


def test_circular_wrap() raises:
    var data = Array[Int32, TOTAL](uninitialized=True)
    _fill(data)

    var it = TileTensorIter[.int32, _, _, circular=True](
        data.unsafe_ptr(), row_major[TILE_ROWS, TILE_COLS](), bound=TOTAL
    )
    for step in range(2 * NUM_TILES + 1):
        assert_equal(Int(it.offset), (step % NUM_TILES) * TILE_SIZE)
        assert_equal(Int(it[][0, 0]), (step % NUM_TILES) * TILE_SIZE)
        it._incr()


def test_next_and_iadd() raises:
    var data = Array[Int32, TOTAL](uninitialized=True)
    _fill(data)

    var it = TileTensorIter[.int32, _, _, circular=True](
        data.unsafe_ptr(), row_major[TILE_ROWS, TILE_COLS](), bound=TOTAL
    )
    # `next` and `next_unsafe` leave the original untouched and wrap.
    assert_equal(Int(it.next(1).offset), TILE_SIZE)
    assert_equal(Int(it.next(NUM_TILES).offset), 0)
    assert_equal(Int(it.next(NUM_TILES + 2).offset), 2 * TILE_SIZE)
    assert_equal(Int(it.next_unsafe(2).offset), 2 * TILE_SIZE)
    assert_equal(Int(it.offset), 0)

    it._incr()
    assert_equal(Int(it.next_unsafe(NUM_TILES - 1).offset), 0)
    assert_equal(Int(it.next_unsafe(2).offset), 0)

    it += 5
    assert_equal(Int(it.offset), 0)
    it += 2
    assert_equal(Int(it.offset), 2 * TILE_SIZE)


def test_explicit_stride_and_offset() raises:
    # Column tiles of a row-major 2x12 matrix: tile (2, 4) with row stride 12
    # and a step of 4 elements to the right.
    var data = Array[Int32, 24](uninitialized=True)
    for i in range(24):
        data[i] = Int32(i)

    comptime TileLayout = Layout[
        shape_types=Coord[ComptimeInt[2], ComptimeInt[4]].element_types,
        stride_types=Coord[ComptimeInt[12], ComptimeInt[1]].element_types,
    ]
    var it = TileTensorIter[.int32, TileLayout, _](
        data.unsafe_ptr(),
        TileLayout(
            Coord[ComptimeInt[2], ComptimeInt[4]](),
            Coord[ComptimeInt[12], ComptimeInt[1]](),
        ),
        bound=24,
        stride=4,
        offset=4,
    )
    for col_tile in range(1, 3):
        var t = it[]
        for r in range(2):
            for c in range(4):
                assert_equal(Int(t[r, c]), r * 12 + col_tile * 4 + c)
        it._incr()


def test_masked_carries_valid_rows() raises:
    var data = Array[Int32, TOTAL](uninitialized=True)
    _fill(data)

    var it = TileTensorIter[.int32, _, _, masked=True](
        data.unsafe_ptr(),
        row_major[TILE_ROWS, TILE_COLS](),
        bound=TOTAL,
        num_valid_rows=1,
    )
    assert_equal(it.num_valid_rows, 1)
    it._incr()
    assert_equal(it.num_valid_rows, 1)
    assert_true(it.next().num_valid_rows == 1)


def test_reshape_and_bitcast() raises:
    var data = Array[Int32, TOTAL](uninitialized=True)
    _fill(data)

    var it = TileTensorIter[.int32, _, _, circular=True](
        data.unsafe_ptr(), row_major[TILE_ROWS, TILE_COLS](), bound=TOTAL
    )
    it._incr()

    var wide = it.reshape(row_major[TILE_COLS, TILE_ROWS]())
    assert_equal(Int(wide.offset), TILE_SIZE)
    assert_equal(Int(wide.stride), TILE_SIZE)
    assert_equal(Int(wide.bound), TOTAL)
    # Reshaping re-reads the same memory in row-major order.
    for r in range(TILE_COLS):
        for c in range(TILE_ROWS):
            assert_equal(Int(wide[][r, c]), TILE_SIZE + r * TILE_ROWS + c)
    wide._incr()
    wide._incr()
    assert_equal(Int(wide.offset), 0)

    var as_uint = it.bitcast[.uint32]()
    assert_equal(Int(as_uint.offset), TILE_SIZE)
    assert_equal(Int(as_uint[][1, 2]), TILE_SIZE + TILE_COLS + 2)


def test_from_tensor() raises:
    var data = Array[Int32, TOTAL](uninitialized=True)
    _fill(data)
    var tensor = TileTensor(
        data.unsafe_ptr(), row_major[NUM_TILES * TILE_ROWS, TILE_COLS]()
    )

    var it = TileTensorIter[.int32, _, _, circular=True](
        tensor, row_major[TILE_ROWS, TILE_COLS](), bound=TOTAL
    )
    assert_equal(Int(it.stride), TILE_SIZE)
    assert_equal(Int(it[][1, 2]), TILE_COLS + 2)

    var shifted = TileTensorIter[.int32, _, _](
        tensor.tile[TILE_ROWS, TILE_COLS](Coord(1, 0)),
        row_major[TILE_ROWS, TILE_COLS](),
        bound=TOTAL - TILE_SIZE,
    )
    assert_equal(Int(shifted[][0, 0]), TILE_SIZE)
    shifted._incr()
    assert_equal(Int(shifted[][1, 2]), 2 * TILE_SIZE + TILE_COLS + 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
