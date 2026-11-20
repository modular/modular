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
"""Tests for the `TwoLevelLayout` conformer.

Every swizzled layout the layout tests use has an exact two-level form, so
a `TwoLevelLayout` must give the same offset as the integer layout with
`swizzle=` applied.
"""

from layout import (
    ComptimeInt,
    Coord,
    CoordLike,
    Idx,
    MixedLayout,
    TwoLevelLayout,
    row_major,
    two_level_layout_from_swizzle,
)
from layout.swizzle import Swizzle, make_swizzle
from std.testing import TestSuite, assert_equal, assert_true


def test_two_level_matches_swizzle() raises:
    comptime swizzle = Swizzle(1, 0, 2)
    comptime plain = row_major[4, 4]()
    comptime layout = two_level_layout_from_swizzle[type_of(plain), swizzle](
        plain
    )
    comptime assert not type_of(layout).is_affine
    comptime assert type_of(layout).static_shape[0] == 4
    comptime assert type_of(layout).static_shape[1] == 4
    comptime assert type_of(layout).static_stride[0] == 5
    comptime assert type_of(layout).static_stride[1] == 1
    var value = layout
    for r in range(4):
        for c in range(4):
            var expected = swizzle(Int(plain(Coord(r, c))))
            assert_equal(Int(value(Coord(r, c))), expected)
            assert_equal(Int(value.shape[0]()), 4)
            assert_equal(Int(value.stride[0]()), 5)


def test_two_level_matches_make_swizzle_8x64() raises:
    comptime swizzle = make_swizzle[num_rows=8, row_size=64, access_size=8]()
    comptime plain = row_major[64, 64]()
    comptime layout = two_level_layout_from_swizzle[type_of(plain), swizzle](
        plain
    )
    var value = layout
    for r in range(64):
        for c in range(64):
            assert_equal(
                Int(value(Coord(r, c))),
                swizzle(Int(plain(Coord(r, c)))),
            )


def test_two_level_matches_make_swizzle_16x64() raises:
    comptime swizzle = make_swizzle[num_rows=16, row_size=64, access_size=8]()
    comptime plain = row_major[64, 64]()
    comptime layout = two_level_layout_from_swizzle[type_of(plain), swizzle](
        plain
    )
    var value = layout
    for r in range(64):
        for c in range(64):
            assert_equal(
                Int(value(Coord(r, c))),
                swizzle(Int(plain(Coord(r, c)))),
            )


def test_two_level_idx2crd_roundtrip() raises:
    comptime swizzle = make_swizzle[num_rows=8, row_size=64, access_size=8]()
    comptime plain = row_major[64, 64]()
    comptime layout = two_level_layout_from_swizzle[type_of(plain), swizzle](
        plain
    )
    var value = layout
    for r in range(64):
        for c in range(64):
            var idx = Int(value(Coord(r, c)))
            var crd = value.idx2crd(idx)
            assert_equal(Int(crd[0]), r)
            assert_equal(Int(crd[1]), c)


def test_two_level_transpose_and_make_dynamic() raises:
    comptime swizzle = Swizzle(1, 0, 2)
    comptime plain = row_major[4, 4]()
    comptime layout = two_level_layout_from_swizzle[type_of(plain), swizzle](
        plain
    )
    var base = layout
    var transposed = base.transpose()
    var dynamic = base.make_dynamic[.int64]()
    for r in range(4):
        for c in range(4):
            assert_equal(Int(transposed(Coord(c, r))), Int(base(Coord(r, c))))
            assert_equal(Int(dynamic(Coord(r, c))), Int(base(Coord(r, c))))


def test_two_level_nested_tile_across_footprints() raises:
    comptime swizzle = make_swizzle[num_rows=8, row_size=64, access_size=8]()
    comptime plain = row_major[64, 64]()
    comptime layout = two_level_layout_from_swizzle[type_of(plain), swizzle](
        plain
    )
    # Two whole footprints along the row mode, one along the columns.
    comptime tile = Coord[ComptimeInt[16], ComptimeInt[64]].element_types
    comptime sub_type = type_of(layout)._NestedTileLayout[tile]
    var sub_outer = rebind[sub_type.Outer](
        MixedLayout(Coord(Idx[2], Idx[1]), Coord(Idx[512], Idx[64]))
    )
    var sub = sub_type(sub_outer, 0)
    var base = Int(layout(Coord(16, 0)))
    for u in range(16):
        for v in range(64):
            assert_equal(
                Int(sub(Coord(u, v))) + base,
                Int(layout(Coord(16 + u, v))),
            )


def test_two_level_nested_tile_inside_footprint() raises:
    comptime swizzle = make_swizzle[num_rows=8, row_size=64, access_size=8]()
    comptime plain = row_major[64, 64]()
    comptime layout = two_level_layout_from_swizzle[type_of(plain), swizzle](
        plain
    )
    # An aligned 1x16 box at rows [0, 1), columns [16, 32).
    comptime tile = Coord[ComptimeInt[1], ComptimeInt[16]].element_types
    comptime sub_type = type_of(layout)._NestedTileLayout[tile]
    var sub_outer = rebind[sub_type.Outer](
        MixedLayout(Coord(Idx[1], Idx[1]), Coord(Idx[512], Idx[64]))
    )
    # Lambda of the box origin (row 0, column 16) is 16.
    var sub = sub_type(sub_outer, 16)
    for u in range(1):
        for v in range(16):
            assert_equal(Int(sub(Coord(u, v))), Int(layout(Coord(u, 16 + v))))


def test_two_level_nested_tile_inside_footprint_rows() raises:
    comptime swizzle = make_swizzle[num_rows=8, row_size=64, access_size=8]()
    comptime plain = row_major[64, 64]()
    comptime layout = two_level_layout_from_swizzle[type_of(plain), swizzle](
        plain
    )
    # A single row inside one footprint; the column mode is whole.
    comptime tile = Coord[ComptimeInt[1], ComptimeInt[64]].element_types
    comptime sub_type = type_of(layout)._NestedTileLayout[tile]
    var sub_outer = rebind[sub_type.Outer](
        MixedLayout(Coord(Idx[1], Idx[1]), Coord(Idx[512], Idx[64]))
    )
    # Lambda of row 1's inner bit is 64 ^ 8 = 72.
    var sub = sub_type(sub_outer, 72)
    for v in range(64):
        assert_equal(Int(sub(Coord(0, v))), Int(layout(Coord(1, v))))


def test_two_level_manual_construction() raises:
    # Outer shape (2, 2), stride (8, 4); inner images 5 and 1.
    comptime outer = MixedLayout(Coord(Idx[2], Idx[2]), Coord(Idx[8], Idx[4]))
    comptime inner = TypeList.of[
        Trait=CoordLike, Coord[ComptimeInt[5]], Coord[ComptimeInt[1]]
    ]()
    comptime layout_type = TwoLevelLayout[type_of(outer), inner]
    var value = layout_type(outer, 0)
    for r in range(4):
        for c in range(4):
            var expected = (r >> 1) * 8 + (c >> 1) * 4 + ((r & 1) * 5 ^ (c & 1))
            assert_equal(Int(value(Coord(r, c))), expected)


def test_two_level_tile_shape_check() raises:
    comptime swizzle = make_swizzle[num_rows=8, row_size=64, access_size=8]()
    comptime plain = row_major[64, 64]()
    comptime layout = two_level_layout_from_swizzle[type_of(plain), swizzle](
        plain
    )
    comptime whole = Coord[ComptimeInt[16], ComptimeInt[64]].element_types
    comptime box = Coord[ComptimeInt[1], ComptimeInt[16]].element_types
    assert_true(type_of(layout).require_tile_shape[whole]())
    assert_true(type_of(layout).require_tile_shape[box]())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
