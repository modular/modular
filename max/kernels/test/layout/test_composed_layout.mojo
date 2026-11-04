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

from layout import Coord, Idx, coord
from layout.swizzle import ComposedLayout, Swizzle
from layout.tile_layout import Layout
from std.testing import assert_equal


comptime _BASE_SHAPE = Coord(Idx[6], Idx[2])
"""Shape shared by the composed-layout fixtures."""

comptime _BASE_STRIDE = Coord(Idx[8], Idx[2])
"""Strides shared by the composed-layout fixtures."""


def test_composed_layout_swizzle() raises:
    print("== test_composed_layout_swizzle")

    var swizzle = Swizzle(1, 0, 2)
    comptime layout = Layout(_BASE_SHAPE, _BASE_STRIDE)

    var comp_layout = ComposedLayout[type_of(layout), layout](swizzle)

    assert_equal(comp_layout(Coord(Idx[0], Idx[0])), 0)
    assert_equal(comp_layout(Coord(Idx[1], Idx[0])), 8)
    assert_equal(comp_layout(Coord(Idx[2], Idx[0])), 16)
    assert_equal(comp_layout(Coord(Idx[3], Idx[0])), 24)
    assert_equal(comp_layout(Coord(Idx[4], Idx[0])), 32)
    assert_equal(comp_layout(Coord(Idx[5], Idx[0])), 40)


def test_composed_layout_swizzle_rt() raises:
    print("== test_composed_layout_swizzle_rt")

    var swizzle = Swizzle(1, 0, 2)
    comptime layout = Layout(_BASE_SHAPE, _BASE_STRIDE)

    var comp_layout = ComposedLayout[type_of(layout), layout](swizzle)

    assert_equal(comp_layout(Coord(0, 0)), 0)
    assert_equal(comp_layout(Coord(1, 0)), 8)
    assert_equal(comp_layout(Coord(2, 0)), 16)
    assert_equal(comp_layout(Coord(3, 0)), 24)
    assert_equal(comp_layout(Coord(4, 0)), 32)
    assert_equal(comp_layout(Coord(5, 0)), 40)


def test_layout_coord_index() raises:
    print("== test_layout_coord_index")

    comptime layout = Layout(_BASE_SHAPE, _BASE_STRIDE)

    # Compile-time Coord leaves must match the same crd2idx arithmetic as
    # runtime Coord leaves.
    comptime for i in range(6):
        comptime for j in range(2):
            assert_equal(
                layout(coord[i, j]),
                layout(Coord(Idx[i], Idx[j])),
            )

    # `coord[i, j]` and an explicit static `Coord` must agree.
    comptime for i in range(6):
        comptime for j in range(2):
            assert_equal(layout(coord[i, j]), layout(Coord(Idx[i], Idx[j])))


def test_composed_layout_coord() raises:
    print("== test_composed_layout_coord")

    var swizzle = Swizzle(1, 0, 2)
    comptime layout = Layout(_BASE_SHAPE, _BASE_STRIDE)
    var comp_layout = ComposedLayout[type_of(layout), layout](swizzle)

    # A composed layout is the swizzle applied to the base layout's offset.
    for i in range(6):
        for j in range(2):
            assert_equal(
                comp_layout(Coord(i, j)), swizzle(Int(layout(Coord(i, j))))
            )


def main() raises:
    test_composed_layout_swizzle()
    test_composed_layout_swizzle_rt()
    test_layout_coord_index()
    test_composed_layout_coord()
