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
"""Tests coordinate offsets with static and dynamic tensor strides."""

from layout import Coord, Idx, TileTensor, col_major, row_major
from layout.tile_layout import Layout
from std.testing import TestSuite, assert_equal


def test_ptr_at_offset_static_2d() raises:
    """Tests that ptr_at_offset(Coord) produces correct pointers for 2D layouts.
    """
    comptime layout = row_major[10, 20]()
    comptime total_elems = 10 * 20

    var data = Array[Int32, total_elems](
        fill_with=lambda (i: Int) -> Int32: Int32(i)
    )

    var tensor = TileTensor(data, layout)

    # Test pointer at (2, 3) -> offset = 2 * 20 + 3 = 43
    var ptr = tensor.ptr_at_offset(Coord(2, 3))
    assert_equal(ptr[], 43)

    # Test pointer at (5, 15) -> offset = 5 * 20 + 15 = 115
    ptr = tensor.ptr_at_offset(Coord(5, 15))
    assert_equal(ptr[], 115)


def test_ptr_at_offset_static_3d() raises:
    """Tests that ptr_at_offset(Coord) produces correct pointers for 3D layouts.
    """
    comptime layout = row_major[5, 10, 20]()
    comptime total_elems = 5 * 10 * 20

    var data = Array[Int32, total_elems](
        fill_with=lambda (i: Int) -> Int32: Int32(i)
    )

    var tensor = TileTensor(data, layout)

    # Test pointer at (1, 2, 3) -> offset = 1*200 + 2*20 + 3 = 243
    var ptr = tensor.ptr_at_offset(Coord(1, 2, 3))
    assert_equal(ptr[], 243)


def test_ptr_at_offset_static_4d() raises:
    """Tests that ptr_at_offset(Coord) produces correct pointers for 4D layouts.
    """
    comptime layout = row_major[2, 4, 8, 16]()
    comptime total_elems = 2 * 4 * 8 * 16

    var data = Array[Int32, total_elems](
        fill_with=lambda (i: Int) -> Int32: Int32(i)
    )

    var tensor = TileTensor(data, layout)

    # Test pointer at (1, 2, 3, 4) -> offset = 1*512 + 2*128 + 3*16 + 4 = 820
    var ptr = tensor.ptr_at_offset(Coord(1, 2, 3, 4))
    assert_equal(ptr[], 820)


def test_ptr_at_offset_col_major() raises:
    """Tests that ptr_at_offset(Coord) produces correct pointers for col-major layouts.
    """
    comptime layout = col_major[10, 20]()
    comptime total_elems = 10 * 20

    var data = Array[Int32, total_elems](
        fill_with=lambda (i: Int) -> Int32: Int32(i)
    )

    var tensor = TileTensor(data, layout)

    # Col-major: stride = (1, 10), offset = 2*1 + 3*10 = 32
    var ptr = tensor.ptr_at_offset(Coord(2, 3))
    assert_equal(ptr[], 32)


def test_ptr_at_offset_with_unknown_stride() raises:
    """Tests mixed static and dynamic strides in one layout."""
    comptime d1 = 8
    comptime d2 = 16

    # Allocate test data
    comptime total_elems = 4 * d1 * d2  # 4 * 8 * 16 = 512
    var data = Array[Int32, total_elems](
        fill_with=lambda (i: Int) -> Int32: Int32(i)
    )

    var runtime_stride_0 = d1 * d2
    var tensor = TileTensor(
        data,
        Layout(
            Coord(4, Idx[d1], Idx[d2]),
            Coord(runtime_stride_0, Idx[d2], Idx[1]),
        ),
    )

    # Test pointer at (2, 3, 5) -> offset = 2*128 + 3*16 + 5 = 309
    var ptr = tensor.ptr_at_offset(Coord(2, 3, 5))
    var expected_offset = 2 * runtime_stride_0 + 3 * d2 + 5  # = 309
    assert_equal(ptr[], Int32(expected_offset))


def test_ptr_at_offset_view_tensor() raises:
    """Tests that view tensors use correct runtime strides via ptr_at_offset.

    This simulates PagedKVCache-like scenarios where a 4D view's stride[0]
    depends on dimensions in the parent tensor that aren't in the view's shape.
    """
    comptime total_elems = 24
    var data = Array[Int32, total_elems](
        fill_with=lambda (i: Int) -> Int32: Int32(i)
    )

    # The parent row stride is larger than the view's row extent.
    var runtime_stride_0 = 8
    var child = TileTensor(
        data,
        Layout(Coord(3, Idx[4]), Coord(runtime_stride_0, Idx[1])),
    )

    # Test pointer at (1, 2) with runtime stride -> offset = 1*8 + 2*1 = 10
    var ptr = child.ptr_at_offset(Coord(1, 2))
    var expected_offset = 1 * runtime_stride_0 + 2  # = 10
    assert_equal(ptr[], Int32(expected_offset))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
