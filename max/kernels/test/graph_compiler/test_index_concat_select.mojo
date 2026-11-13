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
"""Pins the `mogg.index.concat.select` primitive.

Every input here reads `1000 * input + 100 * row + column + lane`, so a lane
is right exactly when the primitive picked the input holding it and rebased
the column into that input. Vectors are handed out as the kernel launch
does: `width` lanes when every extent is a multiple of it, one otherwise.
"""

from builtin_primitives.primitives import mogg_index_concat_select
from layout import Coord, Idx
from std.utils import IndexList
from std.testing import TestSuite, assert_equal


def _expected(row: Int, col: Int, sizes: List[Int]) -> Int32:
    var offset = 0
    for r in range(len(sizes)):
        if col - offset < sizes[r]:
            return Int32(1000 * r + 100 * row + col - offset)
        offset += sizes[r]
    return -1


def _concat_inputs[
    input_index: Int, width: Int
](i: Coord) -> SIMD[DType.int32, width]:
    var row = Int(i[0].value())
    var col = Int(i[1].value())
    var result = SIMD[DType.int32, width]()
    comptime for lane in range(width):
        result[lane] = Int32(1000 * input_index + 100 * row + col + lane)
    return result


def _check_rows[
    width: Int, num_inputs: Int, static_sizes: Coord
](sizes: List[Int], runtime: IndexList[num_inputs]) raises:
    # Shaped like the closure the graph compiler emits around the regions.
    def inputs[input_index: Int](i: Coord) {var} -> SIMD[DType.int32, width]:
        return _concat_inputs[input_index, width](i)

    var row_length = 0
    for s in sizes:
        row_length += s
    for row in range(2):
        for col in range(0, row_length, width):
            var got = mogg_index_concat_select[
                dtype=DType.int32,
                width=width,
                num_inputs=num_inputs,
                axis=1,
                static_sizes=static_sizes,
            ](IndexList[2](row, col), inputs, runtime)
            for lane in range(width):
                assert_equal(got[lane], _expected(row, col + lane, sizes))


def test_static() raises:
    _check_rows[8, 2, Coord(Idx[16], Idx[24])]([16, 24], IndexList[2](16, 24))


def test_static_unaligned() raises:
    # A split at 13: the launch runs one-element vectors.
    _check_rows[1, 2, Coord(Idx[13], Idx[19])]([13, 19], IndexList[2](13, 19))


def test_dynamic() raises:
    _check_rows[4, 2, Coord(Int(-1), Int(-1))]([12, 20], IndexList[2](12, 20))


def test_dynamic_unaligned() raises:
    _check_rows[1, 2, Coord(Int(-1), Int(-1))]([13, 19], IndexList[2](13, 19))


def test_three_inputs() raises:
    _check_rows[4, 3, Coord(Idx[4], Int(-1), Idx[12])](
        [4, 8, 12], IndexList[3](4, 8, 12)
    )


def test_outer_axis() raises:
    # Concat along rows: every lane of a vector shares the row, so the input
    # is picked once per vector whatever the sizes.
    def inputs[input_index: Int](i: Coord) {var} -> SIMD[DType.int32, 8]:
        return _concat_inputs[input_index, 8](i)

    for row in range(5):
        var got = mogg_index_concat_select[
            dtype=DType.int32,
            width=8,
            num_inputs=2,
            axis=0,
            static_sizes=Coord(Int(-1), Int(-1)),
        ](IndexList[2](row, 0), inputs, IndexList[2](3, 2))
        var r = 0 if row < 3 else 1
        var in_row = row if row < 3 else row - 3
        for lane in range(8):
            assert_equal(got[lane], Int32(1000 * r + 100 * in_row + lane))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
