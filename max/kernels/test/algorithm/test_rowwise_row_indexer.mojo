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
"""Checks the rowwise GPU row decode against the divide-based helper it
replaces, on the host.

The kernels' own tests read and write through the same decoded coordinates,
so they cannot tell a wrong decode from a right one as long as it is a
bijection. This compares `_RowIndexer.coords` to
`_get_nd_indices_from_flat_index` coordinate by coordinate, over every rank
from 1 to 4, every reduced axis, extents of 1 in divided positions,
non-power-of-two extents, an extent near 2**31, and the `wide` path for row
counts past 32 bits."""

from std.testing import assert_equal, TestSuite
from std.utils.coord import Coord, DynamicCoord
from std.utils.index import IndexList

from algorithm.gpu.rowwise import _RowIndexer
from max.algorithm.reduction import _get_nd_indices_from_flat_index


def _check[
    rank: Int, axis: Int, wide: Bool
](shape: IndexList[rank], rows: List[Int]) raises:
    var shape_dc = Coord(shape)
    var indexer = _RowIndexer[rank, axis, wide](shape_dc)
    for row in rows:
        var got = indexer.coords(shape_dc, row)
        var want = _get_nd_indices_from_flat_index(row, shape_dc, axis)
        comptime for i in range(rank):
            assert_equal(
                Int(got[i].value()),
                Int(want[i].value()),
                String(
                    "rank ",
                    rank,
                    " axis ",
                    axis,
                    " wide ",
                    wide,
                    " shape ",
                    shape,
                    " row ",
                    row,
                    " dim ",
                    i,
                ),
            )


def _rows_in(num_rows: Int) -> List[Int]:
    # Both ends, a spread through the middle, and the ends' neighbors.
    var rows = List[Int]()
    var picks: List[Int] = [0, 1, 2, 3, 5, 7]
    for p in picks:
        if p < num_rows:
            rows.append(p)
    var steps = 29
    for k in range(1, steps):
        rows.append((num_rows - 1) * k // steps)
    for p in picks:
        if num_rows - 1 - p >= 0:
            rows.append(num_rows - 1 - p)
    return rows^


def _num_rows[rank: Int](shape: IndexList[rank], axis: Int) -> Int:
    var n = 1
    for i in range(rank):
        if i != axis:
            n *= shape[i]
    return n


def _check_all_axes[rank: Int](shape: IndexList[rank]) raises:
    comptime for axis in range(rank):
        _check[rank, axis, False](shape, _rows_in(_num_rows(shape, axis)))


def test_rank1() raises:
    _check_all_axes(IndexList[1](7))
    _check_all_axes(IndexList[1](1))


def test_rank2() raises:
    _check_all_axes(IndexList[2](3, 5))
    _check_all_axes(IndexList[2](1, 4096))
    _check_all_axes(IndexList[2](8192, 1))
    _check_all_axes(IndexList[2](131071, 3))


def test_rank3() raises:
    _check_all_axes(IndexList[3](2, 3, 5))
    _check_all_axes(IndexList[3](7, 11, 13))
    # An extent of 1 in a divided position.
    _check_all_axes(IndexList[3](4, 1, 9))
    _check_all_axes(IndexList[3](1, 1, 1))
    _check_all_axes(IndexList[3](1, 8192, 4096))
    _check_all_axes(IndexList[3](4, 2048, 4096))
    # Non-power-of-two extents around the 16-bit boundary.
    _check_all_axes(IndexList[3](65537, 3, 2))
    _check_all_axes(IndexList[3](3, 65535, 7))


def test_rank4() raises:
    _check_all_axes(IndexList[4](2, 3, 5, 7))
    _check_all_axes(IndexList[4](3, 1, 6, 1))
    _check_all_axes(IndexList[4](1, 1, 1, 9))
    _check_all_axes(IndexList[4](2, 3, 5, 8192))
    _check_all_axes(IndexList[4](17, 19, 23, 29))


def test_extent_near_2_pow_31() raises:
    # The divided dim is close to 2**31; the row count stays below 2**32.
    comptime big = (1 << 31) - 1
    _check[3, 2, False](IndexList[3](1, big, 2), _rows_in(big))
    _check[3, 2, False](IndexList[3](2, big, 2), _rows_in(2 * big))
    _check[3, 0, False](IndexList[3](2, big, 1), _rows_in(big))
    _check[4, 3, False](IndexList[4](1, 1, big, 2), _rows_in(big))


def test_wide_rows() raises:
    # Row counts past 32 bits take the 64-bit path, picked on the host by
    # `launch`; the decode must still match the helper.
    comptime big = 1 << 31
    _check[3, 2, True](IndexList[3](2, big, 2), _rows_in(2 * big))
    _check[3, 2, True](IndexList[3](3, big + 1, 2), _rows_in(3 * (big + 1)))
    _check[4, 3, True](IndexList[4](2, 3, big, 2), _rows_in(6 * big))
    _check[3, 1, True](IndexList[3](big, 2, 3), _rows_in(3 * big))
    # The wide path is also valid below 32 bits.
    _check[3, 2, True](IndexList[3](7, 11, 13), _rows_in(77))
    _check[2, 1, True](IndexList[2](3, 5), _rows_in(3))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
