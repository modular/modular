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
"""Checks that `elementwise_range` visits every element of `[start, end)`
exactly once, for every `start` and `end` of a small shape, and that a call
never crosses an innermost row."""

from max.algorithm.backend.elementwise_range import elementwise_range
from std.testing import TestSuite, assert_equal

from std.utils.coord import ComptimeInt, Coord


def _check_range[simd_width: Int](shape: Coord, start: Int, end: Int) raises:
    comptime rank = shape.rank
    var total = Int(shape.product())

    var counts = List[Int](length=total, fill=0)
    var violations = List[Int](length=1, fill=0)
    var counts_span = Span(counts)
    var violations_span = Span(violations)

    def func[
        width: Int, alignment: Int = 1
    ](idx: Coord) {var counts_span, var violations_span, var shape}:
        comptime assert idx.rank == rank
        var flat = 0
        comptime for i in range(rank):
            flat = flat * Int(shape[i]) + Int(idx[i])
        var col = Int(idx[rank - 1])
        var crosses_row = col + width > Int(shape[rank - 1])
        # In rank 1 a full-width call must start on a `simd_width` boundary.
        var misaligned = (
            rank == 1 and width == simd_width and col % simd_width != 0
        )
        if crosses_row or misaligned or width > simd_width:
            violations_span.unsafe_ptr()[unsafe_offset=0] = 1
            return
        for lane in range(width):
            counts_span.unsafe_ptr()[unsafe_offset=flat + lane] += 1

    elementwise_range[simd_width](func, shape, start, end)

    assert_equal(violations[0], 0, msg=String(t"{shape} [{start}, {end})"))
    for i in range(total):
        assert_equal(
            counts[i],
            1 if start <= i < end else 0,
            msg=String(t"element {i} of {shape} [{start}, {end})"),
        )


def _check_all_ranges[simd_width: Int](shape: Coord) raises:
    var total = Int(shape.product())
    for start in range(total + 1):
        for end in range(start, total + 1):
            _check_range[simd_width](shape, start, end)


def test_rank1() raises:
    _check_all_ranges[4](Coord(13))


def test_rank2_row_shorter_than_simd_width() raises:
    _check_all_ranges[4](Coord(5, 3))


def test_rank3() raises:
    _check_all_ranges[4](Coord(2, 3, 5))


def test_rank3_static_extents() raises:
    _check_all_ranges[4](Coord(ComptimeInt[2](), 3, ComptimeInt[5]()))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
