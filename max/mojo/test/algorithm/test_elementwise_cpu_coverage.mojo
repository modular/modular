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
"""Checks the CPU `elementwise` driver against a scalar reference: every
element is visited exactly once with the right index, a call never crosses an
innermost row, and the innermost remainder runs as descending power-of-two
widths. The per-chunk row loop is also checked directly at every start row, so
the mid-row chunk-start path is covered however many workers the host has."""

from max.algorithm.backend.cpu.elementwise import _elementwise_nd_rows
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext
from std.sys import simd_width_of
from std.testing import TestSuite, assert_equal, assert_true

from std.utils.coord import Coord, Idx, coord_to_index_list
from std.utils.index import IndexList


def _largest_pow2_at_most(n: Int) -> Int:
    var p = 1
    while p * 2 <= n:
        p *= 2
    return p


def _check_rows[
    simd_width: Int
](
    shape: Coord,
    first_row: Int,
    num_rows: Int,
    ctx: DeviceContext,
    whole_driver: Bool,
) raises:
    """Runs rows [first_row, first_row + num_rows) of `shape` through
    `_elementwise_nd_rows`, or all of `shape` through the public `elementwise`
    when `whole_driver`, and checks exactly those rows were visited."""
    comptime rank = shape.rank
    var dims = rebind[IndexList[rank]](coord_to_index_list(shape))
    var total = 1
    comptime for i in range(rank):
        total *= dims[i]
    var inner = dims[rank - 1]

    # Per element: visit count. Per call-start element: the call's width.
    var counts = List[Int32](length=total, fill=0)
    var widths = List[Int32](length=total, fill=0)
    var violations = List[Int32](length=1, fill=0)
    var counts_span = Span(counts)
    var widths_span = Span(widths)
    var violations_span = Span(violations)

    def func[
        width: Int, alignment: Int = 1
    ](idx: Coord) {
        var counts_span, var widths_span, var violations_span, var dims
    }:
        var index = rebind[IndexList[rank]](coord_to_index_list(idx))
        var flat = 0
        var in_bounds = True
        comptime for i in range(rank):
            in_bounds = in_bounds and 0 <= index[i] and index[i] < dims[i]
            flat = flat * dims[i] + index[i]
        # A call must stay within its row and be a legal width.
        if (
            not in_bounds
            or index[rank - 1] + width > dims[rank - 1]
            or width > simd_width
        ):
            violations_span.unsafe_ptr()[unsafe_offset=0] = 1
            return
        widths_span.unsafe_ptr()[unsafe_offset=flat] = Int32(width)
        for lane in range(width):
            counts_span.unsafe_ptr()[unsafe_offset=flat + lane] += 1

    var rows = total // inner if inner > 0 else 0
    var first = 0
    var last = rows
    if whole_driver:
        elementwise[simd_width=simd_width](func, shape, ctx)
    else:
        comptime if rank > 1:
            _elementwise_nd_rows[simd_width, unroll_factor=8](
                func, shape, first_row, num_rows
            )
        first = first_row
        last = first_row + num_rows

    assert_equal(violations[0], 0)
    for i in range(total):
        var covered = rows > 0 and first <= i // inner and i // inner < last
        assert_equal(counts[i], Int32(1) if covered else Int32(0))

    # The 1D path keeps `vectorize`'s width-1 remainder; only the ND driver
    # promises a power-of-two remainder.
    comptime if rank > 1:
        for row in range(first, last):
            var j = 0
            while j < inner:
                var w = Int(widths[row * inner + j])
                assert_true(w > 0)
                var remaining = inner - j
                var expected = simd_width
                if remaining < simd_width:
                    expected = _largest_pow2_at_most(remaining)
                assert_equal(w, expected)
                j += w


def _check[simd_width: Int](shape: Coord, ctx: DeviceContext) raises:
    _check_rows[simd_width](shape, 0, 0, ctx, True)


def _sweep_inner[
    simd_width: Int, outer_rank: Int
](outer: IndexList[outer_rank], ctx: DeviceContext) raises:
    """Runs `_check` with dynamic dims for every innermost size that hits a
    distinct remainder pattern."""
    for inner in range(2 * simd_width + 2):
        var dims = IndexList[outer_rank + 1]()
        comptime for i in range(outer_rank):
            dims[i] = outer[i]
        dims[outer_rank] = inner
        _check[simd_width](Coord(dims), ctx)
    for inner in [3, 4, 31, 33, 63, 65]:
        var dims = IndexList[outer_rank + 1]()
        comptime for i in range(outer_rank):
            dims[i] = outer[i]
        dims[outer_rank] = inner
        _check[simd_width](Coord(dims), ctx)


def _test_remainders[simd_width: Int]() raises:
    var ctx = DeviceContext.cpu()
    _sweep_inner[simd_width](IndexList[1](3), ctx)
    _sweep_inner[simd_width](IndexList[1](7), ctx)
    _sweep_inner[simd_width](IndexList[2](3, 5), ctx)
    _sweep_inner[simd_width](IndexList[3](2, 3, 5), ctx)
    _sweep_inner[simd_width](IndexList[4](2, 1, 3, 7), ctx)
    # Zero in an outer dim.
    _sweep_inner[simd_width](IndexList[1](0), ctx)
    _sweep_inner[simd_width](IndexList[2](3, 0), ctx)
    _sweep_inner[simd_width](IndexList[3](0, 3, 5), ctx)


def test_remainders_native_u8_width() raises:
    _test_remainders[simd_width_of[DType.uint8]()]()


def test_remainders_width_8() raises:
    _test_remainders[8]()


def test_remainders_width_4() raises:
    _test_remainders[4]()


def test_remainders_width_1() raises:
    _test_remainders[1]()


def test_rank1() raises:
    var ctx = DeviceContext.cpu()
    comptime sw = simd_width_of[DType.uint8]()
    for n in range(2 * sw + 2):
        _check[sw](Coord(IndexList[1](n)), ctx)
    _check[sw](Coord(IndexList[1](100003)), ctx)


def test_every_chunk_start() raises:
    """Every (start row, row count) window, so a chunk may begin mid-row of any
    outer dim and carry across each dim boundary. This is the path a worker
    other than the first takes, independent of the host's worker count."""
    var ctx = DeviceContext.cpu()
    comptime sw = simd_width_of[DType.uint8]()
    for inner in [1, 3, 4, sw - 1, sw + 3]:
        var shape = Coord(IndexList[4](2, 3, 2, inner))
        var rows = 2 * 3 * 2
        for start in range(rows):
            for n in range(1, rows - start + 1):
                _check_rows[sw](shape, start, n, ctx, False)
    var shape3 = Coord(IndexList[3](3, 4, 5))
    for start in range(12):
        for n in range(1, 12 - start + 1):
            _check_rows[8](shape3, start, n, ctx, False)


def test_multiple_chunks() raises:
    """Large shapes through the public driver, with prime outer dims. Several
    workers run only if the host has the cores for it; `test_every_chunk_start`
    is what guarantees the cross-chunk path."""
    var ctx = DeviceContext.cpu()
    comptime sw = simd_width_of[DType.uint8]()
    _check[sw](Coord(IndexList[2](40009, 3)), ctx)
    _check[sw](Coord(IndexList[2](10007, 31)), ctx)
    _check[sw](Coord(IndexList[3](13, 3079, 5)), ctx)
    _check[sw](Coord(IndexList[4](7, 11, 523, 4)), ctx)
    _check[sw](Coord(IndexList[4](3, 17, 1009, 33)), ctx)
    _check[sw](Coord(IndexList[5](2, 3, 5, 1301, 7)), ctx)
    _check[8](Coord(IndexList[3](101, 401, 3)), ctx)


def test_static_and_mixed_dims() raises:
    var ctx = DeviceContext.cpu()
    comptime sw = simd_width_of[DType.uint8]()
    _check[sw](Coord(Idx[3], Idx[5], Idx[4]), ctx)
    _check[sw](Coord(Idx[7], Idx[3], Idx[sw + 3]), ctx)
    _check[sw](Coord(Idx[2], Idx[3], Idx[5], Idx[2 * sw + 1]), ctx)
    _check[sw](Coord(Idx[0], Idx[3], Idx[4]), ctx)
    _check[sw](Coord(Idx[3], Idx[5], Idx[0]), ctx)
    # Large enough for several chunks, static and dynamic outer dims.
    _check[sw](Coord(Idx[40009], Idx[3]), ctx)
    _check[sw](Coord(Idx[7], Idx[11], Idx[523], Idx[4]), ctx)
    _check[sw](Coord(Int(10007), Idx[31]), ctx)
    _check[sw](Coord(Idx[13], Int(3079), Idx[5]), ctx)
    _check[sw](Coord(Idx[13], Idx[3079], Int(5)), ctx)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
