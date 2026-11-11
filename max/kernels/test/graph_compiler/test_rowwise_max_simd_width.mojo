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
"""Pins how `_rowwise` caps its SIMD width with a kernel's width limit.

As for `_foreach`: a static limit caps the width at compile time, a dynamic
one picks it at launch. The CPU wide-accumulator path, taken on long rows,
widens only as far as the limit allows: a static limit up to its largest
power-of-two divisor, a dynamic one up to the full widening only when the
limit is 0 or a multiple of it.
"""

from algorithm import rowwise
from builtin_kernels.monoids import ReduceSum
from builtin_primitives.primitives import _rowwise
from algorithm.cpu.rowwise import CPU_WIDE_ACCUMULATOR_FACTOR
from algorithm.rowwise import pick_simd_width
from std.atomic import Atomic
from layout import Coord, CoordLike, Idx
from max.gpu.host import DeviceContext
from std.collections import Array
from std.sys.info import CompilationTarget
from std.testing import TestSuite, assert_equal
from std.utils.coord import DynamicCoord

comptime ROWS = 3
# The width `_rowwise` picks for these rows, and the one the CPU path widens
# long rows to.
comptime W = pick_simd_width[
    ReduceSum[DType.float32, 1],
    CompilationTarget.current(),
    64,
    DType.float32,
    DType.float32,
]()
comptime WIDE = W * CPU_WIDE_ACCUMULATOR_FACTOR


def widest_load[
    MaxSimdWidth: CoordLike, *, width_one_launch: Bool = False
](cols: Int, max_simd_width: MaxSimdWidth) raises -> Int:
    """Sums `ROWS` rows of `cols` ones; returns the widest load the body made,
    after checking every row's sum. With `width_one_launch`, launches at width
    1 with no limit instead of through `_rowwise`."""
    var storage = Array[Int, 1 + ROWS](fill=0)
    var stats = storage.unsafe_ptr()
    # Rows run in parallel, so the widest width is kept with an atomic max.
    var widest_storage = Array[Scalar[DType.int], 1](fill=0)
    var widest = widest_storage.unsafe_ptr()

    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut ctx: rowwise.Context[params]) {
        var cols, var stats, var widest
    }:
        comptime rank = row_coords.rank

        @inline(.always)
        def load[
            width: Int, alignment: Int
        ](idx: DynamicCoord[DType.int, rank]) {var widest} -> SIMD[
            DType.float32, width
        ]:
            Atomic[Scalar[DType.int]].max(widest, Scalar[DType.int](width))
            return SIMD[DType.float32, width](1)

        var row = rowwise.Row[
            params, DType.float32, DType.float32, 1, rank, is_cached=False
        ](row_coords, cols, ctx, load)

        @inline(.always)
        def val[
            width: Int
        ](
            tile: SIMD[DType.float32, width], idx: DynamicCoord[DType.int, rank]
        ) {} -> SIMD[DType.float32, width]:
            return tile

        var acc = row.reduce[ReduceSum[DType.float32, params.simd_width]](
            val, load
        ).acc

        @inline(.always)
        def write(oc: DynamicCoord[DType.int, rank]) {var acc, var stats}:
            stats[unsafe_offset=1 + Int(oc[0])] = Int(
                acc.slice[params.emit_tile_width]()[0]
            )

        row.emit(write)

    with DeviceContext.CPU(api="cpu") as ctx:
        comptime if width_one_launch:
            rowwise.launch[axis=1, simd_width=1, num_phases=1](
                body,
                Coord(Scalar[DType.int](ROWS), Scalar[DType.int](cols)),
                ctx,
            )
            for r in range(ROWS):
                assert_equal(storage[1 + r], cols)
            return Int(widest_storage[0])
        _rowwise[
            ReduceSum[DType.float32, 1],
            DType.float32,
            DType.float32,
            axis=1,
            num_phases=1,
            computationally_expensive=False,
            dtype_size=0,
        ](
            body,
            Coord(Scalar[DType.int](ROWS), Scalar[DType.int](cols)),
            ctx,
            max_simd_width,
        )
    for r in range(ROWS):
        assert_equal(storage[1 + r], cols)
    return Int(widest_storage[0])


def dynamic(limit: Int) -> Scalar[DType.int]:
    return Scalar[DType.int](limit)


def test_no_limit_widens_long_rows() raises:
    # Long rows take the wide-accumulator path; short ones keep `W`.
    assert_equal(widest_load(64, Idx[0]), W)
    assert_equal(widest_load(4096, Idx[0]), WIDE)
    assert_equal(widest_load(4096, dynamic(0)), WIDE)


def test_width_one_never_widens() raises:
    assert_equal(widest_load[width_one_launch=True](4096, Idx[0]), 1)


def test_limit_allowing_the_widening_keeps_it() raises:
    assert_equal(widest_load(4096, Idx[WIDE]), WIDE)
    assert_equal(widest_load(4096, dynamic(WIDE)), WIDE)


def test_limit_between_widths() raises:
    # A static limit widens up to its largest power-of-two divisor; a dynamic
    # one that doesn't allow the full widening keeps `W`.
    assert_equal(widest_load(4096, Idx[2 * W]), 2 * W)
    assert_equal(widest_load(4096, dynamic(2 * W)), W)
    assert_equal(widest_load(4096, Idx[3 * W]), W)
    assert_equal(widest_load(4096, dynamic(3 * W)), W)


def test_static_limit_caps_every_path() raises:
    assert_equal(widest_load(64, Idx[2]), 2)
    assert_equal(widest_load(4096, Idx[2]), 2)
    assert_equal(widest_load(4096, Idx[6]), 2)
    assert_equal(widest_load(4096, Idx[1]), 1)


def test_dynamic_limit_caps_every_path() raises:
    assert_equal(widest_load(64, dynamic(2)), 2)
    assert_equal(widest_load(4096, dynamic(4)), 4)
    assert_equal(widest_load(4096, dynamic(3)), 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
