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
"""Pins how `_rowwise` caps its SIMD width with a kernel's width limit on GPU.

A static limit caps the width at its largest power-of-two divisor, a dynamic
one picks the widest power-of-two step down that divides it; the GPU tiers
load at that width. Each row's sum is checked, including rows whose length
isn't a multiple of the width.
"""

from algorithm import rowwise
from algorithm.rowwise import pick_simd_width
from builtin_kernels.monoids import ReduceSum
from builtin_primitives.primitives import _rowwise
from layout import Coord, CoordLike, Idx
from max.gpu.host import DeviceContext, get_gpu_target
from std.atomic import Atomic
from std.testing import TestSuite, assert_equal
from std.utils.coord import DynamicCoord

comptime ROWS = 3
comptime W = pick_simd_width[
    ReduceSum[DType.float32, 1],
    get_gpu_target(),
    64,
    DType.float32,
    DType.float32,
]()


def widest_load[
    MaxSimdWidth: CoordLike
](cols: Int, max_simd_width: MaxSimdWidth) raises -> Int:
    """Sums `ROWS` rows of `cols` ones on the GPU; returns the widest load the
    body made, after checking every row's sum."""
    with DeviceContext() as ctx:
        var sums_dev = ctx.enqueue_create_buffer[DType.float32](ROWS)
        var widest_dev = ctx.enqueue_create_buffer[DType.int32](1)
        sums_dev.enqueue_fill(0)
        widest_dev.enqueue_fill(0)
        var sums = sums_dev.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var widest = widest_dev.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()

        def body[
            params: rowwise.ContextParams
        ](row_coords: Coord, mut ctx: rowwise.Context[params]) {
            var cols, var sums, var widest
        }:
            comptime rank = row_coords.rank

            @inline(.always)
            def load[
                width: Int, alignment: Int
            ](idx: DynamicCoord[DType.int, rank]) {var widest} -> SIMD[
                DType.float32, width
            ]:
                _ = Atomic[Int32].max(widest, Int32(width))
                return SIMD[DType.float32, width](1)

            var row = rowwise.Row[
                params, DType.float32, DType.float32, 1, rank, is_cached=False
            ](row_coords, cols, ctx, load)

            @inline(.always)
            def val[
                width: Int
            ](
                tile: SIMD[DType.float32, width],
                idx: DynamicCoord[DType.int, rank],
            ) {} -> SIMD[DType.float32, width]:
                return tile

            var acc = row.reduce[ReduceSum[DType.float32, params.simd_width]](
                val, load
            ).acc

            @inline(.always)
            def write(oc: DynamicCoord[DType.int, rank]) {var acc, var sums}:
                sums[Int(oc[0])] = acc.slice[params.emit_tile_width]()[0]

            row.emit(write)

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
        var sums_host = ctx.enqueue_create_host_buffer[DType.float32](ROWS)
        var widest_host = ctx.enqueue_create_host_buffer[DType.int32](1)
        ctx.enqueue_copy(sums_host, sums_dev)
        ctx.enqueue_copy(widest_host, widest_dev)
        ctx.synchronize()
        for r in range(ROWS):
            assert_equal(Int(sums_host[r]), cols)
        return Int(widest_host[0])


def dynamic(limit: Int) -> Scalar[DType.int]:
    return Scalar[DType.int](limit)


def test_no_limit() raises:
    assert_equal(widest_load(4096, Idx[0]), W)
    assert_equal(widest_load(4096, dynamic(0)), W)


def test_static_limit() raises:
    assert_equal(widest_load(4096, Idx[2]), 2)
    assert_equal(widest_load(4096, Idx[6]), 2)


def test_dynamic_limit() raises:
    assert_equal(widest_load(4096, dynamic(4)), 4)
    assert_equal(widest_load(4096, dynamic(6)), 2)
    assert_equal(widest_load(4096, dynamic(3)), 1)


def test_limit_with_a_ragged_row() raises:
    # The sums are the point here: a row whose length the width doesn't divide
    # may load narrower than the limit allows, as it does with no limit.
    var unlimited = widest_load(1001, Idx[0])
    assert_equal(widest_load(1001, Idx[4]), min(4, unlimited))
    assert_equal(widest_load(1001, dynamic(4)), min(4, unlimited))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
