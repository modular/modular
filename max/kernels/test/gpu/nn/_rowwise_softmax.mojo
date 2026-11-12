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
"""A small Row API softmax for the rowwise kernel tests: two dependent folds
(row max, then the sum of `exp(x - max)`) and a per-element write.

The graph compiler generates the real softmax kernel; this one only gives the
tests a body with that shape, so they reach the tiers a single fold does not
(the per-element split-K launches)."""

from std.math import exp
from std.sys import size_of

from algorithm import rowwise
from builtin_kernels.monoids import ReduceMax, ReduceSum
from layout import Coord, CoordLike
from max.gpu.host import DeviceContext
from std.utils.coord import DynamicCoord
from std.utils.numerics import get_accum_type


def softmax[
    dtype: DType,
    InputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int, alignment: Int](Coord) -> SIMD[dtype, width]),
    OutputFn: ImplicitlyCopyable
    & RegisterPassable
    & (
        def[
            width: SIMDLength, alignment: Int
        ](Coord, SIMD[dtype, width]) -> None
    ),
    AxisSizeT: CoordLike,
](
    input_fn: InputFn,
    output_fn: OutputFn,
    shape: Coord,
    axis_size: AxisSizeT,
    ctx: DeviceContext,
) raises:
    """Computes softmax over the last axis of a rank-2 `shape` on the GPU."""
    comptime accum = get_accum_type[dtype]()
    comptime assert accum.is_floating_point(), "softmax requires fp accum"
    comptime simd_width = rowwise.pick_simd_width[
        ReduceSum[accum, 1], ctx.target, 64, dtype, accum
    ]()

    @inline(.always)
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut ctx_p: rowwise.Context[params]) {
        var axis_size, var input_fn, var output_fn
    }:
        comptime row_rank = row_coords.rank

        @inline(.always)
        def load[
            width: Int, alignment: Int
        ](idx: DynamicCoord[DType.int, row_rank]) {var input_fn} -> SIMD[
            dtype, width
        ]:
            return input_fn[width, alignment](idx)

        var row = rowwise.Row[
            params, accum, dtype, 1, row_rank, keep_tile=True
        ](row_coords, axis_size, ctx_p, load)

        @inline(.always)
        def cast_to_accum[
            width: Int
        ](
            tile: SIMD[dtype, width], idx: DynamicCoord[DType.int, row_rank]
        ) {} -> SIMD[accum, width]:
            return tile.cast[accum]()

        var row_max = row.reduce[ReduceMax[accum, params.simd_width]](
            cast_to_accum, load
        ).acc

        @inline(.always)
        def shifted_exp[
            width: Int
        ](tile: SIMD[dtype, width], idx: DynamicCoord[DType.int, row_rank]) {
            var row_max
        } -> SIMD[accum, width]:
            return exp(tile.cast[accum]() - row_max.slice[width]())

        var recip = (
            SIMD[accum, params.simd_width](1)
            / row.reduce[ReduceSum[accum, params.simd_width]](
                shifted_exp, load
            ).acc
        )

        @inline(.always)
        def write[
            width: Int
        ](tile: SIMD[dtype, width], idx: DynamicCoord[DType.int, row_rank]) {
            var row_max, var recip, var output_fn, var ctx_p
        }:
            comptime alignment = ctx_p.element_alignment[dtype, width]()
            var num = exp(tile.cast[accum]() - row_max.slice[width]())
            output_fn[width, alignment](
                idx, (num * recip.slice[width]()).cast[dtype]()
            )

        row.elementwise(write, load)

    # Two folds and a per-element phase; `dtype_size` enables the
    # per-element split-K tier on few long rows.
    rowwise.launch[
        axis=1,
        simd_width=simd_width,
        computationally_expensive=True,
        num_phases=3,
        dtype_size=size_of[dtype](),
    ](body, shape, ctx)
