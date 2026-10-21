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
"""Row-based softmax built on the `algorithm.rowwise` scaffolder and the
`builtin_kernels.monoids` monoid library."""

from std.math import exp, log
from std.collections import Optional
from std.sys import align_of, size_of

from max.gpu.host import DeviceContext
from layout import Coord, CoordLike, TileTensor
from std.utils.numerics import get_accum_type

from algorithm import rowwise
from algorithm.rowwise_types import RowCoord

from .monoids import ReduceMax, ReduceSum


# ===----------------------------------------------------------------------=== #
# Row-based softmax (free-form body layer).
#
# Three phases over a Row: ReduceMax, ReduceSum-of-exp (reads the max via a
# captured scalar), then a per-element `map`. On the cache-eligible tier the row
# is materialized once and replayed from registers across both reductions + the
# map (one load per element); otherwise it streams. A sink term — which the
# declarative pipeline could not carry — would fold into `l` as one extra line.
# ===----------------------------------------------------------------------=== #


def softmax[
    dtype: DType,
    rank: Int,
    InputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int, alignment: Int](Coord) -> SIMD[dtype, width]),
    AxisSizeT: CoordLike,
    /,
    target: StaticString,
    logsoftmax: Bool = False,
    reduce_dim: Int = rank - 1,
](
    input_fn: InputFn,
    shape: Coord,
    axis_size: AxisSizeT,
    output: TileTensor[mut=True, dtype, ...],
    axis: Int,
    context: Optional[DeviceContext] = None,
) raises:
    # softmax's two dependent reduces (row max, then sum of `exp(x - max)`)
    # plus its final normalizing write are passed to the scaffolder as
    # `num_phases=3`, which lets it split each row across many blocks on the
    # GPU when a few rows leave the device under-occupied (e.g. Kimi
    # 8x163840); CPU ignores it. No real caller has ever needed a different
    # value, so it's hardcoded at the `launch` call below rather than
    # exposed here.
    comptime assert shape.rank == rank, "shape.rank must be the same as rank"
    comptime assert shape.is_flat, "shape must be flat"
    comptime assert output.rank == rank, "output.rank must be the same as rank"
    comptime assert 0 <= reduce_dim < rank, "reduce_dim must be in [0, rank)"
    comptime accum = get_accum_type[dtype]()
    comptime assert accum.is_floating_point(), "softmax requires fp accum"
    comptime simd_width = rowwise.pick_simd_width[
        ReduceSum[accum, 1], target, 64, dtype, accum
    ]()

    @inline(.always)
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut ctx_p: rowwise.Context[params]) {
        var axis_size, var output, var input_fn
    }:
        comptime row_rank = row_coords.rank

        # Load: fuses the caller's input closure into the row's primary load.
        @inline(.always)
        def load[
            width: Int, alignment: Int
        ](idx: RowCoord[row_rank]) {var input_fn} -> SIMD[dtype, width]:
            return input_fn[width, alignment](idx.coord)

        var row = rowwise.Row[
            params, accum, dtype, reduce_dim, row_rank, is_cached=True
        ](row_coords, axis_size, ctx_p, load)

        # Reduce (phase 1): row max.
        @inline(.always)
        def vmax[
            width: Int
        ](tile: SIMD[dtype, width], idx: RowCoord[row_rank]) {} -> SIMD[
            accum, width
        ]:
            return tile.cast[accum]()

        # Row stats held W-wide (one value broadcast to every lane, or one
        # per output column, depending on how the launch maps threads to
        # outputs); read via `.slice[width]` uniformly either way.
        var row_max = row.reduce[ReduceMax[accum, params.simd_width]](
            vmax, load
        ).acc

        # Reduce (phase 2): sum of exp(tile - row_max).
        @inline(.always)
        def vexp[
            width: Int
        ](tile: SIMD[dtype, width], idx: RowCoord[row_rank]) {
            var row_max
        } -> SIMD[accum, width]:
            return exp(tile.cast[accum]() - row_max.slice[width]())

        var denom = row.reduce[ReduceSum[accum, params.simd_width]](
            vexp, load
        ).acc
        var recip = SIMD[accum, params.simd_width](1) / denom
        var log_denom = log(denom)

        # Emit: per-element normalize + store.
        @inline(.always)
        def write[
            width: Int
        ](tile: SIMD[dtype, width], idx: RowCoord[row_rank]) {
            var row_max,
            var recip,
            var log_denom,
            var output,
            var ctx_p,
        }:
            # `element_alignment` counts elements; `TileTensor.store` takes
            # bytes. Passing the count through under-claims alignment and splits
            # the store (a bf16 scalar becomes two byte stores).
            comptime alignment = ctx_p.element_alignment[
                dtype, width
            ]() * align_of[dtype]()
            var tile_accum = tile.cast[accum]()

            comptime if logsoftmax:
                var shifted = (
                    tile_accum
                    - row_max.slice[width]()
                    - log_denom.slice[width]()
                )
                var result = shifted.cast[dtype]()
                output.store[alignment=alignment](idx.coord, result)
            else:
                var numerator = exp(tile_accum - row_max.slice[width]())
                var result = (numerator * recip.slice[width]()).cast[dtype]()
                output.store[alignment=alignment](idx.coord, result)

        # `row_max`/`recip`/`log_denom`/`output` ride `write`'s capture list
        # into `elementwise`.
        row.elementwise(write, load)

    rowwise.launch[
        axis=reduce_dim,
        simd_width=simd_width,
        target=target,
        computationally_expensive=True,
        num_phases=3,
        dtype_size=size_of[dtype](),
    ](body, shape, context)
