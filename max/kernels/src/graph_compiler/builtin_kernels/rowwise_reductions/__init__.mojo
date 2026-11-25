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
"""Row-based row-wise reductions (free-form body layer).

Pure-reduction entry points (`reduce_argmax` / `reduce_argmin` /
`reduce_min_and_max`) built on the `algorithm.rowwise` scaffolder and the
`builtin_kernels.monoids` monoid library. Each authors one free-form body — a
single reduce phase plus a per-row `emit` — and runs on both CPU and GPU.
"""

from std.math import inf, log

from max.gpu.host import DeviceContext
from std.utils.coord import Coord, DynamicCoord

from algorithm import rowwise
from algorithm.reduce_op import ReduceOp
from algorithm.rowwise_types import at_axis
from builtin_kernels.monoids import (
    ArgMax,
    ArgMin,
    MinMax,
    TopTokenLogit,
    TopTokenPieces,
)


@inline(.always)
def _in_range_indices[
    width: Int
](indices: SIMD[.int64, width]) -> SIMD[.int64, width]:
    """Folds the `Int64.MAX` identity index to 0 in every lane.

    `ArgMax`/`ArgMin` finalize through `_in_range_index` in the tiers whose
    `rowwise.pjoin` runs the monoid's `join_parallel` (warp, block, split-K).
    The tiled and serial tiers skip that join by design -- one thread owns
    each output lane, and `join_parallel` would reset the per-lane
    accumulators -- so a lane that never saw a winner (an empty reduce axis,
    an all-NaN row, or a row of pure identity values) would emit the identity
    as a tensor value, and a downstream `gather` on `2**63-1` is an
    out-of-bounds device access (see `monoids._in_range_index`). Clamping at
    the emit site covers every tier.

    Parameters:
        width: Number of lanes in the emitted index vector.

    Args:
        indices: The per-lane winning axis indices.

    Returns:
        `indices` with every identity lane replaced by 0.
    """
    return indices.eq(SIMD[.int64, width](Int64.MAX)).select(
        SIMD[.int64, width](0), indices
    )


# ===----------------------------------------------------------------------=== #
# Row-based argument reductions: argmax / argmin. Same free-form body;
# the ArgMax / ArgMin monoid also tracks the winning index.
# ===----------------------------------------------------------------------=== #


def reduce_argmin[
    dtype: DType,
    InputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int, alignment: Int](Coord) -> SIMD[dtype, width]),
    OutputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: SIMDLength](Coord, SIMD[.int64, width]) -> None),
    /,
    *,
    reduce_dim: Int,
    name: StaticString = "reduce_argmin",
](
    input_fn: InputFn,
    output_fn: OutputFn,
    input_shape: Coord,
    context: DeviceContext.Optional = DeviceContext.NoneValue,
) raises:
    """Finds the argmin along `reduce_dim` via the Row layer: one
    `ArgMin` phase (native dtype, lower index wins ties) plus a per-row
    `emit` of the int64 index."""
    comptime simd_width = rowwise.pick_simd_width[
        ArgMin[dtype, 1], context.T.target, 64, dtype
    ]()
    comptime assert (
        0 <= reduce_dim < input_shape.rank
    ), "reduce_dim must index input_shape"
    comptime assert input_shape.is_flat, "input_shape must be flat"
    comptime op_name = name if name != "" else "reduce_argmin"
    var axis_size = Int(input_shape[reduce_dim])

    @inline(.always)
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut ctx: rowwise.Context[params]) {
        var axis_size, var input_fn, var output_fn
    }:
        comptime rank = row_coords.rank

        # Load: fuses the caller's input closure into the row's primary load.
        @inline(.always)
        def load[
            width: Int, alignment: Int
        ](idx: DynamicCoord[DType.int, rank]) {var input_fn} -> SIMD[
            dtype, width
        ]:
            return input_fn[width, alignment](idx)

        # Prepare Row: build the row view over the axis size (always
        # dynamic here — these entry points take no static_cols param).
        var row = rowwise.Row[
            params, dtype, dtype, reduce_dim, rank, keep_tile=False
        ](row_coords, Int(axis_size), ctx, load)

        # Reduce: argmin via the ArgMin monoid (lower index wins ties).
        @inline(.always)
        def val[
            width: Int
        ](
            tile: SIMD[dtype, width], idx: DynamicCoord[DType.int, rank]
        ) {} -> SIMD[dtype, width]:
            return tile

        var indices = row.reduce[ArgMin[dtype, params.simd_width]](
            val, load
        ).acc_indices

        # Emit: per-row output — the winning index.
        @inline(.always)
        def write(
            oc: DynamicCoord[DType.int, rank]
        ) {var indices, var output_fn}:
            output_fn[params.emit_tile_width](
                oc,
                _in_range_indices(indices.slice[params.emit_tile_width]()),
            )

        # `indices`/`output_fn` ride `write`'s capture list into `emit`.
        row.emit(write)

    rowwise.launch[
        axis=reduce_dim,
        simd_width=simd_width,
        num_phases=1,
        name=op_name,
    ](body, input_shape, context)


def reduce_argmax[
    dtype: DType,
    InputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int, alignment: Int](Coord) -> SIMD[dtype, width]),
    OutputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: SIMDLength](Coord, SIMD[.int64, width]) -> None),
    /,
    *,
    reduce_dim: Int,
    name: StaticString = "reduce_argmax",
](
    input_fn: InputFn,
    output_fn: OutputFn,
    input_shape: Coord,
    context: DeviceContext.Optional = DeviceContext.NoneValue,
) raises:
    comptime simd_width = rowwise.pick_simd_width[
        ArgMax[dtype, 1], context.T.target, 64, dtype
    ]()
    comptime assert (
        0 <= reduce_dim < input_shape.rank
    ), "reduce_dim must index input_shape"
    comptime assert input_shape.is_flat, "input_shape must be flat"
    comptime op_name = name if name != "" else "reduce_argmax"
    var axis_size = Int(input_shape[reduce_dim])

    @inline(.always)
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut ctx: rowwise.Context[params]) {
        var axis_size, var input_fn, var output_fn
    }:
        comptime rank = row_coords.rank

        # Load: fuses the caller's input closure into the row's primary load.
        @inline(.always)
        def load[
            width: Int, alignment: Int
        ](idx: DynamicCoord[DType.int, rank]) {var input_fn} -> SIMD[
            dtype, width
        ]:
            return input_fn[width, alignment](idx)

        # Prepare Row: build the row view over the axis size (always
        # dynamic here — these entry points take no static_cols param).
        var row = rowwise.Row[
            params, dtype, dtype, reduce_dim, rank, keep_tile=False
        ](row_coords, Int(axis_size), ctx, load)

        # Reduce: argmax via the ArgMax monoid (lower index wins ties).
        @inline(.always)
        def val[
            width: Int
        ](
            tile: SIMD[dtype, width], idx: DynamicCoord[DType.int, rank]
        ) {} -> SIMD[dtype, width]:
            return tile

        var indices = row.reduce[ArgMax[dtype, params.simd_width]](
            val, load
        ).acc_indices

        # Emit: per-row output — the winning index.
        @inline(.always)
        def write(
            oc: DynamicCoord[DType.int, rank]
        ) {var indices, var output_fn}:
            output_fn[params.emit_tile_width](
                oc,
                _in_range_indices(indices.slice[params.emit_tile_width]()),
            )

        # `indices`/`output_fn` ride `write`'s capture list into `emit`.
        row.emit(write)

    rowwise.launch[
        axis=reduce_dim,
        simd_width=simd_width,
        num_phases=1,
        name=op_name,
    ](body, input_shape, context)


# ===----------------------------------------------------------------------=== #
# Row-based fused min-and-max: one axis walk, two outputs via the
# MinMax monoid (min_acc / max_acc).
# ===----------------------------------------------------------------------=== #


def reduce_min_and_max[
    dtype: DType,
    InputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int, alignment: Int](Coord) -> SIMD[dtype, width]),
    OutputMinFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: SIMDLength](Coord, SIMD[dtype, width]) -> None),
    OutputMaxFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: SIMDLength](Coord, SIMD[dtype, width]) -> None),
    /,
    *,
    reduce_dim: Int,
    name: StaticString = "reduce_min_and_max",
](
    input_fn: InputFn,
    output_min_fn: OutputMinFn,
    output_max_fn: OutputMaxFn,
    input_shape: Coord,
    context: DeviceContext.Optional = DeviceContext.NoneValue,
) raises:
    """Computes min and max in one axis walk (`MinMax` monoid), two
    outputs."""
    comptime simd_width = rowwise.pick_simd_width[
        MinMax[dtype, 1], context.T.target, 64, dtype
    ]()
    comptime assert (
        0 <= reduce_dim < input_shape.rank
    ), "reduce_dim must index input_shape"
    comptime assert input_shape.is_flat, "input_shape must be flat"
    comptime op_name = name if name != "" else "reduce_min_and_max"
    var axis_size = Int(input_shape[reduce_dim])

    @inline(.always)
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut ctx: rowwise.Context[params]) {
        var axis_size, var input_fn, var output_min_fn, var output_max_fn
    }:
        comptime rank = row_coords.rank

        # Load: fuses the caller's input closure into the row's primary load.
        @inline(.always)
        def load[
            width: Int, alignment: Int
        ](idx: DynamicCoord[DType.int, rank]) {var input_fn} -> SIMD[
            dtype, width
        ]:
            return input_fn[width, alignment](idx)

        # Prepare Row: build the row view over the axis size (always
        # dynamic here — these entry points take no static_cols param).
        var row = rowwise.Row[
            params, dtype, dtype, reduce_dim, rank, keep_tile=False
        ](row_coords, Int(axis_size), ctx, load)

        # Reduce: min and max in one pass via the MinMax monoid.
        @inline(.always)
        def val[
            width: Int
        ](
            tile: SIMD[dtype, width], idx: DynamicCoord[DType.int, rank]
        ) {} -> SIMD[dtype, width]:
            return tile

        var stats = row.reduce[MinMax[dtype, params.simd_width]](val, load)
        var mn = stats.min_acc
        var mx = stats.max_acc

        # Emit: per-row output — both the min and the max.
        @inline(.always)
        def write(
            oc: DynamicCoord[DType.int, rank],
        ) {var mn, var mx, var output_min_fn, var output_max_fn,}:
            output_min_fn[params.emit_tile_width](
                oc,
                mn.slice[params.emit_tile_width](),
            )
            output_max_fn[params.emit_tile_width](
                oc,
                mx.slice[params.emit_tile_width](),
            )

        # `mn`/`mx`/`output_min_fn`/`output_max_fn` ride into `emit`.
        row.emit(write)

    rowwise.launch[
        axis=reduce_dim,
        simd_width=simd_width,
        num_phases=1,
        name=op_name,
    ](body, input_shape, context)


# ===-----------------------------------------------------------------------===#
# Row-based top-token reductions: one pass over the row's logits, a few
# fp32 columns per row.
# ===-----------------------------------------------------------------------===#


@inline(.always)
def _excluded_mass[
    W: Int, //, w: Int
](b: SIMD[.float32, W], t: SIMD[.float32, W]) -> SIMD[.float32, w]:
    """Returns `b + (t - 1)`, the mass outside the row's top token."""
    # `t - 1` first: left to right, a tiny `b` cancels into `t`, losing
    # the confident-row precision the excluded mass exists to keep.
    return b.slice[w]() + (t.slice[w]() - 1)


trait TopTokenColumns:
    """A top-token row reduction: its fp32 monoid and the columns it
    emits per row."""

    comptime Stats[W: Int]: ReduceOp
    """The monoid at `W` lanes, accumulating in fp32."""

    comptime columns: Int
    """The number of fp32 columns emitted per row."""

    comptime name: StaticString
    """The kernel's launch name."""

    @staticmethod
    def column[
        c: Int, W: Int, w: Int
    ](stats: Self.Stats[W]) -> SIMD[.float32, w]:
        """Returns column `c` of the joined row statistics.

        Parameters:
            c: The column index, below `columns`.
            W: The monoid's lane count.
            w: The number of rows emitted at once.

        Args:
            stats: The joined row statistics.

        Returns:
            Column `c` for `w` rows.
        """
        ...


struct TopTokenLogitColumns(TopTokenColumns):
    """Scores each row's top token as `-log(excluded_mass)`, with
    `excluded_mass = sum_{j != top} exp(l_j - l_max) + (ties - 1)`.

    An all-masked row (`t == 0`) scores `+inf`, like the reference.
    """

    comptime Stats[W: Int] = TopTokenLogit[.float32, W]
    comptime columns = 1
    comptime name: StaticString = "top_token_logit"

    @staticmethod
    @inline(.always)
    def column[
        c: Int, W: Int, w: Int
    ](stats: Self.Stats[W]) -> SIMD[.float32, w]:
        """Returns the rows' scores.

        Parameters:
            c: The column index, always 0.
            W: The monoid's lane count.
            w: The number of rows emitted at once.

        Args:
            stats: The joined row statistics.

        Returns:
            The score for `w` rows.
        """
        return (
            stats.t.slice[w]()
            .cast[.bool]()
            .select(-log(_excluded_mass[w](stats.b, stats.t)), inf[.float32]())
        )


struct TopTokenPiecesColumns[start: Int](TopTokenColumns):
    """Emits the three pieces a vocabulary-sharded greedy drafter
    allgathers: `(max, global index, excluded mass)`.

    The index is the lowest local index among the tied maxima plus
    `start`.

    Parameters:
        start: The row's shard start in the joined vocabulary.
    """

    comptime Stats[W: Int] = TopTokenPieces[.float32, W]
    comptime columns = 3
    comptime name: StaticString = "top_token_pieces"

    @staticmethod
    @inline(.always)
    def column[
        c: Int, W: Int, w: Int
    ](stats: Self.Stats[W]) -> SIMD[.float32, w]:
        """Returns piece `c` of the rows.

        Parameters:
            c: The piece: 0 the max, 1 its global index, 2 the excluded
                mass.
            W: The monoid's lane count.
            w: The number of rows emitted at once.

        Args:
            stats: The joined row statistics.

        Returns:
            Piece `c` for `w` rows.
        """
        comptime if c == 0:
            return stats.m.slice[w]()
        elif c == 1:
            return stats.i.slice[w]().cast[.float32]() + Float32(Self.start)
        else:
            return _excluded_mass[w](stats.b, stats.t)


def top_token_columns[
    dtype: DType,
    InputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int, alignment: Int](Coord) -> SIMD[dtype, width]),
    OutputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: SIMDLength](Coord, SIMD[.float32, width]) -> None),
    /,
    *,
    Columns: TopTokenColumns,
    reduce_dim: Int,
](
    input_fn: InputFn,
    output_fn: OutputFn,
    input_shape: Coord,
    context: DeviceContext.Optional = DeviceContext.NoneValue,
) raises:
    """Reduces each row's logits to `Columns`' fp32 columns via the Row
    layer: one `Columns.Stats` phase (fp32 accumulation for any input
    dtype) plus a per-row `emit` that writes column `c` at position `c`
    of the reduce axis.
    """
    comptime accum: DType = DType.float32
    comptime simd_width = rowwise.pick_simd_width[
        Columns.Stats[1], context.T.target, 64, dtype, accum
    ]()
    comptime assert (
        0 <= reduce_dim < input_shape.rank
    ), "reduce_dim must index input_shape"
    comptime assert input_shape.is_flat, "input_shape must be flat"
    var axis_size = Int(input_shape[reduce_dim])

    @inline(.always)
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut ctx: rowwise.Context[params]) {
        var axis_size, var input_fn, var output_fn
    }:
        comptime rank = row_coords.rank

        # Load: fuses the caller's input closure into the row's primary load.
        @inline(.always)
        def load[
            width: Int, alignment: Int
        ](idx: DynamicCoord[DType.int, rank]) {var input_fn} -> SIMD[
            dtype, width
        ]:
            return input_fn[width, alignment](idx)

        var row = rowwise.Row[
            params, accum, dtype, reduce_dim, rank, keep_tile=False
        ](row_coords, Int(axis_size), ctx, load)

        # Reduce: the (m, b, t) fold via the `Columns` view's monoid.
        @inline(.always)
        def val[
            width: Int
        ](
            tile: SIMD[dtype, width], idx: DynamicCoord[DType.int, rank]
        ) {} -> SIMD[accum, width]:
            return tile.cast[accum]()

        var stats = row.reduce[Columns.Stats[params.simd_width]](val, load)

        # Emit: per-row output — the columns the `Columns` view derives.
        @inline(.always)
        def write(oc: DynamicCoord[DType.int, rank]) {var stats, var output_fn}:
            comptime for c in range(Columns.columns):
                output_fn[params.emit_tile_width](
                    at_axis[reduce_dim](oc, c),
                    Columns.column[
                        c, params.simd_width, params.emit_tile_width
                    ](stats),
                )

        row.emit(write)

    rowwise.launch[
        axis=reduce_dim,
        simd_width=simd_width,
        num_phases=1,
        name=Columns.name,
    ](body, input_shape, context)
