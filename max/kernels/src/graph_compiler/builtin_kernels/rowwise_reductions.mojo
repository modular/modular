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

Pure-reduction entry points (`reduce_product` / `reduce_argmax` /
`reduce_argmin` / `reduce_min_and_max`) built on the `algorithm.rowwise`
scaffolder and the `builtin_kernels.monoids` monoid library. Each authors one free-form body — a
single reduce phase plus a per-row `emit` — and runs on both CPU and GPU.
"""

from max.gpu.host import DeviceContext
from std.utils.coord import Coord

from algorithm import rowwise
from algorithm.rowwise_types import RowCoord
from .monoids import (
    ArgMax,
    ArgMin,
    MinMax,
    ReduceProduct,
)


def reduce_product[
    dtype: DType,
    InputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int, alignment: Int](Coord) -> SIMD[dtype, width]),
    OutputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: SIMDLength](Coord, SIMD[dtype, width]) -> None),
    /,
    target: StaticString,
    *,
    reduce_dim: Int,
    name: StaticString = "reduce_product",
](
    input_fn: InputFn,
    output_fn: OutputFn,
    input_shape: Coord,
    context: Optional[DeviceContext] = None,
) raises:
    comptime simd_width = rowwise.pick_simd_width[
        ReduceProduct[dtype, 1], target, 64, dtype
    ]()
    comptime assert (
        0 <= reduce_dim < input_shape.rank
    ), "reduce_dim must index input_shape"
    comptime assert input_shape.is_flat, "input_shape must be flat"
    comptime op_name = name if name != "" else "reduce_product"
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
        ](idx: RowCoord[rank]) {var input_fn} -> SIMD[dtype, width]:
            return input_fn[width, alignment](idx.coord)

        # Prepare Row: build the row view over the axis size (always
        # dynamic here — these entry points take no static_cols param).
        var row = rowwise.Row[
            params, dtype, dtype, reduce_dim, rank, is_cached=False
        ](row_coords, Int(axis_size), ctx, load)

        # Reduce: product via the ReduceProduct monoid.
        @inline(.always)
        def val[
            width: Int
        ](tile: SIMD[dtype, width], idx: RowCoord[rank]) {} -> SIMD[
            dtype, width
        ]:
            return tile

        var acc = row.reduce[ReduceProduct[dtype, params.simd_width]](
            val, load
        ).acc

        # Emit: per-row output — the reduced scalar.
        @inline(.always)
        def write(oc: RowCoord[rank]) {var acc, var output_fn}:
            output_fn[params.emit_tile_width](
                oc.coord,
                acc.slice[params.emit_tile_width](),
            )

        # `acc`/`output_fn` ride `write`'s capture list into `emit`.
        row.emit(write)

    rowwise.launch[
        axis=reduce_dim,
        simd_width=simd_width,
        target=target,
        num_phases=1,
        name=op_name,
    ](body, input_shape, context)


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
    target: StaticString,
    *,
    reduce_dim: Int,
    name: StaticString = "reduce_argmin",
](
    input_fn: InputFn,
    output_fn: OutputFn,
    input_shape: Coord,
    context: Optional[DeviceContext] = None,
) raises:
    """Finds the argmin along `reduce_dim` via the Row layer: one
    `ArgMin` phase (native dtype, lower index wins ties) plus a per-row
    `emit` of the int64 index."""
    comptime simd_width = rowwise.pick_simd_width[
        ArgMin[dtype, 1], target, 64, dtype
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
        ](idx: RowCoord[rank]) {var input_fn} -> SIMD[dtype, width]:
            return input_fn[width, alignment](idx.coord)

        # Prepare Row: build the row view over the axis size (always
        # dynamic here — these entry points take no static_cols param).
        var row = rowwise.Row[
            params, dtype, dtype, reduce_dim, rank, is_cached=False
        ](row_coords, Int(axis_size), ctx, load)

        # Reduce: argmin via the ArgMin monoid (lower index wins ties).
        @inline(.always)
        def val[
            width: Int
        ](tile: SIMD[dtype, width], idx: RowCoord[rank]) {} -> SIMD[
            dtype, width
        ]:
            return tile

        var indices = row.reduce[ArgMin[dtype, params.simd_width]](
            val, load
        ).acc_indices

        # Emit: per-row output — the winning index.
        @inline(.always)
        def write(oc: RowCoord[rank]) {var indices, var output_fn}:
            output_fn[params.emit_tile_width](
                oc.coord,
                indices.slice[params.emit_tile_width](),
            )

        # `indices`/`output_fn` ride `write`'s capture list into `emit`.
        row.emit(write)

    rowwise.launch[
        axis=reduce_dim,
        simd_width=simd_width,
        target=target,
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
    target: StaticString,
    *,
    reduce_dim: Int,
    name: StaticString = "reduce_argmax",
](
    input_fn: InputFn,
    output_fn: OutputFn,
    input_shape: Coord,
    context: Optional[DeviceContext] = None,
) raises:
    comptime simd_width = rowwise.pick_simd_width[
        ArgMax[dtype, 1], target, 64, dtype
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
        ](idx: RowCoord[rank]) {var input_fn} -> SIMD[dtype, width]:
            return input_fn[width, alignment](idx.coord)

        # Prepare Row: build the row view over the axis size (always
        # dynamic here — these entry points take no static_cols param).
        var row = rowwise.Row[
            params, dtype, dtype, reduce_dim, rank, is_cached=False
        ](row_coords, Int(axis_size), ctx, load)

        # Reduce: argmax via the ArgMax monoid (lower index wins ties).
        @inline(.always)
        def val[
            width: Int
        ](tile: SIMD[dtype, width], idx: RowCoord[rank]) {} -> SIMD[
            dtype, width
        ]:
            return tile

        var indices = row.reduce[ArgMax[dtype, params.simd_width]](
            val, load
        ).acc_indices

        # Emit: per-row output — the winning index.
        @inline(.always)
        def write(oc: RowCoord[rank]) {var indices, var output_fn}:
            output_fn[params.emit_tile_width](
                oc.coord,
                indices.slice[params.emit_tile_width](),
            )

        # `indices`/`output_fn` ride `write`'s capture list into `emit`.
        row.emit(write)

    rowwise.launch[
        axis=reduce_dim,
        simd_width=simd_width,
        target=target,
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
    target: StaticString,
    *,
    reduce_dim: Int,
    name: StaticString = "reduce_min_and_max",
](
    input_fn: InputFn,
    output_min_fn: OutputMinFn,
    output_max_fn: OutputMaxFn,
    input_shape: Coord,
    context: Optional[DeviceContext] = None,
) raises:
    """Computes min and max in one axis walk (`MinMax` monoid), two
    outputs."""
    comptime simd_width = rowwise.pick_simd_width[
        MinMax[dtype, 1], target, 64, dtype
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
        ](idx: RowCoord[rank]) {var input_fn} -> SIMD[dtype, width]:
            return input_fn[width, alignment](idx.coord)

        # Prepare Row: build the row view over the axis size (always
        # dynamic here — these entry points take no static_cols param).
        var row = rowwise.Row[
            params, dtype, dtype, reduce_dim, rank, is_cached=False
        ](row_coords, Int(axis_size), ctx, load)

        # Reduce: min and max in one pass via the MinMax monoid.
        @inline(.always)
        def val[
            width: Int
        ](tile: SIMD[dtype, width], idx: RowCoord[rank]) {} -> SIMD[
            dtype, width
        ]:
            return tile

        var stats = row.reduce[MinMax[dtype, params.simd_width]](val, load)
        var mn = stats.min_acc
        var mx = stats.max_acc

        # Emit: per-row output — both the min and the max.
        @inline(.always)
        def write(
            oc: RowCoord[rank],
        ) {var mn, var mx, var output_min_fn, var output_max_fn,}:
            output_min_fn[params.emit_tile_width](
                oc.coord,
                mn.slice[params.emit_tile_width](),
            )
            output_max_fn[params.emit_tile_width](
                oc.coord,
                mx.slice[params.emit_tile_width](),
            )

        # `mn`/`mx`/`output_min_fn`/`output_max_fn` ride into `emit`.
        row.emit(write)

    rowwise.launch[
        axis=reduce_dim,
        simd_width=simd_width,
        target=target,
        num_phases=1,
        name=op_name,
    ](body, input_shape, context)
