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
"""Row-based normalization kernels (rms_norm, layer_norm and fused
variants) built on the `algorithm.rowwise` scaffolder and the
`builtin_kernels.monoids` monoid library."""

from std.math import ceildiv, rsqrt
from std.memory import stack_allocation
from std.sys import size_of
from std.sys.info import align_of

from max.gpu import WARP_SIZE, lane_id, warp_id
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu.primitives import warp
from max.gpu.primitives.grid_controls import PDLLevel
from layout import Coord, CoordLike, TileTensor
from std.utils.coord import DynamicCoord
from std.utils.numerics import get_accum_type
from internal_utils.fp8_utils import compute_dynamic_fp8_scale, fp8_quantize

from algorithm import rowwise
from algorithm.rowwise import strided_load
from algorithm.rowwise_types import ReduceTier, at_axis
from nn._ragged_utils import get_batch_from_row_offsets
from nn.rope import _rope

from .monoids import ReduceMax, ReduceSum, Welford


# ===----------------------------------------------------------------------=== #
# Row-based normalizations (free-form body layer). The graph compiler
# generates plain rms_norm from its parts.
#
# layer_norm: one Welford phase (mean + variance in a single pass) -> map
# (normalize * gamma + beta). gamma/beta are broadcast side-loads inside the
# map, like the GC IR's `%weight` load. The row is cached once and replayed
# across the reduce + map on the cache-eligible tier.
# ===----------------------------------------------------------------------=== #


def rms_norm_residual_add[
    dtype: DType,
    rank: Int,
    InputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int, alignment: Int](Coord) -> SIMD[dtype, width]),
    ResidualInputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int, alignment: Int](Coord) -> SIMD[dtype, width]),
    OutputFn: ImplicitlyCopyable
    & RegisterPassable
    & (
        def[
            width: SIMDLength, alignment: Int
        ](Coord, SIMD[dtype, width]) -> None
    ),
    OutputResidualFn: ImplicitlyCopyable
    & RegisterPassable
    & (
        def[
            width: SIMDLength, alignment: Int
        ](Coord, SIMD[dtype, width]) -> None
    ),
    AxisSizeT: CoordLike,
    /,
    epsilon: Float32,
    weight_offset: Float32,
    multiply_before_cast: Bool = True,
    reduce_dim: Int = rank - 1,
](
    input_fn: InputFn,
    residual_input_fn: ResidualInputFn,
    output_fn: OutputFn,
    output_residual_fn: OutputResidualFn,
    shape: Coord,
    axis_size: AxisSizeT,
    gamma: TileTensor[mut=False, dtype, ...],
    context: DeviceContext.Optional = DeviceContext.NoneValue,
) raises:
    """Row-based `rms_norm(input + residual)` that also emits the sum.

    The residual add is fused into the row's load, so the sum is read once,
    cached, normalized, and stored as the residual output on the way out.
    """
    comptime accum = get_accum_type[dtype]()
    comptime assert accum.is_floating_point(), "rms_norm requires fp accum"
    comptime assert shape.rank == rank, "shape.rank must be the same as rank"
    comptime assert shape.is_flat, "shape must be flat"
    comptime simd_width = rowwise.pick_simd_width[
        ReduceSum[accum, 1],
        context.T.target,
        64,
        dtype,
        accum,
    ]()
    var axis_size_accum = Scalar[accum](Int(axis_size))

    @inline(.always)
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut ctx_p: rowwise.Context[params]) {
        var axis_size,
        var gamma,
        var axis_size_accum,
        var input_fn,
        var residual_input_fn,
        var output_fn,
        var output_residual_fn,
    }:
        comptime row_rank = row_coords.rank

        # The add is rounded to `dtype` here, as the unfused add would.
        @inline(.always)
        def load[
            width: Int, alignment: Int
        ](idx: DynamicCoord[DType.int, row_rank]) {
            var input_fn, var residual_input_fn
        } -> SIMD[dtype, width]:
            return input_fn[width, alignment](idx) + residual_input_fn[
                width, alignment
            ](idx)

        var row = rowwise.Row[
            params, accum, dtype, reduce_dim, row_rank, is_cached=True
        ](row_coords, axis_size, ctx_p, load)

        @inline(.always)
        def square[
            width: Int
        ](
            tile: SIMD[dtype, width], idx: DynamicCoord[DType.int, row_rank]
        ) {} -> SIMD[accum, width]:
            var tile_accum = tile.cast[accum]()
            return tile_accum * tile_accum

        var ssq = row.reduce[ReduceSum[accum, params.simd_width]](
            square, load
        ).acc
        var inv_rms = rsqrt(
            ssq / axis_size_accum + comptime (epsilon.cast[accum]())
        )

        comptime g_stride = 1 if reduce_dim == rank - 1 else 0

        @inline(.always)
        def write[
            width: Int
        ](tile: SIMD[dtype, width], idx: DynamicCoord[DType.int, row_rank]) {
            var gamma,
            var inv_rms,
            var output_fn,
            var output_residual_fn,
            var ctx_p,
        }:
            comptime alignment = ctx_p.element_alignment[dtype, width]()
            var col = Int(idx[reduce_dim])
            output_residual_fn[width, alignment](idx, tile)
            var g_raw = strided_load[width, g_stride, alignment=alignment](
                gamma.ptr_at_offset(Coord(col))
            )
            var normed = tile.cast[accum]() * inv_rms.slice[width]()

            comptime if multiply_before_cast:
                var gamma_with_offset = g_raw.cast[accum]() + comptime (
                    weight_offset.cast[dtype]().cast[accum]()
                )
                var result = (normed * gamma_with_offset).cast[dtype]()
                output_fn[width, alignment](idx, result)
            else:
                var gamma_with_offset = g_raw + comptime (
                    weight_offset.cast[dtype]()
                )
                var result = normed.cast[dtype]() * gamma_with_offset
                output_fn[width, alignment](idx, result)

        row.elementwise(write, load)

    rowwise.launch[
        axis=reduce_dim,
        simd_width=simd_width,
        num_phases=2,
        dtype_size=size_of[dtype](),
        computationally_expensive=True,
        pdl_level=PDLLevel.OVERLAP_AT_BEGINNING,
    ](body, shape, context)


# ===----------------------------------------------------------------------=== #
# Row-based rms_norm_rope (the IR litmus test): multiple inputs of
# differing dtypes (primary input; cos/sin side-loads in cos_sin_dtype; gamma),
# one ReduceSum-of-squares phase, then a map that loads the rotate-half partner
# (a second axis position) — the IR's `imap.concat.select` / `imap.slice.index`.
# The primary input is cached and reused by the reduce + the map's normed term;
# the partner is an additional load (inherent to rope, not a re-read).
# ===----------------------------------------------------------------------=== #


def rms_norm_rope[
    input_dtype: DType,
    output_dtype: DType,
    cos_sin_dtype: DType,
    rank: Int,
    InputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int, alignment: Int](Coord) -> SIMD[input_dtype, width]),
    CosFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int, alignment: Int](Coord) -> SIMD[cos_sin_dtype, width]),
    SinFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int, alignment: Int](Coord) -> SIMD[cos_sin_dtype, width]),
    OutputFn: ImplicitlyCopyable
    & RegisterPassable
    & (
        def[
            width: SIMDLength, alignment: Int
        ](Coord, SIMD[output_dtype, width]) -> None
    ),
    AxisSizeT: CoordLike,
    /,
    multiply_before_cast: Bool = True,
    reduce_dim: Int = rank - 1,
](
    input_fn: InputFn,
    cos_fn: CosFn,
    sin_fn: SinFn,
    output_fn: OutputFn,
    shape: Coord,
    axis_size: AxisSizeT,
    gamma: TileTensor[mut=False, input_dtype, ...],
    epsilon: Scalar[input_dtype],
    weight_offset: Scalar[input_dtype],
    context: DeviceContext.Optional = DeviceContext.NoneValue,
) raises:
    comptime accum = get_accum_type[input_dtype]()
    comptime assert accum.is_floating_point(), "rope requires fp accum"
    comptime assert shape.rank == rank, "shape.rank must be the same as rank"
    comptime assert shape.is_flat, "shape must be flat"

    comptime if AxisSizeT.is_static_value:
        comptime assert (
            AxisSizeT.static_value % 2 == 0
        ), "rope requires an even row width"
    comptime simd_width_cand = rowwise.pick_simd_width[
        ReduceSum[accum, 1],
        context.T.target,
        64,
        input_dtype,
        accum,
    ]()
    # Rotate-half needs each SIMD tile to sit wholly in one half and align to
    # the partner offset, i.e. `half` must be a multiple of the SIMD width. That
    # only holds when the row width is statically known and half-divisible;
    # otherwise (dynamic / non-divisible cols) fall back to scalar.
    comptime simd_width = simd_width_cand if (
        AxisSizeT.is_static_value
        and (AxisSizeT.static_value // 2) % simd_width_cand == 0
    ) else 1
    var axis_size_int = Int(axis_size)
    if axis_size_int % 2 != 0:
        raise Error("rope requires an even row width")
    # `half` is the rotate-half split point. `axis_size_int` is the (possibly
    # dynamic) row width, so this holds whether or not `AxisSizeT` is static.
    var half = axis_size_int // 2
    var axis_size_accum = Scalar[accum](axis_size_int)

    @inline(.always)
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut ctx_p: rowwise.Context[params]) {
        var axis_size,
        var half,
        var gamma,
        var epsilon,
        var weight_offset,
        var axis_size_accum,
        var input_fn,
        var cos_fn,
        var sin_fn,
        var output_fn,
    }:
        comptime row_rank = row_coords.rank

        # Load: fuses the caller's input closure into the row's primary load.
        @inline(.always)
        def load[
            width: Int, alignment: Int
        ](idx: DynamicCoord[DType.int, row_rank]) {var input_fn} -> SIMD[
            input_dtype, width
        ]:
            return input_fn[width, alignment](idx)

        var row = rowwise.Row[
            params, accum, input_dtype, reduce_dim, row_rank, is_cached=True
        ](row_coords, axis_size, ctx_p, load)

        # Reduce: sum of squares -> inv_rms.
        @inline(.always)
        def square[
            width: Int
        ](
            tile: SIMD[input_dtype, width],
            idx: DynamicCoord[DType.int, row_rank],
        ) {} -> SIMD[accum, width]:
            var tile_accum = tile.cast[accum]()
            return tile_accum * tile_accum

        # `inv_rms` held W-wide (one value broadcast to every lane, or one per
        # output column, depending on how the launch maps threads to
        # outputs); read via `.slice[width]` uniformly either way.
        var ssq = row.reduce[ReduceSum[accum, params.simd_width]](
            square, load
        ).acc
        var inv_rms = rsqrt(ssq / axis_size_accum + epsilon.cast[accum]())
        var woff = weight_offset.cast[accum]()

        # gamma indexed by the reduced axis: inner stride 1 when that axis is
        # inner (consecutive), else 0 (splat). Strided load, compile-time
        # stride, no branching.
        comptime g_stride = 1 if reduce_dim == rank - 1 else 0

        # Cache: normalize, scale by gamma, and round to `output_dtype` before
        # RoPE to match the unfused (separate rms_norm + rope) rounding, then
        # stage the result into shared memory so the rotate-half partner —
        # which lives on a *different* participant — is read from shmem
        # instead of re-loaded from global and re-normalized.
        # `multiply_before_cast` selects accum-multiply (True) vs
        # output-dtype-multiply (False), matching legacy.
        @inline(.always)
        def normed_tile[
            width: Int
        ](
            tile: SIMD[input_dtype, width],
            idx: DynamicCoord[DType.int, row_rank],
        ) {
            var gamma,
            var inv_rms,
            var woff,
            var weight_offset,
        } -> SIMD[
            output_dtype, width
        ]:
            var col = Int(idx[reduce_dim])
            var g_raw = strided_load[
                width, g_stride, alignment=align_of[SIMD[input_dtype, width]]()
            ](gamma.ptr_at_offset(Coord(col)))
            var scaled = tile.cast[accum]() * inv_rms.slice[width]()

            comptime if multiply_before_cast:
                var gamma_with_offset = g_raw.cast[accum]() + woff
                return (scaled * gamma_with_offset).cast[output_dtype]()
            else:
                var gamma_with_offset = (g_raw + weight_offset).cast[
                    output_dtype
                ]()
                return scaled.cast[output_dtype]() * gamma_with_offset

        var normed = row.cache[output_dtype, shared=True](normed_tile)

        # Emit: reads other columns of the staged row via `normed.load`
        # (the rotate-half partner). `load`'s fallback path needs the
        # primary loader + the producer that built `normed` — `RowCache`
        # doesn't carry them as fields, so `write` captures them too and
        # re-supplies them to `normed.load`.
        @inline(.always)
        def write[
            width: Int
        ](
            nc: SIMD[output_dtype, width],
            idx: DynamicCoord[DType.int, row_rank],
        ) {
            var normed,
            var load,
            var normed_tile,
            var cos_fn,
            var sin_fn,
            var half,
            var output_fn,
            var ctx_p,
        }:
            comptime alignment = ctx_p.element_alignment[input_dtype, width]()
            var col = Int(idx[reduce_dim])
            var normed_c = nc.cast[accum]()

            # Rotate-half partner: concat(-x2, x1). A W-aligned tile sits
            # wholly in one half, so the sign is uniform. The partner's normed
            # value is read straight from the staged shmem row.
            var partner: DynamicCoord[DType.int, row_rank]
            var sign: Scalar[accum]
            if col < half:
                partner = at_axis[reduce_dim](idx, col + half)
                sign = Scalar[accum](-1)
            else:
                partner = at_axis[reduce_dim](idx, col - half)
                sign = Scalar[accum](1)
            var rotated = (
                sign
                * normed.load[width](partner, load, normed_tile).cast[accum]()
            )

            var cos_c = cos_fn[width, alignment](idx).cast[accum]()
            var sin_c = sin_fn[width, alignment](idx).cast[accum]()
            var out = normed_c * cos_c + rotated * sin_c
            comptime output_alignment = ctx_p.element_alignment[
                output_dtype, width
            ]()
            var result = out.cast[output_dtype]()
            output_fn[width, output_alignment](
                idx,
                result,
            )

        row.elementwise(normed, write, load, normed_tile)

    rowwise.launch[
        axis=reduce_dim,
        simd_width=simd_width,
        num_phases=2,
        computationally_expensive=True,
    ](body, shape, context)


# ===----------------------------------------------------------------------=== #
# Row-based layer_norm: mean + variance via Welford (or a two-pass
# sum), then a normalizing map scaled by gamma and shifted by beta.
# ===----------------------------------------------------------------------=== #


def layer_norm[
    dtype: DType,
    rank: Int,
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
    /,
    reduce_dim: Int = rank - 1,
](
    input_fn: InputFn,
    output_fn: OutputFn,
    shape: Coord,
    axis_size: AxisSizeT,
    gamma: TileTensor[mut=False, dtype, ...],
    beta: TileTensor[mut=False, dtype, ...],
    epsilon: Scalar[dtype],
    context: DeviceContext.Optional = DeviceContext.NoneValue,
) raises:
    comptime accum = get_accum_type[dtype]()
    comptime assert accum.is_floating_point(), "layer_norm requires fp accum"
    # Welford's triple-field state is 12B at W=1, so the default 64B
    # per-thread budget halves the SIMD width to 4 for a bf16/fp16 input
    # (natural width 8), giving layer_norm 2x the tiles/ALU of rms_norm
    # (4B `ReduceSum` state keeps W=8). Bump the budget to 96B here so
    # Welford keeps W=8 (12B * 8 = 96); scoped to this call site rather
    # than raising the global default. fp32/fp64 inputs are unaffected
    # (natural width already <= 4).
    comptime simd_width = rowwise.pick_simd_width[
        Welford[accum, 1],
        context.T.target,
        96,
        dtype,
        accum,
    ]()

    @inline(.always)
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut ctx_p: rowwise.Context[params]) {
        var axis_size,
        var gamma,
        var beta,
        var epsilon,
        var input_fn,
        var output_fn,
    }:
        comptime row_rank = row_coords.rank

        # Load: fuses the caller's input closure into the row's primary load.
        @inline(.always)
        def load[
            width: Int, alignment: Int
        ](idx: DynamicCoord[DType.int, row_rank]) {var input_fn} -> SIMD[
            dtype, width
        ]:
            return input_fn[width, alignment](idx)

        var row = rowwise.Row[
            params, accum, dtype, reduce_dim, row_rank, is_cached=True
        ](row_coords, axis_size, ctx_p, load)

        # Reduce: mean and variance in one Welford pass.
        @inline(.always)
        def cast_to_accum[
            width: Int
        ](
            tile: SIMD[dtype, width], idx: DynamicCoord[DType.int, row_rank]
        ) {} -> SIMD[accum, width]:
            return tile.cast[accum]()

        # Stats held W-wide (one value broadcast to every lane, or one per
        # output column, depending on how the launch maps threads to
        # outputs); read via `.slice[width]` uniformly either way.
        # `cast_to_accum` crosses into `reduce` as a value arg (value-closure
        # inner-callback form).
        var stats = row.reduce[Welford[accum, params.simd_width]](
            cast_to_accum, load
        )
        var mean = stats.mean
        var inv = rsqrt(stats.M2 / stats.count + epsilon.cast[accum]())

        # gamma/beta indexed by the reduced axis: inner stride 1 when that axis
        # is inner (consecutive), else 0 (splat). Strided load, compile-time
        # stride, no branching.
        comptime g_stride = 1 if reduce_dim == rank - 1 else 0

        # Emit: per-element normalize, scale by gamma, shift by beta, store.
        @inline(.always)
        def write[
            width: Int
        ](tile: SIMD[dtype, width], idx: DynamicCoord[DType.int, row_rank]) {
            var mean,
            var inv,
            var gamma,
            var beta,
            var output_fn,
            var ctx_p,
        }:
            comptime alignment = ctx_p.element_alignment[dtype, width]()
            var col = Int(idx[reduce_dim])
            var gamma_val = strided_load[width, g_stride, alignment=alignment](
                gamma.ptr_at_offset(Coord(col))
            ).cast[accum]()
            var beta_val = strided_load[width, g_stride, alignment=alignment](
                beta.ptr_at_offset(Coord(col))
            ).cast[accum]()
            var out = (tile.cast[accum]() - mean.slice[width]()) * inv.slice[
                width
            ]() * gamma_val + beta_val
            var result = out.cast[dtype]()
            output_fn[width, alignment](idx, result)

        # `write` crosses into `elementwise` as a value arg; `mean`/`inv`/
        # `gamma`/`beta`/`output_fn` ride the value via its capture list.
        row.elementwise(write, load)

    rowwise.launch[
        axis=reduce_dim,
        simd_width=simd_width,
        num_phases=2,
        computationally_expensive=True,
    ](body, shape, context)


# ===----------------------------------------------------------------------=== #
# Row-based layer_norm_rope_ragged: mean + variance via Welford (layer_norm),
# then a map scaled by gamma/beta that additionally applies ragged-position
# RoPE (the same absolute-position lookup as rope_ragged, via the interleaved
# complex multiply from rope.mojo's `_rope`) to the row's leading `rope_dim`
# columns; the remaining columns pass through unrotated. `rope_dim` comes
# from `freqs_cis`'s own width, matching rope_ragged's own convention.
# ===----------------------------------------------------------------------=== #


def layer_norm_rope_ragged[
    input_dtype: DType,
    output_dtype: DType,
    freq_dtype: DType,
    rank: Int,
    InputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int, alignment: Int](Coord) -> SIMD[input_dtype, width]),
    OutputFn: ImplicitlyCopyable
    & RegisterPassable
    & (
        def[
            width: SIMDLength, alignment: Int
        ](Coord, SIMD[output_dtype, width]) -> None
    ),
    AxisSizeT: CoordLike,
    /,
    interleaved: Bool,
    reduce_dim: Int = rank - 1,
](
    input_fn: InputFn,
    output_fn: OutputFn,
    shape: Coord,
    axis_size: AxisSizeT,
    gamma: TileTensor[mut=False, input_dtype, ...],
    beta: TileTensor[mut=False, input_dtype, ...],
    epsilon: Scalar[input_dtype],
    input_row_offsets: TileTensor[.uint32, ...],
    start_pos: TileTensor[.uint32, ...],
    freqs_cis: TileTensor[freq_dtype, ...],
    context: DeviceContext.Optional = DeviceContext.NoneValue,
) raises:
    # TODO(feras): non-interleaved RoPE needs the far-partner row-cache
    # lookup rms_norm_rope uses for its rotate-half, not yet implemented here.
    comptime assert (
        interleaved
    ), "layer_norm_rope_ragged: non-interleaved RoPE is not implemented"
    comptime accum = get_accum_type[input_dtype]()
    comptime assert accum.is_floating_point(), "layer_norm requires fp accum"
    comptime assert freqs_cis.LayoutType._shape_types[
        1
    ].is_static_value, "Need static rope_dim for freqs_cis"
    comptime rope_dim = Int(freqs_cis.static_shape[1])
    comptime simd_width_cand = rowwise.pick_simd_width[
        Welford[accum, 1],
        context.T.target,
        96,
        input_dtype,
        accum,
    ]()
    # A SIMD tile must sit wholly within the roped region or wholly within
    # the passthrough region -- never straddle the boundary -- so `rope_dim`
    # must be simd-width-divisible; otherwise fall back to scalar.
    comptime simd_width = simd_width_cand if (
        rope_dim % simd_width_cand == 0
    ) else 1

    @inline(.always)
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut ctx_p: rowwise.Context[params]) {
        var axis_size,
        var gamma,
        var beta,
        var epsilon,
        var input_fn,
        var output_fn,
        var input_row_offsets,
        var start_pos,
        var freqs_cis,
    }:
        comptime row_rank = row_coords.rank

        @inline(.always)
        def load[
            width: Int, alignment: Int
        ](idx: DynamicCoord[DType.int, row_rank]) {var input_fn} -> SIMD[
            input_dtype, width
        ]:
            return input_fn[width, alignment](idx)

        var row = rowwise.Row[
            params, accum, input_dtype, reduce_dim, row_rank, is_cached=True
        ](row_coords, axis_size, ctx_p, load)

        @inline(.always)
        def cast_to_accum[
            width: Int
        ](
            tile: SIMD[input_dtype, width],
            idx: DynamicCoord[DType.int, row_rank],
        ) {} -> SIMD[accum, width]:
            return tile.cast[accum]()

        var stats = row.reduce[Welford[accum, params.simd_width]](
            cast_to_accum, load
        )
        var mean = stats.mean
        var inv = rsqrt(stats.M2 / stats.count + epsilon.cast[accum]())

        comptime g_stride = 1 if reduce_dim == rank - 1 else 0

        # Ragged absolute position, resolved once per row (shared by every
        # column) -- same lookup rope_ragged's own kernel does per element.
        var global_token_idx = Int(row_coords[0])
        var batch_idx = get_batch_from_row_offsets(
            input_row_offsets, global_token_idx
        )
        var token_idx = Int(
            UInt32(global_token_idx) - input_row_offsets[batch_idx]
        )
        var position_idx = Int(start_pos[batch_idx]) + token_idx

        # Cache: normalize and scale/shift by gamma/beta, staged into shared
        # memory so `write` can be a pure per-tile map (no partner lookups
        # needed -- interleaved RoPE pairs adjacent lanes within one tile).
        @inline(.always)
        def normalize[
            width: Int
        ](
            tile: SIMD[input_dtype, width],
            idx: DynamicCoord[DType.int, row_rank],
        ) {
            var gamma,
            var beta,
            var mean,
            var inv,
        } -> SIMD[
            output_dtype, width
        ]:
            comptime alignment = align_of[SIMD[input_dtype, width]]()
            var col = Int(idx[reduce_dim])
            var gamma_val = strided_load[width, g_stride, alignment=alignment](
                gamma.ptr_at_offset(Coord(col))
            ).cast[accum]()
            var beta_val = strided_load[width, g_stride, alignment=alignment](
                beta.ptr_at_offset(Coord(col))
            ).cast[accum]()
            var out = (tile.cast[accum]() - mean.slice[width]()) * inv.slice[
                width
            ]() * gamma_val + beta_val
            return out.cast[output_dtype]()

        var normed = row.cache[output_dtype, shared=True](normalize)

        # Emit: passthrough for columns >= rope_dim; interleaved complex-
        # multiply RoPE (rope.mojo's `_rope`) for columns < rope_dim. The
        # block/warp-tier reduce can still dispatch a scalar (width=1) tail
        # tile even when `simd_width` divides `rope_dim`, since the tail is
        # governed by the row's total length, not `rope_dim` alone -- that
        # tile holds one column, not a full (re, im) pair, so `_rope`'s
        # in-register deinterleave doesn't apply. Read the missing partner
        # straight from the staged row cache instead (mirrors
        # row_rms_norm_rope's rotate-half partner load).
        @inline(.always)
        def write[
            width: Int
        ](
            nc: SIMD[output_dtype, width],
            idx: DynamicCoord[DType.int, row_rank],
        ) {
            var normed,
            var load,
            var normalize,
            var freqs_cis,
            var position_idx,
            var output_fn,
            var ctx_p,
        }:
            comptime alignment = ctx_p.element_alignment[output_dtype, width]()
            var col = Int(idx[reduce_dim])
            var result: SIMD[output_dtype, width]
            if col < rope_dim:
                comptime if width == 1:
                    var pair_base = (col // 2) * 2
                    var im_c = pair_base + 1
                    var is_re = col == pair_base
                    var partner_idx = at_axis[reduce_dim](
                        idx, im_c if is_re else pair_base
                    )
                    var partner = normed.load[1](
                        partner_idx,
                        load,
                        normalize,
                    ).cast[freq_dtype]()
                    var self_val = nc.cast[freq_dtype]()
                    var x_re = self_val if is_re else partner
                    var x_im = partner if is_re else self_val
                    var f_re = rebind[Scalar[freq_dtype]](
                        freqs_cis.load[alignment=1](
                            (
                                Scalar[freqs_cis.linear_idx_type](position_idx),
                                Scalar[freqs_cis.linear_idx_type](pair_base),
                            )
                        )
                    )
                    var f_im = rebind[Scalar[freq_dtype]](
                        freqs_cis.load[alignment=1](
                            (
                                Scalar[freqs_cis.linear_idx_type](position_idx),
                                Scalar[freqs_cis.linear_idx_type](im_c),
                            )
                        )
                    )
                    var result_re = x_re * f_re - x_im * f_im
                    var result_im = x_re * f_im + x_im * f_re
                    var out_val = result_re if is_re else result_im
                    result = out_val.cast[output_dtype]()
                else:
                    var freq_val = freqs_cis.load[width=width, alignment=1](
                        (
                            Scalar[freqs_cis.linear_idx_type](position_idx),
                            Scalar[freqs_cis.linear_idx_type](col),
                        )
                    )
                    result = _rope(nc, freq_val)
            else:
                result = nc
            output_fn[width, alignment](
                idx,
                result,
            )

        row.elementwise(normed, write, load, normalize)

    rowwise.launch[
        axis=reduce_dim,
        simd_width=simd_width,
        num_phases=2,
        computationally_expensive=True,
    ](body, shape, context)


# ===----------------------------------------------------------------------=== #
# Row-based row_mean_of_squares: the rms_norm reduction core on its own.
# One ReduceSum-of-squares phase, then a scalar `emit` of `sum(x^2) / axis_size`
# (a true reduction — the reduced axis collapses to one value per row). Splits
# input and output dtypes; the square and sum run in the input's accum type
# (fp32 for bf16/fp16). Drop-in signature for the legacy `row_mean_of_squares`
# it replaces (2-arg `input_fn`, `(row, value)` `output_fn`); `reduce_dim`
# defaults to the last axis (what the op needs) but any axis is supported.
# ===----------------------------------------------------------------------=== #


def row_mean_of_squares[
    in_dtype: DType,
    out_dtype: DType,
    rank: Int,
    InputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int, alignment: Int](Coord) -> SIMD[in_dtype, width]),
    OutputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: SIMDLength](Coord, SIMD[out_dtype, width]) -> None),
    /,
    reduce_dim: Int = rank - 1,
](
    input_fn: InputFn,
    output_fn: OutputFn,
    shape: Coord,
    ctx: DeviceContext,
) raises:
    comptime assert shape.rank == rank, "shape.rank must be the same as rank"
    comptime assert shape.is_flat, "shape must be flat"
    comptime assert 0 <= reduce_dim < rank, "reduce_dim must be in [0, rank)"
    comptime accum = get_accum_type[in_dtype]()
    comptime assert (
        accum.is_floating_point()
    ), "row_mean_of_squares requires fp accum"
    comptime simd_width = rowwise.pick_simd_width[
        ReduceSum[accum, 1], ctx.target, 64, in_dtype, accum
    ]()
    var axis_size = Int(shape[reduce_dim])
    var axis_size_accum = Scalar[accum](axis_size)

    @inline(.always)
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut ctx_p: rowwise.Context[params]) {
        var axis_size,
        var axis_size_accum,
        var input_fn,
        var output_fn,
    }:
        comptime row_rank = row_coords.rank

        # Load: fuses the caller's input closure into the row's primary load.
        @inline(.always)
        def load[
            width: Int, alignment: Int
        ](idx: DynamicCoord[DType.int, row_rank]) {var input_fn} -> SIMD[
            in_dtype, width
        ]:
            return input_fn[width, alignment](idx)

        # Prepare Row: this is a true reduction (no fuse-eligible cache), so
        # the axis size is always the dynamic form.
        var row = rowwise.Row[
            params, accum, in_dtype, reduce_dim, row_rank, is_cached=False
        ](row_coords, Int(axis_size), ctx_p, load)

        # Reduce: sum of squares -> mean of squares.
        @inline(.always)
        def square[
            width: Int
        ](
            tile: SIMD[in_dtype, width],
            idx: DynamicCoord[DType.int, row_rank],
        ) {} -> SIMD[accum, width]:
            var tile_accum = tile.cast[accum]()
            return tile_accum * tile_accum

        # Mean of squares held W-wide (one value broadcast to every lane);
        # read via `.slice[width]` uniformly in the terminal.
        var mean_sq = (
            row.reduce[ReduceSum[accum, params.simd_width]](square, load).acc
            / axis_size_accum
        )

        # Emit: one value per row, at `oc` (reduced axis pinned to 0).
        @inline(.always)
        def write(
            oc: DynamicCoord[DType.int, row_rank],
        ) {var mean_sq, var output_fn}:
            output_fn[params.emit_tile_width](
                oc,
                mean_sq.slice[params.emit_tile_width]().cast[out_dtype](),
            )

        # `mean_sq`/`output_fn` ride `write`'s capture list into `emit`.
        row.emit(write)

    rowwise.launch[
        axis=reduce_dim,
        simd_width=simd_width,
        num_phases=1,
    ](body, shape, ctx)


# ===----------------------------------------------------------------------=== #
# Row-based fused Q/K row_mean_of_squares: composes the single-input
# reduction above twice (Q -> output column 0, K -> column 1). The scaffolder is
# single-primary-input, so the two reductions run as two launches rather than
# legacy's single fused launch; each is an optimal Row reduction. This
# mirrors the op's own definition — "two `row_mean_of_squares` ops plus a
# concat" — and matches the GC IR (two `iter.reduce` phases).
# ===----------------------------------------------------------------------=== #


def row_mean_of_squares_qk[
    in_dtype: DType,
    out_dtype: DType,
    //,
](
    output: TileTensor[mut=True, out_dtype, ...],
    query: TileTensor[mut=False, in_dtype, ...],
    key: TileTensor[mut=False, in_dtype, ...],
    rows: Int,
    q_cols: Int,
    k_cols: Int,
    ctx: DeviceContext,
) raises:
    if rows == 0:
        return

    @inline(.always)
    def q_in[
        width: Int, alignment: Int
    ](idx: Coord) {var query} -> SIMD[in_dtype, width]:
        return query.load[
            width=width, alignment=alignment * align_of[in_dtype]()
        ](idx)

    # Column 0 (q) / column 1 (k) sit stride-2 apart in `output`'s `[rows, 2]`
    # layout, so a `width`-wide batch of adjacent rows can't land in one
    # contiguous vector store; write it back lane by lane instead (`width`
    # is comptime, so this unrolls to `width` scalar stores).
    @inline(.always)
    def q_out[
        width: SIMDLength
    ](oc: Coord, val: SIMD[out_dtype, width]) {var output}:
        comptime for i in range(width):
            output.store((Int(oc[0]) + i, 0), val[i])

    row_mean_of_squares[in_dtype, out_dtype, 2](
        q_in, q_out, Coord(rows, q_cols), ctx
    )

    @inline(.always)
    def k_in[
        width: Int, alignment: Int
    ](idx: Coord) {var key} -> SIMD[in_dtype, width]:
        return key.load[
            width=width, alignment=alignment * align_of[in_dtype]()
        ](idx)

    @inline(.always)
    def k_out[
        width: SIMDLength
    ](oc: Coord, val: SIMD[out_dtype, width]) {var output}:
        comptime for i in range(width):
            output.store((Int(oc[0]) + i, 1), val[i])

    row_mean_of_squares[in_dtype, out_dtype, 2](
        k_in, k_out, Coord(rows, k_cols), ctx
    )


# ===----------------------------------------------------------------------=== #
# Row-based rms_norm + residual add (composite of two fused rms_norms). In
# one launch: reduce sum(input^2) -> inv_rms1; a second phase reduces
# sum(intermediate^2) where `intermediate = rms_norm(input, gamma1) + residual`
# (and emits `intermediate` as the residual output); then a map applies the
# second norm `rms_norm(intermediate, gamma2)` to the output. The intermediate
# is recomputed from the register-cached input + a residual reload rather than
# held in shared memory (legacy's single-kernel trick), so it costs one extra
# pass of memory traffic on large rows.
# ===----------------------------------------------------------------------=== #


def rms_norm_fused_residual_add[
    dtype: DType,
    rank: Int,
    Input0Fn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int, alignment: Int](Coord) -> SIMD[dtype, width]),
    Input1Fn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int, alignment: Int](Coord) -> SIMD[dtype, width]),
    Output0Fn: ImplicitlyCopyable
    & RegisterPassable
    & (
        def[
            width: SIMDLength, alignment: Int
        ](Coord, SIMD[dtype, width]) -> None
    ),
    OutputResidualFn: ImplicitlyCopyable
    & RegisterPassable
    & (
        def[
            width: SIMDLength, alignment: Int
        ](Coord, SIMD[dtype, width]) -> None
    ),
    AxisSizeT: CoordLike,
    /,
    epsilon1: Float32,
    weight_offset1: Float32,
    epsilon2: Float32,
    weight_offset2: Float32,
    multiply_before_cast: Bool = True,
    reduce_dim: Int = rank - 1,
](
    input_0_fn: Input0Fn,
    input_1_fn: Input1Fn,
    output_0_fn: Output0Fn,
    output_residual_fn: OutputResidualFn,
    shape: Coord,
    axis_size: AxisSizeT,
    gamma1: TileTensor[mut=False, dtype, ...],
    gamma2: TileTensor[mut=False, dtype, ...],
    context: DeviceContext.Optional = DeviceContext.NoneValue,
) raises:
    comptime accum = get_accum_type[dtype]()
    comptime assert accum.is_floating_point(), "rms_norm requires fp accum"
    comptime simd_width = rowwise.pick_simd_width[
        ReduceSum[accum, 1],
        context.T.target,
        64,
        dtype,
        accum,
    ]()
    var axis_size_int = Int(axis_size)
    var axis_size_accum = Scalar[accum](axis_size_int)

    @inline(.always)
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut ctx_p: rowwise.Context[params]) {
        var axis_size,
        var gamma1,
        var gamma2,
        var axis_size_accum,
        var input_0_fn,
        var input_1_fn,
        var output_0_fn,
        var output_residual_fn,
    }:
        comptime row_rank = row_coords.rank

        # Load: fuses the caller's input closure into the row's primary load.
        @inline(.always)
        def load[
            width: Int, alignment: Int
        ](idx: DynamicCoord[DType.int, row_rank]) {var input_0_fn} -> SIMD[
            dtype, width
        ]:
            return input_0_fn[width, alignment](idx)

        var row = rowwise.Row[
            params, accum, dtype, reduce_dim, row_rank, is_cached=True
        ](row_coords, axis_size, ctx_p, load)

        # gamma inner (lane) stride: 1 when the reduced axis is inner
        # (consecutive), else 0 (one gamma, splatted). Strided load, no branch.
        comptime g_stride = 1 if reduce_dim == rank - 1 else 0

        @inline(.always)
        def square[
            width: Int
        ](
            tile: SIMD[dtype, width], idx: DynamicCoord[DType.int, row_rank]
        ) {} -> SIMD[accum, width]:
            var tile_accum = tile.cast[accum]()
            return tile_accum * tile_accum

        # Reduce (phase 1): sum of squares of the input -> inv_rms1.
        var ssq1 = row.reduce[ReduceSum[accum, params.simd_width]](
            square, load
        ).acc
        var inv_rms1 = rsqrt(
            ssq1 / axis_size_accum + comptime (epsilon1.cast[accum]())
        )

        # Cache: intermediate = rms_norm(input, gamma1) + residual, in `dtype`
        # (matching legacy: the first norm is cast to dtype before the
        # residual add). Emits the residual output on the way through, and is
        # cached once so phase 2 and the terminal read it back from registers
        # instead of recomputing it (and re-loading the residual) twice.
        @inline(.always)
        def intermediate[
            width: Int
        ](tile: SIMD[dtype, width], idx: DynamicCoord[DType.int, row_rank]) {
            var gamma1,
            var inv_rms1,
            var input_1_fn,
            var output_residual_fn,
            var ctx_p,
        } -> SIMD[dtype, width]:
            comptime alignment = ctx_p.element_alignment[dtype, width]()
            var col = Int(idx[reduce_dim])
            var gamma1_val = strided_load[width, g_stride, alignment=alignment](
                gamma1.ptr_at_offset(Coord(col))
            )
            var normed = tile.cast[accum]() * inv_rms1.slice[width]()
            var residual = input_1_fn[width, alignment](idx)

            var inter: SIMD[dtype, width]
            comptime if multiply_before_cast:
                var gamma1_with_offset = gamma1_val.cast[accum]() + comptime (
                    weight_offset1.cast[accum]()
                )
                inter = (normed * gamma1_with_offset).cast[dtype]() + residual
            else:
                var gamma1_with_offset = gamma1_val + comptime (
                    weight_offset1.cast[dtype]()
                )
                inter = normed.cast[dtype]() * gamma1_with_offset + residual
            output_residual_fn[width, alignment](
                idx,
                inter,
            )
            return inter

        var inter = row.cache[dtype](intermediate)

        @inline(.always)
        def square_intermediate[
            width: Int
        ](
            staged_tile: SIMD[dtype, width],
            idx: DynamicCoord[DType.int, row_rank],
        ) {} -> SIMD[accum, width]:
            var staged_accum = staged_tile.cast[accum]()
            return staged_accum * staged_accum

        # Reduce (phase 2): sum of squares of the (staged) intermediate.
        # `square_intermediate` crosses into the cached `reduce` overload as a
        # trailing value arg. `load`/`intermediate` (the producer that built
        # `inter`) ride along for the non-fuse recompute fallback.
        var ssq2 = row.reduce[ReduceSum[accum, params.simd_width]](
            inter, square_intermediate, load, intermediate
        ).acc
        var inv_rms2 = rsqrt(
            ssq2 / axis_size_accum + comptime (epsilon2.cast[accum]())
        )

        # Emit: apply the second norm to the (staged) intermediate.
        @inline(.always)
        def write[
            width: Int
        ](
            staged_tile: SIMD[dtype, width],
            idx: DynamicCoord[DType.int, row_rank],
        ) {
            var inv_rms2,
            var gamma2,
            var output_0_fn,
            var ctx_p,
        }:
            comptime alignment = ctx_p.element_alignment[dtype, width]()
            var col = Int(idx[reduce_dim])
            var gamma2_val = strided_load[width, g_stride, alignment=alignment](
                gamma2.ptr_at_offset(Coord(col))
            )
            var normed = staged_tile.cast[accum]() * inv_rms2.slice[width]()

            comptime if multiply_before_cast:
                var gamma2_with_offset = gamma2_val.cast[accum]() + comptime (
                    weight_offset2.cast[accum]()
                )
                var result = (normed * gamma2_with_offset).cast[dtype]()
                output_0_fn[width, alignment](
                    idx,
                    result,
                )
            else:
                var gamma2_with_offset = gamma2_val + comptime (
                    weight_offset2.cast[dtype]()
                )
                var result = normed.cast[dtype]() * gamma2_with_offset
                output_0_fn[width, alignment](
                    idx,
                    result,
                )

        # `write` crosses into the cached `elementwise` overload as a
        # trailing value arg; `inv_rms2`/`gamma2`/
        # `output_0_fn` ride its captures. `load`/`intermediate` ride along
        # for the non-fuse recompute fallback.
        row.elementwise(inter, write, load, intermediate)
        # `inv_rms1` read only inside the comptime `intermediate` cache
        # closure; discard-read marks it used.
        _ = inv_rms1

    rowwise.launch[
        axis=reduce_dim,
        simd_width=simd_width,
        num_phases=2,
        computationally_expensive=True,
    ](body, shape, context)


# ===----------------------------------------------------------------------=== #
# Row-based rms_norm + dynamic-scaled FP8 quantization: sum-of-squares and
# max(|gamma*x|) reduced in one cached pass, then a per-row FP8 scale is emitted
# and the normalized row is quantized to FP8.
# ===----------------------------------------------------------------------=== #


def rms_norm_fused_quantize_dynamic_scaled_fp8[
    in_dtype: DType,
    out_dtype: DType,
    scale_dtype: DType,
    rank: Int,
    InputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int, alignment: Int](Coord) -> SIMD[in_dtype, width]),
    OutputFn: ImplicitlyCopyable
    & RegisterPassable
    & (
        def[
            width: SIMDLength, alignment: Int
        ](Coord, SIMD[out_dtype, width]) -> None
    ),
    ScaleFn: ImplicitlyCopyable
    & RegisterPassable
    & (def(Coord, Scalar[scale_dtype]) -> None),
    AxisSizeT: CoordLike,
    /,
    reduce_dim: Int = rank - 1,
](
    input_fn: InputFn,
    output_fn: OutputFn,
    scale_fn: ScaleFn,
    shape: Coord,
    axis_size: AxisSizeT,
    gamma: TileTensor[mut=False, in_dtype, ...],
    epsilon: Scalar[in_dtype],
    weight_offset: Scalar[in_dtype],
    scale_ub: Float32,
    context: DeviceContext.Optional = DeviceContext.NoneValue,
) raises:
    comptime accum = get_accum_type[in_dtype]()
    comptime assert accum.is_floating_point(), "rms_norm requires fp accum"
    comptime assert out_dtype in (
        DType.float8_e4m3fn,
        DType.float8_e4m3fnuz,
    ), "output dtype should be float8_e4m3fn or float8_e4m3fnuz"
    comptime simd_width = rowwise.pick_simd_width[
        ReduceSum[accum, 1],
        context.T.target,
        64,
        in_dtype,
        accum,
    ]()
    var axis_size_accum = Scalar[accum](Int(axis_size))

    @inline(.always)
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut ctx_p: rowwise.Context[params]) {
        var axis_size,
        var gamma,
        var epsilon,
        var weight_offset,
        var scale_ub,
        var axis_size_accum,
        var input_fn,
        var output_fn,
        var scale_fn,
    }:
        comptime row_rank = row_coords.rank

        # Load: fuses the caller's input closure into the row's primary load.
        @inline(.always)
        def load[
            width: Int, alignment: Int
        ](idx: DynamicCoord[DType.int, row_rank]) {var input_fn} -> SIMD[
            in_dtype, width
        ]:
            return input_fn[width, alignment](idx)

        var row = rowwise.Row[
            params, accum, in_dtype, reduce_dim, row_rank, is_cached=True
        ](row_coords, axis_size, ctx_p, load)

        # gamma indexed by the reduced axis: inner stride 1 when that axis is
        # inner (consecutive), else 0 (splat).
        comptime g_stride = 1 if reduce_dim == rank - 1 else 0

        # Plain non-capturing helper: receives the gamma tensor + weight offset
        # as arguments (captures no runtime value), so the value-closure
        # callbacks below can call it by name while capturing only plain values.
        @inline(.always)
        def gamma_load[
            width: Int
        ](
            gamma_tensor: TileTensor[mut=False, in_dtype, ...],
            col: Int,
            weight_offset: Scalar[in_dtype],
        ) -> SIMD[accum, width]:
            var gamma_raw = strided_load[
                width, g_stride, alignment=align_of[SIMD[in_dtype, width]]()
            ](gamma_tensor.ptr_at_offset(Coord(col)))
            return gamma_raw.cast[accum]() + weight_offset.cast[accum]()

        @inline(.always)
        def square[
            width: Int
        ](
            tile: SIMD[in_dtype, width],
            idx: DynamicCoord[DType.int, row_rank],
        ) {} -> SIMD[accum, width]:
            var tile_accum = tile.cast[accum]()
            return tile_accum * tile_accum

        # Reduce (phase 1): mean of squares of the input -> inv_rms.
        var ssq = row.reduce[ReduceSum[accum, params.simd_width]](
            square, load
        ).acc
        var inv_rms = rsqrt(ssq / axis_size_accum + epsilon.cast[accum]())

        # Reduce (phase 2): max(|gamma*x|) over the same cached row. The norm
        # factor is a positive scalar, so max(|gamma*x*inv_rms|) ==
        # max(|gamma*x|)*inv_rms; the FP8 scale is derived from the raw input
        # without staging the normalized row.
        @inline(.always)
        def abs_gamma_x[
            width: Int
        ](
            tile: SIMD[in_dtype, width],
            idx: DynamicCoord[DType.int, row_rank],
        ) {var gamma, var weight_offset} -> SIMD[accum, width]:
            return abs(
                tile.cast[accum]()
                * gamma_load[width](gamma, Int(idx[reduce_dim]), weight_offset)
            )

        # `gamma`/`weight_offset` ride `abs_gamma_x`'s captures into `reduce`;
        # `gamma_load` is called by name (captures nothing).
        var row_abs_max = row.reduce[ReduceMax[accum, params.simd_width]](
            abs_gamma_x, load
        ).acc
        var row_max = (row_abs_max * inv_rms).reduce_max()
        var scale_result = compute_dynamic_fp8_scale[out_dtype](
            row_max, scale_ub.cast[scale_dtype]()
        )
        var scale_factor = scale_result[0]
        var scale_recip = scale_result[1]

        # Emit (scale): per-row output, the dynamic FP8 scale (reduced axis
        # pinned to 0).
        @inline(.always)
        def write_scale(
            oc: DynamicCoord[DType.int, row_rank],
        ) {var scale_factor, var scale_fn}:
            scale_fn(
                oc,
                scale_factor,
            )

        # `scale_factor`/`scale_fn` ride `write_scale`'s captures into `emit`.
        row.emit(write_scale)

        # Emit (per-element): normalize, scale by gamma, quantize to FP8.
        @inline(.always)
        def write[
            width: Int
        ](
            tile: SIMD[in_dtype, width],
            idx: DynamicCoord[DType.int, row_rank],
        ) {
            var gamma,
            var weight_offset,
            var inv_rms,
            var scale_recip,
            var output_fn,
            var ctx_p,
        }:
            var normed = (
                tile.cast[accum]() * inv_rms.slice[width]()
            ) * gamma_load[width](gamma, Int(idx[reduce_dim]), weight_offset)
            var out_fp8 = fp8_quantize[out_dtype](normed, scale_recip)
            comptime alignment = ctx_p.element_alignment[out_dtype, width]()
            output_fn[width, alignment](
                idx,
                out_fp8,
            )

        # `gamma`/`weight_offset`/`inv_rms`/`scale_recip`/`output_fn` ride
        # `write`'s captures into `elementwise`.
        row.elementwise(write, load)

    rowwise.launch[
        axis=reduce_dim,
        simd_width=simd_width,
        num_phases=3,
        computationally_expensive=True,
    ](body, shape, context)


# ===----------------------------------------------------------------------=== #
# Row-based layer_norm + group-scaled (blockwise-FP8) dynamic quantization: the
# Welford statistics and the FP8 scale of every group in one launch, so a
# normalized row is never written to HBM only to be read back by the
# quantizer.
#
# Bit-exactness with the unfused pair is a property of the construction, not a
# tolerance: the statistics come from `rowwise.Row.reduce` with the same
# `Welford` monoid, the same lane width and the same launch parameters
# `layer_norm` picks, so the fp32 mean and inverse deviation are the same bits;
# the normalized value is formed in `layer_norm`'s own expression order and
# rounded to the input dtype once, exactly where `layer_norm` stores it; and
# the scale and the quantized bytes come from `compute_dynamic_fp8_scale` and
# `fp8_quantize`, the same two functions `quantize_dynamic_scaled_fp8` uses.
# The group maximum is a maximum, so the lane grouping it uses cannot change
# the result.
# ===----------------------------------------------------------------------=== #

comptime _LNQ_GROUP_LANE_WIDTH = 8
"""Columns one lane covers in the group pass.

Fixed rather than the rowwise body's `simd_width`, because the tier picks that
width from the dtype and can drop it to 1, which would leave a 128-column
scale group spanning more lanes than a warp has. Eight lanes is one 128-bit load
of bf16 input, and it tiles a 128-column group into sixteen lanes of one
warp. Sixteen lanes was measured and is worse: the group pass then runs
larger-vector accesses over the same bytes, but the pass is dominated by the
row's second read, not by the access width."""


def layer_norm_quantize_dynamic_scaled_fp8[
    dtype: DType,
    rounded: DType,
    gamma_dtype: DType,
    beta_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    rank: Int,
    InputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int, alignment: Int](Coord) -> SIMD[dtype, width]),
    Fp8Fn: ImplicitlyCopyable
    & RegisterPassable
    & (
        def[
            width: SIMDLength, alignment: Int
        ](Coord, SIMD[out_dtype, width]) -> None
    ),
    ScaleFn: ImplicitlyCopyable
    & RegisterPassable
    & (def(Coord, Scalar[scales_dtype]) -> None),
    AxisSizeT: CoordLike,
    /,
    group_size: Int,
    num_cols: Int,
    reduce_dim: Int = rank - 1,
    pdl_level: PDLLevel = PDLLevel.OVERLAP_AT_BEGINNING,
](
    input_fn: InputFn,
    fp8_fn: Fp8Fn,
    scale_fn: ScaleFn,
    shape: Coord,
    axis_size: AxisSizeT,
    gamma: TileTensor[mut=False, gamma_dtype, ...],
    beta: TileTensor[mut=False, beta_dtype, ...],
    epsilon: Scalar[dtype],
    scale_ub: Float32,
    context: DeviceContext.Optional = DeviceContext.NoneValue,
) raises:
    """Fused `layer_norm` + `quantize_dynamic_scaled_fp8` for group (blockwise)
    activation scales: one launch, no normalized round trip through HBM.

    Produces the outputs of `layer_norm` followed by
    `quantize_dynamic_scaled_fp8[group_size_or_per_token=group_size]`: `fp8_fn`
    receives `[rows, num_cols]` float8 and `scale_fn` the
    `[num_cols // group_size, rows]` float32 group scales (the blockwise
    activation layout the block-scaled matmuls take).

    The group-scale epilogue is the one
    ``Kernels/test/gpu/Moka/kernels/normq/normq_plan.mojo`` specifies, but
    this kernel's statistics half has no component: Welford's state is joined
    by `algorithm.rowwise`'s generic reducer, a Chan merge with no regime where
    its rounding is exact, so a component would have no inputs on which it
    could be verified bit-for-bit against this kernel. The RMSNorm sibling
    above carries the component.

    Parameters:
        dtype: The layer norm's working dtype; its statistics run here.
        rounded: The dtype the normalized row is rounded to before the
            quantize reads it -- the dtype `layer_norm` writes, which is the
            quantize's input dtype.
        gamma_dtype: The normalization scale weight's dtype. Cast up to the
            working dtype in the load, which is what the chain's own cast of
            the weight does and costs nothing, as a bf16-to-float32 cast is
            exact.
        beta_dtype: The normalization bias weight's dtype.
        out_dtype: The FP8 output dtype.
        scales_dtype: The group scale dtype.
        rank: Input rank; must be 2.
        InputFn: The primary-input loader.
        Fp8Fn: The quantized-value writer.
        ScaleFn: The group-scale writer.
        AxisSizeT: The reduce axis length's `CoordLike` type.
        group_size: Columns per scale.
        num_cols: The row width.
        reduce_dim: The reduced axis.
        pdl_level: The tier kernels' programmatic dependent launch level.

    Args:
        input_fn: Loads a tile of the unnormalized input.
        fp8_fn: Stores a tile of the quantized output.
        scale_fn: Stores one group's scale.
        shape: Tensor shape.
        axis_size: Length of the reduce axis.
        gamma: The `[num_cols]` layer-norm scale weight.
        beta: The `[num_cols]` layer-norm bias weight.
        epsilon: The layer norm's epsilon.
        scale_ub: Upper bound of the dynamic scale.
        context: Device context.

    Raises:
        On launch or worker failure.
    """
    comptime accum = get_accum_type[dtype]()
    comptime assert accum.is_floating_point(), "layer_norm requires fp accum"
    comptime assert rank == 2, "the fused quantize writes rank-2 block scales"
    comptime assert reduce_dim == rank - 1, "the reduced axis must be the row"
    # `layer_norm`'s own width selection, including the 96-byte budget that
    # keeps Welford's 12-byte state at the natural width. The tier picks the
    # block size and the reduction tree from this width, so matching it is what
    # makes the fused statistics the unfused ones.
    comptime simd_width = rowwise.pick_simd_width[
        Welford[accum, 1], context.T.target, 96, dtype, accum
    ]()
    comptime gw = _LNQ_GROUP_LANE_WIDTH
    comptime lanes_per_group = group_size // gw
    comptime num_groups = num_cols // group_size
    comptime assert group_size % gw == 0, "a scale group must tile whole lanes"
    comptime assert (
        lanes_per_group.is_power_of_two() and lanes_per_group <= WARP_SIZE
    ), "a scale group must be a power-of-two lane segment of one warp"
    comptime assert (
        num_cols % group_size == 0
    ), "the row must tile whole scale groups"
    var scale_ub_scaled = Scalar[scales_dtype](scale_ub)

    @inline(.always)
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut ctx_p: rowwise.Context[params]) {
        var axis_size,
        var gamma,
        var beta,
        var epsilon,
        var scale_ub_scaled,
        var input_fn,
        var fp8_fn,
        var scale_fn,
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
            params, accum, dtype, reduce_dim, row_rank, is_cached=True
        ](row_coords, axis_size, ctx_p, load)

        # The row is staged in shared memory as the reduce walks it, so the
        # group pass below normalizes it from there instead of reading it out
        # of HBM a second time. Staging from inside the reduce's contribution
        # keeps the reduction itself untouched -- the statistics are still the
        # unfused `layer_norm`'s own tree.
        #
        # Only the block tier stages, because only it owns a whole row: it runs
        # one row per block, so one strip serves it. The warp tier owns one row
        # per warp and re-reads, which is what the chain does -- and it keeps
        # that instantiation (elaborated at every width, because the tier pick
        # is a runtime `if` on the row size) from declaring one strip per warp
        # of never-launched shared memory.
        comptime staged = params._tier == ReduceTier.Block
        comptime warp_tier = params._tier == ReduceTier.Warp
        # Split-K stripes one row across blocks, so no block sees the whole row
        # and reading the strip there would fuse garbage. It is dead for a
        # normalize-shaped body -- `rowwise.launch` passes `supports_splitk=
        # False` for `num_phases > 1` -- so state the invariant it rests on.
        comptime assert params._tier != ReduceTier.Splitk, (
            "the fused norm+quantize needs one row per block; the split-K tier"
            " stripes a row across blocks"
        )
        # The group pass loads `gw` wide, so the strip has to carry that
        # promise, not the narrower one the reduce's tiles give.
        comptime stage_align = max(
            align_of[SIMD[dtype, params.simd_width]](),
            align_of[SIMD[dtype, gw]](),
        )
        # A row wider than this would fail to compile with ptxas's own message
        # about a kernel the caller never wrote; refuse with the real limit.
        comptime assert num_cols * size_of[dtype]() <= 48 * 1024, (
            "the staged row does not fit the 48 KiB of shared memory every"
            " target allows without opting in"
        )
        var x_row = stack_allocation[
            num_cols,
            Scalar[dtype],
            alignment=stage_align,
            address_space=.SHARED,
        ]()
        var x_addr = Int(x_row)

        # The previous grid-stride row's group pass reads this strip, so it has
        # to finish before this row overwrites it -- `Row.cache`'s own pre-sync.
        comptime if staged:
            barrier()

        @inline(.always)
        def cast_to_accum[
            width: Int
        ](tile: SIMD[dtype, width], idx: DynamicCoord[DType.int, row_rank]) {
            var x_addr
        } -> SIMD[accum, width]:
            comptime if staged:
                comptime al = align_of[SIMD[dtype, width]]()
                MutPointer[
                    Scalar[dtype], MutUntrackedOrigin, address_space=.SHARED
                ](unsafe_from_address=x_addr).unsafe_store[alignment=al](
                    Int(idx[reduce_dim]), tile
                )
            return tile.cast[accum]()

        var stats = row.reduce[Welford[accum, params.simd_width]](
            cast_to_accum, load
        )
        comptime if staged:
            barrier()
        # Every lane holds the row's statistics after the cross-thread join, so
        # one lane's copy splats; `layer_norm` reads the same value through
        # `.slice`, which on this tier is the state unchanged.
        var mean = SIMD[accum, gw](stats.mean[0])
        var inv = SIMD[accum, gw](
            rsqrt(stats.M2[0] / stats.count[0] + epsilon.cast[accum]())
        )
        var row_idx = Int(row_coords[0])

        # The group pass walks one scale group per lane segment of a warp, in
        # as many passes as the row needs; its input is the strip the reduce
        # just filled, or the row itself on a tier that did not stage it. Every
        # warp of a block covers the same row on the block tier, so the warps
        # split the pass between them; on the warp tier each warp owns a row of
        # its own and walks all of it.
        comptime warps = params.BLOCK_SIZE // WARP_SIZE
        comptime groups_per_warp = WARP_SIZE // lanes_per_group
        comptime groups_per_pass = (
            groups_per_warp if warp_tier else warps * groups_per_warp
        )
        comptime warp_group_stride = 0 if warp_tier else groups_per_warp
        comptime passes = ceildiv(num_groups, groups_per_pass)
        comptime gw_align = align_of[SIMD[dtype, gw]]()
        comptime gamma_align = align_of[SIMD[gamma_dtype, gw]]()
        comptime beta_align = align_of[SIMD[beta_dtype, gw]]()
        comptime fp8_align = align_of[SIMD[out_dtype, gw]]()
        var lane_in_group = Int(lane_id()) % lanes_per_group
        # A `use_simd_full=false` instantiation carries a one-wide body; the
        # group pass keeps its own width, which is what `gw` is for.
        for p in range(passes):
            var group = (
                p * groups_per_pass
                + Int(warp_id()) * warp_group_stride
                + Int(lane_id()) // lanes_per_group
            )
            var live = group < num_groups
            # The rounds past the last group recompute the last one and store
            # nothing, so every lane of the warp reaches the group maximum
            # together. A branch here would split the warp.
            var clamped = group if live else num_groups - 1
            var col = clamped * group_size + lane_in_group * gw

            var x: SIMD[accum, gw]
            comptime if staged:
                x = (
                    MutPointer[
                        Scalar[dtype], MutUntrackedOrigin, address_space=.SHARED
                    ](unsafe_from_address=x_addr)
                    .unsafe_load[width=gw, alignment=gw_align](col)
                    .cast[accum]()
                )
            else:
                # The loader's second parameter is an element alignment, not
                # the byte alignment the shared strip's load above takes; `col`
                # is a whole number of `gw` tiles into the row.
                x = input_fn[gw, gw](Coord(row_idx, col)).cast[accum]()
            var gamma_val = gamma.load[width=gw, alignment=gamma_align](
                Coord(col)
            ).cast[accum]()
            var beta_val = beta.load[width=gw, alignment=beta_align](
                Coord(col)
            ).cast[accum]()
            # `layer_norm`'s expression order, rounded to `rounded` exactly
            # where it writes the row: that rounding is the quantize's input,
            # and quantizing the unrounded float32 value moves every scale.
            var out = ((x - mean) * inv * gamma_val + beta_val).cast[rounded]()
            var values = out.cast[accum]()

            var group_max = warp.lane_group_max[num_lanes=lanes_per_group](
                abs(values).reduce_max()
            )
            var scale: Scalar[scales_dtype]
            var recip: Scalar[accum]
            scale, recip = compute_dynamic_fp8_scale[out_dtype](
                group_max, scale_ub_scaled
            )
            if live and lane_in_group == 0:
                scale_fn(Coord(group, row_idx), scale)
            if live:
                fp8_fn[gw, fp8_align](
                    Coord(row_idx, col),
                    fp8_quantize[out_dtype](values, recip),
                )

    rowwise.launch[
        axis=reduce_dim,
        simd_width=simd_width,
        num_phases=2,
        computationally_expensive=True,
        name="layer_norm_quantize",
        pdl_level=pdl_level,
    ](body, shape, context)


# ===----------------------------------------------------------------------=== #
# Row-based rms_norm + group-scaled (blockwise-FP8) dynamic quantization, with
# the normalized row kept: both outputs of the unfused pair, one launch.
#
# A norm whose reader needs the normalized activation in the input dtype as
# well as the quantized form cannot use the single-output fused op above --
# the quantize would have to be a second launch. This one writes both: on the
# block tier the reduce's contribution stages the row in shared memory and the
# group pass normalizes it from there, so the row is read out of HBM once.
#
# Bit-exactness with `rms_norm` followed by `quantize_dynamic_scaled_fp8` is a
# property of the construction: the sum of squares comes from
# `rowwise.Row.reduce` with the same `ReduceSum` monoid, the same lane width
# and the same launch parameters `rms_norm` picks, so the inverse deviation is
# the same bits; the normalized value is formed in `rms_norm`'s own expression
# order for either `multiply_before_cast`, and rounded to the input dtype once;
# and the scale and quantized bytes come from `compute_dynamic_fp8_scale` and
# `fp8_quantize`, the same two functions the standalone quantize uses.
# ===----------------------------------------------------------------------=== #

comptime _RNQ_GROUP_LANE_WIDTH = 8
"""Columns one lane covers in the group pass; see
`_LNQ_GROUP_LANE_WIDTH` for why it is fixed rather than the body's width."""


def rms_norm_quantize_dynamic_scaled_fp8_dual[
    dtype: DType,
    rounded: DType,
    gamma_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    rank: Int,
    InputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int, alignment: Int](Coord) -> SIMD[dtype, width]),
    NormedFn: ImplicitlyCopyable
    & RegisterPassable
    & (
        def[
            width: SIMDLength, alignment: Int
        ](Coord, SIMD[rounded, width]) -> None
    ),
    Fp8Fn: ImplicitlyCopyable
    & RegisterPassable
    & (
        def[
            width: SIMDLength, alignment: Int
        ](Coord, SIMD[out_dtype, width]) -> None
    ),
    ScaleFn: ImplicitlyCopyable
    & RegisterPassable
    & (def(Coord, Scalar[scales_dtype]) -> None),
    AxisSizeT: CoordLike,
    /,
    group_size: Int,
    num_cols: Int,
    weight_offset: Float32 = 0.0,
    multiply_before_cast: Bool = False,
    reduce_dim: Int = rank - 1,
    pdl_level: PDLLevel = PDLLevel.OVERLAP_AT_BEGINNING,
](
    input_fn: InputFn,
    normed_fn: NormedFn,
    fp8_fn: Fp8Fn,
    scale_fn: ScaleFn,
    shape: Coord,
    axis_size: AxisSizeT,
    gamma: TileTensor[mut=False, gamma_dtype, ...],
    epsilon: Float32,
    scale_ub: Float32,
    context: DeviceContext.Optional = DeviceContext.NoneValue,
) raises:
    """Fused `rms_norm` + `quantize_dynamic_scaled_fp8` for group (blockwise)
    activation scales, emitting the normalized row as well.

    Produces both outputs of `rms_norm` and then the outputs of
    `quantize_dynamic_scaled_fp8[group_size_or_per_token=group_size]` applied
    to it: `normed_fn` receives the `[rows, num_cols]` normalized row in
    `rounded`, `fp8_fn` the same row in `out_dtype`, and `scale_fn` the
    `[num_cols // group_size, rows]` group scales.

    The Moka component for this kernel is
    ``Kernels/test/gpu/Moka/kernels/normq/normq_plan.mojo`` (one CTA per row,
    a sum-of-squares rendezvous, then the group-scale epilogue), with its ops
    in ``kernels/normq/block_ops_normq.mojo`` and its oracle test in
    ``test/test_normq_plan.mojo``. ``ptx/ptx_normq.mojo`` dumps the plan the
    sync certificate checks.

    Parameters:
        dtype: The normalization's working dtype; its reduction runs here.
        rounded: The dtype the normalized row is written in, which is the
            quantize's input dtype.
        gamma_dtype: The scale weight's dtype, cast up in the load.
        out_dtype: The FP8 output dtype.
        scales_dtype: The group scale dtype.
        rank: Input rank; must be 2.
        InputFn: The primary-input loader.
        NormedFn: The normalized-row writer.
        Fp8Fn: The quantized-value writer.
        ScaleFn: The group-scale writer.
        AxisSizeT: The reduce axis length's `CoordLike` type.
        group_size: Columns per scale.
        num_cols: The row width.
        weight_offset: Constant added to `gamma`.
        multiply_before_cast: `rms_norm`'s gamma-multiply order.
        reduce_dim: The reduced axis.
        pdl_level: The tier kernels' programmatic dependent launch level.

    Args:
        input_fn: Loads a tile of the input.
        normed_fn: Stores a tile of the normalized row.
        fp8_fn: Stores a tile of the quantized output.
        scale_fn: Stores one group's scale.
        shape: Tensor shape.
        axis_size: Length of the reduce axis.
        gamma: The `[num_cols]` normalization weight.
        epsilon: The normalization epsilon. Float32, as `rms_norm` takes it:
            rounding it to the input dtype moves the rsqrt argument for every
            row.
        scale_ub: Upper bound of the dynamic scale.
        context: Device context.

    Raises:
        On launch or worker failure.
    """
    comptime accum = get_accum_type[dtype]()
    comptime assert accum.is_floating_point(), "rms_norm requires fp accum"
    comptime assert rank == 2, "the fused quantize writes rank-2 block scales"
    comptime assert reduce_dim == rank - 1, "the reduced axis must be the row"
    # `rms_norm`'s own width selection: the tier picks the block size and the
    # reduction tree from it, so matching it is what makes the fused sum of
    # squares the unfused one.
    comptime simd_width = rowwise.pick_simd_width[
        ReduceSum[accum, 1], context.T.target, 64, dtype, accum
    ]()
    comptime gw = _RNQ_GROUP_LANE_WIDTH
    comptime lanes_per_group = group_size // gw
    comptime num_groups = num_cols // group_size
    comptime assert group_size % gw == 0, "a scale group must tile whole lanes"
    comptime assert (
        lanes_per_group.is_power_of_two() and lanes_per_group <= WARP_SIZE
    ), "a scale group must be a power-of-two lane segment of one warp"
    comptime assert (
        num_cols % group_size == 0
    ), "the row must tile whole scale groups"
    var axis_size_accum = Scalar[accum](Int(axis_size))
    var scale_ub_scaled = Scalar[scales_dtype](scale_ub)

    @inline(.always)
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut ctx_p: rowwise.Context[params]) {
        var axis_size,
        var axis_size_accum,
        var gamma,
        var epsilon,
        var scale_ub_scaled,
        var input_fn,
        var normed_fn,
        var fp8_fn,
        var scale_fn,
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
            params, accum, dtype, reduce_dim, row_rank, is_cached=True
        ](row_coords, axis_size, ctx_p, load)

        # The row is staged in shared memory as the reduce walks it (see the
        # layer_norm body above): the group pass normalizes it from there
        # rather than reading it out of HBM again. Only the block tier stages
        # -- it owns one row per block, so one strip serves it; the warp tier
        # owns one row per warp and re-reads.
        comptime staged = params._tier == ReduceTier.Block
        comptime warp_tier = params._tier == ReduceTier.Warp
        comptime assert params._tier != ReduceTier.Splitk, (
            "the fused norm+quantize needs one row per block; the split-K tier"
            " stripes a row across blocks"
        )
        comptime stage_align = max(
            align_of[SIMD[dtype, params.simd_width]](),
            align_of[SIMD[dtype, gw]](),
        )
        comptime assert num_cols * size_of[dtype]() <= 48 * 1024, (
            "the staged row does not fit the 48 KiB of shared memory every"
            " target allows without opting in"
        )
        var x_row = stack_allocation[
            num_cols,
            Scalar[dtype],
            alignment=stage_align,
            address_space=.SHARED,
        ]()
        var x_addr = Int(x_row)

        # See the layer_norm body: the previous grid-stride row's readers must
        # finish before this row overwrites the strip.
        comptime if staged:
            barrier()

        @inline(.always)
        def square[
            width: Int
        ](tile: SIMD[dtype, width], idx: DynamicCoord[DType.int, row_rank]) {
            var x_addr
        } -> SIMD[accum, width]:
            comptime if staged:
                comptime al = align_of[SIMD[dtype, width]]()
                MutPointer[
                    Scalar[dtype], MutUntrackedOrigin, address_space=.SHARED
                ](unsafe_from_address=x_addr).unsafe_store[alignment=al](
                    Int(idx[reduce_dim]), tile
                )
            var tile_accum = tile.cast[accum]()
            return tile_accum * tile_accum

        var ssq = row.reduce[ReduceSum[accum, params.simd_width]](
            square, load
        ).acc
        comptime if staged:
            barrier()
        var inv = SIMD[accum, gw](
            rsqrt(ssq[0] / axis_size_accum + epsilon.cast[accum]())
        )
        var row_idx = Int(row_coords[0])
        comptime normalized_offset = Scalar[rounded](weight_offset)
        comptime gamma_offset = Scalar[gamma_dtype](weight_offset)

        # One pass writes both outputs and the group scales: a group is one
        # lane segment of a warp, so its maximum is a `lane_group_max` and its
        # lanes keep their normalized values in registers across it.
        comptime warps = params.BLOCK_SIZE // WARP_SIZE
        comptime groups_per_warp = WARP_SIZE // lanes_per_group
        comptime groups_per_pass = (
            groups_per_warp if warp_tier else warps * groups_per_warp
        )
        comptime warp_group_stride = 0 if warp_tier else groups_per_warp
        comptime passes = ceildiv(num_groups, groups_per_pass)
        comptime gw_align = align_of[SIMD[dtype, gw]]()
        comptime gamma_align = align_of[SIMD[gamma_dtype, gw]]()
        comptime normed_align = align_of[SIMD[rounded, gw]]()
        comptime fp8_align = align_of[SIMD[out_dtype, gw]]()
        var lane_in_group = Int(lane_id()) % lanes_per_group
        for p in range(passes):
            var group = (
                p * groups_per_pass
                + Int(warp_id()) * warp_group_stride
                + Int(lane_id()) // lanes_per_group
            )
            var live = group < num_groups
            # Rounds past the last group recompute it and store nothing, so
            # every lane of the warp reaches the maximum together.
            var clamped = group if live else num_groups - 1
            var col = clamped * group_size + lane_in_group * gw

            var x: SIMD[accum, gw]
            comptime if staged:
                x = (
                    MutPointer[
                        Scalar[dtype], MutUntrackedOrigin, address_space=.SHARED
                    ](unsafe_from_address=x_addr)
                    .unsafe_load[width=gw, alignment=gw_align](col)
                    .cast[accum]()
                )
            else:
                # The loader's second parameter is an element alignment, not
                # the byte alignment the shared strip's load above takes; `col`
                # is a whole number of `gw` tiles into the row.
                x = input_fn[gw, gw](Coord(row_idx, col)).cast[accum]()
            var gamma_val = gamma.load[width=gw, alignment=gamma_align](
                Coord(col)
            )
            # `rms_norm`'s expression for either gamma-multiply order, rounded
            # once: that rounding is the quantize's input and the caller's
            # normalized row.
            var out: SIMD[rounded, gw]
            comptime if multiply_before_cast:
                out = (
                    (x * inv)
                    * (
                        gamma_val.cast[accum]()
                        + SIMD[accum, gw](normalized_offset.cast[accum]())
                    )
                ).cast[rounded]()
            else:
                out = (x * inv).cast[rounded]() * (
                    gamma_val + SIMD[gamma_dtype, gw](gamma_offset)
                ).cast[rounded]()
            var values = out.cast[accum]()

            var group_max = warp.lane_group_max[num_lanes=lanes_per_group](
                abs(values).reduce_max()
            )
            var scale: Scalar[scales_dtype]
            var recip: Scalar[accum]
            scale, recip = compute_dynamic_fp8_scale[out_dtype](
                group_max, scale_ub_scaled
            )
            if live and lane_in_group == 0:
                scale_fn(Coord(group, row_idx), scale)
            if live:
                normed_fn[gw, normed_align](Coord(row_idx, col), out)
                fp8_fn[gw, fp8_align](
                    Coord(row_idx, col),
                    fp8_quantize[out_dtype](values, recip),
                )

    rowwise.launch[
        axis=reduce_dim,
        simd_width=simd_width,
        num_phases=2,
        computationally_expensive=True,
        name="rms_norm_quantize",
        pdl_level=pdl_level,
    ](body, shape, context)
