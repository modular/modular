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
"""An input-scaled RMS norm over the routed rows of an EP expert-output buffer.

    out = rms_norm(x * (1 + weight))          # weightless norm

The EP dispatch packs a device's routed rows from row 0, and the last
expert offset counts them. That count lives on the device, so the launch
covers the whole static buffer and every row past it returns at once; rows
past it are left unwritten. The body is `rms_norm`'s, on the same launch, so
the routed rows match `mo.composite.reduce.rms_norm` of the scaled input bit
for bit.

`norm.routed_rows_rms_norm_finalize` folds that norm into `moe.finalize`
(`nn.routed_rows_finalize`): each token sums its routed rows' norms by router
weight, and a slot whose row is past the routed count adds nothing.
"""

from std.math import rsqrt
from std.sys import size_of
from std.utils.coord import ComptimeInt, Coord, DynamicCoord
from std.utils.numerics import get_accum_type

import extensibility
from extensibility import Input, Output, Tensor, TileTensorable
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_gpu

from algorithm import rowwise
from nn.routed_rows_finalize import (
    HIDDEN,
    In,
    Out,
    RoutedRowsFinalizeArgs,
    routed_rows_finalize,
)

from .monoids import ReduceSum


@extensibility.register("norm.routed_rows_rms_norm")
struct RoutedRowsRMSNorm:
    """Registers the `norm.routed_rows_rms_norm` graph op."""

    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        weight: Some[Input & Tensor],
        expert_offsets: Some[Input & Tensor],
        eps: Float32,
        ctx: DeviceContext,
    ) raises:
        """Normalizes the rows below the last expert offset.

        Args:
            output: The normalized rows `[rows, hidden]`.
            input: The expert outputs `[rows, hidden]`.
            weight: The input scale weight `[hidden]`.
            expert_offsets: The dispatch's per-expert row offsets; the last
                one is the routed row count.
            eps: The norm's epsilon.
            ctx: Device context.

        Raises:
            Error: If the launch fails.
        """
        comptime assert (
            ctx.target.is_gpu()
        ), "norm.routed_rows_rms_norm is GPU only"
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert conforms_to(type_of(weight), TileTensorable)
        comptime assert conforms_to(type_of(expert_offsets), TileTensorable)
        comptime assert type_of(output).rank == 2
        comptime assert type_of(input).rank == 2
        comptime assert type_of(weight).rank == 1
        comptime assert type_of(expert_offsets).rank == 1
        comptime dtype = type_of(output).dtype
        comptime assert type_of(input).dtype == dtype
        comptime assert type_of(weight).dtype == dtype
        comptime assert (
            type_of(input)._shape_types[1].is_static_value
        ), "the hidden size must be static"
        comptime cols = type_of(input)._shape_types[1].static_value
        var output_tt = output.to_tile_tensor[dtype]()
        var input_tt = input.to_tile_tensor[dtype]()
        var weight_tt = weight.to_tile_tensor[dtype]()
        var offsets_tt = expert_offsets.to_tile_tensor[DType.uint32]()
        comptime input_alignment = type_of(input).alignment
        comptime weight_alignment = type_of(weight).alignment
        comptime output_alignment = type_of(output).alignment
        comptime accum = get_accum_type[dtype]()
        comptime simd_width = rowwise.pick_simd_width[
            ReduceSum[accum, 1],
            ctx.target,
            64,
            dtype,
            accum,
        ]()
        var axis_size_accum = Scalar[accum](cols)
        var last_offset = Int(expert_offsets.shape()[0].value()) - 1

        @inline(.always)
        def body[
            params: rowwise.ContextParams
        ](row_coords: Coord, mut ctx_p: rowwise.Context[params]) {
            var input_tt,
            var weight_tt,
            var offsets_tt,
            var output_tt,
            var axis_size_accum,
            var last_offset,
            var eps,
        }:
            # Uniform over the threads sharing a row, so no tier's barrier
            # is split.
            if UInt32(Int(row_coords[0])) >= offsets_tt.load[width=1](
                Coord(last_offset)
            ):
                return

            comptime row_rank = row_coords.rank

            @inline(.always)
            def load[
                width: Int, alignment: Int
            ](idx: DynamicCoord[DType.int, row_rank]) {
                var input_tt, var weight_tt
            } -> SIMD[dtype, width]:
                comptime byte_alignment = alignment * size_of[dtype]()
                return input_tt.load[
                    width=width,
                    alignment=min(input_alignment, byte_alignment),
                ](idx) * (
                    1
                    + weight_tt.load[
                        width=width,
                        alignment=min(weight_alignment, byte_alignment),
                    ](Coord(Int(idx[1])))
                )

            var row = rowwise.Row[
                params, accum, dtype, 1, row_rank, is_cached=True
            ](row_coords, ComptimeInt[cols](), ctx_p, load)

            @inline(.always)
            def square[
                width: Int
            ](
                tile: SIMD[dtype, width], idx: DynamicCoord[DType.int, row_rank]
            ) {} -> SIMD[accum, width]:
                var t = tile.cast[accum]()
                return t * t

            var ssq = row.reduce[ReduceSum[accum, params.simd_width]](
                square, load
            ).acc
            var inv_rms = rsqrt(ssq / axis_size_accum + eps.cast[accum]())

            @inline(.always)
            def write[
                width: Int
            ](
                tile: SIMD[dtype, width], idx: DynamicCoord[DType.int, row_rank]
            ) {var inv_rms, var output_tt, var ctx_p}:
                comptime alignment = ctx_p.element_alignment[dtype, width]()
                output_tt.store[
                    width=width,
                    alignment=min(
                        output_alignment, alignment * size_of[dtype]()
                    ),
                ](
                    idx,
                    (tile.cast[accum]() * inv_rms.slice[width]()).cast[dtype](),
                )

            row.elementwise(write, load)

        rowwise.launch[
            axis=1,
            simd_width=simd_width,
            num_phases=2,
            dtype_size=size_of[dtype](),
            computationally_expensive=True,
        ](body, input.shape(), ctx)


@always_inline
def _in[T: DType](t: Some[Input & Tensor]) -> In[T]:
    comptime assert conforms_to(type_of(t), TileTensorable)
    return (
        t.to_tile_tensor[T]()
        .ptr.unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmUntrackedOrigin]()
    )


@extensibility.register("norm.routed_rows_rms_norm_finalize")
struct RoutedRowsRMSNormFinalize:
    """Registers the `norm.routed_rows_rms_norm_finalize` graph op."""

    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        weight: Some[Input & Tensor],
        expert_offsets: Some[Input & Tensor],
        restore_order: Some[Input & Tensor],
        router_weight: Some[Input & Tensor],
        eps: Float32,
        ctx: DeviceContext,
    ) raises:
        """Sums each token's normalized routed rows by router weight.

        Args:
            output: One row per token `[tokens, hidden]`, float32 or bf16
                (rounded once from the float32 sum).
            input: The expert outputs in expert order `[tokens * top_k,
                hidden]`.
            weight: The norm's input scale weight `[hidden]`.
            expert_offsets: Per-group row offsets; the last one is the
                routed row count.
            restore_order: Each `token * top_k + k` slot's row in `input`.
            router_weight: Each slot's weight `[tokens, top_k]`.
            eps: The norm's epsilon.
            ctx: Device context.

        Raises:
            Error: If the launch fails.
        """
        comptime assert (
            ctx.target.is_gpu()
        ), "norm.routed_rows_rms_norm_finalize is GPU only"
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert type_of(output).rank == 2
        comptime assert type_of(input).rank == 2
        comptime assert type_of(weight).rank == 1
        comptime assert type_of(expert_offsets).rank == 1
        comptime assert type_of(restore_order).rank == 1
        comptime assert type_of(router_weight).rank == 2
        comptime out_dtype = type_of(output).dtype
        comptime assert (
            type_of(input)._shape_types[1].static_value == HIDDEN
        ), "the hidden size must be 4096"
        comptime assert out_dtype in (
            DType.float32,
            DType.bfloat16,
        ), "the output must be float32 or bf16"
        comptime assert (
            type_of(router_weight)._shape_types[1].is_static_value
        ), "top_k must be static"
        comptime top_k = type_of(router_weight)._shape_types[1].static_value
        routed_rows_finalize[top_k](
            RoutedRowsFinalizeArgs(
                _in[DType.bfloat16](input),
                _in[DType.bfloat16](weight),
                _in[DType.uint32](expert_offsets),
                _in[DType.uint32](restore_order),
                _in[DType.float32](router_weight),
                output.to_tile_tensor[out_dtype]().ptr.unsafe_origin_cast[
                    MutUntrackedOrigin
                ](),
                Int64(output.shape()[0].value()),
                Int64(expert_offsets.shape()[0].value() - 1),
                eps,
            ),
            ctx,
        )
