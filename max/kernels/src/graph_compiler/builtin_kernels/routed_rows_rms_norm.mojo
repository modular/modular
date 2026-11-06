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
from extensibility import InputTensor, OutputTensor
from layout import UNKNOWN_VALUE
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
    def execute[
        dtype: DType,
        //,
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        input: InputTensor[dtype=dtype, rank=2, ...],
        weight: InputTensor[dtype=dtype, rank=1, ...],
        expert_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        eps: Float32,
        ctx: DeviceContext,
    ) raises:
        """Normalizes the rows below the last expert offset.

        Parameters:
            dtype: Element type of the activations and the weight.

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
        comptime cols = Int(input.static_spec.shape_tuple[1])
        comptime assert cols != UNKNOWN_VALUE, "the hidden size must be static"
        comptime accum = get_accum_type[dtype]()
        comptime simd_width = rowwise.pick_simd_width[
            ReduceSum[accum, 1],
            ctx.target.device_kind_label(),
            64,
            dtype,
            accum,
        ]()
        var axis_size_accum = Scalar[accum](cols)
        var last_offset = Int(expert_offsets.dim_size[0]()) - 1

        @inline(.always)
        def body[
            params: rowwise.ContextParams
        ](row_coords: Coord, mut ctx_p: rowwise.Context[params]) {
            var input,
            var weight,
            var expert_offsets,
            var output,
            var axis_size_accum,
            var last_offset,
            var eps,
        }:
            # Uniform over the threads sharing a row, so no tier's barrier
            # is split.
            if UInt32(Int(row_coords[0])) >= expert_offsets.load[width=1](
                Coord(last_offset)
            ):
                return

            comptime row_rank = row_coords.rank

            @inline(.always)
            def load[
                width: Int, alignment: Int
            ](idx: DynamicCoord[DType.int, row_rank]) {
                var input, var weight
            } -> SIMD[dtype, width]:
                return input.load[width, element_alignment=alignment](idx) * (
                    1
                    + weight.load[width, element_alignment=alignment](
                        Coord(Int(idx[1]))
                    )
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
            ) {var inv_rms, var output, var ctx_p}:
                comptime alignment = ctx_p.element_alignment[dtype, width]()
                output.store[width, element_alignment=alignment](
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
        ](body, input.shape_coord(), ctx)


@always_inline
def _in[T: DType](t: InputTensor[dtype=T, ...]) -> In[T]:
    return (
        t.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmUntrackedOrigin]()
    )


@extensibility.register("norm.routed_rows_rms_norm_finalize")
struct RoutedRowsRMSNormFinalize:
    """Registers the `norm.routed_rows_rms_norm_finalize` graph op."""

    @staticmethod
    def execute[
        out_dtype: DType,
    ](
        output: OutputTensor[dtype=out_dtype, rank=2, ...],
        input: InputTensor[dtype=DType.bfloat16, rank=2, ...],
        weight: InputTensor[dtype=DType.bfloat16, rank=1, ...],
        expert_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        restore_order: InputTensor[dtype=DType.uint32, rank=1, ...],
        router_weight: InputTensor[dtype=DType.float32, rank=2, ...],
        eps: Float32,
        ctx: DeviceContext,
    ) raises:
        """Sums each token's normalized routed rows by router weight.

        Parameters:
            out_dtype: The output's dtype, float32 or bf16 (rounded once
                from the float32 sum).

        Args:
            output: One row per token `[tokens, hidden]`.
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
        comptime assert (
            Int(input.static_spec.shape_tuple[1]) == HIDDEN
        ), "the hidden size must be 4096"
        comptime assert out_dtype in (
            DType.float32,
            DType.bfloat16,
        ), "the output must be float32 or bf16"
        comptime top_k = Int(router_weight.static_spec.shape_tuple[1])
        comptime assert top_k != UNKNOWN_VALUE, "top_k must be static"
        routed_rows_finalize[top_k](
            RoutedRowsFinalizeArgs(
                _in(input),
                _in(weight),
                _in(expert_offsets),
                _in(restore_order),
                _in(router_weight),
                output.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
                Int64(output.dim_size[0]()),
                Int64(expert_offsets.dim_size[0]() - 1),
                eps,
            ),
            ctx,
        )
