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

from std.sys import align_of
from std.sys.info import CompilationTarget

from extensibility import register
from max.algorithm import elementwise
from max.gpu.host import DeviceContext
from extensibility import (
    Input,
    Output,
    Tensor,
    TileTensorable,
    get_kernel_simd_width,
)

from std.utils.coord import Coord, coord_to_index_list


@register("grayscale")
struct Grayscale:
    @staticmethod
    def execute(
        img_out: Some[Output & Tensor],
        img_in: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(img_out), TileTensorable)
        comptime assert conforms_to(type_of(img_in), TileTensorable)
        comptime assert type_of(img_out).dtype == .uint8
        comptime assert type_of(img_out).rank == 2
        comptime assert type_of(img_in).dtype == .uint8
        comptime assert type_of(img_in).rank == 3
        var out_tile = img_out.to_tile_tensor[.uint8]()
        var in_tile = img_in.to_tile_tensor[.uint8]().as_imm()

        @inline(.always)
        def color_to_grayscale[
            simd_width: Int, element_alignment: Int = 1
        ](idx: Coord) {var out_tile, var in_tile}:
            var idx_l = coord_to_index_list(idx)
            var row = idx_l[0]
            var col = idx_l[1]

            comptime align = align_of[DType.uint8]()
            var r_f32 = in_tile.load[width=simd_width, alignment=align](
                Coord(row, col, 0)
            ).cast[.float32]()
            var g_f32 = in_tile.load[width=simd_width, alignment=align](
                Coord(row, col, 1)
            ).cast[.float32]()
            var b_f32 = in_tile.load[width=simd_width, alignment=align](
                Coord(row, col, 2)
            ).cast[.float32]()

            var gray_f32 = 0.21 * r_f32 + 0.71 * g_f32 + 0.07 * b_f32

            out_tile.store[
                width=simd_width,
                alignment=min(
                    type_of(img_out).alignment, element_alignment * align
                ),
            ](idx, min(gray_f32, 255).cast[.uint8]())

        elementwise[simd_width=1](color_to_grayscale, img_out.shape(), ctx)


@register("myadd")
struct MyAdd:
    @staticmethod
    def execute(
        C: Some[Output & Tensor],
        A: Some[Input & Tensor],
        B: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(C), TileTensorable)
        comptime assert conforms_to(type_of(A), TileTensorable)
        comptime assert conforms_to(type_of(B), TileTensorable)
        comptime type = type_of(C).dtype
        comptime assert type_of(A).dtype == type
        comptime assert type_of(B).dtype == type
        comptime assert type_of(A).rank == type_of(C).rank
        comptime assert type_of(B).rank == type_of(C).rank
        var c_tile = C.to_tile_tensor[type]()
        var a_tile = A.to_tile_tensor[type]().as_imm()
        var b_tile = B.to_tile_tensor[type]().as_imm()

        @inline(.always)
        def doit[
            simd_width: Int, element_alignment: Int = 1
        ](idx: Coord) {var c_tile, var a_tile, var b_tile}:
            var a = a_tile.load[width=simd_width, alignment=align_of[type]()](
                idx
            )
            var b = b_tile.load[width=simd_width, alignment=align_of[type]()](
                idx
            )
            c_tile.store[
                width=simd_width,
                alignment=min(
                    type_of(C).alignment, element_alignment * align_of[type]()
                ),
            ](idx, a + b)

        elementwise[simd_width=get_kernel_simd_width[type, ctx.target]()](
            doit, C.shape(), ctx
        )


@register("parameter_increment")
struct ParameterIncrement:
    @staticmethod
    def execute[
        increment: Int
    ](
        B: Some[Output & Tensor],
        A: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(B), TileTensorable)
        comptime assert conforms_to(type_of(A), TileTensorable)
        comptime type = type_of(B).dtype
        comptime assert type_of(A).dtype == type
        comptime assert type_of(A).rank == type_of(B).rank
        var b_tile = B.to_tile_tensor[type]()
        var a_tile = A.to_tile_tensor[type]().as_imm()

        @inline(.always)
        def doit[
            simd_width: Int, element_alignment: Int = 1
        ](idx: Coord) {var b_tile, var a_tile}:
            var a = a_tile.load[width=simd_width, alignment=align_of[type]()](
                idx
            )
            b_tile.store[
                width=simd_width,
                alignment=min(
                    type_of(B).alignment, element_alignment * align_of[type]()
                ),
            ](idx, a + type_of(a)(increment))

        elementwise[simd_width=get_kernel_simd_width[type, ctx.target]()](
            doit, B.shape(), ctx
        )


@register("scalar_add")
struct ScalarAdd:
    @staticmethod
    def execute[
        dtype: DType,
    ](C: Some[Output & Tensor], A: Scalar[dtype], B: Scalar[dtype],) raises:
        comptime assert conforms_to(type_of(C), TileTensorable)
        comptime assert type_of(C).dtype == dtype
        comptime assert type_of(C).rank == 1
        C.to_tile_tensor[dtype]().store[width=1](Coord(0), A + B)


@register("unsupported_type_op")
struct UnsupportedTypeOp:
    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        message: String,  # String is not a supported type for PyTorch custom ops
        ctx: DeviceContext,
    ) raises:
        # This operation is for testing error handling only
        # The String parameter should cause a validation error
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime assert type_of(input).dtype == dtype
        comptime assert type_of(input).rank == type_of(output).rank
        var out_tile = output.to_tile_tensor[dtype]()
        var in_tile = input.to_tile_tensor[dtype]().as_imm()

        @inline(.always)
        def copy[
            simd_width: Int, element_alignment: Int = 1
        ](idx: Coord) {var out_tile, var in_tile}:
            out_tile.store[
                width=simd_width,
                alignment=min(
                    type_of(output).alignment,
                    element_alignment * align_of[dtype](),
                ),
            ](
                idx,
                in_tile.load[width=simd_width, alignment=align_of[dtype]()](
                    idx
                ),
            )

        elementwise[
            simd_width=get_kernel_simd_width[
                dtype, CompilationTarget.current()
            ]()
        ](copy, output.shape(), ctx)
