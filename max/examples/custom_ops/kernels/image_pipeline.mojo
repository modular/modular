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

import extensibility

from max.algorithm import elementwise
from max.gpu.host import DeviceContext
from std.simd import SIMD

from extensibility import (
    Input,
    Output,
    Tensor,
    TileTensorable,
    get_kernel_simd_width,
)

from std.utils.coord import Coord, coord_to_index_list


@extensibility.register("grayscale")
struct Grayscale:
    @staticmethod
    def execute(
        img_out: Some[Output & Tensor],
        img_in: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(img_out), TileTensorable)
        comptime assert conforms_to(type_of(img_in), TileTensorable)
        comptime assert type_of(img_out).rank == 2
        comptime assert type_of(img_in).rank == 3
        var out_tile = img_out.to_tile_tensor[.uint8]()
        var in_tile = img_in.to_tile_tensor[.uint8]().as_imm()

        @inline(.always)
        def color_to_grayscale[
            simd_width: Int, alignment: Int = 1
        ](idx: Coord) {var}:
            var idx_l = coord_to_index_list(idx)
            var row = idx_l[0]
            var col = idx_l[1]

            var r_f32 = in_tile.load[width=simd_width](Coord(row, col, 0)).cast[
                .float32
            ]()
            var g_f32 = in_tile.load[width=simd_width](Coord(row, col, 1)).cast[
                .float32
            ]()
            var b_f32 = in_tile.load[width=simd_width](Coord(row, col, 2)).cast[
                .float32
            ]()

            var gray_f32 = 0.21 * r_f32 + 0.71 * g_f32 + 0.07 * b_f32

            out_tile.store[width=simd_width](
                idx, gray_f32.clamp(0, 255).cast[.uint8]()
            )

        elementwise[simd_width=1](color_to_grayscale, img_out.shape(), ctx)


@extensibility.register("brightness")
struct Brightness:
    @staticmethod
    def execute(
        img_out: Some[Output & Tensor],
        img_in: Some[Input & Tensor],
        brightness: Float32,
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(img_out), TileTensorable)
        comptime assert conforms_to(type_of(img_in), TileTensorable)
        comptime assert type_of(img_out).rank == 2
        comptime assert type_of(img_in).rank == 2
        comptime output_alignment = type_of(img_out).alignment
        var out_tile = img_out.to_tile_tensor[.uint8]()
        var in_tile = img_in.to_tile_tensor[.uint8]().as_imm()

        @inline(.always)
        def brighten[simd_width: Int, alignment: Int = 1](idx: Coord) {var}:
            var pixels_f32 = in_tile.load[
                width=simd_width, alignment=align_of[DType.uint8]()
            ](idx).cast[.float32]()

            var brightened_f32 = pixels_f32 * brightness

            out_tile.store[
                width=simd_width,
                alignment=min(
                    output_alignment,
                    alignment * align_of[DType.uint8](),
                ),
            ](idx, brightened_f32.clamp(0, 255).cast[.uint8]())

        elementwise[simd_width=get_kernel_simd_width[.uint8, ctx.target]()](
            brighten, img_out.shape(), ctx
        )


@extensibility.register("blur")
struct Blur:
    @staticmethod
    def execute(
        img_out: Some[Output & Tensor],
        img_in: Some[Input & Tensor],
        blur_size: Int64,
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(img_out), TileTensorable)
        comptime assert conforms_to(type_of(img_in), TileTensorable)
        comptime assert type_of(img_out).rank == 2
        comptime assert type_of(img_in).rank == 2
        var out_tile = img_out.to_tile_tensor[.uint8]()
        var in_tile = img_in.to_tile_tensor[.uint8]().as_imm()

        @inline(.always)
        def blur_kernel[simd_width: Int, alignment: Int = 1](idx: Coord) {var}:
            """
            Computes the blurred value for a SIMD vector of pixels.
            """
            var height = Int(in_tile.dim[0]())
            var width = Int(in_tile.dim[1]())
            var size = Int(blur_size)  # Surrounding pixels to average

            var idx_l = coord_to_index_list(idx)
            var base_row: Int = idx_l[0]
            var base_col: Int = idx_l[1]

            var pix_val_accum = 0
            var pixel_count = 0

            # Iterate over the blur kernel window
            for blur_row_offset in range(-size, size + 1):
                for blur_col_offset in range(-size, size + 1):
                    var cur_row = base_row + blur_row_offset
                    var cur_col = base_col + blur_col_offset

                    # Check if the neighbor pixel is within bounds
                    if 0 <= cur_row < height and 0 <= cur_col < width:
                        pix_val_accum += Int(
                            in_tile.load[width=1](Coord(cur_row, cur_col))
                        )
                        pixel_count += 1

            out_tile.store[width=simd_width](
                idx, UInt8(max(min(pix_val_accum // pixel_count, 255), 0))
            )

        # Apply the kernel to each pixel in the output tensor
        elementwise[simd_width=1](blur_kernel, img_out.shape(), ctx)
