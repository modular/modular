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

# DOC: develop/custom-kernels-pytorch.mdx

from std.sys import align_of

import extensibility

from max.algorithm import elementwise
from max.gpu.host import DeviceContext
from extensibility import Input, Output, Tensor, TileTensorable

from std.utils.coord import Coord, coord_to_index_list


@extensibility.register("grayscale")
struct Grayscale:
    """Convert RGB image tensor to grayscale using weighted formula."""

    @staticmethod
    def execute(
        img_out: Some[Output & Tensor],
        img_in: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        """Execute grayscale conversion on the input image tensor.

        Args:
            img_out: Output tensor for grayscale image (height x width).
            img_in: Input tensor for RGB image (height x width x channels).
            ctx: Device context for execution.
        """
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
            """Convert one RGB pixel to grayscale using perceptual weighting.

            Args:
                idx: Index into the output tensor (row, col).
            """
            comptime align = align_of[DType.uint8]()

            def load(
                row: Int, col: Int, channel: Int
            ) {imm in_tile} -> SIMD[.float32, simd_width]:
                return in_tile.load[width=simd_width, alignment=align](
                    Coord(row, col, channel)
                ).cast[.float32]()

            var idx_l = coord_to_index_list(idx)
            var row = idx_l[0]
            var col = idx_l[1]

            # Load RGB values from input tensor
            var r = load(row, col, 0)
            var g = load(row, col, 1)
            var b = load(row, col, 2)

            # Apply standard grayscale conversion formula
            # These weights are based on human visual perception
            var gray = 0.21 * r + 0.71 * g + 0.07 * b

            # Clamp to valid uint8 range, convert back, and store
            out_tile.store[
                width=simd_width,
                alignment=min(
                    type_of(img_out).alignment, element_alignment * align
                ),
            ](idx, min(gray, 255).cast[.uint8]())

        # Run the conversion in parallel over every output pixel
        elementwise[simd_width=1](color_to_grayscale, img_out.shape(), ctx)
