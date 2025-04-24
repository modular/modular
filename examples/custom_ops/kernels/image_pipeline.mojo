# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from compiler import register
from max.tensor import InputTensor, OutputTensor, foreach
from runtime.asyncrt import DeviceContextPtr
from utils.index import IndexList
from layout import LayoutTensor, Layout
from gpu import global_idx
from math import ceildiv


# TODO: Convert to using `foreach` to win Mojo swag
@register("grayscale")
struct Grayscale:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        img_out: OutputTensor[type = DType.uint8, rank=2],
        img_in: InputTensor[type = DType.uint8, rank=3],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        fn color_to_grayscale():
            var col = global_idx.x
            var row = global_idx.y
            var height = img_in.shape()[0]
            var width = img_in.shape()[1]

            if col < width and row < height:
                var red = img_in[row, col, 0].cast[DType.float32]()
                var green = img_in[row, col, 1].cast[DType.float32]()
                var blue = img_in[row, col, 2].cast[DType.float32]()

                var gray = 0.21 * red + 0.71 * green + 0.07 * blue
                img_out[row, col, 0] = gray.cast[DType.uint8]()

        alias BLOCK_SIZE = 16
        var row_blocks = ceildiv(img_in.shape()[0], BLOCK_SIZE)
        var col_blocks = ceildiv(img_in.shape()[1], BLOCK_SIZE)

        var dev_ctx = ctx.get_device_context()
        dev_ctx.enqueue_function[color_to_grayscale](
            grid_dim=(col_blocks, row_blocks),
            block_dim=(BLOCK_SIZE, BLOCK_SIZE),
        )


@register("brightness")
struct Brightness:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        img_out: OutputTensor[type = DType.uint8, rank=2],
        img_in: InputTensor[type = DType.uint8, rank=2],
        brightness: Float32,
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        fn brighten[
            width: Int
        ](idx: IndexList[img_in.rank]) -> SIMD[img_in.type, width]:
            var pixels = img_in.load[width](idx).cast[DType.float32]()
            return (pixels * brightness).clamp(0, 255).cast[DType.uint8]()

        foreach[brighten, target=target](img_out, ctx)


# TODO: Add CPU implementation to win Mojo swag (see other kernels for examples)
@register("blur")
struct Blur:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        img_out: OutputTensor[type = DType.uint8, rank=2],
        img_in: InputTensor[type = DType.uint8, rank=2],
        blur_size: Int64,
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        fn blur_kernel():
            var col = global_idx.x
            var row = global_idx.y
            var height = img_in.shape()[0]
            var width = img_in.shape()[1]
            var size = Int(blur_size)

            if col < width and row < height:
                var pix_val = 0
                var pixels = 0

                for blur_row in range(-size, size + 1):
                    for blur_col in range(-size, size + 1):
                        var cur_row = row + blur_row
                        var cur_col = col + blur_col
                        if 0 <= cur_row < height and 0 <= cur_col < width:
                            pix_val += Int(
                                img_in[cur_row, cur_col, block_idx.x]
                            )
                            pixels += 1
                img_out[row, col, block_idx.x] = UInt8(pix_val / pixels)

        alias BLOCK_SIZE = 16
        var row_blocks = ceildiv(img_in.shape()[0], BLOCK_SIZE)
        var col_blocks = ceildiv(img_in.shape()[1], BLOCK_SIZE)

        var dev_ctx = ctx.get_device_context()
        dev_ctx.enqueue_function[blur_kernel](
            grid_dim=(col_blocks, row_blocks),
            block_dim=(BLOCK_SIZE, BLOCK_SIZE),
        )
