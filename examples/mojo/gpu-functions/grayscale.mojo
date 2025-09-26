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


from math import ceildiv
from sys import has_accelerator

from gpu.host import DeviceContext
from gpu.id import global_idx
from layout import Layout, LayoutTensor

alias WIDTH = 5
alias HEIGHT = 10
alias NUM_CHANNELS = 3

alias int_dtype = DType.uint8
alias float_dtype = DType.float32
alias rgb_layout = Layout.row_major(HEIGHT, WIDTH, NUM_CHANNELS)
alias gray_layout = Layout.row_major(HEIGHT, WIDTH)


def main():
    constrained[
        has_accelerator(), "This example requires a supported accelerator"
    ]()

    var ctx = DeviceContext()

    var rgb_buffer = ctx.enqueue_create_buffer[int_dtype](rgb_layout.size())
    var gray_buffer = ctx.enqueue_create_buffer[int_dtype](gray_layout.size())

    # Map device buffer to host to initialize values from CPU
    with rgb_buffer.map_to_host() as host_buffer:
        var rgb_tensor = LayoutTensor[int_dtype, rgb_layout](host_buffer)
        # Fill the image with initial colors.
        for row in range(HEIGHT):
            for col in range(WIDTH):
                rgb_tensor[row, col, 0] = row + col
                rgb_tensor[row, col, 1] = row + col + 20
                rgb_tensor[row, col, 2] = row + col + 40

    var rgb_tensor = LayoutTensor[int_dtype, rgb_layout](rgb_buffer)
    var gray_tensor = LayoutTensor[int_dtype, gray_layout](gray_buffer)

    # The grid is divided up into blocks, making sure there's an extra
    # full block for any remainder. This hasn't been tuned for any specific
    # GPU.
    alias BLOCK_SIZE = 16
    num_col_blocks = ceildiv(WIDTH, BLOCK_SIZE)
    num_row_blocks = ceildiv(HEIGHT, BLOCK_SIZE)

    # Launch the compiled function on the GPU. The target device is specified
    # first, followed by all function arguments. The last two named parameters
    # are the dimensions of the grid in blocks, and the block dimensions.
    ctx.enqueue_function[color_to_grayscale](
        rgb_tensor,
        gray_tensor,
        grid_dim=(num_col_blocks, num_row_blocks),
        block_dim=(BLOCK_SIZE, BLOCK_SIZE),
    )

    with gray_buffer.map_to_host() as host_buffer:
        host_tensor = LayoutTensor[int_dtype, gray_layout](host_buffer)
        print("Resulting grayscale image:")
        print_image(host_tensor)


fn color_to_grayscale(
    rgb_tensor: LayoutTensor[int_dtype, rgb_layout, MutableAnyOrigin],
    gray_tensor: LayoutTensor[int_dtype, gray_layout, MutableAnyOrigin],
):
    """Converting each RGB pixel to grayscale, parallelized across the output tensor on the GPU.
    """
    row = global_idx.y
    col = global_idx.x

    if col < WIDTH and row < HEIGHT:
        red = rgb_tensor[row, col, 0].cast[float_dtype]()
        green = rgb_tensor[row, col, 1].cast[float_dtype]()
        blue = rgb_tensor[row, col, 2].cast[float_dtype]()
        gray = 0.21 * red + 0.71 * green + 0.07 * blue

        gray_tensor[row, col] = gray.cast[int_dtype]()


def print_image(gray_tensor: LayoutTensor[int_dtype, gray_layout]):
    """A helper function to print out the grayscale channel intensities."""
    for row in range(HEIGHT):
        for col in range(WIDTH):
            var v = gray_tensor[row, col]
            if v < 100:
                print(" ", end="")
                if v < 10:
                    print(" ", end="")
            print(v, " ", end="")
        print("")
