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

# COM: Note: CPU function compilation not supported
# COM: %mojo-no-debug -D MODULAR_ASYNCRT_DEVICE_CONTEXT_V2=cpu %s
# RUN: %mojo-no-debug -D MODULAR_ASYNCRT_DEVICE_CONTEXT_V2=gpu %s

from asyncrt_test_utils import create_test_device_context, expect_eq
from gpu.id import global_idx
from layout import Layout, LayoutTensor
from math import ceildiv
from tensor import (
    InputTensor,
    OutputTensor,
    StaticTensorSpec,
)
from utils import IndexList
from buffer.dimlist import DimList

alias WIDTH = 5
alias HEIGHT = 10
alias NUM_CHANNELS = 3

alias int_dtype = DType.uint8
alias float_dtype = DType.float32
alias rgb_layout_orig = Layout.row_major(HEIGHT, WIDTH, NUM_CHANNELS)
alias gray_layout_orig = Layout.row_major(HEIGHT, WIDTH)
alias rgb_spec = StaticTensorSpec[int_dtype, 3].create_unknown()
alias rgb_layout = rgb_spec.to_layout()
alias gray_spec = StaticTensorSpec[int_dtype, 2].create_unknown()
alias gray_layout = gray_spec.to_layout()


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


fn main() raises:
    var ctx = create_test_device_context()

    var rgb_buffer = ctx.enqueue_create_buffer[int_dtype](
        rgb_layout_orig.size()
    )
    var gray_buffer = ctx.enqueue_create_buffer[int_dtype](
        gray_layout_orig.size()
    )

    var rgb_tensor = InputTensor[static_spec=rgb_spec](
        rgb_buffer._unsafe_ptr(), IndexList[3](HEIGHT, WIDTH, NUM_CHANNELS)
    )

    # Map device buffer to host to initialize values from CPU
    with rgb_buffer.map_to_host() as host_buffer:
        var rgb_tensor = InputTensor[static_spec=rgb_spec](
            host_buffer.unsafe_ptr(), IndexList[3](HEIGHT, WIDTH, NUM_CHANNELS)
        ).to_layout_tensor()
        # Fill the image with initial colors.
        for row in range(HEIGHT):
            for col in range(WIDTH):
                rgb_tensor[row, col, 0] = row + col
                rgb_tensor[row, col, 1] = row + col + 20
                rgb_tensor[row, col, 2] = row + col + 40

    var gray_tensor = OutputTensor[static_spec=gray_spec](
        gray_buffer._unsafe_ptr(), IndexList[2](HEIGHT, WIDTH)
    )

    # The grid is divided up into blocks, making sure there's an extra
    # full block for any remainder. This hasn't been tuned for any specific
    # GPU.
    alias BLOCK_SIZE = 16
    num_col_blocks = ceildiv(WIDTH, BLOCK_SIZE)
    num_row_blocks = ceildiv(HEIGHT, BLOCK_SIZE)

    # Launch the compiled function on the GPU. The target device is specified
    # first, followed by all function arguments. The last two named parameters
    # are the dimensions of the grid in blocks, and the block dimensions.
    ctx.enqueue_function_checked[color_to_grayscale, color_to_grayscale](
        rgb_tensor,
        gray_tensor,
        grid_dim=(num_col_blocks, num_row_blocks),
        block_dim=(BLOCK_SIZE, BLOCK_SIZE),
    )

    with gray_buffer.map_to_host() as host_buffer:
        host_tensor = LayoutTensor[int_dtype, gray_layout_orig](host_buffer)
        print("Resulting grayscale image:")
        print_image(host_tensor)
        expect_eq(host_tensor[0, 0], 17)
        expect_eq(host_tensor[0, 3], 19)
        expect_eq(host_tensor[5, 4], 25)
        expect_eq(host_tensor[7, 1], 24)
        expect_eq(host_tensor[9, 4], 29)

    _ = rgb_buffer
    _ = gray_buffer


def print_image(gray_tensor: LayoutTensor[int_dtype, gray_layout_orig]):
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
