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


from gpu.host import Dim
from gpu.id import block_dim, block_idx, thread_idx
from math import ceildiv
from max.driver import (
    Device,
    DynamicTensor,
    Tensor,
    accelerator_device,
    cpu_device,
)
from max.driver.accelerator import compile
from sys import has_nvidia_gpu_accelerator

alias channel_dtype = DType.uint8
alias internal_float_dtype = DType.float32
alias tensor_rank = 3
alias TensorType = DynamicTensor[type=channel_dtype, rank=tensor_rank].Type


def print_image[h: Int, w: Int](t: Tensor[channel_dtype, 3]):
    """A helper function to print out the grayscale channel intensities."""
    out = t.unsafe_slice()
    for row in range(h):
        for col in range(w):
            var v = out[row, col, 0]
            if v < 100:
                print(" ", end="")
                if v < 10:
                    print(" ", end="")
            print(v, " ", end="")
        print("")


fn color_to_grayscale_conversion(
    width: Int,
    height: Int,
    image: TensorType,
    out: TensorType,
):
    """Converting each RGB pixel to grayscale, parallelized across the output tensor on the GPU.
    """
    row = block_dim.y * block_idx.y + thread_idx.y
    col = block_dim.x * block_idx.x + thread_idx.x

    if col < width and row < height:
        red = image[row, col, 0].cast[internal_float_dtype]()
        green = image[row, col, 1].cast[internal_float_dtype]()
        blue = image[row, col, 2].cast[internal_float_dtype]()
        gray = 0.21 * red + 0.71 * green + 0.07 * blue

        out[row, col, 0] = gray.cast[channel_dtype]()


def main():
    @parameter
    if has_nvidia_gpu_accelerator():
        # Attempt to connect to a compatible GPU. If one is not found, this will
        # error out and exit.
        gpu_device = accelerator_device()
        host_device = cpu_device()

        alias IMAGE_WIDTH = 5
        alias IMAGE_HEIGHT = 10
        alias NUM_CHANNELS = 3

        # Allocate the input image tensor on the host.
        rgb_tensor = Tensor[channel_dtype, tensor_rank](
            (IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS), host_device
        )

        # Fill the image with initial colors.
        for row in range(IMAGE_HEIGHT):
            for col in range(IMAGE_WIDTH):
                rgb_tensor[row, col, 0] = row + col
                rgb_tensor[row, col, 1] = row + col + 20
                rgb_tensor[row, col, 2] = row + col + 40

        # Move the image tensor to the accelerator.
        rgb_tensor = rgb_tensor.move_to(gpu_device)

        # Allocate a tensor on the accelerator to host the grayscale image.
        gray_tensor = Tensor[channel_dtype, tensor_rank](
            (IMAGE_HEIGHT, IMAGE_WIDTH, 1), gpu_device
        )

        # Compile the function to run across a grid on the GPU.
        gpu_function = compile[color_to_grayscale_conversion](gpu_device)

        # The grid is divided up into blocks, making sure there's an extra
        # full block for any remainder. This hasn't been tuned for any specific
        # GPU.
        alias BLOCK_SIZE = 16
        num_col_blocks = ceildiv(IMAGE_WIDTH, BLOCK_SIZE)
        num_row_blocks = ceildiv(IMAGE_HEIGHT, BLOCK_SIZE)

        # Launch the compiled function on the GPU. The target device is specified
        # first, followed by all function arguments. The last two named parameters
        # are the dimensions of the grid in blocks, and the block dimensions.
        gpu_function(
            gpu_device,
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
            rgb_tensor.unsafe_slice(),
            gray_tensor.unsafe_slice(),
            grid_dim=Dim(num_col_blocks, num_row_blocks),
            block_dim=Dim(BLOCK_SIZE, BLOCK_SIZE),
        )

        # Move the output tensor back onto the CPU so that we can read the results.
        gray_tensor = gray_tensor.move_to(host_device)

        print("Resulting grayscale image:")
        print_image[IMAGE_HEIGHT, IMAGE_WIDTH](gray_tensor)
    else:
        print(
            "These examples require a MAX-compatible NVIDIA GPU and none was"
            " detected."
        )
