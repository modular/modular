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

from collections.string import StringSlice
from complex import ComplexSIMD
from gpu.host import Dim
from gpu.id import thread_idx, block_dim, block_idx
from layout import LayoutTensor, Layout
from math import ceildiv
from max.driver import Accelerator, Tensor, accelerator, cpu
from sys import has_nvidia_gpu_accelerator

alias float_dtype = DType.float32
alias int_dtype = DType.int32


def draw_mandelbrot[h: Int, w: Int](t: Tensor[int_dtype, 2], max: Int):
    """A helper function to visualize the Mandelbrot set in ASCII art."""
    alias sr = StringSlice("....,c8M@jawrpogOQEPGJ")
    out = t.to_layout_tensor()
    for row in range(h):
        for col in range(w):
            var v = out[row, col]
            if v < max:
                var idx = Int(v % len(sr))
                var p = sr[idx]
                print(p, end="")
            else:
                print(" ", end="")
        print("")


fn mandelbrot[
    layout: Layout
](
    min_x: Scalar[float_dtype],
    min_y: Scalar[float_dtype],
    scale_x: Scalar[float_dtype],
    scale_y: Scalar[float_dtype],
    max_iterations: Scalar[int_dtype],
    out: LayoutTensor[int_dtype, layout, MutableAnyOrigin],
):
    """The per-element calculation of iterations to escape in the Mandelbrot set.
    """
    # Obtain the position in the grid from the X, Y thread locations.
    var row = block_dim.y * block_idx.y + thread_idx.y
    var col = block_dim.x * block_idx.x + thread_idx.x

    # Calculate the complex C corresponding to that grid location.
    var cx = min_x + col * scale_x
    var cy = min_y + row * scale_y
    var c = ComplexSIMD[float_dtype, 1](cx, cy)

    # Perform the Mandelbrot iteration loop calculation.
    var z = ComplexSIMD[float_dtype, 1](0, 0)
    var iters = Scalar[int_dtype](0)

    var in_set_mask: Scalar[DType.bool] = True
    for _ in range(max_iterations):
        if not any(in_set_mask):
            break
        in_set_mask = z.squared_norm() <= 4
        iters = in_set_mask.select(iters + 1, iters)
        z = z.squared_add(c)

    # Write out the resulting iterations to escape.
    out[row, col] = iters


def main():
    @parameter
    if has_nvidia_gpu_accelerator():
        # Attempt to connect to a compatible GPU. If one is not found, this will
        # error out and exit.
        gpu_device = accelerator()
        host_device = cpu()

        # Set the resolution of the Mandelbrot set grid that will be calculated.
        alias GRID_WIDTH = 60
        alias GRID_HEIGHT = 25

        # The grid is divided up into blocks, making sure there's an extra
        # full block for any remainder. This hasn't been tuned for any specific
        # GPU.
        alias BLOCK_SIZE = 16
        num_col_blocks = ceildiv(GRID_WIDTH, BLOCK_SIZE)
        num_row_blocks = ceildiv(GRID_HEIGHT, BLOCK_SIZE)

        # Set the parameters for the area of the Mandelbrot set we'll be examining.
        alias MIN_X: Scalar[float_dtype] = -2.0
        alias MAX_X: Scalar[float_dtype] = 0.7
        alias MIN_Y: Scalar[float_dtype] = -1.12
        alias MAX_Y: Scalar[float_dtype] = 1.12
        alias SCALE_X = (MAX_X - MIN_X) / GRID_WIDTH
        alias SCALE_Y = (MAX_Y - MIN_Y) / GRID_HEIGHT
        alias MAX_ITERATIONS = 100

        # Allocate a tensor on the target device to hold the resulting set.
        out_tensor = Tensor[int_dtype, 2]((GRID_HEIGHT, GRID_WIDTH), gpu_device)

        out_layout_tensor = out_tensor.to_layout_tensor()

        # Compile the function to run across a grid on the GPU.
        gpu_function = Accelerator.compile[
            mandelbrot[out_layout_tensor.layout]
        ](gpu_device)

        # Launch the compiled function on the GPU. The target device is specified
        # first, followed by all function arguments. The last two named parameters
        # are the dimensions of the grid in blocks, and the block dimensions.
        gpu_function(
            gpu_device,
            MIN_X,
            MIN_Y,
            SCALE_X,
            SCALE_Y,
            MAX_ITERATIONS,
            out_layout_tensor,
            grid_dim=Dim(num_col_blocks, num_row_blocks),
            block_dim=Dim(BLOCK_SIZE, BLOCK_SIZE),
        )

        # Move the output tensor back onto the CPU so that we can read the results.
        out_tensor = out_tensor.move_to(host_device)

        # Draw the final Mandelbrot set.
        draw_mandelbrot[GRID_HEIGHT, GRID_WIDTH](out_tensor, max=MAX_ITERATIONS)
    else:
        print(
            "These examples require a MAX-compatible NVIDIA GPU and none was"
            " detected."
        )
