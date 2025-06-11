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

from python import PythonObject
from python.bindings import PythonModuleBuilder
from os import abort

from math import ceildiv
from sys import has_accelerator

from complex import ComplexSIMD
from gpu import global_idx
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor

alias GRID_WIDTH = 60
alias GRID_HEIGHT = 25

alias float_dtype = DType.float32
alias int_dtype = DType.int32

alias MIN_X: Scalar[float_dtype] = -2.0
alias MAX_X: Scalar[float_dtype] = 0.7
alias MIN_Y: Scalar[float_dtype] = -1.12
alias MAX_Y: Scalar[float_dtype] = 1.12

alias layout = Layout.row_major(GRID_HEIGHT, GRID_WIDTH)


# An interface for this Mojo module must be exported to Python.
@export
fn PyInit_mandelbrot_mojo() -> PythonObject:
    try:
        # A Python module is constructed, matching the name of this Mojo module.
        var module = PythonModuleBuilder("mandelbrot_mojo")
        # The functions to be exported are registered within this module.
        module.def_function[run_mandelbrot]("run_mandelbrot")
        return module.finalize()
    except e:
        return abort[PythonObject](
            String("failed to create Python module: ", e)
        )


fn run_mandelbrot(iterations: PythonObject) raises -> PythonObject:
    """The main GPU dispatch function for the Mandelbrot calculation called from Python.
    """
    constrained[has_accelerator(), "This example requires a supported GPU"]()

    # Get the context for the attached GPU
    var ctx = DeviceContext()

    # Allocate a tensor on the target device to hold the resulting set.
    var dev_buf = ctx.enqueue_create_buffer[int_dtype](layout.size())
    var out_tensor = LayoutTensor[int_dtype, layout](dev_buf)

    # Compute how many blocks are needed in each dimension to fully cover the grid,
    # rounding up to ensure even partially filled blocks are launched.
    alias BLOCK_SIZE = 16
    alias COL_BLOCKS = ceildiv(GRID_WIDTH, BLOCK_SIZE)
    alias ROW_BLOCKS = ceildiv(GRID_HEIGHT, BLOCK_SIZE)

    # Launch the Mandelbrot kernel on the GPU with a 2D grid of thread blocks.
    ctx.enqueue_function[mandelbrot](
        out_tensor,
        Int(iterations),
        grid_dim=(COL_BLOCKS, ROW_BLOCKS),
        block_dim=(BLOCK_SIZE, BLOCK_SIZE),
    )
    ctx.synchronize()

    # Map the output tensor data to CPU so that we can read the results.
    with dev_buf.map_to_host() as host_buf:
        var host_tensor = LayoutTensor[int_dtype, layout](host_buf)
        # Return the ASCII art string representation to Python.
        return draw_mandelbrot(host_tensor, Int(iterations))


fn mandelbrot(
    tensor: LayoutTensor[int_dtype, layout, MutableAnyOrigin], iterations: Int32
):
    """The per-element calculation of iterations to escape in the Mandelbrot set.
    """
    # Obtain the position in the grid from the X, Y thread locations.
    var row = global_idx.y
    var col = global_idx.x

    alias SCALE_X = (MAX_X - MIN_X) / GRID_WIDTH
    alias SCALE_Y = (MAX_Y - MIN_Y) / GRID_HEIGHT

    # Calculate the complex C corresponding to that grid location.
    var cx = MIN_X + col * SCALE_X
    var cy = MIN_Y + row * SCALE_Y
    var c = ComplexSIMD[float_dtype, 1](cx, cy)

    # Perform the Mandelbrot iteration loop calculation.
    var z = ComplexSIMD[float_dtype, 1](0, 0)
    var iters = Scalar[int_dtype](0)

    var in_set_mask: Scalar[DType.bool] = True
    for _ in range(iterations):
        if not any(in_set_mask):
            break
        in_set_mask = z.squared_norm() <= 4
        iters = in_set_mask.select(iters + 1, iters)
        z = z.squared_add(c)

    # Write out the resulting iterations to escape.
    tensor[row, col] = iters


def draw_mandelbrot(
    tensor: LayoutTensor[int_dtype, layout], iterations: Int32
) -> String:
    """A helper function to visualize the Mandelbrot set in ASCII art."""
    alias sr = StringSlice("....,c8M@jawrpogOQEPGJ")
    var buffer = String()
    for row in range(GRID_HEIGHT):
        for col in range(GRID_WIDTH):
            var v = tensor[row, col]
            if v < iterations:
                var idx = Int(v % len(sr))
                var p = sr[idx]
                buffer += p
            else:
                buffer += " "
        buffer += "\n"
    return buffer
