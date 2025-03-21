# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from math import ceildiv, iota
from sys.info import simdwidthof

from algorithm import vectorize
from buffer import DimList, NDBuffer
from complex import ComplexSIMD
from gpu import *
from gpu.host import DeviceContext
from testing import assert_equal

from utils.index import Index

alias float_type = DType.float64
alias int_type = DType.index


alias width = 4096
alias height = 4096
alias MAX_ITERS = 1000
alias BLOCK_SIZE = 32

alias min_x = -2.0
alias max_x = 0.47
alias min_y = -1.12
alias max_y = 1.12


@always_inline
fn mandelbrot_kernel[
    simd_width: Int
](c: ComplexSIMD[float_type, simd_width]) -> SIMD[int_type, simd_width]:
    """A vectorized implementation of the inner mandelbrot computation."""
    var z = ComplexSIMD[float_type, simd_width](0, 0)
    var iters = SIMD[int_type, simd_width](0)

    var in_set_mask: SIMD[DType.bool, simd_width] = True
    for i in range(MAX_ITERS):
        if not in_set_mask.reduce_or():
            break
        in_set_mask = z.squared_norm() <= 4
        iters = in_set_mask.select(iters + 1, iters)
        z = z.squared_add(c)

    return iters


fn mandelbrot(out_ptr: UnsafePointer[Scalar[int_type]]):
    # Each task gets a row.
    var row = global_idx.x
    if row >= height:
        return

    var out = NDBuffer[int_type, 2](out_ptr, Index(height, width))

    alias scale_x = (max_x - min_x) / width
    alias scale_y = (max_y - min_y) / height

    @always_inline
    @parameter
    fn compute_vector[simd_width: Int](col: Int):
        """Each time we operate on a `simd_width` vector of pixels."""
        if col >= width:
            return
        var cx = min_x + (col + iota[float_type, simd_width]()) * scale_x
        var cy = min_y + row * SIMD[float_type, simd_width](scale_y)
        var c = ComplexSIMD[float_type, simd_width](cx, cy)
        out.store[width=simd_width](
            Index(row, col), mandelbrot_kernel[simd_width](c)
        )

    # We vectorize the call to compute_vector where call gets a chunk of
    # pixels.
    vectorize[compute_vector, simdwidthof[float_type]()](width)


fn run_mandelbrot(ctx: DeviceContext) raises:
    var out_host = UnsafePointer[Scalar[int_type]].alloc(width * height)

    var out_device = ctx.enqueue_create_buffer[int_type](width * height)

    @always_inline
    @parameter
    fn run_mandelbrot(ctx: DeviceContext) raises:
        ctx.enqueue_function[mandelbrot](
            out_device,
            grid_dim=(ceildiv(height, BLOCK_SIZE),),
            block_dim=(BLOCK_SIZE,),
        )

    run_mandelbrot(ctx)  # Warmup
    print(
        "Computation took:",
        ctx.execution_time[run_mandelbrot](1) / 1_000_000_000.0,
    )

    ctx.enqueue_copy(out_host, out_device)

    ctx.synchronize()

    var accum = SIMD[int_type, 1](0)
    for i in range(width):
        for j in range(height):
            accum += out_host[i * width + j]
    assert_equal(4687767697, accum)

    _ = out_device

    _ = out_host


# CHECK-NOT: CUDA_ERROR
def main():
    with DeviceContext() as ctx:
        run_mandelbrot(ctx)
