# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: nvptx_backend
# REQUIRES: has_cuda_device
# RUN: %mojo %s | FileCheck %s

from math import abs, div_ceil, iota
from sys.info import simdwidthof

from algorithm import vectorize
from buffer import NDBuffer
from complex import ComplexSIMD
from gpu import *
from gpu.host import Context, Dim, Function, Stream, synchronize
from gpu.host.event import time_function
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
)
from tensor import Tensor

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


fn mandelbrot(out: NDBuffer[int_type, 2, DimList(height, width)]):
    # Each task gets a row.
    var row = ThreadIdx.x() + BlockDim.x() * BlockIdx.x()
    if row >= height:
        return

    var scale_x = (max_x - min_x) / width
    var scale_y = (max_y - min_y) / height

    @always_inline
    @parameter
    fn compute_vector[simd_width: Int](col: Int):
        """Each time we operate on a `simd_width` vector of pixels."""
        if col >= width:
            return
        var cx = min_x + (col + iota[float_type, simd_width]()) * scale_x
        var cy = min_y + row * scale_y
        var c = ComplexSIMD[float_type, simd_width](cx, cy)
        out.store[width=simd_width](
            Index(row, col), mandelbrot_kernel[simd_width](c)
        )

    # We vectorize the call to compute_vector where call gets a chunk of
    # pixels.
    vectorize[compute_vector, simdwidthof[float_type](), width]()


fn run_mandelbrot() raises:
    var stream = Stream()

    var out_host = Tensor[int_type](width, height)

    var out_device = _malloc[int_type](width * height)

    var func = Function[__type_of(mandelbrot), mandelbrot]()

    @always_inline
    @parameter
    fn run_mandelbrot(stream: Stream) raises:
        func(
            NDBuffer[int_type, 2, DimList(height, width)](out_device),
            grid_dim=(div_ceil(height, BLOCK_SIZE),),
            block_dim=(BLOCK_SIZE,),
            stream=stream,
        )

    run_mandelbrot(stream)  # Warmup
    print(
        "Computation took:",
        time_function[run_mandelbrot](stream) / 1_000_000_000.0,
    )

    _copy_device_to_host(out_host.data(), out_device, width * height)

    var accum = SIMD[int_type, 1](0)
    for i in range(width):
        for j in range(height):
            accum += out_host[i, j]
    # CHECK: 4687759697
    print(accum)

    _free(out_device)

    _ = out_host

    _ = func ^
    _ = stream ^


# CHECK-NOT: CUDA_ERROR
def main():
    try:
        with Context() as ctx:
            run_mandelbrot()
    except e:
        print("CUDA_ERROR:", e)
