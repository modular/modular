# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: nvptx_backend
# REQUIRES: has_cuda_device
# RUN: %mojo %s | FileCheck %s

from math import div_ceil, iota
from gpu import *
from complex import ComplexSIMD
from memory.buffer import NDBuffer
from math import abs
from utils.index import Index
from algorithm import vectorize
from sys.info import simdwidthof
from gpu.host import Context, Dim, Function, Stream, synchronize
from gpu.host.event import time_function
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
)


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


fn mandelbrot(out: NDBuffer[2, DimList(height, width), int_type]):
    # Each task gets a row.
    let row = ThreadIdx.x() + BlockDim.x() * BlockIdx.x()
    if row >= height:
        return

    let scale_x = (max_x - min_x) / width
    let scale_y = (max_y - min_y) / height

    @always_inline
    @parameter
    fn compute_vector[simd_width: Int](col: Int):
        """Each time we operate on a `simd_width` vector of pixels."""
        if col >= width:
            return
        let cx = min_x + (col + iota[float_type, simd_width]()) * scale_x
        let cy = min_y + row * scale_y
        let c = ComplexSIMD[float_type, simd_width](cx, cy)
        out.simd_store[simd_width](
            Index(row, col), mandelbrot_kernel[simd_width](c)
        )

    # We vectorize the call to compute_vector where call gets a chunk of
    # pixels.
    vectorize[simdwidthof[float_type](), compute_vector](width)


fn run_mandelbrot() raises:
    let stream = Stream()

    let out_host = Tensor[int_type](width, height)

    let out_device = _malloc[int_type](width * height)

    let func = Function[
        fn (NDBuffer[2, DimList(width, height), int_type]) -> None, mandelbrot
    ]()

    @always_inline
    @parameter
    fn run_mandelbrot() raises:
        func(
            (div_ceil(height, BLOCK_SIZE),),
            (BLOCK_SIZE,),
            NDBuffer[2, DimList(height, width), int_type](out_device),
            stream=stream,
        )

    run_mandelbrot()  # Warmup
    print(
        "Computation took:", time_function[run_mandelbrot]() / 1_000_000_000.0
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
