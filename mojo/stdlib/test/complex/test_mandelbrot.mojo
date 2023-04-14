# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# This uses mandelbrot as an example to test how the entire stdlib works
# together.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s | FileCheck %s


from DType import DType
from IO import print
from Range import range
from SIMD import SIMD, F32
from Complex import ComplexSIMD, ComplexF32


fn mandelbrot_iter(row: Int, col: Int) -> Int:

    alias height = 375
    alias width = 500

    let xRange: F32 = 2.0
    let yRange: F32 = 1.5
    let minX = 0.5 - xRange
    let maxX = 0.5 + xRange
    let minY = -0.5 - yRange
    let maxY = -0.5 + yRange

    let c = ComplexF32(
        minX + col * xRange / width, maxY - row * yRange / height
    )

    var z = c

    var iter: Int = 0
    for i in range(10):
        iter += 1
        z = z * z + c
        if z.norm() > 4:
            return iter
    return iter


# CHECK-LABEL: test_mandelbrot_iter
fn test_mandelbrot_iter():
    print("== test_mandelbrot_iter")

    # CHECK: 1
    print(mandelbrot_iter(0, 0))

    # CHECK: 1
    print(mandelbrot_iter(0, 1))

    # CHECK: 2
    print(mandelbrot_iter(50, 50))

    # CHECK: 3
    print(mandelbrot_iter(100, 100))

    let re = SIMD[DType.si32, 1].splat(3)
    let im = SIMD[DType.si32, 1].splat(4)
    let z = ComplexSIMD[DType.si32, 1](re, im)
    # CHECK: 25
    print(z.norm().value)


fn main():
    test_mandelbrot_iter()
