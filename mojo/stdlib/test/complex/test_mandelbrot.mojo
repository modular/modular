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
# RUN: lit %s | FileCheck %s

from Bool import Bool
from DType import DType
from Int import Int
from IO import print
from Range import range
from SIMD import SIMD
from Complex import Complex


fn mandelbrot_iter(row: Int, col: Int) -> Int:

    alias height = 375
    alias width = 500

    let xRange: SIMD[1, DType.f32.value] = 2.0
    let yRange: SIMD[1, DType.f32.value] = 1.5
    let minX = 0.5 - xRange
    let maxX = 0.5 + xRange
    let minY = -0.5 - yRange
    let maxY = -0.5 + yRange

    let c = Complex[1, DType.f32](
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
    print("== test_mandelbrot_iter\n")

    # CHECK: 1
    print(mandelbrot_iter(0, 0))

    # CHECK: 1
    print(mandelbrot_iter(0, 1))

    # CHECK: 2
    print(mandelbrot_iter(50, 50))

    # CHECK: 3
    print(mandelbrot_iter(100, 100))

    let re = SIMD[1, DType.si32.value].splat(3)
    let im = SIMD[1, DType.si32.value].splat(4)
    let z = Complex[1, DType.si32.value](re, im)
    # CHECK: 25
    print(z.norm().value)


fn main():
    test_mandelbrot_iter()
