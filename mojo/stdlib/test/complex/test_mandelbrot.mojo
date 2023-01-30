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
# RUN: kgen %s -execute -func='$test_mandelbrot::main():index()' -I %stdlibdir | FileCheck %s

from Bool import Bool
from DType import DType
from Int import Int
from IO import print
from Range import range
from SIMD import SIMD


struct Complex[type: DType]:
    var re: SIMD[1, type.value]
    var im: SIMD[1, type.value]

    fn __new__(
        re: SIMD[1, type.value], im: SIMD[1, type.value]
    ) -> Complex[type]:
        return Complex[type] {re: re, im: im}

    fn __add__(self, rhs: Complex[type]) -> Complex[type]:
        return Complex[type] {re: self.re + rhs.re, im: self.im + rhs.im}

    fn __mul__(self, rhs: Complex[type]) -> Complex[type]:
        return Complex[type] {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }

    fn abs(self) -> SIMD[1, type.value]:
        return self.re * self.re + self.im * self.im


fn mandelbrot_iter(row: Int, col: Int) -> Int:

    alias height = 375
    alias width = 500

    let xRange: SIMD[1, DType.f32.value] = 2.0
    let yRange: SIMD[1, DType.f32.value] = 1.5
    let minX = 0.5 - xRange
    let maxX = 0.5 + xRange
    let minY = -0.5 - yRange
    let maxY = -0.5 + yRange

    let c = Complex[DType.f32](
        minX + col * xRange / width, maxY - row * yRange / height
    )

    var z = c

    var iter: Int = 0
    for i in range(10):
        iter += 1
        z = z * z + c
        if z.abs() > 4:
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


@export
fn main() -> __mlir_type.index:
    test_mandelbrot_iter()
    return 0
