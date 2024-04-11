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
# RUN: %mojo %s | FileCheck %s


from complex import ComplexFloat32, ComplexSIMD


fn mandelbrot_iter(row: Int, col: Int) -> Int:
    alias height = 375
    alias width = 500

    var xRange: Float32 = 2.0
    var yRange: Float32 = 1.5
    var minX = 0.5 - xRange
    var maxX = 0.5 + xRange
    var minY = -0.5 - yRange
    var maxY = -0.5 + yRange

    var c = ComplexFloat32(
        minX + col * xRange / width, maxY - row * yRange / height
    )

    var z = c

    var iter: Int = 0
    for i in range(10):
        iter += 1
        z = z * z + c
        if z.squared_norm() > 4:
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

    var re = Int32(3)
    var im = Int32(4)
    var z = ComplexSIMD[DType.int32, 1](re, im)
    # CHECK: 25
    print(z.squared_norm())


fn main():
    test_mandelbrot_iter()
