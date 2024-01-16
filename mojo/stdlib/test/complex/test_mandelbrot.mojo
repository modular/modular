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
# RUN: %mojo -debug-level full %s | FileCheck %s


from complex import ComplexFloat32, ComplexSIMD


fn mandelbrot_iter(row: Int, col: Int) -> Int:
    alias height = 375
    alias width = 500

    let xRange: Float32 = 2.0
    let yRange: Float32 = 1.5
    let minX = 0.5 - xRange
    let maxX = 0.5 + xRange
    let minY = -0.5 - yRange
    let maxY = -0.5 + yRange

    let c = ComplexFloat32(
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

    let re = Int32(3)
    let im = Int32(4)
    let z = ComplexSIMD[DType.int32, 1](re, im)
    # CHECK: 25
    print(z.squared_norm().value)


fn main():
    test_mandelbrot_iter()
