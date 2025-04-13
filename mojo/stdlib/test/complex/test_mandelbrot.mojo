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
# RUN: %mojo-no-debug %s


from complex import ComplexFloat32, ComplexSIMD
from testing import assert_equal


fn mandelbrot_iter(row: Int, col: Int) -> Int:
    alias height = 375
    alias width = 500

    var xRange: Float32 = 2.0
    var yRange: Float32 = 1.5
    var minX = 0.5 - xRange
    var maxY = -0.5 + yRange

    var c = ComplexFloat32(
        minX + col * xRange / width, maxY - row * yRange / height
    )

    var z = c

    var iter: Int = 0
    for _i in range(10):
        iter += 1
        z = z * z + c
        if z.squared_norm() > 4:
            return iter
    return iter


def test_mandelbrot_iter():
    assert_equal(mandelbrot_iter(0, 0), 1)
    assert_equal(mandelbrot_iter(0, 1), 1)
    assert_equal(mandelbrot_iter(50, 50), 2)
    assert_equal(mandelbrot_iter(100, 100), 3)

    var z = ComplexSIMD[DType.int32, 1](re=Int32(3), im=Int32(4))
    assert_equal(z.squared_norm(), 25)


def main():
    test_mandelbrot_iter()
    # NOTE: We need to print this for the SDK self test.
    print("Mandelbrot passed")
