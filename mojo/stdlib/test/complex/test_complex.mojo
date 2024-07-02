# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s


from complex import ComplexSIMD, ComplexFloat32, abs
from testing import assert_almost_equal, assert_equal


def test_abs():
    assert_equal(abs(ComplexFloat32(0, 0)), 0)
    assert_equal(abs(ComplexFloat32(1, 0)), 1)
    assert_equal(abs(ComplexFloat32(0, 1)), 1)
    assert_almost_equal(abs(ComplexFloat32(-1, -1)), 1.41421)
    assert_almost_equal(abs(ComplexFloat32(-93, -23)), 95.801)


def test_complex_str():
    assert_equal(str(ComplexFloat32(0, 0)), "0.0")
    assert_equal(str(ComplexFloat32(1, 0)), "1.0")
    assert_equal(str(ComplexFloat32(0, 1)), "0.0 + 1.0i")
    assert_equal(str(ComplexFloat32(1, 1)), "1.0 + 1.0i")

    assert_equal(
        str(
            ComplexSIMD[DType.float32, 2](
                SIMD[DType.float32, 2](1, 0),
                SIMD[DType.float32, 2](0, 1),
            )
        ),
        "[1.0, 0.0 + 1.0i]",
    )


def main():
    test_abs()
    test_complex_str()
