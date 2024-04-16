# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s

from sys import has_neon
from math import frexp, isinf, log, log2, isclose
from testing import assert_true, assert_equal


fn test_frexp[type: DType](atol: Float32, rtol: Float32) raises:
    var res = frexp(Scalar[type](123.45))
    assert_true(
        isclose(
            res[0].cast[DType.float32](),
            0.964453,
            atol=atol,
            rtol=rtol,
        )
    )
    assert_true(
        isclose(
            res[1].cast[DType.float32](),
            7.0,
            atol=atol,
            rtol=rtol,
        )
    )

    res = frexp(Scalar[type](0.1))
    assert_true(
        isclose(
            res[0].cast[DType.float32](),
            0.8,
            atol=atol,
            rtol=rtol,
        )
    )
    assert_true(
        isclose(
            res[1].cast[DType.float32](),
            -3.0,
            atol=atol,
            rtol=rtol,
        )
    )

    res = frexp(Scalar[type](-0.1))
    assert_true(
        isclose(
            res[0].cast[DType.float32](),
            -0.8,
            atol=atol,
            rtol=rtol,
        )
    )
    assert_true(
        isclose(
            res[1].cast[DType.float32](),
            -3.0,
            atol=atol,
            rtol=rtol,
        )
    )

    var res2 = frexp(SIMD[type, 4](0, 2, 4, 5))
    assert_true(
        isclose(
            res2[0].cast[DType.float32](),
            SIMD[DType.float32, 4](0.0, 0.5, 0.5, 0.625),
            atol=atol,
            rtol=rtol,
        )
    )
    assert_true(
        isclose(
            res2[1].cast[DType.float32](),
            SIMD[DType.float32, 4](-0.0, 2.0, 3.0, 3.0),
            atol=atol,
            rtol=rtol,
        )
    )


fn test_log[type: DType](atol: Float32, rtol: Float32) raises:
    var res0 = log(Scalar[type](123.45))
    assert_true(
        isclose(
            res0.cast[DType.float32](),
            4.8158,
            atol=atol,
            rtol=rtol,
        )
    )

    var res1 = log(Scalar[type](0.1))
    assert_true(
        isclose(
            res1.cast[DType.float32](),
            -2.3025,
            atol=atol,
            rtol=rtol,
        )
    )

    var res2 = log(SIMD[type, 4](1, 2, 4, 5))
    assert_true(
        isclose(
            res2.cast[DType.float32](),
            SIMD[DType.float32, 4](0.0, 0.693147, 1.38629, 1.6094),
            atol=atol,
            rtol=rtol,
        )
    )

    var res3 = log[type, 1](2.7182818284590452353602874713526624977572)
    assert_true(
        isclose(
            res3.cast[DType.float32](),
            1.0,
            atol=atol,
            rtol=rtol,
        )
    )

    var res4 = isinf(log(SIMD[type, 4](0, 1, 0, 0)))
    assert_equal(res4, SIMD[DType.bool, 4](True, False, True, True))


fn test_log2[type: DType](atol: Float32, rtol: Float32) raises:
    var res0 = log2(Scalar[type](123.45))
    assert_true(
        isclose(
            res0.cast[DType.float32](),
            6.9477,
            atol=atol,
            rtol=rtol,
        )
    )

    var res1 = log2(Scalar[type](0.1))
    assert_true(
        isclose(
            res1.cast[DType.float32](),
            -3.3219,
            atol=atol,
            rtol=rtol,
        )
    )

    var res2 = log2(SIMD[type, 4](1, 2, 4, 5))
    assert_true(
        isclose(
            res2.cast[DType.float32](),
            SIMD[DType.float32, 4](0.0, 1.0, 2.0, 2.3219),
            atol=atol,
            rtol=rtol,
        )
    )


fn main() raises:
    var f32_atol = 1e-4
    var f32_rtol = 1e-5
    test_frexp[DType.float32](f32_atol, f32_rtol)
    test_log[DType.float32](f32_atol, f32_rtol)
    test_log2[DType.float32](f32_atol, f32_rtol)

    var f16_atol = 1e-2
    var f16_rtol = 1e-5
    test_frexp[DType.float16](f16_atol, f16_rtol)
    test_log[DType.float16](f16_atol, f16_rtol)
    test_log2[DType.float16](f16_atol, f16_rtol)

    @parameter
    if not has_neon():
        var bf16_atol = 1e-1
        var bf16_rtol = 1e-5
        test_frexp[DType.bfloat16](bf16_atol, bf16_rtol)
        test_log[DType.bfloat16](bf16_atol, bf16_rtol)
        test_log2[DType.bfloat16](bf16_atol, bf16_rtol)
