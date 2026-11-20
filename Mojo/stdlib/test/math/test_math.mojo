# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from std.math import (
    acos,
    align_down,
    align_up,
    asin,
    atanh,
    cbrt,
    ceil,
    ceildiv,
    clamp,
    comb,
    copysign,
    cos,
    cosh,
    erf,
    erfc,
    expm1,
    exp2,
    factorial,
    floor,
    fma,
    frexp,
    gcd,
    iota,
    isclose,
    lcm,
    log,
    log1p,
    log2,
    perm,
    pi,
    rsqrt,
    sin,
    sinh,
    sqrt,
    trunc,
    ulp,
)
from std.math.math import _call_libm
from std.memory import bitcast
from std.sys import CompilationTarget
from std.builtin.dtype import _integral_type_of
from std.math.math import (
    _call_libm,
    _frexp_bits,
    _is_subnormal_bits,
    _is_zero_bits,
    _log_accurate,
    _narrow_f32_to_bf16_bits,
    _scale_down_bits,
    _scale_up_subnormal,
)
from std.memory import bitcast
from std.sys import bit_width_of, CompilationTarget

from std.testing import TestSuite
from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_true,
)

from std.utils.numerics import (
    FPUtils,
    inf,
    isinf,
    isnan,
    nan,
    neg_inf,
    nextafter,
)

from test_utils.math import SplitMix64


def test_sin() raises:
    assert_almost_equal(sin(Float32(1.0)), 0.841470956802)

    comptime s_45 = sin(pi / 4)
    assert_almost_equal(s_45, 0.7071067811865475)

    comptime s_30 = sin(pi / 6)
    assert_almost_equal(s_30, 0.5)

    comptime s_60 = sin(pi / 3)
    assert_almost_equal(s_60, 0.8660254037844387)

    # Compare the compile time values against the runtime values to make sure
    # they align.
    assert_almost_equal(s_45, sin(pi / 4))
    assert_almost_equal(s_30, sin(pi / 6))
    assert_almost_equal(s_60, sin(pi / 3))


def test_cos() raises:
    assert_almost_equal(cos(Float32(1.0)), 0.540302276611)

    assert_equal(cos(BFloat16(2.0)), -0.416015625)

    comptime c_45 = cos(pi / 4)
    assert_almost_equal(c_45, 0.7071067811865476)

    comptime c_30 = cos(pi / 6)
    assert_almost_equal(c_30, 0.8660254037844386)

    comptime c_60 = cos(pi / 3)
    assert_almost_equal(c_60, 0.4999999999999999)

    # Compare the compile time values against the runtime values to make sure
    # they align.
    assert_almost_equal(c_45, cos(pi / 4))
    assert_almost_equal(c_30, cos(pi / 6))
    assert_almost_equal(c_60, cos(pi / 3))


def test_factorial() raises:
    assert_equal(factorial(0), 1)
    assert_equal(factorial(1), 1)
    assert_equal(factorial(15), 1307674368000)
    assert_equal(factorial(20), 2432902008176640000)


def test_comb() raises:
    assert_equal(comb(0, 0), 1)
    assert_equal(comb(5, 0), 1)
    assert_equal(comb(5, 5), 1)
    assert_equal(comb(5, 1), 5)
    assert_equal(comb(5, 2), 10)
    assert_equal(comb(10, 3), 120)
    assert_equal(comb(3, 5), 0)  # k > n returns 0
    # Symmetry: C(n, k) == C(n, n-k)
    assert_equal(comb(10, 4), comb(10, 6))


def test_perm() raises:
    assert_equal(perm(5, 0), 1)
    assert_equal(perm(5, 1), 5)
    assert_equal(perm(5, 2), 20)
    assert_equal(perm(5, 5), 120)
    # perm(n) with default k=-1 delegates to factorial(n)
    assert_equal(perm(5), factorial(5))
    assert_equal(perm(0), 1)
    assert_equal(perm(10, 3), 720)


def test_copysign() raises:
    var x = Int32(2)
    assert_equal(x, copysign(x, x))
    assert_equal(-x, copysign(x, -x))
    assert_equal(x, copysign(-x, x))
    assert_equal(-x, copysign(-x, -x))

    assert_equal(Float32(1.0), copysign(Float32(1.0), Float32(2.0)))
    assert_equal(Float32(-1.0), copysign(Float32(1.0), Float32(-2.0)))
    assert_equal(neg_inf[.float32](), copysign(inf[.float32](), -2.0))
    assert_equal(
        String(-nan[.float32]()),
        String(copysign(nan[.float32](), -2.0)),
    )

    # Test some cases with 0 and signed zero
    assert_equal(Float32(1.0), copysign(Float32(1.0), Float32(0.0)))
    assert_equal(Float32(0.0), copysign(Float32(0.0), Float32(1.0)))
    assert_equal(Float32(0.0), copysign(Float32(-0.0), Float32(1.0)))
    assert_equal(Float32(-0.0), copysign(Float32(0.0), Float32(-1.0)))

    # TODO: Add some test cases for SIMD vector with width > 1


def _test_isclose_numerics[*, symm: Bool]() raises:
    comptime dtype = DType.float64
    comptime T = SIMD[dtype, 2]

    comptime atol = 1e-8
    comptime rtol = 1e-5

    comptime inf_ = inf[dtype]()
    comptime nan_ = nan[dtype]()
    comptime v = T(0.1, 0.2)

    def edge_val[symm: Bool](a: T, atol: T, rtol: T) -> T:
        """Creates a value at the tolerance boundary that should be considered close to `a`.
        """
        assert all(a.ge(0))

        comptime if symm:
            # |a - b| ≤ max(atol, rtol * max(|a|, |b|))
            return a - max(atol, rtol * a)
        else:
            # |a - b| ≤ atol + rtol * |b|
            return a + atol + rtol * a

    var all_close: List[Tuple[T, T]] = [
        (T(1, 0), T(1, 0)),
        (T(atol), T(0)),
        (T(inf_, -inf_), T(inf_, -inf_)),
        (T(1), edge_val[symm](1, atol, 0)),
        (edge_val[symm](v, atol, 0), v),
    ]

    comptime if not symm:
        all_close += [
            (edge_val[symm](v, 0, rtol), v),
            (edge_val[symm](v, atol, rtol), v),
        ]

    for item in all_close:
        var a, b = item
        var res = isclose[symmetrical=symm](a, b, atol=atol, rtol=rtol)
        assert_true(all(res))

    var none_close: List[Tuple[T, T]] = [
        (T(inf_, 0), T(1, inf_)),
        (T(inf_, -inf_), T(1, 0)),
        (T(inf_, inf_), T(1, -inf_)),
        (T(inf_, inf_), T(1, 0)),
        (T(nan_, 0), T(nan_, -inf_)),
    ]

    comptime if symm:
        none_close += [(v, v + atol + rtol)]
    else:
        none_close += [
            (T(0), edge_val[symm](0, 2.0 * atol, 0)),
            (T(1), edge_val[symm](1, 2.0 * atol, rtol)),
            (T(1), edge_val[symm](1, atol, 2.0 * rtol)),
            (v, edge_val[symm](v, atol, 2.0 * rtol)),
            (v, edge_val[symm](v, 1.1 * atol, 1.1 * rtol)),
        ]

    for item in none_close:
        var a, b = item
        var res = isclose[symmetrical=symm](a, b, atol=atol, rtol=rtol)
        assert_false(any(res))


def test_isclose() raises:
    # floating-point
    comptime dtype = DType.float32
    comptime S = Scalar[dtype]
    comptime T = SIMD[dtype, 4]
    comptime nan_ = nan[dtype]()

    assert_true(isclose(S(2), S(2)))
    assert_true(isclose(S(2), S(2), rtol=1e-9))
    assert_true(isclose(S(2), S(2.00001), rtol=1e-3))
    assert_true(isclose(nan_, nan_, equal_nan=True))

    assert_true(
        all(isclose(T(1, 2, 3, nan_), T(1, 2, 3, nan_), equal_nan=True))
    )

    assert_false(
        all(isclose(T(1, 2, nan_, 3), T(1, 2, nan_, 4), equal_nan=True))
    )

    _test_isclose_numerics[symm=False]()
    _test_isclose_numerics[symm=True]()


def test_ceil() raises:
    # We just test that the `ceil` function resolves correctly for a few common
    # types. Types should test their own `__ceil__` implementation explicitly.
    assert_equal(ceil(0), 0)
    assert_equal(ceil(Int(5)), 5)
    assert_equal(ceil(1.5), 2.0)
    assert_equal(ceil(Float32(1.4)), 2.0)
    assert_equal(ceil(Float64(-3.6)), -3.0)


def test_floor() raises:
    # We just test that the `floor` function resolves correctly for a few common
    # types. Types should test their own `__floor__` implementation explicitly.
    assert_equal(floor(0), 0)
    assert_equal(floor(Int(5)), 5)
    assert_equal(floor(1.5), 1.0)
    assert_equal(floor(Float32(1.6)), 1.0)
    assert_equal(floor(Float64(-3.4)), -4.0)


def test_trunc() raises:
    # We just test that the `trunc` function resolves correctly for a few common
    # types. Types should test their own `__trunc__` implementation explicitly.
    assert_equal(trunc(0), 0)
    assert_equal(trunc(Int(5)), 5)
    assert_equal(trunc(1.5), 1.0)
    assert_equal(trunc(Float32(1.6)), 1.0)
    assert_equal(trunc(Float64(-3.4)), -3.0)


def test_exp2() raises:
    assert_equal(exp2(Float32(1)), 2.0)
    assert_almost_equal(exp2(Float32(0.2)), 1.148696)
    assert_equal(exp2(Float32(0)), 1.0)
    assert_equal(exp2(Float32(-1)), 0.5)
    assert_equal(exp2(Float32(2)), 4.0)
    assert_equal(exp2(Float32(-125)), 2.3509887e-38)
    assert_equal(exp2(Float32(125)), 4.2535296e37)

    assert_equal(exp2(Float64(1)), 2.0)
    assert_almost_equal(exp2(Float64(0.2)), 1.148696)
    assert_equal(exp2(Float64(0)), 1.0)
    assert_equal(exp2(Float64(-1)), 0.5)
    assert_equal(exp2(Float64(2)), 4.0)
    assert_equal(exp2(Float64(-127)), 5.877471754111438e-39)
    assert_equal(exp2(Float64(127)), 1.7014118346046923e38)
    assert_equal(exp2(Float64(-1023)), 1.1125369292536007e-308)
    assert_equal(exp2(Float64(1023)), 8.98846567431158e307)
    assert_equal(exp2(Float64(-1022)), 2.2250738585072014e-308)


def test_iota() raises:
    comptime length = 103
    var offset = 2

    var vector = List[Int32](unsafe_uninit_length=length)

    var buff = vector.unsafe_ptr()
    iota(buff, length, offset)
    for i in range(length):
        assert_equal(vector[i], Int32(offset + i))

    iota(vector, offset)
    for i in range(length):
        assert_equal(vector[i], Int32(offset + i))

    var vector2 = List[Int](unsafe_uninit_length=length)

    iota(vector2, offset)
    for i in range(length):
        assert_equal(vector2[i], offset + i)


comptime F32x4 = SIMD[.float32, 4]
comptime F64x4 = SIMD[.float64, 4]


def test_sqrt() raises:
    assert_equal(sqrt(-1), 0)
    assert_equal(sqrt(0), 0)
    assert_equal(sqrt(1), 1)
    assert_equal(sqrt(63), 7)
    assert_equal(sqrt(64), 8)
    assert_equal(sqrt(2**34 - 1), 2**17 - 1)
    assert_equal(sqrt(2**34), 2**17)
    assert_equal(sqrt(10**16), 10**8)
    assert_equal(sqrt(Int.MAX), 3037000499)

    var i = SIMD[.int, 4](0, 1, 2, 3)
    assert_equal(sqrt(i**2), i)

    var f32x4 = 0.5 * F32x4(0.0, 1.0, 2.0, 3.0)

    var s1_f32 = sqrt(f32x4)
    assert_equal(s1_f32[0], 0.0)
    assert_almost_equal(s1_f32[1], 0.70710)
    assert_equal(s1_f32[2], 1.0)
    assert_almost_equal(s1_f32[3], 1.22474)

    var s2_f32 = sqrt(0.5 * f32x4)
    assert_equal(s2_f32[0], 0.0)
    assert_equal(s2_f32[1], 0.5)
    assert_almost_equal(s2_f32[2], 0.70710)
    assert_almost_equal(s2_f32[3], 0.86602)

    var f64x4 = 0.5 * F64x4(0.0, 1.0, 2.0, 3.0)

    var s1_f64 = sqrt(f64x4)
    assert_equal(s1_f64[0], 0.0)
    assert_almost_equal(s1_f64[1], 0.70710)
    assert_equal(s1_f64[2], 1.0)
    assert_almost_equal(s1_f64[3], 1.22474)

    var s2_f64 = sqrt(0.5 * f64x4)
    assert_equal(s2_f64[0], 0.0)
    assert_equal(s2_f64[1], 0.5)
    assert_almost_equal(s2_f64[2], 0.70710)
    assert_almost_equal(s2_f64[3], 0.86602)


def test_rsqrt() raises:
    var f32x4 = 0.5 * F32x4(0.0, 1.0, 2.0, 3.0) + 1

    var s1_f32 = rsqrt(f32x4)
    assert_equal(s1_f32[0], 1.0)
    assert_almost_equal(s1_f32[1], 0.81649)
    assert_almost_equal(s1_f32[2], 0.70710)
    assert_almost_equal(s1_f32[3], 0.63245)

    var s2_f32 = rsqrt(0.5 * f32x4)
    assert_almost_equal(s2_f32[0], 1.41421)
    assert_almost_equal(s2_f32[1], 1.15470)
    assert_equal(s2_f32[2], 1.0)
    assert_almost_equal(s2_f32[3], 0.89442)

    var f64x4 = 0.5 * F64x4(0.0, 1.0, 2.0, 3.0) + 1

    var s1_f64 = rsqrt(f64x4)
    assert_equal(s1_f64[0], 1.0)
    assert_almost_equal(s1_f64[1], 0.81649)
    assert_almost_equal(s1_f64[2], 0.70710)
    assert_almost_equal(s1_f64[3], 0.63245)

    var s2_f64 = rsqrt(0.5 * f64x4)
    assert_almost_equal(s2_f64[0], 1.41421)
    assert_almost_equal(s2_f64[1], 1.15470)
    assert_equal(s2_f64[2], 1.0)
    assert_almost_equal(s2_f64[3], 0.89442)


def _test_sqrt_rsqrt_comptime[
    dtype: DType
]() raises where dtype.is_floating_point():
    comptime x = SIMD[dtype, 4](0.5, 2, 3, 100)
    comptime sqrt_x = sqrt(x)
    comptime rsqrt_x = rsqrt(x)
    assert_equal(sqrt_x, sqrt(x))
    assert_equal(rsqrt_x, rsqrt(x))


def test_sqrt_rsqrt_comptime() raises:
    _test_sqrt_rsqrt_comptime[.float16]()
    _test_sqrt_rsqrt_comptime[.bfloat16]()
    _test_sqrt_rsqrt_comptime[.float32]()
    _test_sqrt_rsqrt_comptime[.float64]()


def _test_sin_cos_comptime[
    dtype: DType
]() raises where dtype.is_floating_point():
    # Compile time folds 16-bit inputs directly, while runtime computes them in
    # float32 and rounds, so the two can differ by an ulp.
    comptime x = SIMD[dtype, 4](0.5, 2, 3, 100)
    comptime sin_x = sin(x)
    comptime cos_x = cos(x)
    assert_almost_equal(sin_x, sin(x), atol=0, rtol=1e-2)
    assert_almost_equal(cos_x, cos(x), atol=0, rtol=1e-2)


def test_sin_cos_comptime_16bit() raises:
    _test_sin_cos_comptime[.float16]()
    _test_sin_cos_comptime[.bfloat16]()


def _test_frexp_impl[
    dtype: DType
](*, atol: Float64, rtol: Float64) raises where dtype.is_floating_point():
    var res0 = frexp(Scalar[dtype](123.45))
    assert_almost_equal(
        res0[0].cast[.float32](), 0.964453, atol=atol, rtol=rtol
    )
    assert_almost_equal(res0[1].cast[.float32](), 7.0, atol=atol, rtol=rtol)

    var res1 = frexp(Scalar[dtype](0.1))
    assert_almost_equal(res1[0].cast[.float32](), 0.8, atol=atol, rtol=rtol)
    assert_almost_equal(res1[1].cast[.float32](), -3.0, atol=atol, rtol=rtol)

    var res2 = frexp(Scalar[dtype](-0.1))
    assert_almost_equal(res2[0].cast[.float32](), -0.8, atol=atol, rtol=rtol)
    assert_almost_equal(res2[1].cast[.float32](), -3.0, atol=atol, rtol=rtol)

    # not representable in bfloat16
    comptime if dtype in (DType.float32, DType.float64):
        var res3 = frexp(Scalar[dtype](7.346839692639297e-40))
        assert_almost_equal(res3[0].cast[.float32](), 0.5)
        assert_almost_equal(res3[1].cast[.float32](), -129)

    var res4 = frexp(SIMD[dtype, 4](0, 2, 4, 5))
    assert_almost_equal(
        res4[0].cast[.float32](),
        SIMD[.float32, 4](0.0, 0.5, 0.5, 0.625),
        atol=atol,
        rtol=rtol,
    )
    assert_almost_equal(
        res4[1].cast[.float32](),
        SIMD[.float32, 4](-0.0, 2.0, 3.0, 3.0),
        atol=atol,
        rtol=rtol,
    )


def _test_log_impl[
    dtype: DType
](*, atol: Float64, rtol: Float64) raises where dtype.is_floating_point():
    var res0 = log(Scalar[dtype](123.45))
    assert_almost_equal(res0.cast[.float32](), 4.8158, atol=atol, rtol=rtol)

    var res1 = log(Scalar[dtype](0.1))
    assert_almost_equal(res1.cast[.float32](), -2.3025, atol=atol, rtol=rtol)

    var res2 = log(SIMD[dtype, 4](1, 2, 4, 5))
    assert_almost_equal(
        res2.cast[.float32](),
        SIMD[.float32, 4](0.0, 0.693147, 1.38629, 1.6094),
        atol=atol,
        rtol=rtol,
    )

    var res3 = log(Scalar[dtype](2.7182818284590452353602874713526624977572))
    assert_almost_equal(res3.cast[.float32](), 1.0, atol=atol, rtol=rtol)

    var res4 = isinf(log(SIMD[dtype, 4](0, 1, 0, 0)))
    assert_equal(res4, SIMD[.bool, 4](True, False, True, True))


def _test_log2_impl[
    dtype: DType
](*, atol: Float64, rtol: Float64) raises where dtype.is_floating_point():
    var res0 = log2(Scalar[dtype](123.45))
    assert_almost_equal(res0.cast[.float32](), 6.9477, atol=atol, rtol=rtol)

    var res1 = log2(Scalar[dtype](0.1))
    assert_almost_equal(res1.cast[.float32](), -3.3219, atol=atol, rtol=rtol)

    var res2 = log2(SIMD[dtype, 4](1, 2, 4, 5))
    assert_almost_equal(
        res2.cast[.float32](),
        SIMD[.float32, 4](0.0, 1.0, 2.0, 2.3219),
        atol=atol,
        rtol=rtol,
    )


def _test_log1p_impl[
    dtype: DType
](*, atol: Float64, rtol: Float64) raises where dtype.is_floating_point():
    var res0 = log1p(Scalar[dtype](123.45))
    assert_almost_equal(res0.cast[.float32](), 4.8239, atol=atol, rtol=rtol)

    var res1 = log1p(Scalar[dtype](0.1))
    assert_almost_equal(res1.cast[.float32](), 0.0953102, atol=atol, rtol=rtol)

    var res2 = log1p(SIMD[dtype, 4](1, 2, 4, 5))
    assert_almost_equal(
        res2.cast[.float32](),
        SIMD[.float32, 4](0.693147, 1.09861, 1.60944, 1.79176),
        atol=atol,
        rtol=rtol,
    )

    var res3 = log1p(SIMD[dtype, 4](0.00001, 0.000002, 0.000004, 0.00005))
    assert_almost_equal(
        res3.cast[.float32](),
        SIMD[.float32, 4](9.99995e-6, 2.0e-6, 3.99999e-6, 0.0000499988),
        atol=atol,
        rtol=rtol,
    )

    var res4 = log1p(SIMD[dtype, 4](0.707107, 0.807107, 0.9, 1))
    assert_almost_equal(
        res4.cast[.float32](),
        SIMD[.float32, 4](0.5348, 0.591727, 0.641854, 0.693147),
        atol=atol,
        rtol=rtol,
    )


def test_frexp() raises:
    _test_frexp_impl[.float64](atol=1e-4, rtol=1e-5)
    _test_frexp_impl[.float32](atol=1e-4, rtol=1e-5)
    _test_frexp_impl[.float16](atol=1e-2, rtol=1e-5)

    _test_frexp_impl[.bfloat16](atol=1e-1, rtol=1e-5)

    var res1 = frexp(Float64(5e-324))
    assert_equal(res1[0], 0.5)
    assert_equal(res1[1], -1073)

    var res2 = frexp(Float64(1.5e-310))
    assert_almost_equal(res2[0], 0.8628927047339232)
    assert_equal(res2[1], -1029)


def _test_frexp_bits_impl[
    dtype: DType
]() raises where dtype.is_floating_point():
    # `_frexp_bits` reads subnormal exponents from the bit pattern instead of
    # scaling the value, so it must agree with `frexp` bit for bit on the
    # smallest and largest subnormals, the smallest normal, and ordinary
    # values, as scalars and as vectors.
    comptime T = Scalar[dtype]
    comptime min_normal = T(
        1.17549435e-38 if dtype == .float32 else 2.2250738585072014e-308
    )
    var min_sub = nextafter(T(0), T(1))
    var xs: List[T] = [
        min_sub,
        -min_sub,
        min_sub * 12345,
        nextafter(min_normal, T(0)),
        min_normal,
        1.0,
        0.75,
        -123.45,
        1e30,
    ]
    for x in xs:
        var got = _frexp_bits(x)
        var want = frexp(x)
        assert_equal(
            got[0].to_bits(), want[0].to_bits(), String(t"frac of {x}")
        )
        assert_equal(got[1], want[1], String(t"exponent of {x}"))
    var v = SIMD[dtype, 4](min_sub, 1.0, -123.45, min_normal)
    var got4 = _frexp_bits(v)
    var want4 = frexp(v)
    assert_equal(got4[0], want4[0])
    assert_equal(got4[1], want4[1])


def test_narrow_f32_to_bf16_bits() raises:
    # Must agree with the (round-to-nearest-even) `cast`. Every high half
    # covers each sign, exponent, NaN and kept mantissa; the low halves sit
    # at, just below and just above a rounding tie. NaN payloads differ
    # between targets' casts, so a NaN only has to stay a quiet NaN with
    # its sign.
    var lows: List[UInt32] = [
        0x0000,
        0x0001,
        0x1234,
        0x4000,
        0x7FFF,
        0x8000,
        0x8001,
        0xBFFF,
        0xFFFF,
    ]
    for hi in range(0x10000):
        for lo in lows:
            var bits = (UInt32(hi) << 16) | lo
            var x = Float32(from_bits=bits)
            var got = _narrow_f32_to_bf16_bits(x)
            var want = x.cast[.bfloat16]()
            if isnan(x):
                var got_bits = got.to_bits[.uint16]()
                if got_bits & 0xFFC0 != UInt16(hi & 0x8000) | 0x7FC0:
                    raise Error(
                        t"narrow of NaN bits {bits}: got bits {got_bits}"
                    )
            elif got.to_bits() != want.to_bits():
                raise Error(t"narrow of bits {bits}: got {got}, want {want}")
    var v = SIMD[.float32, 4](
        1.0, bitcast[.float32](UInt32(0x00008001)), -2.5, 0.0
    )
    assert_equal(_narrow_f32_to_bf16_bits(v), v.cast[.bfloat16]())


def _check_scale_down[shift: Int]() raises:
    # Exponents near `1 + shift` give subnormal results, smaller ones round
    # to zero and larger ones stay normal. Clearing the bits below a set bit
    # `k` makes some shifts discard exactly half an ULP.
    comptime scale = Float64(from_bits=UInt64(1023 - shift) << 52)
    var rng = SplitMix64(UInt64(shift))
    for e in range(1, 255):
        var r = (rng.next() & 0x7FFFFF).cast[.uint32]()
        var mantissas: List[UInt32] = [0, 0x7FFFFF, r]
        for k in range(1, 24):
            mantissas.append(
                (r >> UInt32(k) << UInt32(k)) | (UInt32(1) << UInt32(k - 1))
            )
        for m in mantissas:
            for sign in [UInt32(0), UInt32(0x80000000)]:
                var y = Float32(from_bits=sign | (UInt32(e) << 23) | m)
                var got = _scale_down_bits[shift](y)
                var want = (Float64(y) * scale).cast[.float32]()
                if got.to_bits() != want.to_bits():
                    raise Error(
                        t"_scale_down_bits[{shift}]({y}) = {got}, want {want}"
                    )
    comptime if shift >= 23:
        # Zero stays zero once the shift drops the implicit bit. `hypot(0, 0)`
        # and `scalb(0, n)` pass it in on targets that flush subnormals.
        for sign in [UInt32(0), UInt32(0x80000000)]:
            var z = Float32(from_bits=sign)
            assert_equal(_scale_down_bits[shift](z).to_bits(), z.to_bits())


def test_scale_down_bits() raises:
    _check_scale_down[1]()
    _check_scale_down[23]()
    _check_scale_down[24]()
    _check_scale_down[25]()
    _check_scale_down[46]()
    _check_scale_down[64]()
    _check_scale_down[100]()
    _check_scale_down[126]()
    _check_scale_down[130]()


def _check_scale_up[shift: Int]() raises:
    var mantissas: List[UInt32] = [
        1,
        2,
        3,
        0x12345,
        0x35555,
        0x3FFFFF,
        0x400000,
        0x400001,
        0x7FFFFF,
    ]
    var rng = SplitMix64(UInt64(shift))
    for _ in range(512):
        var m = (rng.next() & 0x7FFFFF).cast[.uint32]()
        if m != 0:
            mantissas.append(m)
    comptime scale = Float64(2) ** Float64(shift)
    for m in mantissas:
        for sign in [UInt32(0), UInt32(0x80000000)]:
            var x = Float32(from_bits=sign | m)
            var got = _scale_up_subnormal[shift](x)
            var want = (Float64(x) * scale).cast[.float32]()
            if got.to_bits() != want.to_bits():
                raise Error(
                    t"_scale_up_subnormal[{shift}]({x}) = {got}, want {want}"
                )


def test_scale_up_subnormal() raises:
    _check_scale_up[23]()
    _check_scale_up[24]()
    _check_scale_up[40]()
    _check_scale_up[46]()
    _check_scale_up[100]()
    for m in range(1, 0x400):
        for sign in [UInt16(0), UInt16(0x8000)]:
            var h = Float16(from_bits=sign | UInt16(m))
            var got = _scale_up_subnormal[10](h)
            var want = (Float64(h) * 1024).cast[.float16]()
            if got.to_bits() != want.to_bits():
                raise Error(
                    t"_scale_up_subnormal[10]({h}) = {got}, want {want}"
                )


def test_frexp_bits() raises:
    _test_frexp_bits_impl[.float32]()
    _test_frexp_bits_impl[.float64]()


def test_log() raises:
    _test_log_impl[.float32](atol=1e-4, rtol=1e-5)
    _test_log_impl[.float16](atol=1e-2, rtol=1e-5)

    _test_log_impl[.bfloat16](atol=1e-1, rtol=1e-5)

    assert_almost_equal(log(5e-324), -744.4400719213812)


def test_log2() raises:
    _test_log2_impl[.float64](atol=1e-4, rtol=1e-5)
    _test_log2_impl[.float32](atol=1e-4, rtol=1e-5)
    _test_log2_impl[.float16](atol=1e-2, rtol=1e-5)

    _test_log2_impl[.bfloat16](atol=1e-1, rtol=1e-5)

    assert_equal(log2(5e-324), -1074)


def test_log1p() raises:
    _test_log1p_impl[.float64](atol=1e-4, rtol=1e-5)
    _test_log1p_impl[.float32](atol=1e-4, rtol=1e-5)
    _test_log1p_impl[.float16](atol=1e-2, rtol=1e-5)

    _test_log1p_impl[.bfloat16](atol=1e-1, rtol=1e-5)


def test_log1p_accuracy() raises:
    # Compare log1p against libm across several regimes.

    # Small values near zero (critical regime where log1p matters).
    var small_vals: Array[Float64, 14] = [
        1e-15,
        1e-12,
        1e-10,
        1e-8,
        1e-6,
        1e-4,
        1e-2,
        -1e-15,
        -1e-12,
        -1e-10,
        -1e-8,
        -1e-6,
        -1e-4,
        -1e-2,
    ]
    for val in small_vals:
        assert_almost_equal(
            log1p(val),
            _call_libm["log1p"](val),
            msg=String("f64 small mismatch for the value = ", val),
        )

    # Float32 small values.
    var small_vals_f32: Array[Float32, 6] = [
        1e-7,
        1e-5,
        1e-3,
        -1e-7,
        -1e-5,
        -1e-3,
    ]
    for val in small_vals_f32:
        assert_almost_equal(
            log1p(val),
            _call_libm["log1p"](val),
            msg=String("f32 small mismatch for the value = ", val),
        )

    # Moderate values.
    comptime n = 1_000.0
    for i in range(Float64(0.0), n, Float64(1.0)):
        var val = Float64(i) / (n / 10.5) - 0.5
        assert_almost_equal(
            log1p(val),
            _call_libm["log1p"](val),
            msg=String("f64 moderate mismatch for the value = ", val),
        )

    # Values near the singularity at x = -1.
    var near_neg1: Array[Float64, 5] = [
        -0.9,
        -0.99,
        -0.999,
        -0.9999,
        -0.99999,
    ]
    for val in near_neg1:
        assert_almost_equal(
            log1p(val),
            _call_libm["log1p"](val),
            msg=String("f64 near-singularity mismatch for the value = ", val),
        )

    # Edge cases.
    assert_true(isnan(log1p(nan[.float64]())))
    assert_equal(log1p(Float64(0)), Float64(0))
    assert_true(isinf(log1p(Float64(-1))))
    assert_true(isnan(log1p(Float64(-2))))


def _scaled[
    dtype: DType
](m: Scalar[dtype], k: Int) -> Scalar[dtype] where dtype.is_floating_point():
    """Returns `m * 2**k` for `m` in [1, 2), rounding into the subnormals."""
    comptime T = Scalar[dtype]
    comptime U = _integral_type_of[dtype]()
    comptime mant = FPUtils[dtype].mantissa_width()
    comptime bias = FPUtils[dtype].exponent_bias()
    var lo = max(k, 1 - bias)
    var p = T(from_bits=Scalar[U](lo + bias) << Scalar[U](mant))
    var tail = T(from_bits=Scalar[U](bias + k - lo) << Scalar[U](mant))
    return m * p * tail


def _log_ulp_diff[
    dtype: DType
](
    got: Scalar[dtype], want: Scalar[dtype]
) -> Int where dtype.is_floating_point():
    if isnan(got) or isnan(want):
        return 0 if isnan(got) and isnan(want) else Int.MAX
    if got == want:
        return 0
    if isinf(got) or isinf(want):
        return Int.MAX
    comptime sign_shift = UInt64(bit_width_of[dtype]() - 1)
    var got_bits = got.to_bits().cast[.uint64]()
    var want_bits = want.to_bits().cast[.uint64]()
    if (
        got != 0
        and want != 0
        and ((got_bits >> sign_shift) != (want_bits >> sign_shift))
    ):
        return Int.MAX
    # Same-sign finite magnitudes differ by less than Int.MAX, even for f64.
    comptime magnitude_mask = (UInt64(1) << sign_shift) - 1
    var got_magnitude = got_bits & magnitude_mask
    var want_magnitude = want_bits & magnitude_mask
    return Int(
        max(got_magnitude, want_magnitude) - min(got_magnitude, want_magnitude)
    )


def _test_log_ulp_comparator[
    dtype: DType
]() raises where dtype.is_floating_point():
    comptime T = Scalar[dtype]
    assert_equal(_log_ulp_diff(nan[dtype](), nan[dtype]()), 0)
    assert_equal(_log_ulp_diff(nan[dtype](), T(1)), Int.MAX)
    assert_equal(_log_ulp_diff(T(1), nan[dtype]()), Int.MAX)
    assert_equal(_log_ulp_diff(inf[dtype](), inf[dtype]()), 0)
    assert_equal(_log_ulp_diff(-inf[dtype](), -inf[dtype]()), 0)
    assert_equal(_log_ulp_diff(T.MAX_FINITE, inf[dtype]()), Int.MAX)
    assert_equal(_log_ulp_diff(inf[dtype](), T.MAX_FINITE), Int.MAX)
    assert_equal(_log_ulp_diff(-T.MAX_FINITE, -inf[dtype]()), Int.MAX)
    assert_equal(_log_ulp_diff(inf[dtype](), -inf[dtype]()), Int.MAX)
    assert_equal(_log_ulp_diff(T(-1), T(1)), Int.MAX)
    assert_equal(_log_ulp_diff(T(0), T(-0.0)), 0)
    assert_equal(_log_ulp_diff(T(1), T(1)), 0)
    assert_equal(_log_ulp_diff(T(-1), T(-1)), 0)


def test_log_ulp_comparator() raises:
    _test_log_ulp_comparator[.float64]()
    _test_log_ulp_comparator[.float32]()
    _test_log_ulp_comparator[.float16]()
    _test_log_ulp_comparator[.bfloat16]()
    var wrong_sign = Float64(from_bits=UInt64(0x3FFF5D4DC0C4548D))
    var reference = Float64(from_bits=UInt64(0xC000A2B23F3BAB73))
    assert_equal(_log_ulp_diff(wrong_sign, reference), Int.MAX)
    assert_equal(_log_ulp_diff(reference, wrong_sign), Int.MAX)
    var next_positive = Float64(from_bits=UInt64(0x3FF0000000000001))
    var next_negative = Float64(from_bits=UInt64(0xBFF0000000000001))
    assert_equal(_log_ulp_diff(next_positive, Float64(1)), 1)
    assert_equal(_log_ulp_diff(next_negative, Float64(-1)), 1)
    assert_equal(
        _log_ulp_diff(Float64(0), Float64.MAX_FINITE), 0x7FEFFFFFFFFFFFFF
    )
    var min_subnormal = Float64(from_bits=UInt64(1))
    assert_equal(_log_ulp_diff(min_subnormal, Float64(-0.0)), 1)
    assert_equal(_log_ulp_diff(-min_subnormal, Float64(0)), 1)


def _assert_ulp_vs_libm[
    dtype: DType, //, name: StaticString
](
    x: Scalar[dtype], got: Scalar[dtype], max_ulp: Int
) raises where dtype.is_floating_point():
    var want = _call_libm[name](x)
    var d = _log_ulp_diff(got, want)
    assert_true(
        d <= max_ulp,
        String(t"{name}({x}) = {got}, libm {want} ({d} ulp)"),
    )


def _test_log_special[dtype: DType]() raises where dtype.is_floating_point():
    comptime T = Scalar[dtype]
    assert_equal(log(inf[dtype]()), inf[dtype]())
    assert_equal(log(T(0)), neg_inf[dtype]())
    assert_equal(log(T(-0.0)), neg_inf[dtype]())
    assert_true(isnan(log(T(-1))))
    assert_true(isnan(log(neg_inf[dtype]())))
    assert_true(isnan(log(nan[dtype]())))
    assert_equal(log(T(1)), T(0))
    assert_equal(log2(inf[dtype]()), inf[dtype]())
    assert_equal(log1p(inf[dtype]()), inf[dtype]())


def test_log_special_values() raises:
    _test_log_special[.float64]()
    _test_log_special[.float32]()
    _test_log_special[.float16]()
    _test_log_special[.bfloat16]()


def test_log_float64_ulp() raises:
    # Mantissas near 1, sqrt(2)/2 and sqrt(2) at every 5th binade, which
    # covers subnormals and the reduction boundaries.
    var mantissas: Array[Float64, 6] = [
        1.0,
        1.0000000000000002,
        1.4142135623730951,
        1.4142135623730954,
        1.9999999999999998,
        1.2345678901234567,
    ]
    for k in range(-1074, 1024, 5):
        for m in mantissas:
            var x = _scaled(m, k)
            _assert_ulp_vs_libm["log"](x, log(x), 1)
    comptime n = 20_000
    for i in range(1, n):
        var x = Float64(i) / (n / 4.0)
        _assert_ulp_vs_libm["log"](x, log(x), 1)
        var y = 1.0 + Float64(i - n // 2) * 1e-12
        _assert_ulp_vs_libm["log"](y, log(y), 1)


def _check_bit_classification[
    dtype: DType
]() raises where dtype.is_floating_point():
    comptime T = Scalar[dtype]
    comptime TInt = Scalar[_integral_type_of[dtype]()]
    comptime mw = FPUtils[dtype].mantissa_width()
    comptime sign = TInt.MIN
    for neg in range(2):
        var s = sign if neg == 1 else TInt(0)
        assert_true(_is_zero_bits(T(from_bits=s)))
        assert_false(_is_subnormal_bits(T(from_bits=s)))
        for m in [1, 2, (1 << mw) // 2, (1 << mw) - 1]:
            var v = T(from_bits=s | TInt(m))
            assert_false(_is_zero_bits(v))
            assert_true(_is_subnormal_bits(v))
        for b in [1 << mw, (1 << mw) + 1, 1 << (mw + 2)]:
            var v = T(from_bits=s | TInt(b))
            assert_false(_is_zero_bits(v))
            assert_false(_is_subnormal_bits(v))
    assert_false(_is_zero_bits(inf[dtype]()))
    assert_false(_is_subnormal_bits(nan[dtype]()))


def test_subnormal_bit_classification() raises:
    _check_bit_classification[.float16]()
    _check_bit_classification[.bfloat16]()
    _check_bit_classification[.float32]()
    _check_bit_classification[.float64]()
    var rng = SplitMix64(7)
    for _ in range(512):
        var x = rng.bits[.float32]()
        var magnitude = x.to_bits() & 0x7FFFFFFF
        assert_equal(_is_zero_bits(x), magnitude == 0)
        assert_equal(
            _is_subnormal_bits(x), magnitude != 0 and magnitude < 0x00800000
        )
    var tiny = Float32(from_bits=UInt32(1))
    var min_normal = Float32(from_bits=UInt32(0x00800000))
    var x = SIMD[.float32, 8](
        0, -0.0, tiny, -tiny, 1.1e-38, min_normal, -min_normal, 1
    )
    assert_equal(
        _is_subnormal_bits(x),
        SIMD[.bool, 8](False, False, True, True, True, False, False, False),
    )
    var h = SIMD[.float16, 4](0, 6e-8, 6.2e-5, 1)
    assert_equal(
        _is_subnormal_bits(h), SIMD[.bool, 4](False, True, False, False)
    )


def test_log_subnormal() raises:
    """`_log_reduce` reads a subnormal's exponent from its bits, so the
    smallest and largest subnormals, and the smallest normal, of each dtype
    stay within 1 ulp of libm. The float32 kernel is called directly because
    public float32 `log` does not use it yet."""
    var bits32: List[UInt32] = [1, 0x007FFFFF, 0x00800000]
    for b in bits32:
        var x = Float32(from_bits=b)
        _assert_ulp_vs_libm["log"](x, _log_accurate(x), 1)
    var bits64: List[UInt64] = [1, 0x000FFFFFFFFFFFFF, 0x0010000000000000]
    for b in bits64:
        var x = Float64(from_bits=b)
        _assert_ulp_vs_libm["log"](x, log(x), 1)


def test_gcd() raises:
    var l = [2, 4, 6, 8, 16]
    var il: Array[Int, 5] = [4, 16, 2, 8, 6]
    assert_equal(gcd(Span[Int](il)), 2)
    assert_equal(gcd(2, 4, 6, 8, 16), 2)
    assert_equal(gcd(l), 2)
    assert_equal(gcd(88, 24), 8)
    assert_equal(gcd(0, 0), 0)
    assert_equal(gcd(1, 0), 1)
    assert_equal(gcd(-2, 4), 2)
    assert_equal(gcd(-2, -4), 2)
    assert_equal(gcd(-2, 0), 2)
    assert_equal(gcd(2, -4), 2)
    assert_equal(gcd(24826148, 45296490), 526)
    assert_equal(gcd(0, 9), 9)
    assert_equal(gcd(4, 4), 4)
    assert_equal(gcd(8), 8)
    assert_equal(gcd(), 0)
    assert_equal(gcd(List[Int]()), 0)


def test_lcm() raises:
    assert_equal(lcm(-2, 4), 4)
    assert_equal(lcm(2345, 23452), 54994940)
    var l = [4, 6, 7, 3]
    assert_equal(lcm(Span(l)), 84)
    assert_equal(lcm(l), 84)
    assert_equal(lcm(4, 6, 7, 3), 84)
    assert_equal(lcm(), 1)
    assert_equal(lcm(List[Int]()), 1)
    assert_equal(lcm(0, 4), 0)
    assert_equal(lcm(5, 33), 165)
    assert_equal(lcm(-34, -56, -32), 3808)
    var il: Array[Int, 5] = [4, 16, 2, 8, 6]
    assert_equal(lcm(Span[Int](il)), 48)
    assert_equal(lcm(345, 623, 364, 84, 93), 346475220)
    assert_equal(lcm(0, 0), 0)


def test_ulp() raises:
    assert_true(isnan(ulp(nan[.float32]())))
    assert_true(isinf(ulp(inf[.float32]())))
    assert_true(isinf(ulp(-inf[.float32]())))
    assert_almost_equal(ulp(Float64(0)), 5e-324)
    assert_equal(ulp(Float64.MAX_FINITE), 1.99584030953472e292)
    assert_equal(ulp(Float64(5)), 8.881784197001252e-16)
    assert_equal(ulp(Float64(-5)), 8.881784197001252e-16)


def test_ulp_at_comptime() raises:
    # `ulp` reaches `nextafter`, which used to fail in the comptime interpreter.
    comptime u1 = ulp(Float32(1.0))
    assert_equal(u1, ulp(Float32(1.0)))
    comptime u5 = ulp(Float64(5))
    assert_equal(u5, 8.881784197001252e-16)
    comptime u_max = ulp(Float64.MAX_FINITE)
    assert_equal(u_max, 1.99584030953472e292)


def test_ceildiv() raises:
    # NOTE: these tests are here mostly to ensure the ceildiv method exists.
    # Types that opt in to CeilDivable, should test their own dunder methods for
    # correctness.
    assert_equal(ceildiv(53.6, 1.35), 40.0)

    # Test the IntLiteral overload.
    comptime a: type_of(1) = ceildiv(1, 7)
    assert_equal(a, 1)
    comptime b: type_of(-78) = ceildiv(548, -7)
    assert_equal(b, -78)

    # Test the Int overload.
    assert_equal(ceildiv(Int(1), Int(7)), 1)
    assert_equal(ceildiv(Int(548), Int(-7)), -78)
    assert_equal(ceildiv(Int(-55), Int(8)), -6)
    assert_equal(ceildiv(Int(-55), Int(-8)), 7)

    # Test the UInt overload.
    assert_equal(
        ceildiv(UInt(1), UInt(7)),
        UInt(1),
    )
    assert_equal(
        ceildiv(UInt(546), UInt(7)),
        UInt(78),
    )

    # Test the SIMD overload.
    assert_equal(ceildiv(Float32(5), 2), ceildiv(5, 2))
    assert_equal((UInt32(5) + 1) // 2, ceildiv(5, 2))
    assert_equal(ceildiv(UInt32(5), UInt32(2)), ceildiv(5, 2))


def test_ceildiv_unsigned_overflow() raises:
    # Regression test for https://github.com/modular/modular/issues/6836:
    # `self + denominator - 1` overflows near the unsigned type's max, which
    # used to wrap the result to a small value instead of the true ceiling.
    assert_equal(ceildiv(UInt.MAX, UInt(2)), UInt(9223372036854775808))
    assert_equal(ceildiv(UInt.MAX, UInt(3)), UInt(6148914691236517205))
    assert_equal(ceildiv(UInt.MAX, UInt(1)), UInt.MAX)

    # Exact multiples must not round up.
    assert_equal(ceildiv(UInt(8), UInt(4)), UInt(2))
    assert_equal(ceildiv(UInt(0), UInt(4)), UInt(0))

    # Small values, unaffected by the overflow, stay correct.
    assert_equal(ceildiv(UInt(7), UInt(2)), UInt(4))
    assert_equal(ceildiv(UInt(1), UInt(7)), UInt(1))

    # Signed types never hit the overflowing branch; confirm no regression.
    assert_equal(ceildiv(Int(-55), Int(8)), -6)
    assert_equal(ceildiv(Int(548), Int(-7)), -78)

    # SIMD vector width > 1, near the type max in every lane.
    comptime max8 = UInt8.MAX
    var lanes = SIMD[.uint8, 4](max8, max8 - 1, 8, 7)
    var divisors = SIMD[.uint8, 4](2, 2, 4, 2)
    assert_equal(
        ceildiv(lanes, divisors),
        SIMD[.uint8, 4](128, 127, 2, 4),
    )


def test_align_down() raises:
    assert_equal(align_down(1, 7), 0)
    assert_equal(align_down(548, -7), 553)
    assert_equal(align_down(-548, -7), -546)
    assert_equal(align_down(-548, 7), -553)

    # Test the UInt overload.
    assert_equal(
        align_down(UInt(1), UInt(7)),
        UInt(0),
    )
    assert_equal(
        align_down(UInt(546), UInt(7)),
        UInt(546),
    )

    # Test the SIMD overload with various integer types.
    assert_equal(align_down(UInt32(385), UInt32(64)), UInt32(384))
    assert_equal(align_down(UInt32(512), UInt32(64)), UInt32(512))
    assert_equal(align_down(UInt32(0), UInt32(64)), UInt32(0))
    assert_equal(align_down(UInt16(100), UInt16(32)), UInt16(96))
    assert_equal(align_down(Int32(385), Int32(64)), Int32(384))
    assert_equal(align_down(Int32(-1), Int32(64)), Int32(-64))

    # Test SIMD vector (width > 1).
    assert_equal(
        align_down(
            SIMD[.uint32, 4](385, 512, 63, 0),
            SIMD[.uint32, 4](64, 64, 64, 64),
        ),
        SIMD[.uint32, 4](384, 512, 0, 0),
    )


def test_align_up() raises:
    assert_equal(align_up(1, 7), 7)
    assert_equal(align_up(548, -7), 546)
    assert_equal(align_up(-548, -7), -553)
    assert_equal(align_up(-548, 7), -546)

    # Test the UInt overload.
    assert_equal(
        align_up(UInt(1), UInt(7)),
        UInt(7),
    )
    assert_equal(
        align_up(UInt(546), UInt(7)),
        UInt(546),
    )

    # Test the SIMD overload with various integer types.
    assert_equal(align_up(UInt32(385), UInt32(64)), UInt32(448))
    assert_equal(align_up(UInt32(512), UInt32(64)), UInt32(512))
    assert_equal(align_up(UInt32(0), UInt32(64)), UInt32(0))
    assert_equal(align_up(UInt16(100), UInt16(32)), UInt16(128))
    assert_equal(align_up(Int32(385), Int32(64)), Int32(448))
    assert_equal(align_up(Int32(-1), Int32(64)), Int32(0))

    # Test SIMD vector (width > 1).
    assert_equal(
        align_up(
            SIMD[.uint32, 4](385, 512, 63, 0),
            SIMD[.uint32, 4](64, 64, 64, 64),
        ),
        SIMD[.uint32, 4](448, 512, 64, 0),
    )


def test_clamp() raises:
    assert_equal(clamp(Int(1), 0, 1), 1)
    assert_equal(clamp(Int(2), 0, 1), 1)
    assert_equal(clamp(Int(-2), 0, 1), 0)

    assert_equal(
        clamp(UInt(1), UInt(0), UInt(1)),
        UInt(1),
    )
    assert_equal(
        clamp(UInt(2), UInt(0), UInt(1)),
        UInt(1),
    )
    assert_equal(
        clamp(UInt(1), UInt(2), UInt(4)),
        UInt(2),
    )

    assert_equal(
        clamp(SIMD[.float32, 4](0, 1, 3, 4), 0, 1),
        SIMD[.float32, 4](0, 1, 1, 1),
    )


def test_fma() raises:
    # Test Int overload
    assert_equal(fma(5, 3, 2), 17)  # 5*3 + 2 = 17
    assert_equal(fma(-2, 3, 4), -2)  # -2*3 + 4 = -2
    assert_equal(fma(0, 100, 5), 5)  # 0*100 + 5 = 5

    # Test UInt (uses SIMD overload since UInt = UInt)
    assert_equal(
        fma(UInt(5), UInt(3), UInt(2)),
        UInt(17),
    )
    assert_equal(
        fma(
            UInt(1000000),
            UInt(1000),
            UInt(500),
        ),
        UInt(1000000500),
    )
    assert_equal(
        fma(
            UInt(0),
            UInt(100),
            UInt(5),
        ),
        UInt(5),
    )

    # Test SIMD overload with float
    assert_almost_equal(fma(Float32(2.5), Float32(4.0), Float32(1.5)), 11.5)
    assert_almost_equal(
        fma(
            SIMD[.float32, 4](1, 2, 3, 4),
            SIMD[.float32, 4](2, 2, 2, 2),
            SIMD[.float32, 4](1, 1, 1, 1),
        ),
        SIMD[.float32, 4](3, 5, 7, 9),
    )


def _test_atanh_impl[dtype: DType]() raises where dtype.is_floating_point():
    comptime n = 1 << 16
    comptime max_ulp = Scalar[dtype](2 if dtype == .float32 else 3)
    for i in range(-n + 1, n):
        var x = Scalar[dtype](i) / n
        var want = _call_libm["atanh"](x)
        var got = atanh(x)
        # Written so that a NaN result fails the comparison.
        assert_true(
            abs(got - want) <= max_ulp * ulp(want),
            String(t"atanh[{dtype}]({x}) = {got}, expected {want}"),
        )

    # Closest representable values to the pole, where `1 - |x|` is smallest.
    var near_one = nextafter(Scalar[dtype](1), Scalar[dtype](0))
    for x in [near_one, -near_one]:
        var want = _call_libm["atanh"](x)
        var got = atanh(x)
        assert_true(
            abs(got - want) <= max_ulp * ulp(want),
            String(t"atanh[{dtype}]({x}) = {got}, expected {want}"),
        )


def test_atanh() raises:
    assert_equal(atanh(Float32(1)), inf[.float32]())
    assert_equal(atanh(Float32(-1)), -inf[.float32]())
    assert_true(isnan(atanh(Float32(2))))
    assert_true(isnan(atanh(Float32(-2))))
    assert_almost_equal(
        atanh(SIMD[.float32, 4](0.5, 0.15, 0.9, 0.0)),
        atanh(SIMD[.float64, 4](0.5, 0.15, 0.9, 0.0)).cast[.float32](),
    )

    assert_equal(atanh(Float32(0)), Float32(0), msg="atanh(0)")
    assert_almost_equal(
        atanh(Float32(0.1)), Float32(0.1003353477310756), msg="atanh(0.1)"
    )
    assert_almost_equal(
        atanh(Float32(-0.1)), Float32(-0.1003353477310756), msg="atanh(-0.1)"
    )
    assert_almost_equal(
        atanh(Float32(0.2)), Float32(0.202732554054082), msg="atanh(0.2)"
    )
    assert_almost_equal(
        atanh(Float32(0.3)), Float32(0.3095196042031118), msg="atanh(0.3)"
    )
    assert_almost_equal(
        atanh(Float32(0.4)), Float32(0.4236489301936017), msg="atanh(0.4)"
    )
    assert_almost_equal(
        atanh(Float32(0.5)), Float32(0.54930614433405489), msg="atanh(0.5)"
    )
    assert_almost_equal(
        atanh(Float32(0.6)), Float32(0.6931471805599453), msg="atanh(0.6)"
    )
    assert_almost_equal(
        atanh(Float32(0.7)), Float32(0.8673005276940542), msg="atanh(0.7)"
    )
    assert_almost_equal(
        atanh(Float32(0.8)), Float32(1.0986122886681098), msg="atanh(0.8)"
    )
    assert_almost_equal(
        atanh(Float32(0.9)), Float32(1.4722194895832204), msg="atanh(0.9)"
    )

    assert_almost_equal(
        atanh(Float32(-0.5)), Float32(-0.54930614433405489), msg="atanh(-0.5)"
    )

    assert_almost_equal(
        atanh(Float32(-0.9297103072)),
        Float32(-1.65625),
        msg="atanh(-0.9297103072)",
    )

    _test_atanh_impl[.float32]()
    _test_atanh_impl[.float64]()


def test_atanh_float16() raises:
    var input_bits = SIMD[.uint16, 8](
        0x3C00, 0xBC00, 0x3BFF, 0xBBFF, 0x0000, 0x8000, 0x0001, 0x83FF
    )
    # atanh(1 - 2^-11) = 4.1587609981435849, which rounds to 0x4429.
    var expected_bits = SIMD[.uint16, 8](
        0x7C00, 0xFC00, 0x4429, 0xC429, 0x0000, 0x8000, 0x0001, 0x83FF
    )
    for i in range(8):
        assert_equal(
            atanh(bitcast[.float16](input_bits[i])).to_bits[.uint16](),
            expected_bits[i],
        )
    assert_equal(
        atanh(bitcast[.float16](input_bits)).to_bits[.uint16](), expected_bits
    )

    var tiny_bits = SIMD[.uint16, 8](
        0x0001, 0x8001, 0x0200, 0x8200, 0x03FF, 0x83FF, 0x0400, 0x8400
    )
    for i in range(8):
        assert_equal(
            atanh(bitcast[.float16](tiny_bits[i])).to_bits[.uint16](),
            tiny_bits[i],
        )
    assert_equal(
        atanh(bitcast[.float16](tiny_bits)).to_bits[.uint16](), tiny_bits
    )

    var domain_bits = SIMD[.uint16, 8](
        0x3C01, 0xBC01, 0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x4000, 0xC000
    )
    for i in range(8):
        assert_true(isnan(atanh(bitcast[.float16](domain_bits[i]))))
    assert_true(isnan(atanh(bitcast[.float16](domain_bits))).reduce_and())


def test_sinh() raises:
    comptime n = 1_000
    for i in range(n):
        var val = Float32(i) / (n * 2) - 1
        assert_almost_equal(
            sinh(val),
            _call_libm["sinh"](val),
            msg=String("mismatch for the value = ", val),
        )


def test_cosh() raises:
    comptime n = 1_000
    for i in range(n):
        var val = Float32(i) / (n * 2) - 1
        assert_almost_equal(
            cosh(val),
            _call_libm["cosh"](val),
            msg=String("mismatch for the value = ", val),
        )


def test_expm1() raises:
    comptime n = 1_000
    for i in range(n):
        var val = Float32(i) / (n * 2) - 1
        assert_almost_equal(
            expm1(val),
            _call_libm["expm1"](val),
            msg=String("mismatch for the value = ", val),
        )


def test_asin() raises:
    comptime n = 1_000
    for i in range(n):
        var val = Float32(i) / (n * 2) - 1
        assert_almost_equal(
            asin(val),
            _call_libm["asin"](val),
            msg=String("mismatch for the value = ", val),
        )

    # Float64 accuracy across [-1, 1].
    for i in range(n):
        var val = Float64(i) / (n * 2) - 1
        assert_almost_equal(
            asin(val),
            _call_libm["asin"](val),
            msg=String("f64 mismatch for the value = ", val),
        )

    # Edge cases.
    assert_equal(asin(Float64(0)), Float64(0))
    assert_almost_equal(asin(Float64(1)), pi / 2.0)
    assert_almost_equal(asin(Float64(-1)), -(pi / 2.0))
    assert_true(isnan(asin(nan[.float64]())))


def test_erfc() raises:
    comptime n = 10_000
    for i in range(n):
        var val = Float32(i) / (n * Float32(2) / 10) - 10
        assert_almost_equal(
            erfc(val),
            _call_libm["erfc"](val),
        )


def test_cbrt() raises:
    comptime n = 1_0000
    for i in range(n):
        var val = Float32(i) / (n * Float32(2) / 10) - 10
        assert_almost_equal(
            cbrt(val),
            _call_libm["cbrt"](val),
            msg=String("mismatch for the value = ", val, " at index = ", i),
        )


def test_cbrt_f32_ulp() raises:
    # Every float32 exponent, subnormals included, against the float64 libm
    # root: the float32 path is within 0.505 ulp of the true root.
    comptime n = 1 << 14
    for i in range(n):
        var bits = UInt32(i) * ((0x7F800000 // n) | 1) % 0x7F800000
        var x = bitcast[.float32](bits)
        var got = cbrt(x)
        var want = _call_libm["cbrt"](x.cast[.float64]())
        var err = abs(got.cast[.float64]() - want) / ulp(got).cast[.float64]()
        assert_true(err <= 0.506, msg=String(t"cbrt({x}) = {got}"))
        assert_equal(cbrt(-x), -got)


def test_cbrt_f32_mixed_lanes() raises:
    # 0, -0, inf, -inf, nan, 2^-147, -2^-144, -27 in one vector.
    var x = bitcast[.float32](
        SIMD[.uint32, 8](
            0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 4, 0x80000020, 0
        )
    )
    x[7] = -27.0
    var got = bitcast[.uint32](cbrt(x))
    assert_equal(got[0], 0)
    assert_equal(got[1], 0x80000000)
    assert_equal(got[2], 0x7F800000)
    assert_equal(got[3], 0xFF800000)
    assert_true(isnan(bitcast[.float32](got[4])))
    assert_equal(got[5], (127 - 49) << 23)
    assert_equal(got[6], 0x80000000 | ((127 - 48) << 23))
    assert_equal(bitcast[.float32](got[7]), -3.0)


def test_acos() raises:
    comptime n = 1_000
    for i in range(n):
        var val = Float32(i) / (n * 2) - 1
        assert_almost_equal(
            acos(val),
            _call_libm["acos"](val),
            msg=String("mismatch for the value = ", val),
        )

    # Float64 accuracy across [-1, 1].
    for i in range(n):
        var val = Float64(i) / (n * 2) - 1
        assert_almost_equal(
            acos(val),
            _call_libm["acos"](val),
            msg=String("f64 mismatch for the value = ", val),
        )

    # Edge cases.
    assert_almost_equal(acos(Float64(1)), Float64(0))
    assert_almost_equal(acos(Float64(-1)), pi)
    assert_almost_equal(acos(Float64(0)), pi / 2.0)
    assert_true(isnan(acos(nan[.float64]())))


def _assert_same[
    dtype: DType
](got: Scalar[dtype], want: Scalar[dtype], msg: String) raises:
    """Asserts bitwise equality, so NaN matches NaN and -0 differs from +0."""
    assert_equal(got.to_bits(), want.to_bits(), msg)


def _ulp_diff[
    dtype: DType
](
    got: Scalar[dtype], want: Scalar[dtype]
) -> Int where dtype.is_floating_point():
    """Counts the floats between `got` and `want`. Bit-identical values give
    0; a NaN, an infinity, or differing signs (other than +0 and -0) give
    `Int.MAX`."""
    comptime sign = UInt64(1) << UInt64(bit_width_of[dtype]() - 1)
    var a = got.to_bits().cast[.uint64]()
    var b = want.to_bits().cast[.uint64]()
    if a == b:
        return 0
    if isnan(got) or isnan(want) or isinf(got) or isinf(want):
        return Int.MAX
    var ma = a & (sign - 1)
    var mb = b & (sign - 1)
    if ((a ^ b) & sign) != 0:
        return Int(ma + mb) if ma == 0 or mb == 0 else Int.MAX
    return Int(max(ma, mb) - min(ma, mb))


def test_erf_float64_special_values() raises:
    _assert_same(erf(Float64(0)), Float64(0), "erf(0)")
    _assert_same(erf(Float64(-0.0)), Float64(-0.0), "erf(-0)")
    assert_equal(erf(inf[.float64]()), 1.0)
    assert_equal(erf(neg_inf[.float64]()), -1.0)
    assert_true(isnan(erf(nan[.float64]())))


def test_erf_float64_ulp() raises:
    # Correctly rounded values from mpmath.
    var cases: List[Tuple[Float64, Float64]] = [
        (-0.8358000000000008, -0.7627941782926682),
        (1e-300, 1.1283791670955126e-300),
        (2.5e-09, 2.8209479177387815e-09),
        (0.1, 0.1124629160182849),
        (0.25, 0.27632639016823696),
        (0.5, 0.5204998778130465),
        (0.84375, 0.7672256612323416),
        (0.9, 0.7969082124228322),
        (1.0, 0.8427007929497149),
        (1.2, 0.9103139782296353),
        (1.25, 0.9229001282564583),
        (1.5, 0.9661051464753108),
        (2.0, 0.9953222650189527),
        (2.857142857142857, 0.9999466876886117),
        (3.0, 0.9999779095030014),
        (4.5, 0.9999999998033839),
        (5.9, 0.9999999999999999),
        (-0.3, -0.3286267594591274),
        (-1.1, -0.8802050695740817),
        (-2.2, -0.9981371537020182),
    ]
    for c in cases:
        var got = erf(c[0])
        assert_true(
            _ulp_diff(got, c[1]) <= 1,
            String("erf(", c[0], ") = ", got, ", expected ", c[1]),
        )

    # macOS libm erf is off by up to about 3.2 ulp, so the libm comparison
    # only bounds gross errors; the table above checks the 1 ulp target.
    comptime n = 20_000
    for i in range(n):
        var x = Float64(i) / (n / 14.0) - 7.0
        _assert_ulp_vs_libm["erf"](x, erf(x), 4)
    for k in range(-1074, 0, 3):
        var t = _scaled(Float64(1.2345678901234567), k)
        _assert_ulp_vs_libm["erf"](t, erf(t), 4)
        _assert_ulp_vs_libm["erf"](-t, erf(-t), 4)
    # Interval boundaries of the fdlibm approximations.
    var edges: Array[Float64, 8] = [
        0.84375,
        0.8437499999999999,
        1.25,
        1.2499999999999998,
        2.857142857142857,
        5.999999999999999,
        6.0,
        27.9,
    ]
    for x in edges:
        _assert_ulp_vs_libm["erf"](x, erf(x), 4)
        _assert_ulp_vs_libm["erf"](-x, erf(-x), 4)


def test_erf_plus_erfc() raises:
    # erf and erfc share their polynomials, so they sum to 1 to within a
    # couple of rounding errors.
    comptime n = 10_000
    for i in range(n):
        var x = Float64(i) / (n / 6.0) - 3.0
        assert_almost_equal(
            erf(x) + erfc(x),
            1.0,
            atol=4.5e-16,
            rtol=0,
            msg=String("erf + erfc at ", x),
        )


def test_erfc_subnormal_results() raises:
    # Correctly rounded values from mpmath. Hosts keep subnormals, so these
    # must come back within 1 ulp (2^-149 or 2^-1074) rather than flush to 0.
    var x32: List[Float32] = [9.5, 9.6, 10.0]
    var bits32: List[UInt32] = [0x00006912, 0x00000F66, 0x00000001]
    for i in range(len(x32)):
        var got = erfc(x32[i])
        assert_true(
            _ulp_diff(got, Float32(from_bits=bits32[i])) <= 1,
            String("erfc(", x32[i], ") = ", got, ", bits ", hex(got.to_bits())),
        )
    var x64: List[Float64] = [26.6, 27.0]
    var bits64: List[UInt64] = [0x0000C860A467B1A5, 0x0000000000019E0F]
    for i in range(len(x64)):
        var got = erfc(x64[i])
        assert_true(
            _ulp_diff(got, Float64(from_bits=bits64[i])) <= 1,
            String("erfc(", x64[i], ") = ", got, ", bits ", hex(got.to_bits())),
        )


def _test_expm1_tiny_bits[
    dtype: DType
]() raises where dtype.is_floating_point():
    comptime uint_type = DType.uint32 if dtype == .float32 else DType.uint64
    comptime sign_bit = Scalar[uint_type](1) << Scalar[uint_type](
        31 if dtype == .float32 else 63
    )
    comptime min_normal_bits = Scalar[uint_type](1) << Scalar[uint_type](
        23 if dtype == .float32 else 52
    )
    for bits in [
        Scalar[uint_type](0),
        1,
        2,
        min_normal_bits >> 1,
        min_normal_bits - 1,
        min_normal_bits,
    ]:
        for sign in [Scalar[uint_type](0), sign_bit]:
            var input_bits = bits | sign
            var x = bitcast[dtype](input_bits)
            assert_equal(bitcast[uint_type](expm1(x)), input_bits)
            # A tiny lane must also be preserved alongside ordinary/special lanes.
            var mixed = SIMD[dtype, 4](x, 1.0, inf[dtype](), nan[dtype]())
            var result = expm1(mixed)
            assert_equal(bitcast[uint_type](result[0]), input_bits)
            assert_equal(result[2], inf[dtype]())
            assert_true(isnan(result[3]))


def test_expm1_bfloat16_tiny() raises:
    # Subnormal bit patterns plus 0x3F80 (1.0), a normal boundary value.
    var bits: List[UInt16] = [1, 2, 0x40, 0x7F, 0x80, 0x3F80]
    for b in bits:
        for sign in [UInt16(0), UInt16(0x8000)]:
            var x = bitcast[DType.bfloat16](b | sign)
            var expected = _call_libm["expm1"](x.cast[.float64]()).cast[
                .bfloat16
            ]()
            assert_equal(
                bitcast[DType.uint16](expm1(x)),
                bitcast[DType.uint16](expected),
            )


def test_expm1_float32_edges() raises:
    assert_equal(expm1(inf[.float32]()), inf[.float32]())
    assert_equal(expm1(neg_inf[.float32]()), -1.0)
    assert_true(isnan(expm1(nan[.float32]())))
    var neg_zero = expm1(Float32(-0.0))
    assert_equal(neg_zero, 0.0)
    assert_equal(copysign(Float32(1.0), neg_zero), -1.0)
    assert_equal(expm1(Float32(-30.0)), -1.0)
    # Just below the overflow threshold log(FLT_MAX), where the result needs
    # the top binade.
    comptime log_flt_max = Float32(88.7228)
    var near_overflow: List[Float32] = [88.38, 88.5, 88.72, log_flt_max]
    for val in near_overflow:
        var got = expm1(val)
        assert_false(isinf(got), msg=String(t"expm1({val}) overflowed"))
        assert_almost_equal(
            got, _call_libm["expm1"](val), rtol=1e-6, msg=String(val)
        )
    assert_equal(expm1(Float32(88.73)), inf[.float32]())
    assert_equal(expm1(Float32(1000.0)), inf[.float32]())
    # The largest float32 below log(FLT_MAX) stays finite; the next one
    # overflows.
    comptime last_finite = Float32(88.72283172607421875)
    assert_false(isinf(expm1(last_finite)))
    assert_equal(expm1(Float32(88.72283935546875)), inf[.float32]())


def test_expm1_float32_ulp() raises:
    # The k = +-1 reductions (|x| near ln2) used to reach 3 ulp around 0.36.
    # The reference is the float64 libm result rounded to float32, so this
    # allows up to 1 ulp of error, not a correctly rounded result.
    comptime n = 2_000
    for i in range(n + 1):
        var x = Float32(-20.0) + Float32(i) * (Float32(108.0) / n)
        var dense = Float32(-1.5) + Float32(i) * (Float32(3.0) / n)
        for val in [x, dense]:
            var expected = _call_libm["expm1"](val.cast[.float64]()).cast[
                .float32
            ]()
            var got = expm1(val)
            assert_true(
                abs(got - expected) <= ulp(expected),
                msg=String(t"expm1({val}) = {got}, expected {expected}"),
            )


def test_expm1_float64_edges() raises:
    assert_equal(expm1(inf[.float64]()), inf[.float64]())
    assert_equal(expm1(neg_inf[.float64]()), -1.0)
    assert_true(isnan(expm1(nan[.float64]())))
    var neg_zero = expm1(Float64(-0.0))
    assert_equal(neg_zero, 0.0)
    assert_equal(copysign(Float64(1.0), neg_zero), -1.0)
    assert_equal(expm1(Float64(-745.0)), -1.0)
    # The float64 path is new, so pin its overflow threshold the way the
    # float32 test does: the largest float64 below 1024*ln(2) stays finite
    # (just under FLT64_MAX after subtracting 1), and the next one overflows.
    comptime last_finite = Float64(709.782712893384)
    assert_false(isinf(expm1(last_finite)))
    assert_equal(expm1(Float64(709.7827128933841)), inf[.float64]())
    assert_equal(expm1(Float64(1000.0)), inf[.float64]())


def test_expm1_tiny_bits() raises:
    _test_expm1_tiny_bits[.float32]()
    _test_expm1_tiny_bits[.float64]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
