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

# Subnormal inputs whose results are normal. `y0` and `y1` read a subnormal
# through its integer significand or its bits, so on targets that flush
# subnormals, subnormal inputs are handled and normal results are correct;
# subnormal results are rebuilt from integer bits (Apple GPUs). On IEEE
# targets the path is bit-identical to scaling by a power of two. The GPU
# counterpart is `test_bessel_huge_args_gpu.mojo`.

from std.math import (
    scalb,
    atan,
    atan2,
    gamma,
    cbrt,
    frexp,
    hypot,
    inf,
    isfinite,
    isinf,
    isnan,
    nan,
    remainder,
    y0,
    y1,
)
from std.math.math import (
    _call_libm,
    _frexp_bits,
    _log_accurate,
)
from std.memory import bitcast
from std.testing import TestSuite, assert_equal, assert_raises, assert_true
from std.utils.numerics import FlushDenormals

from test_utils.math import ulp_distance

comptime f32_min = Float32(from_bits=UInt32(1))  # 2^-149
comptime f32_min_normal = Float32(from_bits=UInt32(0x00800000))  # 2^-126
comptime f32_max_sub = Float32(from_bits=UInt32(0x007FFFFF))
comptime f64_min = Float64(from_bits=UInt64(1))  # 2^-1074
comptime f64_max_sub = Float64(from_bits=UInt64(0x000FFFFFFFFFFFFF))


def _check[
    dtype: DType, //
](got: Scalar[dtype], want: Scalar[dtype], name: String) raises:
    var msg = String(t"{name} = {got}, expected {want}")
    assert_true(isfinite(got) and isfinite(want), msg)
    assert_true(
        (got._to_bits_signed() < 0) == (want._to_bits_signed() < 0), msg
    )
    var got_bits = got.to_bits[.uint64]()
    var want_bits = want.to_bits[.uint64]()
    var d = max(got_bits, want_bits) - min(got_bits, want_bits)
    assert_true(d <= 1, String(t"{msg} ({d} ulp)"))


def test_gamma_subnormal() raises:
    # Gamma(x) = 1/x - euler_gamma + O(x) rounds to 1/x here.
    assert_equal(
        gamma(Float32(from_bits=UInt32(1) << 22)), 1.7014118346046923e38
    )
    assert_equal(
        gamma(-Float32(from_bits=UInt32(1) << 22)), -1.7014118346046923e38
    )
    _check(gamma(f32_max_sub), 8.507060187143942e37, "gamma(max_sub)")
    _check(gamma(-f32_max_sub), -8.507060187143942e37, "gamma(-max_sub)")
    _check(
        gamma(Float32(from_bits=UInt32(3) << 21)),
        1.1342745902071442e38,
        "gamma(3 * 2^-128)",
    )
    assert_equal(gamma(Float32(from_bits=UInt32(1) << 21)), inf[.float32]())
    assert_equal(gamma(f32_min), inf[.float32]())
    assert_equal(gamma(-f32_min), -inf[.float32]())
    assert_equal(
        gamma(Float64(from_bits=UInt64(1) << 51)), 8.98846567431158e307
    )
    _check(gamma(f64_max_sub), 4.494232837155791e307, "gamma(max_sub)")
    _check(gamma(-f64_max_sub), -4.494232837155791e307, "gamma(-max_sub)")
    assert_equal(gamma(f64_min), inf[.float64]())
    var v = gamma(SIMD[.float32, 4](f32_max_sub, 5.0, -f32_max_sub, f32_min))
    assert_equal(v[0], gamma(f32_max_sub))
    assert_equal(v[1], 24.0)
    assert_equal(v[2], gamma(-f32_max_sub))
    assert_equal(v[3], inf[.float32]())


def _check_rejects_mutants[dtype: DType]() raises:
    comptime S = Scalar[dtype]
    var infinity = inf[dtype]()
    var max_finite = S(from_bits=infinity._to_bits_signed() - 1)
    var nan_next_to_inf = S(from_bits=infinity._to_bits_signed() + 1)
    _check(S(1.0), S(1.0), "oracle")
    _check(max_finite, max_finite, "oracle")
    for mutant in [S(-4.0), infinity, -infinity, nan_next_to_inf]:
        with assert_raises():
            _check(mutant, S(1.0), "oracle mutant")
    for mutant in [infinity, nan_next_to_inf, -max_finite]:
        with assert_raises():
            _check(mutant, max_finite, "oracle mutant")


def test_gamma_subnormal_oracle_f32() raises:
    _check_rejects_mutants[.float32]()


def test_gamma_subnormal_oracle_f64() raises:
    _check_rejects_mutants[.float64]()


def test_hypot_subnormal() raises:
    # Both subnormal, normal result.
    _check(
        hypot(f32_max_sub, f32_max_sub),
        1.6623998848452245e-38,
        "hypot(max_sub, max_sub)",
    )
    # Normal and subnormal operands of comparable size.
    _check(
        hypot(Float32(1.1754943508222875e-38), Float32(1.1571272515906893e-38)),
        1.6494636579419635e-38,
        "hypot(2^-126, 0.98 * 2^-126)",
    )
    assert_equal(hypot(f32_min, Float32(1.0)), 1.0)
    assert_equal(hypot(f32_min, Float32(0.0)), f32_min)
    _check(
        hypot(f64_max_sub, f64_max_sub),
        3.1467296279827165e-308,
        "hypot(max_sub, max_sub)",
    )


def test_hypot_subnormal_outputs() raises:
    var half_min = Float32(from_bits=Int32(0x00400000))
    assert_equal(Int(hypot(half_min, half_min)._to_bits_signed()), 0x005A827A)
    var bf16_half_min = BFloat16(from_bits=Int16(0x0040))
    assert_equal(
        Int(hypot(bf16_half_min, bf16_half_min)._to_bits_signed()), 0x005B
    )
    assert_equal(hypot(bf16_half_min, BFloat16(0)), bf16_half_min)
    assert_equal(
        Int(
            hypot(
                BFloat16(from_bits=Int16(0x007F)),
                BFloat16(from_bits=Int16(0x000C)),
            )._to_bits_signed()
        ),
        0x0080,
    )

    var xs = SIMD[.float32, 4](
        half_min, f32_max_sub, nan[.float32](), inf[.float32]()
    )
    var ys = SIMD[.float32, 4](
        half_min, f32_max_sub, Float32(1), nan[.float32]()
    )
    var got = hypot(xs, ys)
    assert_equal(Int(got[0]._to_bits_signed()), 0x005A827A)
    _check(got[1], Float32(from_bits=Int32(0x00B504F2)), "hypot SIMD normal")
    assert_true(isnan(got[2]))
    assert_equal(got[3], inf[.float32]())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


def _check_cbrt[
    dtype: DType
](got: Scalar[dtype], want: Scalar[dtype], name: String) raises:
    comptime assert dtype in (DType.float32, DType.float64)
    assert_true(
        isfinite(got) and isfinite(want),
        String(t"{name}: expected finite values, got {got} and {want}"),
    )
    comptime sign = UInt64(1) << UInt64(63 if dtype == .float64 else 31)
    var got_bits = got.to_bits[.uint64]()
    var want_bits = want.to_bits[.uint64]()
    assert_equal(
        got_bits & sign,
        want_bits & sign,
        String(t"{name}: wrong sign for {got}, expected {want}"),
    )
    var d = (
        got_bits - want_bits if got_bits >= want_bits else want_bits - got_bits
    )
    assert_true(d <= 1, String(t"{name} = {got}, expected {want} ({d} ulp)"))


def _check_log_helpers[
    dtype: DType
](x: Scalar[dtype]) raises where dtype.is_floating_point():
    """`_frexp_bits` feeds `_log_reduce`; it must agree with `frexp` on
    subnormals, and `_log_accurate` of a subnormal must match libm."""
    var want = frexp(x)
    var got = _frexp_bits(x)
    assert_equal(got[0], want[0], msg=String(t"_frexp_bits({x}) fraction"))
    assert_equal(got[1], want[1], msg=String(t"_frexp_bits({x}) exponent"))
    if x > 0:
        var lg = _log_accurate(x)
        var want_log = _call_libm["log"](x)
        var d = ulp_distance(lg, want_log)
        assert_true(
            d <= 1,
            String(t"_log_accurate({x}) = {lg}, libm {want_log} ({d} ulp)"),
        )


@inline(.never)
def _runtime_y0[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    return y0(x)


@inline(.never)
def _runtime_y1[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    return y1(x)


def test_cbrt_subnormal() raises:
    _check_cbrt(cbrt(f32_min), 1.119034708375443e-15, "cbrt(2^-149)")
    # 2^-147 is a perfect cube.
    assert_equal(cbrt(f32_min * 4), Float32(1.7763568394002505e-15))
    _check_cbrt(cbrt(-f32_max_sub), -2.273736618907049e-13, "cbrt(-max_sub)")
    assert_equal(cbrt(f64_min), 1.7031839360032603e-108)
    _check_cbrt(cbrt(-f64_max_sub), -2.8126442852362615e-103, "cbrt(-max_sub)")
    var v = cbrt(SIMD[.float32, 4](f32_min, -f32_max_sub, 8.0, f32_min * 4))
    assert_equal(v[0], cbrt(f32_min))
    assert_equal(v[1], cbrt(-f32_max_sub))
    assert_equal(v[2], 2.0)
    assert_equal(v[3], cbrt(f32_min * 4))


def test_check() raises:
    # These bit patterns overflow a signed ULP distance to Int.MIN.
    comptime want = Float64(from_bits=UInt64(0xAAA428A2F98D728A))
    comptime bad = Float64(from_bits=UInt64(0x555BD75D06728D76))
    with assert_raises(contains="wrong sign"):
        _check_cbrt(bad, want, "cbrt(-max_sub)")
    _check_cbrt(want, want, "exact")
    _check_cbrt(Float64(from_bits=UInt64(0xAAA428A2F98D728B)), want, "one ulp")
    with assert_raises(contains="ulp"):
        _check_cbrt(
            Float64(from_bits=UInt64(0xAAA428A2F98D728C)), want, "two ulp"
        )

    comptime for dtype in [DType.float32, DType.float64]:
        with assert_raises(contains="wrong sign"):
            _check_cbrt(Scalar[dtype](-4.0), Scalar[dtype](1.0), "wrong sign")
        with assert_raises(contains="expected finite"):
            _check_cbrt(
                inf[dtype](), Scalar[dtype].MAX_FINITE, "infinite result"
            )
        with assert_raises(contains="expected finite"):
            _check_cbrt(
                Scalar[dtype].MAX_FINITE, inf[dtype](), "infinite reference"
            )
        with assert_raises(contains="expected finite"):
            _check_cbrt(nan[dtype](), Scalar[dtype](1.0), "NaN result")
        with assert_raises(contains="expected finite"):
            _check_cbrt(Scalar[dtype](1.0), nan[dtype](), "NaN reference")


def test_log_helpers_subnormal() raises:
    _check_log_helpers(f32_min)
    _check_log_helpers(f32_max_sub)
    _check_log_helpers(Float32(from_bits=UInt32(1) << 22))
    _check_log_helpers(-f32_min)
    _check_log_helpers(-f32_max_sub)
    _check_log_helpers(Float32(0.75))
    _check_log_helpers(Float32(3.0e38))
    _check_log_helpers(f64_min)
    _check_log_helpers(f64_max_sub)
    _check_log_helpers(Float64(from_bits=UInt64(1) << 51))
    _check_log_helpers(-f64_max_sub)
    _check_log_helpers(Float64(1.5e308))
    var v = _frexp_bits(SIMD[.float32, 4](f32_min, -f32_max_sub, 6.0, 0.5))
    var w = frexp(SIMD[.float32, 4](f32_min, -f32_max_sub, 6.0, 0.5))
    assert_equal(v[0], w[0])
    assert_equal(v[1], w[1])


def test_remainder_normal_results_near_min_normal() raises:
    # Normal results whose reduction runs close to the subnormal boundary,
    # where `p * 0.5` is inexact and the fma residues are smallest.
    assert_equal(
        remainder(f32_min_normal * 3, f32_min_normal * 2), -f32_min_normal
    )
    assert_equal(
        remainder(f32_min_normal * 5, f32_min_normal * 2), f32_min_normal
    )
    assert_equal(
        remainder(f32_min_normal * 7, f32_min_normal * 4), -f32_min_normal
    )
    # 2^26 = 3 * 22369621 + 1, so 2^-100 = n * (3 * 2^-126) + 2^-126.
    assert_equal(
        remainder(Float32(from_bits=UInt32(0x0D800000)), f32_min_normal * 3),
        f32_min_normal,
    )
    assert_equal(remainder(Float32(1.0), f32_min_normal * 3), f32_min_normal)
    var v = remainder(
        SIMD[.float32, 4](f32_min_normal * 3, 1.0, f32_min_normal * 5, -1.0),
        SIMD[.float32, 4](
            f32_min_normal * 2,
            f32_min_normal * 3,
            f32_min_normal * 2,
            f32_min_normal * 3,
        ),
    )
    assert_equal(
        v,
        SIMD[.float32, 4](
            -f32_min_normal, f32_min_normal, f32_min_normal, -f32_min_normal
        ),
    )


# CPU coverage of normal and subnormal outputs, and of the integer-only
# helpers that keep subnormal results on targets that flush arithmetic.

from std.math import (
    cbrt,
    frexp,
    hypot,
    inf,
    isfinite,
    isinf,
    isnan,
    nan,
    remainder,
    y0,
    y1,
)

from test_utils.math import ulp_distance


def test_remainder_subnormal() raises:
    # Every multiple of the smallest subnormal reduces to a signed zero.
    assert_true(remainder(Float32(1.0), f32_min).to_bits() == 0)
    assert_true(remainder(-f32_max_sub, f32_min).to_bits() == 0x80000000)
    assert_equal(remainder(Float32(8.0), f32_min), Float32(0.0))
    assert_equal(
        remainder(Float32(1.1754943508222875e-38), f32_min * 3), -f32_min
    )
    assert_equal(remainder(f32_max_sub, f32_min * 7), f32_min * 3)
    assert_equal(remainder(f32_min, f32_min * 3), f32_min)
    assert_true(remainder(Float64(1.0), f64_min).to_bits() == 0)
    assert_equal(
        remainder(Float64(2.2250738585072014e-308), f64_min * 3), f64_min
    )
    var v = remainder(
        SIMD[.float32, 4](1.0, 7.0, f32_max_sub, -3.0),
        SIMD[.float32, 4](f32_min, 3.0, f32_min * 7, f32_min),
    )
    assert_equal(v, SIMD[.float32, 4](0.0, 1.0, f32_min * 3, -0.0))


def test_remainder_subnormal_results() raises:
    # Subnormal results from subnormal and from normal inputs; the last pair
    # is two normal inputs one ulp apart. 2^149 mod (2^23 - 1) = 2^11.
    assert_true(remainder(f32_min, f32_min * 3).to_bits() == 1)
    assert_true(remainder(Float32(1.0), f32_max_sub).to_bits() == 0x800)
    assert_true(
        remainder(
            Float32(from_bits=UInt32(0x00800001)), f32_min_normal
        ).to_bits()
        == 1
    )


def test_scalb_infinite_exponent() raises:
    assert_equal(scalb(f32_min, inf[.float32]()), inf[.float32]())
    assert_equal(scalb(-f32_min, inf[.float32]()), -inf[.float32]())
    assert_equal(scalb(f32_max_sub, inf[.float32]()), inf[.float32]())
    assert_true(isnan(scalb(Float32(0.0), inf[.float32]())))
    assert_true(isnan(scalb(Float32(-0.0), inf[.float32]())))
    assert_equal(scalb(Float32(3.0), inf[.float32]()), inf[.float32]())
    assert_true(scalb(-f32_min, -inf[.float32]()).to_bits() == 0x80000000)
    assert_true(scalb(f32_max_sub, -inf[.float32]()).to_bits() == 0)
    assert_true(isnan(scalb(inf[.float32](), -inf[.float32]())))
    assert_equal(scalb(inf[.float32](), inf[.float32]()), inf[.float32]())
    assert_equal(scalb(f64_min, inf[.float64]()), inf[.float64]())
    assert_equal(scalb(-f64_max_sub, inf[.float64]()), -inf[.float64]())
    assert_true(isnan(scalb(Float64(0.0), inf[.float64]())))
    assert_true(
        scalb(-f64_min, -inf[.float64]()).to_bits() == 0x8000000000000000
    )
    var v = scalb(
        SIMD[.float32, 4](f32_min, -f32_max_sub, 0.0, 2.0),
        SIMD[.float32, 4](
            inf[.float32](), inf[.float32](), inf[.float32](), -inf[.float32]()
        ),
    )
    assert_equal(v[0], inf[.float32]())
    assert_equal(v[1], -inf[.float32]())
    assert_true(isnan(v[2]))
    assert_equal(v[3], 0.0)


def test_scalb_subnormal() raises:
    assert_equal(scalb(f32_min, Float32(149)), 1.0)
    assert_equal(
        scalb(f32_min * 4, Float32(100)), Float32(7.105427357601002e-15)
    )
    assert_equal(scalb(-f32_max_sub, Float32(126)), -Float32(0.99999988079071))
    assert_equal(scalb(f32_min, Float32(0)), f32_min)
    assert_equal(scalb(-f32_min, Float32(300)), -inf[.float32]())
    assert_equal(scalb(f64_min, Float64(1074)), 1.0)
    assert_equal(scalb(f64_max_sub, Float64(1022)), 0.9999999999999998)
    assert_equal(scalb(f64_min, Float64(-1)), 0.0)


def test_scalb_subnormal_exponent() raises:
    # A subnormal exponent is never integral, so the result is NaN even on a
    # target that flushes the exponent to zero before comparing it.
    assert_true(isnan(scalb(Float32(1), f32_min)))
    assert_true(isnan(scalb(Float32(-8), -f32_max_sub)))
    assert_true(isnan(scalb(f32_min, f32_min)))
    assert_true(isnan(scalb(Float64(1), f64_min)))
    assert_true(isnan(scalb(Float64(2), -f64_max_sub)))
    var v = scalb(
        SIMD[.float32, 4](1.0, 2.0, 3.0, 4.0),
        SIMD[.float32, 4](f32_min, 1.0, -f32_max_sub, 0.0),
    )
    assert_true(isnan(v[0]))
    assert_equal(v[1], 4.0)
    assert_true(isnan(v[2]))
    assert_equal(v[3], 4.0)


def _check_subnormal_order(y: SIMD[.float32, 4], x: SIMD[.float32, 4]) raises:
    var got = atan2(y, x)
    var expected = SIMD[.float32, 4](
        1.249045729637146,
        0.32175055146217346,
        -1.249045729637146,
        1.8925468921661377,
    )
    for i in range(4):
        _check(got[i], expected[i], "unequal subnormals")


def _check_tiny_atan(bits: SIMD[.uint32, 4]) raises:
    var x = SIMD[.float32, 4](from_bits=bits)
    assert_equal(bitcast[.uint32, 4](atan(x)), bits)


def _check_tiny_bits(
    y_bits: SIMD[.uint32, 4],
    x_bits: SIMD[.uint32, 4],
    expected: SIMD[.uint32, 4],
) raises:
    var y = SIMD[.float32, 4](from_bits=y_bits)
    var x = SIMD[.float32, 4](from_bits=x_bits)
    assert_equal(bitcast[.uint32, 4](atan2(y, x)), expected)


def _tiny_atan_cases() raises:
    _check_tiny_atan(SIMD[.uint32, 4](1, 0x80000001, 0x007FFFFF, 0x807FFFFF))
    _check_tiny_atan(SIMD[.uint32, 4](0, 0x80000000, 0x00800000, 0x80800000))


def _tiny_cases() raises:
    _check_tiny_bits(
        SIMD[.uint32, 4](0x00800000, 0x80800000, 0x007FFFFF, 0x807FFFFF),
        SIMD[.uint32, 4](0x40000000, 0x40000000, 0x3F800000, 0x3F800000),
        SIMD[.uint32, 4](0x00400000, 0x80400000, 0x007FFFFF, 0x807FFFFF),
    )
    _check_tiny_bits(
        SIMD[.uint32, 4](0x00800000, 0x00800000, 0x00FFFFFF, 0x80FFFFFF),
        SIMD[.uint32, 4](0x3F800001, 0x3F7FFFFF, 0x40000000, 0x40000000),
        SIMD[.uint32, 4](0x007FFFFF, 0x00800001, 0x007FFFFF, 0x807FFFFF),
    )
    # Halfway quotients round downward because atan(t) < t for positive t.
    _check_tiny_bits(
        SIMD[.uint32, 4](1, 3, 5, 7),
        SIMD[.uint32, 4](0x40000000),
        SIMD[.uint32, 4](0, 1, 2, 3),
    )
    _check_tiny_bits(
        SIMD[.uint32, 4](0x00800000, 0x00800000, 0x00800000, 0x00800001),
        SIMD[.uint32, 4](0x4AFFFFFF, 0x4B000000, 0x4B000001, 0x3F800000),
        SIMD[.uint32, 4](1, 1, 1, 0x00800001),
    )
    _check_tiny_bits(
        SIMD[.uint32, 4](1, 1, 1, 0x80000001),
        SIMD[.uint32, 4](0x3FFFFFFF, 0x40000000, 0x40000001, 0x40000000),
        SIMD[.uint32, 4](1, 0, 0, 0x80000000),
    )


def test_atan2_subnormal() raises:
    comptime f32_min = Float32(from_bits=UInt32(1))  # 2^-149
    comptime f32_max_sub = Float32(from_bits=UInt32(0x007FFFFF))
    comptime f64_min = Float64(from_bits=UInt64(1))  # 2^-1074
    comptime f64_max_sub = Float64(from_bits=UInt64(0x000FFFFFFFFFFFFF))
    assert_equal(atan2(f32_min, f32_min), 0.7853981852531433)
    _check(
        atan2(Float32(from_bits=UInt32(3)), f32_min),
        1.249045729637146,
        "atan2(3u, u)",
    )
    _check(
        atan2(f32_min, Float32(from_bits=UInt32(4))),
        0.244978666305542,
        "atan2(u, 4u)",
    )
    _check(
        atan2(f32_min, Float32(1.1754943508222875e-38)),
        1.1920928955078125e-07,
        "atan2(u, 2^-126)",
    )
    _check(atan2(f32_min, -f32_max_sub), 3.141592502593994, "atan2(u, -max)")
    _check(atan2(-f32_max_sub, f32_min), -1.570796251296997, "atan2(-max, u)")
    assert_equal(atan2(f64_min, f64_min), 0.7853981633974483)
    _check(
        atan2(Float64(from_bits=UInt64(3)), f64_min),
        1.2490457723982544,
        "atan2(3u, u)",
    )
    _check(atan2(f64_min, -f64_max_sub), 3.141592653589793, "atan2(u, -max)")
    var v = atan2(
        SIMD[.float32, 4](f32_min, 3.0, -f32_max_sub, 0.0),
        SIMD[.float32, 4](f32_min, 4.0, f32_min, -f32_min),
    )
    assert_equal(v[0], atan2(f32_min, f32_min))
    assert_equal(v[1], atan2(Float32(3.0), Float32(4.0)))
    assert_equal(v[2], atan2(-f32_max_sub, f32_min))
    assert_equal(v[3], 3.1415927410125732)


def test_subnormal_order_flush_denormals() raises:
    var ys = SIMD[.float32, 4](from_bits=SIMD[.uint32, 4](3, 1, 0x80000003, 3))
    var xs = SIMD[.float32, 4](from_bits=SIMD[.uint32, 4](1, 3, 1, 0x80000001))
    _check_subnormal_order(ys, xs)
    with FlushDenormals():
        _check_subnormal_order(ys, xs)


def test_tiny_atan_bits() raises:
    _tiny_atan_cases()
    with FlushDenormals():
        _tiny_atan_cases()


def test_tiny_bits() raises:
    _tiny_cases()


def test_tiny_bits_flush_denormals() raises:
    with FlushDenormals():
        _tiny_cases()


def test_y0_flush_denormals() raises:
    # Calling out of line prevents the subnormal fixtures from being folded
    # before entering the flushing context. References are rounded MPFR values.
    var x32 = SIMD[.float32, 4](
        f32_min, Float32(from_bits=UInt32(1) << 22), f32_max_sub, 1.0
    )
    var x64 = SIMD[.float64, 4](
        f64_min, Float64(from_bits=UInt64(1) << 51), f64_max_sub, 1.0
    )
    var want32 = SIMD[.float32, 4](
        -65.82321166992188,
        -56.1152458190918,
        -55.67397689819336,
        0.08825696259737015,
    )
    var want64 = SIMD[.float64, 4](
        -473.9990734230043,
        -451.49424220743384,
        -451.0529710071285,
        0.08825696421567696,
    )
    var normal32 = y0(Float32(8.0))
    var normal64 = y0(Float64(8.0))
    with FlushDenormals():
        var got32 = _runtime_y0(x32)
        var got64 = _runtime_y0(x64)
        for i in range(4):
            _check(got32[i], want32[i], "y0 f32 FTZ mixed")
            _check(got64[i], want64[i], "y0 f64 FTZ mixed")
        _check(_runtime_y0(f32_min), want32[0], "y0 f32 FTZ min")
        _check(_runtime_y0(f64_min), want64[0], "y0 f64 FTZ min")
        var mixed32 = _runtime_y0(
            SIMD[.float32, 4](f32_min, 8.0, f32_max_sub, 0.0)
        )
        var mixed64 = _runtime_y0(
            SIMD[.float64, 4](f64_min, 8.0, f64_max_sub, 0.0)
        )
        _check(mixed32[0], want32[0], "y0 f32 FTZ small/large")
        _check(mixed64[0], want64[0], "y0 f64 FTZ small/large")
        assert_equal(mixed32[1], normal32)
        assert_equal(mixed64[1], normal64)
        _check(mixed32[2], want32[2], "y0 f32 FTZ max mixed")
        _check(mixed64[2], want64[2], "y0 f64 FTZ max mixed")
        assert_equal(mixed32[3], -inf[.float32]())
        assert_equal(mixed64[3], -inf[.float64]())
        assert_true(isnan(_runtime_y0(Float32(from_bits=UInt32(0x80000001)))))
        assert_true(
            isnan(_runtime_y0(Float64(from_bits=UInt64(0x8000000000000001))))
        )


def test_y1_flush_denormals() raises:
    var x32 = SIMD[.float32, 4](
        Float32(from_bits=UInt32(1) << 22),
        f32_max_sub,
        Float32(from_bits=UInt32(0x80000001)),
        0.0,
    )
    var x64 = SIMD[.float64, 4](
        Float64(from_bits=UInt64(1) << 51),
        f64_max_sub,
        Float64(from_bits=UInt64(0x8000000000000001)),
        0.0,
    )
    with FlushDenormals():
        var got32 = _runtime_y1(x32)
        var got64 = _runtime_y1(x64)
        _check(got32[0], Float32(-1.0831523711529481e38), "y1 f32 FTZ mid")
        _check(got32[1], Float32(-5.415762869885221e37), "y1 f32 FTZ max")
        _check(got64[0], Float64(-5.722234971514057e307), "y1 f64 FTZ mid")
        _check(got64[1], Float64(-2.861117485757029e307), "y1 f64 FTZ max")
        assert_true(isnan(got32[2]))
        assert_true(isnan(got64[2]))
        assert_equal(got32[3], -inf[.float32]())
        assert_equal(got64[3], -inf[.float64]())


def test_y1_y0_subnormal() raises:
    # y1(x) = -2/(pi x) below 2^-25; at 2^-128 1/x overflows but y1 does not.
    _check(
        y1(Float32(from_bits=UInt32(1) << 22)),
        -1.0831523711529481e38,
        "y1(2^-127)",
    )
    _check(
        y1(Float32(from_bits=UInt32(1) << 21)),
        -2.1663047423058962e38,
        "y1(2^-128)",
    )
    _check(y1(f32_max_sub), -5.415762869885221e37, "y1(max_sub)")
    assert_equal(y1(f32_min * 4), -inf[.float32]())
    assert_equal(y1(Float32(0.0)), -inf[.float32]())
    assert_equal(y1(Float32(-0.0)), -inf[.float32]())
    assert_true(isnan(y1(-f32_min)))
    _check(
        y1(Float64(from_bits=UInt64(1) << 51)),
        -5.722234971514057e307,
        "y1(2^-1023)",
    )
    _check(y1(f64_max_sub), -2.861117485757029e307, "y1(max_sub)")
    _check(y0(f32_min), -65.82321166992188, "y0(2^-149)")
    _check(y0(f32_max_sub), -55.67397689819336, "y0(max_sub)")
    assert_equal(y0(Float32(-0.0)), -inf[.float32]())
    assert_true(isnan(y0(-f32_max_sub)))
    _check(y0(f64_min), -473.9990734230043, "y0(2^-1074)")
    var v = y1(SIMD[.float32, 4](f32_max_sub, 8.0, -f32_min, 0.0))
    assert_equal(v[0], y1(f32_max_sub))
    assert_equal(v[1], y1(Float32(8.0)))
    assert_true(isnan(v[2]))
    assert_equal(v[3], -inf[.float32]())
