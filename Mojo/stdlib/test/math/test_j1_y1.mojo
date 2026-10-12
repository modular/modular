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

from std.math import frexp, inf, isinf, isnan, j1, ldexp, nan, sqrt, y1
from std.math.math import _call_libm
from std.testing import TestSuite, assert_equal, assert_true
from std.utils.numerics import FPUtils

comptime sqrt_two_over_pi = 0.7978845608028654
# Bound on the error in ulps of max(|result|, envelope), as in test_j0_y0.
comptime max_scaled_error = 4.0


def _ulp[dtype: DType](v: Float64) -> Float64:
    """Returns the spacing of `dtype` values in the binade of `v`, in float64;
    below the normal range it is the smallest subnormal."""
    comptime mant = FPUtils[dtype].mantissa_width()
    comptime emin = 1 - FPUtils[dtype].exponent_bias()
    var e = Int(frexp(abs(v))[1]) - 1 if v != 0 else emin
    return ldexp(Float64(1), Int32(max(e, emin) - mant))


def _inputs() -> List[Float64]:
    var xs = List[Float64]()
    # Dense sweep over the oscillating region, both signs.
    for i in range(-4000, 4001):
        xs.append(Float64(i) * 0.0125 + 0.0003)
    # Geometric sweep across the whole float64 range.
    var x = 1.0e-300
    while x < 1.0e300:
        xs.append(x)
        x *= 1.37
    # Zeros, small/large crossover and rational interval boundaries.
    var specials: List[Float64] = [
        2.197141326031017,
        3.8317059702075123,
        5.429681040794135,
        7.015586669815619,
        0.125,
        0.5,
        1.0,
        2.0,
        1.9999999,
        2.857,
        4.5454,
        2.8571414947509766,
        4.545452117919922,
        8.0,
        131071.9,
        131072.0,
        131072.1,
        1099511627775.0,
        1099511627776.0,
        1.0e22,
        5.0e15,
        6.5536e4,
        1.0e10,
        3.4e38,
        1.0e308,
        1.0e-8,
        1.0e-40,
        4.9e-324,
    ]
    for s in specials:
        xs.append(s)
        xs.append(-s)
    return xs^


def _check[dtype: DType, is_y1: Bool]() raises where dtype.is_floating_point():
    var worst = 0.0
    for x64 in _inputs():
        var x = Scalar[dtype](x64)
        var xr = x.cast[.float64]()
        var got: Scalar[dtype]
        var want64: Float64
        comptime if is_y1:
            got = y1(x)
            want64 = _call_libm["y1"](xr)
        else:
            got = j1(x)
            want64 = _call_libm["j1"](xr)
        var want = Scalar[dtype](want64)
        if isnan(want):
            assert_true(isnan(got))
            continue
        if isinf(want):
            assert_equal(got._to_bits_signed(), want._to_bits_signed())
            continue
        assert_true(not isnan(got) and not isinf(got))
        if got == want:
            continue
        var scale = abs(want64)
        var ax = abs(xr)
        if ax >= 1.0:
            scale = max(scale, sqrt_two_over_pi / sqrt(ax))
        # Both values can round to zero when an input casts to float32 zero.
        if scale == 0.0:
            assert_equal(got, want)
            continue
        var err = abs(got.cast[.float64]() - want64) / _ulp[dtype](scale)
        assert_true(
            err <= max_scaled_error,
            String(
                t"j1/y1 dtype={dtype} is_y1={is_y1} x={xr} got={got}"
                t" want={want64} scaled error={err}"
            ),
        )
        worst = max(worst, err)
    print(t"j1/y1 dtype={dtype} is_y1={is_y1} max scaled error={worst}")


def test_j1_y1_accuracy() raises:
    _check[.float32, False]()
    _check[.float32, True]()
    _check[.float64, False]()
    _check[.float64, True]()


def _check_odd[dtype: DType]() raises where dtype.is_floating_point():
    var xs: List[Scalar[dtype]] = [0.125, 0.5, 1.5, 2.0, 3.8317, 8.0, 1.0e6]
    for x in xs:
        assert_equal(j1(-x)._to_bits_signed(), (-j1(x))._to_bits_signed())


def test_j1_finite_odd_symmetry() raises:
    _check_odd[.float32]()
    _check_odd[.float64]()


def _check_vector_reference[
    dtype: DType, //
](
    got: Scalar[dtype], want64: Float64, x: Scalar[dtype]
) raises where dtype.is_floating_point():
    var want = Scalar[dtype](want64)
    if isnan(want):
        assert_true(isnan(got))
        return
    if isinf(want) or want == 0.0:
        assert_equal(got._to_bits_signed(), want._to_bits_signed())
        return
    assert_true(not isnan(got) and not isinf(got))
    if got == want:
        return
    var scale = abs(want64)
    var ax = abs(x.cast[.float64]())
    if ax >= 1.0:
        scale = max(scale, sqrt_two_over_pi / sqrt(ax))
    var err = abs(got.cast[.float64]() - want64) / _ulp[dtype](scale)
    assert_true(
        err <= max_scaled_error,
        String(t"SIMD x={x} got={got} want={want64} scaled error={err}"),
    )


def _check_lanes[
    dtype: DType, //
](x: SIMD[dtype, 4]) raises where dtype.is_floating_point():
    var j = j1(x)
    var y = y1(x)
    for i in range(4):
        var xr = x[i].cast[.float64]()
        # j1 deliberately uses +0 at either infinity, unlike some libms.
        var want_j = Float64(0.0) if isinf(xr) else _call_libm["j1"](xr)
        _check_vector_reference(j[i], want_j, x[i])
        _check_vector_reference(y[i], _call_libm["y1"](xr), x[i])


def _check_vectors[dtype: DType]() raises where dtype.is_floating_point():
    _check_lanes(SIMD[dtype, 4](0.0, 0.25, -1.5, 1.9))
    _check_lanes(SIMD[dtype, 4](-8.0, 2.0, 12.0, 1.0e6))
    _check_lanes(SIMD[dtype, 4](0.5, 3.0, -2.0, nan[dtype]()))
    _check_lanes(SIMD[dtype, 4](inf[dtype](), -inf[dtype](), -0.0, 0.0))


def test_j1_y1_simd() raises:
    _check_vectors[.float32]()
    _check_vectors[.float64]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
