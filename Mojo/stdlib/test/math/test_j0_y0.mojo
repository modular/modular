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

from std.math import frexp, inf, isinf, isnan, j0, ldexp, nan, sqrt, y0
from std.math.math import _call_libm
from std.testing import assert_true, TestSuite
from std.utils.numerics import FPUtils


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
    # Zeros of j0 and y0, branch thresholds, and hard reduction cases.
    var specials: List[Float64] = [
        0.8935769662791675,
        2.404825557695773,
        3.957678419314858,
        5.520078110286311,
        7.086051060301773,
        8.653727912911013,
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


def _scale(x: Float64) -> Float64:
    """Returns the magnitude scale of j0/y0 around x (their envelope)."""
    comptime two_over_pi = 0.6366197723675814
    var ax = abs(x)
    if ax < 2.0:
        return 1.0
    return sqrt(two_over_pi / ax)


def _check[dtype: DType, is_y0: Bool]() raises where dtype.is_floating_point():
    """Compares std.math j0/y0 with libm's float64 j0/y0.

    The error is measured in ulps of the result, but near zeros of the
    function (where it is ill-conditioned) in ulps of the envelope
    `sqrt(2 / (pi |x|))` instead. Both use the true ulp spacing of `dtype`.
    """
    comptime max_ulp = 4.0
    comptime max_envelope_ulp = 4.0
    var worst_ulp = 0.0
    var worst_abs = 0.0
    for x64 in _inputs():
        var x = Scalar[dtype](x64)
        var xr = x.cast[.float64]()
        var got: Float64
        var want: Float64
        comptime if is_y0:
            got = y0(x).cast[.float64]()
            want = _call_libm["y0"](xr)
        else:
            got = j0(x).cast[.float64]()
            want = _call_libm["j0"](xr)
        if isnan(want) or isinf(want):
            assert_true(
                (isnan(want) and isnan(got)) or got == want,
                String(t"special mismatch at x={xr} got={got} want={want}"),
            )
            continue
        var ref_r = Scalar[dtype](want).cast[.float64]()
        if isinf(ref_r):
            continue
        var err = abs(got - want)
        var ulp_err = err / _ulp[dtype](want)
        var abs_err = err / _ulp[dtype](_scale(xr))
        # Tests the passing condition so that a NaN error fails.
        if not (ulp_err <= max_ulp or abs_err <= max_envelope_ulp):
            raise Error(
                t"j0/y0 mismatch dtype={dtype} is_y0={is_y0} x={xr} got={got}"
                t" want={want} ulp_err={ulp_err} abs_err={abs_err}"
            )
        worst_ulp = max(worst_ulp, min(ulp_err, abs_err))
        worst_abs = max(worst_abs, abs_err)
    print(
        t"dtype={dtype} is_y0={is_y0} max err (ulp, or envelope-ulp near"
        t" zeros)={worst_ulp} max envelope-ulp={worst_abs}"
    )


def test_j0_y0_specials() raises:
    assert_true(j0(Float64(0)) == 1)
    assert_true(j0(Float32(-0.0)) == 1)
    assert_true(j0(inf[.float64]()) == 0)
    assert_true(j0(-inf[.float32]()) == 0)
    assert_true(isnan(j0(nan[.float64]())))
    assert_true(y0(Float64(0)) == -inf[.float64]())
    assert_true(y0(Float32(-0.0)) == -inf[.float32]())
    assert_true(isnan(y0(Float64(-1))))
    assert_true(isnan(y0(-inf[.float32]())))
    assert_true(y0(inf[.float64]()) == 0)
    assert_true(isnan(y0(nan[.float32]())))


def test_j0_y0_accuracy() raises:
    _check[.float64, False]()
    _check[.float64, True]()
    _check[.float32, False]()
    _check[.float32, True]()


def test_j0_y0_simd() raises:
    var x = SIMD[.float64, 4](0.5, 3.0, 2.0e5, 1.0e30)
    var j = j0(x)
    var y = y0(x)
    for i in range(4):
        assert_true(j[i] == j0(x[i]))
        assert_true(y[i] == y0(x[i]))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
