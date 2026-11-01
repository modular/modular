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

from std.builtin.dtype import _integral_type_of
from std.math import log1p
from std.math.math import _call_libm
from std.sys import bit_width_of
from std.testing import TestSuite, assert_equal, assert_true
from std.utils.numerics import FPUtils, inf, isinf, isnan, nan, neg_inf
from test_utils.math import ulp_distance


def _scaled[
    dtype: DType, //
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


def _assert_ulp_vs_libm[
    dtype: DType, //, name: StaticString
](
    x: Scalar[dtype], got: Scalar[dtype], max_ulp: Int
) raises where dtype.is_floating_point():
    var want = _call_libm[name](x)
    var msg = String(name, "(", x, ") = ", got, ", libm ", want)
    if isnan(want) or isnan(got):
        assert_true(isnan(want) and isnan(got), msg)
        return
    if isinf(want) or isinf(got):
        assert_equal(got.to_bits(), want.to_bits(), msg)
        return
    comptime sign_shift = UInt64(bit_width_of[dtype]() - 1)
    assert_equal(
        got.to_bits().cast[.uint64]() >> sign_shift,
        want.to_bits().cast[.uint64]() >> sign_shift,
        msg,
    )
    var d = ulp_distance(got, want)
    assert_true(
        d <= max_ulp,
        String(name, "(", x, ") = ", got, ", libm ", want, " (", d, " ulp)"),
    )


def _assert_same[
    dtype: DType, //
](got: Scalar[dtype], want: Scalar[dtype], msg: String) raises:
    """Asserts bitwise equality, so NaN matches NaN and -0 differs from +0."""
    assert_equal(got.to_bits(), want.to_bits(), msg)


def _test_log1p_special[dtype: DType]() raises where dtype.is_floating_point():
    comptime T = Scalar[dtype]
    var name = String(dtype)
    _assert_same(log1p(T(0)), T(0), name + " log1p(0)")
    _assert_same(log1p(T(-0.0)), T(-0.0), name + " log1p(-0)")
    _assert_same(log1p(T(-1)), neg_inf[dtype](), name + " log1p(-1)")
    _assert_same(log1p(inf[dtype]()), inf[dtype](), name + " log1p(inf)")
    assert_true(isnan(log1p(T(-2))), name + " log1p(-2)")
    assert_true(isnan(log1p(neg_inf[dtype]())), name + " log1p(-inf)")
    assert_true(isnan(log1p(nan[dtype]())), name + " log1p(nan)")


def test_log1p_special_values() raises:
    _test_log1p_special[.float64]()
    _test_log1p_special[.float32]()
    _test_log1p_special[.float16]()
    _test_log1p_special[.bfloat16]()


def _test_log1p_ulp[dtype: DType]() raises where dtype.is_floating_point():
    comptime T = Scalar[dtype]
    # Powers of two on both sides of zero, approaching -1 from above, and
    # large values.
    comptime min_exp = -1074 if dtype == .float64 else -149
    for k in range(min_exp, 0):
        var t = _scaled(T(1.2345678901234567), k)
        _assert_ulp_vs_libm["log1p"](t, log1p(t), 1)
        _assert_ulp_vs_libm["log1p"](-t, log1p(-t), 1)
        var near_neg1 = t - 1
        _assert_ulp_vs_libm["log1p"](near_neg1, log1p(near_neg1), 1)
    comptime max_exp = 1024 if dtype == .float64 else 128
    for k in range(0, max_exp):
        var t = _scaled(T(1.2345678901234567), k)
        _assert_ulp_vs_libm["log1p"](t, log1p(t), 1)
    comptime n = 20_000
    for i in range(n):
        var x = T(i) / T(n / 8) - T(0.99)
        _assert_ulp_vs_libm["log1p"](x, log1p(x), 1)


def test_log1p_ulp() raises:
    _test_log1p_ulp[.float64]()
    _test_log1p_ulp[.float32]()


def _test_log1p_table[
    dtype: DType, //
](table: List[Scalar[dtype]]) raises where dtype.is_floating_point():
    """Checks `(x, log1p(x))` pairs, stored flat, against correct rounding."""
    for i in range(0, len(table), 2):
        var x = table[i]
        var want = table[i + 1]
        var got = log1p(x)
        var d = ulp_distance(got, want)
        assert_true(
            d <= 1,
            String("log1p(", x, ") = ", got, ", want ", want, " (", d, " ulp)"),
        )


def test_log1p_reference_table() raises:
    # Correctly rounded results generated with mpmath at 300-bit precision.
    var f64: List[Float64] = [
        -0.9999999,
        -16.118095651484676,
        -0.999,
        -6.907755278982136,
        -0.99,
        -4.605170185988091,
        -0.75,
        -1.3862943611198906,
        -0.5,
        -0.6931471805599453,
        -0.4142135623730951,
        -0.5347999967395705,
        -0.25,
        -0.2876820724517809,
        -0.1,
        -0.10536051565782631,
        -0.001,
        -0.0010005003335835335,
        -1e-08,
        -1.0000000050000001e-08,
        1e-300,
        1e-300,
        1e-17,
        1e-17,
        1e-08,
        9.999999950000001e-09,
        0.0009765625,
        0.0009760859730554589,
        0.1,
        0.09531017980432487,
        0.25,
        0.22314355131420976,
        0.4142135623730951,
        0.3465735902799727,
        0.5,
        0.4054651081081644,
        0.75,
        0.5596157879354227,
        0.9999999999999999,
        0.6931471805599453,
        1.0,
        0.6931471805599453,
        1.5,
        0.9162907318741551,
        2.0,
        1.0986122886681098,
        2.5,
        1.252762968495368,
        3.0,
        1.3862943611198906,
        10.0,
        2.3978952727983707,
        12345.678,
        9.42114231801815,
        10000000000.0,
        23.025850930040455,
        1e100,
        230.25850929940458,
        1e300,
        690.7755278982137,
    ]
    _test_log1p_table(f64)
    var f32: List[Float32] = [
        -0.9999999,
        -15.942385,
        -0.999,
        -6.9077682,
        -0.99,
        -4.605171,
        -0.75,
        -1.3862944,
        -0.5,
        -0.6931472,
        -0.41421357,
        -0.5348,
        -0.25,
        -0.2876821,
        -0.1,
        -0.105360515,
        -0.001,
        -0.0010005004,
        -1e-08,
        -1e-08,
        1e-17,
        1e-17,
        1e-08,
        1e-08,
        0.0009765625,
        0.00097608595,
        0.1,
        0.09531018,
        0.25,
        0.22314355,
        0.41421357,
        0.3465736,
        0.5,
        0.4054651,
        0.75,
        0.5596158,
        1.0,
        0.6931472,
        1.5,
        0.91629076,
        2.0,
        1.0986123,
        2.5,
        1.2527629,
        3.0,
        1.3862944,
        10.0,
        2.3978953,
        12345.678,
        9.421143,
        10000000000.0,
        23.02585,
    ]
    _test_log1p_table(f32)


def _test_log1p_simd[dtype: DType]() raises where dtype.is_floating_point():
    # Mix the near-zero, k = 1, k >= 2 and near-minus-one reductions.
    var x = SIMD[dtype, 4](0.1, 0.9, 3.0, -0.999)
    var got = log1p(x)
    for i in range(4):
        _assert_ulp_vs_libm["log1p"](x[i], got[i], 1)


def test_log1p_simd() raises:
    _test_log1p_simd[.float64]()
    _test_log1p_simd[.float32]()


def test_log1p_narrow() raises:
    comptime n = 2_000
    for i in range(n):
        var x = Float32(i) / Float32(n / 8) - 0.99
        var h = x.cast[.float16]()
        _assert_ulp_vs_libm["log1p"](h, log1p(h), 1)
        var b = x.cast[.bfloat16]()
        _assert_ulp_vs_libm["log1p"](b, log1p(b), 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
