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

from std.math import inf, isnan, log10, nan, nextafter
from std.math.math import _call_libm
from std.memory import bitcast
from std.testing import assert_equal, assert_true, TestSuite
from std.utils.numerics import max_finite

from test_utils.math import (
    f32_max_subnormal_bits,
    f32_min_subnormal_bits,
    f64_max_subnormal_bits,
    f64_min_subnormal_bits,
    log10_f32_inputs,
    log10_f64_inputs,
    ulp_diff,
)


def test_ulp_diff() raises:
    assert_equal(ulp_diff(Float32(1), Float32(1)), 0)
    assert_equal(ulp_diff(Float32(1), nextafter(Float32(1), Float32(2))), 1)
    assert_equal(ulp_diff(Float32(-1), nextafter(Float32(-1), Float32(-2))), 1)
    assert_equal(ulp_diff(Float64(1), Float64(1.0000000000000022)), 10)
    assert_equal(ulp_diff(Float32(0), Float32(-0.0)), 0)
    assert_equal(ulp_diff(inf[.float64](), inf[.float64]()), 0)
    assert_equal(ulp_diff(nan[.float64](), nan[.float64]()), 0)
    # Opposite signs must never pass: the ordered-integer difference of
    # log10(0.01) = -2 and a wrong +2 overflows Int64 and used to pass.
    assert_equal(ulp_diff(Float64(2), Float64(-2)), Int.MAX)
    assert_equal(ulp_diff(Float32(1), Float32(-1)), Int.MAX)
    assert_equal(ulp_diff(max_finite[.float32](), inf[.float32]()), Int.MAX)
    assert_equal(ulp_diff(-max_finite[.float64](), -inf[.float64]()), Int.MAX)
    assert_equal(ulp_diff(inf[.float32](), -inf[.float32]()), Int.MAX)
    assert_equal(ulp_diff(nan[.float32](), Float32(1)), Int.MAX)
    assert_equal(ulp_diff(Float32(0), -Float32(from_bits=Int32(1))), Int.MAX)


def _same[
    dtype: DType, //
](a: Scalar[dtype], b: Scalar[dtype]) -> Bool where dtype.is_floating_point():
    return (isnan(a) and isnan(b)) or a.to_bits() == b.to_bits()


def test_log10_float64() raises:
    # glibc's log10 is not correctly rounded and is up to 2 ULP from the true
    # value near 1 (e.g. log10(0.9999999999999691)), so allow 2 ULP vs libm.
    comptime max_ulp_log10 = 2
    # 10^n is exactly representable in float64 for n <= 22.
    comptime max_exact_pow10 = 22
    for x in log10_f64_inputs():
        var expected = _call_libm["log10"](x)
        var got = log10(x)
        var d = ulp_diff(got, expected)
        assert_true(
            d <= max_ulp_log10,
            String(t"log10({x}) = {got}, expected {expected} ({d} ulp)"),
        )
    var p: Float64 = 1.0
    for n in range(max_exact_pow10 + 1):
        assert_equal(log10(p), Float64(n))
        p *= 10.0
    # SIMD lanes take the same path as scalars.
    var v = SIMD[.float64, 4](0.0, 1.0e-310, 1000.0, -2.0)
    var r = log10(v)
    for i in range(4):
        assert_true(_same(r[i], log10(v[i])))


def test_log10_float32() raises:
    # Against float64 libm rounded to float32; that reference is itself
    # within 1 ULP of the true value.
    for x in log10_f32_inputs():
        var expected = _call_libm["log10"](x.cast[.float64]()).cast[.float32]()
        var got = log10(x)
        var d = ulp_diff(got, expected)
        assert_true(
            d <= 1,
            String(t"log10({x}) = {got}, expected {expected} ({d} ulp)"),
        )
    var p: Float32 = 1.0
    for n in range(11):
        assert_equal(log10(p), Float32(n))
        p *= 10.0
    assert_equal(log10(Float16(100.0)), Float16(2.0))
    assert_equal(log10(BFloat16(0.0)), -inf[.bfloat16]())
    var v = SIMD[.float32, 4](0.0, 1.0e-40, 1000.0, -2.0)
    var r = log10(v)
    for i in range(4):
        assert_true(_same(r[i], log10(v[i])))


def test_log10_float16_exhaustive() raises:
    # Every float16 bit pattern against float64 libm rounded to float16.
    var max_d = 0
    for bits in range(1 << 16):
        var x = bitcast[.float16](UInt16(bits))
        var expected = _call_libm["log10"](x.cast[.float64]()).cast[.float16]()
        var got = log10(x)
        if isnan(expected):
            assert_true(isnan(got), String(t"log10({x}) = {got}, expected NaN"))
            continue
        var d = ulp_diff(got, expected)
        max_d = max(max_d, d)
        assert_true(
            d <= 1,
            String(t"log10({x}) = {got}, expected {expected} ({d} ulp)"),
        )
    print("log10 float16 max ulp vs libm:", max_d)


def test_log10_subnormal() raises:
    """The log reduction reads a subnormal's exponent from its bits, so the
    smallest and largest subnormal inputs survive flush-to-zero targets. The
    GPU suite repeats this on device; it does not run on macOS."""
    var x32: List[Float32] = [
        bitcast[.float32](f32_min_subnormal_bits),
        bitcast[.float32](f32_max_subnormal_bits),
    ]
    for x in x32:
        var expected = _call_libm["log10"](x.cast[.float64]()).cast[.float32]()
        var got = log10(x)
        var d = ulp_diff(got, expected)
        assert_true(
            d <= 1, String(t"log10({x}) = {got}, expected {expected} ({d} ulp)")
        )
    var x64: List[Float64] = [
        bitcast[.float64](f64_min_subnormal_bits),
        bitcast[.float64](f64_max_subnormal_bits),
    ]
    for x in x64:
        var expected = _call_libm["log10"](x)
        var got = log10(x)
        var d = ulp_diff(got, expected)
        assert_true(
            d <= 1, String(t"log10({x}) = {got}, expected {expected} ({d} ulp)")
        )
    # A negative subnormal is still negative, not a zero.
    assert_true(isnan(log10(-bitcast[.float32](f32_min_subnormal_bits))))
    assert_true(isnan(log10(-bitcast[.float64](f64_min_subnormal_bits))))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
