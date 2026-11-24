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

# Inputs where acosh and asinh once had more than 1.5 ulp of error,
# checked against correctly rounded results from mpmath.

from std.math import acosh, asinh, inf, isinf, isnan, nan
from std.math.math import _call_libm, _log_accurate
from std.testing import assert_equal, assert_false, assert_true, TestSuite
from std.utils.numerics import FPUtils, max_finite


def _within_one_ulp[
    dtype: DType, //
](
    got: Scalar[dtype], want: Scalar[dtype]
) -> Bool where dtype.is_floating_point():
    if isnan(got) or isnan(want):
        return isnan(got) and isnan(want)
    if isinf(got) or isinf(want):
        return got == want
    if FPUtils[dtype].get_sign(got) != FPUtils[dtype].get_sign(want):
        return False
    var got_bits = FPUtils[dtype].bitcast_to_uint(got).cast[.uint64]()
    var want_bits = FPUtils[dtype].bitcast_to_uint(want).cast[.uint64]()
    return max(got_bits, want_bits) - min(got_bits, want_bits) <= 1


def test_acosh_hard_cases() raises:
    var x32: List[Float32] = [
        1.0070128440856934,
        1.0001211166381836,
        1.007747769355774,
    ]
    var want32: List[Float32] = [0.118360996, 0.015563688, 0.12440085]
    for i in range(len(x32)):
        assert_true(
            _within_one_ulp(acosh(x32[i]), want32[i]), String(t"x = {x32[i]}")
        )
    var x64: List[Float64] = [
        1.025717012207915,
        1.0000303429856943,
        1.0000001191542771,
    ]
    var want64: List[Float64] = [
        0.22630746557989662,
        0.007790100416046396,
        0.00048816856667639394,
    ]
    for i in range(len(x64)):
        assert_true(
            _within_one_ulp(acosh(x64[i]), want64[i]), String(t"x = {x64[i]}")
        )
    assert_equal(acosh(Float32(1)), 0)
    assert_equal(acosh(Float64(1)), 0)


def test_asinh_hard_cases() raises:
    assert_true(
        _within_one_ulp(asinh(Float32(0.2525882124900818)), Float32(0.24997663))
    )


def _check_comparator[dtype: DType]() raises where dtype.is_floating_point():
    comptime T = Scalar[dtype]
    var infinity = inf[dtype]()
    var largest = max_finite[dtype]()
    assert_true(_within_one_ulp(infinity, infinity))
    assert_true(_within_one_ulp(-infinity, -infinity))
    assert_true(_within_one_ulp(nan[dtype](), nan[dtype]()))
    assert_false(_within_one_ulp(largest, infinity))
    assert_false(_within_one_ulp(infinity, largest))
    assert_false(_within_one_ulp(-largest, -infinity))
    assert_false(_within_one_ulp(infinity, -infinity))
    assert_false(_within_one_ulp(nan[dtype](), T(1)))
    assert_false(_within_one_ulp(T(1), nan[dtype]()))
    assert_false(_within_one_ulp(T(0), T(-0.0)))
    assert_true(_within_one_ulp(T(1), T(1)))
    assert_false(_within_one_ulp(T(2), T(1)))


def test_comparator() raises:
    _check_comparator[.float32]()
    _check_comparator[.float64]()
    _check_comparator[.float16]()
    _check_comparator[.bfloat16]()


def _check_specials[dtype: DType]() raises where dtype.is_floating_point():
    comptime T = Scalar[dtype]
    var infinity = inf[dtype]()
    var not_a_number = nan[dtype]()
    assert_true(isnan(acosh(not_a_number)))
    assert_true(isnan(acosh(-infinity)))
    assert_true(isnan(acosh(T(0))))
    assert_true(isnan(acosh(T(0.5))))
    assert_equal(acosh(infinity), infinity)
    assert_equal(FPUtils[dtype].bitcast_to_uint(acosh(T(1))), 0)
    assert_true(isnan(asinh(not_a_number)))
    assert_equal(asinh(infinity), infinity)
    assert_equal(asinh(-infinity), -infinity)
    var zeros: List[T] = [0, -0.0]
    for x in zeros:
        assert_equal(
            FPUtils[dtype].bitcast_to_uint(asinh(x)),
            FPUtils[dtype].bitcast_to_uint(x),
        )


def test_specials() raises:
    _check_specials[.float32]()
    _check_specials[.float64]()
    _check_specials[.float16]()
    _check_specials[.bfloat16]()


def _check_branches[dtype: DType]() raises where dtype.is_floating_point():
    comptime T = Scalar[dtype]
    comptime is_f64 = dtype == .float64
    var tiny = T(1.862645149230957e-09 if is_f64 else 1.220703125e-04)
    var tiny_inputs: List[T] = [tiny, -tiny]
    for x in tiny_inputs:
        assert_equal(
            FPUtils[dtype].bitcast_to_uint(asinh(x)),
            FPUtils[dtype].bitcast_to_uint(x),
        )
    # Correctly rounded mpmath references for the middle and huge branches.
    assert_true(_within_one_ulp(acosh(T(3)), T(1.762747174039086)))
    assert_true(_within_one_ulp(asinh(T(3)), T(1.8184464592320668)))
    assert_true(_within_one_ulp(asinh(T(-3)), T(-1.8184464592320668)))
    var huge = T(268435456 if is_f64 else 4096)
    var want_acosh = T(20.101268236238414 if is_f64 else 9.010913332378127)
    var want_asinh = T(20.101268236238414 if is_f64 else 9.01091336218045)
    assert_true(_within_one_ulp(acosh(huge), want_acosh))
    assert_true(_within_one_ulp(asinh(huge), want_asinh))
    assert_true(_within_one_ulp(asinh(-huge), -want_asinh))


def test_branches() raises:
    _check_branches[.float32]()
    _check_branches[.float64]()
    _check_branches[.float16]()
    _check_branches[.bfloat16]()


def _check_asinh_identity[
    dtype: DType
]() raises where dtype.is_floating_point():
    comptime T = Scalar[dtype]
    comptime FP = FPUtils[dtype]
    comptime minimum_normal_bits = 1 << FP.mantissa_width()
    # Bit construction keeps the smallest, interior, and largest subnormals
    # nonzero before they reach asinh, including the promoted narrow types.
    var magnitudes: List[Int] = [
        1,
        minimum_normal_bits // 2,
        minimum_normal_bits - 1,
        minimum_normal_bits,
    ]
    for magnitude in magnitudes:
        var positive = FP.bitcast_from_integer(magnitude)
        var negative = FP.set_sign(positive, True)
        var signed_inputs: List[T] = [positive, negative]
        for x in signed_inputs:
            assert_equal(FP.bitcast_to_uint(asinh(x)), FP.bitcast_to_uint(x))
        var lanes = SIMD[dtype, 4](positive, 1, negative, -3)
        var got = asinh(lanes)
        assert_equal(FP.bitcast_to_uint(got[0]), FP.bitcast_to_uint(positive))
        assert_equal(FP.bitcast_to_uint(got[2]), FP.bitcast_to_uint(negative))
        assert_true(_within_one_ulp(got[1], T(0.881373587019543)))
        assert_true(_within_one_ulp(got[3], T(-1.8184464592320668)))


def test_asinh_tiny_identity() raises:
    _check_asinh_identity[.float32]()
    _check_asinh_identity[.float64]()
    _check_asinh_identity[.float16]()
    _check_asinh_identity[.bfloat16]()


def test_log_subnormals() raises:
    """The shared log reduction reads a subnormal's exponent from its bits, so
    the smallest and largest subnormal inputs survive flush-to-zero targets.
    The GPU suite repeats this on device; it does not run on macOS."""
    var x32: List[Float32] = [
        Float32(from_bits=Int32(1)),
        1,
        Float32(from_bits=Int32(0x007FFFFF)),
        2,
    ]
    var want32: List[Float32] = [-103.278931, 0, -87.3365479, 0.693147182]
    var got32 = _log_accurate(SIMD[.float32, 4](x32[0], x32[1], x32[2], x32[3]))
    for i in range(len(x32)):
        assert_true(
            _within_one_ulp(_log_accurate(x32[i]), want32[i]),
            String(t"x = {x32[i]}"),
        )
        assert_true(_within_one_ulp(got32[i], want32[i]))
    var x64: List[Float64] = [
        Float64(from_bits=Int64(1)),
        1,
        Float64(from_bits=Int64(0x000FFFFFFFFFFFFF)),
        2,
    ]
    var want64: List[Float64] = [
        -744.4400719213812,
        0,
        -708.3964185322641,
        0.6931471805599453,
    ]
    var got64 = _log_accurate(SIMD[.float64, 4](x64[0], x64[1], x64[2], x64[3]))
    for i in range(len(x64)):
        assert_true(
            _within_one_ulp(_log_accurate(x64[i]), want64[i]),
            String(t"x = {x64[i]}"),
        )
        assert_true(_within_one_ulp(got64[i], want64[i]))


def _bit_distance(got: Float32, want: Float32) -> Int:
    var got_bits = FPUtils[.float32].bitcast_to_uint(got)
    var want_bits = FPUtils[.float32].bitcast_to_uint(want)
    return Int(max(got_bits, want_bits) - min(got_bits, want_bits))


def test_float32_sweep() raises:
    """Compares float32 results against libm evaluated in float64 and rounded
    once to float32, so the reference does not share the float32 code path."""
    comptime count = 4096
    var worst_acosh = 0
    var worst_asinh = 0
    for i in range(count):
        var s = Float64(i) / Float64(count)
        # Log-spaced offsets from 1 for acosh and magnitudes for asinh cover
        # the near-one, middle, and huge branches.
        var xa = Float32(1.0 + _call_libm["exp"](Float64(-16.0 + 32.0 * s)))
        var want_a = Float32(_call_libm["acosh"](Float64(xa)))
        worst_acosh = max(worst_acosh, _bit_distance(acosh(xa), want_a))
        var xs = Float32(_call_libm["exp"](Float64(-12.0 + 100.0 * s)))
        for sign in [Float32(1), Float32(-1)]:
            var x = sign * xs
            var want_s = Float32(_call_libm["asinh"](Float64(x)))
            if x == want_s:
                continue
            worst_asinh = max(worst_asinh, _bit_distance(asinh(x), want_s))
    assert_true(worst_acosh <= 2, String(t"acosh max ulp = {worst_acosh}"))
    assert_true(worst_asinh <= 2, String(t"asinh max ulp = {worst_asinh}"))
    print("float32 sweep max ulp: acosh", worst_acosh, "asinh", worst_asinh)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
