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

from std.math import hypot, inf, isinf, isnan, nan
from std.sys import argv
from std.testing import TestSuite, assert_equal, assert_raises, assert_true
from std.utils.numerics import FPUtils, FlushDenormals, max_finite

from test_utils.math import (
    PairMode,
    hypot_specials,
    libm_call_binary,
    pair_inputs,
)


comptime length = 1 << 12
comptime width = 4


def _check[
    dtype: DType, //, max_ulp: Int
](got: Scalar[dtype], expected: Scalar[dtype]) raises -> Int:
    var ok: Bool
    var d = 0
    var got_bits = Int(got._to_bits_signed())
    var expected_bits = Int(expected._to_bits_signed())
    if isnan(expected):
        ok = isnan(got)
    elif isinf(expected):
        ok = got_bits == expected_bits and got_bits > 0
    elif isnan(got) or isinf(got) or got_bits < 0 or expected_bits < 0:
        ok = False
    elif expected_bits == 0:
        ok = got_bits == 0
    else:
        # Nonnegative finite encodings cannot overflow this signed distance.
        d = abs(got_bits - expected_bits)
        ok = d <= max_ulp
    if not ok:
        raise Error(t"hypot got {got} expected {expected} ({d} ulp)")
    return d


def _test_oracle[dtype: DType]() raises:
    var big = max_finite[dtype]()
    var infinity = inf[dtype]()
    var snan = Scalar[dtype](from_bits=infinity._to_bits_signed() | 1)
    with assert_raises():
        _ = _check[2](infinity, big)
    with assert_raises():
        _ = _check[2](big, infinity)
    with assert_raises():
        _ = _check[2](snan, infinity)
    with assert_raises():
        _ = _check[2](snan, big)
    with assert_raises():
        _ = _check[2](Scalar[dtype](-0.0), Scalar[dtype](0.0))
    with assert_raises():
        _ = _check[2](Scalar[dtype](-2.0), Scalar[dtype](2.0))
    var adjacent = Scalar[dtype](
        from_bits=Scalar[dtype](1)._to_bits_signed() + 1
    )
    assert_equal(_check[2](adjacent, Scalar[dtype](1)), 1)
    assert_equal(_check[2](infinity, infinity), 0)
    assert_equal(_check[2](snan, nan[dtype]()), 0)


def test_oracle() raises:
    _test_oracle[.float32]()
    _test_oracle[.float64]()
    _test_oracle[.float16]()
    _test_oracle[.bfloat16]()


def _run[dtype: DType]() raises where dtype.is_floating_point():
    # Compare representable steps to the rounded libm oracle, not real-valued
    # ULP error. Narrow types can round the float32 paths to adjacent values.
    comptime max_ulp = 2 if dtype in (DType.float32, DType.float64) else 1
    var worst = 0
    var modes: List[PairMode] = [.random_bits, .integer_y, .small_quotients]
    for i, mode in enumerate(modes):
        var inputs = pair_inputs(
            hypot_specials[dtype](), mode, UInt64(1234 + i), length
        )
        ref xs = inputs[0]
        ref ys = inputs[1]
        for base in range(0, length, width):
            var xv = SIMD[dtype, width]()
            var yv = SIMD[dtype, width]()
            comptime for j in range(width):
                xv[j] = xs[base + j]
                yv[j] = ys[base + j]
            var got = hypot(xv, yv)
            comptime for j in range(width):
                var x = xv[j]
                var y = yv[j]
                var expected = libm_call_binary["hypotf", "hypot"](x, y)
                worst = max(worst, _check[max_ulp](got[j], expected))
    print(t"hypot {dtype} max ulp: {worst}")


def test_rounded_oracle_witness() raises:
    # A one-step rounded-oracle error does not imply faithful rounding.
    var x = Float32(from_bits=Int32(0x3F810954))
    var y = Float32(from_bits=Int32(0x3F810358))
    var expected = Float32(from_bits=Int32(0x3FB677F3))
    _ = _check[1](hypot(x, y), expected)


@inline(.never)
def _runtime_hypot[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width], y: SIMD[dtype, width]) -> SIMD[
    dtype, width
] where dtype.is_floating_point():
    return hypot(x, y)


@inline(.never)
def _check_subnormal_normal_results[
    dtype: DType, width: SIMDLength
](
    sign: UInt64,
    expected_sub: UInt64,
    expected_mixed: UInt64,
) raises where dtype.is_floating_point():
    comptime IT = Scalar[FPUtils[dtype].uint_type]
    comptime mant = FPUtils[dtype].mantissa_width()
    var sub = Scalar[dtype](
        from_bits=IT((1 << mant) - 1) | sign.cast[FPUtils[dtype].uint_type]()
    )
    var normal = Scalar[dtype](
        from_bits=IT(1 << mant) | sign.cast[FPUtils[dtype].uint_type]()
    )
    var xs = SIMD[dtype, 4](sub, normal, 3.0, normal)
    var ys = SIMD[dtype, 4](sub, sub, 4.0, 0.0)
    var expected = SIMD[.uint64, 4](
        expected_sub,
        expected_mixed,
        Scalar[dtype](5)._to_bits_signed().cast[.uint64](),
        UInt64(1 << mant),
    )
    comptime if width == 1:
        comptime for lane in range(4):
            assert_equal(
                _runtime_hypot(xs[lane], ys[lane])
                ._to_bits_signed()
                .cast[.uint64](),
                expected[lane],
            )
            assert_equal(
                _runtime_hypot(ys[lane], xs[lane])
                ._to_bits_signed()
                .cast[.uint64](),
                expected[lane],
            )
    else:
        assert_equal(
            _runtime_hypot(xs, ys)._to_bits_signed().cast[.uint64](), expected
        )

    xs = SIMD[dtype, 4](sub, -0.0, -inf[dtype](), nan[dtype]())
    ys = SIMD[dtype, 4](sub, -0.0, nan[dtype](), 1.0)
    var mixed = _runtime_hypot(xs, ys)
    assert_equal(mixed[0]._to_bits_signed().cast[.uint64](), expected_sub)
    assert_equal(mixed[1]._to_bits_signed().cast[.uint64](), UInt64(0))
    assert_equal(
        mixed[2]._to_bits_signed().cast[.uint64](),
        inf[dtype]()._to_bits_signed().cast[.uint64](),
    )
    assert_true(isnan(mixed[3]))


def _test_subnormal_normal_results[
    dtype: DType, width: SIMDLength
](
    expected_sub: UInt64,
    expected_mixed: UInt64,
) raises where dtype.is_floating_point():
    # Runtime signs and noinline calls keep FTZ-sensitive inputs out of folding.
    var sign_bit = UInt64(FPUtils[dtype].sign_mask())
    var sign = (UInt64(len(argv())) & 1) * sign_bit
    _check_subnormal_normal_results[dtype, width](
        sign, expected_sub, expected_mixed
    )
    _check_subnormal_normal_results[dtype, width](
        sign ^ sign_bit, expected_sub, expected_mixed
    )
    with FlushDenormals():
        _check_subnormal_normal_results[dtype, width](
            sign, expected_sub, expected_mixed
        )
        _check_subnormal_normal_results[dtype, width](
            sign ^ sign_bit, expected_sub, expected_mixed
        )
    _check_subnormal_normal_results[dtype, width](
        sign, expected_sub, expected_mixed
    )
    _check_subnormal_normal_results[dtype, width](
        sign ^ sign_bit, expected_sub, expected_mixed
    )


def test_subnormal_inputs_scalar_ftz() raises:
    # Independently rounded boundary norms, not values generated by hypot.
    _test_subnormal_normal_results[.float32, 1](0x00B504F2, 0x00B504F2)
    _test_subnormal_normal_results[.float64, 1](
        0x0016A09E667F3BCB, 0x0016A09E667F3BCC
    )
    _test_subnormal_normal_results[.float16, 1](0x05A7, 0x05A7)
    _test_subnormal_normal_results[.bfloat16, 1](0x00B4, 0x00B4)


def test_subnormal_inputs_simd_ftz() raises:
    _test_subnormal_normal_results[.float32, 4](0x00B504F2, 0x00B504F2)
    _test_subnormal_normal_results[.float64, 4](
        0x0016A09E667F3BCB, 0x0016A09E667F3BCC
    )
    _test_subnormal_normal_results[.float16, 4](0x05A7, 0x05A7)
    _test_subnormal_normal_results[.bfloat16, 4](0x00B4, 0x00B4)


def _test_near_overflow[dtype: DType]() raises where dtype.is_floating_point():
    # Equal operands near max_finite / sqrt(2) put the true result within a few
    # ulp of the overflow threshold, so both finite and inf outcomes occur.
    comptime max_ulp = 2 if dtype in (DType.float32, DType.float64) else 1
    var near = Scalar[dtype](Float64(max_finite[dtype]()) * 0.7071067811865476)
    comptime I = Scalar[FPUtils[dtype].integral_type]
    var base = near._to_bits_signed()
    var saw_inf = False
    var saw_finite = False
    for dx in range(-8, 9):
        for dy in range(-8, 9):
            var x = Scalar[dtype](from_bits=base + I(dx))
            var y = Scalar[dtype](from_bits=base + I(dy))
            var got = hypot(SIMD[dtype, width](x), SIMD[dtype, width](y))
            var expected = libm_call_binary["hypotf", "hypot"](x, y)
            for j in range(width):
                # Near the overflow threshold the kernel and libm may each
                # round to max_finite or inf, or differ by an ulp below
                # max_finite (glibc: max_finite, kernel: one ulp lower), so
                # accept inf or a finite value within max_ulp of max_finite.
                if isinf(expected) or expected == max_finite[dtype]():
                    var max_bits = Int(max_finite[dtype]()._to_bits_signed())
                    assert_true(
                        isinf(got[j])
                        or (
                            got[j] > 0
                            and Int(got[j]._to_bits_signed())
                            >= max_bits - max_ulp
                        )
                    )
                else:
                    _ = _check[max_ulp](got[j], expected)
            if isinf(expected):
                saw_inf = True
            else:
                saw_finite = True
    assert_true(saw_inf and saw_finite)


def test_near_overflow() raises:
    _test_near_overflow[.float32]()
    _test_near_overflow[.float64]()
    _test_near_overflow[.float16]()
    _test_near_overflow[.bfloat16]()


def test_float32() raises:
    _run[.float32]()


def test_float64() raises:
    _run[.float64]()


def test_float16() raises:
    _run[.float16]()


def test_bfloat16() raises:
    _run[.bfloat16]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
