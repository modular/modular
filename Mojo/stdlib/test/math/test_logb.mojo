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

from std.math import inf, isnan, logb, nan
from std.math.math import _call_libm
from std.memory import bitcast
from std.sys import bit_width_of
from std.testing import assert_equal, assert_true, TestSuite
from std.utils.numerics import FPUtils

from test_utils.math import xorshift64

comptime length = 16384


def _same[
    dtype: DType
](a: Scalar[dtype], b: Scalar[dtype]) -> Bool where dtype.is_floating_point():
    return (isnan(a) and isnan(b)) or a.to_bits() == b.to_bits()


def _logb_inputs[
    dtype: DType
]() -> List[Scalar[dtype]] where dtype.is_floating_point():
    comptime I = FPUtils[dtype].uint_type
    comptime nbits = bit_width_of[dtype]()
    var v = List[Scalar[dtype]](capacity=length)
    v.append(0.0)
    v.append(-0.0)
    v.append(inf[dtype]())
    v.append(-inf[dtype]())
    v.append(nan[dtype]())
    v.append(-nan[dtype]())
    comptime mantissa_width = FPUtils[dtype].mantissa_width()
    comptime mantissa_mask = (Scalar[I](1) << Scalar[I](mantissa_width)) - 1
    comptime sign_bit = Scalar[I](1) << Scalar[I](nbits - 1)
    # Every normal exponent field, both signs, with the smallest and largest
    # mantissa.
    for e in range(1, (1 << FPUtils[dtype].exponent_width()) - 1):
        var top = Scalar[I](e) << Scalar[I](mantissa_width)
        for bits in [top, top | mantissa_mask]:
            v.append(bitcast[dtype](bits))
            v.append(bitcast[dtype](bits | sign_bit))
    # Single-bit and all-ones-below patterns: subnormals at every bit position
    # and powers of two across the encoding.
    for i in range(nbits - 1):
        var b = Scalar[I](1) << Scalar[I](i)
        v.append(bitcast[dtype](b))
        v.append(bitcast[dtype](b | sign_bit))
        v.append(bitcast[dtype](b | (b - 1)))
    # 2^64 / golden ratio, the Fibonacci-hashing multiplier.
    comptime golden_ratio_u64: UInt64 = 0x9E37_79B9_7F4A_7C15
    var state = golden_ratio_u64
    while len(v) < length:
        v.append(bitcast[dtype](xorshift64(state).cast[I]()))
    return v^


def _test_logb[dtype: DType]() raises where dtype.is_floating_point():
    for x in _logb_inputs[dtype]():
        var expected = _call_libm["logb"](x)
        var got = logb(x)
        assert_true(
            _same(got, expected),
            String(t"logb[{dtype}]({x}) = {got}, expected {expected}"),
        )


def test_logb() raises:
    _test_logb[.float32]()
    _test_logb[.float16]()
    _test_logb[.bfloat16]()
    _test_logb[.float64]()
    var v = SIMD[.float32, 4](0.0, 1.0e-40, 8.0, -0.5)
    assert_equal(
        logb(v), SIMD[.float32, 4](-inf[.float32](), -133.0, 3.0, -1.0)
    )
    # NaN payloads are kept but not quieted, so compare NaN-ness, not bits.
    # Smallest signaling NaN: all-ones exponent, only the lowest mantissa bit
    # set.
    comptime f32_snan_bits: UInt32 = 0x7F80_0001
    comptime f64_snan_bits: UInt64 = 0x7FF0_0000_0000_0001
    var snan = bitcast[.float32](f32_snan_bits)
    assert_true(isnan(logb(snan)) and isnan(logb(-snan)))
    assert_true(isnan(logb(bitcast[.float64](f64_snan_bits))))


def test_logb_simd_mixed() raises:
    var v = SIMD[.float32, 8](
        0.0,
        -0.0,
        bitcast[.float32](UInt32(1)),
        1.0e-40,
        inf[.float32](),
        -inf[.float32](),
        nan[.float32](),
        -0.25,
    )
    var r = logb(v)
    assert_equal(r[0], -inf[.float32]())
    assert_equal(r[1], -inf[.float32]())
    assert_equal(r[2], -149.0)
    assert_equal(r[3], -133.0)
    assert_equal(r[4], inf[.float32]())
    assert_equal(r[5], inf[.float32]())
    assert_true(isnan(r[6]))
    assert_equal(r[7], -2.0)


def _test_logb_cast_fallback[
    dtype: DType
]() raises where dtype.is_floating_point():
    # Dtypes without a native bit-level path go through float32. Zero is left
    # out because the narrowing cast of the -inf result saturates.
    var v = SIMD[dtype, 4](2.0, 1.0, 6.0, 0.25)
    var expected = SIMD[.float32, 4](1.0, 0.0, 2.0, -2.0)
    assert_equal(logb(v).cast[.float32](), expected)
    assert_equal(logb(Scalar[dtype](-8.0)).cast[.float32](), 3.0)


def test_logb_cast_fallback() raises:
    _test_logb_cast_fallback[.float8_e5m2]()
    _test_logb_cast_fallback[.float8_e4m3fn]()


def _test_logb_subnormal[dtype: DType]() raises where dtype.is_floating_point():
    comptime I = FPUtils[dtype].uint_type
    comptime mantissa_width = FPUtils[dtype].mantissa_width()
    comptime emin = 1 - FPUtils[dtype].exponent_bias()
    comptime sign_bit = Scalar[I](1) << Scalar[I](bit_width_of[dtype]() - 1)
    for i in range(mantissa_width):
        var b = Scalar[I](1) << Scalar[I](i)
        var expected = Scalar[dtype](emin - mantissa_width + i)
        for x in [b, b | (b - 1), b | sign_bit | (b >> 1)]:
            var v = bitcast[dtype](x)
            assert_equal(logb(v), expected, String(t"logb[{dtype}]({v})"))


def test_logb_subnormal() raises:
    _test_logb_subnormal[.float32]()
    _test_logb_subnormal[.float64]()
    assert_equal(logb(Float32(1.0e-40)), -133.0)
    assert_equal(logb(bitcast[.float32](UInt32(1))), -149.0)
    assert_equal(logb(Float64(1.0e-310)), -1030.0)
    assert_equal(logb(bitcast[.float64](UInt64(1))), -1074.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
