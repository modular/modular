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

from std.testing import assert_equal, TestSuite

from std.utils.numerics import isnan


def test_abs() raises:
    assert_equal(0, abs(0))
    assert_equal(1, abs(1))
    assert_equal(1, abs(-1))

    var lhs = SIMD[.int32, 4](1, -2, 3, -4)
    var expected = SIMD[.int32, 4](1, 2, 3, 4)
    assert_equal(expected, abs(lhs))


def test_divmod() raises:
    var t = divmod(0, 1)
    assert_equal(0, t[0])
    assert_equal(0, t[1])
    t = divmod(1, 1)
    assert_equal(1, t[0])
    assert_equal(0, t[1])
    t = divmod(1, 2)
    assert_equal(0, t[0])
    assert_equal(1, t[1])
    t = divmod(4, 3)
    assert_equal(1, t[0])
    assert_equal(1, t[1])


def test_min() raises:
    assert_equal(-2, min(-2, -1))
    assert_equal(-1, min(0, -1))
    assert_equal(0, min(0, 1))
    assert_equal(1, min(1, 42))

    assert_equal(UInt(0), min(UInt(0), UInt(1)))
    assert_equal(
        UInt(1),
        min(UInt(1), UInt(42)),
    )

    comptime F = SIMD[.float32, 4]
    var f = F(-10.5, -5.0, 5.0, 10.0)
    assert_equal(min(f, F(-9.0, -6.0, -4.0, 10.5)), F(-10.5, -6.0, -4.0, 10.0))
    assert_equal(min(f, -4.0), F(-10.5, -5.0, -4.0, -4.0))

    comptime I = SIMD[.int32, 4]
    var i = I(-10, -5, 5, 10)
    assert_equal(min(i, I(-9, -6, -4, 11)), I(-10, -6, -4, 10))
    assert_equal(min(i, -4), I(-10, -5, -4, -4))

    assert_equal(min(1), 1)
    assert_equal(min(1, 2, 3, 4), 1)
    assert_equal(min(500, 1, 2, 3, 4), 1)


def test_max() raises:
    assert_equal(-1, max(-2, -1))
    assert_equal(0, max(0, -1))
    assert_equal(1, max(0, 1))
    assert_equal(2, max(2, 1))

    assert_equal(UInt(1), max(UInt(0), UInt(1)))
    assert_equal(UInt(2), max(UInt(1), UInt(2)))

    comptime F = SIMD[.float32, 4]
    var f = F(-10.5, -5.0, 5.0, 10.0)
    assert_equal(max(f, F(-9.0, -6.0, -4.0, 10.5)), F(-9.0, -5.0, 5.0, 10.5))
    assert_equal(max(f, -4.0), F(-4.0, -4.0, 5.0, 10.0))

    comptime I = SIMD[.int32, 4]
    var i = I(-10, -5, 5, 10)
    assert_equal(max(i, I(-9, -6, -4, 11)), I(-9, -5, 5, 11))
    assert_equal(max(i, -4), I(-4, -4, 5, 10))

    assert_equal(max(1), 1)
    assert_equal(max(1, 2, 3, 4), 4)
    assert_equal(max(-10, 2, 3, 4, -10), 4)


def test_round() raises:
    assert_equal(0.0, round(0.0))
    assert_equal(1.0, round(1.0))
    assert_equal(1.0, round(1.1))
    assert_equal(1.0, round(1.4))
    assert_equal(2.0, round(1.5))
    assert_equal(2.0, round(2.0))
    assert_equal(1.0, round(1.4, 0))
    assert_equal(2.0, round(2.5))
    assert_equal(-2.0, round(-2.5))

    assert_equal(1.5, round(1.5, 1))
    assert_equal(1.61, round(1.613, 2))

    var lhs = SIMD[.float32, 4](1.1, 1.5, 1.9, 2.0)
    var expected = SIMD[.float32, 4](1.0, 2.0, 2.0, 2.0)
    assert_equal(expected, round(lhs))

    # Ensure that round works on float literal
    comptime r1 = round(2.3)
    assert_equal(r1, 2.0)
    comptime r2 = round(2.3324, 2)
    assert_equal(r2, 2.33)


def test_pow() raises:
    comptime F = SIMD[.float32, 4]
    var base = F(0.0, 1.0, 2.0, 3.0)
    assert_equal(pow(base, Float32(2.0)), F(0.0, 1.0, 4.0, 9.0))
    assert_equal(pow(base, Int(2)), F(0.0, 1.0, 4.0, 9.0))
    comptime I = SIMD[.int32, 4]
    assert_equal(pow(I(0, 1, 2, 3), Int(2)), I(0, 1, 4, 9))


# Expected values are Python's `pow(base, exp, mod)`.
def test_pow_mod() raises:
    assert_equal(pow(3, 4, 5), 1)
    assert_equal(pow(2, 10, 1000), 24)
    assert_equal(pow(7, 0, 13), 1)
    assert_equal(pow(5, 0, 1), 0)
    assert_equal(pow(17, 1, 5), 2)

    # The result takes the sign of `mod`.
    assert_equal(pow(-2, 3, 5), 2)
    assert_equal(pow(2, 3, -5), -2)
    assert_equal(pow(-2, 3, -5), -3)
    assert_equal(pow(7, 0, -5), -4)
    assert_equal(pow(17, 1, -5), -3)

    # A negative exponent uses the modular inverse.
    assert_equal(pow(3, -1, 7), 5)
    assert_equal(pow(3, -1, -7), -2)
    assert_equal(pow(38, -1, 97), 23)
    assert_equal(pow(0, -1, 1), 0)
    assert_equal(pow(2, -1, 1), 0)
    assert_equal(pow(4, -3, -1), 0)

    # Intermediate products would overflow the input type.
    assert_equal(pow(123456789123, 987654321, 1000000007), 811545863)
    assert_equal(pow(Int64(2), Int64(100), Int64.MAX), 137438953472)
    assert_equal(pow(UInt64.MAX - 1, UInt64(3), UInt64.MAX), UInt64.MAX - 1)
    assert_equal(pow(Int8(100), Int8(3), Int8(127)), 2)
    assert_equal(pow(Int8.MIN, Int8(5), Int8(127)), 126)
    assert_equal(pow(Int64.MIN, Int64(2), Int64(97)), 33)
    assert_equal(pow(Int64.MIN, Int64(3), Int64.MIN), 0)

    # Exponents whose top bit is the type's highest.
    assert_equal(pow(UInt64(3), UInt64.MAX, UInt64(1000000007)), 35072593)
    assert_equal(pow(Int64(3), Int64.MIN, Int64(1000000007)), 201490753)
    assert_equal(
        pow(Int64(5), Int64.MAX, Int64.MIN), Int64(-3689348814741910323)
    )

    # 64-bit inputs with a modulus on either side of 2^32.
    comptime b64 = 123456789123456789
    comptime e64 = 987654321987654321
    assert_equal(pow(Int64(b64), Int64(e64), Int64(4294967295)), 1367320344)
    assert_equal(pow(Int64(b64), Int64(e64), Int64(4294967296)), 2987520085)
    assert_equal(pow(Int64(-b64), Int64(e64), Int64(-4294967291)), -1734543060)
    assert_equal(
        pow(UInt64.MAX - 58, UInt64.MAX, UInt64(4294967291)), 292809642
    )

    # Moduli above 2^32.
    assert_equal(
        pow(UInt64.MAX - 4, UInt64(1 << 40), UInt64(1 << 63)),
        8225398108880633857,
    )
    assert_equal(
        pow(Int64(12345678901234567), Int64(1), Int64(4294967297)),
        1564438323,
    )
    assert_equal(
        pow(Int64.MAX - 24, Int64((1 << 62) + 12345), Int64((1 << 61) - 1)),
        937114457775539982,
    )
    assert_equal(
        pow(Int64(3), Int64(-1), Int64((1 << 61) - 1)), 1537228672809129301
    )
    assert_equal(
        pow(Int64(-(1 << 62) + 7), Int64((1 << 62) + 3), Int64(-(1 << 62) - 1)),
        -4611686018425290753,
    )
    assert_equal(
        pow(UInt64.MAX, UInt64.MAX - 1, UInt64.MAX - (1 << 31)),
        6295607430025845564,
    )
    assert_equal(pow(Int64(10**18), Int64(10**18), Int64(1 << 33)), 0)

    # Exponents on either side of 2^34 with moduli up to 2^32.
    comptime e62 = (1 << 62) + 5
    assert_equal(pow(Int64(123456789), Int64(e62), Int64(2)), 1)
    assert_equal(pow(Int64(123456789), Int64(e62), Int64(3)), 0)
    assert_equal(pow(Int64.MAX, Int64.MAX, Int64(4294967295)), 4294967293)
    assert_equal(
        pow(Int64(987654321), Int64(1 << 34), Int64(1000000007)), 292649332
    )
    assert_equal(
        pow(Int64(987654321), Int64((1 << 34) - 1), Int64(1000000007)),
        156163515,
    )
    assert_equal(pow(Int64(-5), Int64((1 << 40) + 1), Int64(-65536)), -5)
    assert_equal(pow(Int32(7), Int32.MAX, Int32(65537)), 18725)

    comptime I = SIMD[.int32, 4]
    assert_equal(
        pow(I(3, 2, -2, 3), I(4, 10, 3, -1), I(5, 1000, -5, 7)), I(1, 24, -3, 5)
    )

    comptime folded = pow(3, 4, 5)
    assert_equal(folded, 1)


def test_isnan() raises:
    # Check that we can run llvm intrinsics returning bool at comptime.
    comptime x1 = isnan(SIMD[.float32, 4](SIMD[.float64, 4](1.0)))
    assert_equal(x1, SIMD[.bool, 4](fill=False))

    comptime x2 = isnan(SIMD[.float32, 4](FloatLiteral.nan))
    assert_equal(x2, SIMD[.bool, 4](fill=True))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
