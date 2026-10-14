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

from std.math import inf, isinf, isnan
from std.memory import bitcast

from std.testing import assert_equal, assert_raises, assert_true
from std.testing import TestSuite


def test_basic_parsing() raises:
    """Test basic parsing functionality."""
    assert_equal(atof("123"), 123.0)
    assert_equal(atof("123.456"), 123.456)
    assert_equal(atof("-123.456"), -123.456)
    assert_equal(atof("+123.456"), 123.456)


def test_scientific_notation() raises:
    """Test scientific notation parsing, which contained the primary bug."""
    assert_equal(atof("1.23e2"), 123.0)
    assert_equal(atof("1.23e+2"), 123.0)
    assert_equal(atof("1.23e-2"), 0.0123)
    assert_equal(atof("1.23E2"), 123.0)
    assert_equal(atof("1.23E+2"), 123.0)
    assert_equal(atof("1.23E-2"), 0.0123)


def test_nan_and_inf() raises:
    """Test NaN and infinity parsing."""
    assert_true(isnan(atof("nan")))
    assert_true(isnan(atof("NaN")))
    assert_true(isinf(atof("inf")))
    assert_true(isinf(atof("infinity")))
    assert_true(isinf(atof("-inf")))
    assert_true(atof("-inf") < 0)
    assert_true(isinf(atof("-infinity")))


def test_leading_decimal() raises:
    """Test parsing with leading decimal point."""
    assert_equal(atof(".123"), 0.123)
    assert_equal(atof("-.123"), -0.123)
    assert_equal(atof("+.123"), 0.123)


def test_trailing_f() raises:
    """Test parsing with trailing 'f'."""
    assert_equal(atof("123.456f"), 123.456)
    assert_equal(atof("123.456F"), 123.456)


def test_large_exponents() raises:
    """Test handling of large exponents."""
    assert_equal(atof("1e309"), inf[.float64]())
    assert_equal(atof("1e-309"), 1e-309)


def test_malformed_numbers_rejected() raises:
    """Test that malformed numbers raise instead of being partially parsed."""
    # The old right-to-left parser silently skipped the offending
    # characters, reading "1-2" as 12.0, "1e--5" as 1e-5, and so on.
    var malformed: List[String] = [
        "1-2",
        "1..2",
        "1e2e3",
        "1e5.0",
        ".e5",
        "1e--5",
        "1.2.3",
        "1e2.3",
        "1+2",
        "..5",
    ]
    for text in malformed:
        with assert_raises(
            contains="Invalid character(s) in the number: '" + text + "'"
        ):
            _ = atof(text)


def test_error_cases() raises:
    """Test error cases."""
    with assert_raises(
        contains=(
            "String is not convertible to float: 'abc'. The first character of"
            " 'abc' should be a digit or dot to convert it to a float."
        )
    ):
        _ = atof("abc")

    with assert_raises(contains="String is not convertible to float"):
        _ = atof("")

    with assert_raises(contains="String is not convertible to float"):
        _ = atof(".")

    # More than 19 significant digits are resolved exactly.
    assert_equal(atof("47421763.548648646474532187448684"), 47421763.54864865)


def test_slow_path() raises:
    """Test exact rounding of long mantissas via the big-integer slow path.

    Numbers with more than 19 significant digits only fall back to the
    exact `_BigUInt` comparison when the 19-digit truncation `w` and its
    neighbor `w + 1` round to different doubles, so every case below is
    built to straddle a rounding boundary.
    """
    var as_bits = lambda (f: Float64) -> UInt64: bitcast[.uint64](f)

    # Exact decimal expansions of midpoints between adjacent doubles: each
    # tie resolves to the neighbor with the even mantissa.
    assert_equal(
        atof("1.00000000000000011102230246251565404236316680908203125"), 1.0
    )
    assert_equal(
        atof("1.00000000000000033306690738754696212708950042724609375"),
        1.0000000000000004,
    )
    assert_equal(
        atof("0.999999999999999944488848768742172978818416595458984375"), 1.0
    )
    # One digit past each midpoint tips the tie.
    assert_equal(
        atof("1.000000000000000111022302462515654042363166809082031251"),
        1.0000000000000002,
    )
    assert_equal(
        atof("0.999999999999999944488848768742172978818416595458984374"),
        0.9999999999999999,
    )

    # Integer midpoints 2**54 + 2 and 2**55 + 4: the same ties at
    # magnitudes where the midpoint shifts by a positive power of two.
    assert_equal(
        atof("18014398509481986.00000000000000000001"), 18014398509481988.0
    )
    assert_equal(
        atof("36028797018963972.00000000000000000001"), 36028797018963976.0
    )

    # An explicit exponent keeps the decimal scale positive, so the value
    # itself carries the power of ten.
    assert_equal(atof("12089258196146293081e5"), 1.2089258196146292e24)

    # Leading zeros after the dot must not consume the digit budget. This
    # is the midpoint below 2**-40, scaled down by a power of two.
    assert_equal(
        atof(
            "0.0000000000009094947017729281874279411283552444536493718219016813009147881530225276947021484375"
        ),
        9.094947017729282e-13,
    )

    # Straddling the normal/subnormal boundary: only the exact comparison
    # can tell which side the truncated mantissa belongs to.
    var below = atof("2.225073858507201136057409e-308")
    assert_equal(as_bits(below), UInt64(0x000FFFFFFFFFFFFF))
    var above = atof("2.225073858507201136057410e-308")
    assert_equal(as_bits(above), UInt64(0x0010000000000000))

    # 769 significant digits: the tail beyond the 768-digit capacity of
    # the big integer only contributes a sticky bit, which still tips the
    # tie below.
    var sticky = (
        String("1.00000000000000011102230246251565404236316680908203125")
        + "0" * 713
        + "1"
    )
    assert_equal(atof(sticky), 1.0000000000000002)


def test_digit_runs_at_buffer_end() raises:
    """Test digit runs that stop exactly at the end of the input.

    `scan_digits` consumes runs eight bytes at a time with a byte loop for
    the remainder, so runs on every boundary around the chunk size and
    the 19-digit mantissa budget need to parse exactly without padding.
    """
    assert_equal(atof("12345678"), 12345678.0)
    assert_equal(atof("1234567890123456"), 1234567890123456.0)
    assert_equal(atof("12345678901234567"), 1.2345678901234568e16)
    assert_equal(atof("123456789012345678"), 1.2345678901234568e17)
    assert_equal(atof("1234567890123456789"), 1.2345678901234568e18)
    assert_equal(atof("12345678901234567890"), 1.2345678901234567e19)
    # Longer runs load many chunks and end on a byte-count tail.
    assert_equal(
        atof("1234567890123456789012345678901234567890"), 1.2345678901234568e39
    )
    assert_equal(
        atof(
            "1234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890"
        ),
        1.2345678901234567e99,
    )
    # A run of zeros still has to scan every byte for the terminator.
    assert_equal(atof("1000000000000000000"), 1e18)
    # A trailing dot is a valid terminator for a long run.
    assert_equal(
        atof("1234567890123456789012345678901234567890."),
        1.2345678901234568e39,
    )
    # Fraction runs ending at the end of the input.
    assert_equal(atof("0.12345678"), 0.12345678)
    assert_equal(atof("0.123456789012345678"), 0.12345678901234568)
    assert_equal(atof("0.1234567890123456789"), 0.12345678901234568)
    assert_equal(
        atof("0.1234567890123456789012345678901234567890"),
        0.12345678901234568,
    )
    # Terminators inside a chunk: fewer than eight digits before the dot
    # or the exponent.
    assert_equal(atof("1234567e89"), 1.234567e95)
    assert_equal(atof("0.1234567e2"), 12.34567)
    assert_equal(atof("12345678.1234567e2"), 1234567812.34567)
    # Exponent digits must also survive the chunked scan.
    assert_equal(atof("12345678e5"), 1234567800000.0)
    assert_equal(atof("12345678e89"), 1.2345678e96)
    assert_equal(atof("1234567.e5"), 123456700000.0)
    assert_equal(atof("1.e5"), 100000.0)


def test_leading_zeros() raises:
    """Test long runs of leading zeros.

    Leading zeros must not consume the 19-digit mantissa budget, whether
    they appear before the dot, after it, or split across eight-byte
    chunks.
    """
    assert_equal(atof("0000000000000000000000000000001"), 1.0)
    assert_equal(
        atof("00000000000000000000000000000000000000000000000001"), 1.0
    )
    assert_equal(
        atof("00000000000000000000000000000000000000000000000001e5"), 100000.0
    )
    # Zeros after the dot shift the exponent once a nonzero digit appears.
    assert_equal(atof("0.00000001"), 1e-08)
    assert_equal(atof("0.0000000123456789"), 1.23456789e-08)
    assert_equal(atof("0.00000000000000000000000000000000000001"), 1e-38)
    assert_equal(atof(String("0.") + "0" * 100 + "1"), 1e-101)
    # All-zero numbers stay zero.
    assert_equal(atof("0.00000000"), 0.0)
    assert_equal(atof("000.000"), 0.0)
    assert_equal(atof("0.0e10"), 0.0)
    # Zeros after the mantissa becomes nonzero do consume the budget.
    assert_equal(atof("10000000000000000000000000000000000000000001"), 1e43)
    assert_equal(
        atof("000000000000000000000000123456789012345678901234567890"),
        1.2345678901234568e29,
    )


comptime T = Tuple[Float64, String]
comptime numbers_to_test = [
    T(5e-324, "5e-324"),  # smallest value possible with float64
    T(1e-309, "1e-309"),  # subnormal float64
    T(84.5e-309, "84.5e-309"),  # subnormal float64
    T(1e-45, "1e-45"),  # smallest float32 value,
    # largest value possible
    T(1.7976931348623157e308, "1.7976931348623157e+308"),
    T(3.4028235e38, "3.4028235e38"),  # largest value possible, float32
    T(15038927332917.156, "15038927332917.156"),  # triggers step 19
    T(9000000000000000.5, "9000000000000000.5"),  # tie to even
    T(456.7891011e70, "456.7891011e70"),  # Lemire algorithm
    T(0.0, "5e-600"),  # approximate to 0
    T(FloatLiteral.infinity, "5e1000"),  # approximate to infinity
    T(5484.2155e-38, "5484.2155e-38"),  # Lemire algorithm
    T(5e-35, "5e-35"),  # Lemire algorithm
    T(5e30, "5e30"),  # Lemire algorithm
    T(47421763.54884, "47421763.54884"),  # Clinger fast path
    T(474217635486486e10, "474217635486486e10"),  # Clinger fast path
    T(474217635486486e-10, "474217635486486e-10"),  # Clinger fast path
    T(474217635486486e-20, "474217635486486e-20"),  # Clinger fast path
    T(4e-22, "4e-22"),  # Clinger fast path
    T(4.5e15, "4.5e15"),  # Clinger fast path
    T(0.1, "0.1"),  # Clinger fast path
    T(0.2, "0.2"),  # Clinger fast path
    T(0.3, "0.3"),  # Clinger fast path
    # largest uint64 * 10 ** 10
    T(18446744073709551615e10, "18446744073709551615e10"),
    T(3.5e18, "3.5e18"),
    # Examples for issue https://github.com/modularml/mojo/issues/3419
    T(3.5e19, "3.5e19"),
    T(3.5e20, "3.5e20"),
    T(3.5e21, "3.5e21"),
    T(3.5e-15, "3.5e-15"),
    T(3.5e-16, "3.5e-16"),
    T(3.5e-17, "3.5e-17"),
    T(3.5e-18, "3.5e-18"),
    T(3.5e-19, "3.5e-19"),
    T(47421763.54864864647, "47421763.54864864647"),
    # Normal/subnormal boundary
    T(4.4501363245856945e-308, "4.4501363245856945e-308"),
    T(2.2250738585072014e-308, "2.2250738585072014e-308"),  # smallest normal
    T(2.2250738585072009e-308, "2.2250738585072009e-308"),  # largest subnormal
    # More than 19 significant digits, resolved exactly by comparing the
    # truncated mantissa against its neighbors.
    T(47421763.54864865, "47421763.548648646474532187448684"),
]


def test_atof_generate_cases() raises:
    for number, number_as_str in materialize[numbers_to_test]():
        for suffix in ["", "f", "F"]:
            for exponent in ["e", "E"]:
                for multiplier in ["", "-"]:
                    var sign: Float64 = 1
                    if multiplier == "-":
                        sign = -1
                    var final_string = number_as_str.replace("e", exponent)
                    final_string = multiplier + final_string + suffix
                    var final_value = sign * number

                    assert_equal(atof(final_string), final_value)


def test_normal_subnormal_boundary() raises:
    var as_bits = lambda (f: Float64) -> UInt64: bitcast[.uint64](f)

    # Smallest normal: biased exponent 1, mantissa 0
    # Bit pattern: 0x0010000000000000
    var smallest_normal = atof("2.2250738585072014e-308")
    assert_equal(as_bits(smallest_normal), UInt64(0x0010000000000000))

    # Largest subnormal: biased exponent 0, mantissa all 1s
    # Bit pattern: 0x000FFFFFFFFFFFFF
    var largest_subnormal = atof("2.2250738585072009e-308")
    assert_equal(as_bits(largest_subnormal), UInt64(0x000FFFFFFFFFFFFF))

    # Reported bug: should be normal, not subnormal
    # Bit pattern: 0x0020000000000001
    var reported_bug = atof("4.4501363245856945e-308")
    assert_equal(as_bits(reported_bug), as_bits(4.4501363245856945e-308))
    # Verify it's a normal number (biased exponent > 0)
    assert_true(as_bits(reported_bug) >= UInt64(0x0010000000000000))


def test_large_mantissa_rounding() raises:
    """Mantissas above 2^53 take the Eisel-Lemire path and must round to
    nearest, ties to even."""
    # Both inputs round to ...680, where adjacent doubles are spaced by 16.
    assert_equal(atof("123456789012345678"), 1.2345678901234568e17)
    assert_equal(atof("123456789012345679"), 1.2345678901234568e17)
    # Exact ties to even.
    assert_equal(atof("4503599627370497.5"), 4503599627370498.0)
    assert_equal(atof("45035996273704975e-1"), 4503599627370498.0)
    assert_equal(atof("9007199254740993"), 9007199254740992.0)
    assert_equal(atof("18014398509481985"), 18014398509481984.0)
    assert_equal(atof("1234567890123456789"), 1.2345678901234568e18)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
