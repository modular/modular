# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
# RUN: %mojo %s

from math import inf, isinf, isnan
from testing import assert_equal, assert_true, assert_raises


def test_basic_parsing():
    """Test basic parsing functionality."""
    assert_equal(atof("123"), 123.0)
    assert_equal(atof("123.456"), 123.456)
    assert_equal(atof("-123.456"), -123.456)
    assert_equal(atof("+123.456"), 123.456)


def test_scientific_notation():
    """Test scientific notation parsing, which contained the primary bug."""
    assert_equal(atof("1.23e2"), 123.0)
    assert_equal(atof("1.23e+2"), 123.0)
    assert_equal(atof("1.23e-2"), 0.0123)
    assert_equal(atof("1.23E2"), 123.0)
    assert_equal(atof("1.23E+2"), 123.0)
    assert_equal(atof("1.23E-2"), 0.0123)


def test_nan_and_inf():
    """Test NaN and infinity parsing."""
    assert_true(isnan(atof("nan")))
    assert_true(isnan(atof("NaN")))
    assert_true(isinf(atof("inf")))
    assert_true(isinf(atof("infinity")))
    assert_true(isinf(atof("-inf")))
    assert_true(atof("-inf") < 0)
    assert_true(isinf(atof("-infinity")))


def test_leading_decimal():
    """Test parsing with leading decimal point."""
    assert_equal(atof(".123"), 0.123)
    assert_equal(atof("-.123"), -0.123)
    assert_equal(atof("+.123"), 0.123)


def test_trailing_f():
    """Test parsing with trailing 'f'."""
    assert_equal(atof("123.456f"), 123.456)
    assert_equal(atof("123.456F"), 123.456)


def test_large_exponents():
    """Test handling of large exponents."""
    assert_equal(atof("1e309"), inf[DType.float64]())
    assert_equal(atof("1e-309"), 1e-309)


def test_error_cases():
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

    # TODO:
    # This should actualy work and approximate to the closest float64
    # but we don't support it yet. See the section
    # 11, "Processing long numbers quickly" in the paper
    # Number Parsing at a Gigabyte per Second by Daniel Lemire
    # https://arxiv.org/abs/2101.11408 to learn how to do it.
    with assert_raises(
        contains="The number is too long, it's not supported yet."
    ):
        _ = atof("47421763.548648646474532187448684")


from testing import assert_equal

alias numbers_to_test_as_str = List[String](
    "5e-324",  # smallest value possible with float64
    "1e-309",  # subnormal float64
    "84.5e-309",  # subnormal float64
    "1e-45",  # smallest float32 value,
    "1.7976931348623157e+308",  # largest value possible
    "3.4028235e38",  # largest value possible, float32
    "15038927332917.156",  # triggers step 19
    "9000000000000000.5",  # tie to even
    "456.7891011e70",  # Lemire algorithm
    "5e-600",  # approximate to 0
    "5e1000",  # approximate to infinity
    "5484.2155e-38",  # Lemire algorithm
    "5e-35",  # Lemire algorithm
    "5e30",  # Lemire algorithm
    "47421763.54884",  # Clinger fast path
    "474217635486486e10",  # Clinger fast path
    "474217635486486e-10",  # Clinger fast path
    "474217635486486e-20",  # Clinger fast path
    "4e-22",  # Clinger fast path
    "4.5e15",  # Clinger fast path
    "0.1",  # Clinger fast path
    "0.2",  # Clinger fast path
    "0.3",  # Clinger fast path
    "18446744073709551615e10",  # largest uint64 * 10 ** 10
    # Examples for issue https://github.com/modularml/mojo/issues/3419
    "3.5e18",
    "3.5e19",
    "3.5e20",
    "3.5e21",
    "3.5e-15",
    "3.5e-16",
    "3.5e-17",
    "3.5e-18",
    "3.5e-19",
    "47421763.54864864647",
    # TODO: Make atof work when many digits are present, e.g.
    # "47421763.548648646474532187448684",
)
alias numbers_to_test = List[Float64](
    5e-324,
    1e-309,
    84.5e-309,
    1e-45,
    1.7976931348623157e308,
    3.4028235e38,
    15038927332917.156,
    9000000000000000.5,
    456.7891011e70,
    0.0,
    FloatLiteral.infinity,
    5484.2155e-38,
    5e-35,
    5e30,
    47421763.54884,
    474217635486486e10,
    474217635486486e-10,
    474217635486486e-20,
    4e-22,
    4.5e15,
    0.1,
    0.2,
    0.3,
    18446744073709551615e10,
    3.5e18,
    3.5e19,
    3.5e20,
    3.5e21,
    3.5e-15,
    3.5e-16,
    3.5e-17,
    3.5e-18,
    3.5e-19,
    47421763.54864864647,
)


def test_atof_generate_cases():
    for i in range(len(numbers_to_test)):
        for suffix in List[String]("", "f", "F"):
            for exponent in List[String]("e", "E"):
                for multiplier in List[String]("", "-"):
                    var sign: Float64 = 1
                    if multiplier[] == "-":
                        sign = -1
                    final_string = numbers_to_test_as_str[i].replace(
                        "e", exponent[]
                    )
                    final_string = multiplier[] + final_string + suffix[]
                    final_value = sign * numbers_to_test[i]

                    assert_equal(atof(final_string), final_value)


def main():
    test_basic_parsing()
    test_scientific_notation()
    test_nan_and_inf()
    test_leading_decimal()
    test_trailing_f()
    test_large_exponents()
    test_error_cases()
    test_atof_generate_cases()
