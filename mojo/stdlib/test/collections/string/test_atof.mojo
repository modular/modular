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
    assert_equal(atof("1e-309"), 0.0)


def test_error_cases():
    """Test error cases."""
    with assert_raises(contains="String is not convertible to float"):
        _ = atof("abc")

    with assert_raises(contains="String is not convertible to float"):
        _ = atof("")

    with assert_raises(contains="String is not convertible to float"):
        _ = atof(".")


def main():
    test_basic_parsing()
    test_scientific_notation()
    test_nan_and_inf()
    test_leading_decimal()
    test_trailing_f()
    test_large_exponents()
    test_error_cases()
