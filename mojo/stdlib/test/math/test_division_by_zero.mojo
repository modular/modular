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
# test_division_by_zero.mojo
# Test that division by zero produces correct IEEE 754 results.
# This test file verifies the fix for issue #5733.

from std.testing import assert_equal, assert_true
from std.math import isnan, isinf


def test_positive_division_by_zero():
    """Test that positive number / 0.0 produces +inf."""
    var result = 1.0 / 0.0
    assert_true(isinf(result), "1.0 / 0.0 should be inf, not nan")
    assert_true(result > 0, "1.0 / 0.0 should be positive infinity")

    # Test with other positive values
    var result2 = 5.0 / 0.0
    assert_true(isinf(result2), "5.0 / 0.0 should be inf")

    var result3 = 0.5 / 0.0
    assert_true(isinf(result3), "0.5 / 0.0 should be inf")


def test_negative_division_by_zero():
    """Test that negative number / 0.0 produces -inf."""
    var result = -1.0 / 0.0
    assert_true(isinf(result), "-1.0 / 0.0 should be -inf, not nan")
    assert_true(result < 0, "-1.0 / 0.0 should be negative infinity")

    # Test with other negative values
    var result2 = -5.0 / 0.0
    assert_true(isinf(result2), "-5.0 / 0.0 should be -inf")


def test_zero_division_by_zero():
    """Test that 0.0 / 0.0 produces nan."""
    var result = 0.0 / 0.0
    assert_true(isnan(result), "0.0 / 0.0 should be nan")


def test_division_by_negative_zero():
    """Test division by negative zero."""
    # IEEE 754 specifies that -0.0 exists
    var neg_zero = -0.0

    # Positive / -0.0 should be -inf
    var result1 = 1.0 / neg_zero
    assert_true(isinf(result1), "1.0 / -0.0 should be -inf")
    assert_true(result1 < 0, "1.0 / -0.0 should be negative infinity")

    # Negative / -0.0 should be +inf
    var result2 = -1.0 / neg_zero
    assert_true(isinf(result2), "-1.0 / -0.0 should be +inf")
    assert_true(result2 > 0, "-1.0 / -0.0 should be positive infinity")


def test_inf_operations():
    """Test that inf values work correctly in operations."""
    var inf_val = 1.0 / 0.0

    # inf + 1 should be inf
    var result1 = inf_val + 1.0
    assert_true(isinf(result1), "inf + 1 should be inf")

    # inf * 2 should be inf
    var result2 = inf_val * 2.0
    assert_true(isinf(result2), "inf * 2 should be inf")

    # 1 / inf should be 0
    var result3 = 1.0 / inf_val
    assert_equal(result3, 0.0, "1 / inf should be 0")


def test_comparison_with_inf():
    """Test comparisons with infinity."""
    var inf_val = 1.0 / 0.0

    assert_true(inf_val > 1000000.0, "inf should be greater than any finite number")
    assert_true(inf_val > 0.0, "inf should be positive")

    var neg_inf = -1.0 / 0.0
    assert_true(neg_inf < -1000000.0, "-inf should be less than any finite number")
    assert_true(neg_inf < 0.0, "-inf should be negative")


def main():
    """Run all division by zero tests."""
    test_positive_division_by_zero()
    test_negative_division_by_zero()
    test_zero_division_by_zero()
    test_division_by_negative_zero()
    test_inf_operations()
    test_comparison_with_inf()
    print("All division by zero tests passed!")
