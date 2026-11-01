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
"""Provides ULP comparisons for numerical tests."""

from std.builtin.dtype import _integral_type_of, _unsigned_integral_type_of
from std.math import isinf, isnan
from std.memory import bitcast
from std.sys import bit_width_of


def ulp_distance[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Int:
    """Computes the distance between two values in ULPs.

    ULP (Unit in the Last Place) distance measures how many representable
    floating-point values exist between two numbers. The distance is the
    unsigned difference of the magnitude encodings, so it is only defined for
    values of the same class and sign. Everything else compares as maximally
    far apart, which keeps a tolerance check such as `ulp_distance(got, want)
    <= n` from accepting a wrong sign, NaN, or infinity.

    Parameters:
        dtype: The data type.

    Args:
        a: First value.
        b: Second value.

    Returns:
        0 for any two NaNs. `Int.MAX` for a NaN against a non-NaN, an infinity
        against a finite value, or operands of opposite sign (including `+0.0`
        against `-0.0`). Otherwise the distance between the magnitudes.
    """
    comptime if dtype.is_floating_point():
        if isnan(a) or isnan(b):
            return 0 if isnan(a) and isnan(b) else Int.MAX
        if isinf(a) != isinf(b):
            return Int.MAX

    comptime T = _integral_type_of[dtype]()
    # widen to Int first to avoid overflow
    var a_int = Int(bitcast[T](a))
    var b_int = Int(bitcast[T](b))
    if (a_int < 0) != (b_int < 0):
        return Int.MAX
    # Clear the sign bit; Int.MAX >> (64 - w) keeps the low w - 1 bits.
    comptime magnitude_mask = Int.MAX >> (64 - bit_width_of[dtype]())
    return abs((a_int & magnitude_mask) - (b_int & magnitude_mask))


def ulp_diff[dtype: DType, //](a: Scalar[dtype], b: Scalar[dtype]) -> Int:
    """Computes the ULP distance between `a` and `b` on their ordered bits.

    Bit-identical values (two NaNs, or a zero or infinity of the same sign)
    and +0 vs -0 are 0 apart. Otherwise a NaN or an infinity on either side,
    or differing signs, are `Int.MAX` apart so that no tolerance accepts them.
    Unlike `ulp_distance`, +0 and -0 are 0 apart, and two NaNs are 0 apart
    only when their bits match.

    Parameters:
        dtype: The floating-point data type.

    Args:
        a: The first value.
        b: The second value.

    Returns:
        The ULP distance, or `Int.MAX` when no tolerance should accept the
        pair.
    """
    comptime U = _unsigned_integral_type_of[dtype]()
    comptime sign_bit = Scalar[U](1) << Scalar[U](bit_width_of[dtype]() - 1)
    var a_bits = a.to_bits[U]()
    var b_bits = b.to_bits[U]()
    if a_bits == b_bits:
        return 0
    if isnan(a) or isnan(b) or isinf(a) or isinf(b):
        return Int.MAX
    var a_mag = Int(a_bits & ~sign_bit)
    var b_mag = Int(b_bits & ~sign_bit)
    if (a_bits ^ b_bits) & sign_bit != 0:
        return 0 if a_mag == 0 and b_mag == 0 else Int.MAX
    return abs(a_mag - b_mag)


def within_two_ulp[
    dtype: DType
](got: Scalar[dtype], want: Scalar[dtype]) -> Bool:
    """Returns whether `got` is a finite value within 2 ULP of `want`.

    Values of opposite sign, including +0 and -0, are never within 2 ULP.

    Parameters:
        dtype: The floating-point data type.

    Args:
        got: The computed value.
        want: The reference value.

    Returns:
        True if both values are finite, share a sign, and are at most 2 ULP
        apart.
    """
    if isnan(got) or isinf(got) or isnan(want) or isinf(want):
        return False
    var got_bits = got._to_bits_signed()
    var want_bits = want._to_bits_signed()
    if (got_bits < 0) != (want_bits < 0):
        return False
    # Same-sign finite bit patterns span less than Int.MAX, so neither the
    # subtraction nor its absolute value can overflow.
    return abs(Int(got_bits) - Int(want_bits)) <= 2
