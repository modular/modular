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

"""
Implementation of the following papers:
- Number Parsing at a Gigabyte per Second by Daniel Lemire
  - https://arxiv.org/abs/2101.11408
- Fast Number Parsing Without Fallback by Noble Mushtak & Daniel Lemire
  - https://arxiv.org/abs/2212.06644

The reference implementation used was the one in C# and can be found here:
- https://github.com/CarlVerret/csFastFloat
"""

from std.collections import Array

import std.bit
import std.memory
from std.sys import Endian


from std.builtin.globals import global_constant

from .constants import (
    MANTISSA_EXPLICIT_BITS,
    POWERS_OF_10,
    SMALLEST_POWER_OF_5,
    get_power_of_5,
)


@fieldwise_init
struct UInt128Decomposed(ImplicitlyCopyable, RegisterPassable):
    var high: UInt64
    var low: UInt64

    def __init__(out self, value: UInt128):
        self.high = UInt64(value >> 64)
        self.low = UInt64(value & 0xFFFFFFFFFFFFFFFF)

    def most_significant_bit(self) -> UInt64:
        return self.high >> 63


def strip_unused_characters(x: StringSlice[_]) -> type_of(x):
    return x.strip().removeprefix("+").removesuffix("f").removesuffix("F")


def get_sign(x: StringSlice[_]) -> Tuple[Float64, type_of(x)]:
    if x.startswith("-"):
        return (-1.0, x[byte=1:])
    return (1.0, x)


# Powers of 10 and integers below 2**53 are exactly representable as Float64.
# Thus any operation done on them must be exact.
@inline(.always)
def can_use_clinger_fast_path(w: UInt64, q: Int64) -> Bool:
    return w <= UInt64(2**53) and (Int64(-22) <= q <= Int64(22))


@inline(.always)
def clinger_fast_path(w: UInt64, q: Int64) -> Float64:
    if q >= 0:
        return Float64(w) * global_constant[POWERS_OF_10]()[q]
    else:
        return Float64(w) / global_constant[POWERS_OF_10]()[-q]


@inline(.always)
def full_multiplication(x: UInt64, y: UInt64) -> UInt128Decomposed:
    # Note that there are assembly instructions to
    # do all that on some architectures.
    # That should speed things up.
    var result = UInt128(x) * UInt128(y)
    return UInt128Decomposed(result)


@inline(.always)
def get_128_bit_truncated_product(w: UInt64, q: Int64) -> UInt128Decomposed:
    comptime bit_precision = MANTISSA_EXPLICIT_BITS + 3
    var index = 2 * (q - SMALLEST_POWER_OF_5)
    var first_product = full_multiplication(w, get_power_of_5(Int(index)))

    # The product keeps `bit_precision` bits of `high`; when every bit below
    # them is set, the truncated tail could carry, so refine with the next
    # 64 bits of the power of five.
    var precision_mask = ~UInt64(0) >> UInt64(bit_precision)
    if (first_product.high & precision_mask) == precision_mask:
        var second_product = full_multiplication(
            w, get_power_of_5(Int(index + 1))
        )
        first_product.low = first_product.low + second_product.high
        if second_product.high > first_product.low:
            first_product.high = first_product.high + 1

    return first_product


def create_subnormal_float64(m: UInt64) -> Float64:
    return create_float64(m, -1023)


@inline(.always)
def create_float64(m: UInt64, p: Int64) -> Float64:
    var m_mask = UInt64(2**MANTISSA_EXPLICIT_BITS - 1)
    var p_shifted = UInt64(p + 1023) << MANTISSA_EXPLICIT_BITS
    var representation_as_int = (m & m_mask) | p_shifted
    return std.memory.bitcast[.float64](representation_as_int)


@inline(.always)
def lemire_algorithm(var w: UInt64, var q: Int64) -> Float64:
    # This algorithm has 22 steps described
    # in https://arxiv.org/pdf/2101.11408 (algorithm 1)
    # Step 1
    if w == 0 or q < -342:
        return 0.0

    # Step 2
    if q > 308:
        return FloatLiteral.infinity

    # Step 3
    var l = std.bit.count_leading_zeros(w)

    # Step 4
    w <<= l

    # Step 5
    var product = get_128_bit_truncated_product(w, q)

    # Step 6
    # This step is skipped because it has been proven not necessary.
    # The proof can be found in the following paper by
    # Noble Mushtak & Daniel Lemire:
    # Fast Number Parsing Without Fallback
    # https://arxiv.org/abs/2212.06644

    # Step 8
    # Comes before step 7 because we need the upper_bit
    var upper_bit = product.most_significant_bit()

    # Step 7
    var m = product.high >> (upper_bit + 9)

    # Step 9
    var p: Int64 = (
        (((Int64(152170) + Int64(65536)) * q) >> Int64(16))
        + Int64(63)
        - Int64(l)
        + Int64(upper_bit)
    )

    # Step 10
    if p <= (-1022 - 64):
        return 0.0

    # Step 11-15
    # Subnormal case
    if p < -1022:
        var s = -1022 - p
        m = m // (UInt64(2) ** UInt64(s))
        if m % 2 == 1:
            m += 1
        m >>= 1
        if m >= UInt64(2**MANTISSA_EXPLICIT_BITS):
            return create_float64(m, -1022)
        return create_subnormal_float64(m)

    # Step 16-18
    # Round ties to even: when the product is exact (the bits shifted out of
    # `high` to form `m` were all zero) and `m` sits exactly on a halfway
    # point with an even neighbour below, clear the rounding bit so the
    # round-up in step 19 does not fire.
    if product.low <= 1 and (m & 3 == 1) and (Int64(-4) <= q <= Int64(23)):
        if (m << (upper_bit + 9)) == product.high:
            m &= ~UInt64(1)

    # step 19
    if m % 2 == 1:
        m += 1
    m //= 2

    # Step 20
    if m == UInt64(2**53):
        m //= 2
        p = p + 1

    # step 21
    if p > 1023:
        return FloatLiteral.infinity

    # Step 22
    return create_float64(m, p)


# ===----------------------------------------------------------------------=== #
# Decimal text to Float64
# ===----------------------------------------------------------------------=== #
#
# Used by `atof` and by callers with their own number grammar. A caller scans
# the digits itself and feeds them through `DecimalParts`, which keeps the
# first 19 significant digits as a `UInt64` and tracks what it dropped;
# `decimal_to_float64` then converts with Clinger's fast path, Eisel-Lemire,
# or an exact big-integer comparison, in that order of cost.
# ===----------------------------------------------------------------------=== #

comptime _MAX_SIGNIFICANT_DIGITS = 19
comptime _MAX_EXACT_DIGITS = 768


# Eight-digit SWAR helpers (Lemire, "Number Parsing at a Gigabyte per
# Second", section 5): a run of digits is consumed eight bytes at a time as
# one little-endian word.
@inline(.always)
def _load_eight(bytes: ImmSpan[Byte, _], pos: Int) -> UInt64:
    comptime assert Endian.native() == .little
    return std.memory.bitcast[.uint64, 1](
        bytes.unsafe_ptr().unsafe_load[width=8](pos)
    )


@inline(.always)
def _is_digit(byte: Byte) -> Bool:
    return byte >= Byte(ord("0")) and byte <= Byte(ord("9"))


# Bytes of `word` that are zero get their high bit set (the classic SWAR
# zero-byte test); everything else is cleared.
@inline(.always)
def _zero_byte_mask(word: UInt64) -> UInt64:
    var partial = (word & 0x7F7F7F7F7F7F7F7F) + 0x7F7F7F7F7F7F7F7F
    return ~(partial | word | 0x7F7F7F7F7F7F7F7F)


# Number of leading bytes of the word (lowest addresses first) that are ASCII
# digits, 0 to 8. A byte is a digit when `byte - '0'` is at most 9; the mask
# sets bit 7 of every byte that is not. The subtraction can borrow into the
# byte after a non-digit, but that byte is above the first non-digit and
# never affects the count.
@inline(.always)
def _digit_prefix_length(chunk: UInt64) -> Int:
    var t = chunk - 0x3030303030303030
    var nondigit = (
        ((t & 0x7F7F7F7F7F7F7F7F) + 0x7676767676767676) | t
    ) & 0x8080808080808080
    if nondigit == 0:
        return 8
    return Int(std.bit.count_trailing_zeros(nondigit)) >> 3


# Number of leading bytes of the word that a signed decimal is made of: the
# minus sign, digits and the point, that is bytes in [0x2D, 0x39] (`/` is
# admitted by the range and rejected by the digit tests that follow). Same
# construction as `_digit_prefix_length`.
@inline(.always)
def _number_prefix_length(chunk: UInt64) -> Int:
    var t = chunk - 0x2D2D2D2D2D2D2D2D
    var other = (
        ((t & 0x7F7F7F7F7F7F7F7F) + 0x7373737373737373) | t
    ) & 0x8080808080808080
    if other == 0:
        return 8
    return Int(std.bit.count_trailing_zeros(other)) >> 3


# Value of the first `count` digit bytes of the word (1 to 8). The digits are
# shifted to the most significant positions and the freed bytes become '0',
# so the eight-digit fold yields the `count`-digit number.
@inline(.always)
def _digit_prefix_value(chunk: UInt64, count: Int) -> UInt64:
    if count == 8:
        return _eight_digits_value(chunk)
    var shift = UInt64(64 - 8 * count)
    return _eight_digits_value(
        (chunk << shift) | (0x3030303030303030 >> (UInt64(64) - shift))
    )


# Like `_load_eight`, but near the end of the buffer the bytes past the end
# read as zero, which no digit test accepts, so prefix logic stays correct.
# A buffer of at least eight bytes uses one overlapping load ending at the
# buffer end; only shorter buffers assemble the word byte by byte.
@inline(.always)
def _load_up_to_eight(bytes: ImmSpan[Byte, _], pos: Int) -> UInt64:
    var end = len(bytes)
    if pos >= end:
        return 0
    if pos + 8 <= end:
        return _load_eight(bytes, pos)
    if end >= 8:
        return _load_eight(bytes, end - 8) >> UInt64(8 * (pos + 8 - end))
    var word: UInt64 = 0
    var ptr = bytes.unsafe_ptr()
    for j in range(end - pos):
        word |= UInt64(ptr[unsafe_offset=pos + j]) << UInt64(8 * j)
    return word


@inline(.always)
def _eight_digits_value(var chunk: UInt64) -> UInt64:
    chunk -= 0x3030303030303030
    chunk = chunk * 10 + (chunk >> 8)
    var mask = UInt64(0x000000FF000000FF)
    return (
        ((chunk & mask) * 0x000F424000000064)
        + (((chunk >> 16) & mask) * 0x0000271000000001)
    ) >> 32


struct DecimalParts(TrivialRegisterPassable):
    """The digits of a decimal number, reduced to a 19-digit mantissa.

    `mantissa * 10**exponent` approximates the digits seen so far; the
    explicit exponent of the text is added by the caller. `truncated` records
    that more than 19 significant digits were present and `truncated_nonzero`
    that at least one dropped digit was not zero, in which case the true
    value lies strictly between `mantissa` and `mantissa + 1` at this
    exponent.
    """

    var mantissa: UInt64
    var exponent: Int
    var digit_count: Int
    var truncated: Bool
    var truncated_nonzero: Bool

    def __init__(out self):
        self.mantissa = 0
        self.exponent = 0
        self.digit_count = 0
        self.truncated = False
        self.truncated_nonzero = False

    @inline(.always)
    def push_digit(mut self, digit: Byte, in_fraction: Bool):
        """Appends one decimal digit (0 to 9).

        Args:
            digit: The digit value.
            in_fraction: Whether the digit follows the decimal point.
        """
        if self.digit_count < _MAX_SIGNIFICANT_DIGITS:
            self.mantissa = self.mantissa * 10 + UInt64(digit)
            self.digit_count += Int(self.mantissa != 0)
            self.exponent -= Int(in_fraction)
        else:
            self.truncated = True
            self.truncated_nonzero = self.truncated_nonzero or digit != 0
            if not in_fraction:
                self.exponent += 1

    @inline(.always)
    def push_digits(
        mut self, value: UInt64, count: Int, in_fraction: Bool
    ) -> Bool:
        """Appends up to eight decimal digits at once when they fit the budget.

        Args:
            value: The value of the digits (below `10**count`).
            count: How many digits `value` stands for, 1 to 8.
            in_fraction: Whether the digits follow the decimal point.

        Returns:
            `False`, leaving `self` unchanged, when accepting the digits would
            exceed the 19 significant digits kept; the caller then continues
            digit by digit.
        """
        var significant = count
        if self.mantissa == 0:
            significant = 0
            var rest = value
            while rest > 0:
                rest //= 10
                significant += 1
        if self.digit_count + significant > _MAX_SIGNIFICANT_DIGITS:
            return False
        comptime pow10: Array[UInt64, 9] = [
            1,
            10,
            100,
            1000,
            10000,
            100000,
            1000000,
            10000000,
            100000000,
        ]
        self.mantissa = self.mantissa * materialize[pow10]()[count] + value
        self.digit_count += significant
        if in_fraction:
            self.exponent -= count
        return True

    @inline(.always)
    def scan_digits(
        mut self, bytes: ImmSpan[Byte, _], mut pos: Int, in_fraction: Bool
    ) -> Int:
        """Consumes the run of ASCII digits starting at `pos`.

        Args:
            bytes: The text.
            pos: The position to start at; advanced past the digits.
            in_fraction: Whether the digits follow the decimal point.

        Returns:
            The number of digits consumed.
        """
        comptime ord_0 = Byte(ord("0"))
        comptime ord_9 = Byte(ord("9"))
        var start = pos
        var end = len(bytes)
        # Up to eight digits per step: the word's digit prefix is folded in
        # one go, so runs of any length up to eight cost one step and long
        # runs cost one step per eight digits. The byte loop handles the
        # last bytes of the buffer and digits past the 19-digit budget.
        while pos + 8 <= end:
            var chunk = _load_eight(bytes, pos)
            var count = _digit_prefix_length(chunk)
            if count == 0:
                return pos - start
            if not self.push_digits(
                _digit_prefix_value(chunk, count), count, in_fraction
            ):
                break
            pos += count
            if count < 8:
                return pos - start
        var ptr = bytes.unsafe_ptr()
        while pos < end:
            var byte = ptr[unsafe_offset=pos]
            if byte < ord_0 or byte > ord_9:
                break
            self.push_digit(byte - ord_0, in_fraction)
            pos += 1
        return pos - start


@fieldwise_init
struct FastDecimal(TrivialRegisterPassable):
    """The result of `scan_short_decimal`: a mantissa that needed no
    truncation, so it converts without the bookkeeping of `DecimalParts`.
    """

    var mantissa: UInt64
    """The digits as an integer."""
    var exponent: Int
    """Power of ten to apply to `mantissa` (minus the fraction length)."""
    var end: Int
    """Position just past the mantissa text."""
    var int_digits: Int
    """Digits before the point."""
    var frac_digits: Int
    """Digits after the point, or -1 when there is no point."""
    var ok: Bool
    """Whether the fast path applied; when `False` use `DecimalParts`."""
    var negative: Bool
    """Whether the text began with a minus sign."""
    var terminator: Byte
    """The byte at `end`, or zero when the text ends there. Saves the caller
    a dependent load when it looks for an exponent."""


comptime _FAST_FAIL = FastDecimal(0, 0, 0, 0, -1, False, False, 0)


@inline(.always)
def scan_short_decimal(bytes: ImmSpan[Byte, _], start: Int) -> FastDecimal:
    """Scans `[-]digits[.digits]` at `start` when the number is short.

    A number whose digits and point fit in one eight-byte word, the common
    case, is folded once: the point is cut out of the word and the digits
    on both sides of it become a single run. Otherwise an integer part
    under eight digits and a fraction under sixteen with at most nineteen
    digits in total are read with one word load per run and one
    multiply-add each. Neither shape needs truncation. Anything longer
    returns `ok=False` and the caller falls back to
    `DecimalParts.scan_digits`.

    Args:
        bytes: The text.
        start: Where the number begins, at its sign if it has one.

    Returns:
        The scanned number, or `_FAST_FAIL`.
    """
    comptime ord_dot = Byte(ord("."))
    comptime ord_dash = Byte(ord("-"))
    var end = len(bytes)
    if start < 0 or start >= end:
        return _FAST_FAIL
    var ptr = bytes.unsafe_ptr()
    var word = _load_up_to_eight(bytes, start)
    # The one-word fold applies when the mantissa ends inside the word. Its
    # length is measured first with one class test, so the next position
    # depends on a short chain (load, mask, count) and everything else,
    # the point, the merge and the multiplies, runs off that path. Long
    # numbers fail the length test and take the general path below.
    var length = _number_prefix_length(word)
    if length < 8:
        # The sign is part of the measured run, so the next position never
        # waits on it; it is only shifted out of the digits here.
        var negative = word.cast[DType.uint8]() == ord_dash
        var digits = word >> 8 if negative else word
        var count = length - Int(negative)
        var terminator = (word >> UInt64(8 * length)).cast[DType.uint8]()
        if count > 0:
            var keep = ~UInt64(0) >> UInt64(64 - 8 * count)
            var points = _zero_byte_mask((digits ^ 0x2E2E2E2E2E2E2E2E) | ~keep)
            var mantissa: UInt64 = 0
            if points == 0:
                if _digit_prefix_length(digits) >= count:
                    mantissa = _digit_prefix_value(digits, count)
                    return FastDecimal(
                        mantissa,
                        0,
                        start + length,
                        count,
                        -1,
                        True,
                        negative,
                        terminator,
                    )
            else:
                var point = Int(std.bit.count_trailing_zeros(points)) >> 3
                var low = (UInt64(1) << UInt64(8 * point)) - 1
                var merged = (digits & low) | ((digits >> 8) & ~low)
                var total = count - 1
                # Exactly one point, digits everywhere else; anything else
                # (`1.2.3`, `1/2`, `1-2`) is left to the general path and
                # its caller.
                if _digit_prefix_length(merged) >= total:
                    if total > 0:
                        mantissa = _digit_prefix_value(merged, total)
                    var frac_digits = total - point
                    return FastDecimal(
                        mantissa,
                        -frac_digits,
                        start + length,
                        point,
                        frac_digits,
                        True,
                        negative,
                        terminator,
                    )
    # A sign shifts the word by a byte, so seven integer digits may hide an
    # eighth; that case is left to `DecimalParts` like eight digits are.
    var negative = word.cast[DType.uint8]() == ord_dash
    var lead = word >> 8 if negative else word
    var pos = start + Int(negative)
    var int_digits = _digit_prefix_length(lead)
    if int_digits == 8 or (negative and int_digits == 7):
        return _FAST_FAIL
    var mantissa: UInt64 = 0
    if int_digits > 0:
        mantissa = _digit_prefix_value(lead, int_digits)
    pos += int_digits
    var frac_digits = -1
    if pos < end and ptr[unsafe_offset=pos] == ord_dot:
        pos += 1
        comptime pow10: Array[UInt64, 9] = [
            1,
            10,
            100,
            1000,
            10000,
            100000,
            1000000,
            10000000,
            100000000,
        ]
        var first = _load_up_to_eight(bytes, pos)
        var count = _digit_prefix_length(first)
        if count < 8:
            frac_digits = count
            if count > 0:
                mantissa = mantissa * materialize[pow10]()[
                    count
                ] + _digit_prefix_value(first, count)
            pos += count
        else:
            mantissa = mantissa * 100000000 + _eight_digits_value(first)
            pos += 8
            var second = _load_up_to_eight(bytes, pos)
            var more = _digit_prefix_length(second)
            if more == 8:
                return _FAST_FAIL
            frac_digits = 8 + more
            if more > 0:
                mantissa = mantissa * materialize[pow10]()[
                    more
                ] + _digit_prefix_value(second, more)
            pos += more
    var fraction = frac_digits if frac_digits > 0 else 0
    if int_digits + fraction > _MAX_SIGNIFICANT_DIGITS:
        return _FAST_FAIL
    var terminator: Byte = 0
    if pos < end:
        terminator = ptr[unsafe_offset=pos]
    return FastDecimal(
        mantissa,
        -fraction,
        pos,
        int_digits,
        frac_digits,
        True,
        negative,
        terminator,
    )


@inline(.always)
def short_decimal_to_float64(mantissa: UInt64, exponent: Int) -> Float64:
    """Converts an untruncated mantissa and decimal exponent.

    Args:
        mantissa: The digits as an integer (below `10**19`).
        exponent: The power of ten to apply.

    Returns:
        The correctly rounded magnitude.
    """
    var q = Int64(exponent)
    if can_use_clinger_fast_path(mantissa, q):
        return clinger_fast_path(mantissa, q)
    return lemire_algorithm(mantissa, q)


@inline(.always)
def decimal_to_float64(
    parts: DecimalParts, explicit_exponent: Int, digits: ImmSpan[Byte, _]
) -> Float64:
    """Converts scanned decimal digits to the nearest `Float64`.

    Args:
        parts: The accumulated digits.
        explicit_exponent: The value of the text's `e` exponent, or 0.
        digits: The mantissa text (digits and at most one `.`), only read
            when more than 19 significant digits must be resolved exactly.

    Returns:
        The correctly rounded magnitude; the caller applies the sign.
    """
    var q = Int64(parts.exponent + explicit_exponent)
    if not parts.truncated and can_use_clinger_fast_path(parts.mantissa, q):
        return clinger_fast_path(parts.mantissa, q)
    var low = lemire_algorithm(parts.mantissa, q)
    if not parts.truncated_nonzero:
        return low
    var high = lemire_algorithm(parts.mantissa + 1, q)
    if low == high:
        return low
    return _round_exactly(digits, explicit_exponent, low)


# Fixed-capacity unsigned big integer for the exact slow path. 4096 bits
# covers the worst case: 768 significant digits scaled by 10^340 or 2^1075.
struct _BigUInt(Movable):
    comptime LIMBS = 128

    var _limbs: Array[UInt32, Self.LIMBS]
    var _length: Int

    def __init__(out self, value: UInt64):
        self._limbs = {fill = 0}
        self._limbs[0] = UInt32(value & 0xFFFFFFFF)
        self._limbs[1] = UInt32(value >> 32)
        self._length = 2 if self._limbs[1] != 0 else (1 if value != 0 else 0)

    def is_zero(self) -> Bool:
        return self._length == 0

    def multiply_add(mut self, factor: UInt32, addend: UInt32):
        var carry = UInt64(addend)
        debug_assert(
            carry == 0 or self._length < Self.LIMBS, "_BigUInt overflow"
        )
        for i in range(self._length):
            var product = UInt64(self._limbs[i]) * UInt64(factor) + carry
            self._limbs[i] = UInt32(product & 0xFFFFFFFF)
            carry = product >> 32
        if carry != 0 and self._length < Self.LIMBS:
            self._limbs[self._length] = UInt32(carry)
            self._length += 1

    def multiply_pow10(mut self, exponent: Int):
        var remaining = exponent
        while remaining >= 9:
            self.multiply_add(1000000000, 0)
            remaining -= 9
        var factor: UInt32 = 1
        for _ in range(remaining):
            factor *= 10
        if factor != 1:
            self.multiply_add(factor, 0)

    def shift_left(mut self, bits: Int):
        var limb_shift = bits // 32
        var bit_shift = bits % 32
        if self._length == 0:
            return
        var new_length = min(self._length + limb_shift + 1, Self.LIMBS)
        for i in range(new_length - 1, -1, -1):
            var source = i - limb_shift
            var value: UInt32 = 0
            if source >= 0 and source < self._length:
                value = self._limbs[source] << UInt32(bit_shift)
            if bit_shift != 0 and source - 1 >= 0 and source - 1 < self._length:
                value |= self._limbs[source - 1] >> UInt32(32 - bit_shift)
            self._limbs[i] = value
        self._length = new_length
        while self._length > 0 and self._limbs[self._length - 1] == 0:
            self._length -= 1

    # Returns -1, 0 or 1.
    def compare(self, other: Self) -> Int:
        if self._length != other._length:
            return 1 if self._length > other._length else -1
        for i in range(self._length - 1, -1, -1):
            if self._limbs[i] != other._limbs[i]:
                return 1 if self._limbs[i] > other._limbs[i] else -1
        return 0


# Slow path for more than 19 significant digits whose truncations `w` and
# `w + 1` land on adjacent doubles. The exact decimal value is compared with
# big integers against the midpoint of `low` and its upper neighbour; ties go
# to the even mantissa. Only the first 768 significant digits are kept: any
# later non-zero digit can only push the value above the midpoint, which is
# all the comparison needs.
def _round_exactly(
    digits: ImmSpan[Byte, _], explicit_exponent: Int, low: Float64
) -> Float64:
    comptime ord_0 = Byte(ord("0"))
    comptime ord_dot = Byte(ord("."))
    var low_bits = std.memory.bitcast[.uint64](low)
    var exponent_field = Int((low_bits >> 52) & 0x7FF)
    var m = low_bits & ((UInt64(1) << 52) - 1)
    var e: Int
    if exponent_field == 0:
        e = -1074
    else:
        m |= UInt64(1) << 52
        e = exponent_field - 1075
    var high = std.memory.bitcast[.float64](low_bits + 1)
    # midpoint = (2m + 1) * 2^(e - 1)
    var midpoint = _BigUInt(2 * m + 1)
    var value = _BigUInt(0)
    var digit_count = 0
    var fraction_digits = 0
    var after_dot = False
    var sticky = False
    for byte in digits:
        if byte == ord_dot:
            after_dot = True
            continue
        if digit_count < _MAX_EXACT_DIGITS:
            value.multiply_add(10, UInt32(byte - ord_0))
            if not value.is_zero():
                digit_count += 1
            if after_dot:
                fraction_digits += 1
        else:
            sticky = sticky or byte != ord_0
            if not after_dot:
                fraction_digits -= 1
    var scale = explicit_exponent - fraction_digits
    if scale >= 0:
        value.multiply_pow10(scale)
    else:
        midpoint.multiply_pow10(-scale)
    if e - 1 >= 0:
        midpoint.shift_left(e - 1)
    else:
        value.shift_left(1 - e)
    var order = value.compare(midpoint)
    if order == 0 and sticky:
        order = 1
    if order > 0:
        return high
    if order < 0:
        return low
    return low if (m & 1) == 0 else high


comptime _ascii_lower: Byte = Byte(ord("A") ^ ord("a"))


@inline(.always)
def _is_nan(stripped: StringSlice) -> Bool:
    comptime `n` = Byte(ord("n"))
    comptime `a` = Byte(ord("a"))
    var ptr = stripped.as_bytes().unsafe_ptr()
    return stripped.byte_length() == 3 and (
        (ptr[unsafe_offset=0] | _ascii_lower == `n`)
        and (ptr[unsafe_offset=1] | _ascii_lower == `a`)
        and (ptr[unsafe_offset=2] | _ascii_lower == `n`)
    )


@inline(.always)
def _is_inf(stripped: StringSlice) -> Bool:
    comptime `i` = Byte(ord("i"))
    comptime `n` = Byte(ord("n"))
    comptime `f` = Byte(ord("f"))
    comptime `t` = Byte(ord("t"))
    comptime `y` = Byte(ord("y"))
    var ptr = stripped.as_bytes().unsafe_ptr()
    var in_start = (ptr[unsafe_offset=0] | _ascii_lower == `i`) and (
        ptr[unsafe_offset=1] | _ascii_lower == `n`
    )
    # f was removed previously
    var is_in = stripped.byte_length() == 2 and in_start
    return in_start and (
        is_in
        or (
            stripped.byte_length() == 8
            and (ptr[unsafe_offset=2] | _ascii_lower == `f`)
            and (ptr[unsafe_offset=3] | _ascii_lower == `i`)
            and (ptr[unsafe_offset=4] | _ascii_lower == `n`)
            and (ptr[unsafe_offset=5] | _ascii_lower == `i`)
            and (ptr[unsafe_offset=6] | _ascii_lower == `t`)
            and (ptr[unsafe_offset=7] | _ascii_lower == `y`)
        )
    )


# Parses `[eE][+-]digits` at `pos`, which must be at the `e`. `pos` ends
# after the digits.
@inline(.always)
def _parse_exponent(
    text: StringSpan, bytes: ImmSpan[Byte, _], mut pos: Int
) raises -> Int:
    comptime ord_0 = Byte(ord("0"))
    comptime ord_9 = Byte(ord("9"))
    comptime ord_minus = Byte(ord("-"))
    comptime ord_plus = Byte(ord("+"))
    var ptr = bytes.unsafe_ptr()
    var end = len(bytes)
    pos += 1
    var negative = False
    if pos < end and (
        ptr[unsafe_offset=pos] == ord_plus
        or ptr[unsafe_offset=pos] == ord_minus
    ):
        negative = ptr[unsafe_offset=pos] == ord_minus
        pos += 1
    if pos >= end:
        raise Error(t"Invalid character(s) in the number: '{text}'")
    var value = 0
    while pos < end:
        var byte = ptr[unsafe_offset=pos]
        if byte < ord_0 or byte > ord_9:
            raise Error(t"Invalid character(s) in the number: '{text}'")
        if value < 100000:
            value = value * 10 + Int(byte - ord_0)
        pos += 1
    return -value if negative else value


def _parse_decimal_text(text: StringSpan) raises -> Float64:
    comptime ord_0 = Byte(ord("0"))
    comptime ord_9 = Byte(ord("9"))
    comptime ord_dot = Byte(ord("."))
    comptime ord_e = Byte(ord("e"))
    comptime ord_E = Byte(ord("E"))

    var bytes = text.as_bytes()
    var end = len(bytes)
    var ptr = bytes.unsafe_ptr()
    if end == 0 or not (
        _is_digit(ptr[unsafe_offset=0]) or ptr[unsafe_offset=0] == ord_dot
    ):
        raise Error(
            (
                t"The first character of '{text}' should be a digit or dot to"
                t" convert it to a float."
            ),
        )
    var last = ptr[unsafe_offset=end - 1]
    if not (_is_digit(last) or last == ord_dot):
        raise Error(
            "The last character of '",
            text,
            "' should be a digit or dot to convert it to a float.",
        )
    # The word path pays off from about sixteen bytes; shorter strings are
    # cheaper digit by digit.
    if end >= 16:
        var fast = scan_short_decimal(bytes, 0)
        if (
            fast.ok
            and not fast.negative
            and fast.int_digits + max(fast.frac_digits, 0) > 0
        ):
            var pos = fast.end
            var exponent = 0
            if pos < end:
                if not (
                    ptr[unsafe_offset=pos] == ord_e
                    or ptr[unsafe_offset=pos] == ord_E
                ):
                    raise Error(
                        "Invalid character(s) in the number: '", text, "'"
                    )
                exponent = _parse_exponent(text, bytes, pos)
            return short_decimal_to_float64(
                fast.mantissa, fast.exponent + exponent
            )
    var parts = DecimalParts()
    var pos = 0
    var digits = parts.scan_digits(bytes, pos, False)
    if pos < end and ptr[unsafe_offset=pos] == ord_dot:
        pos += 1
        digits += parts.scan_digits(bytes, pos, True)
    if digits == 0:
        raise Error(t"Invalid character(s) in the number: '{text}'")
    var mantissa_end = pos
    var explicit_exponent = 0
    if pos < end:
        if not (
            ptr[unsafe_offset=pos] == ord_e or ptr[unsafe_offset=pos] == ord_E
        ):
            raise Error(t"Invalid character(s) in the number: '{text}'")
        explicit_exponent = _parse_exponent(text, bytes, pos)
    return decimal_to_float64(parts, explicit_exponent, bytes[0:mantissa_end])


def _atof(x: StringSlice) raises -> Float64:
    """Parses the given string as a floating point and returns that value.

    For example, `atof("2.25")` returns `2.25`.

    Raises:
        If the given string cannot be parsed as an floating point value, for
        example in `atof("hi")`.

    Args:
        x: A string to be parsed as a floating point.

    Returns:
        An floating point value that represents the string, or otherwise raises.
    """
    if x == "" or x == ".":
        raise Error("String is not convertible to float: ", repr(x))
    var stripped = strip_unused_characters(x)
    var sign_and_stripped = get_sign(stripped)
    var sign = sign_and_stripped[0]
    stripped = sign_and_stripped[1]
    if _is_nan(stripped):
        return FloatLiteral.nan
    elif _is_inf(stripped):
        return FloatLiteral.infinity * sign
    try:
        return _parse_decimal_text(stripped) * sign
    except e:
        raise Error("String is not convertible to float: ", repr(x), ". ", e)
