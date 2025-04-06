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
"""Unicode codepoint handling.

This module provides the `Codepoint` type for representing single Unicode scalar values.
A codepoint represents a single Unicode character, restricted to valid Unicode scalar
values in the ranges 0 to 0xD7FF and 0xE000 to 0x10FFFF inclusive.

The `Codepoint` type provides functionality for:
- Converting between codepoints and UTF-8 encoded bytes
- Testing character properties like ASCII, digits, whitespace etc.
- Converting between codepoints and strings
- Safe construction from integers with validation

Example:
```mojo
    from collections.string import Codepoint
    from testing import assert_true

    # Create a codepoint from a character
    var c = Codepoint.ord('A')

    # Check properties
    assert_true(c.is_ascii())
    assert_true(c.is_ascii_upper())

    # Convert to string
    var s = String(c)  # "A"
```
"""

from collections import Optional
from collections.string import StringSlice

from bit import count_leading_zeros
from memory import UnsafePointer
from sys.intrinsics import likely


@always_inline
fn _is_unicode_scalar_value(codepoint: UInt32) -> Bool:
    """Returns True if `codepoint` is a valid Unicode scalar value.

    Args:
        codepoint: The codepoint integer value to check.

    Returns:
        True if `codepoint` is a valid Unicode scalar value; False otherwise.
    """
    return codepoint <= 0xD7FF or (
        codepoint >= 0xE000 and codepoint <= 0x10FFFF
    )


@value
struct Codepoint(
    CollectionElement,
    EqualityComparable,
    Intable,
    Stringable,
    EqualityComparableCollectionElement,
):
    """A Unicode codepoint, typically a single user-recognizable character;
    restricted to valid Unicode scalar values.

    This type is restricted to store a single Unicode [*scalar value*][1],
    typically encoding a single user-recognizable character.

    All valid Unicode scalar values are in the range(s) 0 to 0xD7FF and
    0xE000 to 0x10FFFF, inclusive. This type guarantees that the stored integer
    value falls in these ranges.

    [1]: https://www.unicode.org/glossary/#unicode_scalar_value

    # Codepoints vs Scalar Values

    Formally, Unicode defines a codespace of values in the range 0 to
    0x10FFFF inclusive, and a
    [Unicode codepoint](https://www.unicode.org/glossary/#code_point) is any
    integer falling within that range. However, due to historical reasons,
    it became necessary to "carve out" a subset of the codespace, excluding
    codepoints in the range 0xD7FF–0xE000. That subset of codepoints excluding
    that range are known as [Unicode scalar values][1]. The codepoints in the
    range 0xD7FF-0xE000 are known as "surrogate" codepoints. The surrogate
    codepoints will never be assigned a semantic meaning, and can only
    validly appear in UTF-16 encoded text.

    The difference between codepoints and scalar values is a technical
    distiction related to the backwards-compatible workaround chosen to enable
    UTF-16 to encode the full range of the Unicode codespace. For simplicities
    sake, and to avoid a confusing clash with the Mojo `Scalar` type, this type
    is pragmatically named `Codepoint`, even though it is restricted to valid
    scalar values.
    """

    var _scalar_value: UInt32
    """The Unicode scalar value represented by this type."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __init__(out self, *, unsafe_unchecked_codepoint: UInt32):
        """Construct a `Codepoint` from a code point value without checking that it
        falls in the valid range.

        Safety:
            The provided codepoint value MUST be a valid Unicode scalar value.
            Providing a value outside of the valid range could lead to undefined
            behavior in algorithms that depend on the validity guarantees of
            this type.

        Args:
            unsafe_unchecked_codepoint: A valid Unicode scalar value code point.
        """
        debug_assert(
            _is_unicode_scalar_value(unsafe_unchecked_codepoint),
            "codepoint is not a valid Unicode scalar value",
        )

        self._scalar_value = unsafe_unchecked_codepoint

    @always_inline
    fn __init__(out self, codepoint: UInt8):
        """Construct a `Codepoint` from a single byte value.

        This constructor cannot fail because non-negative 8-bit integers are
        valid Unicode scalar values.

        Args:
            codepoint: The 8-bit codepoint value to convert to a `Codepoint`.
        """
        self._scalar_value = UInt32(Int(codepoint))

    # ===-------------------------------------------------------------------===#
    # Factory methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    fn from_u32(codepoint: UInt32) -> Optional[Self]:
        """Construct a `Codepoint` from a code point value. Returns None if the
        provided `codepoint` is not in the valid range.

        Args:
            codepoint: An integer representing a Unicode scalar value.

        Returns:
            A `Codepoint` if `codepoint` falls in the valid range for Unicode
            scalar values, otherwise None.
        """

        if _is_unicode_scalar_value(codepoint):
            return Codepoint(unsafe_unchecked_codepoint=codepoint)
        else:
            return None

    @staticmethod
    fn ord(string: StringSlice) -> Codepoint:
        """Returns the `Codepoint` that represents the given single-character
        string.

        Given a string containing one character, return a `Codepoint`
        representing the codepoint of that character. For example,
        `Codepoint.ord("a")` returns the codepoint `97`. This is the inverse of
        the `chr()` function.

        This function is similar to the `ord()` free function, except that it
        returns a `Codepoint` instead of an `Int`.

        Args:
            string: The input string, which must contain only a single character.

        Returns:
            A `Codepoint` representing the codepoint of the given character.
        """

        # SAFETY:
        #   This is safe because `StringSlice` is guaranteed to point to valid
        #   UTF-8.
        char, num_bytes = Codepoint.unsafe_decode_utf8_codepoint(
            string.unsafe_ptr()
        )

        debug_assert(
            string.byte_length() == Int(num_bytes),
            "input string must be one character",
        )

        return char

    @staticmethod
    fn unsafe_decode_utf8_codepoint(
        _ptr: UnsafePointer[Byte],
    ) -> (Codepoint, Int):
        """Decodes a single `Codepoint` and number of bytes read from a given
        UTF-8 string pointer.

        Safety:
            `_ptr` MUST point to the first byte in a **known-valid** UTF-8
            character sequence. This function MUST NOT be used on unvalidated
            input.

        Args:
            _ptr: Pointer to UTF-8 encoded data containing at least one valid
                encoded codepoint.

        Returns:
            The decoded codepoint `Codepoint`, as well as the number of bytes
            read.

        """
        # UTF-8 to Unicode conversion:              (represented as UInt32 BE)
        # 1: 0aaaaaaa                            -> 00000000 00000000 00000000 0aaaaaaa     a
        # 2: 110aaaaa 10bbbbbb                   -> 00000000 00000000 00000aaa aabbbbbb     a << 6  | b
        # 3: 1110aaaa 10bbbbbb 10cccccc          -> 00000000 00000000 aaaabbbb bbcccccc     a << 12 | b << 6  | c
        # 4: 11110aaa 10bbbbbb 10cccccc 10dddddd -> 00000000 000aaabb bbbbcccc ccdddddd     a << 18 | b << 12 | c << 6 | d
        var ptr = _ptr

        var b1 = ptr[]
        if (b1 >> 7) == 0:  # This is 1 byte ASCII char
            return Codepoint(b1), 1

        # TODO: Use _utf8_first_byte_sequence_length() here instead for
        #   consistency.
        var num_bytes = count_leading_zeros(~b1)
        debug_assert(
            1 < Int(num_bytes) < 5, "invalid UTF-8 byte ", b1, " at index 0"
        )

        var shift = Int((6 * (num_bytes - 1)))
        var b1_mask = 0b11111111 >> (num_bytes + 1)
        var result = Int(b1 & b1_mask) << shift
        for i in range(1, num_bytes):
            ptr += 1
            # Assert that this is a continuation byte
            debug_assert(
                ptr[] >> 6 == 0b00000010,
                "invalid UTF-8 byte ",
                ptr[],
                " at index ",
                i,
            )
            shift -= 6
            result |= Int(ptr[] & 0b00111111) << shift

        # SAFETY: Safe because the input bytes are required to be valid UTF-8,
        #   and valid UTF-8 will never decode to an out of bounds codepoint
        #   using the above algorithm.
        # FIXME:
        #   UTF-8 encoding algorithms that do not properly exclude surrogate
        #   pair code points are actually relatively common (as I understand
        #   it); the algorithm above does not check for that.
        var char = Codepoint(unsafe_unchecked_codepoint=result)

        return char, Int(num_bytes)

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    fn __eq__(self, other: Self) -> Bool:
        """Return True if this character has the same codepoint value as `other`.

        Args:
            other: The codepoint value to compare against.

        Returns:
            True if this character and `other` have the same codepoint value;
            False otherwise.
        """
        return self.to_u32() == other.to_u32()

    fn __ne__(self, other: Self) -> Bool:
        """Return True if this character has a different codepoint value from
        `other`.

        Args:
            other: The codepoint value to compare against.

        Returns:
            True if this character and `other` have different codepoint values;
            False otherwise.
        """
        return self.to_u32() != other.to_u32()

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __int__(self) -> Int:
        """Returns the numeric value of this scalar value as an integer.

        Returns:
            The numeric value of this scalar value as an integer.
        """
        return Int(self._scalar_value)

    @always_inline
    fn __str__(self) -> String:
        """Formats this `Codepoint` as a single-character string.

        Returns:
            A string containing this single character.
        """
        var char_len = self.utf8_byte_length()
        var buffer = List[Byte](capacity=char_len + 1)
        _ = self.unsafe_write_utf8(buffer.unsafe_ptr())
        buffer.unsafe_ptr()[char_len] = 0
        buffer._len = char_len + 1
        return String(buffer=buffer^)

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn is_ascii(self) -> Bool:
        """Returns True if this `Codepoint` is an ASCII character.

        All ASCII characters are less than or equal to codepoint value 127, and
        take exactly 1 byte to encode in UTF-8.

        Returns:
            A boolean indicating if this `Codepoint` is an ASCII character.
        """
        return self._scalar_value <= 0b0111_1111

    fn is_ascii_digit(self) -> Bool:
        """Determines whether the given character is a digit [0-9].

        Returns:
            True if the character is a digit.
        """
        alias ord_0 = UInt32(ord("0"))
        alias ord_9 = UInt32(ord("9"))
        return ord_0 <= self.to_u32() <= ord_9

    fn is_ascii_upper(self) -> Bool:
        """Determines whether the given character is an uppercase character.

        This currently only respects the default "C" locale, i.e. returns True
        iff the character specified is one of "ABCDEFGHIJKLMNOPQRSTUVWXYZ".

        Returns:
            True if the character is uppercase.
        """
        alias ord_a = UInt32(ord("A"))
        alias ord_z = UInt32(ord("Z"))
        return ord_a <= self.to_u32() <= ord_z

    fn is_ascii_lower(self) -> Bool:
        """Determines whether the given character is an lowercase character.

        This currently only respects the default "C" locale, i.e. returns True
        iff the character specified is one of "abcdefghijklmnopqrstuvwxyz".

        Returns:
            True if the character is lowercase.
        """
        alias ord_a = UInt32(ord("a"))
        alias ord_z = UInt32(ord("z"))
        return ord_a <= self.to_u32() <= ord_z

    fn is_ascii_printable(self) -> Bool:
        """Determines whether the given character is a printable character.

        Returns:
            True if the character is a printable character, otherwise False.
        """
        alias ord_space = UInt32(ord(" "))
        alias ord_tilde = UInt32(ord("~"))
        return ord_space <= self.to_u32() <= ord_tilde

    @always_inline
    fn is_python_space(self) -> Bool:
        """Determines whether this character is a Python whitespace string.

        This corresponds to Python's [universal separators](
        https://docs.python.org/3/library/stdtypes.html#str.splitlines):
         `" \\t\\n\\v\\f\\r\\x1c\\x1d\\x1e\\x85\\u2028\\u2029"`.

        Returns:
            True if this character is one of the whitespace characters listed
            above, otherwise False.

        # Examples

        Check if a string contains only whitespace:

        ```mojo
        from testing import assert_true, assert_false

        # ASCII space characters
        assert_true(Codepoint.ord(" ").is_python_space())
        assert_true(Codepoint.ord("\t").is_python_space())

        # Unicode paragraph separator:
        assert_true(Codepoint.from_u32(0x2029).value().is_python_space())

        # Letters are not space characters
        assert_fales(Codepoint.ord("a").is_python_space())
        ```
        .
        """

        alias next_line = Codepoint.from_u32(0x85).value()
        alias unicode_line_sep = Codepoint.from_u32(0x2028).value()
        alias unicode_paragraph_sep = Codepoint.from_u32(0x2029).value()

        return self.is_posix_space() or self in (
            next_line,
            unicode_line_sep,
            unicode_paragraph_sep,
        )

    fn is_posix_space(self) -> Bool:
        """Returns True if this `Codepoint` is a **space** character according to the
        [POSIX locale][1].

        The POSIX locale is also known as the C locale.

        [1]: https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap07.html#tag_07_03_01

        This only respects the default "C" locale, i.e. returns True only if the
        character specified is one of " \\t\\n\\v\\f\\r". For semantics similar
        to Python, use `String.isspace()`.

        Returns:
            True iff the character is one of the whitespace characters listed
            above.
        """
        if not self.is_ascii():
            return False

        # ASCII char
        var c = UInt8(Int(self))

        # NOTE: a global LUT doesn't work at compile time so we can't use it here.
        alias ` ` = UInt8(ord(" "))
        alias `\t` = UInt8(ord("\t"))
        alias `\n` = UInt8(ord("\n"))
        alias `\r` = UInt8(ord("\r"))
        alias `\f` = UInt8(ord("\f"))
        alias `\v` = UInt8(ord("\v"))
        alias `\x1c` = UInt8(ord("\x1c"))
        alias `\x1d` = UInt8(ord("\x1d"))
        alias `\x1e` = UInt8(ord("\x1e"))

        # This compiles to something very clever that's even faster than a LUT.
        return (
            c == ` `
            or c == `\t`
            or c == `\n`
            or c == `\r`
            or c == `\f`
            or c == `\v`
            or c == `\x1c`
            or c == `\x1d`
            or c == `\x1e`
        )

    @always_inline
    fn to_u32(self) -> UInt32:
        """Returns the numeric value of this scalar value as an unsigned 32-bit
        integer.

        Returns:
            The numeric value of this scalar value as an unsigned 32-bit
            integer.
        """
        return self._scalar_value

    @always_inline
    fn unsafe_write_utf8[
        optimize_ascii: Bool = True
    ](self, ptr: UnsafePointer[Byte]) -> UInt:
        """Shift unicode to utf8 representation.

        Parameters:
            optimize_ascii: Optimize for languages with mostly ASCII characters.

        Args:
            ptr: Pointer value to write the encoded UTF-8 bytes. Must validly
                point to a sufficient number of bytes (1-4) to hold the encoded
                data.

        Returns:
            Returns the number of bytes written.

        Safety:
            `ptr` MUST point to at least `self.utf8_byte_length()` allocated
            bytes or else an out-of-bounds write will occur, which is undefined
            behavior.

        ### Unicode (represented as UInt32 BE) to UTF-8 conversion:
        - 1: 00000000 00000000 00000000 0aaaaaaa -> 0aaaaaaa
            - a
        - 2: 00000000 00000000 00000aaa aabbbbbb -> 110aaaaa 10bbbbbb
            - (a >> 6)  | 0b11000000, b         | 0b10000000
        - 3: 00000000 00000000 aaaabbbb bbcccccc -> 1110aaaa 10bbbbbb 10cccccc
            - (a >> 12) | 0b11100000, (b >> 6)  | 0b10000000, c        | 0b10000000
        - 4: 00000000 000aaabb bbbbcccc ccdddddd -> 11110aaa 10bbbbbb 10cccccc
        10dddddd
            - (a >> 18) | 0b11110000, (b >> 12) | 0b10000000, (c >> 6) | 0b10000000,
            d | 0b10000000
        .
        """
        var c = Int(self)

        var num_bytes = self.utf8_byte_length()

        @parameter
        if optimize_ascii:
            # FIXME(#933): can't run LLVM intrinsic at compile time
            # if likely(num_bytes == 1):
            if num_bytes == 1:
                ptr[0] = UInt8(c)
                return 1
            var shift = 6 * (num_bytes - 1)
            var mask = UInt8(0xFF) >> (num_bytes + 1)
            var num_bytes_marker = UInt8(0xFF) << (8 - num_bytes)
            ptr[0] = ((c >> shift) & mask) | num_bytes_marker
            for i in range(1, num_bytes):
                shift -= 6
                ptr[i] = ((c >> shift) & 0b0011_1111) | 0b1000_0000
        else:
            var shift = 6 * (num_bytes - 1)
            var mask = UInt8(0xFF) >> (num_bytes + Int(num_bytes > 1))
            var num_bytes_marker = UInt8(0xFF) << (8 - num_bytes)
            ptr[0] = ((c >> shift) & mask) | (
                num_bytes_marker & -Int(num_bytes != 1)
            )
            for i in range(1, num_bytes):
                shift -= 6
                ptr[i] = ((c >> shift) & 0b0011_1111) | 0b1000_0000
        return num_bytes

    @always_inline
    fn utf8_byte_length(self) -> UInt:
        """Returns the number of UTF-8 bytes required to encode this character.

        The returned value is always between 1 and 4 bytes.

        Returns:
            Byte count of UTF-8 bytes required to encode this character.
        """

        # Minimum codepoint values (respectively) that can fit in a 1, 2, 3,
        # and 4 byte encoded UTF-8 sequence.
        alias sizes = SIMD[DType.uint32, 4](0, 2**7, 2**11, 2**16)

        # Count how many of the minimums this codepoint exceeds, which is equal
        # to the number of bytes needed to encode it.
        var lt = (sizes <= self.to_u32()).cast[DType.uint8]()

        # TODO(MOCO-1537): Support `reduce_add()` at compile time.
        #   var count = Int(lt.reduce_add())
        var count = 0
        for i in range(len(lt)):
            count += Int(lt[i])

        return UInt(count)
