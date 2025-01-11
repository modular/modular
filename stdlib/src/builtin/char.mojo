# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
"""Implements the `Char` type for representing single characters."""

from collections import Optional

from memory import UnsafePointer


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
struct Char(CollectionElement):
    """A single textual character.

    This type represents a single textual character. Specifically, this type
    stores a single Unicode [*scalar value*][1], typically encoding a single
    user-recognizable character.

    All valid Unicode scalar values are in the range(s) 0 to 0xD7FF and
    0xE000 to 0x10FFFF, inclusive. This type guarantees that the stored integer
    value falls in these ranges.

    [1]: https://www.unicode.org/glossary/#unicode_scalar_value
    """

    var _scalar_value: UInt32
    """The Unicode scalar value represented by this type."""

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

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __init__(out self, *, unsafe_unchecked_codepoint: UInt32):
        """Construct a `Char` from a code point value without checking that it
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
        """Construct a `Char` from a single byte value.

        This constructor cannot fail because non-negative 8-bit integers are
        valid Unicode scalar values.

        Args:
            codepoint: The 8-bit codepoint value to convert to a `Char`.
        """
        self._scalar_value = UInt32(Int(codepoint))

    # ===-------------------------------------------------------------------===#
    # Factory methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    fn from_u32(codepoint: UInt32) -> Optional[Self]:
        """Construct a `Char` from a code point value. Returns None if the
        provided `codepoint` is not in the valid range.

        Args:
            codepoint: An integer representing a Unicode scalar value.

        Returns:
            A `Char` if `codepoint` falls in the valid range for Unicode scalar
            values, otherwise None.
        """

        if _is_unicode_scalar_value(codepoint):
            return Char(unsafe_unchecked_codepoint=codepoint)
        else:
            return None

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn is_ascii(self) -> Bool:
        """Returns True if this `Char` is an ASCII character.

        All ASCII characters are less than or equal to codepoint value 127, and
        take exactly 1 byte to encode in UTF-8.

        Returns:
            A boolean indicating if this `Char` is an ASCII character.
        """
        return self._scalar_value <= 0b0111_1111

    @always_inline
    fn unsafe_write_utf8(self, ptr: UnsafePointer[Byte]) -> UInt:
        """Shift unicode to utf8 representation.

        Safety:
            `ptr` MUST point to at least `self.utf8_byte_length()` allocated
            bytes or else an out-of-bounds write will occur, which is undefined
            behavior.

        Args:
            ptr: Pointer value to write the encoded UTF-8 bytes. Must validly
                point to a sufficient number of bytes (1-4) to hold the encoded
                data.

        Returns:
            Returns the number of bytes written.

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
        alias sizes = SIMD[DType.int32, 4](
            0,
            2**7,
            2**11,
            2**16,
        )

        # Count how many of the minimums this codepoint exceeds, which is equal
        # to the number of bytes needed to encode it.
        var lt = (sizes <= Int(self)).cast[DType.uint8]()

        # TODO(MOCO-1537): Support `reduce_add()` at compile time.
        #   var count = Int(lt.reduce_add())
        var count = 0
        for i in range(len(lt)):
            count += Int(lt[i])

        return UInt(count)
