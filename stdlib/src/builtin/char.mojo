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
        return UInt(Int((sizes <= Int(self)).cast[DType.uint8]().reduce_add()))
