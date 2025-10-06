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

from collections.string._unicode_lookups import *

from memory import Span
from sys.intrinsics import likely


fn _uppercase_mapping_index(rune: Codepoint) -> Int:
    """Return index for upper case mapping or -1 if no mapping is given."""
    return _to_index[has_uppercase_mapping](rune)


fn _uppercase_mapping2_index(rune: Codepoint) -> Int:
    """Return index for upper case mapping converting the rune to 2 runes, or -1 if no mapping is given.
    """
    return _to_index[has_uppercase_mapping2](rune)


fn _uppercase_mapping3_index(rune: Codepoint) -> Int:
    """Return index for upper case mapping converting the rune to 3 runes, or -1 if no mapping is given.
    """
    return _to_index[has_uppercase_mapping3](rune)


fn _lowercase_mapping_index(rune: Codepoint) -> Int:
    """Return index for lower case mapping or -1 if no mapping is given."""
    return _to_index[has_lowercase_mapping](rune)


@always_inline
fn _to_index[lookup: List[UInt32, **_]](rune: Codepoint) -> Int:
    """Find index of rune in lookup with binary search.
    Returns -1 if not found."""

    var result = materialize[lookup]()._binary_search_index(rune.to_u32())

    if result:
        return Int(result.unsafe_value())
    else:
        return -1


# TODO:
#   Refactor this to return a Span[Codepoint, StaticConstantOrigin], so that the
#   return `UInt` count and fixed-size `InlineArray` are not necessary.
fn _get_uppercase_mapping(
    char: Codepoint,
) -> Optional[Tuple[UInt, InlineArray[Codepoint, 3]]]:
    """Returns the 1, 2, or 3 character sequence that is the uppercase form of
    `char`.

    Returns None if `char` does not have an uppercase equivalent.
    """
    var array = InlineArray[Codepoint, 3](fill=Codepoint(0))

    var index1 = _uppercase_mapping_index(char)
    if index1 != -1:
        var rune = materialize[uppercase_mapping]()[index1]
        array[0] = Codepoint(unsafe_unchecked_codepoint=rune)
        return Tuple(UInt(1), array)

    var index2 = _uppercase_mapping2_index(char)
    if index2 != -1:
        var runes = materialize[uppercase_mapping2]()[index2]
        array[0] = Codepoint(unsafe_unchecked_codepoint=runes[0])
        array[1] = Codepoint(unsafe_unchecked_codepoint=runes[1])
        return Tuple(UInt(2), array)

    var index3 = _uppercase_mapping3_index(char)
    if index3 != -1:
        var runes = materialize[uppercase_mapping3]()[index3]
        array[0] = Codepoint(unsafe_unchecked_codepoint=runes[0])
        array[1] = Codepoint(unsafe_unchecked_codepoint=runes[1])
        array[2] = Codepoint(unsafe_unchecked_codepoint=runes[2])
        return Tuple(UInt(3), array)

    return None


fn _get_lowercase_mapping(char: Codepoint) -> Optional[Codepoint]:
    var index: Optional[UInt] = materialize[
        has_lowercase_mapping
    ]()._binary_search_index(char.to_u32())

    if index:
        # SAFETY: We just checked that `result` is present.
        var codepoint = materialize[lowercase_mapping]()[index.unsafe_value()]

        # SAFETY:
        #   We know this is a valid `Codepoint` because the mapping data tables
        #   contain only valid codepoints.
        return Codepoint(unsafe_unchecked_codepoint=codepoint)
    else:
        return None


fn is_uppercase(s: StringSlice[mut=False]) -> Bool:
    """Returns True if all characters in the string are uppercase, and
        there is at least one cased character.

    Args:
        s: The string to examine.

    Returns:
        True if all characters in the string are uppercaseand
        there is at least one cased character, False otherwise.
    """
    var found = False
    for char in s.codepoints():
        var index = _lowercase_mapping_index(char)
        if index != -1:
            found = True
            continue
        index = _uppercase_mapping_index(char)
        if index != -1:
            return False
        index = _uppercase_mapping2_index(char)
        if index != -1:
            return False
        index = _uppercase_mapping3_index(char)
        if index != -1:
            return False
    return found


fn is_lowercase(s: StringSlice[mut=False]) -> Bool:
    """Returns True if all characters in the string are lowercase, and
        there is at least one cased character.

    Args:
        s: The string to examine.

    Returns:
        True if all characters in the string are lowercase and
        there is at least one cased character, False otherwise.
    """
    var found = False
    for char in s.codepoints():
        var index = _uppercase_mapping_index(char)
        if index != -1:
            found = True
            continue
        index = _uppercase_mapping2_index(char)
        if index != -1:
            found = True
            continue
        index = _uppercase_mapping3_index(char)
        if index != -1:
            found = True
            continue
        index = _lowercase_mapping_index(char)
        if index != -1:
            return False
    return found


fn to_lowercase(s: StringSlice[mut=False]) -> String:
    """Returns a new string with all characters converted to uppercase.

    Args:
        s: Input string.

    Returns:
        A new string where cased letters have been converted to lowercase.
    """
    var data = s.as_bytes()
    # lowercased strings always have the same amount of bytes
    var result = String(capacity=len(data))
    var input_offset = 0
    while input_offset < len(data):
        alias `a` = Byte(ord("a"))
        alias `A` = Byte(ord("A"))
        alias lower_ascii_latin1 = `A` ^ `a`

        ref rune, ref size = Codepoint.unsafe_decode_utf8_codepoint(
            data[input_offset:]
        )

        if likely(size == 1):  # ASCII fast path
            alias `z` = Byte(ord("z"))
            var b = Byte(rune.to_u32())
            var low = b | lower_ascii_latin1
            result.append_byte(low if `a` <= low <= `z` else b)
        elif size == 2:  # latin-1 fast path
            alias `à` = Byte(ord("à"))
            alias `þ` = Byte(ord("þ"))
            alias `÷` = Byte(ord("÷"))
            var b = Byte(rune.to_u32())
            var low = b | lower_ascii_latin1
            var c = Codepoint(low if `à` <= low <= `þ` and low != `÷` else b)
            result += String(c)
        else:
            var lowercase_char_opt = _get_lowercase_mapping(rune)
            if lowercase_char_opt is None:
                result.write_bytes(data[input_offset : input_offset + size])
            else:
                result += String(lowercase_char_opt.unsafe_value())

        input_offset += size

    return result^


fn to_uppercase(s: StringSlice[mut=False]) -> String:
    """Returns a new string with all characters converted to uppercase.

    Args:
        s: Input string.

    Returns:
        A new string where cased letters have been converted to uppercase.
    """
    var data = s.as_bytes()
    # estimate the size since some codepoints require multiple chars to uppercase
    var result = String(capacity=3 * (len(data) // 2))
    var input_offset = 0
    while input_offset < len(data):
        alias `a` = Byte(ord("a"))
        alias `A` = Byte(ord("A"))
        alias upper_ascii_latin1 = ~(`A` ^ `a`)

        ref rune, ref size = Codepoint.unsafe_decode_utf8_codepoint(
            data[input_offset:]
        )
        if likely(size == 1):  # ASCII fast path
            alias `Z` = Byte(ord("Z"))
            var b = Byte(rune.to_u32())
            var up = b & upper_ascii_latin1
            result.append_byte(up if `A` <= up <= `Z` else b)
        elif size == 2:  # latin-1 fast path
            alias `À` = Byte(ord("À"))
            alias `Þ` = Byte(ord("Þ"))
            alias `×` = Byte(ord("×"))
            alias `ÿ` = Byte(ord("ÿ"))
            alias `Ÿ` = Byte(ord("Ÿ"))
            alias `ß` = Byte(ord("ß"))
            var b = Byte(rune.to_u32())
            if likely(b != `ß`):
                b = `Ÿ` if b == `ÿ` else b
                var up = b & upper_ascii_latin1
                var c = Codepoint(up if `À` <= up <= `Þ` and up != `×` else b)
                result += String(c)
            else:
                result += "SS"
        else:
            var uppercase_replacement_opt = _get_uppercase_mapping(rune)

            if uppercase_replacement_opt:
                # A given character can be replaced with a sequence of characters
                # up to 3 characters in length. A fixed size `Codepoint` array is
                # returned, along with a `count` (1, 2, or 3) of how many
                # replacement characters are in the uppercase replacement sequence.
                count, uppercase_replacement_chars = (
                    uppercase_replacement_opt.unsafe_value()
                )
                for char_idx in range(count):
                    result += String(uppercase_replacement_chars[char_idx])
            else:
                result.write_bytes(data[input_offset : input_offset + size])

        input_offset += size

    return result^
