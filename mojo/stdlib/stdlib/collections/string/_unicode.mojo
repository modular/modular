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

    var result = lookup._binary_search_index(rune.to_u32())

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
        var rune = uppercase_mapping[index1]
        array[0] = Codepoint(unsafe_unchecked_codepoint=rune)
        return Tuple(UInt(1), array)

    var index2 = _uppercase_mapping2_index(char)
    if index2 != -1:
        var runes = uppercase_mapping2[index2]
        array[0] = Codepoint(unsafe_unchecked_codepoint=runes[0])
        array[1] = Codepoint(unsafe_unchecked_codepoint=runes[1])
        return Tuple(UInt(2), array)

    var index3 = _uppercase_mapping3_index(char)
    if index3 != -1:
        var runes = uppercase_mapping3[index3]
        array[0] = Codepoint(unsafe_unchecked_codepoint=runes[0])
        array[1] = Codepoint(unsafe_unchecked_codepoint=runes[1])
        array[2] = Codepoint(unsafe_unchecked_codepoint=runes[2])
        return Tuple(UInt(3), array)

    return None


fn _get_lowercase_mapping(char: Codepoint) -> Optional[Codepoint]:
    var index: Optional[UInt] = has_lowercase_mapping._binary_search_index(
        char.to_u32()
    )

    if index:
        # SAFETY: We just checked that `result` is present.
        var codepoint = lowercase_mapping[index.unsafe_value()]

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
    var result = String(capacity=_estimate_needed_size(len(data)))
    var input_offset = 0
    while input_offset < len(data):
        var rune_and_size = Codepoint.unsafe_decode_utf8_codepoint(
            data[input_offset:]
        )
        var lowercase_char_opt = _get_lowercase_mapping(rune_and_size[0])
        if lowercase_char_opt is None:
            result.write_bytes(
                data[input_offset : input_offset + rune_and_size[1]]
            )
        else:
            result += String(lowercase_char_opt.unsafe_value())

        input_offset += rune_and_size[1]

    return result^


fn to_uppercase(s: StringSlice[mut=False]) -> String:
    """Returns a new string with all characters converted to uppercase.

    Args:
        s: Input string.

    Returns:
        A new string where cased letters have been converted to uppercase.
    """
    var data = s.as_bytes()
    var result = String(capacity=_estimate_needed_size(len(data)))
    var input_offset = 0
    while input_offset < len(data):
        var rune_and_size = Codepoint.unsafe_decode_utf8_codepoint(
            data[input_offset:]
        )
        var uppercase_replacement_opt = _get_uppercase_mapping(rune_and_size[0])

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
            result.write_bytes(
                data[input_offset : input_offset + rune_and_size[1]]
            )

        input_offset += rune_and_size[1]

    return result^


@always_inline
fn _estimate_needed_size(byte_len: Int) -> Int:
    return 3 * (byte_len >> 1) + 1
