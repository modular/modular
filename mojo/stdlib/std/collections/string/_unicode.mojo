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

from std.builtin.globals import global_constant
from std.collections.string._unicode_lookups import (
    has_uppercase_mapping,
    has_lowercase_mapping,
    uppercase_mapping,
    lowercase_mapping,
    has_uppercase_mapping2,
    uppercase_mapping2,
    has_uppercase_mapping3,
    uppercase_mapping3,
)


@always_inline
fn _maybe_get_index[
    lookup: InlineArray[UInt32, ...]
](rune: Codepoint) -> Optional[UInt]:
    if __is_run_in_comptime_interpreter:
        return Span(materialize[lookup]())._binary_search_index(rune.to_u32())
    return Span(global_constant[lookup]())._binary_search_index(rune.to_u32())


@always_inline
fn _get_mapping[
    inner_size: Int,
    outer_size: Int,
    //,
    mapping: InlineArray[SIMD[DType.uint32, inner_size], outer_size],
](index: UInt) -> SIMD[DType.uint32, inner_size]:
    if __is_run_in_comptime_interpreter:
        return materialize[mapping]().unsafe_get(index)
    return global_constant[mapping]().unsafe_get(index)


def _get_uppercase_mapping(
    char: Codepoint,
) -> Optional[Tuple[UInt, InlineArray[Codepoint, 3]]]:
    """Returns the 1, 2, or 3 character sequence that is the uppercase form of
    `char`.

    Returns None if `char` does not have an uppercase equivalent.
    """
    var array = InlineArray[Codepoint, 3](fill=Codepoint(0))

    if index := _maybe_get_index[has_uppercase_mapping](char):
        var rune = _get_mapping[uppercase_mapping](index.unsafe_value())
        array[0] = Codepoint(unsafe_unchecked_codepoint=rune)
        return Tuple(UInt(1), array^)
    elif index := _maybe_get_index[has_uppercase_mapping2](char):
        var runes = _get_mapping[uppercase_mapping2](index.unsafe_value())
        array[0] = Codepoint(unsafe_unchecked_codepoint=runes[0])
        array[1] = Codepoint(unsafe_unchecked_codepoint=runes[1])
        return Tuple(UInt(2), array^)
    elif index := _maybe_get_index[has_uppercase_mapping3](char):
        var runes = _get_mapping[uppercase_mapping3](index.unsafe_value())
        array[0] = Codepoint(unsafe_unchecked_codepoint=runes[0])
        array[1] = Codepoint(unsafe_unchecked_codepoint=runes[1])
        array[2] = Codepoint(unsafe_unchecked_codepoint=runes[2])
        return Tuple(UInt(3), array^)
    else:
        return None


fn _get_lowercase_mapping(char: Codepoint) -> Optional[Codepoint]:
    var index = _maybe_get_index[has_lowercase_mapping](char)
    if not index:
        return None
    # SAFETY:
    #   We know this is a valid `Codepoint` because the mapping data tables
    #   contain only valid codepoints.
    var value = _get_mapping[lowercase_mapping](index.unsafe_value())
    return Codepoint(unsafe_unchecked_codepoint=value)


def is_uppercase(s: StringSlice[mut=False, _]) -> Bool:
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
        if _maybe_get_index[has_lowercase_mapping](char):
            found = True
            continue
        elif _maybe_get_index[has_uppercase_mapping](char):
            return False
        elif _maybe_get_index[has_uppercase_mapping2](char):
            return False
        elif _maybe_get_index[has_uppercase_mapping3](char):
            return False
    return found


def is_lowercase(s: StringSlice[mut=False, _]) -> Bool:
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
        if _maybe_get_index[has_uppercase_mapping](char):
            found = True
            continue
        elif _maybe_get_index[has_uppercase_mapping2](char):
            found = True
            continue
        elif _maybe_get_index[has_uppercase_mapping3](char):
            found = True
            continue
        elif _maybe_get_index[has_lowercase_mapping](char):
            return False
    return found


def to_lowercase(s: StringSlice[mut=False, _]) -> String:
    """Returns a new string with all characters converted to uppercase.

    Args:
        s: Input string.

    Returns:
        A new string where cased letters have been converted to lowercase.
    """
    # lowercased strings always have the same amount of bytes
    var result = String(capacity=s.byte_length())
    var input_offset = 0
    while input_offset < s.byte_length():
        ref rune, size = Codepoint.unsafe_decode_utf8_codepoint(
            s.as_bytes()[input_offset:]
        )
        var maybe_replace = _get_lowercase_mapping(rune)
        if maybe_replace:
            result += String(maybe_replace.unsafe_value())
        else:
            result.write_string(s[byte = input_offset : input_offset + size])

        input_offset += size

    return result^


def to_uppercase(s: StringSlice[mut=False, _]) -> String:
    """Returns a new string with all characters converted to uppercase.

    Args:
        s: Input string.

    Returns:
        A new string where cased letters have been converted to uppercase.
    """
    # estimate the size since some codepoints require multiple chars to uppercase
    var result = String(capacity=3 * (s.byte_length() // 2))
    var input_offset = 0
    while input_offset < s.byte_length():
        ref rune, size = Codepoint.unsafe_decode_utf8_codepoint(
            s.as_bytes()[input_offset:]
        )
        var maybe_replace = _get_uppercase_mapping(rune)

        if maybe_replace:
            # A given character can be replaced with a sequence of characters
            # up to 3 characters in length. A fixed size `Codepoint` array is
            # returned, along with a `count` (1, 2, or 3) of how many
            # replacement characters are in the uppercase replacement sequence.
            ref count, uppercase_replacement_chars = (
                maybe_replace.unsafe_value()
            )
            for char_idx in range(count):
                result += String(uppercase_replacement_chars[char_idx])
        else:
            result.write_string(s[byte = input_offset : input_offset + size])

        input_offset += size

    return result^
