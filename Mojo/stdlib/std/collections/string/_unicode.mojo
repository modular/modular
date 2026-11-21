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

from std.collections.string._unicode_lookups import (
    has_uppercase_mapping,
    has_lowercase_mapping,
    uppercase_mapping,
    lowercase_mapping,
    has_uppercase_mapping2,
    uppercase_mapping2,
    has_uppercase_mapping3,
    uppercase_mapping3,
    has_lowercase_mapping2,
    lowercase_mapping2,
)

from std.builtin.globals import global_constant
from std.collections import Span


def _uppercase_mapping_index(rune: Codepoint) -> Int:
    """Return index for upper case mapping or -1 if no mapping is given."""
    return _to_index(Span(global_constant[has_uppercase_mapping]()), rune)


def _uppercase_mapping2_index(rune: Codepoint) -> Int:
    """Return index for upper case mapping converting the rune to 2 runes, or -1 if no mapping is given.
    """
    return _to_index(Span(global_constant[has_uppercase_mapping2]()), rune)


def _uppercase_mapping3_index(rune: Codepoint) -> Int:
    """Return index for upper case mapping converting the rune to 3 runes, or -1 if no mapping is given.
    """
    return _to_index(Span(global_constant[has_uppercase_mapping3]()), rune)


def _lowercase_mapping_index(rune: Codepoint) -> Int:
    """Return index for lower case mapping or -1 if no mapping is given."""
    return _to_index(Span(global_constant[has_lowercase_mapping]()), rune)


def _lowercase_mapping2_index(rune: Codepoint) -> Int:
    """Return index for lower case mapping converting the rune to 2 runes, or -1 if no mapping is given.
    """
    return _to_index(Span(global_constant[has_lowercase_mapping2]()), rune)


def _to_index(lookup: Span[UInt32, ImmStaticOrigin], rune: Codepoint) -> Int:
    """Find index of rune in lookup with binary search.
    Returns -1 if not found."""

    var result = lookup._binary_search_index(rune.to_u32())

    if case .Some(result) = result:
        return Int(result)
    else:
        return -1


__union _CaseMapping(Copyable, Equatable, Writable):
    """The full case mapping of a codepoint, which is 1 to 3 codepoints."""

    case single(Array[Codepoint, 1])
    case pair(Array[Codepoint, 2])
    case triple(Array[Codepoint, 3])

    def codepoints(self) -> Span[Codepoint, origin_of(self)]:
        match self:
            case .single(array):
                return array
            case .pair(array):
                return array
            case .triple(array):
                return array


def _get_uppercase_mapping(char: Codepoint) -> Optional[_CaseMapping]:
    """Returns the uppercase form of `char`, or None if `char` does not have an
    uppercase equivalent.
    """
    var index1 = _uppercase_mapping_index(char)
    if index1 != -1:
        var rune = global_constant[uppercase_mapping]()[index1]
        return _CaseMapping.single([Codepoint(unsafe_unchecked_codepoint=rune)])

    var index2 = _uppercase_mapping2_index(char)
    if index2 != -1:
        var runes = global_constant[uppercase_mapping2]()[index2]
        return _CaseMapping.pair(
            [
                Codepoint(unsafe_unchecked_codepoint=runes[0]),
                Codepoint(unsafe_unchecked_codepoint=runes[1]),
            ]
        )

    var index3 = _uppercase_mapping3_index(char)
    if index3 != -1:
        var runes = global_constant[uppercase_mapping3]()[index3]
        return _CaseMapping.triple(
            [
                Codepoint(unsafe_unchecked_codepoint=runes[0]),
                Codepoint(unsafe_unchecked_codepoint=runes[1]),
                Codepoint(unsafe_unchecked_codepoint=runes[2]),
            ]
        )

    return None


def _get_lowercase_mapping(char: Codepoint) -> Optional[_CaseMapping]:
    """Returns the lowercase form of `char`, or None if `char` does not have a
    lowercase equivalent.
    """
    var index1 = _lowercase_mapping_index(char)
    if index1 != -1:
        var rune = global_constant[lowercase_mapping]()[index1]
        return _CaseMapping.single([Codepoint(unsafe_unchecked_codepoint=rune)])

    var index2 = _lowercase_mapping2_index(char)
    if index2 != -1:
        var runes = global_constant[lowercase_mapping2]()[index2]
        return _CaseMapping.pair(
            [
                Codepoint(unsafe_unchecked_codepoint=runes[0]),
                Codepoint(unsafe_unchecked_codepoint=runes[1]),
            ]
        )

    return None


def _changes_when_uppercased(char: Codepoint) -> Bool:
    return (
        _uppercase_mapping_index(char) != -1
        or _uppercase_mapping2_index(char) != -1
        or _uppercase_mapping3_index(char) != -1
    )


def _changes_when_lowercased(char: Codepoint) -> Bool:
    return (
        _lowercase_mapping_index(char) != -1
        or _lowercase_mapping2_index(char) != -1
    )


def is_uppercase(s: StringSpan[_]) -> Bool:
    """Returns True if all characters in the string are uppercase, and
        there is at least one cased character.

    Args:
        s: The string to examine.

    Returns:
        True if all characters in the string are uppercase and
        there is at least one cased character, False otherwise.
    """
    var found = False
    for char in s.codepoints():
        # A rune that changes when uppercased is lowercase or titlecase.
        if _changes_when_uppercased(char):
            return False
        if _changes_when_lowercased(char):
            found = True
    return found


def is_lowercase(s: StringSpan[_]) -> Bool:
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
        # A rune that changes when lowercased is uppercase or titlecase.
        if _changes_when_lowercased(char):
            return False
        if _changes_when_uppercased(char):
            found = True
    return found


def to_lowercase(s: StringSpan[_]) -> String:
    """Returns a new string with all characters converted to lowercase.

    Args:
        s: Input string.

    Returns:
        A new string where cased letters have been converted to lowercase.
    """
    var input = s.as_bytes()
    var result = String(capacity_bytes=_estimate_needed_size(len(input)))
    var input_offset = 0
    while input_offset < len(input):
        var char, num_bytes = Codepoint.unsafe_decode_utf8_codepoint(
            input[input_offset:]
        )
        var mapping = _get_lowercase_mapping(char).or_else(.single([char]))
        for c in mapping.codepoints():
            result.append(c)
        input_offset += num_bytes

    return result^


def to_uppercase(s: StringSpan[_]) -> String:
    """Returns a new string with all characters converted to uppercase.

    Args:
        s: Input string.

    Returns:
        A new string where cased letters have been converted to uppercase.
    """
    var input = s.as_bytes()
    var result = String(capacity_bytes=_estimate_needed_size(len(input)))
    var input_offset = 0
    while input_offset < len(input):
        var char, num_bytes = Codepoint.unsafe_decode_utf8_codepoint(
            input[input_offset:]
        )
        var mapping = _get_uppercase_mapping(char).or_else(.single([char]))
        for c in mapping.codepoints():
            result.append(c)
        input_offset += num_bytes

    return result^


@inline(.always)
def _estimate_needed_size(byte_len: Int) -> Int:
    return 3 * (byte_len >> 1) + 1
