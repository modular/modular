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

from memory import UnsafePointer, memcpy
from collections.string._unicode_lookups import *


fn _uppercase_mapping_index(rune: Int) -> Int:
    """Return index for upper case mapping or -1 if no mapping is given."""
    return _to_index[has_uppercase_mapping](rune)


fn _uppercase_mapping2_index(rune: Int) -> Int:
    """Return index for upper case mapping converting the rune to 2 runes, or -1 if no mapping is given.
    """
    return _to_index[has_uppercase_mapping2](rune)


fn _uppercase_mapping3_index(rune: Int) -> Int:
    """Return index for upper case mapping converting the rune to 3 runes, or -1 if no mapping is given.
    """
    return _to_index[has_uppercase_mapping3](rune)


fn _lowercase_mapping_index(rune: Int) -> Int:
    """Return index for lower case mapping or -1 if no mapping is given."""
    return _to_index[has_lowercase_mapping](rune)


@always_inline
fn _to_index[lookup: List[UInt32, **_]](rune: Int) -> Int:
    """Find index of rune in lookup with binary search.
    Returns -1 if not found."""

    var result = lookup._binary_search_index(UInt32(rune))

    if result:
        return result.unsafe_value()
    else:
        return -1


fn is_uppercase(s: StringSlice) -> Bool:
    """Returns True if all characters in the string are uppercase, and
        there is at least one cased character.

    Args:
        s: The string to examine.

    Returns:
        True if all characters in the string are uppercaseand
        there is at least one cased character, False otherwise.
    """
    var found = False
    for c in s:
        var rune = ord(c)
        var index = _lowercase_mapping_index(rune)
        if index != -1:
            found = True
            continue
        index = _uppercase_mapping_index(rune)
        if index != -1:
            return False
        index = _uppercase_mapping2_index(rune)
        if index != -1:
            return False
        index = _uppercase_mapping3_index(rune)
        if index != -1:
            return False
    return found


fn is_lowercase(s: StringSlice) -> Bool:
    """Returns True if all characters in the string are lowercase, and
        there is at least one cased character.

    Args:
        s: The string to examine.

    Returns:
        True if all characters in the string are lowercase and
        there is at least one cased character, False otherwise.
    """
    var found = False
    for c in s:
        var rune = ord(c)
        var index = _uppercase_mapping_index(rune)
        if index != -1:
            found = True
            continue
        index = _uppercase_mapping2_index(rune)
        if index != -1:
            found = True
            continue
        index = _uppercase_mapping3_index(rune)
        if index != -1:
            found = True
            continue
        index = _lowercase_mapping_index(rune)
        if index != -1:
            return False
    return found


fn _write_rune(rune: UInt32, p: UnsafePointer[UInt8]) -> Int:
    """Write rune as UTF-8 into provided pointer. Return number of added bytes.
    """
    return Char(unsafe_unchecked_codepoint=rune).unsafe_write_utf8(p)


fn to_lowercase(s: StringSlice) -> String:
    """Returns a new string with all characters converted to uppercase.

    Args:
        s: Input string.

    Returns:
        A new string where cased letters have been converted to lowercase.
    """
    var input = s.unsafe_ptr()
    var capacity = (s.byte_length() >> 1) * 3 + 1
    var output = UnsafePointer[UInt8].alloc(capacity)
    var input_offset = 0
    var output_offset = 0
    while input_offset < s.byte_length():
        var rune_and_size = Char.unsafe_decode_utf8_char(input + input_offset)
        var index = _lowercase_mapping_index(Int(rune_and_size[0]))
        if index == -1:
            memcpy(
                output + output_offset, input + input_offset, rune_and_size[1]
            )
            output_offset += rune_and_size[1]
        else:
            output_offset += _write_rune(
                lowercase_mapping[index], output + output_offset
            )

        input_offset += rune_and_size[1]

        if output_offset >= (
            capacity - 5
        ):  # check if we need to resize the ouput
            capacity += ((s.byte_length() - input_offset) >> 1) * 3 + 1
            var new_output = UnsafePointer[UInt8].alloc(capacity)
            memcpy(new_output, output, output_offset)
            output.free()
            output = new_output

    output[output_offset] = 0
    var list = List[UInt8](
        ptr=output, length=(output_offset + 1), capacity=capacity
    )
    return String(list)


fn to_uppercase(s: StringSlice) -> String:
    """Returns a new string with all characters converted to uppercase.

    Args:
        s: Input string.

    Returns:
        A new string where cased letters have been converted to uppercase.
    """
    var input = s.unsafe_ptr()
    var capacity = (s.byte_length() >> 1) * 3 + 1
    var output = UnsafePointer[UInt8].alloc(capacity)
    var input_offset = 0
    var output_offset = 0
    while input_offset < s.byte_length():
        var rune_and_size = Char.unsafe_decode_utf8_char(input + input_offset)
        var index = _uppercase_mapping_index(Int(rune_and_size[0]))
        var index2 = _uppercase_mapping2_index(
            Int(rune_and_size[0])
        ) if index == -1 else -1
        var index3 = _uppercase_mapping3_index(
            Int(rune_and_size[0])
        ) if index == -1 and index2 == -1 else -1
        if index != -1:
            output_offset += _write_rune(
                uppercase_mapping[index], output + output_offset
            )
        elif index2 != -1:
            var runes = uppercase_mapping2[index2]
            output_offset += _write_rune(runes[0], output + output_offset)
            output_offset += _write_rune(runes[1], output + output_offset)
        elif index3 != -1:
            var runes = uppercase_mapping3[index3]
            output_offset += _write_rune(runes[0], output + output_offset)
            output_offset += _write_rune(runes[1], output + output_offset)
            output_offset += _write_rune(runes[2], output + output_offset)
        else:
            memcpy(
                output + output_offset, input + input_offset, rune_and_size[1]
            )
            output_offset += rune_and_size[1]

        input_offset += rune_and_size[1]

        if output_offset >= (
            capacity - 5
        ):  # check if we need to resize the ouput
            capacity += ((s.byte_length() - input_offset) >> 1) * 3 + 1
            var new_output = UnsafePointer[UInt8].alloc(capacity)
            memcpy(new_output, output, output_offset)
            output.free()
            output = new_output

    output[output_offset] = 0
    var list = List[UInt8](
        ptr=output, length=(output_offset + 1), capacity=capacity
    )
    return String(list)
