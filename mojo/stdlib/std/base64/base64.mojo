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
"""Provides functions for base64 encoding strings.

You can import these APIs from the `base64` package. For example:

```mojo
from base64 import b64encode
```
"""


from memory import Span

from ._b64encode import _b64encode

# ===-----------------------------------------------------------------------===#
# Utilities
# ===-----------------------------------------------------------------------===#


@always_inline
fn _ascii_to_value[validate: Bool = False](char: Byte) raises -> Byte:
    """Converts an ASCII character to its integer value for base64 decoding.

    Args:
        char: A single ascii byte.

    Returns:
        The integer value of the character for base64 decoding, or -1 if
        invalid.
    """
    comptime `A` = Byte(ord("A"))
    comptime `a` = Byte(ord("a"))
    comptime `Z` = Byte(ord("Z"))
    comptime `z` = Byte(ord("z"))
    comptime `0` = Byte(ord("0"))
    comptime `9` = Byte(ord("9"))
    comptime `=` = Byte(ord("="))
    comptime `+` = Byte(ord("+"))
    comptime `/` = Byte(ord("/"))

    # TODO: Measure perf against lookup table approach
    if char == `=`:
        return Byte(0)
    elif `A` <= char <= `Z`:
        return char - `A`
    elif `a` <= char <= `z`:
        return char - `a` + Byte(26)
    elif `0` <= char <= `9`:
        return char - `0` + Byte(52)
    elif char == `+`:
        return Byte(62)
    elif char == `/`:
        return Byte(63)
    else:
        comptime if validate:
            raise Error(
                "ValueError: Unexpected character '",
                chr(Int(char)),
                "' encountered",
            )
        return Byte(-1)


# ===-----------------------------------------------------------------------===#
# b64encode
# ===-----------------------------------------------------------------------===#


@always_inline
fn b64encode(input_bytes: Span[mut=False, Byte], mut result: String):
    """Performs base64 encoding on the input string.

    Args:
        input_bytes: The input string buffer.
        result: The string in which to store the values.

    Notes:
        This method reserves the necessary capacity. `result` can be a 0
        capacity string.
    """
    _b64encode(input_bytes, result)


@always_inline
fn b64encode(input_string: StringSlice[mut=False]) -> String:
    """Performs base64 encoding on the input string.

    Args:
        input_string: The input string buffer.

    Returns:
        The ASCII base64 encoded string.
    """
    return b64encode(input_string.as_bytes())


@always_inline
fn b64encode(input_bytes: Span[mut=False, Byte]) -> String:
    """Performs base64 encoding on the input string.

    Args:
        input_bytes: The input string buffer.

    Returns:
        The ASCII base64 encoded string.
    """
    var result = String()
    b64encode(input_bytes, result)
    return result^


# ===-----------------------------------------------------------------------===#
# b64decode
# ===-----------------------------------------------------------------------===#


fn b64decode[
    *, validate: Bool = False
](str: StringSlice[mut=False]) raises -> String:
    """Performs base64 decoding on the input string.

    Parameters:
        validate: If true, the function will validate the input string.

    Args:
        str: A base64 encoded string.

    Returns:
        The decoded string.

    Raises:
        If the operation fails.
    """
    comptime `=` = Byte(ord("="))
    var data = str.as_bytes()
    var n = str.byte_length()

    comptime if validate:
        if n % 4 != 0:
            raise Error(
                "ValueError: Input length '", n, "' must be divisible by 4"
            )

    var result = String(capacity=n)

    # This algorithm is based on https://arxiv.org/abs/1704.00605
    for i in range(0, n, 4):
        var a = _ascii_to_value[validate](data[i])
        var b = _ascii_to_value[validate](data[i + 1])
        var c = _ascii_to_value[validate](data[i + 2])
        var d = _ascii_to_value[validate](data[i + 3])

        result._unsafe_append_byte((a << 2) | (b >> 4))
        if data[i + 2] == `=`:
            break
        result._unsafe_append_byte(((b & 0x0F) << 4) | (c >> 2))
        if data[i + 3] == `=`:
            break
        result._unsafe_append_byte(((c & 0x03) << 6) | d)

    return result^


# ===-----------------------------------------------------------------------===#
# b16encode
# ===-----------------------------------------------------------------------===#


fn b16encode(str: StringSlice[mut=False]) -> String:
    """Performs base16 encoding on the input string slice.

    Args:
        str: The input string slice.

    Returns:
        Base16 encoding of the input string.
    """
    comptime lookup = "0123456789ABCDEF"
    var b16chars = lookup.unsafe_ptr()

    var data = str.as_bytes()
    var length = str.byte_length()
    var result = String(capacity=length * 2)

    for i in range(length):
        var str_byte = data[i]
        var hi = str_byte >> 4
        var lo = str_byte & 0b1111
        result._unsafe_append_byte(b16chars[hi])
        result._unsafe_append_byte(b16chars[lo])

    return result^


# ===-----------------------------------------------------------------------===#
# b16decode
# ===-----------------------------------------------------------------------===#


fn b16decode(str: StringSlice[mut=False]) -> String:
    """Performs base16 decoding on the input string.

    Args:
        str: A base16 encoded string.

    Returns:
        The decoded string.
    """

    comptime `A` = Byte(ord("A"))
    comptime `a` = Byte(ord("a"))
    comptime `Z` = Byte(ord("Z"))
    comptime `z` = Byte(ord("z"))
    comptime `0` = Byte(ord("0"))
    comptime `9` = Byte(ord("9"))

    # TODO: Measure perf against lookup table approach
    @parameter
    @always_inline
    fn decode(c: Byte) -> Byte:
        if `A` <= c <= `Z`:
            return c - `A` + Byte(10)
        elif `a` <= c <= `z`:
            return c - `a` + Byte(10)
        elif `0` <= c <= `9`:
            return c - `0`
        else:
            return Byte(-1)

    var data = str.as_bytes()
    var n = str.byte_length()
    debug_assert(n % 2 == 0, "Input length '", n, "' must be divisible by 2")

    var result = String(capacity=n // 2)

    for i in range(0, n, 2):
        var hi = data[i]
        var lo = data[i + 1]
        result._unsafe_append_byte(decode(hi) << 4 | decode(lo))

    return result^
