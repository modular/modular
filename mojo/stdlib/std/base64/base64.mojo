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
from std.base64 import b64encode
```
"""


from std.memory import Span
from std.math import ceildiv

from ._b64encode import _b64encode

# ===-----------------------------------------------------------------------===#
# Utilities
# ===-----------------------------------------------------------------------===#


@always_inline
def _ascii_to_value[validate: Bool = False](char: Byte) raises -> Byte:
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
def b64encode(input_bytes: Span[mut=False, Byte, _]) -> String:
    """Performs base64 encoding on input bytes, returning a String (fastest implementation).

    This uses the optimized SIMD-based encoding from _b64encode.

    Args:
        input_bytes: The input bytes to encode.

    Returns:
        The ASCII base64 encoded string.
    """
    var result = String(capacity=4 * ceildiv(len(input_bytes), 3))

    @parameter
    def append_byte(b: UInt8):
        result._unsafe_append_byte(b)

    _b64encode[append_byte](input_bytes)
    return result^


# ===-----------------------------------------------------------------------===#
# b64encode - Layer 2: Convenience overloads
# ===-----------------------------------------------------------------------===#


@always_inline
def b64encode(input_string: StringSlice[mut=False, _]) -> String:
    """Performs base64 encoding on a string, returning a String.

    Args:
        input_string: The input string to encode.

    Returns:
        The ASCII base64 encoded string.
    """
    return b64encode(input_string.as_bytes())


@always_inline
def b64encode(input_bytes: Span[mut=False, Byte, _], mut result: String):
    """Performs base64 encoding on input bytes, writing to a String.

    Args:
        input_bytes: The input bytes to encode.
        result: The string in which to store the values.

    Notes:
        This method reserves the necessary capacity. `result` can be a 0
        capacity string.
    """

    @parameter
    def append_byte(b: UInt8):
        result._unsafe_append_byte(b)

    _b64encode[append_byte](input_bytes)


# ===-----------------------------------------------------------------------===#
# b64decode - Core implementation (no allocations)
# ===-----------------------------------------------------------------------===#


def b64decode[
    *, validate: Bool = False, origin_in: Origin, origin_out: Origin
](
    input_bytes: Span[mut=False, Byte, origin_in],
    mut output_bytes: Span[mut=True, Byte, _],
) raises:
    """Performs base64 decoding from input bytes to output bytes (zero-copy layer).

    This is the core decoding function that performs no allocations.
    The output buffer must be at least (input length * 3) // 4 bytes.

    Parameters:
        validate: If true, the function will validate the input.
        origin_in: Origin of the input span.
        origin_out: Origin of the output span.

    Args:
        input_bytes: The input base64 encoded bytes.
        output_bytes: The output buffer to write decoded bytes into.
    """
    comptime `=` = Byte(ord("="))
    var n = len(input_bytes)
    var write_pos = 0

    comptime if validate:
        if n % 4 != 0:
            raise Error(
                "ValueError: Input length '", n, "' must be divisible by 4"
            )

    # This algorithm is based on https://arxiv.org/abs/1704.00605
    for i in range(0, n, 4):
        var a = _ascii_to_value[validate](input_bytes[i])
        var b = _ascii_to_value[validate](input_bytes[i + 1])
        var c = _ascii_to_value[validate](input_bytes[i + 2])
        var d = _ascii_to_value[validate](input_bytes[i + 3])

        output_bytes[write_pos] = (a << 2) | (b >> 4)
        write_pos += 1

        if input_bytes[i + 2] == `=`:
            break

        output_bytes[write_pos] = ((b & 0x0F) << 4) | (c >> 2)
        write_pos += 1

        if input_bytes[i + 3] == `=`:
            break

        output_bytes[write_pos] = ((c & 0x03) << 6) | d
        write_pos += 1


# ===-----------------------------------------------------------------------===#
# b64decode - Layer 1: String-based convenience API
# ===-----------------------------------------------------------------------===#


def b64decode[
    *, validate: Bool = False
](str: StringSlice[mut=False, _]) raises -> List[UInt8]:
    """Performs base64 decoding on a string, returning decoded bytes.

    Parameters:
        validate: If true, the function will validate the input string.

    Args:
        str: A base64 encoded string.

    Returns:
        The decoded bytes as a List[UInt8].
    """
    var data = str.as_bytes()
    var n = str.byte_length()
    # Allocate buffer with maximum possible size
    var result = List[UInt8](capacity=n * 3 // 4)
    result.resize(n * 3 // 4, 0)

    var write_pos = 0
    comptime `=` = Byte(ord("="))

    comptime if validate:
        if n % 4 != 0:
            raise Error(
                "ValueError: Input length '", n, "' must be divisible by 4"
            )

    # This algorithm is based on https://arxiv.org/abs/1704.00605
    for i in range(0, n, 4):
        var a = _ascii_to_value[validate](data[i])
        var b = _ascii_to_value[validate](data[i + 1])
        var c = _ascii_to_value[validate](data[i + 2])
        var d = _ascii_to_value[validate](data[i + 3])

        result[write_pos] = (a << 2) | (b >> 4)
        write_pos += 1

        if data[i + 2] == `=`:
            break

        result[write_pos] = ((b & 0x0F) << 4) | (c >> 2)
        write_pos += 1

        if data[i + 3] == `=`:
            break

        result[write_pos] = ((c & 0x03) << 6) | d
        write_pos += 1

    result.resize(write_pos, 0)
    return result^


# ===-----------------------------------------------------------------------===#
# b16encode - Layer 1: Span -> String (zero-copy)
# ===-----------------------------------------------------------------------===#


def b16encode[origin: Origin](input_bytes: Span[UInt8, origin]) -> String:
    """Performs base16 encoding on input bytes, returning a String.

    Args:
        input_bytes: The input bytes to encode.

    Returns:
        Base16 encoding of the input bytes.
    """
    comptime lookup = "0123456789ABCDEF"
    var b16chars = lookup.unsafe_ptr()

    var length = len(input_bytes)
    var result = String(capacity=length * 2)

    for i in range(length):
        var byte = input_bytes[i]
        var hi = byte >> 4
        var lo = byte & 0b1111
        result._unsafe_append_byte(b16chars[hi])
        result._unsafe_append_byte(b16chars[lo])

    return result^


# ===-----------------------------------------------------------------------===#
# b16encode - Layer 2: Convenience overload
# ===-----------------------------------------------------------------------===#


@always_inline
def b16encode(s: StringSlice) -> String:
    """Performs base16 encoding on a string, returning a String.

    Args:
        s: The input string.

    Returns:
        Base16 encoding of the input string.
    """
    return b16encode(s.as_bytes())


# ===-----------------------------------------------------------------------===#
# b16decode - Layer 1: Span -> Span (zero-copy)
# ===-----------------------------------------------------------------------===#


def b16decode[
    origin_in: Origin, origin_out: Origin
](
    input_bytes: Span[mut=False, UInt8, origin_in],
    mut output_bytes: Span[mut=True, UInt8, _],
):
    """Performs base16 decoding from input bytes to output bytes (zero-copy layer).

    This is the core decoding function that performs no allocations.
    The output buffer must be exactly input length / 2 bytes.

    Parameters:
        origin_in: Origin of the input span.
        origin_out: Origin of the output span.

    Args:
        input_bytes: The input base16 encoded bytes.
        output_bytes: The output buffer to write decoded bytes into.
    """
    comptime `A` = UInt8(ord("A"))
    comptime `a` = UInt8(ord("a"))
    comptime `Z` = UInt8(ord("Z"))
    comptime `z` = UInt8(ord("z"))
    comptime `0` = UInt8(ord("0"))
    comptime `9` = UInt8(ord("9"))

    @parameter
    @always_inline
    def decode(c: UInt8) -> UInt8:
        if `A` <= c <= `Z`:
            return c - `A` + UInt8(10)
        elif `a` <= c <= `z`:
            return c - `a` + UInt8(10)
        elif `0` <= c <= `9`:
            return c - `0`
        else:
            return UInt8(-1)

    var n = len(input_bytes)
    debug_assert(n % 2 == 0, "Input length '", n, "' must be divisible by 2")

    for i in range(0, n, 2):
        var hi = input_bytes[i]
        var lo = input_bytes[i + 1]
        output_bytes[i // 2] = decode(hi) << 4 | decode(lo)


# ===-----------------------------------------------------------------------===#
# b16decode - Layer 2: String-based convenience API
# ===-----------------------------------------------------------------------===#


def b16decode(str: StringSlice[mut=False, _]) -> List[UInt8]:
    """Performs base16 decoding on a string, returning decoded bytes.

    Args:
        str: A base16 encoded string.

    Returns:
        The decoded bytes as a List[UInt8].
    """
    var data = str.as_bytes()
    var n = str.byte_length()
    var output_len = n // 2
    var result = List[UInt8](capacity=output_len)
    result.resize(output_len, 0)

    comptime `A` = UInt8(ord("A"))
    comptime `a` = UInt8(ord("a"))
    comptime `Z` = UInt8(ord("Z"))
    comptime `z` = UInt8(ord("z"))
    comptime `0` = UInt8(ord("0"))
    comptime `9` = UInt8(ord("9"))

    @parameter
    @always_inline
    def decode(c: UInt8) -> UInt8:
        if `A` <= c <= `Z`:
            return c - `A` + UInt8(10)
        elif `a` <= c <= `z`:
            return c - `a` + UInt8(10)
        elif `0` <= c <= `9`:
            return c - `0`
        else:
            return UInt8(-1)

    debug_assert(n % 2 == 0, "Input length '", n, "' must be divisible by 2")

    for i in range(0, n, 2):
        var hi = data[i]
        var lo = data[i + 1]
        result[i // 2] = decode(hi) << 4 | decode(lo)

    return result^


@always_inline
def b16decode(str: StringSlice[mut=False, _], mut result: List[UInt8]):
    """Performs base16 decoding on a string, writing to a List.

    Args:
        str: A base16 encoded string.
        result: The List[UInt8] to write decoded bytes into.
    """
    var data = str.as_bytes()
    var n = str.byte_length()
    var output_len = n // 2

    comptime `A` = UInt8(ord("A"))
    comptime `a` = UInt8(ord("a"))
    comptime `Z` = UInt8(ord("Z"))
    comptime `z` = UInt8(ord("z"))
    comptime `0` = UInt8(ord("0"))
    comptime `9` = UInt8(ord("9"))

    @parameter
    @always_inline
    def decode(c: UInt8) -> UInt8:
        if `A` <= c <= `Z`:
            return c - `A` + UInt8(10)
        elif `a` <= c <= `z`:
            return c - `a` + UInt8(10)
        elif `0` <= c <= `9`:
            return c - `0`
        else:
            return UInt8(-1)

    debug_assert(n % 2 == 0, "Input length '", n, "' must be divisible by 2")

    result.resize(output_len, 0)
    for i in range(0, n, 2):
        var hi = data[i]
        var lo = data[i + 1]
        result[i // 2] = decode(hi) << 4 | decode(lo)
