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


from std.math import ceildiv
from std.memory import Span

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
def unsafe_b64encode(
    input_bytes: Span[mut=False, Byte, _],
    output_bytes: Span[mut=True, Byte, _],
) -> Int:
    """Performs base64 encoding from input bytes to output bytes.

    The caller must ensure the output buffer is at least
    `4 * ceildiv(len(input_bytes), 3)` bytes; no bounds check is performed.

    Args:
        input_bytes: The input bytes to encode.
        output_bytes: The output buffer to write encoded bytes into.

    Returns:
        The number of bytes written to output_bytes.
    """
    return _b64encode(input_bytes, output_bytes)


@always_inline
def b64encode(input_bytes: Span[mut=False, Byte, _], mut result: String):
    """Performs base64 encoding, writing into a reusable `String` buffer.

    `result` is resized to fit the encoded output; existing contents are
    overwritten. Capacity is reused when possible, so calling this in a
    loop with the same `result` avoids per-call allocation.

    Args:
        input_bytes: The input bytes to encode.
        result: The string buffer to write into.
    """
    var output_len = 4 * ceildiv(len(input_bytes), 3)
    result.resize(unsafe_uninit_length=output_len)
    _ = unsafe_b64encode(input_bytes, result.unsafe_as_bytes_mut())


@always_inline
def b64encode(input_string: StringSlice[mut=False, _]) -> String:
    """Performs base64 encoding on the input string.

    Args:
        input_string: The input string buffer.

    Returns:
        The ASCII base64 encoded string.
    """
    return b64encode(input_string.as_bytes())


@always_inline
def b64encode(input_bytes: Span[mut=False, Byte, _]) -> String:
    """Performs base64 encoding on the input bytes.

    Args:
        input_bytes: The input bytes to encode.

    Returns:
        The ASCII base64 encoded string.
    """
    var result = String()
    b64encode(input_bytes, result)
    return result^


# ===-----------------------------------------------------------------------===#
# b64decode
# ===-----------------------------------------------------------------------===#


def unsafe_b64decode[
    *, validate: Bool = False
](
    str: StringSlice[mut=False, _],
    mut output_bytes: Span[mut=True, Byte, _],
) raises -> Int:
    """Performs base64 decoding from a string to output bytes.

    The caller must ensure the output buffer is at least
    `(len(str) * 3) // 4` bytes; no bounds check is performed.

    Parameters:
        validate: If true, the function will validate the input and that the
            output buffer is large enough.

    Args:
        str: A base64 encoded string.
        output_bytes: The output buffer to write decoded bytes into.

    Returns:
        The number of bytes written to output_bytes.

    Raises:
        If `validate` is true and the input length is not a multiple of 4,
        the output buffer is too small, or the input contains non-base64
        characters.
    """
    comptime `=` = Byte(ord("="))
    var input_bytes = str.as_bytes()
    var n = str.byte_length()
    var write_pos = 0

    debug_assert(
        len(output_bytes) >= n * 3 // 4,
        "output buffer too small for base64 decoding: need ",
        n * 3 // 4,
        ", got ",
        len(output_bytes),
    )

    comptime if validate:
        if n % 4 != 0:
            raise Error(
                "ValueError: Input length '", n, "' must be divisible by 4"
            )
        if len(output_bytes) < n * 3 // 4:
            raise Error(
                "ValueError: Output buffer length '",
                len(output_bytes),
                "' is too small for input length '",
                n,
                "'",
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

    return write_pos


@always_inline
def b64decode[
    *, validate: Bool = False
](str: StringSlice[mut=False, _], mut result: List[Byte]) raises:
    """Performs base64 decoding into a reusable `List[Byte]` buffer.

    `result` is resized to fit the decoded output; existing contents are
    overwritten. Capacity is reused when possible, so calling this in a
    loop with the same `result` avoids per-call allocation.

    Parameters:
        validate: If true, the function will validate the input string.

    Args:
        str: A base64 encoded string.
        result: The byte buffer to write into.

    Raises:
        If `validate` is true and the input is not valid base64.
    """
    var output_len = str.byte_length() * 3 // 4
    result.resize(unsafe_uninit_length=output_len)
    var output_span = Span(result)
    var written = unsafe_b64decode[validate=validate](str, output_span)
    result.resize(unsafe_uninit_length=written)


@always_inline
def b64decode[
    *, validate: Bool = False
](str: StringSlice[mut=False, _]) raises -> List[Byte]:
    """Performs base64 decoding on a string, returning decoded bytes.

    Parameters:
        validate: If true, the function will validate the input string.

    Args:
        str: A base64 encoded string.

    Returns:
        The decoded bytes as a List[Byte].

    Raises:
        If `validate` is true and the input is not valid base64.
    """
    var result = List[Byte]()
    b64decode[validate=validate](str, result)
    return result^


# ===-----------------------------------------------------------------------===#
# b16encode
# ===-----------------------------------------------------------------------===#


def unsafe_b16encode(
    input_bytes: Span[mut=False, Byte, _],
    output_bytes: Span[mut=True, Byte, _],
) -> Int:
    """Performs base16 encoding from input bytes to output bytes.

    The caller must ensure the output buffer is at least
    `2 * len(input_bytes)` bytes; no bounds check is performed.

    Args:
        input_bytes: The input bytes to encode.
        output_bytes: The output buffer to write encoded bytes into.

    Returns:
        The number of bytes written to output_bytes.
    """
    var length = len(input_bytes)
    debug_assert(
        len(output_bytes) >= 2 * length,
        "output buffer too small for base16 encoding: need ",
        2 * length,
        ", got ",
        len(output_bytes),
    )
    comptime lookup = "0123456789ABCDEF"
    var b16chars = lookup.unsafe_ptr()
    for i in range(length):
        var byte = input_bytes[i]
        output_bytes[2 * i] = b16chars[byte >> 4]
        output_bytes[2 * i + 1] = b16chars[byte & 0b1111]
    return length * 2


@always_inline
def b16encode(input_bytes: Span[mut=False, Byte, _], mut result: String):
    """Performs base16 encoding, writing into a reusable `String` buffer.

    `result` is resized to fit the encoded output; existing contents are
    overwritten. Capacity is reused when possible, so calling this in a
    loop with the same `result` avoids per-call allocation.

    Args:
        input_bytes: The input bytes to encode.
        result: The string buffer to write into.
    """
    var output_len = 2 * len(input_bytes)
    result.resize(unsafe_uninit_length=output_len)
    _ = unsafe_b16encode(input_bytes, result.unsafe_as_bytes_mut())


@always_inline
def b16encode(str: StringSlice[mut=False, _]) -> String:
    """Performs base16 encoding on the input string slice.

    Args:
        str: The input string slice.

    Returns:
        Base16 encoding of the input string.
    """
    return b16encode(str.as_bytes())


@always_inline
def b16encode(input_bytes: Span[mut=False, Byte, _]) -> String:
    """Performs base16 encoding on the input bytes.

    Args:
        input_bytes: The input bytes to encode.

    Returns:
        Base16 encoding of the input bytes.
    """
    var result = String()
    b16encode(input_bytes, result)
    return result^


# ===-----------------------------------------------------------------------===#
# b16decode
# ===-----------------------------------------------------------------------===#


def unsafe_b16decode(
    str: StringSlice[mut=False, _],
    mut output_bytes: Span[mut=True, Byte, _],
):
    """Performs base16 decoding from a string to output bytes.

    The caller must ensure the output buffer is at least
    `len(str) // 2` bytes; no bounds check is performed.

    Args:
        str: A base16 encoded string.
        output_bytes: The output buffer to write decoded bytes into.
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
    def decode(c: Byte) -> Byte:
        if `A` <= c <= `Z`:
            return c - `A` + Byte(10)
        elif `a` <= c <= `z`:
            return c - `a` + Byte(10)
        elif `0` <= c <= `9`:
            return c - `0`
        else:
            return Byte(-1)

    var input_bytes = str.as_bytes()
    var n = str.byte_length()
    debug_assert(n % 2 == 0, "Input length '", n, "' must be divisible by 2")
    debug_assert(
        len(output_bytes) >= n // 2,
        "output buffer too small for base16 decoding: need ",
        n // 2,
        ", got ",
        len(output_bytes),
    )

    for i in range(0, n, 2):
        var hi = input_bytes[i]
        var lo = input_bytes[i + 1]
        output_bytes[i // 2] = decode(hi) << 4 | decode(lo)


@always_inline
def b16decode(str: StringSlice[mut=False, _], mut result: List[Byte]):
    """Performs base16 decoding into a reusable `List[Byte]` buffer.

    `result` is resized to fit the decoded output; existing contents are
    overwritten. Capacity is reused when possible, so calling this in a
    loop with the same `result` avoids per-call allocation.

    Args:
        str: A base16 encoded string.
        result: The byte buffer to write into.
    """
    var output_len = str.byte_length() // 2
    result.resize(unsafe_uninit_length=output_len)
    var output_span = Span(result)
    unsafe_b16decode(str, output_span)


@always_inline
def b16decode(str: StringSlice[mut=False, _]) -> List[Byte]:
    """Performs base16 decoding on a string, returning decoded bytes.

    Args:
        str: A base16 encoded string.

    Returns:
        The decoded bytes as a List[Byte].
    """
    var result = List[Byte]()
    b16decode(str, result)
    return result^
