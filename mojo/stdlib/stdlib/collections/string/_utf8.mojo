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

"""Implement UTF-8 utils."""

from base64._b64encode import _sub_with_saturation
from sys import is_compile_time, simdwidthof
from sys.intrinsics import likely, llvm_intrinsic

from bit import count_leading_zeros
from memory import Span, UnsafePointer

# ===-----------------------------------------------------------------------===#
# Validate UTF-8
# ===-----------------------------------------------------------------------===#


alias TOO_SHORT: UInt8 = 1 << 0
alias TOO_LONG: UInt8 = 1 << 1
alias OVERLONG_3: UInt8 = 1 << 2
alias SURROGATE: UInt8 = 1 << 4
alias OVERLONG_2: UInt8 = 1 << 5
alias TWO_CONTS: UInt8 = 1 << 7
alias TOO_LARGE: UInt8 = 1 << 3
alias TOO_LARGE_1000: UInt8 = 1 << 6
alias OVERLONG_4: UInt8 = 1 << 6
alias CARRY: UInt8 = TOO_SHORT | TOO_LONG | TWO_CONTS


# fmt: off
alias shuf1 = SIMD[DType.uint8, 16](
    TOO_LONG, TOO_LONG, TOO_LONG, TOO_LONG,
    TOO_LONG, TOO_LONG, TOO_LONG, TOO_LONG,
    TWO_CONTS, TWO_CONTS, TWO_CONTS, TWO_CONTS,
    TOO_SHORT | OVERLONG_2,
    TOO_SHORT,
    TOO_SHORT | OVERLONG_3 | SURROGATE,
    TOO_SHORT | TOO_LARGE | TOO_LARGE_1000 | OVERLONG_4
)

alias shuf2 = SIMD[DType.uint8, 16](
    CARRY | OVERLONG_3 | OVERLONG_2 | OVERLONG_4,
    CARRY | OVERLONG_2,
    CARRY,
    CARRY,
    CARRY | TOO_LARGE,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000 | SURROGATE,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000
)
alias shuf3 = SIMD[DType.uint8, 16](
    TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT,
    TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT,
    TOO_LONG | OVERLONG_2 | TWO_CONTS | OVERLONG_3 | TOO_LARGE_1000 | OVERLONG_4,
    TOO_LONG | OVERLONG_2 | TWO_CONTS | OVERLONG_3 | TOO_LARGE,
    TOO_LONG | OVERLONG_2 | TWO_CONTS | SURROGATE | TOO_LARGE,
    TOO_LONG | OVERLONG_2 | TWO_CONTS | SURROGATE | TOO_LARGE,
    TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT
)
# fmt: on


@always_inline
fn _extract_vector[
    width: Int, //, offset: Int
](a: SIMD[DType.uint8, width], b: SIMD[DType.uint8, width]) -> SIMD[
    DType.uint8, width
]:
    # generates a single `vpalignr` on x86 with AVX
    return a.join(b).slice[width, offset=offset]()


fn validate_chunk[
    simd_size: Int
](
    current_block: SIMD[DType.uint8, simd_size],
    previous_input_block: SIMD[DType.uint8, simd_size],
) -> SIMD[DType.uint8, simd_size]:
    alias v0f = SIMD[DType.uint8, simd_size](0x0F)
    alias v80 = SIMD[DType.uint8, simd_size](0x80)
    alias third_byte = 0b11100000 - 0x80
    alias fourth_byte = 0b11110000 - 0x80
    var prev1 = _extract_vector[simd_size - 1](
        previous_input_block, current_block
    )
    var byte_1_high = shuf1._dynamic_shuffle(prev1 >> 4)
    var byte_1_low = shuf2._dynamic_shuffle(prev1 & v0f)
    var byte_2_high = shuf3._dynamic_shuffle(current_block >> 4)
    var sc = byte_1_high & byte_1_low & byte_2_high

    var prev2 = _extract_vector[simd_size - 2](
        previous_input_block, current_block
    )
    var prev3 = _extract_vector[simd_size - 3](
        previous_input_block, current_block
    )
    var is_third_byte = _sub_with_saturation(prev2, third_byte)
    var is_fourth_byte = _sub_with_saturation(prev3, fourth_byte)
    var must23 = is_third_byte | is_fourth_byte
    var must23_as_80 = must23 & v80
    return must23_as_80 ^ sc


fn _is_valid_utf8_runtime(span: Span[Byte]) -> Bool:
    """Fast utf-8 validation using SIMD instructions.

    References for this algorithm:
    J. Keiser, D. Lemire, Validating UTF-8 In Less Than One Instruction Per
    Byte, Software: Practice and Experience 51 (5), 2021
    https://arxiv.org/abs/2010.03090

    Blog post:
    https://lemire.me/blog/2018/10/19/validating-utf-8-bytes-using-only-0-45-cycles-per-byte-avx-edition/

    Code adapted from:
    https://github.com/simdutf/SimdUnicode/blob/main/src/UTF8.cs
    """

    ptr = span.unsafe_ptr()
    length = len(span)
    alias simd_size = sys.simdbytewidth()
    var i: Int = 0
    var previous = SIMD[DType.uint8, simd_size]()

    while i + simd_size <= length:
        var current_bytes = (ptr + i).load[width=simd_size]()
        var has_error = validate_chunk(current_bytes, previous)
        previous = current_bytes
        if any(has_error != 0):
            return False
        i += simd_size

    var has_error: SIMD[DType.uint8, simd_size]
    # last incomplete chunk
    if i != length:
        var buffer = SIMD[DType.uint8, simd_size](0)
        for j in range(i, length):
            buffer[j - i] = (ptr + j)[]
        has_error = validate_chunk(buffer, previous)
    else:
        # Add a chunk of 0s to the end to validate continuations bytes
        has_error = validate_chunk(SIMD[DType.uint8, simd_size](), previous)

    return all(has_error == 0)


fn _is_valid_utf8(span: Span[Byte]) -> Bool:
    """Verify that the bytes are valid UTF-8.

    Args:
        span: The Span of bytes.

    Returns:
        Whether the data is valid UTF-8.

    #### UTF-8 coding format
    [Table 3-7 page 94](http://www.unicode.org/versions/Unicode6.0.0/ch03.pdf).
    Well-Formed UTF-8 Byte Sequences

    Code Points        | First Byte | Second Byte | Third Byte | Fourth Byte |
    :----------        | :--------- | :---------- | :--------- | :---------- |
    U+0000..U+007F     | 00..7F     |             |            |             |
    U+0080..U+07FF     | C2..DF     | 80..BF      |            |             |
    U+0800..U+0FFF     | E0         | ***A0***..BF| 80..BF     |             |
    U+1000..U+CFFF     | E1..EC     | 80..BF      | 80..BF     |             |
    U+D000..U+D7FF     | ED         | 80..***9F***| 80..BF     |             |
    U+E000..U+FFFF     | EE..EF     | 80..BF      | 80..BF     |             |
    U+10000..U+3FFFF   | F0         | ***90***..BF| 80..BF     | 80..BF      |
    U+40000..U+FFFFF   | F1..F3     | 80..BF      | 80..BF     | 80..BF      |
    U+100000..U+10FFFF | F4         | 80..***8F***| 80..BF     | 80..BF      |
    """
    return _is_valid_utf8_runtime(span)


# ===-----------------------------------------------------------------------===#
# Utils
# ===-----------------------------------------------------------------------===#


@always_inline
fn _is_utf8_continuation_byte[
    w: Int
](vec: SIMD[DType.uint8, w]) -> SIMD[DType.bool, w]:
    return vec.cast[DType.int8]() < -(0b1000_0000 >> 1)


fn _count_utf8_continuation_bytes(str_slice: StringSlice) -> Int:
    alias sizes = (256, 128, 64, 32, 16, 8)
    var ptr = str_slice.unsafe_ptr()
    var num_bytes = str_slice.byte_length()
    var amnt: Int = 0
    var processed = 0

    @parameter
    for i in range(len(sizes)):
        alias s = sizes[i]

        @parameter
        if simdwidthof[DType.uint8]() >= s:
            var rest = num_bytes - processed
            for _ in range(rest // s):
                var vec = (ptr + processed).load[width=s]()
                var comp = _is_utf8_continuation_byte(vec)
                amnt += Int(comp.cast[DType.uint8]().reduce_add())
                processed += s

    for i in range(num_bytes - processed):
        amnt += Int(_is_utf8_continuation_byte(ptr[processed + i]))

    return amnt


@always_inline
fn _utf8_first_byte_sequence_length(b: Byte) -> Int:
    """Get the length of the sequence starting with given byte. Do note that
    this does not work correctly if given a continuation byte."""

    debug_assert(
        not _is_utf8_continuation_byte(b),
        "Function does not work correctly if given a continuation byte.",
    )
    return Int(count_leading_zeros(~b) | (b < 0b1000_0000).cast[DType.uint8]())


fn _utf8_byte_type(b: SIMD[DType.uint8, _], /) -> __type_of(b):
    """UTF-8 byte type.

    Returns:
        The byte type.

    Notes:

        - 0 -> ASCII byte.
        - 1 -> continuation byte.
        - 2 -> start of 2 byte long sequence.
        - 3 -> start of 3 byte long sequence.
        - 4 -> start of 4 byte long sequence.
    """
    return count_leading_zeros(~b)


@always_inline
fn _is_newline_char_utf8[
    include_r_n: Bool = False
](
    p: UnsafePointer[Byte, mut=False, **_],
    eol_start: Int,
    b0: Byte,
    char_len: Int,
) -> Bool:
    """Returns whether the char is a newline char.

    Safety:
        This assumes valid utf-8 is passed.
    """
    # highly performance sensitive code, benchmark before touching
    alias `\r` = UInt8(ord("\r"))
    alias `\n` = UInt8(ord("\n"))
    alias `\t` = UInt8(ord("\t"))
    alias `\x1c` = UInt8(ord("\x1c"))
    alias `\x1e` = UInt8(ord("\x1e"))

    # here it's actually faster to have branching due to the branch predictor
    # "realizing" that the char_len == 1 path is often taken. Using the likely
    # intrinsic is to make the machine code be ordered to optimize machine
    # instruction fetching, which is an optimization for the CPU front-end.
    if likely(char_len == 1):
        return `\t` <= b0 <= `\x1e` and not (`\r` < b0 < `\x1c`)
    elif char_len == 2:
        var b1 = p[eol_start + 1]
        var is_next_line = b0 == 0xC2 and b1 == 0x85  # unicode next line \x85

        @parameter
        if include_r_n:
            return is_next_line or (b0 == `\r` and b1 == `\n`)
        else:
            return is_next_line
    elif char_len == 3:  # unicode line sep or paragraph sep: \u2028 , \u2029
        var b1 = p[eol_start + 1]
        var b2 = p[eol_start + 2]
        return b0 == 0xE2 and b1 == 0x80 and (b2 == 0xA8 or b2 == 0xA9)
    return False
