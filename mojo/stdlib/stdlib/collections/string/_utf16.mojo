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

"""Implement UTF-16 utils."""

from bit._mask import splat
from sys.intrinsics import likely, unlikely


fn _decode_utf16[
    dtype: DType, //,
    *,
    strict: Bool = True,
    replace: Codepoint = Codepoint.ord("�"),
](*, from_utf16: Span[mut=False, Scalar[dtype], **_]) raises -> String:
    constrained[dtype.is_integral(), "The dtype must be integral"]()
    constrained[
        dtype.bitwidth() > 8,
        "Decoding UTF-16 from a buffer of bytes is ambiguous ",
        "bitcast it to UInt16 or use the codepoints decoder if they are ",
        "independent codepoints",
    ]()

    # TODO: this could use reduce_sum with a function that calculates the utf8
    # length from the utf16 codepoints. Performance needs to be measured
    var utf16_len = UInt(len(from_utf16))
    var result = String(capacity=2 * utf16_len)
    var utf16_ptr = from_utf16.unsafe_ptr()
    var utf16_idx = UInt(0)
    var utf8_ptr = result.unsafe_ptr_mut()
    var utf8_offset = UInt(0)

    while utf16_idx < utf16_len:
        var c = utf16_ptr[utf16_idx].cast[DType.uint32]()
        var ptr = utf8_ptr + utf8_offset
        var num_bytes: UInt
        # high surrogate marker 0xD800 == 0b11_0110 << 10
        var is_high_surrogate = (c >> 10).eq(0b11_0110)
        var is_full_surrogate = Scalar[DType.bool](False)

        if likely(c < 0b1000_0000):  # ASCII
            num_bytes = 1
            ptr[0] = c.cast[DType.uint8]()
        else:
            # get lower 10 bits
            alias low_10b = UInt32(0b0011_1111_1111)
            var has_more = utf16_idx < utf16_len - 1
            var c2 = utf16_ptr[utf16_idx + Int(has_more)].cast[DType.uint32]()
            # low surrogate marker 0xDC00 == 0b11_0111 << 10
            var is_low_surrogate = (c2 >> 10).eq(0b11_0111)
            is_full_surrogate = is_high_surrogate & is_low_surrogate
            var any_pair = is_high_surrogate | is_low_surrogate

            @parameter
            if dtype.bitwidth() >= 32:
                if unlikely(c > 0xFFFF):

                    @parameter
                    if strict:
                        raise Error(
                            "Non UTF-16 codepoint at index: ", utf16_idx
                        )
                    else:
                        c = replace.to_u32()

            if unlikely(Bool(any_pair & ~is_full_surrogate)):

                @parameter
                if strict:
                    raise Error("Unpaired surrogate at index: ", utf16_idx)
                else:
                    c = replace.to_u32()
            elif is_full_surrogate:
                c = 2**16 + (((c & low_10b) << 10) | (c2 & low_10b))

            num_bytes = Codepoint(
                unsafe_unchecked_codepoint=c
            ).unsafe_write_utf8[optimize_ascii=False, branchless=True](ptr)

        utf8_offset += num_bytes
        utf16_idx += UInt(1) << UInt(is_full_surrogate.cast[DType.index]())

    result._set_byte_length(utf8_offset)
    return result^
