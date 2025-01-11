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
# RUN: %mojo %s

from testing import assert_true, assert_false, assert_equal


def test_char_validity():
    # Check that basic unchecked constructor behaves as expected.
    var c1 = Char(unsafe_unchecked_codepoint=32)
    assert_equal(c1._scalar_value, 32)

    assert_true(Char.from_u32(0))

    # For a visual intuition of what constitues a valid scalar value:
    #   https://connorgray.com/ephemera/project-log#2025-01-09

    # Last valid code point in the smaller scalar value range.
    assert_true(Char.from_u32(0xD7FF))

    # First surrogate code point, not valid.
    assert_false(Char.from_u32(0xD7FF + 1))

    # Last surrogate code point, not valid
    assert_false(Char.from_u32(0xDFFF))

    # First valid code point in the larger scalar value range.
    assert_true(Char.from_u32(0xE000))

    # Beyond Unicode's committed range of code points.
    assert_false(Char.from_u32(0x10FFFF + 1))


def test_char_utf8_byte_length():
    fn codepoint_len(cp: UInt32) -> Int:
        return Char.from_u32(cp).value().utf8_byte_length()

    # 1-byte (ASCII) codepoints
    assert_equal(codepoint_len(0), 1)
    assert_equal(codepoint_len(32), 1)
    assert_equal(codepoint_len(127), 1)

    # 2-byte codepoints -- 0b110x_xxxx 0b10xx_xxxx (11 x's)
    # Smallest 2-byte codepoint
    assert_equal(codepoint_len(128), 2)
    # Largest 2-byte codepoint
    assert_equal(codepoint_len(2**11 - 1), 2)  # 2^11 - 1 == 2047

    # 3-byte codepoints -- 0b1110_xxxx 0b10xx_xxxx 0b10xx_xxxx (16 x's)
    # Smallest 3-byte codepoint
    assert_equal(codepoint_len(2**11), 3)  # 2^11 == 2048
    # Largest 3-byte codepoint
    assert_equal(codepoint_len(2**16 - 1), 3)  # 2^16 - 1 == 65535 == 0xFFFF

    # 4-byte codepoints 0b1111_0xxx 0b10xx_xxxx 0b10xx_xxxx 0b10xx_xxxx (21 x's)
    # Smallest 4-byte codepoint
    assert_equal(codepoint_len(2**16), 4)
    # Largest 4-byte codepoint
    assert_equal(codepoint_len(0x10FFFF), 4)  # Maximum Unicode codepoint


def main():
    test_char_validity()
    test_char_utf8_byte_length()
