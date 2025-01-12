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

from testing import assert_true, assert_false, assert_equal, assert_not_equal


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


def test_char_from_u8():
    var c1 = Char(UInt8(0))
    assert_true(c1.is_ascii())

    # All non-negative 8-bit integers are codepoints, but not all are ASCII.
    var c2 = Char(UInt8(255))
    assert_false(c2.is_ascii())


def test_char_comparison():
    assert_equal(Char(0), Char(0))
    assert_not_equal(Char(0), Char(1))


def test_char_formatting():
    assert_equal(str(Char(0)), "\0")
    assert_equal(str(Char(32)), " ")
    assert_equal(str(Char(97)), "a")
    assert_equal(str(Char.from_u32(0x00BE).value()), "¾")
    assert_equal(str(Char.from_u32(0x1F642).value()), "🙂")


def test_char_properties():
    assert_true(Char.from_u32(0).value().is_ascii())
    # Last ASCII codepoint.
    assert_true(Char.from_u32(0b0111_1111).value().is_ascii())  # ASCII 127 0x7F

    # First non-ASCII codepoint.
    assert_false(Char.from_u32(0b1000_0000).value().is_ascii())
    assert_false(Char.from_u32(0b1111_1111).value().is_ascii())


def test_char_is_posix_space():
    # checking true cases
    assert_true(Char.ord(" ").is_posix_space())
    assert_true(Char.ord("\n").is_posix_space())
    assert_true(Char.ord("\n").is_posix_space())
    assert_true(Char.ord("\t").is_posix_space())
    assert_true(Char.ord("\r").is_posix_space())
    assert_true(Char.ord("\v").is_posix_space())
    assert_true(Char.ord("\f").is_posix_space())

    # Checking false cases
    assert_false(Char.ord("a").is_posix_space())
    assert_false(Char.ord("a").is_posix_space())
    assert_false(Char.ord("u").is_posix_space())
    assert_false(Char.ord("s").is_posix_space())
    assert_false(Char.ord("t").is_posix_space())
    assert_false(Char.ord("i").is_posix_space())
    assert_false(Char.ord("n").is_posix_space())
    assert_false(Char.ord("z").is_posix_space())
    assert_false(Char.ord(".").is_posix_space())


fn assert_utf8_bytes(codepoint: UInt32, owned expected: List[Byte]) raises:
    var char_opt = Char.from_u32(codepoint)
    var char = char_opt.value()

    # Allocate a length-4 buffer to write to.
    var buffer = List[Byte](0, 0, 0, 0)
    var written = char.unsafe_write_utf8(buffer.unsafe_ptr())

    # Check that the number of bytes written was as expected.
    assert_equal(
        written,
        len(expected),
        "wrong byte count written encoding codepoint: {}".format(codepoint),
    )

    # Normalize `expected` to length 4 so we can compare the written byte
    # values with `buffer`.
    for _ in range(4 - len(expected)):
        expected.append(0)

    assert_equal(
        buffer,
        expected,
        "wrong byte values written encoding codepoint: {}".format(codepoint),
    )


def test_char_utf8_encoding():
    assert_utf8_bytes(0, List[Byte](0))
    assert_utf8_bytes(1, List[Byte](1))
    assert_utf8_bytes(127, List[Byte](127))

    # Smallest 2-byte codepoint
    assert_utf8_bytes(128, List[Byte](0b1100_0010, 0b1000_0000))
    # Largest 2-byte codepoint
    assert_utf8_bytes(2**11 - 1, List[Byte](0b1101_1111, 0b1011_1111))

    # Smallest 3-byte codepoint -- 2^11 == 2048
    assert_utf8_bytes(
        2**11, List[Byte](0b1110_0000, 0b1010_0000, 0b1000_0000)
    )
    # Largest 3-byte codepoint -- 2^16 - 1 == 65535 == 0xFFFF
    assert_utf8_bytes(
        2**16 - 1, List[Byte](0b1110_1111, 0b1011_1111, 0b1011_1111)
    )

    # Smallest 4-byte codepoint
    assert_utf8_bytes(
        2**16, List[Byte](0b1111_0000, 0b1001_0000, 0b1000_0000, 0b1000_0000)
    )
    # Largest 4-byte codepoint -- Maximum Unicode codepoint
    assert_utf8_bytes(
        0x10FFFF, List[Byte](0b1111_0100, 0b1000_1111, 0b1011_1111, 0b1011_1111)
    )


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


def test_char_comptime():
    alias c1 = Char.from_u32(32).value()

    # Test that `utf8_byte_length()` works at compile time.
    alias c1_bytes = c1.utf8_byte_length()
    assert_equal(c1_bytes, 1)


def main():
    test_char_validity()
    test_char_from_u8()
    test_char_comparison()
    test_char_formatting()
    test_char_properties()
    test_char_is_posix_space()
    test_char_utf8_encoding()
    test_char_utf8_byte_length()
    test_char_comptime()
