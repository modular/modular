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
# RUN: %mojo %s

from testing import assert_equal, assert_false, assert_not_equal, assert_true


def test_char_validity():
    # Check that basic unchecked constructor behaves as expected.
    var c1 = Codepoint(unsafe_unchecked_codepoint=32)
    assert_equal(c1._scalar_value, 32)

    assert_true(Codepoint.from_u32(0))

    # For a visual intuition of what constitues a valid scalar value:
    #   https://connorgray.com/ephemera/project-log#2025-01-09

    # Last valid code point in the smaller scalar value range.
    assert_true(Codepoint.from_u32(0xD7FF))

    # First surrogate code point, not valid.
    assert_false(Codepoint.from_u32(0xD7FF + 1))

    # Last surrogate code point, not valid
    assert_false(Codepoint.from_u32(0xDFFF))

    # First valid code point in the larger scalar value range.
    assert_true(Codepoint.from_u32(0xE000))

    # Beyond Unicode's committed range of code points.
    assert_false(Codepoint.from_u32(0x10FFFF + 1))


def test_char_from_u8():
    var c1 = Codepoint(UInt8(0))
    assert_true(c1.is_ascii())

    # All non-negative 8-bit integers are codepoints, but not all are ASCII.
    var c2 = Codepoint(UInt8(255))
    assert_false(c2.is_ascii())


def test_char_comparison():
    assert_equal(Codepoint(0), Codepoint(0))
    assert_not_equal(Codepoint(0), Codepoint(1))


def test_char_formatting():
    assert_equal(String(Codepoint(0)), "\0")
    assert_equal(String(Codepoint(32)), " ")
    assert_equal(String(Codepoint(97)), "a")
    assert_equal(String(Codepoint.from_u32(0x00BE).value()), "¾")
    assert_equal(String(Codepoint.from_u32(0x1F642).value()), "🙂")


def test_char_properties():
    assert_true(Codepoint.from_u32(0).value().is_ascii())
    # Last ASCII codepoint.
    assert_true(
        Codepoint.from_u32(0b0111_1111).value().is_ascii()
    )  # ASCII 127 0x7F

    # First non-ASCII codepoint.
    assert_false(Codepoint.from_u32(0b1000_0000).value().is_ascii())
    assert_false(Codepoint.from_u32(0b1111_1111).value().is_ascii())


def test_char_is_posix_space():
    # checking true cases
    assert_true(Codepoint.ord(" ").is_posix_space())
    assert_true(Codepoint.ord("\n").is_posix_space())
    assert_true(Codepoint.ord("\n").is_posix_space())
    assert_true(Codepoint.ord("\t").is_posix_space())
    assert_true(Codepoint.ord("\r").is_posix_space())
    assert_true(Codepoint.ord("\v").is_posix_space())
    assert_true(Codepoint.ord("\f").is_posix_space())

    # Checking false cases
    assert_false(Codepoint.ord("a").is_posix_space())
    assert_false(Codepoint.ord("a").is_posix_space())
    assert_false(Codepoint.ord("u").is_posix_space())
    assert_false(Codepoint.ord("s").is_posix_space())
    assert_false(Codepoint.ord("t").is_posix_space())
    assert_false(Codepoint.ord("i").is_posix_space())
    assert_false(Codepoint.ord("n").is_posix_space())
    assert_false(Codepoint.ord("z").is_posix_space())
    assert_false(Codepoint.ord(".").is_posix_space())


def test_char_is_lower():
    assert_true(Codepoint.ord("a").is_ascii_lower())
    assert_true(Codepoint.ord("b").is_ascii_lower())
    assert_true(Codepoint.ord("y").is_ascii_lower())
    assert_true(Codepoint.ord("z").is_ascii_lower())

    assert_false(Codepoint.from_u32(ord("a") - 1).value().is_ascii_lower())
    assert_false(Codepoint.from_u32(ord("z") + 1).value().is_ascii_lower())

    assert_false(Codepoint.ord("!").is_ascii_lower())
    assert_false(Codepoint.ord("0").is_ascii_lower())


def test_char_is_upper():
    assert_true(Codepoint.ord("A").is_ascii_upper())
    assert_true(Codepoint.ord("B").is_ascii_upper())
    assert_true(Codepoint.ord("Y").is_ascii_upper())
    assert_true(Codepoint.ord("Z").is_ascii_upper())

    assert_false(Codepoint.from_u32(ord("A") - 1).value().is_ascii_upper())
    assert_false(Codepoint.from_u32(ord("Z") + 1).value().is_ascii_upper())

    assert_false(Codepoint.ord("!").is_ascii_upper())
    assert_false(Codepoint.ord("0").is_ascii_upper())


def test_char_is_digit():
    assert_true(Codepoint.ord("1").is_ascii_digit())
    assert_false(Codepoint.ord("g").is_ascii_digit())

    # Devanagari Digit 6 — non-ASCII digits are not "ascii digit".
    assert_false(Codepoint.ord("६").is_ascii_digit())


def test_char_is_printable():
    assert_true(Codepoint.ord("a").is_ascii_printable())
    assert_false(Codepoint.ord("\n").is_ascii_printable())
    assert_false(Codepoint.ord("\t").is_ascii_printable())

    # Non-ASCII characters are not considered "ascii printable".
    assert_false(Codepoint.ord("स").is_ascii_printable())


alias SIGNIFICANT_CODEPOINTS = List[Tuple[Int, List[Byte]]](
    # --------------------------
    # 1-byte (ASCII) codepoints
    # --------------------------
    # Smallest 1-byte codepoint value
    (0, List[Byte](0)),
    (1, List[Byte](1)),
    (32, List[Byte](32)),  # First non-control character
    (0b0111_1111, List[Byte](127)),  # 127
    # ------------------
    # 2-byte codepoints -- 0b110x_xxxx 0b10xx_xxxx (11 x's)
    # ------------------
    # Smallest 2-byte codepoint
    (128, List[Byte](0b1100_0010, 0b1000_0000)),
    # Largest 2-byte codepoint -- 2^11 - 1 == 2047
    (2**11 - 1, List[Byte](0b1101_1111, 0b1011_1111)),
    # ------------------
    # 3-byte codepoints -- 0b1110_xxxx 0b10xx_xxxx 0b10xx_xxxx (16 x's)
    # ------------------
    # Smallest 3-byte codepoint -- 2^11 == 2048
    (2**11, List[Byte](0b1110_0000, 0b1010_0000, 0b1000_0000)),
    # Largest 3-byte codepoint -- 2^16 - 1 == 65535 == 0xFFFF
    (2**16 - 1, List[Byte](0b1110_1111, 0b1011_1111, 0b1011_1111)),
    # ------------------
    # 4-byte codepoints 0b1111_0xxx 0b10xx_xxxx 0b10xx_xxxx 0b10xx_xxxx (21 x's)
    # ------------------
    # Smallest 4-byte codepoint
    (2**16, List[Byte](0b1111_0000, 0b1001_0000, 0b1000_0000, 0b1000_0000)),
    # Largest 4-byte codepoint -- Maximum Unicode codepoint
    (
        0x10FFFF,
        List[Byte](0b1111_0100, 0b1000_1111, 0b1011_1111, 0b1011_1111),
    ),
)


fn assert_utf8_bytes(codepoint: UInt32, owned expected: List[Byte]) raises:
    var char_opt = Codepoint.from_u32(codepoint)
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
    for entry in SIGNIFICANT_CODEPOINTS:
        var codepoint = entry[][0]
        var expected_utf8 = entry[][1]

        assert_utf8_bytes(codepoint, expected_utf8)


def test_char_utf8_byte_length():
    for entry in SIGNIFICANT_CODEPOINTS:
        var codepoint = entry[][0]
        var expected_utf8 = entry[][1]

        var computed_len = Codepoint.from_u32(
            codepoint
        ).value().utf8_byte_length()

        assert_equal(computed_len, len(expected_utf8))


def test_char_comptime():
    alias c1 = Codepoint.from_u32(32).value()

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
    test_char_is_lower()
    test_char_is_upper()
    test_char_is_digit()
    test_char_is_printable()
    test_char_utf8_encoding()
    test_char_utf8_byte_length()
    test_char_comptime()
