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

from std.bit import byte_swap
from std.collections.string._unicode import (
    _get_uppercase_mapping,
    BIGGEST_UNICODE_CODEPOINT,
)
from std.sys.info import Endian

from std.testing import TestSuite, assert_equal, assert_raises


def test_uppercase_conversion() raises:
    # a -> A
    var count1: Int
    count1, ref chars1 = _get_uppercase_mapping(Codepoint(97)).value()
    assert_equal(count1, 1)
    assert_equal(chars1[0], Codepoint(65))
    assert_equal(chars1[1], Codepoint(0))
    assert_equal(chars1[2], Codepoint(0))

    # ß -> SS
    var count2: Int
    count2, ref chars2 = _get_uppercase_mapping(
        Codepoint.from_u32(0xDF).value()
    ).value()
    assert_equal(count2, 2)
    assert_equal(chars2[0], Codepoint.from_u32(0x53).value())
    assert_equal(chars2[1], Codepoint.from_u32(0x53).value())
    assert_equal(chars2[2], Codepoint(0))

    # ΐ -> Ϊ́
    var count3: Int
    count3, ref chars3 = _get_uppercase_mapping(
        Codepoint.from_u32(0x390).value()
    ).value()
    assert_equal(count3, 3)
    assert_equal(chars3[0], Codepoint.from_u32(0x0399).value())
    assert_equal(chars3[1], Codepoint.from_u32(0x0308).value())
    assert_equal(chars3[2], Codepoint.from_u32(0x0301).value())


def _assert_roundtrip[
    dtype: DType, //
](value: Scalar[dtype]) raises where dtype.is_unsigned():
    """Assert strict and replace decoding both yield `chr(value)`."""
    var buf: List[Scalar[dtype]] = [value]
    assert_equal(String(from_codepoints=buf), chr(Int(value)))
    assert_equal(String(from_codepoints=buf, errors="replace"), chr(Int(value)))


def _assert_invalid[
    dtype: DType, //
](value: Scalar[dtype]) raises where dtype.is_unsigned():
    """Assert strict raises and replace yields the replacement character."""
    var buf: List[Scalar[dtype]] = [value]
    with assert_raises():
        _ = String(from_codepoints=buf)
    assert_equal(String(from_codepoints=buf, errors="replace"), "�")


def test_codepoint_parsing() raises:
    # ASCII / Latin-1 sample + boundaries (UInt8: every value is valid).
    for i in [UInt8(0), 0x41, 0x7F, 0x80, 0xA9, 0xFF]:
        _assert_roundtrip(i)

    # UInt16 BMP samples, boundaries, and surrogate rejection.
    # `range` is exclusive of the end, so include UInt16.MAX explicitly.
    for i in [
        UInt16(0x100),
        0x20AC,  # €
        0xD7FF,  # last scalar before surrogates
        0xE000,  # first scalar after surrogates
        0xFFFD,
        UInt16.MAX,
    ]:
        _assert_roundtrip(i)

    for i in [UInt16(0xD800), 0xDBFF, 0xDC00, 0xDFFF]:
        _assert_invalid(i)

    # UInt32 supplementary-plane samples + Unicode max.
    for i in [
        UInt32(UInt16.MAX) + 1,
        0x10000,  # first supplementary
        0x1F525,  # 🔥
        BIGGEST_UNICODE_CODEPOINT,
    ]:
        _assert_roundtrip(i)

    # Invalid range above the Unicode codespace.
    for i in [
        UInt64(BIGGEST_UNICODE_CODEPOINT) + 1,
        UInt64(UInt32.MAX),
        UInt64(UInt32.MAX) + 2,
        UInt64.MAX - 2,
    ]:
        _assert_invalid(i)


def test_codepoints_replace_preserves_neighbors() raises:
    # Regression: errors="replace" used to corrupt multi-codepoint buffers
    # because `+=` observed a still-zero string length after unsafe writes.
    # [0x41, 0xD800, 0x42] must become 41 EF BF BD 42 ("A�B").
    var buf: List[UInt16] = [0x41, 0xD800, 0x42]
    with assert_raises():
        _ = String(from_codepoints=buf)
    var replaced = String(from_codepoints=buf, errors="replace")
    assert_equal(replaced, "A�B")
    var bytes = replaced.as_bytes()
    assert_equal(len(bytes), 5)
    assert_equal(bytes[0], 0x41)
    assert_equal(bytes[1], 0xEF)
    assert_equal(bytes[2], 0xBF)
    assert_equal(bytes[3], 0xBD)
    assert_equal(bytes[4], 0x42)


def test_from_codepoints_supplementary() raises:
    var buf: List[UInt32] = [0x41, 0x1F525, 0x42]
    assert_equal(String(from_codepoints=buf), "A🔥B")
    assert_equal(String(from_codepoints=buf, errors="replace"), "A🔥B")

    var invalid: List[UInt32] = [0x41, 0xD800, 0x42]
    with assert_raises():
        _ = String(from_codepoints=invalid)
    assert_equal(String(from_codepoints=invalid, errors="replace"), "A�B")


def _store_u32[endian: Endian](value: UInt32) -> UInt32:
    """Return `value` as it would appear in an `endian`-ordered buffer."""
    comptime if Endian.native() == endian:
        return value
    else:
        return byte_swap(value)


def _store_u16[endian: Endian](value: UInt16) -> UInt16:
    """Return `value` as it would appear in an `endian`-ordered buffer."""
    comptime if Endian.native() == endian:
        return value
    else:
        return byte_swap(value)


def test_endianness() raises:
    var be16: List[UInt16] = [
        _store_u16[.big](0x41),
        _store_u16[.big](0x20AC),
        _store_u16[.big](0x42),
    ]
    assert_equal(String(from_codepoints=be16, endian=.big), "A€B")
    assert_equal(
        String(from_codepoints=be16, endian=.big, errors="replace"),
        "A€B",
    )

    var le32: List[UInt32] = [
        _store_u32[.little](0x41),
        _store_u32[.little](0x1F525),
        _store_u32[.little](0x42),
    ]
    assert_equal(String(from_codepoints=le32, endian=.little), "A🔥B")
    assert_equal(
        String(from_codepoints=le32, endian=.little, errors="replace"),
        "A🔥B",
    )

    # Host-endian default.
    var native32: List[UInt32] = [0x41, 0x1F525, 0x42]
    assert_equal(String(from_codepoints=native32), "A🔥B")


def test_bom_detection() raises:
    # Native-order BOM (U+FEFF) is skipped; payload decodes as host endian.
    var native_bom: List[UInt32] = [0xFEFF, 0x41, 0x1F525, 0x42]
    assert_equal(
        String.__init__[detect_bom=True](from_codepoints=native_bom),
        "A🔥B",
    )

    # Opposite-endian BOM: first unit is byte-swapped FEFF, payload swapped.
    var foreign_bom: List[UInt32] = [
        byte_swap(UInt32(0xFEFF)),
        byte_swap(UInt32(0x41)),
        byte_swap(UInt32(0x1F525)),
        byte_swap(UInt32(0x42)),
    ]
    assert_equal(
        String.__init__[detect_bom=True](from_codepoints=foreign_bom),
        "A🔥B",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
