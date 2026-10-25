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

from std.collections.string._utf16 import _decode_utf16, _invalid_utf16_strict

from std.testing import TestSuite, assert_equal, assert_raises

# ===----------------------------------------------------------------------=== #
# Tests
# ===----------------------------------------------------------------------=== #


def test_utf16_parsing() raises:
    var buf: List[UInt16] = []
    assert_equal(_decode_utf16(buf, _invalid_utf16_strict), "")
    buf = [65, 97, 33, 945, 10175]
    assert_equal(_decode_utf16(buf, _invalid_utf16_strict), "Aa!α➿")
    buf = [0x20AC]
    assert_equal(_decode_utf16(buf, _invalid_utf16_strict), "€")
    buf = [0xFFFD]
    assert_equal(_decode_utf16(buf, _invalid_utf16_strict), "�")
    buf = [0xD83D, 0xDD25]
    assert_equal(_decode_utf16(buf, _invalid_utf16_strict), "🔥")
    buf = [0xD801, 0xDC37]
    assert_equal(_decode_utf16(buf, _invalid_utf16_strict), chr(0x10437))
    buf = [0xD852, 0xDF62]
    assert_equal(_decode_utf16(buf, _invalid_utf16_strict), chr(0x24B62))

    buf = [0]
    for i in range(0xD800):
        buf[0] = UInt16(i)
        assert_equal(String(from_utf16=buf), chr(i))
        assert_equal(String(from_utf16=buf, errors="replace"), chr(i))

    # test unpaired surrogates
    for i in range(0xD800, 0xDFFF + 1):
        buf[0] = UInt16(i)
        with assert_raises():
            _ = String(from_utf16=buf)
        assert_equal(String(from_utf16=buf, errors="replace"), "�")

    for i in range(0xDFFF + 1, Int(UInt16.MAX) + 1):
        buf[0] = UInt16(i)
        assert_equal(String(from_utf16=buf), chr(i))
        assert_equal(String(from_utf16=buf, errors="replace"), chr(i))

    # test invalid range
    var buf64: List[UInt64] = [UInt64(UInt16.MAX) + 1]
    with assert_raises():
        _ = String(from_utf16=buf64)
    assert_equal(String(from_utf16=buf64, errors="replace"), "�")
    buf64 = [UInt64(UInt32.MAX)]
    with assert_raises():
        _ = String(from_utf16=buf64)
    assert_equal(String(from_utf16=buf64, errors="replace"), "�")
    buf64 = [UInt64(UInt32.MAX) + 2]
    with assert_raises():
        _ = String(from_utf16=buf64)
    assert_equal(String(from_utf16=buf64, errors="replace"), "�")
    buf64 = [UInt64.MAX - 2]
    with assert_raises():
        _ = String(from_utf16=buf64)
    assert_equal(String(from_utf16=buf64, errors="replace"), "�")


def test_utf16_surrogate_neighbors() raises:
    # BMP then unpaired low surrogate must not treat the BMP as invalid.
    var buf: List[UInt16] = [0x4E00, 0xDC00]
    with assert_raises():
        _ = String(from_utf16=buf)
    assert_equal(String(from_utf16=buf, errors="replace"), "一�")

    # Unpaired low surrogate then ASCII.
    buf = [0xDC00, 0x0041]
    with assert_raises():
        _ = String(from_utf16=buf)
    assert_equal(String(from_utf16=buf, errors="replace"), "�A")

    # Unpaired high surrogate then ASCII.
    buf = [0xD800, 0x0041]
    with assert_raises():
        _ = String(from_utf16=buf)
    assert_equal(String(from_utf16=buf, errors="replace"), "�A")

    # Valid surrogate pair still decodes.
    buf = [0xD83D, 0xDD25]
    assert_equal(String(from_utf16=buf), "🔥")
    assert_equal(String(from_utf16=buf, errors="replace"), "🔥")


def test_utf16_replace_preserves_neighbors() raises:
    # [0x41, 0xD800, 0x42] must become 41 EF BF BD 42 ("A�B").
    var buf: List[UInt16] = [0x41, 0xD800, 0x42]
    with assert_raises():
        _ = String(from_utf16=buf)
    var replaced = String(from_utf16=buf, errors="replace")
    assert_equal(replaced, "A�B")
    var bytes = replaced.as_bytes()
    assert_equal(len(bytes), 5)
    assert_equal(bytes[0], 0x41)
    assert_equal(bytes[1], 0xEF)
    assert_equal(bytes[2], 0xBF)
    assert_equal(bytes[3], 0xBD)
    assert_equal(bytes[4], 0x42)


def test_utf16_cjk_heap_capacity() raises:
    # U+4E00 encodes as 3 UTF-8 bytes. Choose enough units such that
    # a short capacity estimate overflows the SSO buffer.
    comptime count = String.INLINE_CAPACITY // 2 + 1
    var buf = List[UInt16](capacity=count)
    for _ in range(count):
        buf.append(0x4E00)
    var expected = String("一") * count
    assert_equal(String(from_utf16=buf), expected)
    assert_equal(String(from_utf16=buf, errors="replace"), expected)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
