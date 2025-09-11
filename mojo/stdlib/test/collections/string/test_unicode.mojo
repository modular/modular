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

from collections.string._unicode import _get_uppercase_mapping

from testing import assert_equal, assert_raises

alias MAX_UNICODE_VALUE = 0x10FFFF


def test_uppercase_conversion():
    # a -> A
    count1, chars1 = _get_uppercase_mapping(Codepoint(97)).value()
    assert_equal(count1, 1)
    assert_equal(chars1[0], Codepoint(65))
    assert_equal(chars1[1], Codepoint(0))
    assert_equal(chars1[2], Codepoint(0))

    # ß -> SS
    count2, chars2 = _get_uppercase_mapping(
        Codepoint.from_u32(0xDF).value()
    ).value()
    assert_equal(count2, 2)
    assert_equal(chars2[0], Codepoint.from_u32(0x53).value())
    assert_equal(chars2[1], Codepoint.from_u32(0x53).value())
    assert_equal(chars2[2], Codepoint(0))

    # ΐ -> Ϊ́
    count3, chars3 = _get_uppercase_mapping(
        Codepoint.from_u32(0x390).value()
    ).value()
    assert_equal(count3, 3)
    assert_equal(chars3[0], Codepoint.from_u32(0x0399).value())
    assert_equal(chars3[1], Codepoint.from_u32(0x0308).value())
    assert_equal(chars3[2], Codepoint.from_u32(0x0301).value())


def test_codepoint_parsing():
    # ASCII
    for i in range(0x7F + 1):
        alias L = List[UInt8]
        assert_equal(String(from_codepoints=L(i)), chr(i))
        assert_equal(String(from_codepoints=L(i), errors="replace"), chr(i))

    # ISO-8859-1 aka. Latin-1
    for i in range(0x7F + 1, 0xFF + 1):
        alias L = List[UInt8]
        assert_equal(String(from_codepoints=L(i)), chr(i))
        assert_equal(String(from_codepoints=L(i), errors="replace"), chr(i))

    # UInt16 range
    for i in range(0xFF + 1, UInt16.MAX):
        alias L = List[UInt16]
        if 0xD800 <= i <= 0xDFFF:  # UTF-16 surrogate pairs
            with assert_raises():
                _ = String(from_codepoints=L(i))
            assert_equal(String(from_codepoints=L(i), errors="replace"), "�")
        else:
            assert_equal(String(from_codepoints=L(i)), chr(i))
            assert_equal(String(from_codepoints=L(i), errors="replace"), chr(i))

    # UInt32 range
    for i in range(0xFF + 1, MAX_UNICODE_VALUE + 1):
        alias L = List[UInt32]
        assert_equal(String(from_codepoints=L(i)), chr(i))
        assert_equal(String(from_codepoints=L(i), errors="replace"), chr(i))

    # invalid values
    for i in range(MAX_UNICODE_VALUE + 1, UInt256.MAX):
        with assert_raises():
            _ = String(from_codepoints=List[UInt256](i))
        assert_equal(
            String(from_codepoints=List[UInt256](i), errors="replace"), "�"
        )


def main():
    test_uppercase_conversion()
