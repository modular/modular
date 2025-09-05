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

from collections.string._utf16 import _decode_utf16

from testing import assert_equal, assert_false, assert_raises, assert_true

# ===----------------------------------------------------------------------=== #
# Reusable testing data
# ===----------------------------------------------------------------------=== #
# ===----------------------------------------------------------------------=== #
# Tests
# ===----------------------------------------------------------------------=== #


def test_utf16_parsing():
    alias L = List[UInt16]

    assert_equal(_decode_utf16(from_utf16=L(65, 97, 33, 945, 10175)), "Aa!α➿")
    assert_equal(_decode_utf16(from_utf16=L(0x20AC)), "€")
    assert_equal(_decode_utf16(from_utf16=L(0xFFFD)), "�")
    assert_equal(_decode_utf16(from_utf16=L(0xD83D, 0xDD25)), "🔥")
    assert_equal(_decode_utf16(from_utf16=L(0xD801, 0xDC37)), chr(0x10437))
    assert_equal(_decode_utf16(from_utf16=L(0xD852, 0xDF62)), chr(0x24B62))

    for i in range(0xD800):
        assert_equal(String(from_utf16=L(i)), chr(i))
        assert_equal(String(from_utf16=L(i), errors="replace"), chr(i))

    # test unpaired surrogates
    for i in range(0xD800, 0xDFFF + 1):
        with assert_raises():
            _ = String(from_utf16=L(i))
        assert_equal(String(from_utf16=L(i), errors="replace"), "�")

    for i in range(0xDFFF + 1, UInt16.MAX + 1):
        assert_equal(String(from_utf16=L(i)), chr(i))
        assert_equal(String(from_utf16=L(i), errors="replace"), chr(i))

    # test invalid ranges
    for i in range(UInt16.MAX + 1, UInt256.MAX):
        with assert_raises():
            _ = String(from_utf16=List[UInt256](i))
        assert_equal(String(from_utf16=List[UInt256](i), errors="replace"), "�")


def main():
    test_utf16_parsing()
