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
    assert_equal(c1._codepoint, 32)

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


def main():
    test_char_validity()
