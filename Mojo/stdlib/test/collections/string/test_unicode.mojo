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

from std.collections.string._unicode import (
    _CaseMapping,
    _get_lowercase_mapping,
    _get_uppercase_mapping,
)

from std.testing import assert_equal
from std.testing import TestSuite


def test_uppercase_conversion() raises:
    # a -> A
    assert_equal(
        _get_uppercase_mapping(Codepoint(97)),
        _CaseMapping.single([Codepoint(65)]),
    )

    # ß -> SS
    assert_equal(
        _get_uppercase_mapping(Codepoint.from_u32(0xDF).value()),
        _CaseMapping.pair(
            [Codepoint.from_u32(0x53).value(), Codepoint.from_u32(0x53).value()]
        ),
    )

    # ΐ -> Ϊ́
    assert_equal(
        _get_uppercase_mapping(Codepoint.from_u32(0x390).value()),
        _CaseMapping.triple(
            [
                Codepoint.from_u32(0x0399).value(),
                Codepoint.from_u32(0x0308).value(),
                Codepoint.from_u32(0x0301).value(),
            ]
        ),
    )


def test_lowercase_conversion() raises:
    # A -> a
    assert_equal(
        _get_lowercase_mapping(Codepoint(65)),
        _CaseMapping.single([Codepoint(97)]),
    )

    # U+0130 -> "i" followed by U+0307 COMBINING DOT ABOVE
    assert_equal(
        _get_lowercase_mapping(Codepoint.from_u32(0x130).value()),
        _CaseMapping.pair([Codepoint(105), Codepoint.from_u32(0x0307).value()]),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
