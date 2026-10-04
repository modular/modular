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
"""Exercises runtime `__match` on `TraceCategory` and `TraceLevel`.

Each helper names every case so the match fails to compile if
`_enum_case_names` drifts from the comptime case constants.
"""

from std.testing import TestSuite, assert_equal

from max.runtime.tracing import TraceCategory, TraceLevel


def _category_rank(category: TraceCategory) -> Int:
    __match category:
        case .OTHER:
            return 0
        case .ASYNCRT:
            return 1
        case .MEM:
            return 2
        case .Kernel:
            return 3
        case .MAX:
            return 4


def _level_rank(level: TraceLevel) -> Int:
    __match level:
        case .ALWAYS:
            return 0
        case .OP:
            return 1
        case .THREAD:
            return 2


def test_category_match() raises:
    var categories = List[TraceCategory]()
    categories.append(TraceCategory.OTHER)
    categories.append(TraceCategory.ASYNCRT)
    categories.append(TraceCategory.MEM)
    categories.append(TraceCategory.Kernel)
    categories.append(TraceCategory.MAX)

    for i in range(len(categories)):
        assert_equal(
            _category_rank(categories[i]), i, "wrong TraceCategory case matched"
        )


def test_level_match() raises:
    var levels = List[TraceLevel]()
    levels.append(TraceLevel.ALWAYS)
    levels.append(TraceLevel.OP)
    levels.append(TraceLevel.THREAD)

    for i in range(len(levels)):
        assert_equal(_level_rank(levels[i]), i, "wrong TraceLevel case matched")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
