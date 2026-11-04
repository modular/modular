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

from test_utils import CopyCounter
from std.testing import TestSuite, assert_equal, assert_raises


def test_filter() raises:
    var l = [1, 2, 3, 4, 5]

    def is_odd(x: Int) -> Bool:
        return x % 2 == 1

    var f = filter(l, is_odd)
    assert_equal(next(f), 1)
    assert_equal(next(f), 3)
    assert_equal(next(f), 5)
    with assert_raises():
        _ = f.__next__()  # raises StopIteration


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
