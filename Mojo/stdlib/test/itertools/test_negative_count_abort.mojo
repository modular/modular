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
#
# Verifies that `take` and `drop` abort when given a negative `count`.
#
# ===----------------------------------------------------------------------=== #

from std.itertools import drop, take
from std.testing import _assert_aborts, TestSuite


def test_take_negative_count() raises:
    var nums: List = [1, 2, 3]

    def trigger() raises {nums} -> None:
        _ = take(nums, -1)

    _assert_aborts(
        trigger, contains="The `count` argument must be non-negative"
    )


def test_drop_negative_count() raises:
    var nums: List = [1, 2, 3]

    def trigger() raises {nums} -> None:
        _ = drop(nums, -1)

    _assert_aborts(
        trigger, contains="The `count` argument must be non-negative"
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
