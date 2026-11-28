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
# Verifies that `factorial` aborts for negative and overflowing arguments
# instead of reading past its lookup table.
#
# ===----------------------------------------------------------------------=== #

from std.math import factorial
from std.testing import _assert_aborts, TestSuite


def test_negative_aborts() raises:
    def trigger() raises -> None:
        _ = factorial(-1)

    _assert_aborts(trigger, contains="input value causes an overflow")


def test_overflow_aborts() raises:
    def trigger() raises -> None:
        _ = factorial(21)

    _assert_aborts(trigger, contains="input value causes an overflow")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
