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

from std.testing import _assert_aborts, TestSuite


def test_pow_negative_exponent_aborts() raises:
    # A runtime integer exponent's sign is not known until it is evaluated, so
    # the guard is a trap rather than a compile error.
    var a: Int = 2
    var b: Int = -1

    def trigger() raises {a, b} -> None:
        _ = a**b

    _assert_aborts(
        trigger, contains="an integer cannot be raised to a negative power"
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
