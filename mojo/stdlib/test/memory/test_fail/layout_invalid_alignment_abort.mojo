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

from std.memory import Layout


def test_alignment_too_small() raises:
    var too_small = 2

    def trigger() raises {too_small} -> None:
        var _layout = Layout[Int32](count=1, alignment=too_small)

    _assert_aborts(
        trigger,
        contains="Alignment is invalid. Must be a power of two and >= to the types natural alignment.",
    )


def test_alignment_not_pow_of_2() raises:
    var not_pow_of_2 = 127

    def trigger() raises {not_pow_of_2} -> None:
        var _layout = Layout[Int32](count=1, alignment=not_pow_of_2)

    _assert_aborts(
        trigger,
        contains="Alignment is invalid. Must be a power of two and >= to the types natural alignment.",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
