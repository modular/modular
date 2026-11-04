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

from std.memory.alloc import Layout
from std.sys import size_of
from std.testing import _assert_aborts, TestSuite


def test_alloc_zst_count_negative_fails() raises:
    comptime ZST = Array[Int, 0]
    comptime assert (
        size_of[ZST]() == 0
    ), "Please find a ZST to use for this test."

    def trigger() raises -> None:
        # Layout.__init__ aborts when length is negative.
        _ = Layout[ZST](length=-1)

    _assert_aborts(
        trigger,
        contains="Layout: `length` must be >= 0",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
