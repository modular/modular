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

from std.memory.alloc import alloc, dealloc, Layout, ThinAllocation
from std.sys import align_of, size_of

from std.testing import (
    _assert_aborts,
    TestSuite,
)


def test_alloc_zst_count_negative_fails() raises:
    comptime ZST = Array[Int, 0]
    comptime assert (
        size_of[ZST]() == 0
    ), "Please find a ZST to use for this test."

    def trigger() raises -> None:
        var layout = Layout[ZST](count=-1)
        var ptr = alloc(layout).unsafe_leak()
        assert_equal(0, len(ptr[]))
        dealloc(ThinAllocation(unsafe_owned_ptr=ptr).unsafe_with_layout(layout))

    _assert_aborts(
        trigger,
        contains="alloc: `Layout.count()` must be > 0",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
