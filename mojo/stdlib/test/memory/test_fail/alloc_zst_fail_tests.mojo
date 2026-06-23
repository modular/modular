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

# RUN: not %mojo -Dtest=1 %s 2>&1 | FileCheck --check-prefix CHECK_1 %s
# RUN: not %mojo -Dtest=2 %s 2>&1 | FileCheck --check-prefix CHECK_2 %s

from std.memory.alloc import alloc, dealloc, Layout, ThinAllocation
from std.sys import align_of, size_of
from std.sys.defines import get_defined_int

from std.testing import (
    assert_equal,
    assert_not_equal,
    assert_false,
    assert_true,
    TestSuite,
)


def test_alloc_zst_count_zero_fails() raises:
    comptime ZST = InlineArray[Int, 0]
    comptime assert (
        size_of[ZST]() == 0
    ), "Please find a ZST to use for this test."

    # CHECK_1: Assert Error: alloc(Layout[std.collections.inline_array.InlineArray[SIMD[DType.int, 1], 0 : SIMD[DType.int, 1]]](count=0, alignment=8)): count must be > 0
    var layout = Layout[ZST](count=0)
    var ptr = alloc(layout).unsafe_leak()

    # CHECK_1-NOT: is never reached
    assert_equal(0, len(ptr[]))

    dealloc(
        ThinAllocation(unsafe_assume_ownership=ptr).unsafe_with_layout(layout)
    )


def test_alloc_zst_count_negative_fails() raises:
    comptime ZST = InlineArray[Int, 0]
    comptime assert (
        size_of[ZST]() == 0
    ), "Please find a ZST to use for this test."

    # CHECK_2: Assert Error: alloc(Layout[std.collections.inline_array.InlineArray[SIMD[DType.int, 1], 0 : SIMD[DType.int, 1]]](count=-1, alignment=8)): count must be > 0
    var layout = Layout[ZST](count=-1)
    var ptr = alloc(layout).unsafe_leak()

    # CHECK_2-NOT: is never reached
    assert_equal(0, len(ptr[]))

    dealloc(
        ThinAllocation(unsafe_assume_ownership=ptr).unsafe_with_layout(layout)
    )


def main() raises:
    comptime test = get_defined_int["test"]()
    comptime if test == 1:
        test_alloc_zst_count_zero_fails()
    elif test == 2:
        test_alloc_zst_count_negative_fails()
    else:
        raise t"Invalid `test` value {test}"
