# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from layout import *
from layout.layout import Layout
from layout.swizzle import ComposedLayout, Swizzle
from testing import assert_equal


fn test_composed_layout() raises:
    print("== test_composed_layout")

    alias layout_a = Layout(IntTuple(6, 2), IntTuple(8, 2))
    alias layout_b = Layout(IntTuple(4, 3), IntTuple(3, 1))

    alias comp_layout = ComposedLayout[Layout, Layout, 0](layout_b, layout_a)

    assert_equal(comp_layout(0), 0)
    assert_equal(comp_layout(1), 24)
    assert_equal(comp_layout(2), 2)
    assert_equal(comp_layout(3), 26)
    assert_equal(comp_layout(4), 8)
    assert_equal(comp_layout(5), 32)
    assert_equal(comp_layout(6), 10)
    assert_equal(comp_layout(7), 34)
    assert_equal(comp_layout(8), 16)
    assert_equal(comp_layout(9), 40)
    assert_equal(comp_layout(10), 18)
    assert_equal(comp_layout(11), 42)


fn test_composed_layout_swizzle() raises:
    print("== test_composed_layout_swizzle")

    var swizzle = Swizzle(1, 0, 2)
    var layout = Layout(IntTuple(6, 2), IntTuple(8, 2))

    var comp_layout = ComposedLayout[Layout, Swizzle, 0](layout, swizzle)

    assert_equal(comp_layout(0), 0)
    assert_equal(comp_layout(1), 8)
    assert_equal(comp_layout(2), 16)
    assert_equal(comp_layout(3), 24)
    assert_equal(comp_layout(4), 32)
    assert_equal(comp_layout(5), 40)


fn test_composed_layout_swizzle_rt() raises:
    print("== test_composed_layout_swizzle_rt")

    var swizzle = Swizzle(1, 0, 2)
    var layout = Layout(IntTuple(6, 2), IntTuple(8, 2))

    var comp_layout = ComposedLayout[Layout, Swizzle, 0](layout, swizzle)

    assert_equal(comp_layout(0), 0)
    assert_equal(comp_layout(1), 8)
    assert_equal(comp_layout(2), 16)
    assert_equal(comp_layout(3), 24)
    assert_equal(comp_layout(4), 32)
    assert_equal(comp_layout(5), 40)


def main():
    test_composed_layout()
    test_composed_layout_swizzle()
    test_composed_layout_swizzle_rt()
