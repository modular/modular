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

from testing import assert_equal
from layout._mixed_layout import MixedLayout
from layout._mixed_tuple import Idx
from layout.int_tuple import IntTuple


fn main() raises:
    # Row-major 3x4: last element (2,3) -> 11, cosize = 12
    var layout1 = MixedLayout(
        shape=[Idx[3](), Idx[4]()], stride=[Idx[4](), Idx[1]()]
    )
    assert_equal(layout1.size(), 12)
    assert_equal(layout1.cosize(), 12)

    # Layout with gaps: last element (1,1) -> 11, cosize = 12
    var layout2 = MixedLayout(
        shape=[Idx[2](), Idx[2]()], stride=[Idx[10](), Idx[1]()]
    )
    assert_equal(layout2.size(), 4)
    assert_equal(layout2.cosize(), 12)

    var layout = MixedLayout(
        shape=[Idx[4](), Idx[2]()], stride=[Idx[1](), Idx[4]()]
    )

    # Multi-dimensional coordinates
    assert_equal(layout(IntTuple(0, 0)), 0)
    assert_equal(layout(IntTuple(1, 0)), 1)
    assert_equal(layout(IntTuple(2, 0)), 2)
    assert_equal(layout(IntTuple(3, 0)), 3)
    assert_equal(layout(IntTuple(0, 1)), 4)
    assert_equal(layout(IntTuple(1, 1)), 5)
    assert_equal(layout(IntTuple(2, 1)), 6)
    assert_equal(layout(IntTuple(3, 1)), 7)

    assert_equal(layout.size(), 8)
