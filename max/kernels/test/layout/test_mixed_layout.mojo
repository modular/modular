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

from layout._mixed_layout import MixedLayout, make_row_major
from layout._mixed_tuple import ComptimeInt, Idx, MixedTuple, RuntimeInt
from layout.int_tuple import IntTuple
from testing import assert_equal, assert_true


def main():
    test_size_cosize()
    test_crd2idx()
    test_row_major()


fn test_size_cosize() raises:
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


fn test_crd2idx() raises:
    var layout = MixedLayout(
        shape=[Idx[4](), Idx[2]()], stride=[Idx[1](), Idx[4]()]
    )

    # Multi-dimensional coordinates
    assert_equal(layout(MixedTuple(Idx[0](), Idx[0]())), 0)
    assert_equal(layout(MixedTuple(Idx[1](), Idx[0]())), 1)
    assert_equal(layout(MixedTuple(Idx[2](), Idx[0]())), 2)
    assert_equal(layout(MixedTuple(Idx[3](), Idx[0]())), 3)
    assert_equal(layout(MixedTuple(Idx[0](), Idx[1]())), 4)
    assert_equal(layout(MixedTuple(Idx[1](), Idx[1]())), 5)
    assert_equal(layout(MixedTuple(Idx[2](), Idx[1]())), 6)
    assert_equal(layout(MixedTuple(Idx[3](), Idx[1]())), 7)

    assert_equal(layout.size(), 8)


fn test_row_major() raises:
    var shape2 = MixedTuple(Idx[3](), Idx(4))
    var layout2 = make_row_major(shape2)
    assert_true(layout2.shape == shape2)
    assert_true(layout2.stride == MixedTuple(Idx(4), Idx[1]()))

    var shape3 = MixedTuple(Idx[3](), Idx(4), Idx(5))
    var layout3 = make_row_major(shape3)
    assert_true(layout3.shape == shape3)
    assert_true(layout3.stride == MixedTuple(Idx(20), Idx(5), Idx[1]()))

    var shape3_static = MixedTuple(
        ComptimeInt[3](), ComptimeInt[4](), ComptimeInt[5]()
    )
    var layout3_static = make_row_major[Second=4, Third=5](shape3_static)
    assert_true(layout3_static.shape == shape3_static)
    assert_true(
        layout3_static.stride
        == MixedTuple(ComptimeInt[20](), ComptimeInt[5](), ComptimeInt[1]())
    )
