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

from layout._mixed_layout import MixedLayout, row_major
from layout._mixed_layout_tensor import (
    MixedLayoutTensor,
    MixedLayoutTensorIter,
    distribute,
    tile,
)
from layout._mixed_tuple import ComptimeInt, Idx, MixedTuple, RuntimeInt
from layout.int_tuple import IntTuple
from math import ceildiv
from testing import TestSuite, assert_equal, assert_false, assert_true


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()


fn test_distribute() raises:
    comptime thread_layout = row_major((Idx[2](), Idx[2]()))

    var array = InlineArray[UInt32, 16](fill=-1)
    var ptr = array.unsafe_ptr()

    comptime data_layout_shape = MixedTuple[ComptimeInt[4], ComptimeInt[4]]
    comptime data_layout_stride = MixedTuple[ComptimeInt[4], ComptimeInt[1]]
    var layout_tensor = MixedLayoutTensor[dtype = DType.uint32](
        ptr=ptr,
        layout=MixedLayout(
            shape=data_layout_shape(Idx[4](), Idx[4]()),
            stride=data_layout_stride(Idx[4](), Idx[1]()),
        ),
    )

    var counter = 0
    for th_id in range(4):
        var frag = distribute[
            dtype = DType.uint32,
            thread_layout=thread_layout,
        ](layout_tensor, th_id)

        # Fill the fragment positions with the thread id (0..3)
        for i in range(2):
            for j in range(2):
                frag[(Idx(i), Idx(j))] = counter
                counter += 1

    var expected = [0, 4, 1, 5, 8, 12, 9, 13, 2, 6, 3, 7, 10, 14, 11, 15]
    for i in range(16):
        assert_equal(ptr[i], expected[i])


fn test_tile() raises:
    # Create a 4x4 tensor with row-major layout
    var data = InlineArray[UInt32, 16](fill=0)

    var layout_tensor = MixedLayoutTensor[dtype = DType.uint32](
        data, row_major((Idx[4](), Idx[4]()))
    )

    var counter = 0

    @parameter
    for tile_i in range(2):

        @parameter
        for tile_j in range(2):
            var current_tile = tile(
                layout_tensor,
                tile_shape=(Idx[2](), Idx[2]()),
                tile_coords=(Idx(tile_i), Idx(tile_j)),
            )

            for i in range(2):
                for j in range(2):
                    current_tile[(Idx(i), Idx(j))] = counter
                    counter += 1

    # Expected layout after tiling:
    # Tile (0,0): values 0,1,2,3   -> positions [0,1], [4,5]
    # Tile (0,1): values 4,5,6,7   -> positions [2,3], [6,7]
    # Tile (1,0): values 8,9,10,11 -> positions [8,9], [12,13]
    # Tile (1,1): values 12,13,14,15 -> positions [10,11], [14,15]
    var expected = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]
    for i in range(16):
        assert_equal(data[i], expected[i])


def test_tensor_span_constructor():
    var bytes: List[UInt8] = [0, 1, 2, 3]
    var _tensor = MixedLayoutTensor(
        bytes,
        row_major((Idx(2), Idx[2]())),
    )


def test_layout_tensor_iterator():
    comptime buf_size = 16
    var storage = InlineArray[Int16, buf_size](uninitialized=True)
    for i in range(buf_size):
        storage[i] = i
    var tile_layout = row_major((Idx[2](), Idx[2]()))
    var iter = MixedLayoutTensorIter(storage, tile_layout)

    var tile = next(iter)
    assert_equal(tile[(Idx(0), Idx(0))], 0)
    assert_equal(tile[(Idx(0), Idx(1))], 1)
    assert_equal(tile[(Idx(1), Idx(0))], 2)
    assert_equal(tile[(Idx(1), Idx(1))], 3)
    assert_equal(tile.layout.size(), 4)
    assert_true(iter.__has_next__())
    tile = next(iter)
    assert_equal(tile[(Idx(0), Idx(0))], 4)
    assert_equal(tile[(Idx(0), Idx(1))], 5)
    assert_equal(tile[(Idx(1), Idx(0))], 6)
    assert_equal(tile[(Idx(1), Idx(1))], 7)
    assert_equal(tile.layout.size(), 4)
    assert_true(iter.__has_next__())
    tile = next(iter)
    assert_equal(tile[(Idx(0), Idx(0))], 8)
    assert_equal(tile[(Idx(0), Idx(1))], 9)
    assert_equal(tile[(Idx(1), Idx(0))], 10)
    assert_equal(tile[(Idx(1), Idx(1))], 11)
    assert_equal(tile.layout.size(), 4)
    assert_true(iter.__has_next__())
    tile = next(iter)
    assert_equal(tile[(Idx(0), Idx(0))], 12)
    assert_equal(tile[(Idx(0), Idx(1))], 13)
    assert_equal(tile[(Idx(1), Idx(0))], 14)
    assert_equal(tile[(Idx(1), Idx(1))], 15)
    assert_equal(tile.layout.size(), 4)
    assert_false(iter.__has_next__())
