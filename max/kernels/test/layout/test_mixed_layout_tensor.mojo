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
from layout._mixed_layout_tensor import MixedLayoutTensor, distribute, tile
from layout._mixed_tuple import ComptimeInt, Idx, MixedTuple, RuntimeInt
from layout.int_tuple import IntTuple
from testing import TestSuite, assert_equal, assert_true


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
