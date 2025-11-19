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

# DOC: mojo/docs/manual/layout/tensors.mdx

from layout import (
    IntTuple,
    Layout,
    LayoutTensor,
    print_layout,
    UNKNOWN_VALUE,
    RuntimeLayout,
    RuntimeTuple,
)
from math import ceildiv
from collections import Set, InlineArray

from layout.layout_tensor import LayoutTensorIter, _compute_distribute_layout
from layout.layout import (
    tile_to_shape,
    blocked_product,
    crd2idx,
    idx2crd,
    coalesce,
)
from layout.int_tuple import flatten
from memory import UnsafePointer, memset
from testing import assert_equal
from utils import Index, IndexList


def accessing_tensor_elements_example():
    alias rows = 4
    alias columns = 8
    alias layout = Layout.row_major(rows, columns)
    var storage = InlineArray[Float32, rows * columns](uninitialized=True)
    for i in range(rows * columns):
        storage[i] = i
    var tensor = LayoutTensor[DType.float32, layout](storage)

    var row, col = 0, 1
    # start-access-example-1
    var element = tensor[row, col][
        0
    ]  # element is guaranteed to be a scalar value
    # end-access-example-1

    assert_equal(element, 1)
    row, col = 0, 0

    # start-access-example-2
    var elements = tensor.load[4](row, col)
    elements = elements * 2
    tensor.store(row, col, elements)
    # end-access-example-2

    element = tensor[0, 2][0]
    assert_equal(element, 4)


def accessing_nested_tensor_elements_example():
    alias rows = 4
    alias columns = 6
    alias tiler = Layout.row_major(2, 3)
    alias layout = blocked_product(Layout.col_major(2, 2), tiler)
    var storage = InlineArray[Float32, rows * columns](uninitialized=True)
    for i in range(rows * columns):
        storage[i] = i
    var tensor = LayoutTensor[DType.float32, layout](storage)

    # start-access-nested-tensor-example
    var element = tensor[1, 0, 0, 1][0]
    # end-access-nested-tensor-example

    assert_equal(element, 5)


def layout_tensor_on_cpu_example():
    # start-layout-tensor-on-cpu-example
    alias rows = 8
    alias columns = 16
    alias layout = Layout.row_major(rows, columns)
    var storage = InlineArray[Float32, rows * columns](uninitialized=True)
    var tensor = LayoutTensor[DType.float32, layout](storage)
    # end-layout-tensor-on-cpu-example
    assert_equal(tensor.size(), rows * columns)
    _ = tensor


def layout_tensor_from_pointer_example():
    # start-layout-tensor-from-pointer-example
    alias rows = 1024
    alias columns = 1024
    alias buf_size = rows * columns
    alias layout = Layout.row_major(rows, columns)
    var ptr = alloc[Float32](buf_size)
    memset(ptr, 0, buf_size)
    var tensor = LayoutTensor[DType.float32, layout](ptr)
    # end-layout-tensor-from-pointer-example
    assert_equal(tensor.size(), rows * columns)
    _ = tensor


def layout_tensor_tile_example():
    # start-layout-tensor-tile-example
    alias rows = 2
    alias columns = 4
    alias tile_size = 32
    alias tile_layout = Layout.row_major(tile_size, tile_size)
    alias tiler_layout = Layout.row_major(rows, columns)
    alias tiled_layout = blocked_product(tile_layout, tiler_layout)
    var storage = InlineArray[Float32, tiled_layout.size()](uninitialized=True)
    for i in range(tiled_layout.size()):
        storage[i] = i
    var tensor = LayoutTensor[DType.float32, tiled_layout](storage)
    var tile = tensor.tile[32, 32](0, 1)
    # end-layout-tensor-tile-example
    assert_equal(tile[0, 0][0], Float32(tile_size * tile_size))
    # start-layout-tensor-tile-example-2
    var my_tile: tensor.TileType[tile_size, tile_size]
    for i in range(rows):
        for j in range(columns):
            my_tile = tensor.tile[tile_size, tile_size](i, j)
            # ... do something with the tile ...
            # end-layout-tensor-tile-example-2
            _ = my_tile


# Iterates through a block of memory one tile at a time.
# This essentially treats the memory as a flat array of
# tiles (or a 2D row-major matrix of tiles).
def layout_tensor_iterator_example():
    # start-layout-tensor-iterator-example-1
    alias buf_size = 128
    var storage = InlineArray[Int16, buf_size](uninitialized=True)
    for i in range(buf_size):
        storage[i] = i
    alias tile_layout = Layout.row_major(4, 4)
    var iter = LayoutTensorIter[DType.int16, tile_layout, MutAnyOrigin](
        storage.unsafe_ptr(), buf_size
    )

    for i in range(ceildiv(buf_size, tile_layout.size())):
        var tile = iter[]
        # ... do something with tile
        iter += 1
        # end-layout-tensor-iterator-example-1
        assert_equal(tile[0, 0][0], i * tile_layout.size())


def layout_tensor_iterator_example2():
    # TODO: set up a tiled layout tensor as input and
    # validate output.
    alias rows = 4
    alias cols = 8
    alias size = rows * cols
    alias tile_size = 2
    var storage = InlineArray[Int32, size](uninitialized=True)
    for i in range(size):
        storage[i] = Int32(i)

    alias layout = Layout.row_major(rows, cols)
    var tensor = LayoutTensor[DType.int32, layout, masked=True](storage)
    # start-layout-tensor-iterator-example-2
    # given a tensor of size rows x cols
    alias num_row_tiles = ceildiv(rows, tile_size)
    alias num_col_tiles = ceildiv(cols, tile_size)

    for i in range(num_row_tiles):
        var iter = tensor.tiled_iterator[tile_size, tile_size, axis=1](i, 0)

        for _ in range(num_col_tiles):
            var tile = iter[]
            # ... do something with the tile
            iter += 1
            # end-layout-tensor-iterator-example-2
            _ = tile


def main():
    accessing_tensor_elements_example()
    accessing_nested_tensor_elements_example()
    layout_tensor_on_cpu_example()
    layout_tensor_from_pointer_example()
    layout_tensor_tile_example()
    layout_tensor_iterator_example()
    layout_tensor_iterator_example2()
