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

from layout import TileTensor, Idx, Coord, coord
from std.math import ceildiv
from std.collections import InlineArray

from layout.tile_layout import row_major, col_major, blocked_product
from std.memory import UnsafePointer, alloc, memset
from std.testing import assert_equal


def accessing_tensor_elements_example() raises:
    comptime rows = 4
    comptime columns = 8
    comptime layout = row_major[rows, columns]()
    var storage = InlineArray[Float32, rows * columns](uninitialized=True)
    for i in range(rows * columns):
        storage[i] = Float32(i)
    var tensor = TileTensor(storage, layout)

    var row, col = 0, 1
    # start-access-example-1
    var element = tensor[row, col]
    # element is guaranteed to be a scalar value
    # end-access-example-1

    assert_equal(element, 1)

    # start-access-example-3
    var elements = tensor.load[4]((Idx(row), Idx(col)))
    elements = elements * 2
    tensor.store((Idx(row), Idx(col)), elements)
    # end-access-example-3

    element = tensor[0, 2][0]
    assert_equal(element, 4)


# start-scalar-tensor-example
def takes_scalar_tensor(
    tensor: TileTensor[DType.int32, element_size=1, ...]
) -> Int:
    comptime assert tensor.flat_rank == 2
    return Int(tensor[1, 1])


# end-scalar-tensor-example


def test_scalar_tensor() raises:
    var layout = row_major[4, 4]()
    var num_elements = layout.static_product
    var list = List[Int32](capacity=num_elements)
    for i in range(num_elements):
        list.append(Int32(i))
    var t = TileTensor(list, layout)
    var v = takes_scalar_tensor(t)
    assert_equal(v, 5)


def accessing_nested_tensor_elements_example() raises:
    comptime rows = 4
    comptime columns = 6
    comptime tiler = row_major[2, 3]()
    comptime layout = blocked_product(col_major[2, 2](), tiler)
    var storage = InlineArray[Float32, rows * columns * 2](uninitialized=True)
    for i in range(rows * columns):
        storage[i] = Float32(i)
    var tensor = TileTensor(storage, layout)

    # start-access-nested-tensor-example
    # start-access-nested-coord-example
    var el1 = tensor[Coord(Coord(Idx(1), Idx(0)), Coord(Idx(0), Idx(1)))]
    # end-access-nested-coord-example
    # start-access-nested-flat-example
    var el2 = tensor[1, 0, 0, 1]
    # end-access-nested-flat-example
    var el3 = tensor.load[1](
        Coord(Coord(Idx(1), Idx(0)), Coord(Idx(0), Idx(1)))
    )
    # end-access-nested-tensor-example
    print(el1, el2, el3)

    var test_conversion: Int = Int(el2)
    assert_equal(el1, 5)
    _ = test_conversion


def tile_tensor_on_cpu_example() raises:
    # start-layout-tensor-on-cpu-example
    comptime rows = 8
    comptime columns = 16
    comptime layout = row_major[rows, columns]()
    var storage = InlineArray[Float32, rows * columns](fill=0.0)
    var tensor = TileTensor(storage, layout)
    # end-layout-tensor-on-cpu-example
    assert_equal(tensor.num_elements(), rows * columns)
    _ = tensor


def tile_tensor_from_list_example() raises:
    # start-layout-tensor-from-pointer-example
    comptime rows = 1024
    comptime columns = 1024
    comptime buf_size = rows * columns
    comptime layout = row_major[rows, columns]()
    var storage = List[Float32](length=buf_size, fill=0.0)
    var tensor = TileTensor(storage, layout)
    # end-layout-tensor-from-pointer-example
    assert_equal(tensor.num_elements(), rows * columns)
    _ = tensor
    _ = storage^


def tile_tensor_tile_example() raises:
    # start-layout-tensor-tile-example
    comptime tile_size = 32
    comptime rows = 64
    comptime cols = 128
    comptime layout = row_major[rows, cols]()
    var storage = List[Float32](capacity=rows * cols)
    for i in range(rows * cols):
        storage.append(Float32(i))
    var tensor = TileTensor(storage, layout)
    var tile = tensor.tile[tile_size, tile_size](0, 1)
    # end-layout-tensor-tile-example
    assert_equal(tile[0, 0][0], Float32(tile_size))


# Iterates through a block of memory one tile at a time.
# This essentially treats the memory as a flat array of
# tiles (or a 2D row-major matrix of tiles).
def layout_tensor_iterator_example() raises:
    # start-layout-tensor-iterator-example-1
    comptime buf_size = 128
    var storage = InlineArray[Int16, buf_size](uninitialized=True)
    for i in range(buf_size):
        storage[i] = Int16(i)
    comptime tile_m = 4
    comptime tile_n = 4
    comptime tile_size = tile_m * tile_n
    comptime num_tiles = buf_size // tile_size
    # Interpret the flat buffer as (num_tiles * tile_m) rows × tile_n cols,
    # then step through tile_m-row chunks.
    comptime full_layout = row_major[num_tiles * tile_m, tile_n]()
    var tensor = TileTensor(storage, full_layout)

    for i in range(num_tiles):
        var tile = tensor.tile[tile_m, tile_n](i, 0)
        # ... do something with tile
        # end-layout-tensor-iterator-example-1
        assert_equal(tile[0, 0][0], Int16(i * tile_size))


def layout_tensor_iterator_example2() raises:
    comptime rows = 4
    comptime cols = 8
    comptime size = rows * cols
    comptime tile_size = 2
    var storage = InlineArray[Int32, size](uninitialized=True)
    for i in range(size):
        storage[i] = Int32(i)

    comptime layout = row_major[rows, cols]()
    var tensor = TileTensor(storage, layout)
    # start-layout-tensor-iterator-example-2
    # given a tensor of size rows x cols
    comptime num_row_tiles = ceildiv(rows, tile_size)
    comptime num_col_tiles = ceildiv(cols, tile_size)

    for i in range(num_row_tiles):
        for j in range(num_col_tiles):
            var tile = tensor.tile[tile_size, tile_size](i, j)
            # ... do something with the tile
            # end-layout-tensor-iterator-example-2
            _ = tile


def main() raises:
    accessing_tensor_elements_example()
    accessing_nested_tensor_elements_example()
    test_scalar_tensor()
    tile_tensor_on_cpu_example()
    tile_tensor_from_list_example()
    tile_tensor_tile_example()
    layout_tensor_iterator_example()
    layout_tensor_iterator_example2()
