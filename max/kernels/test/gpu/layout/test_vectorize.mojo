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

from layout import Coord, Idx, TileTensor, row_major
from layout.tile_layout import Layout as TileLayout
from layout.tile_tensor import stack_allocation
from std.memory import alloc
from std.memory.alloc import unsafe_alloc
from std.testing import assert_equal


def test_native_vectorize_static() raises:
    comptime flat_layout = TileLayout(
        Coord(Idx[16], Idx[32], Idx[4]),
        Coord(Idx[32], Idx[1], Idx[512]),
    )
    var tensor = stack_allocation[dtype=.float32, alignment=16](flat_layout)
    var storage = tensor.reshape(row_major[2048]())
    for i in range(2048):
        storage[i] = Float32(i)

    comptime nested_layout = TileLayout(
        Coord(Coord(Idx[16], Idx[32]), Idx[4]),
        Coord(Coord(Idx[32], Idx[1]), Idx[512]),
    )
    var nested = tensor.reshape(nested_layout)
    comptime assert nested.rank == 2 and nested.flat_rank == 3
    # Native vectorize takes flat scalar tile sizes, not nested tilers.
    # This leaf-equivalent view keeps the nested source's offsets unchanged.
    var fragments = nested.reshape(flat_layout).vectorize[1, 4, 1]()
    comptime assert fragments.flat_rank == 3 and fragments.element_size == 4
    comptime assert fragments.linear_idx_type == tensor.linear_idx_type
    assert_equal(fragments.dim[0](), 16)
    assert_equal(fragments.dim[1](), 8)
    assert_equal(fragments.dim[2](), 4)
    for row in range(16):
        for column in range(8):
            for plane in range(4):
                var coordinate = Coord(row, column, plane)
                var offset = row * 32 + column * 4 + plane * 512
                assert_equal(Int(fragments.layout(coordinate)), offset)
                assert_equal(
                    Int(nested.layout(Coord(Coord(row, column * 4), plane))),
                    offset,
                )
                var native_value = fragments[coordinate]
                for lane in range(4):
                    assert_equal(native_value[lane], Float32(offset + lane))

    var square = stack_allocation[dtype=.float32, alignment=8](
        row_major[8, 8]()
    )
    for i in range(64):
        square[i // 8, i % 8] = Float32(i)
    var contiguous = square.vectorize[1, 2]()
    comptime assert contiguous.flat_rank == 2 and contiguous.element_size == 2
    assert_equal(contiguous.dim[0](), 8)
    assert_equal(contiguous.dim[1](), 4)
    for row in range(8):
        for column in range(4):
            var value = contiguous[row, column]
            assert_equal(
                Int(contiguous.layout(Coord(row, column))), row * 8 + column * 2
            )
            for lane in range(2):
                assert_equal(value[lane], Float32(row * 8 + column * 2 + lane))

    # Native SIMD lanes are contiguous.
    var row_blocks = square.vectorize[2, 1]()
    var row_value = row_blocks[2, 0]
    assert_equal(row_value[0], 32.0)
    assert_equal(row_value[1], 33.0)


def test_native_vectorize_dynamic[width: Int]() raises:
    var heap = unsafe_alloc[Int32](64, alignment=16)
    var runtime_layout = TileLayout(
        Coord(Int32(8), Int32(8)), Coord(Int32(8), Idx[1])
    )
    var tensor = TileTensor[linear_idx_type=.int32](heap, runtime_layout)
    comptime assert not tensor.all_dims_known
    for i in range(64):
        tensor[i // 8, i % 8] = Int32(i)
    var fragments = tensor.vectorize[1, width]()
    comptime assert fragments.flat_rank == 2 and fragments.element_size == width
    comptime assert fragments.linear_idx_type == .int32
    assert_equal(fragments.dim[0](), 8)
    assert_equal(fragments.dim[1](), Int32(8 // width))
    assert_equal(Int(fragments.layout.stride_coord()[0]), 8)
    assert_equal(Int(fragments.layout.stride_coord()[1]), width)
    for row in range(8):
        for column in range(8 // width):
            var coordinate = Coord(row, column)
            var offset = row * 8 + column * width
            assert_equal(Int(fragments.layout(coordinate)), offset)
            var value = fragments[coordinate]
            for lane in range(width):
                assert_equal(value[lane], Int32(offset + lane))
    heap.unsafe_free()


def main() raises:
    test_native_vectorize_static()
    test_native_vectorize_dynamic[2]()
    test_native_vectorize_dynamic[4]()
