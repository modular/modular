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

from layout import *
from layout.int_tuple import product
from layout.layout_tensor import *
from layout.tile_layout import Layout as TileLayout
from std.testing import assert_equal


def test_vectorize_2() raises:
    var tensor = LayoutTensor[
        .float32,
        Layout(IntTuple(IntTuple(16, 32), 4), IntTuple(IntTuple(32, 1), 512)),
        MutAnyOrigin,
    ].stack_allocation[stack_alignment=16]()

    var n = product(tensor.layout.shape)
    for i in range(n):
        tensor.ptr[i] = Float32(i)

    var frag = tensor._vectorize_2[origin_of(), IntTuple(IntTuple(1, 4), 1)]()
    var crd = RuntimeTuple[IntTuple(2)]()
    var val = frag[crd]
    assert_equal(val[0], 64)
    assert_equal(val[1], 65)
    assert_equal(val[2], 66)
    assert_equal(val[3], 67)

    var frag_linear = tensor._vectorize_2[4]()
    var val_linear = frag_linear[crd]
    assert_equal(val_linear[0], 64)
    assert_equal(val_linear[1], 65)
    assert_equal(val_linear[2], 66)
    assert_equal(val_linear[3], 67)

    var three_dim_tensor = LayoutTensor[
        .float32,
        Layout(IntTuple(16, 32, 4), IntTuple(32, 1, 512)),
        MutAnyOrigin,
    ].stack_allocation[stack_alignment=16]()

    n = product(three_dim_tensor.layout.shape)
    for i in range(n):
        three_dim_tensor.ptr[i] = Float32(i)

    var frag_3dt = three_dim_tensor._vectorize_2[
        origin_of(), IntTuple(1, 4, 1)
    ]()
    var val_3dt = frag_3dt[crd]
    assert_equal(val_3dt[0], 64)
    assert_equal(val_3dt[1], 65)
    assert_equal(val_3dt[2], 66)
    assert_equal(val_3dt[3], 67)

    var frag_linear_3dt = three_dim_tensor._vectorize_2[4]()
    var val_linear_3dt = frag_linear_3dt[crd]
    assert_equal(val_linear_3dt[0], 64)
    assert_equal(val_linear_3dt[1], 65)
    assert_equal(val_linear_3dt[2], 66)
    assert_equal(val_linear_3dt[3], 67)

    comptime layout = Layout(IntTuple(8, 8), IntTuple(8, 1))
    var tensor2 = LayoutTensor[
        .float32,
        layout,
        MutAnyOrigin,
    ].stack_allocation[stack_alignment=8]()

    n = product(tensor2.layout.shape)
    for i in range(n):
        tensor2.ptr[i] = Float32(i)

    var frag2 = tensor2.vectorize[2]()
    var val2 = frag2[crd]
    assert_equal(val2[0], 32.0)
    assert_equal(val2[1], 40.0)

    n = product(tensor2.layout.shape)
    for i in range(n):
        tensor2.ptr[i] = Float32(i)

    var frag3 = tensor2._vectorize_2[2]()
    var val3 = frag3[crd]
    assert_equal(val3[0], 16.0)
    assert_equal(val3[1], 17.0)

    comptime layout_unknown = Layout(
        IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE), IntTuple(UNKNOWN_VALUE, 1)
    )
    var heap = alloc[Int32](64, alignment=8)
    var tensor4 = LayoutTensor[
        .int32,
        layout_unknown,
        linear_idx_type=.int32,
        layout_int_type=.int32,
    ](heap, RuntimeLayout[layout_unknown]({8, 8}, {8, 1}))
    for i in range(64):
        tensor4.ptr[i] = Int32(i)
    var frag4 = tensor4._vectorize_2[2]()
    var val4 = frag4[crd]
    assert_equal(val4[0], 16)
    assert_equal(val4[1], 17)
    heap.free()


def test_native_vectorize_static() raises:
    comptime flat_layout = TileLayout(
        Coord(Idx[16], Idx[32], Idx[4]),
        Coord(Idx[32], Idx[1], Idx[512]),
    )
    var tensor = stack_allocation[dtype=.float32, alignment=16](flat_layout)
    for i in range(2048):
        tensor.raw_store(i, Float32(i))

    comptime nested_layout = TileLayout(
        Coord(Coord(Idx[16], Idx[32]), Idx[4]),
        Coord(Coord(Idx[32], Idx[1]), Idx[512]),
    )
    var nested = tensor.reshape(nested_layout)
    comptime assert nested.rank == 2 and nested.flat_rank == 3
    # Native vectorize takes flat scalar tile sizes, not nested IntTuple tilers.
    # This leaf-equivalent view keeps the nested source's offsets unchanged.
    var fragments = nested.reshape(flat_layout).vectorize[1, 4, 1]()
    comptime assert fragments.flat_rank == 3 and fragments.element_size == 4
    comptime assert fragments.linear_idx_type == tensor.linear_idx_type
    assert_equal(fragments.dim[0](), 16)
    assert_equal(fragments.dim[1](), 8)
    assert_equal(fragments.dim[2](), 4)
    var nested_legacy = nested.to_layout_tensor()._vectorize_2[
        origin_of(), IntTuple(IntTuple(1, 4), 1)
    ]()
    var flat_legacy = tensor.to_layout_tensor()._vectorize_2[
        origin_of(), IntTuple(1, 4, 1)
    ]()
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
                var nested_value = nested_legacy[row + column * 16, plane]
                var flat_value = flat_legacy[row, column, plane]
                for lane in range(4):
                    assert_equal(native_value[lane], Float32(offset + lane))
                    assert_equal(native_value[lane], nested_value[lane])
                    assert_equal(native_value[lane], flat_value[lane])

    var square = stack_allocation[dtype=.float32, alignment=8](
        row_major[8, 8]()
    )
    for i in range(64):
        square.raw_store(i, Float32(i))
    var contiguous = square.vectorize[1, 2]()
    var legacy_contiguous = square.to_layout_tensor()._vectorize_2[2]()
    comptime assert contiguous.flat_rank == 2 and contiguous.element_size == 2
    assert_equal(contiguous.dim[0](), 8)
    assert_equal(contiguous.dim[1](), 4)
    for row in range(8):
        for column in range(4):
            var value = contiguous[row, column]
            var legacy_value = legacy_contiguous[row, column]
            assert_equal(
                Int(contiguous.layout(Coord(row, column))), row * 8 + column * 2
            )
            for lane in range(2):
                assert_equal(value[lane], Float32(row * 8 + column * 2 + lane))
                assert_equal(value[lane], legacy_value[lane])

    # Unlike the legacy strided Element, native SIMD lanes are contiguous.
    var row_blocks = square.vectorize[2, 1]()
    var row_value = row_blocks[2, 0]
    assert_equal(row_value[0], 32.0)
    assert_equal(row_value[1], 33.0)


def test_native_vectorize_dynamic[width: Int]() raises:
    var heap = alloc[Int32](64, alignment=16)
    var runtime_layout = TileLayout(
        Coord(Int32(8), Int32(8)), Coord(Int32(8), Idx[1])
    )
    var tensor = TileTensor[linear_idx_type=.int32](heap, runtime_layout)
    comptime assert not tensor.all_dims_known
    for i in range(64):
        tensor.raw_store(i, Int32(i))
    var fragments = tensor.vectorize[1, width]()
    comptime legacy_layout = Layout(
        IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE), IntTuple(UNKNOWN_VALUE, 1)
    )
    var legacy = LayoutTensor[
        .int32,
        legacy_layout,
        linear_idx_type=.int32,
        layout_int_type=.int32,
    ](heap, RuntimeLayout[legacy_layout]({8, 8}, {8, 1}))
    var legacy_fragments = legacy._vectorize_2[width]()
    comptime assert fragments.flat_rank == 2 and fragments.element_size == width
    comptime assert fragments.linear_idx_type == .int32
    assert_equal(fragments.dim[0](), 8)
    assert_equal(fragments.dim[1](), Int32(8 // width))
    assert_equal(Int(fragments.layout.stride_coord()[0].value()), 8)
    assert_equal(Int(fragments.layout.stride_coord()[1].value()), width)
    for row in range(8):
        for column in range(8 // width):
            var coordinate = Coord(row, column)
            var offset = row * 8 + column * width
            assert_equal(Int(fragments.layout(coordinate)), offset)
            var value = fragments[coordinate]
            var legacy_value = legacy_fragments[row, column]
            for lane in range(width):
                assert_equal(value[lane], Int32(offset + lane))
                assert_equal(value[lane], legacy_value[lane])
    heap.free()


def main() raises:
    test_vectorize_2()
    test_native_vectorize_static()
    test_native_vectorize_dynamic[2]()
    test_native_vectorize_dynamic[4]()
