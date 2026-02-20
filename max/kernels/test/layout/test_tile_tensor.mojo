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

from buffer import NDBuffer, Dim, DimList
from layout._layout import Layout, row_major
from layout._tile_tensor import TileTensor
from layout._coord import ComptimeInt, Idx, Coord, RuntimeInt
from layout.int_tuple import IntTuple
from layout.swizzle import Swizzle
from math import ceildiv
from testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_true,
    assert_raises,
)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()


fn test_distribute() raises:
    comptime thread_layout = row_major((Idx[2](), Idx[2]()))

    var array = InlineArray[UInt32, 16](fill=-1)
    var ptr = array.unsafe_ptr()

    comptime data_layout_shape = Coord[ComptimeInt[4], ComptimeInt[4]]
    comptime data_layout_stride = Coord[ComptimeInt[4], ComptimeInt[1]]
    var layout_tensor = TileTensor(
        ptr=ptr,
        layout=Layout(
            shape=data_layout_shape(Idx[4](), Idx[4]()),
            stride=data_layout_stride(Idx[4](), Idx[1]()),
        ),
    )

    var counter = 0
    for th_id in range(4):
        var frag = layout_tensor.distribute[thread_layout=thread_layout,](th_id)

        # Fill the fragment positions with the thread id (0..3)
        for i in range(2):
            for j in range(2):
                frag[(Idx(i), Idx(j))] = UInt32(counter)
                counter += 1

    var expected = [0, 4, 1, 5, 8, 12, 9, 13, 2, 6, 3, 7, 10, 14, 11, 15]
    for i in range(16):
        assert_equal(ptr[i], UInt32(expected[i]))


fn test_distribute_with_swizzle() raises:
    """Test distribute with swizzle parameter.

    This test verifies that the swizzle parameter correctly transforms
    the memory access pattern. We use a simple swizzle that XORs bits
    to remap thread offsets.
    """
    comptime thread_layout = row_major((Idx[2](), Idx[2]()))

    # Use Swizzle(1, 0, 2) which XORs bit 2 with bit 0
    # yyy_mask = 1 << 2 = 4 (binary: 100)
    # swizzle(x) = x ^ ((x & 4) >> 2)
    # For offset 0: swizzle(0) = 0 ^ ((0 & 4) >> 2) = 0 ^ 0 = 0
    # For offset 1: swizzle(1) = 1 ^ ((1 & 4) >> 2) = 1 ^ 0 = 1
    # For offset 4: swizzle(4) = 4 ^ ((4 & 4) >> 2) = 4 ^ 1 = 5
    # For offset 5: swizzle(5) = 5 ^ ((5 & 4) >> 2) = 5 ^ 1 = 4
    comptime swizzle = Swizzle(1, 0, 2)

    var array = InlineArray[UInt32, 16](fill=-1)
    var ptr = array.unsafe_ptr()

    comptime data_layout_shape = Coord[ComptimeInt[4], ComptimeInt[4]]
    comptime data_layout_stride = Coord[ComptimeInt[4], ComptimeInt[1]]
    var layout_tensor = TileTensor[dtype = DType.uint32](
        ptr=ptr,
        layout=Layout(
            shape=data_layout_shape(Idx[4](), Idx[4]()),
            stride=data_layout_stride(Idx[4](), Idx[1]()),
        ),
    )

    # Assign thread IDs to positions with swizzle
    for th_id in range(4):
        var frag = layout_tensor.distribute[
            thread_layout=thread_layout, swizzle=swizzle
        ](th_id)
        # Write thread ID to each position in the fragment
        for i in range(2):
            for j in range(2):
                frag[(Idx(i), Idx(j))] = UInt32(th_id)

    # Thread layout row_major[2, 2] has strides [2, 1]
    # Thread 0: coord (0, 0) -> base offset 0*4 + 0*1 = 0, swizzle(0) = 0
    # Thread 1: coord (0, 1) -> base offset 0*4 + 1*1 = 1, swizzle(1) = 1
    # Thread 2: coord (1, 0) -> base offset 1*4 + 0*1 = 4, swizzle(4) = 5
    # Thread 3: coord (1, 1) -> base offset 1*4 + 1*1 = 5, swizzle(5) = 4

    # Verify that thread assignments are swizzled correctly
    # Thread 0 writes starting at offset 0
    assert_equal(ptr[0], 0)

    # Thread 1 writes starting at offset 1
    assert_equal(ptr[1], 1)

    # Thread 2 writes starting at swizzled offset 5 (from base 4)
    assert_equal(ptr[5], 2)

    # Thread 3 writes starting at swizzled offset 4 (from base 5)
    assert_equal(ptr[4], 3)


fn test_distribute_swizzle_vs_no_swizzle() raises:
    """Test that swizzle actually changes the memory access pattern.

    Compare the results of distribute with and without swizzle to verify
    that swizzling produces different memory layouts.
    """
    comptime thread_layout = row_major((Idx[2](), Idx[2]()))
    comptime swizzle = Swizzle(1, 0, 2)

    # Array without swizzle
    var array_no_swizzle = InlineArray[UInt32, 16](fill=0)
    var ptr_no_swizzle = array_no_swizzle.unsafe_ptr()

    # Array with swizzle
    var array_with_swizzle = InlineArray[UInt32, 16](fill=0)
    var ptr_with_swizzle = array_with_swizzle.unsafe_ptr()

    comptime data_layout_shape = Coord[ComptimeInt[4], ComptimeInt[4]]
    comptime data_layout_stride = Coord[ComptimeInt[4], ComptimeInt[1]]

    var tensor_no_swizzle = TileTensor[dtype = DType.uint32](
        ptr=ptr_no_swizzle,
        layout=Layout(
            shape=data_layout_shape(Idx[4](), Idx[4]()),
            stride=data_layout_stride(Idx[4](), Idx[1]()),
        ),
    )

    var tensor_with_swizzle = TileTensor[dtype = DType.uint32](
        ptr=ptr_with_swizzle,
        layout=Layout(
            shape=data_layout_shape(Idx[4](), Idx[4]()),
            stride=data_layout_stride(Idx[4](), Idx[1]()),
        ),
    )

    # Fill both tensors with thread IDs
    for th_id in range(4):
        var frag_no_swizzle = tensor_no_swizzle.distribute[
            thread_layout=thread_layout
        ](th_id)
        var frag_with_swizzle = tensor_with_swizzle.distribute[
            thread_layout=thread_layout, swizzle=swizzle
        ](th_id)

        for i in range(2):
            for j in range(2):
                frag_no_swizzle[(Idx(i), Idx(j))] = UInt32(th_id)
                frag_with_swizzle[(Idx(i), Idx(j))] = UInt32(th_id)

    # Verify that the two arrays are different (swizzle changes layout)
    var differ = False
    for i in range(16):
        if ptr_no_swizzle[i] != ptr_with_swizzle[i]:
            differ = True
            break
    assert_true(differ, "Swizzle should produce different memory layout")


fn test_tile() raises:
    # Create a 4x4 tensor with row-major layout
    var data = InlineArray[UInt32, 16](fill=0)

    var layout_tensor = TileTensor(data, row_major((Idx[4](), Idx[4]())))

    var counter = 0

    comptime for tile_i in range(2):
        comptime for tile_j in range(2):
            var current_tile = layout_tensor.tile[2, 2](
                (Idx(tile_i), Idx(tile_j)),
            )

            for i in range(2):
                for j in range(2):
                    current_tile[(Idx(i), Idx(j))] = UInt32(counter)
                    counter += 1

    # Expected layout after tiling:
    # Tile (0,0): values 0,1,2,3   -> positions [0,1], [4,5]
    # Tile (0,1): values 4,5,6,7   -> positions [2,3], [6,7]
    # Tile (1,0): values 8,9,10,11 -> positions [8,9], [12,13]
    # Tile (1,1): values 12,13,14,15 -> positions [10,11], [14,15]
    var expected = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]
    for i in range(16):
        assert_equal(data[i], UInt32(expected[i]))


def test_tensor_span_constructor():
    var bytes: List[UInt8] = [0, 1, 2, 3]
    var _tensor = TileTensor(
        bytes,
        row_major((Idx(2), Idx[2]())),
    )


def test_fill():
    var stack = InlineArray[UInt32, 16](fill=0)
    var tensor = TileTensor(stack, row_major[4, 4]()).fill(1)
    for i in range(tensor.layout.shape[0]().value()):
        for j in range(tensor.layout.shape[1]().value()):
            assert_equal(tensor[(Idx(i), Idx(j))], 1)


def test_fill_large():
    # layout._fillers.BATCH_SIZE is 2048, so we do 4096
    var stack = InlineArray[UInt32, 4096](fill=0)
    var tensor = TileTensor(stack, row_major[2048, 2]()).fill(1)
    for i in range(tensor.layout.shape[0]().value()):
        for j in range(tensor.layout.shape[1]().value()):
            assert_equal(tensor[(Idx(i), Idx(j))], 1)


fn test_slice() raises:
    """Test tensor slicing functionality."""
    # Test 2D slice (most common case)
    var data_2d = InlineArray[Int32, 16](uninitialized=True)

    # Initialize with values 0-15 in row-major order
    for i in range(16):
        data_2d[i] = Int32(i)

    # Create 4x4 tensor:
    # [0  1  2  3]
    # [4  5  6  7]
    # [8  9  10 11]
    # [12 13 14 15]
    var tensor_2d = TileTensor(data_2d, row_major[4, 4]())

    # Slice to extract middle 2x2 region [1:3, 1:3]:
    # [5  6]
    # [9  10]
    var sliced = tensor_2d.slice[1:3, 1:3]()

    # Verify slice dimensions
    assert_equal(sliced.layout.shape[0]().value(), 2)
    assert_equal(sliced.layout.shape[1]().value(), 2)

    # Verify slice values - use runtime indices since slice returns runtime shapes
    assert_equal(sliced[0, 0], 5)
    assert_equal(sliced[0, 1], 6)
    assert_equal(sliced[1, 0], 9)
    assert_equal(sliced[1, 1], 10)

    # Test that slice is a view (modifying slice affects original)
    sliced[(Idx(0), Idx(0))] = 99
    assert_equal(tensor_2d[(Idx(1), Idx(1))], 99)

    # Test different slice ranges
    var top_left = tensor_2d.slice[0:2, 0:2]()
    assert_equal(top_left[0, 0], 0)
    assert_equal(top_left[0, 1], 1)
    assert_equal(top_left[1, 0], 4)
    assert_equal(top_left[1, 1], 99)  # Modified earlier

    # Test slice with start=0 (should work with default)
    var first_row = tensor_2d.slice[0:1, 0:4]()
    assert_equal(first_row.layout.shape[0]().value(), 1)
    assert_equal(first_row.layout.shape[1]().value(), 4)
    assert_equal(first_row[0, 0], 0)
    assert_equal(first_row[0, 3], 3)


fn test_slice_3d() raises:
    """Test 3D tensor slicing."""
    # Create a 4x4x4 tensor
    var data_3d = InlineArray[Int32, 64](uninitialized=True)

    for i in range(64):
        data_3d[i] = Int32(i)

    var tensor_3d = TileTensor(data_3d, row_major[4, 4, 4]())

    # Slice [1:3, 1:3, 1:3] to get a 2x2x2 cube from the middle
    var sliced_3d = tensor_3d.slice[1:3, 1:3, 1:3]()

    # Verify dimensions
    assert_equal(sliced_3d.layout.shape[0]().value(), 2)
    assert_equal(sliced_3d.layout.shape[1]().value(), 2)
    assert_equal(sliced_3d.layout.shape[2]().value(), 2)

    # Verify some values - use runtime indices
    # Original tensor[1][1][1] = 1*16 + 1*4 + 1 = 21
    assert_equal(sliced_3d[0, 0, 0], 21)

    # Original tensor[1][1][2] = 1*16 + 1*4 + 2 = 22
    assert_equal(sliced_3d[0, 0, 1], 22)

    # Original tensor[2][2][2] = 2*16 + 2*4 + 2 = 42
    assert_equal(sliced_3d[1, 1, 1], 42)

    # Test that it's a view
    sliced_3d[(Idx(0), Idx(0), Idx(0))] = 999
    assert_equal(tensor_3d[(Idx(1), Idx(1), Idx(1))], 999)


# fn test_slice_runtime_shapes() raises:
#     """Test slicing with runtime-shaped tensors."""
#     var data = InlineArray[Float32, 12](uninitialized=True)
#
#     for i in range(12):
#         data[i] = Float32(i)
#
#     # Create tensor with runtime shapes
#     var shape = Coord(
#         RuntimeInt[DType.int32](3), RuntimeInt[DType.int32](4)
#     )
#     var stride = Coord(
#         RuntimeInt[DType.int32](4), RuntimeInt[DType.int32](1)
#     )
#     var layout = MixedLayout(shape^, stride^)
#
#     var tensor_runtime = TileTensor(
#         data.unsafe_ptr(), layout^
#     )
#
#     # Slice with compile-time slice bounds
#     var sliced = tensor_runtime.slice[1:3, 1:3]()
#
#     # Verify dimensions (result should be runtime too)
#     assert_equal(sliced.layout.shape[0].value(), 2)
#     assert_equal(sliced.layout.shape[1].value(), 2)
#
#     # Verify values - use runtime indices
#     # Original[1][1] = 1 * 4 + 1 = 5
#     assert_equal(sliced[(Idx(0), Idx(0))], Float32(5))
#
#     # Original[2][2] = 2*4 + 2 = 10
#     assert_equal(sliced[(Idx(1), Idx(1))], Float32(10))


fn test_vectorize() raises:
    """Test tensor vectorization functionality."""
    # Create a 16x16 tensor with row-major layout
    var data = InlineArray[Int32, 256](uninitialized=True)

    # Initialize with sequential values
    for i in range(256):
        data[i] = Int32(i)

    var tensor = TileTensor(data, row_major[16, 16]())

    # Vectorize with 4x4 blocks
    var vectorized = tensor.vectorize[4, 4]()

    # Verify vectorized tensor shape: 16/4 x 16/4 = 4x4
    assert_equal(vectorized.layout.shape[0]().value(), 4)
    assert_equal(vectorized.layout.shape[1]().value(), 4)

    # Verify vectorized tensor strides: original_stride * vector_shape
    # Original row-major 16x16 has strides [16, 1]
    # Vectorized strides should be [16*4, 1*4] = [64, 4]
    assert_equal(vectorized.layout.stride[0]().value(), 64)
    assert_equal(vectorized.layout.stride[1]().value(), 4)

    # Verify that vectorized[i, j] returns a SIMD vector starting at the (i,j) block
    # Block (0, 0) starts at element 0 - check first element of the SIMD vector
    assert_equal(vectorized[(Idx(0), Idx(0))][0], 0)

    # Block (0, 1) starts at element 4 (column offset by vector width)
    assert_equal(vectorized[(Idx(0), Idx(1))][0], 4)

    # Block (1, 0) starts at element 64 (row offset by vector height * row stride)
    assert_equal(vectorized[(Idx(1), Idx(0))][0], 64)

    # Block (1, 1) starts at element 68
    assert_equal(vectorized[(Idx(1), Idx(1))][0], 68)

    # Block (3, 3) is the last block, starts at element 3*64 + 3*4 = 204
    assert_equal(vectorized[(Idx(3), Idx(3))][0], 204)


fn test_vectorize_non_square() raises:
    """Test vectorization with non-square vector shapes."""
    var data = InlineArray[Int32, 64](uninitialized=True)

    for i in range(64):
        data[i] = Int32(i)

    # Create 8x8 tensor
    var tensor = TileTensor(data, row_major[8, 8]())

    # Vectorize with 2x4 blocks (different dimensions)
    var vectorized = tensor.vectorize[2, 4]()

    # Shape should be 8/2 x 8/4 = 4x2
    assert_equal(vectorized.layout.shape[0]().value(), 4)
    assert_equal(vectorized.layout.shape[1]().value(), 2)

    # Strides should be [8*2, 1*4] = [16, 4]
    assert_equal(vectorized.layout.stride[0]().value(), 16)
    assert_equal(vectorized.layout.stride[1]().value(), 4)

    # Verify block positions - check first element of each SIMD vector
    assert_equal(vectorized[(Idx(0), Idx(0))][0], 0)  # Block (0,0) at element 0
    assert_equal(vectorized[(Idx(0), Idx(1))][0], 4)  # Block (0,1) at element 4
    assert_equal(
        vectorized[(Idx(1), Idx(0))][0], 16
    )  # Block (1,0) at element 16
    assert_equal(
        vectorized[(Idx(3), Idx(1))][0], 52
    )  # Block (3,1) at element 52


fn test_vectorize_1d() raises:
    """Test vectorization of 1D tensor."""
    var data = InlineArray[Int32, 16](uninitialized=True)

    for i in range(16):
        data[i] = Int32(i)

    # Create 16-element 1D tensor
    var tensor = TileTensor(data, row_major[16]())

    # Vectorize with width 4
    var vectorized = tensor.vectorize[4]()

    # Shape should be 16/4 = 4
    assert_equal(vectorized.layout.shape[0]().value(), 4)

    # Stride should be 1*4 = 4
    assert_equal(vectorized.layout.stride[0]().value(), 4)

    # Verify block positions - check first element of each SIMD vector
    assert_equal(vectorized[(Idx(0),)][0], 0)
    assert_equal(vectorized[(Idx(1),)][0], 4)
    assert_equal(vectorized[(Idx(2),)][0], 8)
    assert_equal(vectorized[(Idx(3),)][0], 12)


def test_indexing():
    var stack: InlineArray[UInt8, 4] = [1, 2, 3, 4]
    var tensor = TileTensor(stack, row_major[2, 2]())
    assert_equal(tensor[Int32(0), Int64(0)], 1)
    assert_equal(tensor[Int(1), Int64(0)], 3)


def test_to_layout_tensor_square():
    var stack: InlineArray[UInt8, 4] = [1, 2, 3, 4]
    var tensor = TileTensor(stack, row_major[2, 2]()).to_layout_tensor()
    assert_equal(materialize[tensor.layout](), layout.Layout.row_major(2, 2))
    assert_equal(tensor.rank, 2)
    assert_equal(
        rebind[std.utils.IndexList[2]](
            tensor.runtime_layout.shape.value.canonicalize()
        ),
        std.utils.IndexList[2](2, 2),
    )


def test_to_layout_tensor_3d():
    var stack = InlineArray[UInt8, 64 * 8 * 4](fill=0)
    var tensor = TileTensor(stack, row_major[64, 8, 4]())
    var lt = tensor.to_layout_tensor()
    assert_equal(materialize[lt.layout](), layout.Layout.row_major(64, 8, 4))
    assert_equal(lt.rank, 3)
    assert_equal(
        rebind[std.utils.IndexList[3]](
            lt.runtime_layout.shape.value.canonicalize()
        ),
        std.utils.IndexList[3](64, 8, 4),
    )


def test_to_layout_tensor_3d_dynamic():
    var stack = InlineArray[UInt8, 64 * 8 * 4](fill=0)
    var tensor = TileTensor(
        stack, row_major((Idx[64](), Idx[8](), Idx(Int(4))))
    )
    var lt = tensor.to_layout_tensor()
    assert_equal(
        materialize[lt.layout](),
        layout.Layout.row_major(64, 8, layout.UNKNOWN_VALUE),
    )
    assert_equal(lt.rank, 3)
    assert_equal(
        rebind[std.utils.IndexList[3]](
            lt.runtime_layout.shape.value.canonicalize()
        ),
        std.utils.IndexList[3](64, 8, 4),
    )


fn test_coalesce_2d() raises:
    """Test coalescing a 2D tensor to rank-1."""
    var data = InlineArray[Int32, 16](uninitialized=True)

    # Initialize with sequential values
    for i in range(16):
        data[i] = Int32(i)

    # Create 4x4 tensor
    var tensor = TileTensor(data, row_major[4, 4]())

    # Coalesce to rank-1
    var coalesced = tensor.coalesce()

    # Verify coalesced tensor shape: 4*4 = 16
    assert_equal(coalesced.layout.shape[0]().value(), 16)

    # Verify coalesced tensor stride: 1
    assert_equal(coalesced.layout.stride[0]().value(), 1)

    # Verify elements are accessible in order
    for i in range(16):
        assert_equal(coalesced[(Idx(i),)], Int32(i))


fn test_coalesce_3d() raises:
    """Test coalescing a 3D tensor to rank-1."""
    var data = InlineArray[Int32, 24](uninitialized=True)

    for i in range(24):
        data[i] = Int32(i)

    # Create 2x3x4 tensor
    var tensor = TileTensor(data, row_major[2, 3, 4]())

    # Coalesce to rank-1
    var coalesced = tensor.coalesce()

    # Verify coalesced tensor shape: 2*3*4 = 24
    assert_equal(coalesced.layout.shape[0]().value(), 24)

    # Verify coalesced tensor stride: 1
    assert_equal(coalesced.layout.stride[0]().value(), 1)

    # Verify elements are accessible in order
    for i in range(24):
        assert_equal(coalesced[(Idx(i),)], Int32(i))


fn test_coalesce_1d() raises:
    """Test coalescing a 1D tensor (should be no-op effectively)."""
    var data = InlineArray[Int32, 8](uninitialized=True)

    for i in range(8):
        data[i] = Int32(i)

    # Create 8-element 1D tensor
    var tensor = TileTensor(data, row_major[8]())

    # Coalesce (should maintain rank-1)
    var coalesced = tensor.coalesce()

    # Verify shape and stride unchanged
    assert_equal(coalesced.layout.shape[0]().value(), 8)
    assert_equal(coalesced.layout.stride[0]().value(), 1)

    # Verify elements
    for i in range(8):
        assert_equal(coalesced[(Idx(i),)], Int32(i))


fn test_coalesce_element_size() raises:
    """Test that coalesce properly tracks element_size."""
    var data = InlineArray[Int32, 16](uninitialized=True)

    for i in range(16):
        data[i] = Int32(i)

    # Create 4x4 tensor
    var tensor = TileTensor(data, row_major[4, 4]())

    # Verify element_size is 1 for non-vectorized tensor
    assert_equal(tensor.element_size, 1)

    # Coalesce the tensor
    var coalesced = tensor.coalesce()

    # Verify coalesced shape: 4*4 = 16
    assert_equal(coalesced.layout.shape[0]().value(), 16)
    assert_equal(coalesced.layout.stride[0]().value(), 1)

    # Verify element_size is still 1 (coalesced from 1)
    assert_equal(coalesced.element_size, 1)

    # Verify all elements accessible
    for i in range(16):
        assert_equal(coalesced[(Idx(i),)], Int32(i))


fn test_to_nd_buffer_partially_dynamic() raises:
    var stack = InlineArray[Int32, 16](fill=0)
    var tensor = TileTensor(stack, row_major((Idx(Int(4)), Idx[4]())))
    var buffer = tensor._to_ndbuffer()
    assert_equal(buffer.shape.at[0](), Dim())
    assert_equal(buffer.shape.at[1](), Dim(4))
    assert_equal(buffer.dynamic_shape[0], 4)
    assert_equal(buffer.dynamic_shape[1], 4)


fn test_to_nd_buffer_fully_dynamic() raises:
    var stack = InlineArray[Int32, 16](fill=0)
    var tensor = TileTensor(stack, row_major((Idx(Int(8)), Idx(Int(2)))))
    var buffer = tensor._to_ndbuffer()
    assert_equal(buffer.shape.at[0](), Dim())
    assert_equal(buffer.shape.at[1](), Dim())
    assert_equal(buffer.dynamic_shape[0], 8)
    assert_equal(buffer.dynamic_shape[1], 2)


fn test_to_nd_buffer_fully_static() raises:
    var stack = InlineArray[Int32, 16](fill=0)
    var tensor = TileTensor(stack, row_major((Idx[16](), Idx[1]())))
    var buffer = tensor._to_ndbuffer()
    assert_equal(buffer.shape.at[0](), Dim(16))
    assert_equal(buffer.shape.at[1](), Dim(1))
    assert_equal(buffer.dynamic_shape[0], 16)
    assert_equal(buffer.dynamic_shape[1], 1)
