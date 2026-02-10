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

from layout._layout import (
    Layout,
    ZippedDivideLayout,
    BlockedProductLayout,
    row_major,
    col_major,
    blocked_product,
)
from layout._coord import ComptimeInt, Idx, Coord, RuntimeInt
from layout._tile_tensor import TileTensor
from layout.int_tuple import IntTuple
from testing import assert_equal, assert_true, TestSuite


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()


fn test_size_cosize() raises:
    # Row-major 3x4: last element (2,3) -> 11, cosize = 12
    var layout1 = Layout(
        shape=(Idx[3](), Idx[4]()),
        stride=(Idx[4](), Idx[1]()),
    )
    assert_equal(layout1.product(), 12)
    assert_equal(layout1.cosize(), 12)

    # Layout with gaps: last element (1,1) -> 11, cosize = 12
    var layout2 = Layout(
        shape=(Idx[2](), Idx[2]()), stride=(Idx[10](), Idx[1]())
    )
    assert_equal(layout2.product(), 4)
    assert_equal(layout2.cosize(), 12)


fn test_crd2idx() raises:
    var layout = Layout(
        shape=(Idx[4](), Idx[2]()),
        stride=(Idx[1](), Idx[4]()),
    )

    # Multi-dimensional coordinates
    assert_equal(layout(Coord(Idx[0](), Idx[0]())), 0)
    assert_equal(layout(Coord(Idx[1](), Idx[0]())), 1)
    assert_equal(layout(Coord(Idx[2](), Idx[0]())), 2)
    assert_equal(layout(Coord(Idx[3](), Idx[0]())), 3)
    assert_equal(layout(Coord(Idx[0](), Idx[1]())), 4)
    assert_equal(layout(Coord(Idx[1](), Idx[1]())), 5)
    assert_equal(layout(Coord(Idx[2](), Idx[1]())), 6)
    assert_equal(layout(Coord(Idx[3](), Idx[1]())), 7)

    assert_equal(layout.product(), 8)


def test_row_major():
    var shape = Coord(Idx[3](), Idx(4))
    var layout = row_major(shape)
    assert_true(layout.shape_coord() == shape)
    assert_true(layout.stride_coord() == Coord(Idx(4), Idx[1]()))

    var shape3 = Coord(Idx[3](), Idx(4), Idx(5))
    var layout3 = row_major(shape3)
    assert_true(layout3.shape_coord() == shape3)
    assert_true(layout3.stride_coord() == Coord(Idx(20), Idx(5), Idx[1]()))

    var shape3_static = Coord(
        ComptimeInt[3](), ComptimeInt[4](), ComptimeInt[5]()
    )
    var layout3_static = row_major(shape3_static)
    assert_true(layout3_static.shape_coord() == shape3_static)
    assert_true(
        layout3_static.stride_coord()
        == Coord(ComptimeInt[20](), ComptimeInt[5](), ComptimeInt[1]())
    )


def test_row_major_static_constructor_empty():
    var layout = row_major[]()
    assert_equal(len(layout.shape_coord()), 0)
    assert_equal(len(layout.stride_coord()), 0)


def test_row_major_static_constructor_():
    var layout = row_major[1, 2, 3]()
    assert_equal(len(layout.shape_coord()), 3)
    assert_equal(len(layout.stride_coord()), 3)
    assert_equal(layout.shape[0]().value(), 1)
    assert_equal(layout.shape[1]().value(), 2)
    assert_equal(layout.shape[2]().value(), 3)
    assert_equal(layout.stride[0]().value(), 6)
    assert_equal(layout.stride[1]().value(), 3)
    assert_equal(layout.stride[2]().value(), 1)


def test_zipped_divide_layout():
    # row_major[8, 16]() has shape_types = (ComptimeInt[8], ComptimeInt[16])
    # and stride_types = (ComptimeInt[16], ComptimeInt[1])
    comptime a = row_major[8, 16]()
    comptime b = Coord(Idx[2](), Idx[4]())
    comptime layout = ZippedDivideLayout[
        a._shape_types, a._stride_types, b.element_types
    ]
    assert_equal(layout._shape_types[0].VariadicType[0].static_value, 2)
    assert_equal(layout._shape_types[0].VariadicType[1].static_value, 4)
    assert_equal(layout._shape_types[1].VariadicType[0].static_value, 4)
    assert_equal(layout._shape_types[1].VariadicType[1].static_value, 4)
    assert_equal(layout._stride_types[0].VariadicType[0].static_value, 16)
    assert_equal(layout._stride_types[0].VariadicType[1].static_value, 1)
    assert_equal(layout._stride_types[1].VariadicType[0].static_value, 32)
    assert_equal(layout._stride_types[1].VariadicType[1].static_value, 4)


# ===----------------------------------------------------------------------=== #
# TileTensor.reshape tests
# ===----------------------------------------------------------------------=== #


def test_tile_tensor_reshape_static():
    """Test reshaping a TileTensor with compile-time dimensions."""
    var storage = InlineArray[Float32, 12](uninitialized=True)
    var tensor = TileTensor(storage, row_major[3, 4]()).fill(1.0)

    # Verify original shape
    assert_equal(tensor.dim[0](), 3)
    assert_equal(tensor.dim[1](), 4)
    assert_equal(tensor.numel(), 12)

    # Reshape to (2, 6)
    var reshaped_2x6 = tensor.reshape[2, 6]()
    assert_equal(reshaped_2x6.dim[0](), 2)
    assert_equal(reshaped_2x6.dim[1](), 6)
    assert_equal(reshaped_2x6.numel(), 12)

    # Reshape to (4, 3)
    var reshaped_4x3 = tensor.reshape[4, 3]()
    assert_equal(reshaped_4x3.dim[0](), 4)
    assert_equal(reshaped_4x3.dim[1](), 3)
    assert_equal(reshaped_4x3.numel(), 12)

    # Reshape to 1D (equivalent to coalesce)
    var reshaped_1d = tensor.reshape[12]()
    assert_equal(reshaped_1d.dim[0](), 12)
    assert_equal(reshaped_1d.numel(), 12)

    # Reshape to 3D
    var reshaped_3d = tensor.reshape[2, 2, 3]()
    assert_equal(reshaped_3d.dim[0](), 2)
    assert_equal(reshaped_3d.dim[1](), 2)
    assert_equal(reshaped_3d.dim[2](), 3)
    assert_equal(reshaped_3d.numel(), 12)


def test_tile_tensor_reshape_preserves_data():
    """Test that reshape preserves the underlying data."""
    var storage = InlineArray[Float32, 6](uninitialized=True)
    var tensor = TileTensor(storage, row_major[2, 3]())

    # Fill with distinct values
    tensor[0, 0] = 0.0
    tensor[0, 1] = 1.0
    tensor[0, 2] = 2.0
    tensor[1, 0] = 3.0
    tensor[1, 1] = 4.0
    tensor[1, 2] = 5.0

    # Reshape to (3, 2)
    var reshaped = tensor.reshape[3, 2]()

    # Verify data is preserved in row-major order
    assert_equal(reshaped[0, 0], 0.0)
    assert_equal(reshaped[0, 1], 1.0)
    assert_equal(reshaped[1, 0], 2.0)
    assert_equal(reshaped[1, 1], 3.0)
    assert_equal(reshaped[2, 0], 4.0)
    assert_equal(reshaped[2, 1], 5.0)

    # Reshape to 1D and verify
    var reshaped_1d = tensor.reshape[6]()
    assert_equal(reshaped_1d[0], 0.0)
    assert_equal(reshaped_1d[1], 1.0)
    assert_equal(reshaped_1d[2], 2.0)
    assert_equal(reshaped_1d[3], 3.0)
    assert_equal(reshaped_1d[4], 4.0)
    assert_equal(reshaped_1d[5], 5.0)


def test_tile_tensor_reshape_with_coord():
    """Test reshaping with a Coord argument (potentially runtime dims)."""
    var storage = InlineArray[Float32, 12](uninitialized=True)
    var tensor = TileTensor(storage, row_major[3, 4]()).fill(2.0)

    # Reshape using Coord with compile-time dimensions
    var reshaped = tensor.reshape(Coord(Idx[2](), Idx[6]()))
    assert_equal(reshaped.dim[0](), 2)
    assert_equal(reshaped.dim[1](), 6)
    assert_equal(reshaped.numel(), 12)

    # Reshape using Coord with runtime dimensions
    var rows = 4
    var cols = 3
    var reshaped_runtime = tensor.reshape(Coord(Idx(rows), Idx(cols)))
    assert_equal(reshaped_runtime.dim[0](), 4)
    assert_equal(reshaped_runtime.dim[1](), 3)
    assert_equal(reshaped_runtime.numel(), 12)


def test_tile_tensor_reshape_strides():
    """Test that reshaped tensor has correct row-major strides."""
    var storage = InlineArray[Float32, 24](uninitialized=True)
    var tensor = TileTensor(storage, row_major[4, 6]()).fill(0.0)

    # Reshape to (2, 3, 4)
    var reshaped = tensor.reshape[2, 3, 4]()

    # Verify row-major strides: (12, 4, 1) for shape (2, 3, 4)
    assert_equal(reshaped.layout.stride[0]().value(), 12)
    assert_equal(reshaped.layout.stride[1]().value(), 4)
    assert_equal(reshaped.layout.stride[2]().value(), 1)


def test_tile_tensor_reshape_is_view():
    """Test that reshape creates a view, not a copy."""
    var storage = InlineArray[Float32, 6](uninitialized=True)
    var tensor = TileTensor(storage, row_major[2, 3]()).fill(0.0)

    var reshaped = tensor.reshape[3, 2]()

    # Modify through reshaped view
    reshaped[1, 0] = 42.0

    # Verify change is visible in original tensor
    # (1, 0) in (3, 2) = index 2 in row-major = (0, 2) in (2, 3)
    assert_equal(tensor[0, 2], 42.0)


# ===----------------------------------------------------------------------=== #
# TileTensor.tile tests
# ===----------------------------------------------------------------------=== #


def test_tile_tensor_tile_with_int_coords():
    """Test tile method with variadic Int coordinates (LayoutTensor compatible).
    """
    var storage = InlineArray[Float32, 16](uninitialized=True)
    var tensor = TileTensor(storage, row_major[4, 4]())

    # Fill with distinct values
    for i in range(4):
        for j in range(4):
            tensor[i, j] = Float32(i * 4 + j)

    # Extract 2x2 tile at position (0, 0)
    var tile_00 = tensor.tile[2, 2](0, 0)
    assert_equal(tile_00.dim[0](), 2)
    assert_equal(tile_00.dim[1](), 2)
    assert_equal(tile_00[0, 0], 0.0)
    assert_equal(tile_00[0, 1], 1.0)
    assert_equal(tile_00[1, 0], 4.0)
    assert_equal(tile_00[1, 1], 5.0)

    # Extract 2x2 tile at position (1, 1)
    var tile_11 = tensor.tile[2, 2](1, 1)
    assert_equal(tile_11[0, 0], 10.0)
    assert_equal(tile_11[0, 1], 11.0)
    assert_equal(tile_11[1, 0], 14.0)
    assert_equal(tile_11[1, 1], 15.0)

    # Extract 2x2 tile at position (0, 1)
    var tile_01 = tensor.tile[2, 2](0, 1)
    assert_equal(tile_01[0, 0], 2.0)
    assert_equal(tile_01[0, 1], 3.0)
    assert_equal(tile_01[1, 0], 6.0)
    assert_equal(tile_01[1, 1], 7.0)


def test_tile_tensor_tile_is_view():
    """Test that tile creates a view, not a copy."""
    var storage = InlineArray[Float32, 16](uninitialized=True)
    var tensor = TileTensor(storage, row_major[4, 4]()).fill(0.0)

    var tile = tensor.tile[2, 2](1, 0)

    # Modify through tile view
    tile[0, 0] = 99.0

    # Verify change is visible in original tensor
    # tile (1, 0) with size 2x2 starts at (2, 0) in original
    assert_equal(tensor[2, 0], 99.0)


# ===----------------------------------------------------------------------=== #
# blocked_product tests
# ===----------------------------------------------------------------------=== #


def test_blocked_product_basic():
    """Test blocked_product creates correct hierarchical layout.

    Example from legacy layout docs:
    - block = 2x2 row-major: shape (2,2), stride (2,1)
    - tiler = 2x3 row-major: shape (2,3), stride (3,1)
    - result: shape ((2,2), (2,3)), stride ((2,1), (12,4))
    """
    var block = row_major[2, 2]()
    var tiler = row_major[2, 3]()
    var blocked = blocked_product(block, tiler)

    # Check inner shape (block shape)
    assert_equal(blocked.shape[0]()[0].value(), 2)
    assert_equal(blocked.shape[0]()[1].value(), 2)

    # Check outer shape (tiler shape)
    assert_equal(blocked.shape[1]()[0].value(), 2)
    assert_equal(blocked.shape[1]()[1].value(), 3)

    # Check inner stride (block stride)
    assert_equal(blocked.stride[0]()[0].value(), 2)
    assert_equal(blocked.stride[0]()[1].value(), 1)

    # Check outer stride = block.cosize * tiler.stride
    # block.cosize = 4 (2x2), tiler.stride = (3, 1)
    # outer_stride = (4*3, 4*1) = (12, 4)
    assert_equal(blocked.stride[1]()[0].value(), 12)
    assert_equal(blocked.stride[1]()[1].value(), 4)


def test_blocked_product_type_alias():
    """Test BlockedProductLayout type alias directly."""
    comptime block = row_major[2, 2]()
    comptime tiler = row_major[2, 3]()
    comptime layout = BlockedProductLayout[
        block._shape_types,
        block._stride_types,
        tiler._shape_types,
        tiler._stride_types,
    ]

    # inner_shape types should be (ComptimeInt[2], ComptimeInt[2])
    assert_equal(layout._shape_types[0].VariadicType[0].static_value, 2)
    assert_equal(layout._shape_types[0].VariadicType[1].static_value, 2)

    # outer_shape types should be (ComptimeInt[2], ComptimeInt[3])
    assert_equal(layout._shape_types[1].VariadicType[0].static_value, 2)
    assert_equal(layout._shape_types[1].VariadicType[1].static_value, 3)

    # inner_stride types should be (ComptimeInt[2], ComptimeInt[1])
    assert_equal(layout._stride_types[0].VariadicType[0].static_value, 2)
    assert_equal(layout._stride_types[0].VariadicType[1].static_value, 1)

    # outer_stride types should be (ComptimeInt[12], ComptimeInt[4])
    # = tiler.stride * block.row_major_strides
    # = (3, 1) * (2*2, 2) ... wait, that's not right
    # Let me recalculate: outer_stride = tiler.stride * row_major_strides_of_block_shape
    # row_major_strides of (2, 2) = (2, 1)
    # So outer_stride = (3 * 2, 1 * ... hmm
    # Actually the formula is: outer_stride[i] = tiler.stride[i] * block.cosize
    # For row-major block, cosize = 4
    # So outer_stride = (3*4, 1*4) = (12, 4)
    assert_equal(layout._stride_types[1].VariadicType[0].static_value, 12)
    assert_equal(layout._stride_types[1].VariadicType[1].static_value, 4)


# ===----------------------------------------------------------------------=== #
# col_major tests
# ===----------------------------------------------------------------------=== #


def test_col_major_2d():
    """Test column-major layout for 2D shapes.

    For shape (M, N):
    - row_major strides: (N, 1)
    - col_major strides: (1, M)
    """
    var layout = col_major[3, 4]()

    # Shape should be (3, 4)
    assert_equal(layout.shape[0]().value(), 3)
    assert_equal(layout.shape[1]().value(), 4)

    # Strides should be (1, 3) for column-major
    assert_equal(layout.stride[0]().value(), 1)
    assert_equal(layout.stride[1]().value(), 3)


def test_col_major_3d():
    """Test column-major layout for 3D shapes.

    For shape (M, N, K):
    - row_major strides: (N*K, K, 1)
    - col_major strides: (1, M, M*N)
    """
    var layout = col_major[2, 3, 4]()

    # Shape should be (2, 3, 4)
    assert_equal(layout.shape[0]().value(), 2)
    assert_equal(layout.shape[1]().value(), 3)
    assert_equal(layout.shape[2]().value(), 4)

    # Strides should be (1, 2, 6) for column-major
    # stride[0] = 1
    # stride[1] = shape[0] = 2
    # stride[2] = shape[0] * shape[1] = 2 * 3 = 6
    assert_equal(layout.stride[0]().value(), 1)
    assert_equal(layout.stride[1]().value(), 2)
    assert_equal(layout.stride[2]().value(), 6)


def test_col_major_vs_row_major():
    """Test that col_major and row_major produce different strides."""
    var row = row_major[3, 4]()
    var col = col_major[3, 4]()

    # Same shape
    assert_equal(row.shape[0]().value(), col.shape[0]().value())
    assert_equal(row.shape[1]().value(), col.shape[1]().value())

    # Different strides
    # row_major: (4, 1)
    assert_equal(row.stride[0]().value(), 4)
    assert_equal(row.stride[1]().value(), 1)

    # col_major: (1, 3)
    assert_equal(col.stride[0]().value(), 1)
    assert_equal(col.stride[1]().value(), 3)


def test_col_major_with_coord():
    """Test col_major with Coord argument."""
    var shape = Coord(Idx[3](), Idx[4]())
    var layout = col_major(shape)

    assert_equal(layout.shape[0]().value(), 3)
    assert_equal(layout.shape[1]().value(), 4)
    assert_equal(layout.stride[0]().value(), 1)
    assert_equal(layout.stride[1]().value(), 3)


def test_col_major_with_runtime_dims():
    """Test col_major with runtime dimensions."""
    var m = 3
    var n = 4
    var shape = Coord(Idx(m), Idx(n))
    var layout = col_major(shape)

    assert_equal(layout.shape[0]().value(), 3)
    assert_equal(layout.shape[1]().value(), 4)
    assert_equal(layout.stride[0]().value(), 1)
    assert_equal(layout.stride[1]().value(), 3)


# ===----------------------------------------------------------------------=== #
# Coord.flatten tests
# ===----------------------------------------------------------------------=== #


def test_coord_flatten_non_nested():
    """Test flatten on a non-nested Coord (should be identity)."""
    var coord = Coord(Idx[1](), Idx[2](), Idx[3]())
    var flat = coord.flatten()

    assert_equal(len(flat), 3)
    assert_equal(flat[0].value(), 1)
    assert_equal(flat[1].value(), 2)
    assert_equal(flat[2].value(), 3)


def test_coord_flatten_single_nested():
    """Test flatten on a Coord with one nested Coord."""
    var nested = Coord(Idx[5](), Coord(Idx[3](), Idx[2]()), Idx[7]())
    var flat = nested.flatten()

    # Should flatten to (5, 3, 2, 7)
    assert_equal(len(flat), 4)
    assert_equal(flat[0].value(), 5)
    assert_equal(flat[1].value(), 3)
    assert_equal(flat[2].value(), 2)
    assert_equal(flat[3].value(), 7)


def test_coord_flatten_nested_at_start():
    """Test flatten with nested Coord at the beginning."""
    var nested = Coord(Coord(Idx[1](), Idx[2]()), Idx[3]())
    var flat = nested.flatten()

    # Should flatten to (1, 2, 3)
    assert_equal(len(flat), 3)
    assert_equal(flat[0].value(), 1)
    assert_equal(flat[1].value(), 2)
    assert_equal(flat[2].value(), 3)


def test_coord_flatten_nested_at_end():
    """Test flatten with nested Coord at the end."""
    var nested = Coord(Idx[1](), Coord(Idx[2](), Idx[3]()))
    var flat = nested.flatten()

    # Should flatten to (1, 2, 3)
    assert_equal(len(flat), 3)
    assert_equal(flat[0].value(), 1)
    assert_equal(flat[1].value(), 2)
    assert_equal(flat[2].value(), 3)


def test_coord_flatten_blocked_product_shape():
    """Test flatten on shape from blocked_product."""
    var block = row_major[2, 2]()
    var tiler = row_major[2, 3]()
    var blocked = blocked_product(block, tiler)

    # blocked.shape is Coord(Coord(2, 2), Coord(2, 3))
    var flat_shape = blocked.shape_coord().flatten()

    # Should flatten to (2, 2, 2, 3)
    assert_equal(len(flat_shape), 4)
    assert_equal(flat_shape[0].value(), 2)
    assert_equal(flat_shape[1].value(), 2)
    assert_equal(flat_shape[2].value(), 2)
    assert_equal(flat_shape[3].value(), 3)


def test_coord_flatten_blocked_product_stride():
    """Test flatten on stride from blocked_product."""
    var block = row_major[2, 2]()
    var tiler = row_major[2, 3]()
    var blocked = blocked_product(block, tiler)

    # blocked.stride is Coord(Coord(2, 1), Coord(12, 4))
    var flat_stride = blocked.stride_coord().flatten()

    # Should flatten to (2, 1, 12, 4)
    assert_equal(len(flat_stride), 4)
    assert_equal(flat_stride[0].value(), 2)
    assert_equal(flat_stride[1].value(), 1)
    assert_equal(flat_stride[2].value(), 12)
    assert_equal(flat_stride[3].value(), 4)


# ===----------------------------------------------------------------------=== #
# TileTensor flat indexing tests
# ===----------------------------------------------------------------------=== #


def test_tile_tensor_flat_rank():
    """Test that flat_rank is computed correctly for nested and non-nested layouts.
    """
    # Non-nested layout: flat_rank == rank
    comptime tensor1 = TileTensor[
        DType.float32, type_of(row_major[3, 4]()), MutAnyOrigin
    ]
    comptime assert tensor1.rank == 2
    comptime assert tensor1.flat_rank == 2

    # Nested layout from blocked_product: flat_rank > rank
    comptime block = row_major[2, 2]()
    comptime tiler = row_major[2, 3]()
    comptime blocked_layout = blocked_product(block, tiler)

    comptime tensor2 = TileTensor[
        DType.float32, type_of(blocked_layout), MutAnyOrigin
    ]
    comptime assert tensor2.rank == 2  # Two top-level Coords
    comptime assert tensor2.flat_rank == 4  # Four scalar dimensions


def test_tile_tensor_flat_indexing_blocked():
    """Test flat indexing on a tensor with blocked_product layout."""
    # Create a blocked layout: 2x2 blocks arranged in 2x3 grid
    # Total size: 4 * 6 = 24 elements
    var block = row_major[2, 2]()
    var tiler = row_major[2, 3]()
    var blocked_layout = blocked_product(block, tiler)

    var storage = InlineArray[Float32, 24](uninitialized=True)
    var tensor = TileTensor(storage, blocked_layout)

    # Initialize using flat indices
    # The layout has strides ((2, 1), (12, 4))
    # Flat indexing: tensor[block_row, block_col, tile_row, tile_col]
    for block_row in range(2):
        for block_col in range(2):
            for tile_row in range(2):
                for tile_col in range(3):
                    var val = Float32(
                        block_row * 1000
                        + block_col * 100
                        + tile_row * 10
                        + tile_col
                    )
                    tensor[block_row, block_col, tile_row, tile_col] = val

    # Read back and verify
    assert_equal(tensor[0, 0, 0, 0], 0.0)
    assert_equal(tensor[0, 1, 0, 0], 100.0)
    assert_equal(tensor[1, 0, 0, 0], 1000.0)
    assert_equal(tensor[1, 1, 0, 0], 1100.0)
    assert_equal(tensor[0, 0, 1, 2], 12.0)
    assert_equal(tensor[1, 1, 1, 2], 1112.0)


def test_tile_tensor_flat_indexing_with_coord():
    """Test flat indexing using Coord on nested layout."""
    var block = row_major[2, 2]()
    var tiler = row_major[2, 3]()
    var blocked_layout = blocked_product(block, tiler)

    var storage = InlineArray[Float32, 24](fill=0.0)
    var tensor = TileTensor(storage, blocked_layout)

    # Use flat Coord for indexing
    var flat_coord = Coord(Idx[1](), Idx[0](), Idx[1](), Idx[2]())
    tensor[flat_coord] = 42.0

    # Read back with flat indices
    assert_equal(tensor[1, 0, 1, 2], 42.0)
