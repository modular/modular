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
"""Tests for ManagedTensorSlice - a tensor view type for custom graph operations.
"""

from buffer.dimlist import DimList
from compiler_internal.directives import StaticTensorSpec
from tensor import ManagedTensorSlice, IOUnknown
from testing import assert_equal, TestSuite

from utils import IndexList


def test_basic_construction():
    """Test basic ManagedTensorSlice construction from pointer and shape."""
    var storage = InlineArray[Float32, 3 * 4](uninitialized=True)
    # Shape-only constructor computes row-major strides automatically
    comptime spec = StaticTensorSpec[DType.float32, 2](DimList(3, 4))
    var tensor = ManagedTensorSlice[io_spec=IOUnknown, static_spec=spec](
        storage.unsafe_ptr(), IndexList[2](3, 4)
    )

    assert_equal(tensor.rank, 2)
    assert_equal(tensor.size(), 12)


def test_shape_and_strides():
    """Test shape() and strides() methods."""
    var storage = InlineArray[Float32, 2 * 3 * 4](uninitialized=True)
    comptime spec = StaticTensorSpec[DType.float32, 3](DimList(2, 3, 4))
    var tensor = ManagedTensorSlice[io_spec=IOUnknown, static_spec=spec](
        storage.unsafe_ptr(), IndexList[3](2, 3, 4)
    )

    var shape = tensor.shape()
    assert_equal(shape[0], 2)
    assert_equal(shape[1], 3)
    assert_equal(shape[2], 4)

    var strides = tensor.strides()
    assert_equal(strides[0], 12)
    assert_equal(strides[1], 4)
    assert_equal(strides[2], 1)


def test_dim_size():
    """Test dim_size methods (compile-time and runtime)."""
    var storage = InlineArray[Float32, 5 * 7](uninitialized=True)
    comptime spec = StaticTensorSpec[DType.float32, 2](DimList(5, 7))
    var tensor = ManagedTensorSlice[io_spec=IOUnknown, static_spec=spec](
        storage.unsafe_ptr(), IndexList[2](5, 7)
    )

    # Test compile-time dim_size
    assert_equal(tensor.dim_size[0](), 5)
    assert_equal(tensor.dim_size[1](), 7)

    # Test runtime dim_size
    assert_equal(tensor.dim_size(0), 5)
    assert_equal(tensor.dim_size(1), 7)


def test_getitem_setitem():
    """Test __getitem__ and __setitem__ operations."""
    var storage = InlineArray[Float32, 2 * 3](uninitialized=True)
    comptime spec = StaticTensorSpec[DType.float32, 2](DimList(2, 3))
    var tensor = ManagedTensorSlice[
        mut=True, io_spec=IOUnknown, static_spec=spec
    ](storage.unsafe_ptr(), IndexList[2](2, 3))

    # Set values
    tensor[0, 0] = 1.0
    tensor[0, 1] = 2.0
    tensor[0, 2] = 3.0
    tensor[1, 0] = 4.0
    tensor[1, 1] = 5.0
    tensor[1, 2] = 6.0

    # Get values using variadic indices
    assert_equal(tensor[0, 0], 1.0)
    assert_equal(tensor[0, 1], 2.0)
    assert_equal(tensor[0, 2], 3.0)
    assert_equal(tensor[1, 0], 4.0)
    assert_equal(tensor[1, 1], 5.0)
    assert_equal(tensor[1, 2], 6.0)

    # Get values using IndexList
    assert_equal(tensor[IndexList[2](0, 2)], 3.0)
    assert_equal(tensor[IndexList[2](1, 0)], 4.0)


def test_simd_load_store():
    """Test SIMD load and store operations."""
    var storage = InlineArray[Float32, 8](uninitialized=True)
    comptime spec = StaticTensorSpec[DType.float32, 1](DimList(8))
    var tensor = ManagedTensorSlice[
        mut=True, io_spec=IOUnknown, static_spec=spec
    ](storage.unsafe_ptr(), IndexList[1](8))

    # Store a SIMD vector
    var vec = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
    tensor.store(IndexList[1](0), vec)

    var vec2 = SIMD[DType.float32, 4](5.0, 6.0, 7.0, 8.0)
    tensor.store(IndexList[1](4), vec2)

    # Load and verify
    var loaded = tensor.load[4](IndexList[1](0))
    assert_equal(loaded, SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0))

    var loaded2 = tensor.load[4](IndexList[1](4))
    assert_equal(loaded2, SIMD[DType.float32, 4](5.0, 6.0, 7.0, 8.0))


def test_to_layout_tensor():
    """Test to_layout_tensor() conversion."""
    var storage = InlineArray[Float32, 3 * 4](uninitialized=True)
    comptime spec = StaticTensorSpec[DType.float32, 2](DimList(3, 4))
    var tensor = ManagedTensorSlice[
        mut=True, io_spec=IOUnknown, static_spec=spec
    ](storage.unsafe_ptr(), IndexList[2](3, 4))

    # Initialize data
    for i in range(3):
        for j in range(4):
            tensor[i, j] = Float32(i * 4 + j)

    # Convert to LayoutTensor
    var layout_tensor = tensor.to_layout_tensor()

    # Verify the layout tensor has the same data
    assert_equal(layout_tensor[0, 0], 0.0)
    assert_equal(layout_tensor[1, 1], 5.0)
    assert_equal(layout_tensor[2, 3], 11.0)

    # Verify dimensions
    assert_equal(Int(layout_tensor.runtime_layout.shape[0]), 3)
    assert_equal(Int(layout_tensor.runtime_layout.shape[1]), 4)


def test_stride_length():
    """Test stride_length methods."""
    var storage = InlineArray[Float32, 3 * 5](uninitialized=True)
    comptime spec = StaticTensorSpec[DType.float32, 2](DimList(3, 5))
    var tensor = ManagedTensorSlice[io_spec=IOUnknown, static_spec=spec](
        storage.unsafe_ptr(), IndexList[2](3, 5)
    )

    # Test compile-time stride_length
    assert_equal(tensor.stride_length[0](), 5)
    assert_equal(tensor.stride_length[1](), 1)

    # Test runtime stride_length
    assert_equal(tensor.stride_length(0), 5)
    assert_equal(tensor.stride_length(1), 1)


def test_simd_load_store_2d():
    """Test SIMD load and store operations on 2D tensor."""
    var storage = InlineArray[Float32, 4 * 8](uninitialized=True)
    comptime spec = StaticTensorSpec[DType.float32, 2](DimList(4, 8))
    var tensor = ManagedTensorSlice[
        mut=True, io_spec=IOUnknown, static_spec=spec
    ](storage.unsafe_ptr(), IndexList[2](4, 8))

    # Store vectors in each row
    for i in range(4):
        var vec = SIMD[DType.float32, 4](
            Float32(i * 10),
            Float32(i * 10 + 1),
            Float32(i * 10 + 2),
            Float32(i * 10 + 3),
        )
        tensor.store(IndexList[2](i, 0), vec)

        var vec2 = SIMD[DType.float32, 4](
            Float32(i * 10 + 4),
            Float32(i * 10 + 5),
            Float32(i * 10 + 6),
            Float32(i * 10 + 7),
        )
        tensor.store(IndexList[2](i, 4), vec2)

    # Load and verify
    var loaded_row0 = tensor.load[4](IndexList[2](0, 0))
    assert_equal(loaded_row0, SIMD[DType.float32, 4](0.0, 1.0, 2.0, 3.0))

    var loaded_row2 = tensor.load[4](IndexList[2](2, 4))
    assert_equal(loaded_row2, SIMD[DType.float32, 4](24.0, 25.0, 26.0, 27.0))

    var loaded_row3 = tensor.load[4](IndexList[2](3, 0))
    assert_equal(loaded_row3, SIMD[DType.float32, 4](30.0, 31.0, 32.0, 33.0))


def test_to_tile_tensor():
    """Test to_tile_tensor() conversion."""
    var storage = InlineArray[Float32, 3 * 4](uninitialized=True)
    comptime spec = StaticTensorSpec[DType.float32, 2](DimList(3, 4))
    var tensor = ManagedTensorSlice[
        mut=True, io_spec=IOUnknown, static_spec=spec
    ](storage.unsafe_ptr(), IndexList[2](3, 4))

    # Initialize data
    for i in range(3):
        for j in range(4):
            tensor[i, j] = Float32(i * 4 + j)

    # Convert to TileTensor
    var tile_tensor = tensor.to_tile_tensor[DType.int64]()

    # Verify the layout tensor has the same data
    comptime assert tile_tensor.flat_rank == 2
    assert_equal(tile_tensor[0, 0], 0.0)
    assert_equal(tile_tensor[1, 1], 5.0)
    assert_equal(tile_tensor[2, 3], 11.0)

    # Verify dimensions
    assert_equal(tile_tensor.layout.shape[0]().value(), 3)
    assert_equal(tile_tensor.layout.shape[1]().value(), 4)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
