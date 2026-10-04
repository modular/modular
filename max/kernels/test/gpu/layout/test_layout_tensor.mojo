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

from std.itertools import product
from layout import (
    CoordLike,
    Coord,
    Idx,
    Layout,
    LayoutTensor,
    TileTensor,
    row_major,
    stack_allocation,
)
from layout.tile_layout import blocked_product
from layout._fillers import arange
from std.testing import assert_equal

from std.utils.index import IndexList


def test_runtime_and_compile_time_dim_and_stride[
    MType: CoordLike, KType: CoordLike, //
](m: MType, k: KType) raises:
    var shape = Coord(k, m)
    var tensor = TileTensor(
        MutPointer[Float32, MutAnyOrigin].unsafe_dangling(), row_major(shape)
    )

    var K = Int(k.value())
    var M = Int(m.value())

    assert_equal(Int(tensor.dim(0)), K)
    assert_equal(Int(tensor.dim(1)), M)
    assert_equal(Int(tensor.dynamic_stride(0)), M)
    assert_equal(tensor.dynamic_stride(1), 1)

    assert_equal(Int(tensor.dim[0]()), K)
    assert_equal(Int(tensor.dim[1]()), M)
    assert_equal(Int(tensor.layout.stride[0]().value()), M)
    assert_equal(tensor.layout.stride[1]().value(), 1)


def test_nested_layout_shape() raises:
    """Checks static and runtime extents for nested layouts."""
    # Test case 1: blocked_product creates nested layout
    comptime tiler_layout = row_major[2, 4]()
    comptime base_layout = row_major[32, 32]()
    comptime smem_layout = blocked_product(base_layout, tiler_layout)

    var tensor = TileTensor(
        MutPointer[Float32, MutAnyOrigin].unsafe_dangling(), smem_layout
    )

    # Shape should be (64, 128) because:
    # - First dimension: 32 * 2 = 64
    # - Second dimension: 32 * 4 = 128
    comptime shape0 = smem_layout.shape[0]().product()
    comptime shape1 = smem_layout.shape[1]().product()

    assert_equal(tensor.dim[0](), 64)
    assert_equal(tensor.dim[1](), 128)
    assert_equal(shape0, 64, "Shape[0] should be 64 for nested layout")
    assert_equal(shape1, 128, "Shape[1] should be 128 for nested layout")

    # Total size should be 64 * 128 = 8192
    var total_size = tensor.layout.size()
    assert_equal(total_size, 8192, "Total size should be 8192")

    # Test case 2: Ensure non-nested layouts still work (regression test)
    comptime simple_layout = row_major[16, 32]()
    var simple_tensor = TileTensor(
        MutPointer[Float32, MutAnyOrigin].unsafe_dangling(), simple_layout
    )
    comptime simple_shape0 = simple_tensor.static_shape[0]
    comptime simple_shape1 = simple_tensor.static_shape[1]

    assert_equal(simple_tensor.dim[0](), 16)
    assert_equal(simple_tensor.dim[1](), 32)
    assert_equal(simple_shape0, 16, "Non-nested shape[0] should still work")
    assert_equal(simple_shape1, 32, "Non-nested shape[1] should still work")


def _create_tensor_2x2[
    dtype: DType
]() -> LayoutTensor[dtype, Layout.row_major(2, 2), MutAnyOrigin]:
    """Helper to create a 2x2 row-major tensor on the stack."""
    return LayoutTensor[
        dtype,
        Layout.row_major(2, 2),
        MutAnyOrigin,
        address_space=.GENERIC,
    ].stack_allocation()


def _copy_transpose[
    dtype: DType
](
    src: LayoutTensor[dtype, Layout.row_major(2, 2), MutAnyOrigin],
    mut dst: LayoutTensor[dtype, Layout.row_major(2, 2), MutAnyOrigin],
):
    """Copy tensor src into dst with transposed indices."""
    for i, j in product(range(2), range(2)):
        dst[j, i] = src[i, j]


def test_transpose_arithmetic() raises:
    """Test all arithmetic operations with transposed tensors.

    This test verifies that arithmetic operations (+, -, *, /) work correctly
    when one operand is a transposed view of a tensor. This ensures the
    transpose operation properly maintains stride information for arithmetic.
    """
    # Test with arange values: a = [[0, 1], [2, 3]]
    var a = _create_tensor_2x2[.float32]()
    arange(a)

    var b = _create_tensor_2x2[.float32]()
    _copy_transpose(a, b)

    # After transpose, a.transpose() = [[0, 2], [1, 3]] = b
    # Test subtraction: should be all zeros
    var sub_result = a.transpose() - b
    assert_equal(sub_result[0, 0], 0.0)
    assert_equal(sub_result[0, 1], 0.0)
    assert_equal(sub_result[1, 0], 0.0)
    assert_equal(sub_result[1, 1], 0.0)

    # Test addition: a.transpose() + b = 2 * [[0, 2], [1, 3]]
    var add_result = a.transpose() + b
    assert_equal(add_result[0, 0], 0.0)
    assert_equal(add_result[0, 1], 4.0)
    assert_equal(add_result[1, 0], 2.0)
    assert_equal(add_result[1, 1], 6.0)

    # Test multiplication: element-wise product
    var mul_result = a.transpose() * b
    assert_equal(mul_result[0, 0], 0.0)  # 0 * 0
    assert_equal(mul_result[0, 1], 4.0)  # 2 * 2
    assert_equal(mul_result[1, 0], 1.0)  # 1 * 1
    assert_equal(mul_result[1, 1], 9.0)  # 3 * 3

    # Test division with non-zero values: c = [[2, 4], [6, 8]]
    var c = _create_tensor_2x2[.float32]()
    for i, j in product(range(2), range(2)):
        c[i, j] = Float32((i * 2 + j + 1) * 2)

    var d = _create_tensor_2x2[.float32]()
    _copy_transpose(c, d)

    # c.transpose() / d should be all ones
    var div_result = c.transpose() / d
    assert_equal(div_result[0, 0], 1.0)
    assert_equal(div_result[0, 1], 1.0)
    assert_equal(div_result[1, 0], 1.0)
    assert_equal(div_result[1, 1], 1.0)


def test_different_layouts_arithmetic() raises:
    """Test arithmetic between row-major and column-major tensors.

    This verifies that tensors with different memory layouts can still
    perform arithmetic operations correctly based on their logical indices.
    """
    var a = _create_tensor_2x2[.float32]()
    arange(a)

    # Create column-major tensor with same logical values
    var b = LayoutTensor[
        .float32,
        Layout.col_major(2, 2),
        MutAnyOrigin,
        address_space=.GENERIC,
    ].stack_allocation()
    for i, j in product(range(2), range(2)):
        b[i, j] = a[i, j]

    # Subtraction should yield zeros despite different memory layouts
    var result = a - b
    assert_equal(result[0, 0], 0.0)
    assert_equal(result[0, 1], 0.0)
    assert_equal(result[1, 0], 0.0)
    assert_equal(result[1, 1], 0.0)


def test_coalesce() raises:
    var stack = Array[Int8, 16](fill=0)
    var tensor = TileTensor(stack, row_major[4, 4]()).coalesce()
    assert_equal(tensor.num_elements(), 16)
    assert_equal(tensor.rank, 1)
    assert_equal(tensor.layout.stride[0]().value(), 1)


def test_get_shape() raises:
    var stack = Array[Int8, 16](fill=0)
    var tensor = TileTensor(stack, row_major[4, 4]())
    assert_equal(4, tensor.layout.shape[0]().value())
    assert_equal(4, tensor.layout.shape[1]().value())


def test_reshape() raises:
    var stack = Array[Int8, 16](fill=0)
    var tensor = TileTensor(stack, row_major[16]()).reshape(Coord(4, 4))
    assert_equal(tensor.num_elements(), 16)
    assert_equal(tensor.layout.shape[0]().value(), 4)
    assert_equal(tensor.layout.shape[1]().value(), 4)


def test_aligned_load() raises:
    """Tests aligned SIMD loads with coordinate and index-list arguments."""
    var tensor = stack_allocation[.float32, alignment=16](row_major[4, 16]())

    for column in range(0, 16, 4):
        var expected = SIMD[.float32, 4](Float32(column // 4 + 1))
        tensor.store[width=4, alignment=16](Coord(0, column), expected)
        var value = tensor.load[width=4, alignment=16](Coord(0, column))
        var linear_value = tensor.load_linear[width=4, alignment=16](
            IndexList[2](0, column)
        )
        assert_equal(value, linear_value)
        assert_equal(value, expected)


def test_unaligned_load() raises:
    """Tests SIMD loads at scalar-aligned offsets across multiple rows."""
    var tensor = stack_allocation[.float32, alignment=16](row_major[2, 16]())
    for row in range(2):
        for column in range(16):
            tensor[row, column] = Float32(row * 16 + column)

    for row in range(2):
        for column in range(1, 4):
            var expected = SIMD[.float32, 4]()
            comptime for lane in range(4):
                expected[lane] = Float32(row * 16 + column + lane)
            var value = tensor.load[width=4, alignment=4](Coord(row, column))
            var linear_value = tensor.load_linear[width=4, alignment=4](
                IndexList[2](row, column)
            )
            assert_equal(value, expected)
            assert_equal(linear_value, expected)


def main() raises:
    test_runtime_and_compile_time_dim_and_stride(Idx[120], Idx[512])
    test_nested_layout_shape()
    test_transpose_arithmetic()
    test_different_layouts_arithmetic()
    test_aligned_load()
    test_unaligned_load()
    test_coalesce()
    test_get_shape()
    test_reshape()
