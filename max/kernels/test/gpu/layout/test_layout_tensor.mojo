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

from buffer.dimlist import DimList
from internal_utils._utils import ValOrDim, dynamic, static
from itertools import product
from layout import Layout, LayoutTensor, RuntimeLayout
from layout.layout import blocked_product
from layout._fillers import arange
from testing import assert_equal

from utils.index import IndexList


def test_runtime_and_compile_time_dim_and_stride(m: ValOrDim, k: ValOrDim):
    alias static_shape = DimList(k.dim, m.dim)
    var dynamic_shape = IndexList[2](k.value, m.value)
    alias layout = Layout.row_major[2](static_shape)

    var tensor = LayoutTensor[DType.float32, layout,](
        UnsafePointer[Float32](),
        RuntimeLayout[layout].row_major(dynamic_shape),
    )

    assert_equal(tensor.dim(0), dynamic_shape[0])
    assert_equal(tensor.dim(1), dynamic_shape[1])
    assert_equal(tensor.stride(0), dynamic_shape[1])
    assert_equal(tensor.stride(1), 1)

    assert_equal(tensor.dim[0](), dynamic_shape[0])
    assert_equal(tensor.dim[1](), dynamic_shape[1])
    assert_equal(tensor.stride[0](), -1)
    assert_equal(tensor.stride[1](), 1)


def test_nested_layout_shape():
    """Test that shape[idx]() works correctly for nested layouts."""
    # Test case 1: blocked_product creates nested layout
    alias tiler_layout = Layout.row_major(2, 4)
    alias base_layout = Layout.row_major(32, 32)
    alias smem_layout = blocked_product(base_layout, tiler_layout)

    var tensor = LayoutTensor[DType.float32, smem_layout, MutableAnyOrigin](
        UnsafePointer[Float32]()
    )

    # Shape should be (64, 128) because:
    # - First dimension: 32 * 2 = 64
    # - Second dimension: 32 * 4 = 128
    alias shape0 = tensor.shape[0]()
    alias shape1 = tensor.shape[1]()

    assert_equal(shape0, 64, "Shape[0] should be 64 for nested layout")
    assert_equal(shape1, 128, "Shape[1] should be 128 for nested layout")

    # Total size should be 64 * 128 = 8192
    var total_size = tensor.size()
    assert_equal(total_size, 8192, "Total size should be 8192")

    # Test case 2: Ensure non-nested layouts still work (regression test)
    alias simple_layout = Layout.row_major(16, 32)
    alias simple_shape0 = LayoutTensor[
        DType.float32, simple_layout, MutableAnyOrigin
    ].shape[0]()
    alias simple_shape1 = LayoutTensor[
        DType.float32, simple_layout, MutableAnyOrigin
    ].shape[1]()

    assert_equal(simple_shape0, 16, "Non-nested shape[0] should still work")
    assert_equal(simple_shape1, 32, "Non-nested shape[1] should still work")


fn _create_tensor_2x2[
    dtype: DType
]() -> LayoutTensor[dtype, Layout.row_major(2, 2), MutableAnyOrigin]:
    """Helper to create a 2x2 row-major tensor on the stack."""
    return LayoutTensor[
        dtype,
        Layout.row_major(2, 2),
        MutableAnyOrigin,
        address_space = AddressSpace.GENERIC,
    ].stack_allocation()


fn _copy_transpose[
    dtype: DType
](
    src: LayoutTensor[dtype, Layout.row_major(2, 2), MutableAnyOrigin],
    mut dst: LayoutTensor[dtype, Layout.row_major(2, 2), MutableAnyOrigin],
):
    """Copy tensor src into dst with transposed indices."""
    for i, j in product(range(2), range(2)):
        dst[j, i] = src[i, j]


def test_transpose_arithmetic():
    """Test all arithmetic operations with transposed tensors.

    This test verifies that arithmetic operations (+, -, *, /) work correctly
    when one operand is a transposed view of a tensor. This ensures the
    transpose operation properly maintains stride information for arithmetic.
    """
    # Test with arange values: a = [[0, 1], [2, 3]]
    var a = _create_tensor_2x2[DType.float32]()
    arange(a)

    var b = _create_tensor_2x2[DType.float32]()
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
    var c = _create_tensor_2x2[DType.float32]()
    for i, j in product(range(2), range(2)):
        c[i, j] = Float32((i * 2 + j + 1) * 2)

    var d = _create_tensor_2x2[DType.float32]()
    _copy_transpose(c, d)

    # c.transpose() / d should be all ones
    var div_result = c.transpose() / d
    assert_equal(div_result[0, 0], 1.0)
    assert_equal(div_result[0, 1], 1.0)
    assert_equal(div_result[1, 0], 1.0)
    assert_equal(div_result[1, 1], 1.0)


def test_different_layouts_arithmetic():
    """Test arithmetic between row-major and column-major tensors.

    This verifies that tensors with different memory layouts can still
    perform arithmetic operations correctly based on their logical indices.
    """
    var a = _create_tensor_2x2[DType.float32]()
    arange(a)

    # Create column-major tensor with same logical values
    var b = LayoutTensor[
        DType.float32,
        Layout.col_major(2, 2),
        MutableAnyOrigin,
        address_space = AddressSpace.GENERIC,
    ].stack_allocation()
    for i, j in product(range(2), range(2)):
        b[i, j] = a[i, j]

    # Subtraction should yield zeros despite different memory layouts
    var result = a - b
    assert_equal(result[0, 0], 0.0)
    assert_equal(result[0, 1], 0.0)
    assert_equal(result[1, 0], 0.0)
    assert_equal(result[1, 1], 0.0)


def main():
    test_runtime_and_compile_time_dim_and_stride(dynamic(120), static[512]())
    test_nested_layout_shape()
    test_transpose_arithmetic()
    test_different_layouts_arithmetic()
