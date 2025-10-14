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
from layout import Layout, LayoutTensor, RuntimeLayout
from layout.layout import blocked_product
from testing import assert_equal

from utils.index import IndexList


fn test_runtime_and_compile_time_dim_and_stride(
    m: ValOrDim, k: ValOrDim
) raises:
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


def main():
    test_runtime_and_compile_time_dim_and_stride(dynamic(120), static[512]())
    test_nested_layout_shape()
