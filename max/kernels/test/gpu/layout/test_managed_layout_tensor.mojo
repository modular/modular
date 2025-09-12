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

from gpu.host import DeviceContext
from layout import Layout, RuntimeLayout, UNKNOWN_VALUE
from layout._utils import ManagedLayoutTensor
from layout.int_tuple import IntTuple
from testing import assert_equal
from utils import IndexList


# Tests for the ManagedLayoutTensor n-dimensional buffer support
# Verifies that device_buffer() and buffer() methods work correctly for various ranks


fn test_managed_layout_tensor_1d() raises:
    """Test 1D ManagedLayoutTensor buffer operations."""
    alias layout_1d = Layout(IntTuple(10))

    # Test with CPU context
    var cpu_tensor = ManagedLayoutTensor[DType.float32, layout_1d]()
    var host_buffer_1d = cpu_tensor.buffer[update=False]()
    assert_equal(host_buffer_1d.rank, 1)
    assert_equal(host_buffer_1d.dim[0](), 10)

    # Test with GPU context
    var gpu_ctx = DeviceContext()
    var gpu_tensor = ManagedLayoutTensor[DType.float32, layout_1d](gpu_ctx)
    var device_buffer_1d = gpu_tensor.device_buffer[update=False]()
    assert_equal(device_buffer_1d.rank, 1)
    assert_equal(device_buffer_1d.dim[0](), 10)


fn test_managed_layout_tensor_2d() raises:
    """Test 2D ManagedLayoutTensor buffer operations."""
    alias layout_2d = Layout(IntTuple(4, 6))

    # Test with CPU context
    var cpu_tensor = ManagedLayoutTensor[DType.float32, layout_2d]()
    var host_buffer_2d = cpu_tensor.buffer[update=False]()
    assert_equal(host_buffer_2d.rank, 2)
    assert_equal(host_buffer_2d.dim[0](), 4)
    assert_equal(host_buffer_2d.dim[1](), 6)

    # Test with GPU context
    var gpu_ctx = DeviceContext()
    var gpu_tensor = ManagedLayoutTensor[DType.float32, layout_2d](gpu_ctx)
    var device_buffer_2d = gpu_tensor.device_buffer[update=False]()
    assert_equal(device_buffer_2d.rank, 2)
    assert_equal(device_buffer_2d.dim[0](), 4)
    assert_equal(device_buffer_2d.dim[1](), 6)


fn test_managed_layout_tensor_3d() raises:
    """Test 3D ManagedLayoutTensor buffer operations."""
    alias layout_3d = Layout(IntTuple(2, 3, 4))

    # Test with CPU context
    var cpu_tensor = ManagedLayoutTensor[DType.float32, layout_3d]()
    var host_buffer_3d = cpu_tensor.buffer[update=False]()
    assert_equal(host_buffer_3d.rank, 3)
    assert_equal(host_buffer_3d.dim[0](), 2)
    assert_equal(host_buffer_3d.dim[1](), 3)
    assert_equal(host_buffer_3d.dim[2](), 4)

    # Test with GPU context
    var gpu_ctx = DeviceContext()
    var gpu_tensor = ManagedLayoutTensor[DType.float32, layout_3d](gpu_ctx)
    var device_buffer_3d = gpu_tensor.device_buffer[update=False]()
    assert_equal(device_buffer_3d.rank, 3)
    assert_equal(device_buffer_3d.dim[0](), 2)
    assert_equal(device_buffer_3d.dim[1](), 3)
    assert_equal(device_buffer_3d.dim[2](), 4)


fn test_managed_layout_tensor_dynamic() raises:
    """Test ManagedLayoutTensor with dynamic dimensions."""
    # Create layout with some dynamic dimensions
    alias layout_dynamic = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, 4)

    # Define runtime shape with actual values
    var runtime_shape = IndexList[3](5, 8, 4)
    var runtime_layout = RuntimeLayout[layout_dynamic].row_major(runtime_shape)

    # Test with CPU context
    var cpu_tensor = ManagedLayoutTensor[DType.float32, layout_dynamic](
        runtime_layout
    )
    var host_buffer_dynamic = cpu_tensor.buffer[update=False]()
    assert_equal(host_buffer_dynamic.rank, 3)
    assert_equal(host_buffer_dynamic.dim[0](), 5)
    assert_equal(host_buffer_dynamic.dim[1](), 8)
    assert_equal(host_buffer_dynamic.dim[2](), 4)

    # Test with GPU context
    var gpu_ctx = DeviceContext()
    var gpu_tensor = ManagedLayoutTensor[DType.float32, layout_dynamic](
        runtime_layout, gpu_ctx
    )
    var device_buffer_dynamic = gpu_tensor.device_buffer[update=False]()
    assert_equal(device_buffer_dynamic.rank, 3)
    assert_equal(device_buffer_dynamic.dim[0](), 5)
    assert_equal(device_buffer_dynamic.dim[1](), 8)
    assert_equal(device_buffer_dynamic.dim[2](), 4)


def main():
    """Main test function that runs all n-dimensional ManagedLayoutTensor tests.
    """
    test_managed_layout_tensor_1d()
    test_managed_layout_tensor_2d()
    test_managed_layout_tensor_3d()
    test_managed_layout_tensor_dynamic()
