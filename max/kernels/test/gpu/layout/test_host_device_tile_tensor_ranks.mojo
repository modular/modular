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

from max.gpu.host import DeviceContext
from layout import Idx, col_major, row_major
from layout._host_device_tile_tensor import HostDeviceTileTensor
from std.testing import assert_equal


def test_host_device_tile_tensor_1d() raises:
    """Checks rank and dimensions of 1D host and device views."""
    comptime layout_1d = row_major[10]()

    var cpu_tensor = HostDeviceTileTensor[.float32](layout_1d)
    var host_tensor_1d = cpu_tensor.host_tensor()
    assert_equal(comptime (host_tensor_1d.rank), 1)
    assert_equal(host_tensor_1d.dim[0](), 10)

    var gpu_ctx = DeviceContext()
    var gpu_tensor = HostDeviceTileTensor[.float32](layout_1d, gpu_ctx)
    var device_tensor_1d = gpu_tensor.device_tensor()
    assert_equal(comptime (device_tensor_1d.rank), 1)
    assert_equal(device_tensor_1d.dim[0](), 10)


def test_host_device_tile_tensor_2d() raises:
    """Checks rank and dimensions of 2D host and device views."""
    comptime layout_2d = col_major[4, 6]()

    var cpu_tensor = HostDeviceTileTensor[.float32](layout_2d)
    var host_tensor_2d = cpu_tensor.host_tensor()
    assert_equal(comptime (host_tensor_2d.rank), 2)
    assert_equal(host_tensor_2d.dim[0](), 4)
    assert_equal(host_tensor_2d.dim[1](), 6)

    var gpu_ctx = DeviceContext()
    var gpu_tensor = HostDeviceTileTensor[.float32](layout_2d, gpu_ctx)
    var device_tensor_2d = gpu_tensor.device_tensor()
    assert_equal(comptime (device_tensor_2d.rank), 2)
    assert_equal(device_tensor_2d.dim[0](), 4)
    assert_equal(device_tensor_2d.dim[1](), 6)


def test_host_device_tile_tensor_3d() raises:
    """Checks rank and dimensions of 3D host and device views."""
    comptime layout_3d = col_major[2, 3, 4]()

    var cpu_tensor = HostDeviceTileTensor[.float32](layout_3d)
    var host_tensor_3d = cpu_tensor.host_tensor()
    assert_equal(comptime (host_tensor_3d.rank), 3)
    assert_equal(host_tensor_3d.dim[0](), 2)
    assert_equal(host_tensor_3d.dim[1](), 3)
    assert_equal(host_tensor_3d.dim[2](), 4)

    var gpu_ctx = DeviceContext()
    var gpu_tensor = HostDeviceTileTensor[.float32](layout_3d, gpu_ctx)
    var device_tensor_3d = gpu_tensor.device_tensor()
    assert_equal(comptime (device_tensor_3d.rank), 3)
    assert_equal(device_tensor_3d.dim[0](), 2)
    assert_equal(device_tensor_3d.dim[1](), 3)
    assert_equal(device_tensor_3d.dim[2](), 4)


def test_host_device_tile_tensor_dynamic() raises:
    """Checks views with two dynamic dimensions and a static inner dimension."""
    var runtime_layout = row_major((5, 8, Idx[4]))

    var cpu_tensor = HostDeviceTileTensor[.float32](runtime_layout)
    var host_tensor_dynamic = cpu_tensor.host_tensor()
    assert_equal(comptime (host_tensor_dynamic.rank), 3)
    assert_equal(host_tensor_dynamic.dim[0](), 5)
    assert_equal(host_tensor_dynamic.dim[1](), 8)
    assert_equal(host_tensor_dynamic.dim[2](), 4)

    var gpu_ctx = DeviceContext()
    var gpu_tensor = HostDeviceTileTensor[.float32](runtime_layout, gpu_ctx)
    var device_tensor_dynamic = gpu_tensor.device_tensor()
    assert_equal(comptime (device_tensor_dynamic.rank), 3)
    assert_equal(device_tensor_dynamic.dim[0](), 5)
    assert_equal(device_tensor_dynamic.dim[1](), 8)
    assert_equal(device_tensor_dynamic.dim[2](), 4)


def main() raises:
    test_host_device_tile_tensor_1d()
    test_host_device_tile_tensor_2d()
    test_host_device_tile_tensor_3d()
    test_host_device_tile_tensor_dynamic()
