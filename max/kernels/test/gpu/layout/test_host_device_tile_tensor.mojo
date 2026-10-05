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

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import ComptimeInt, Idx, RowMajorLayout, TileTensor, row_major
from layout._host_device_tile_tensor import HostDeviceTileTensor
from std.testing import assert_equal, assert_raises

comptime N = 16
comptime VecLayout = RowMajorLayout[ComptimeInt[N]]


def add_one_kernel(t: TileTensor[.float32, VecLayout, MutAnyOrigin]):
    var i = Int(global_idx.x)
    if i < N:
        t[i] += 1


def test_static_layout_dims() raises:
    var ctx = DeviceContext()
    var t = HostDeviceTileTensor[.float32](row_major[4, 6](), ctx)
    var host = t.host_tensor()
    var device = t.device_tensor()
    assert_equal(host.dim[0](), 4)
    assert_equal(host.dim[1](), 6)
    assert_equal(host.num_elements(), 24)
    assert_equal(device.dim[0](), 4)
    assert_equal(device.dim[1](), 6)


def test_runtime_layout_dims() raises:
    var ctx = DeviceContext()
    var m = 5
    var t = HostDeviceTileTensor[.float32](row_major(m, Idx[4]), ctx)
    assert_equal(t.host_tensor().dim[0](), 5)
    assert_equal(t.device_tensor().dim[1](), 4)
    assert_equal(t.device_tensor().num_elements(), 20)


def test_round_trip() raises:
    var ctx = DeviceContext()
    var t = HostDeviceTileTensor[.float32](row_major[N](), ctx)
    var host = t.host_tensor()
    for i in range(N):
        host[i] = Float32(i)
    t.to_device()
    ctx.enqueue_function[add_one_kernel](
        t.device_tensor(), grid_dim=1, block_dim=N
    )
    t.to_host()
    for i in range(N):
        assert_equal(host[i], Float32(i + 1))


def test_views_do_not_copy() raises:
    var ctx = DeviceContext()
    var t = HostDeviceTileTensor[.float32](row_major[N](), ctx)
    var host = t.host_tensor()
    for i in range(N):
        host[i] = Float32(i)
    t.to_device()
    for i in range(N):
        host[i] = -1
    _ = t.device_tensor()
    _ = t.host_tensor()
    t.to_host()
    for i in range(N):
        assert_equal(host[i], Float32(i))


def test_host_only() raises:
    var t = HostDeviceTileTensor[.float32](row_major[N]())
    var host = t.host_tensor()
    for i in range(N):
        host[i] = Float32(i)
    t.to_device()
    t.to_host()
    for i in range(N):
        assert_equal(host[i], Float32(i))
    with assert_raises(contains="no device buffer"):
        _ = t.device_tensor()


def main() raises:
    test_static_layout_dims()
    test_runtime_layout_dims()
    test_round_trip()
    test_views_do_not_copy()
    test_host_only()
