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
# `enqueue_copy_to` between buffers on two devices, run against both the AsyncRT
# and the HAL `DeviceContext`. The copy is enqueued on the destination's
# context, so the destination may be on a different device than the source.

from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_equal


def test_device_buffer_copy_to_other_device() raises:
    comptime N = 1024
    var ctx0 = DeviceContext(device_id=0)
    var ctx1 = DeviceContext(device_id=1)
    var src = ctx0.enqueue_create_buffer[.int32](N)
    with src.map_to_host() as host:
        for i in range(N):
            host[i] = Int32(i * 3)
    var dst = ctx1.enqueue_create_buffer[.int32](N)
    dst.enqueue_fill(0)
    src.enqueue_copy_to(dst)
    ctx1.synchronize()
    with dst.map_to_host() as host:
        for i in range(N):
            assert_equal(host[i], Int32(i * 3))


def test_host_buffer_copy_to_other_device() raises:
    comptime N = 1024
    var ctx0 = DeviceContext(device_id=0)
    var ctx1 = DeviceContext(device_id=1)
    var src = ctx0.enqueue_create_host_buffer[.int32](N)
    ctx0.synchronize()
    for i in range(N):
        src[i] = Int32(i + 5)
    var dst = ctx1.enqueue_create_buffer[.int32](N)
    dst.enqueue_fill(0)
    src.enqueue_copy_to(dst)
    ctx1.synchronize()
    with dst.map_to_host() as host:
        for i in range(N):
            assert_equal(host[i], Int32(i + 5))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
