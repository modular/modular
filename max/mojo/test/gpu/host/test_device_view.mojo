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
"""Tests the `Device` view returned by `DeviceContext.device()`."""

from max.gpu.host import DeviceAttribute, DeviceContext
from std.testing import assert_equal, assert_false, assert_true, TestSuite


def _multiprocessor_count(ctx: DeviceContext) raises -> Int:
    return ctx.device().get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)


def test_device_queries() raises:
    with DeviceContext() as ctx:
        var device = ctx.device()
        assert_equal(device.id, 0)
        assert_true(device.get_name().byte_length() > 0)
        assert_true(
            device.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT) > 0
        )
        _ = device.is_host_unified()
        assert_true(device.get_arch().byte_length() > 0)


def test_device_cannot_peer_itself() raises:
    with DeviceContext() as ctx:
        assert_false(ctx.device().can_access_peer(ctx.device()))


def test_device_cannot_peer_cpu() raises:
    var cpu = DeviceContext(api="cpu")
    with DeviceContext() as ctx:
        assert_false(ctx.device().can_access_peer(cpu.device()))


def test_device_through_borrowed_context() raises:
    with DeviceContext() as ctx:
        assert_true(_multiprocessor_count(ctx) > 0)


def test_device_from_temporary_context() raises:
    var maybe_ctx = Optional(DeviceContext())
    assert_true(
        maybe_ctx.value()
        .device()
        .get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
        > 0
    )


def test_device_id_follows_context() raises:
    if DeviceContext.number_of_devices() < 2:
        return
    with DeviceContext(1) as ctx:
        assert_equal(ctx.device().id, 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
