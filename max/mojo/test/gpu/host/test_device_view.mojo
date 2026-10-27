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
"""Tests that `DeviceContext.device()` answers the same as the context."""

from max.gpu.host import DeviceAttribute, DeviceContext
from std.testing import assert_equal, assert_true, TestSuite


def _multiprocessor_count(ctx: DeviceContext) raises -> Int:
    return ctx.device().get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)


def test_device_matches_context() raises:
    with DeviceContext() as ctx:
        var device = ctx.device()
        assert_equal(device.get_name(), ctx.name())
        assert_equal(
            device.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT),
            ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT),
        )
        assert_equal(device.is_host_unified(), ctx.is_host_unified())


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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
