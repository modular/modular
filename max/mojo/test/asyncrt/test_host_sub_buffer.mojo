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

from asyncrt_test_utils import create_test_device_context
from std.testing import TestSuite, assert_equal, assert_raises


def test_host_sub_buffer() raises:
    var ctx = create_test_device_context()

    comptime length = 16

    var host = ctx.enqueue_create_host_buffer[.int32](2 * length)
    ctx.synchronize()
    for i in range(2 * length):
        host[i] = Int32(i)

    var tail = host.create_sub_buffer[.int32](length, length)
    assert_equal(len(tail), length)
    for i in range(length):
        assert_equal(tail[i], Int32(length + i))

    # The view aliases the parent's memory.
    tail[0] = -1
    assert_equal(host[length], -1)


def test_host_sub_buffer_reinterprets_dtype() raises:
    var ctx = create_test_device_context()

    var host = ctx.enqueue_create_host_buffer[.uint8](8)
    ctx.synchronize()
    for i in range(8):
        host[i] = 0

    # Byte offset 4 is element 1 of a uint32 view.
    var words = host.create_sub_buffer[.uint32](1, 1)
    words[0] = 0x01020304
    assert_equal(host[4], 0x04)
    assert_equal(host[7], 0x01)


def test_host_sub_buffer_outlives_parent() raises:
    var ctx = create_test_device_context()

    var host = ctx.enqueue_create_host_buffer[.int32](4)
    ctx.synchronize()
    for i in range(4):
        host[i] = Int32(i)

    var view = host.create_sub_buffer[.int32](2, 2)
    _ = host^
    assert_equal(view[0], 2)
    assert_equal(view[1], 3)


def test_host_sub_buffer_out_of_range() raises:
    var ctx = create_test_device_context()

    var host = ctx.enqueue_create_host_buffer[.int32](4)
    ctx.synchronize()

    with assert_raises():
        _ = host.create_sub_buffer[.int32](3, 2)
    with assert_raises():
        _ = host.create_sub_buffer[.int32](-1, 1)
    # `offset * size_of[Int32]()` wraps to -4 in `Int`.
    with assert_raises():
        _ = host.create_sub_buffer[.int32](4611686018427387903, 1)


def test_copy_into_host_sub_buffer_stays_in_view() raises:
    var ctx = create_test_device_context()

    comptime length = 8

    var host = ctx.enqueue_create_host_buffer[.int32](2 * length)
    var dev = ctx.enqueue_create_buffer[.int32](2 * length)
    ctx.synchronize()
    for i in range(2 * length):
        host[i] = -1
    with dev.map_to_host() as dev_map:
        for i in range(2 * length):
            dev_map[i] = Int32(i)

    # The source is larger than the view; only the view's elements are copied.
    var head = host.create_sub_buffer[.int32](0, length)
    dev.enqueue_copy_to(head)
    ctx.synchronize()
    for i in range(length):
        assert_equal(host[i], Int32(i))
        assert_equal(host[length + i], -1)


def test_host_sub_buffer_copy_from_smaller_source_raises() raises:
    var ctx = create_test_device_context()

    var host = ctx.enqueue_create_host_buffer[.int32](8)
    var other_host = ctx.enqueue_create_host_buffer[.int32](8)
    var small_dev = ctx.enqueue_create_buffer[.int32](2)
    var big_dev = ctx.enqueue_create_buffer[.int32](8)
    ctx.synchronize()

    var view = host.create_sub_buffer[.int32](0, 4)
    with assert_raises():
        small_dev.enqueue_copy_to(view)
    with assert_raises():
        big_dev.enqueue_copy_from(view)
    with assert_raises():
        ctx.enqueue_copy(other_host, view)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
