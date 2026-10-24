# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Mojo HAL operations."""

from std.memory import alloc, Pointer
from std.os import getenv
from std.testing import assert_equal, assert_raises, assert_true

from max._driver import Driver, Context
from max._driver.max_license.hal.device import DeviceSpec
from max._driver.max_license.hal.event import EVENT_FLAG_CPU_VISIBLE


def do_queue_stream_sync_test[
    device_spec: DeviceSpec,
](mut ctx: Context[device_spec]) raises:
    comptime n = 4096
    var device_buffer = ctx.alloc_sync(UInt64(n))
    var host = alloc[UInt8]({length = n}).unsafe_leak()

    var queue = ctx.create_queue()
    queue[].set_memory(device_buffer.view(), value=UInt8(0x5A))
    queue[].copy_from_device(host.unsafe_bitcast[UInt8](), device_buffer.view())
    queue[].synchronize()
    for i in range(n):
        assert_equal(host[unsafe_offset=i], UInt8(0x5A))

    # Metal reports `queue_is_stream` false, so the stream path is a genuinely
    # different submission route there rather than the same object renamed.
    var stream = ctx.create_stream()
    stream[].set_memory(device_buffer.view(), value=UInt8(0xA5))
    stream[].copy_from_device(
        host.unsafe_bitcast[UInt8](), device_buffer.view()
    )
    stream[].synchronize()
    for i in range(n):
        assert_equal(host[unsafe_offset=i], UInt8(0xA5))

    host.unsafe_free()
    ctx.free_sync(device_buffer^)


def do_event_ordering_test[
    device_spec: DeviceSpec,
](mut ctx: Context[device_spec]) raises:
    """Events gate work across queues, and a host-visible event reports ready.

    The mock version recorded an event and waited on it with no work in
    between and nothing asserted afterwards, so a `wait_for_events` that
    returned success without waiting passed it. Both halves below fail against
    such a plugin.

    On Metal `queue_is_stream` is false and two queues may serialize onto one
    command queue, so the ordering half proves ordering on CUDA and HIP and
    only non-regression there.
    """
    # Large enough that the fill is not instantaneous, so a consumer that
    # skips the wait reads the pre-fill pattern rather than racing to a pass.
    comptime n = 64 * 1024 * 1024
    # An ordering failure leaves the whole buffer at the pre-fill value, so a
    # strided sample catches it without 64M assert calls.
    comptime stride = 64 * 1024

    var device_buffer = ctx.alloc_sync(UInt64(n))
    var host = alloc[UInt8]({length = n}).unsafe_leak()

    var producer = ctx.create_queue()
    var consumer = ctx.create_queue()

    producer[].set_memory(device_buffer.view(), value=UInt8(0x00))
    producer[].synchronize()

    producer[].set_memory(device_buffer.view(), value=UInt8(0x77))
    var ordering_event = producer[].record_event()

    consumer[].wait_for_events(ordering_event)
    consumer[].copy_from_device(
        host.unsafe_bitcast[UInt8](), device_buffer.view()
    )
    consumer[].synchronize()
    for i in range(0, n, stride):
        assert_equal(host[unsafe_offset=i], UInt8(0x77))
    assert_equal(host[unsafe_offset=n - 1], UInt8(0x77))

    # Deterministic, unlike the race above: once the host wait returns the
    # event must report ready.
    producer[].set_memory(device_buffer.view(), value=UInt8(0x11))
    var host_event = producer[].record_event[EVENT_FLAG_CPU_VISIBLE]()
    host_event.synchronize()
    assert_true(
        host_event.is_ready(), "host-visible event not ready after synchronize"
    )

    # One variadic wait mixing default and host-visible flags.
    var plain_event = producer[].record_event()
    var visible_event = producer[].record_event[EVENT_FLAG_CPU_VISIBLE]()
    consumer[].wait_for_events(plain_event, visible_event)
    consumer[].synchronize()

    # The same ordering contract across two streams.
    var producer_stream = ctx.create_stream()
    var consumer_stream = ctx.create_stream()

    producer_stream[].set_memory(device_buffer.view(), value=UInt8(0x00))
    producer_stream[].synchronize()

    producer_stream[].set_memory(device_buffer.view(), value=UInt8(0x99))
    var stream_event = producer_stream[].record_event()

    consumer_stream[].wait_for_events(stream_event)
    consumer_stream[].copy_from_device(
        host.unsafe_bitcast[UInt8](), device_buffer.view()
    )
    consumer_stream[].synchronize()
    for i in range(0, n, stride):
        assert_equal(host[unsafe_offset=i], UInt8(0x99))
    assert_equal(host[unsafe_offset=n - 1], UInt8(0x99))

    host.unsafe_free()
    ctx.free_sync(device_buffer^)


def do_copy_overflow_test[
    device_spec: DeviceSpec,
](mut ctx: Context[device_spec]) raises:
    """A copy whose destination cannot hold the source is rejected."""
    var queue = ctx.create_queue()
    var small = ctx.alloc_sync(UInt64(16))
    var big = ctx.alloc_sync(UInt64(32))

    with assert_raises(contains="copy overflow"):
        queue[].copy(dst=small, src=big)

    ctx.free_sync(small^)
    ctx.free_sync(big^)


def do_invalid_device_test(driver: Driver) raises:
    """Device IDs are range-checked against the real device count."""
    with assert_raises(contains="Invalid device ID"):
        _ = driver.get_device[Int64(99)]()


def main() raises:
    var plugin_spec = getenv("MODULAR_DRIVER_PLUGINS")
    if not plugin_spec:
        raise Error("MODULAR_DRIVER_PLUGINS not set")

    var driver = Driver.create(plugin_spec)
    print("Driver:", driver[].get_name())
    print("Devices:", driver[].get_device_count())

    do_invalid_device_test(driver[])

    var device = driver[].get_device[0]()
    var ctx = device[].get_context()

    do_queue_stream_sync_test(ctx[])
    do_event_ordering_test(ctx[])
    do_copy_overflow_test(ctx[])
    print("All HAL ops tests OK")
