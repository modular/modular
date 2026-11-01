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
"""Tests for the Mojo HAL bindings using a mock plugin.

Exercises the full HAL lifecycle (Driver -> Device -> Context) against a
lightweight C mock that implements the M_driver_* API without GPU hardware.
This catches FFI signature mismatches, status code drift, and basic error
handling paths.
"""

from std.os import getenv
from std.memory import alloc
from max._driver import Driver, copy, fill, set_memory
from max._driver.max_license.hal.context import Context
from max._driver.max_license.hal.device import Device, get_device_spec
from max._driver.max_license.hal.queue import Queue
from max._driver.max_license.hal.stream import Stream
from max._driver.max_license.hal.event import Event, EVENT_FLAG_CPU_VISIBLE
from max._driver.max_license.hal.extensions.traits import SupportsPeerAccessV1
from max._driver.max_license.hal.status import (
    STATUS_SUCCESS,
    STATUS_INCOMPATIBLE_VERSION,
    STATUS_UNKNOWN_ERROR,
    STATUS_ARGUMENT_OUT_OF_RANGE,
    STATUS_UNKNOWN_PROPERTY_NAME,
    STATUS_NO_DEVICES,
    STATUS_BAD_DRIVER,
    STATUS_BAD_DEVICE,
    STATUS_BAD_CONTEXT,
    STATUS_BAD_MEMORY,
    STATUS_BAD_QUEUE,
    STATUS_BAD_EVENT,
    STATUS_BAD_FUNCTION,
    STATUS_OUT_OF_MEMORY,
    STATUS_INVALID_ARG,
    STATUS_INCOMPATIBLE,
    STATUS_BAD_BUNDLE,
    STATUS_EXECUTE_MODE_NOT_SUPPORTED,
    STATUS_UNSUPPORTED,
    STATUS_UNIMPLEMENTED,
)
from std.testing import assert_equal, assert_false, assert_raises, assert_true


def _get_plugin_spec() -> String:
    return getenv("MODULAR_DRIVER_PLUGINS")


# ===-----------------------------------------------------------------------===#
# Status code value tests
# ===-----------------------------------------------------------------------===#


def test_status_code_values() raises:
    assert_equal(Int(STATUS_SUCCESS), 0)
    assert_equal(Int(STATUS_INCOMPATIBLE_VERSION), -1)
    assert_equal(Int(STATUS_UNKNOWN_ERROR), -2)
    assert_equal(Int(STATUS_ARGUMENT_OUT_OF_RANGE), -3)
    assert_equal(Int(STATUS_UNKNOWN_PROPERTY_NAME), -4)
    assert_equal(Int(STATUS_NO_DEVICES), -5)
    assert_equal(Int(STATUS_BAD_DRIVER), -6)
    assert_equal(Int(STATUS_BAD_DEVICE), -7)
    assert_equal(Int(STATUS_BAD_CONTEXT), -8)
    assert_equal(Int(STATUS_BAD_MEMORY), -9)
    assert_equal(Int(STATUS_BAD_QUEUE), -10)
    assert_equal(Int(STATUS_BAD_EVENT), -11)
    assert_equal(Int(STATUS_BAD_FUNCTION), -12)
    assert_equal(Int(STATUS_OUT_OF_MEMORY), -13)
    assert_equal(Int(STATUS_INVALID_ARG), -14)
    assert_equal(Int(STATUS_INCOMPATIBLE), -15)
    assert_equal(Int(STATUS_BAD_BUNDLE), -16)
    assert_equal(Int(STATUS_EXECUTE_MODE_NOT_SUPPORTED), -17)
    assert_equal(Int(STATUS_UNSUPPORTED), -19)
    assert_equal(Int(STATUS_UNIMPLEMENTED), -20)


# ===-----------------------------------------------------------------------===#
# Driver lifecycle tests
# ===-----------------------------------------------------------------------===#


def test_driver_create() raises:
    var device_count: Int
    try:
        var driver = Driver.create(_get_plugin_spec())
        device_count = Int(driver[].get_device_count())
    except e:
        raise Error(String(e))
    assert_equal(device_count, 2)


def test_driver_device_enumeration() raises:
    var ok = True
    try:
        var device_count: Int
        var driver = Driver.create(_get_plugin_spec())
        device_count = Int(driver[].get_device_count())
        for i in range(device_count):
            var device = driver[].get_device_dynamic(Int64(i))
            if Int(device[].id) != i:
                ok = False
    except e:
        raise Error(String(e))
    assert_true(ok, "device ID mismatch during enumeration")


def test_device_host_unified() raises:
    # The mock models a discrete accelerator.
    var driver = Driver.create(_get_plugin_spec())
    var device = driver[].get_device_dynamic(Int64(0))
    assert_false(device[].is_host_unified())


def test_invalid_device_id() raises:
    var driver = Driver.create(_get_plugin_spec())
    with assert_raises():
        _ = driver[].get_device[Int64(99)]()


def test_context_create() raises:
    try:
        var driver = Driver.create(_get_plugin_spec())
        var device = driver[].get_device[Int64(0)]()
        var ctx = device[].get_context()
        # Let ctx go out of scope to exercise Context destructor.
    except e:
        raise Error(String(e))


def test_invalid_device_error_message() raises:
    var driver = Driver.create(_get_plugin_spec())
    with assert_raises(contains="Invalid device ID"):
        _ = driver[].get_device[Int64(99)]()


def test_memory_wrap_unwrap() raises:
    var driver = Driver.create(_get_plugin_spec())
    var device = driver[].get_device[Int64(0)]()
    var ctx = device[].get_context()
    var buf = ctx[].alloc_sync(64)
    var addr = ctx[].memory_get_address(buf)
    var wrapped = ctx[].wrap_memory(addr, 64)
    assert_equal(ctx[].memory_get_address(wrapped), addr)
    assert_equal(ctx[].unwrap_memory(wrapped), addr)
    # `wrapped` is non-owning; dropping it frees nothing, and `buf` still
    # owns the allocation.
    _ = wrapped^
    assert_equal(ctx[].memory_get_address(buf), addr)


def test_queue_native_handle() raises:
    # The queue exposes its backend handle as a nonzero integer, the value
    # lent to a DLPack producer for stream-scoped import ordering.
    var driver = Driver.create(_get_plugin_spec())
    var device = driver[].get_device[Int64(0)]()
    var ctx = device[].get_context()
    var queue = ctx[].create_queue()
    var handle = queue[].native_handle()
    assert_true(Bool(handle))
    assert_true(handle.value() != 0)


# ===-----------------------------------------------------------------------===#
# Queue tests
# ===-----------------------------------------------------------------------===#


def test_queue_create_and_sync() raises:
    try:
        var driver = Driver.create(_get_plugin_spec())
        var device = driver[].get_device[Int64(0)]()
        var ctx = device[].get_context()
        var queue = ctx[].create_queue()
        queue[].synchronize()
    except e:
        raise Error(String(e))


def test_queue_record_event() raises:
    try:
        var driver = Driver.create(_get_plugin_spec())
        var device = driver[].get_device[Int64(0)]()
        var ctx = device[].get_context()
        var queue = ctx[].create_queue()
        var event = queue[].record_event()
        queue[].wait_for_events(event)
        queue[].synchronize()
    except e:
        raise Error(String(e))


def test_queue_wait_for_events() raises:
    try:
        var driver = Driver.create(_get_plugin_spec())
        var device = driver[].get_device[Int64(0)]()
        var ctx = device[].get_context()
        var queue1 = ctx[].create_queue()
        var queue2 = ctx[].create_queue()
        var e1 = queue1[].record_event()
        var e2 = queue1[].record_event()
        queue2[].wait_for_events(e1, e2)
        queue2[].synchronize()
    except e:
        raise Error(String(e))


def test_queue_wait_for_events_mixed_flags() raises:
    """Variadic accepts events with different flag combos in one call."""
    try:
        var driver = Driver.create(_get_plugin_spec())
        var device = driver[].get_device[Int64(0)]()
        var ctx = device[].get_context()
        var queue1 = ctx[].create_queue()
        var queue2 = ctx[].create_queue()
        var e_intra = queue1[].record_event()
        var e_host = queue1[].record_event[EVENT_FLAG_CPU_VISIBLE]()
        queue2[].wait_for_events(e_intra, e_host)
        queue2[].synchronize()
    except e:
        raise Error(String(e))


# ===-----------------------------------------------------------------------===#
# Stream tests
# ===-----------------------------------------------------------------------===#


def test_stream_create_and_sync() raises:
    try:
        var driver = Driver.create(_get_plugin_spec())
        var device = driver[].get_device[Int64(0)]()
        var ctx = device[].get_context()
        var stream = ctx[].create_stream()
        stream[].synchronize()
    except e:
        raise Error(String(e))


def test_stream_record_event() raises:
    try:
        var driver = Driver.create(_get_plugin_spec())
        var device = driver[].get_device[Int64(0)]()
        var ctx = device[].get_context()
        var stream = ctx[].create_stream()
        var event = stream[].record_event()
        stream[].wait_for_events(event)
        stream[].synchronize()
    except e:
        raise Error(String(e))


def test_stream_wait_for_events() raises:
    try:
        var driver = Driver.create(_get_plugin_spec())
        var device = driver[].get_device[Int64(0)]()
        var ctx = device[].get_context()
        var producer = ctx[].create_stream()
        var consumer = ctx[].create_stream()
        var ev = producer[].record_event()
        consumer[].wait_for_events(ev)
        consumer[].synchronize()
    except e:
        raise Error(String(e))


# ===-----------------------------------------------------------------------===#
# Unified copy tests
# ===-----------------------------------------------------------------------===#


def test_unified_copy_device_to_device_on_queue() raises:
    """`Queue.copy` does a device-to-device copy between two buffers."""
    var driver = Driver.create(_get_plugin_spec())
    var device = driver[].get_device[Int64(0)]()
    var ctx = device[].get_context()
    var queue = ctx[].create_queue()

    comptime n = 64
    var src = alloc[UInt8]({length = n}).unsafe_leak()
    var dst = alloc[UInt8]({length = n}).unsafe_leak()
    for i in range(n):
        src[unsafe_offset=i] = UInt8(i)
        dst[unsafe_offset=i] = UInt8(0)

    var a_dev = ctx[].alloc_sync(UInt64(n))
    var b_dev = ctx[].alloc_sync(UInt64(n))

    # Stage host bytes into a_dev with the low-level host transfer, copy them
    # device-to-device with the unified copy, then read them back.
    queue[].copy_to_device(a_dev.view(), src)
    queue[].copy(dst=b_dev, src=a_dev)
    queue[].copy_from_device(dst, b_dev.view())
    queue[].synchronize()

    for i in range(n):
        assert_equal(dst[unsafe_offset=i], src[unsafe_offset=i])

    src.unsafe_free()
    dst.unsafe_free()
    ctx[].free_sync(a_dev^)
    ctx[].free_sync(b_dev^)


def test_unified_copy_sync_free_function() raises:
    """The free `copy(dst, src)` copies device-to-device via its own queue."""
    var driver = Driver.create(_get_plugin_spec())
    var device = driver[].get_device[Int64(0)]()
    var ctx = device[].get_context()
    var queue = ctx[].create_queue()

    comptime n = 32
    var src = alloc[UInt8]({length = n}).unsafe_leak()
    var dst = alloc[UInt8]({length = n}).unsafe_leak()
    for i in range(n):
        src[unsafe_offset=i] = UInt8(i * 2)
        dst[unsafe_offset=i] = UInt8(0)

    var a_dev = ctx[].alloc_sync(UInt64(n))
    var b_dev = ctx[].alloc_sync(UInt64(n))

    queue[].copy_to_device(a_dev.view(), src)
    queue[].synchronize()
    copy(dst=b_dev, src=a_dev)  # device -> device, synchronous
    queue[].copy_from_device(dst, b_dev.view())
    queue[].synchronize()

    for i in range(n):
        assert_equal(dst[unsafe_offset=i], src[unsafe_offset=i])

    src.unsafe_free()
    dst.unsafe_free()
    ctx[].free_sync(a_dev^)
    ctx[].free_sync(b_dev^)


def test_sync_set_memory_and_fill_free_functions() raises:
    """The free `set_memory` / `fill` write a device buffer synchronously."""
    var driver = Driver.create(_get_plugin_spec())
    var device = driver[].get_device[Int64(0)]()
    var ctx = device[].get_context()
    var queue = ctx[].create_queue()

    comptime n = 32
    var out = alloc[UInt8]({length = n}).unsafe_leak()
    var dev = ctx[].alloc_sync(UInt64(n))

    set_memory(dst=dev, value=UInt8(0x11))
    queue[].copy_from_device(out, dev.view())
    queue[].synchronize()
    for i in range(n):
        assert_equal(out[unsafe_offset=i], UInt8(0x11))

    fill(dst=dev, value=UInt64(0x22222222), value_size=UInt64(4))
    queue[].copy_from_device(out, dev.view())
    queue[].synchronize()
    for i in range(n):
        assert_equal(out[unsafe_offset=i], UInt8(0x22))

    out.unsafe_free()
    ctx[].free_sync(dev^)


def test_unified_copy_on_stream() raises:
    """`Stream.copy` does a device-to-device copy in submission order."""
    var driver = Driver.create(_get_plugin_spec())
    var device = driver[].get_device[Int64(0)]()
    var ctx = device[].get_context()
    var stream = ctx[].create_stream()

    comptime n = 48
    var src = alloc[UInt8]({length = n}).unsafe_leak()
    var dst = alloc[UInt8]({length = n}).unsafe_leak()
    for i in range(n):
        src[unsafe_offset=i] = UInt8(255 - i)
        dst[unsafe_offset=i] = UInt8(0)

    var a_dev = ctx[].alloc_sync(UInt64(n))
    var b_dev = ctx[].alloc_sync(UInt64(n))

    stream[].copy_to_device(a_dev.view(), src)
    stream[].copy(dst=b_dev, src=a_dev)  # device -> device
    stream[].copy_from_device(dst, b_dev.view())
    stream[].synchronize()

    for i in range(n):
        assert_equal(dst[unsafe_offset=i], src[unsafe_offset=i])

    src.unsafe_free()
    dst.unsafe_free()
    ctx[].free_sync(a_dev^)
    ctx[].free_sync(b_dev^)


def test_unified_copy_dst_too_small_raises() raises:
    """A copy whose destination cannot hold the source raises."""
    var driver = Driver.create(_get_plugin_spec())
    var device = driver[].get_device[Int64(0)]()
    var ctx = device[].get_context()
    var queue = ctx[].create_queue()

    var small = ctx[].alloc_sync(UInt64(16))
    var big = ctx[].alloc_sync(UInt64(32))

    with assert_raises(contains="copy overflow"):
        queue[].copy(dst=small, src=big)

    ctx[].free_sync(small^)
    ctx[].free_sync(big^)


def test_unified_copy_front_fill() raises:
    """A smaller src lands in the front of a larger dst; the tail is kept."""
    var driver = Driver.create(_get_plugin_spec())
    var device = driver[].get_device[Int64(0)]()
    var ctx = device[].get_context()
    var queue = ctx[].create_queue()

    comptime big_n = 32
    comptime small_n = 16
    var small = ctx[].alloc_sync(UInt64(small_n))
    var big = ctx[].alloc_sync(UInt64(big_n))
    var host = alloc[UInt8]({length = small_n}).unsafe_leak()
    var out = alloc[UInt8]({length = big_n}).unsafe_leak()
    for i in range(small_n):
        host[unsafe_offset=i] = UInt8(i + 1)

    # Fill `big` with 0xBB, stage the 16-byte payload into `small`, copy
    # `small` into the front of `big`, and read it back: front = payload,
    # tail = 0xBB (untouched).
    queue[].set_memory(big.view(), UInt8(0xBB))
    queue[].copy_to_device(small.view(), host)
    queue[].copy(dst=big, src=small)
    queue[].copy_from_device(out, big.view())
    queue[].synchronize()
    for i in range(small_n):
        assert_equal(out[unsafe_offset=i], UInt8(i + 1))
    for i in range(small_n, big_n):
        assert_equal(out[unsafe_offset=i], UInt8(0xBB))

    host.unsafe_free()
    out.unsafe_free()
    ctx[].free_sync(big^)
    ctx[].free_sync(small^)


def test_unified_copy_pinned() raises:
    """Pinned host buffers route through host transports in every direction."""
    var driver = Driver.create(_get_plugin_spec())
    var device = driver[].get_device[Int64(0)]()
    var ctx = device[].get_context()
    var queue = ctx[].create_queue()

    comptime n = 32
    var pinned_in = ctx[].alloc_host_pinned(UInt64(n))
    var pinned_out = ctx[].alloc_host_pinned(UInt64(n))
    var dev = ctx[].alloc_sync(UInt64(n))

    # A device buffer has no host-dereferenceable address; the plugin refuses
    # rather than hand back a device pointer that would fault on the host.
    with assert_raises(contains="host_address"):
        _ = ctx[].memory_get_host_address[MutAnyOrigin](dev)

    # Pinned memory is CPU-addressable in place: write the input and read the
    # output directly through the host address, with no copies.
    var in_ptr = ctx[].memory_get_host_address[MutAnyOrigin](pinned_in)
    var out_ptr = ctx[].memory_get_host_address[MutAnyOrigin](pinned_out)
    for i in range(n):
        in_ptr[unsafe_offset=i] = UInt8(i + 3)
        out_ptr[unsafe_offset=i] = UInt8(0)

    # pinned -> device -> pinned through the queue.
    queue[].copy(dst=dev, src=pinned_in)
    queue[].copy(dst=pinned_out, src=dev)
    queue[].synchronize()
    for i in range(n):
        assert_equal(out_ptr[unsafe_offset=i], UInt8(i + 3))

    # pinned -> pinned is purely host-side and cannot be enqueued...
    with assert_raises(contains="cannot be enqueued"):
        queue[].copy(dst=pinned_out, src=pinned_in)

    # ...but the synchronous free `copy` performs it as a memcpy.
    for i in range(n):
        out_ptr[unsafe_offset=i] = UInt8(0)
    copy(dst=pinned_out, src=pinned_in)
    for i in range(n):
        assert_equal(out_ptr[unsafe_offset=i], UInt8(i + 3))

    ctx[].free_host_pinned(pinned_in^)
    ctx[].free_host_pinned(pinned_out^)
    ctx[].free_sync(dev^)


def test_unified_copy_free_function_pinned_device() raises:
    var driver = Driver.create(_get_plugin_spec())
    var device = driver[].get_device[Int64(0)]()
    var ctx = device[].get_context()

    comptime n = 24
    var pinned_in = ctx[].alloc_host_pinned(UInt64(n))
    var pinned_out = ctx[].alloc_host_pinned(UInt64(n))
    var dev = ctx[].alloc_sync(UInt64(n))

    # Pinned memory is host-accessible, so seed the input and clear the output
    # in place through their host addresses — no copy needed.
    var in_ptr = ctx[].memory_get_host_address[MutAnyOrigin](pinned_in)
    var out_ptr = ctx[].memory_get_host_address[MutAnyOrigin](pinned_out)
    for i in range(n):
        in_ptr[unsafe_offset=i] = UInt8(200 - i)
        out_ptr[unsafe_offset=i] = UInt8(0)

    copy(dst=dev, src=pinned_in)
    copy(dst=pinned_out, src=dev)
    for i in range(n):
        assert_equal(out_ptr[unsafe_offset=i], UInt8(200 - i))

    ctx[].free_host_pinned(pinned_in^)
    ctx[].free_host_pinned(pinned_out^)
    ctx[].free_sync(dev^)


def test_unified_copy_inter_device_on_queue() raises:
    """`Queue.copy` peer-copies a buffer from one device to another."""
    var driver = Driver.create(_get_plugin_spec())
    # Two devices sharing one arch spec but with different runtime ids — the
    # shape of same-arch multi-GPU (one target arch, several physical devices).
    # The unified `copy` requires both operands to share a comptime device_spec,
    # while the peer path is selected by the differing runtime device ids.
    comptime spec = get_device_spec[Int64(0)]()
    var device0 = Device[spec]._create(driver[], Int64(0))
    var device1 = Device[spec]._create(driver[], Int64(1))
    var ctx0 = device0[].get_context()
    var ctx1 = device1[].get_context()
    var queue0 = ctx0[].create_queue()
    var queue1 = ctx1[].create_queue()

    comptime n = 64
    var host_src = alloc[UInt8]({length = n}).unsafe_leak()
    var host_out = alloc[UInt8]({length = n}).unsafe_leak()
    for i in range(n):
        host_src[unsafe_offset=i] = UInt8(i + 1)
        host_out[unsafe_offset=i] = UInt8(0)

    var src = ctx0[].alloc_sync(UInt64(n))  # device 0
    var dst = ctx1[].alloc_sync(UInt64(n))  # device 1

    # Stage the payload onto device 0 and synchronize (the peer copy does not
    # order against the producer queue), peer-copy it to device 1 on device
    # 1's queue, then read it back.
    queue0[].copy_to_device(src.view(), host_src)
    queue0[].synchronize()
    queue1[].copy(dst=dst, src=src)  # device 0 -> device 1
    queue1[].copy_from_device(host_out, dst.view())
    queue1[].synchronize()

    for i in range(n):
        assert_equal(host_out[unsafe_offset=i], UInt8(i + 1))

    host_src.unsafe_free()
    host_out.unsafe_free()
    ctx0[].free_sync(src^)
    ctx1[].free_sync(dst^)


def test_unified_copy_inter_device_free_function() raises:
    """The free `copy` peer-copies between buffers on different devices."""
    var driver = Driver.create(_get_plugin_spec())
    # Two devices sharing one arch spec but with different runtime ids — the
    # shape of same-arch multi-GPU (one target arch, several physical devices).
    # The unified `copy` requires both operands to share a comptime device_spec,
    # while the peer path is selected by the differing runtime device ids.
    comptime spec = get_device_spec[Int64(0)]()
    var device0 = Device[spec]._create(driver[], Int64(0))
    var device1 = Device[spec]._create(driver[], Int64(1))
    var ctx0 = device0[].get_context()
    var ctx1 = device1[].get_context()
    var queue0 = ctx0[].create_queue()
    var queue1 = ctx1[].create_queue()

    comptime n = 40
    var host_src = alloc[UInt8]({length = n}).unsafe_leak()
    var host_out = alloc[UInt8]({length = n}).unsafe_leak()
    for i in range(n):
        host_src[unsafe_offset=i] = UInt8(i * 3)
        host_out[unsafe_offset=i] = UInt8(0)

    var src = ctx0[].alloc_sync(UInt64(n))  # device 0
    var dst = ctx1[].alloc_sync(UInt64(n))  # device 1

    queue0[].copy_to_device(src.view(), host_src)
    queue0[].synchronize()
    # Synchronous peer copy; its queue is derived from `dst`'s context, so it
    # runs on device 1.
    copy(dst=dst, src=src)
    queue1[].copy_from_device(host_out, dst.view())
    queue1[].synchronize()

    for i in range(n):
        assert_equal(host_out[unsafe_offset=i], UInt8(i * 3))

    host_src.unsafe_free()
    host_out.unsafe_free()
    ctx0[].free_sync(src^)
    ctx1[].free_sync(dst^)


def test_unified_copy_inter_device_dst_not_on_queue_raises() raises:
    """A peer copy whose destination is not on the queue's device raises."""
    var driver = Driver.create(_get_plugin_spec())
    # Two devices sharing one arch spec but with different runtime ids — the
    # shape of same-arch multi-GPU (one target arch, several physical devices).
    # The unified `copy` requires both operands to share a comptime device_spec,
    # while the peer path is selected by the differing runtime device ids.
    comptime spec = get_device_spec[Int64(0)]()
    var device0 = Device[spec]._create(driver[], Int64(0))
    var device1 = Device[spec]._create(driver[], Int64(1))
    var ctx0 = device0[].get_context()
    var ctx1 = device1[].get_context()
    var queue0 = ctx0[].create_queue()  # device 0

    comptime n = 32
    var src = ctx0[].alloc_sync(UInt64(n))  # device 0
    var dst = ctx1[].alloc_sync(UInt64(n))  # device 1

    # `dst` lives on device 1, but the queue belongs to device 0, so the
    # plugin would use the wrong destination context for the peer copy.
    with assert_raises(contains="queue's device"):
        queue0[].copy(dst=dst, src=src)

    ctx0[].free_sync(src^)
    ctx1[].free_sync(dst^)


def test_unified_copy_pinned_cross_device() raises:
    """The free `copy` moves data between a pinned host buffer and a device
    buffer on a DIFFERENT device. The pinned side is only a host pointer, so
    the transfer runs on the device operand's context regardless of which
    device allocated the pinned buffer — no same-device restriction applies.
    """
    var driver = Driver.create(_get_plugin_spec())
    comptime spec = get_device_spec[Int64(0)]()
    var device0 = Device[spec]._create(driver[], Int64(0))
    var device1 = Device[spec]._create(driver[], Int64(1))
    var ctx0 = device0[].get_context()
    var ctx1 = device1[].get_context()

    comptime n = 32
    var pinned_in = ctx0[].alloc_host_pinned(UInt64(n))  # device 0, pinned
    var pinned_out = ctx0[].alloc_host_pinned(UInt64(n))  # device 0, pinned
    var dev = ctx1[].alloc_sync(UInt64(n))  # device 1, device-resident

    var in_ptr = ctx0[].memory_get_host_address[MutAnyOrigin](pinned_in)
    var out_ptr = ctx0[].memory_get_host_address[MutAnyOrigin](pinned_out)
    for i in range(n):
        in_ptr[unsafe_offset=i] = UInt8(100 + i)
        out_ptr[unsafe_offset=i] = UInt8(0)

    # pinned (device 0) -> device (device 1): H2D on device 1's context.
    copy(dst=dev, src=pinned_in)
    # device (device 1) -> pinned (device 0): D2H on device 1's context.
    copy(dst=pinned_out, src=dev)
    for i in range(n):
        assert_equal(out_ptr[unsafe_offset=i], UInt8(100 + i))

    ctx0[].free_host_pinned(pinned_in^)
    ctx0[].free_host_pinned(pinned_out^)
    ctx1[].free_sync(dev^)


def test_unified_copy_pinned_cross_device_on_queue() raises:
    """`Queue.copy` moves data between a pinned host buffer and a device buffer
    on a different device. The device operand must be on the queue's device;
    the pinned host operand may come from any device's context.
    """
    var driver = Driver.create(_get_plugin_spec())
    comptime spec = get_device_spec[Int64(0)]()
    var device0 = Device[spec]._create(driver[], Int64(0))
    var device1 = Device[spec]._create(driver[], Int64(1))
    var ctx0 = device0[].get_context()
    var ctx1 = device1[].get_context()
    var queue1 = ctx1[].create_queue()  # device 1

    comptime n = 32
    var pinned_in = ctx0[].alloc_host_pinned(UInt64(n))  # device 0, pinned
    var pinned_out = ctx0[].alloc_host_pinned(UInt64(n))  # device 0, pinned
    var dev = ctx1[].alloc_sync(UInt64(n))  # device 1, device-resident

    var in_ptr = ctx0[].memory_get_host_address[MutAnyOrigin](pinned_in)
    var out_ptr = ctx0[].memory_get_host_address[MutAnyOrigin](pinned_out)
    for i in range(n):
        in_ptr[unsafe_offset=i] = UInt8(50 + i)
        out_ptr[unsafe_offset=i] = UInt8(0)

    # pinned(dev 0) -> dev(dev 1) -> pinned(dev 0), all on device 1's queue
    # (the device operand's device).
    queue1[].copy(dst=dev, src=pinned_in)
    queue1[].copy(dst=pinned_out, src=dev)
    queue1[].synchronize()
    for i in range(n):
        assert_equal(out_ptr[unsafe_offset=i], UInt8(50 + i))

    ctx0[].free_host_pinned(pinned_in^)
    ctx0[].free_host_pinned(pinned_out^)
    ctx1[].free_sync(dev^)


# ===-----------------------------------------------------------------------===#
# Entry point
# ===-----------------------------------------------------------------------===#


def test_peer_access_mock() raises:
    var driver = Driver.create(_get_plugin_spec())
    var dev0 = driver[].get_device_dynamic(Int64(0))
    var dev1 = driver[].get_device_dynamic(Int64(1))
    comptime Ext = type_of(dev0[].extensions())
    comptime if conforms_to(Ext, SupportsPeerAccessV1):
        # The mock exports the peer group but has no peer-capable pair.
        assert_false(dev0[].can_access_peer(dev1[]))
        var ctx0 = dev0[].get_context()
        var ctx1 = dev1[].get_context()
        # `assert_raises` does not catch the typed `HALError`.
        var enable_raised = False
        try:
            ctx0[].enable_peer_access(ctx1[])
        except e:
            enable_raised = True
            assert_equal(e.status, STATUS_INCOMPATIBLE)
            assert_true("failed to enable peer access" in e.message)
        assert_true(enable_raised)
    else:
        print("extension set has no peer access; nothing to call")


def main() raises:
    test_status_code_values()

    var spec = _get_plugin_spec()
    if not spec:
        print("SKIP: MODULAR_DRIVER_PLUGINS not set")
        return

    test_driver_create()
    test_device_host_unified()
    test_invalid_device_id()
    test_context_create()
    test_invalid_device_error_message()
    test_memory_wrap_unwrap()
    test_queue_native_handle()
    test_queue_create_and_sync()
    test_queue_record_event()
    test_queue_wait_for_events()
    test_queue_wait_for_events_mixed_flags()
    test_stream_create_and_sync()
    test_stream_record_event()
    test_stream_wait_for_events()
    test_unified_copy_device_to_device_on_queue()
    test_unified_copy_sync_free_function()
    test_sync_set_memory_and_fill_free_functions()
    test_unified_copy_on_stream()
    test_unified_copy_dst_too_small_raises()
    test_unified_copy_front_fill()
    test_unified_copy_pinned()
    test_unified_copy_free_function_pinned_device()
    test_unified_copy_inter_device_on_queue()
    test_unified_copy_inter_device_free_function()
    test_unified_copy_inter_device_dst_not_on_queue_raises()
    test_unified_copy_pinned_cross_device()
    test_unified_copy_pinned_cross_device_on_queue()
    test_peer_access_mock()
