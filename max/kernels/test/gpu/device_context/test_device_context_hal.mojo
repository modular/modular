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

from std.collections import List

from std.gpu.host._device_context_hal import (
    DeviceBuffer,
    DeviceContext,
    DeviceEvent,
    DeviceStream,
    HostBuffer,
)
from std.memory import alloc, Span, UnsafePointer
from std.testing import assert_equal


def test_move(ctx: DeviceContext) raises:
    var b = ctx
    var c = b^
    c.synchronize()


def test_id(ctx: DeviceContext) raises:
    assert_equal(ctx.id(), 0)


def test_synchronize(ctx: DeviceContext) raises:
    ctx.synchronize()


def test_default_stream(ctx: DeviceContext) raises:
    var default_stream = ctx.stream()
    default_stream.synchronize()


def test_create_stream(ctx: DeviceContext) raises:
    var extra_stream = ctx.create_stream()
    extra_stream.synchronize()


def test_unrecorded_event_is_noop(ctx: DeviceContext) raises:
    # Synchronize and enqueue_wait_for on an unrecorded event are no-ops.
    var pending = ctx.create_event()
    var stream = ctx.create_stream()
    pending.synchronize()
    stream.enqueue_wait_for(pending)


def test_record_event(ctx: DeviceContext) raises:
    # First record on the unrecorded event installs the HAL handle; the
    # cross-stream wait + synchronize then resolve normally.
    var event = ctx.create_event()
    var default_stream = ctx.stream()
    var extra_stream = ctx.create_stream()
    default_stream.record_event(event)
    extra_stream.enqueue_wait_for(event)
    event.synchronize()


def test_record_event_replaces(ctx: DeviceContext) raises:
    # Re-recording on the same event replaces the prior HAL handle
    # (CUDA/HIP last-record-wins semantics).
    var event = ctx.create_event()
    var default_stream = ctx.stream()
    var extra_stream = ctx.create_stream()
    default_stream.record_event(event)
    default_stream.record_event(event)
    extra_stream.enqueue_wait_for(event)
    event.synchronize()


def test_device_event_constructor(ctx: DeviceContext) raises:
    # Constructing a DeviceEvent with a context records an event
    # on that context's default stream.
    var immediate = DeviceEvent(ctx)
    immediate.synchronize()


def test_buffer_roundtrip(ctx: DeviceContext) raises:
    comptime length = 128
    var dev_a = ctx.enqueue_create_buffer[DType.float32](length)
    var dev_b = ctx.enqueue_create_buffer[DType.float32](length)
    assert_equal(len(dev_a), length)
    assert_equal(len(dev_b), length)

    # Stage host data, copy to device, copy device-to-device, copy back.
    var src_host = alloc[Float32](length)
    var dst_host = alloc[Float32](length)
    for i in range(length):
        src_host[i] = Float32(i) * Float32(0.5)
        dst_host[i] = Float32(0)

    ctx.enqueue_copy(dev_a, src_host)
    ctx.enqueue_copy(dev_b, dev_a)
    ctx.enqueue_copy(dst_host, dev_b)
    ctx.synchronize()

    for i in range(length):
        assert_equal(dst_host[i], Float32(i) * Float32(0.5))

    src_host.free()
    dst_host.free()


def test_create_buffer_sync(ctx: DeviceContext) raises:
    comptime length = 64
    var buf = ctx.create_buffer_sync[DType.float32](length)
    assert_equal(len(buf), length)


def test_buffer_empty(ctx: DeviceContext) raises:
    var buf = DeviceBuffer[DType.float32].empty(ctx)
    assert_equal(len(buf), 0)


def test_buffer_context(ctx: DeviceContext) raises:
    var buf = ctx.enqueue_create_buffer[DType.float32](32)
    var owner = buf.context()
    assert_equal(owner.id(), ctx.id())


def test_enqueue_copy_from_span(ctx: DeviceContext) raises:
    comptime length = 8

    # Test with List as source (implicitly converts to Span).
    var src_list: List[Float32] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    var dev_buf = ctx.enqueue_create_buffer[DType.float32](length)
    ctx.enqueue_copy(dev_buf, Span(src_list))

    var out_host = ctx.enqueue_create_host_buffer[DType.float32](length)
    ctx.enqueue_copy(out_host, dev_buf)
    ctx.synchronize()

    for i in range(length):
        assert_equal(out_host[i], Float32(i + 1))


def test_enqueue_copy_to_span(ctx: DeviceContext) raises:
    comptime length = 8

    var src_list: List[Float32] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    var dev_buf = ctx.enqueue_create_buffer[DType.float32](length)
    ctx.enqueue_copy(dev_buf, Span(src_list))

    var dst_list: List[Float32] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ctx.enqueue_copy(Span(dst_list), dev_buf)
    ctx.synchronize()

    for i in range(length):
        assert_equal(dst_list[i], Float32(i + 1))


def test_enqueue_copy_from_span_host_buffer(ctx: DeviceContext) raises:
    comptime length = 4

    var src_list: List[Float32] = [100.0, 200.0, 300.0, 400.0]
    var host_buf = ctx.enqueue_create_host_buffer[DType.float32](length)
    ctx.enqueue_copy(host_buf, Span(src_list))
    ctx.synchronize()

    for i in range(length):
        assert_equal(host_buf[i], src_list[i])


def test_enqueue_copy_to_span_host_buffer(ctx: DeviceContext) raises:
    comptime length = 4

    var host_buf = ctx.enqueue_create_host_buffer[DType.float32](length)
    ctx.synchronize()
    for i in range(length):
        host_buf[i] = Float32((i + 1) * 10)

    var dst_list: List[Float32] = [0.0, 0.0, 0.0, 0.0]
    ctx.enqueue_copy(Span(dst_list), host_buf)
    ctx.synchronize()

    for i in range(length):
        assert_equal(dst_list[i], Float32((i + 1) * 10))


def test_device_buffer_enqueue_copy_from_span(ctx: DeviceContext) raises:
    comptime length = 4

    var src_list: List[Float32] = [10.0, 20.0, 30.0, 40.0]
    var dev_buf = ctx.enqueue_create_buffer[DType.float32](length)
    dev_buf.enqueue_copy_from(Span(src_list))

    var dst_list: List[Float32] = [0.0, 0.0, 0.0, 0.0]
    ctx.enqueue_copy(Span(dst_list), dev_buf)
    ctx.synchronize()

    for i in range(length):
        assert_equal(dst_list[i], src_list[i])


def test_device_buffer_enqueue_copy_to_span(ctx: DeviceContext) raises:
    comptime length = 4

    var src_list: List[Float32] = [10.0, 20.0, 30.0, 40.0]
    var dev_buf = ctx.enqueue_create_buffer[DType.float32](length)
    ctx.enqueue_copy(dev_buf, Span(src_list))

    var dst_list: List[Float32] = [0.0, 0.0, 0.0, 0.0]
    dev_buf.enqueue_copy_to(Span(dst_list))
    ctx.synchronize()

    for i in range(length):
        assert_equal(dst_list[i], src_list[i])


def test_host_buffer_enqueue_copy_from_span(ctx: DeviceContext) raises:
    comptime length = 4

    var src_list: List[Float32] = [10.0, 20.0, 30.0, 40.0]
    var host_buf = ctx.enqueue_create_host_buffer[DType.float32](length)
    host_buf.enqueue_copy_from(Span(src_list))
    ctx.synchronize()

    for i in range(length):
        assert_equal(host_buf[i], src_list[i])


def test_host_buffer_enqueue_copy_to_span(ctx: DeviceContext) raises:
    comptime length = 4

    var host_buf = ctx.enqueue_create_host_buffer[DType.float32](length)
    ctx.synchronize()
    for i in range(length):
        host_buf[i] = Float32((i + 1) * 10)

    var dst_list: List[Float32] = [0.0, 0.0, 0.0, 0.0]
    host_buf.enqueue_copy_to(Span(dst_list))
    ctx.synchronize()

    for i in range(length):
        assert_equal(dst_list[i], Float32((i + 1) * 10))


def test_host_buffer_as_span(ctx: DeviceContext) raises:
    comptime length = 16
    var hb = ctx.enqueue_create_host_buffer[DType.float32](length)
    var span = hb.as_span()
    assert_equal(len(span), length)

    # Mutations through the span are reflected through buffer indexing.
    for i in range(length):
        span[i] = Float32(i) * Float32(0.125)
    for i in range(length):
        assert_equal(hb[i], Float32(i) * Float32(0.125))


def test_host_buffer_to_host_buffer_copy(ctx: DeviceContext) raises:
    comptime length = 32
    var a = ctx.enqueue_create_host_buffer[DType.float32](length)
    var b = ctx.enqueue_create_host_buffer[DType.float32](length)
    for i in range(length):
        a[i] = Float32(i) * Float32(1.5)
        b[i] = Float32(0)

    ctx.enqueue_copy(b, a)
    ctx.synchronize()

    for i in range(length):
        assert_equal(b[i], Float32(i) * Float32(1.5))


def test_host_buffer_pointer_copy(ctx: DeviceContext) raises:
    comptime length = 16
    var hb = ctx.enqueue_create_host_buffer[DType.float32](length)
    var paged = alloc[Float32](length)
    for i in range(length):
        paged[i] = Float32(i) * Float32(3.0)

    ctx.enqueue_copy(hb, paged)
    ctx.synchronize()

    for i in range(length):
        assert_equal(hb[i], Float32(i) * Float32(3.0))

    for i in range(length):
        paged[i] = Float32(0)
    ctx.enqueue_copy(paged, hb)
    ctx.synchronize()

    for i in range(length):
        assert_equal(paged[i], Float32(i) * Float32(3.0))

    paged.free()


def test_host_buffer_instance_copy_with_device(ctx: DeviceContext) raises:
    comptime length = 32
    var hb = ctx.enqueue_create_host_buffer[DType.float32](length)
    var dev = ctx.enqueue_create_buffer[DType.float32](length)

    for i in range(length):
        hb[i] = Float32(i) * Float32(0.75)

    hb.enqueue_copy_to(dev)
    ctx.synchronize()

    for i in range(length):
        hb[i] = Float32(0)
    hb.enqueue_copy_from(dev)
    ctx.synchronize()

    for i in range(length):
        assert_equal(hb[i], Float32(i) * Float32(0.75))


def test_device_buffer_instance_copy_with_host(ctx: DeviceContext) raises:
    comptime length = 32
    var dev = ctx.enqueue_create_buffer[DType.float32](length)
    var hb_in = ctx.enqueue_create_host_buffer[DType.float32](length)
    var hb_out = ctx.enqueue_create_host_buffer[DType.float32](length)

    for i in range(length):
        hb_in[i] = Float32(i) * Float32(2.0)
        hb_out[i] = Float32(0)

    dev.enqueue_copy_from(hb_in)
    dev.enqueue_copy_to(hb_out)
    ctx.synchronize()

    for i in range(length):
        assert_equal(hb_out[i], Float32(i) * Float32(2.0))


def test_device_buffer_map_to_host(ctx: DeviceContext) raises:
    comptime length = 64
    var dev = ctx.enqueue_create_buffer[DType.float32](length)

    # Write to the device via the mapped host buffer; on `__exit__` the
    # host edits get pushed back to the device.
    with dev.map_to_host() as host:
        for i in range(length):
            host[i] = Float32(i) * Float32(0.25)

    # Re-map and verify the device side received the writes.
    with dev.map_to_host() as host:
        for i in range(length):
            assert_equal(host[i], Float32(i) * Float32(0.25))


def test_host_buffer_alloc_and_index(ctx: DeviceContext) raises:
    comptime length = 16
    var hb = ctx.enqueue_create_host_buffer[DType.float32](length)
    assert_equal(len(hb), length)

    # Host-side write through __setitem__.
    for i in range(length):
        hb[i] = Float32(i) * Float32(0.25)
    # Host-side read through __getitem__.
    for i in range(length):
        assert_equal(hb[i], Float32(i) * Float32(0.25))


def test_host_buffer_roundtrip(ctx: DeviceContext) raises:
    comptime length = 64
    var src_host = ctx.enqueue_create_host_buffer[DType.float32](length)
    var dst_host = ctx.enqueue_create_host_buffer[DType.float32](length)
    var dev = ctx.enqueue_create_buffer[DType.float32](length)

    for i in range(length):
        src_host[i] = Float32(i) * Float32(0.5)
        dst_host[i] = Float32(0)

    ctx.enqueue_copy(dev, src_host)
    ctx.enqueue_copy(dst_host, dev)
    ctx.synchronize()

    for i in range(length):
        assert_equal(dst_host[i], Float32(i) * Float32(0.5))


def test_host_buffer_context(ctx: DeviceContext) raises:
    var hb = ctx.enqueue_create_host_buffer[DType.float32](8)
    var owner = hb.context()
    assert_equal(owner.id(), ctx.id())


def main() raises:
    with DeviceContext() as ctx:
        test_move(ctx)
        test_id(ctx)
        test_synchronize(ctx)
        test_default_stream(ctx)
        test_create_stream(ctx)
        test_unrecorded_event_is_noop(ctx)
        test_record_event(ctx)
        test_record_event_replaces(ctx)
        test_device_event_constructor(ctx)
        test_buffer_roundtrip(ctx)
        test_create_buffer_sync(ctx)
        test_buffer_empty(ctx)
        test_buffer_context(ctx)
        test_enqueue_copy_from_span(ctx)
        test_enqueue_copy_to_span(ctx)
        test_enqueue_copy_from_span_host_buffer(ctx)
        test_enqueue_copy_to_span_host_buffer(ctx)
        test_device_buffer_enqueue_copy_from_span(ctx)
        test_device_buffer_enqueue_copy_to_span(ctx)
        test_host_buffer_enqueue_copy_from_span(ctx)
        test_host_buffer_enqueue_copy_to_span(ctx)
        test_host_buffer_as_span(ctx)
        test_host_buffer_to_host_buffer_copy(ctx)
        test_host_buffer_pointer_copy(ctx)
        test_host_buffer_instance_copy_with_device(ctx)
        test_device_buffer_instance_copy_with_host(ctx)
        test_device_buffer_map_to_host(ctx)
        test_host_buffer_alloc_and_index(ctx)
        test_host_buffer_roundtrip(ctx)
        test_host_buffer_context(ctx)
