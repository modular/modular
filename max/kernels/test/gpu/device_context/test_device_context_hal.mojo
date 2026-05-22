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

from std.gpu.host._device_context_hal import (
    DeviceBuffer,
    DeviceContext,
    DeviceEvent,
    DeviceStream,
)
from std.memory import alloc, UnsafePointer
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
