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

from std.gpu import global_idx, thread_idx
from std.gpu.memory import external_memory
from std.gpu.sync import barrier
from _device_context_hal import (
    DeviceBuffer,
    DeviceContext,
    DeviceEvent,
    DeviceFunction,
    DeviceStream,
    HostBuffer,
)
from std.memory import alloc, AddressSpace, Span, UnsafePointer
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


# ===-----------------------------------------------------------------------===#
# Kernel compile + launch
# ===-----------------------------------------------------------------------===#


def _vec_add_kernel(
    in0: UnsafePointer[Float32, ImmutAnyOrigin],
    in1: UnsafePointer[Float32, ImmutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    length: Int,
    supplement: Int,
):
    var tid = global_idx.x
    if tid >= length:
        return
    output[tid] = in0[tid] + in1[tid] + Float32(supplement)


def test_enqueue_function_with_args(ctx: DeviceContext) raises:
    comptime length = 1024

    var in0 = ctx.enqueue_create_buffer[DType.float32](length)
    var in1 = ctx.enqueue_create_buffer[DType.float32](length)
    var out = ctx.enqueue_create_buffer[DType.float32](length)
    var in0_host = ctx.enqueue_create_host_buffer[DType.float32](length)
    var in1_host = ctx.enqueue_create_host_buffer[DType.float32](length)
    var out_host = ctx.enqueue_create_host_buffer[DType.float32](length)
    ctx.synchronize()

    for i in range(length):
        in0_host[i] = Float32(i)
        in1_host[i] = Float32(2)

    ctx.enqueue_copy(in0, in0_host)
    ctx.enqueue_copy(in1, in1_host)

    comptime block_dim = 32
    var supplement = 5
    ctx.enqueue_function[_vec_add_kernel](
        in0,
        in1,
        out,
        length,
        supplement,
        grid_dim=(length // block_dim),
        block_dim=block_dim,
    )
    ctx.enqueue_copy(out_host, out)
    ctx.synchronize()

    for i in range(10):
        assert_equal(out_host[i], Float32(i) + Float32(2) + Float32(5))


def test_compile_function_reuse(ctx: DeviceContext) raises:
    # Pre-compile once, launch twice.
    comptime length = 64
    var dev = ctx.enqueue_create_buffer[DType.float32](length)
    var host_in = ctx.enqueue_create_host_buffer[DType.float32](length)
    var host_out = ctx.enqueue_create_host_buffer[DType.float32](length)
    ctx.synchronize()
    for i in range(length):
        host_in[i] = Float32(i)

    var compiled = ctx.compile_function[_vec_add_kernel]()
    ctx.enqueue_copy(dev, host_in)
    ctx.enqueue_function(
        compiled,
        dev,
        dev,
        dev,
        length,
        1,
        grid_dim=length // 32,
        block_dim=32,
    )
    ctx.enqueue_copy(host_out, dev)
    ctx.synchronize()

    for i in range(length):
        assert_equal(host_out[i], Float32(i) + Float32(i) + Float32(1))

    # Re-launch the same compiled function with different arguments.
    ctx.enqueue_copy(dev, host_in)
    ctx.enqueue_function(
        compiled,
        dev,
        dev,
        dev,
        length,
        7,
        grid_dim=length // 32,
        block_dim=32,
    )
    ctx.enqueue_copy(host_out, dev)
    ctx.synchronize()

    for i in range(length):
        assert_equal(host_out[i], Float32(i) + Float32(i) + Float32(7))


def test_enqueue_unified(ctx: DeviceContext) raises:
    comptime length = 1024

    var in0_host = ctx.enqueue_create_host_buffer[DType.float32](length)
    var in1_host = ctx.enqueue_create_host_buffer[DType.float32](length)
    var out_host = ctx.enqueue_create_host_buffer[DType.float32](length)
    ctx.synchronize()

    for i in range(length):
        in0_host[i] = Float32(i)
        in1_host[i] = Float32(2)

    var in0_device = ctx.enqueue_create_buffer[DType.float32](length)
    var in1_device = ctx.enqueue_create_buffer[DType.float32](length)
    var out_device = ctx.enqueue_create_buffer[DType.float32](length)

    ctx.enqueue_copy(in0_device, in0_host)
    ctx.enqueue_copy(in1_device, in1_host)

    var block_dim = 32
    var supplement = 5

    var output = Span(ptr=out_device.unsafe_ptr(), length=length)
    var in0 = Span(ptr=in0_device.unsafe_ptr(), length=length)
    var in1 = Span(ptr=in1_device.unsafe_ptr(), length=length)

    def vec_closure() {var supplement, var in0, var in1, var output}:
        var tid = global_idx.x
        if tid >= length:
            return
        output[tid] = in0[tid] + in1[tid] + Float32(supplement)

    ctx.enqueue_function(
        vec_closure,
        grid_dim=(length // block_dim),
        block_dim=block_dim,
    )

    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    var expected: List[Float32] = [
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
    ]
    for i in range(10):
        assert_equal(out_host[i], expected[i])


def test_external_shared_mem(ctx: DeviceContext) raises:
    print("== test_external_shared_mem")

    def dynamic_smem_kernel(data: UnsafePointer[Float32, MutAnyOrigin]):
        var dynamic_sram = external_memory[
            Float32, address_space=AddressSpace.SHARED, alignment=4
        ]()
        dynamic_sram[thread_idx.x] = Float32(thread_idx.x)
        barrier()
        data[thread_idx.x] = dynamic_sram[thread_idx.x]

    var res_host_ptr = ctx.enqueue_create_host_buffer[DType.float32](16)
    ctx.synchronize()
    for i in range(16):
        res_host_ptr[i] = Float32(0)
    var res_device = ctx.enqueue_create_buffer[DType.float32](16)

    ctx.enqueue_copy(res_device, res_host_ptr)

    comptime kernel = dynamic_smem_kernel

    # 16 KB allocation — valid on all platforms including Metal (32 KB limit).
    ctx.enqueue_function[kernel](
        res_device,
        grid_dim=1,
        block_dim=16,
        shared_mem_bytes=16 * 1024,
    )

    ctx.enqueue_copy(res_host_ptr, res_device)

    ctx.synchronize()

    var expected: List[Float32] = [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
    ]
    for i in range(16):
        print(res_host_ptr[i])
        assert_equal(res_host_ptr[i], expected[i])

    _ = res_device


def test_stream_enqueue_function(ctx: DeviceContext) raises:
    comptime length = 128
    var stream = ctx.create_stream()
    var dev_in = ctx.enqueue_create_buffer[DType.float32](length)
    var dev_out = ctx.enqueue_create_buffer[DType.float32](length)
    var host_in = ctx.enqueue_create_host_buffer[DType.float32](length)
    var host_out = ctx.enqueue_create_host_buffer[DType.float32](length)
    ctx.synchronize()
    for i in range(length):
        host_in[i] = Float32(i)
    ctx.enqueue_copy(dev_in, host_in)
    ctx.synchronize()

    stream.enqueue_function[_vec_add_kernel](
        dev_in,
        dev_in,
        dev_out,
        length,
        0,
        grid_dim=length // 32,
        block_dim=32,
    )
    stream.synchronize()
    ctx.enqueue_copy(host_out, dev_out)
    ctx.synchronize()

    for i in range(length):
        assert_equal(host_out[i], Float32(i) * Float32(2))


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
        test_enqueue_function_with_args(ctx)
        test_compile_function_reuse(ctx)
        test_enqueue_unified(ctx)
        test_external_shared_mem(ctx)
        test_stream_enqueue_function(ctx)
