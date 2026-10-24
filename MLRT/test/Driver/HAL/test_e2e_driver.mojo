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
from std.os import getenv
from max._driver import Driver, Context, copy
from max._driver.max_license.hal.device import (
    Device,
    DeviceSpec,
    get_device_spec,
)
from max._driver.max_license.hal.status import HALError
from max._driver.max_license.hal.plugin import MemoryHandle
from max.gpu import global_idx
from std.memory import alloc, ArcPointer, OpaquePointer, Pointer
from std.sys import size_of
from std.testing import assert_equal


def vec_add(
    a: Pointer[Float32, ImmutAnyOrigin],
    b: Pointer[Float32, ImmutAnyOrigin],
    result: Pointer[Float32, MutAnyOrigin],
    n: Int,
):
    var tid = global_idx.x
    if tid >= n:
        return
    result[unsafe_offset=tid] = a[unsafe_offset=tid] + b[unsafe_offset=tid]


def do_execute[
    device_spec: DeviceSpec,
](mut ctx: Context[device_spec]) raises:
    # Compile the kernel and load it into the plugin.
    ref (bundle, info) = ctx.compile[type_of(vec_add), vec_add]()
    print("Compiled and loaded bundle")
    print("  function_name:", info.function_name)

    # Load function from the bundle.
    var func = ctx.load_function(bundle, info.function_name)
    print("Loaded function handle")

    # Allocate device memory.
    comptime length = 1024
    comptime elem_size = size_of[Float32]()
    comptime buf_size = length * elem_size

    var in0_dev = ctx.alloc_sync(UInt64(buf_size))
    var in1_dev = ctx.alloc_sync(UInt64(buf_size))
    var out_dev = ctx.alloc_sync(UInt64(buf_size))

    # Create queue.
    var queue = ctx.create_queue()

    # Fill host buffers and copy to device.
    var in0_host = alloc[Float32]({length = length}).unsafe_leak()
    var in1_host = alloc[Float32]({length = length}).unsafe_leak()
    var out_host = alloc[Float32]({length = length}).unsafe_leak()
    for i in range(length):
        in0_host[unsafe_offset=i] = Float32(i)
        in1_host[unsafe_offset=i] = Float32((i % 2) * -i * 2)
        out_host[unsafe_offset=i] = Float32(0.0)

    queue[].copy_to_device(in0_dev.view(), in0_host.unsafe_bitcast[UInt8]())
    queue[].copy_to_device(
        in1_dev.view(),
        in1_host.unsafe_bitcast[UInt8](),
    )
    queue[].synchronize()

    # Get GPU addresses for each buffer.
    var in0_ptr = ctx.memory_get_address(in0_dev)
    var in1_ptr = ctx.memory_get_address(in1_dev)
    var out_ptr = ctx.memory_get_address(out_dev)
    var length_val = Int(length)

    # Pack kernel arguments — order must match the kernel signature: a, b, result, n.
    var args = alloc[OpaquePointer[MutUntrackedOrigin]](
        {length = 4}
    ).unsafe_leak()
    var arg_sizes = alloc[UInt64]({length = 4}).unsafe_leak()

    args[unsafe_offset=0] = rebind[OpaquePointer[MutUntrackedOrigin]](
        Pointer(to=in0_ptr)
    )
    args[unsafe_offset=1] = rebind[OpaquePointer[MutUntrackedOrigin]](
        Pointer(to=in1_ptr)
    )
    args[unsafe_offset=2] = rebind[OpaquePointer[MutUntrackedOrigin]](
        Pointer(to=out_ptr)
    )
    args[unsafe_offset=3] = rebind[OpaquePointer[MutUntrackedOrigin]](
        Pointer(to=length_val)
    )
    arg_sizes[unsafe_offset=0] = UInt64(size_of[UInt64]())
    arg_sizes[unsafe_offset=1] = UInt64(size_of[UInt64]())
    arg_sizes[unsafe_offset=2] = UInt64(size_of[UInt64]())
    arg_sizes[unsafe_offset=3] = UInt64(size_of[Int]())

    # Execute.
    comptime block_dim = UInt32(256)
    comptime grid_dim = UInt32(length // Int(block_dim))

    queue[].execute(
        func,
        grid=(grid_dim, UInt32(1), UInt32(1)),
        block=(block_dim, UInt32(1), UInt32(1)),
        args=rebind[Pointer[OpaquePointer[MutUntrackedOrigin], MutAnyOrigin]](
            args
        ),
        arg_sizes=rebind[Pointer[UInt64, MutAnyOrigin]](arg_sizes),
        num_args=UInt32(4),
    )

    # Copy results back.
    queue[].copy_from_device(
        out_host.unsafe_bitcast[UInt8](),
        out_dev.view(),
    )
    queue[].synchronize()

    # Verify.
    for i in range(length):
        var fi = Float32(i)
        var expected = (fi % 2) * -fi * 2 + fi
        assert_equal(out_host[unsafe_offset=i], expected)

    # Also test out DtoD copies
    var out_dev_copy = ctx.alloc_sync(UInt64(buf_size))
    var out_host_copy = alloc[Float32]({length = length}).unsafe_leak()
    for i in range(length):
        out_host_copy[unsafe_offset=i] = Float32(0.0)

    queue[].copy_intra_device(out_dev_copy.view(), out_dev.view())
    queue[].synchronize()
    queue[].copy_from_device(
        out_host_copy.unsafe_bitcast[UInt8](),
        out_dev_copy.view(),
    )
    queue[].synchronize()

    for i in range(length):
        assert_equal(out_host_copy[unsafe_offset=i], out_host[unsafe_offset=i])

    # Cleanup.
    out_host_copy.unsafe_free()
    args.unsafe_free()
    arg_sizes.unsafe_free()
    in0_host.unsafe_free()
    in1_host.unsafe_free()
    out_host.unsafe_free()
    ctx.unload_function(func)
    ctx.free_sync(out_dev_copy^)
    ctx.free_sync(out_dev^)
    ctx.free_sync(in1_dev^)
    ctx.free_sync(in0_dev^)


def do_offset_copy_test[
    device_spec: DeviceSpec,
](mut ctx: Context[device_spec]) raises:
    """Copies into the upper half of a buffer through an offset `BufferView`
    and reads the whole buffer back."""
    comptime n = 256
    comptime half = n // 2

    var stream = ctx.create_stream()
    var device_buffer = ctx.alloc_sync(UInt64(n))

    var src = alloc[UInt8]({length = half}).unsafe_leak()
    for i in range(half):
        src[unsafe_offset=i] = UInt8(0xAB)

    # Fill the lower half, then write the upper half at byte offset `half`.
    stream[].set_memory(
        device_buffer.view(byte_offset=UInt64(0), byte_size=UInt64(half)),
        value=UInt8(0xCD),
    )
    stream[].copy_to_device(
        device_buffer.view(byte_offset=UInt64(half), byte_size=UInt64(half)),
        src.unsafe_bitcast[UInt8](),
    )

    var host = alloc[UInt8]({length = n}).unsafe_leak()
    stream[].copy_from_device(
        host.unsafe_bitcast[UInt8](), device_buffer.view()
    )
    stream[].synchronize()
    for i in range(n):
        assert_equal(
            host[unsafe_offset=i], UInt8(0xCD) if i < half else UInt8(0xAB)
        )

    src.unsafe_free()
    host.unsafe_free()
    ctx.free_sync(device_buffer^)


def do_fill_test[
    device_spec: DeviceSpec,
](mut ctx: Context[device_spec]) raises:
    """Fills a buffer with a repeated 32-bit pattern and reads it back."""
    comptime n = 256
    var pattern = 0xDEADBEEF

    var stream = ctx.create_stream()
    var device_buffer = ctx.alloc_sync(UInt64(n))

    stream[].fill(
        device_buffer.view(),
        value=UInt64(pattern),
        value_size=UInt64(4),
    )

    var host = alloc[UInt8]({length = n}).unsafe_leak()
    stream[].copy_from_device(
        host.unsafe_bitcast[UInt8](), device_buffer.view()
    )
    stream[].synchronize()

    # 0xDEADBEEF stored little-endian repeats bytes EF BE AD DE.
    for i in range(n):
        var expected_byte = UInt8((pattern >> (8 * (i % 4))) & 0xFF)
        assert_equal(host[unsafe_offset=i], expected_byte)

    host.unsafe_free()
    ctx.free_sync(device_buffer^)


def do_pinned_copy_test[
    device_spec: DeviceSpec,
](mut ctx: Context[device_spec]) raises:
    """Pinned host buffers ride host transports through the unified copy."""
    comptime n = 256
    var queue = ctx.create_queue()

    var pinned_in = ctx.alloc_host_pinned(UInt64(n))
    var pinned_out = ctx.alloc_host_pinned(UInt64(n))
    var dev = ctx.alloc_sync(UInt64(n))

    # Pinned memory is CPU-addressable in place through its host address; no
    # copy is needed to read or write it from the host.
    var in_ptr = ctx.memory_get_host_address[MutAnyOrigin](pinned_in)
    var out_ptr = ctx.memory_get_host_address[MutAnyOrigin](pinned_out)
    for i in range(n):
        in_ptr[unsafe_offset=i] = UInt8(i % 251)
        out_ptr[unsafe_offset=i] = UInt8(0)

    # pinned -> device -> pinned through the queue's unified copy.
    queue[].copy(dst=dev, src=pinned_in)
    queue[].copy(dst=pinned_out, src=dev)
    queue[].synchronize()
    for i in range(n):
        assert_equal(out_ptr[unsafe_offset=i], UInt8(i % 251))

    # The synchronous free `copy` handles pinned -> pinned as a host memcpy.
    for i in range(n):
        out_ptr[unsafe_offset=i] = UInt8(0)
    copy(dst=pinned_out, src=pinned_in)
    for i in range(n):
        assert_equal(out_ptr[unsafe_offset=i], UInt8(i % 251))

    print("Pinned copy test OK")

    ctx.free_host_pinned(pinned_in^)
    ctx.free_host_pinned(pinned_out^)
    ctx.free_sync(dev^)


def main() raises:
    var plugin_spec = getenv("MODULAR_DRIVER_PLUGINS")
    if not plugin_spec:
        raise Error("MODULAR_DRIVER_PLUGINS not set")

    var driver: ArcPointer[Driver]
    try:
        driver = Driver.create(plugin_spec)
    except e:
        raise Error(e)
    print("Driver:", driver[].get_name())
    print("Devices:", driver[].get_device_count())

    var device: ArcPointer[Device[get_device_spec[0]()]]
    try:
        device = driver[].get_device[0]()
    except e:
        raise Error(e)
    print("Device", device[].id, "OK")

    var ctx: ArcPointer[Context[get_device_spec[0]()]]
    try:
        ctx = device[].get_context()
    except e:
        raise Error(e)
    print("Context created")

    do_execute(ctx[])
    do_offset_copy_test(ctx[])
    do_fill_test(ctx[])
    do_pinned_copy_test(ctx[])
