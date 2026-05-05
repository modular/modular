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

from std.math import ceildiv
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.testing import assert_equal


def vec_add(
    output: UnsafePointer[Float32, MutAnyOrigin],
    in0: UnsafePointer[Float32, ImmutAnyOrigin],
    in1: UnsafePointer[Float32, ImmutAnyOrigin],
    length: Int,
):
    var tid = global_idx.x
    if tid >= length:
        return
    output[tid] = in0[tid] + in1[tid]


def test_vec_add_kernel_node(ctx: DeviceContext) raises:
    print("Test capturing and replaying a vec_add kernel in a device graph.")
    comptime length = 1024
    comptime block_dim = 256

    var in0_dev = ctx.enqueue_create_buffer[DType.float32](length)
    var in1_dev = ctx.enqueue_create_buffer[DType.float32](length)
    var out_dev = ctx.enqueue_create_buffer[DType.float32](length)

    with in0_dev.map_to_host() as in0_host, in1_dev.map_to_host() as in1_host:
        for i in range(length):
            in0_host[i] = Float32(i)
            in1_host[i] = Float32(length - i)

    var func = ctx.compile_function[vec_add]()
    var builder = ctx.create_graph_builder()
    builder.add_function(
        func,
        out_dev,
        in0_dev,
        in1_dev,
        length,
        grid_dim=ceildiv(length, block_dim),
        block_dim=block_dim,
    )
    var graph = builder^.instantiate()
    graph.replay()

    # Check values and zero out buffer for next run
    with out_dev.map_to_host() as out_host:
        for i in range(length):
            assert_equal(out_host[i], Float32(length))
            out_host[i] = 0.0

    graph.replay()

    with out_dev.map_to_host() as out_host:
        for i in range(length):
            assert_equal(out_host[i], Float32(length))


def test_closure_node(ctx: DeviceContext) raises:
    print("Test using a register_passable closure as a device graph node.")
    comptime length = 1024
    comptime block_dim = 256
    var scale = Float32(2.0)

    var in0_dev = ctx.enqueue_create_buffer[DType.float32](length)
    var in1_dev = ctx.enqueue_create_buffer[DType.float32](length)
    var out_dev = ctx.enqueue_create_buffer[DType.float32](length)

    with in0_dev.map_to_host() as in0_host, in1_dev.map_to_host() as in1_host:
        for i in range(length):
            in0_host[i] = Float32(i)
            in1_host[i] = Float32(length - i)

    var out_ptr = out_dev.unsafe_ptr()
    var in0_ptr = in0_dev.unsafe_ptr()
    var in1_ptr = in1_dev.unsafe_ptr()

    # Closure captures device pointers and scale from enclosing scope.
    def scaled_vec_add() register_passable {
        var scale, var out_ptr, var in0_ptr, var in1_ptr
    }:
        var tid = global_idx.x
        if tid >= length:
            return
        out_ptr[tid] = (in0_ptr[tid] + in1_ptr[tid]) * scale

    var builder = ctx.create_graph_builder()
    builder.add_function(
        scaled_vec_add,
        grid_dim=ceildiv(length, block_dim),
        block_dim=block_dim,
    )
    var graph = builder^.instantiate()
    graph.replay()

    with out_dev.map_to_host() as out_host:
        for i in range(length):
            assert_equal(out_host[i], Float32(length) * scale)


def test_add_copy_to_device(ctx: DeviceContext) raises:
    print("Test capturing a host-to-device memcpy node.")
    comptime length = 1024

    var host_src = ctx.enqueue_create_host_buffer[DType.float32](length)
    for i in range(length):
        host_src[i] = Float32(i) * 3.0
    var dev_buf = ctx.enqueue_create_buffer[DType.float32](length)

    var builder = ctx.create_graph_builder()
    builder.add_copy(dev_buf, host_src)
    var graph = builder^.instantiate()
    graph.replay()
    ctx.synchronize()

    with dev_buf.map_to_host() as host_view:
        for i in range(length):
            assert_equal(host_view[i], Float32(i) * 3.0)


def test_add_copy_from_device(ctx: DeviceContext) raises:
    print("Test capturing a device-to-host memcpy node.")
    comptime length = 1024

    var dev_buf = ctx.enqueue_create_buffer[DType.float32](length)
    with dev_buf.map_to_host() as host_view:
        for i in range(length):
            host_view[i] = Float32(2 * i + 1)

    # Zero the host destination so we can detect that the graph wrote to it.
    var host_dst = ctx.enqueue_create_host_buffer[DType.float32](length)
    for i in range(length):
        host_dst[i] = 0.0

    var builder = ctx.create_graph_builder()
    builder.add_copy(host_dst, dev_buf)
    var graph = builder^.instantiate()
    graph.replay()
    ctx.synchronize()

    for i in range(length):
        assert_equal(host_dst[i], Float32(2 * i + 1))


def test_add_copy_device_to_device(ctx: DeviceContext) raises:
    print("Test capturing a device-to-device memcpy node.")
    comptime length = 1024

    var src_dev = ctx.enqueue_create_buffer[DType.float32](length)
    var dst_dev = ctx.enqueue_create_buffer[DType.float32](length)

    with src_dev.map_to_host() as src_host:
        for i in range(length):
            src_host[i] = Float32(i * i)

    var builder = ctx.create_graph_builder()
    builder.add_copy(dst_dev, src_dev)
    var graph = builder^.instantiate()
    graph.replay()
    ctx.synchronize()

    with dst_dev.map_to_host() as dst_host:
        for i in range(length):
            assert_equal(dst_host[i], Float32(i * i))


def test_add_memset(ctx: DeviceContext) raises:
    print("Test capturing memset nodes for 8/16/32/64-bit dtypes.")
    comptime length = 64

    var buf_u8 = ctx.enqueue_create_buffer[DType.uint8](length)
    var buf_u16 = ctx.enqueue_create_buffer[DType.uint16](length)
    var buf_u32 = ctx.enqueue_create_buffer[DType.uint32](length)
    var buf_u64 = ctx.enqueue_create_buffer[DType.uint64](length)

    var builder = ctx.create_graph_builder()
    builder.add_memset(buf_u8, UInt8(123))
    builder.add_memset(buf_u16, UInt16(0xBEEF))
    builder.add_memset(buf_u32, UInt32(0xDEADBEEF))
    # Symmetric high/low halves so the graph builder can express it as a
    # single node.
    builder.add_memset(buf_u64, UInt64(0x0101010101010101))
    var graph = builder^.instantiate()
    graph.replay()
    ctx.synchronize()

    with buf_u8.map_to_host() as host_u8:
        for i in range(length):
            assert_equal(host_u8[i], UInt8(123))

    with buf_u16.map_to_host() as host_u16:
        for i in range(length):
            assert_equal(host_u16[i], UInt16(0xBEEF))

    with buf_u32.map_to_host() as host_u32:
        for i in range(length):
            assert_equal(host_u32[i], UInt32(0xDEADBEEF))

    with buf_u64.map_to_host() as host_u64:
        for i in range(length):
            assert_equal(host_u64[i], UInt64(0x0101010101010101))


def main() raises:
    with DeviceContext() as ctx:
        test_vec_add_kernel_node(ctx)
        test_closure_node(ctx)
        test_add_copy_to_device(ctx)
        test_add_copy_from_device(ctx)
        test_add_copy_device_to_device(ctx)
        test_add_memset(ctx)
