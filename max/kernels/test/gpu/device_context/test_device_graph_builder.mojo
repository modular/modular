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
from std.gpu.host import DeviceContext, DeviceGraphBuilder
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


def fill_constant(
    output: UnsafePointer[Float32, MutAnyOrigin],
    val: Int,
    length: Int,
):
    var tid = global_idx.x
    if tid >= length:
        return
    output[tid] = Float32(val)


def add_in_place(
    buf: UnsafePointer[Float32, MutAnyOrigin],
    delta: Int,
    length: Int,
):
    var tid = global_idx.x
    if tid >= length:
        return
    buf[tid] += Float32(delta)


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

    def build(mut builder: DeviceGraphBuilder) raises {read}:
        _ = builder.add_function(
            func,
            out_dev,
            in0_dev,
            in1_dev,
            length,
            grid_dim=ceildiv(length, block_dim),
            block_dim=block_dim,
        )

    var graph = ctx.create_device_graph(build)
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


def test_parameterized_kernel_node(ctx: DeviceContext) raises:
    print(
        "Test add_function compiling a kernel passed as a parameter (no"
        " explicit compile_function step)."
    )
    comptime length = 1024
    comptime block_dim = 256

    var in0_dev = ctx.enqueue_create_buffer[DType.float32](length)
    var in1_dev = ctx.enqueue_create_buffer[DType.float32](length)
    var out_dev = ctx.enqueue_create_buffer[DType.float32](length)

    with in0_dev.map_to_host() as in0_host, in1_dev.map_to_host() as in1_host:
        for i in range(length):
            in0_host[i] = Float32(i)
            in1_host[i] = Float32(length - i)

    # Pass `vec_add` directly as a parameter; the builder compiles it.
    def build(mut builder: DeviceGraphBuilder) raises {read}:
        _ = builder.add_function[vec_add](
            out_dev,
            in0_dev,
            in1_dev,
            length,
            grid_dim=ceildiv(length, block_dim),
            block_dim=block_dim,
        )

    var graph = ctx.create_device_graph(build)
    graph.replay()

    with out_dev.map_to_host() as out_host:
        for i in range(length):
            assert_equal(out_host[i], Float32(length))


def test_closure_node(ctx: DeviceContext) raises:
    print("Test using a closure as a device graph node.")
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
    def scaled_vec_add() {var scale, var out_ptr, var in0_ptr, var in1_ptr}:
        var tid = global_idx.x
        if tid >= length:
            return
        out_ptr[tid] = (in0_ptr[tid] + in1_ptr[tid]) * scale

    def build(mut builder: DeviceGraphBuilder) raises {read}:
        _ = builder.add_function(
            scaled_vec_add,
            grid_dim=ceildiv(length, block_dim),
            block_dim=block_dim,
        )

    var graph = ctx.create_device_graph(build)
    graph.replay()

    _ = in0_dev^
    _ = in1_dev^

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

    def build(mut builder: DeviceGraphBuilder) raises {read}:
        _ = builder.add_copy(dev_buf, host_src)

    var graph = ctx.create_device_graph(build)
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

    def build(mut builder: DeviceGraphBuilder) raises {read}:
        _ = builder.add_copy(host_dst, dev_buf)

    var graph = ctx.create_device_graph(build)
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

    def build(mut builder: DeviceGraphBuilder) raises {read}:
        _ = builder.add_copy(dst_dev, src_dev)

    var graph = ctx.create_device_graph(build)
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

    def build(mut builder: DeviceGraphBuilder) raises {read}:
        # The four memsets target disjoint buffers, so each can be an
        # independent graph root.
        _ = builder.add_memset(buf_u8, UInt8(123))
        _ = builder.add_memset(buf_u16, UInt16(0xBEEF))
        _ = builder.add_memset(buf_u32, UInt32(0xDEADBEEF))
        # Symmetric high/low halves so the graph builder can express it as a
        # single node.
        _ = builder.add_memset(buf_u64, UInt64(0x0101010101010101))

    var graph = ctx.create_device_graph(build)
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


def test_add_function_with_dependencies(ctx: DeviceContext) raises:
    print(
        "Test add_function with explicit dependencies for two independent"
        " kernel chains."
    )
    comptime length = 1024
    comptime block_dim = 256
    comptime grid_dim = ceildiv(length, block_dim)

    var buf_a = ctx.enqueue_create_buffer[DType.float32](length)
    var buf_b = ctx.enqueue_create_buffer[DType.float32](length)

    var func = ctx.compile_function[fill_constant]()

    def build(mut builder: DeviceGraphBuilder) raises {read}:
        # Sequence A on `buf_a`: write 1, then 2, then 3, internally chained
        # via explicit dependencies. The first node is rooted with an empty
        # deps list; downstream nodes name their predecessor explicitly.
        var a0 = builder.add_function(
            func,
            buf_a,
            1,
            length,
            grid_dim=grid_dim,
            block_dim=block_dim,
        )
        var a1 = builder.add_function(
            func,
            buf_a,
            2,
            length,
            grid_dim=grid_dim,
            block_dim=block_dim,
            dependencies=[a0],
        )
        _ = builder.add_function(
            func,
            buf_a,
            3,
            length,
            grid_dim=grid_dim,
            block_dim=block_dim,
            dependencies=[a1],
        )

        # Sequence B on `buf_b`: write 4, then 5, then 6. Independent of
        # sequence A — also rooted explicitly.
        var b0 = builder.add_function(
            func,
            buf_b,
            4,
            length,
            grid_dim=grid_dim,
            block_dim=block_dim,
        )
        var b1 = builder.add_function(
            func,
            buf_b,
            5,
            length,
            grid_dim=grid_dim,
            block_dim=block_dim,
            dependencies=[b0],
        )
        _ = builder.add_function(
            func,
            buf_b,
            6,
            length,
            grid_dim=grid_dim,
            block_dim=block_dim,
            dependencies=[b1],
        )

    var graph = ctx.create_device_graph(build)
    graph.replay()
    ctx.synchronize()

    with buf_a.map_to_host() as host_a:
        for i in range(length):
            assert_equal(host_a[i], Float32(3))
    with buf_b.map_to_host() as host_b:
        for i in range(length):
            assert_equal(host_b[i], Float32(6))


def test_add_memset_with_dependencies(ctx: DeviceContext) raises:
    print(
        "Test add_memset with explicit dependencies for two independent"
        " memset chains."
    )
    comptime length = 64

    var buf_a = ctx.enqueue_create_buffer[DType.uint8](length)
    var buf_b = ctx.enqueue_create_buffer[DType.uint8](length)

    def build(mut builder: DeviceGraphBuilder) raises {read}:
        # Sequence A on `buf_a`: 0x11 -> 0x22 -> 0x33, internally chained.
        # First node is rooted with an empty deps list; sequences A and B are
        # independent because neither names a predecessor in the other chain.
        var a0 = builder.add_memset(buf_a, UInt8(0x11), dependencies=[])
        var a1 = builder.add_memset(buf_a, UInt8(0x22), dependencies=[a0])
        _ = builder.add_memset(buf_a, UInt8(0x33), dependencies=[a1])

        # Sequence B on `buf_b`: 0xAA -> 0xBB -> 0xCC. Independent of seq A.
        var b0 = builder.add_memset(buf_b, UInt8(0xAA), dependencies=[])
        var b1 = builder.add_memset(buf_b, UInt8(0xBB), dependencies=[b0])
        _ = builder.add_memset(buf_b, UInt8(0xCC), dependencies=[b1])

    var graph = ctx.create_device_graph(build)
    graph.replay()
    ctx.synchronize()

    with buf_a.map_to_host() as host_a:
        for i in range(length):
            assert_equal(host_a[i], UInt8(0x33))
    with buf_b.map_to_host() as host_b:
        for i in range(length):
            assert_equal(host_b[i], UInt8(0xCC))


def test_add_copy_with_dependencies(ctx: DeviceContext) raises:
    print(
        "Test add_copy with explicit dependencies for two independent copy"
        " chains."
    )
    comptime length = 64

    var buf_a = ctx.enqueue_create_buffer[DType.uint32](length)
    var buf_b = ctx.enqueue_create_buffer[DType.uint32](length)

    # Distinct host-side payloads for each step. Pinned via
    # enqueue_create_host_buffer so the graph builder can reference them
    # directly via add_copy.
    var host_a1 = ctx.enqueue_create_host_buffer[DType.uint32](length)
    var host_a2 = ctx.enqueue_create_host_buffer[DType.uint32](length)
    var host_b1 = ctx.enqueue_create_host_buffer[DType.uint32](length)
    var host_b2 = ctx.enqueue_create_host_buffer[DType.uint32](length)
    for i in range(length):
        host_a1[i] = UInt32(0x11111111)
        host_a2[i] = UInt32(0x22222222)
        host_b1[i] = UInt32(0xAAAAAAAA)
        host_b2[i] = UInt32(0xBBBBBBBB)

    def build(mut builder: DeviceGraphBuilder) raises {read}:
        # Sequence A: HtoD(host_a1) -> HtoD(host_a2). Final state of `buf_a`
        # is the second copy's payload (host_a2). First node rooted
        # explicitly.
        var a0 = builder.add_copy(buf_a, host_a1, dependencies=[])
        _ = builder.add_copy(buf_a, host_a2, dependencies=[a0])

        # Sequence B: HtoD(host_b1) -> HtoD(host_b2). Independent of seq A.
        var b0 = builder.add_copy(buf_b, host_b1, dependencies=[])
        _ = builder.add_copy(buf_b, host_b2, dependencies=[b0])

    var graph = ctx.create_device_graph(build)
    graph.replay()
    ctx.synchronize()

    with buf_a.map_to_host() as host_a:
        for i in range(length):
            assert_equal(host_a[i], UInt32(0x22222222))
    # with buf_b.map_to_host() as host_b:
    #    for i in range(length):
    #        assert_equal(host_b[i], UInt32(0xBBBBBBBB))

    # FIXME(MSTDL-2742): HostBuffer is origin incorrect.
    _ = UnsafePointer(to=host_a1).as_unsafe_any_origin()[]


def test_region(ctx: DeviceContext) raises:
    print(
        "Test region joins scope nodes into a single"
        " empty node usable as a downstream node's sole dependency."
    )
    comptime length = 64

    var buf_a = ctx.enqueue_create_buffer[DType.uint8](length)
    var buf_b = ctx.enqueue_create_buffer[DType.uint8](length)
    var buf_c = ctx.enqueue_create_buffer[DType.uint8](length)

    def build(mut builder: DeviceGraphBuilder) raises {read}:
        # Pre-existing root node added before the scope. It must NOT be a
        # predecessor of the join node returned by the scope.
        var pre_scope = builder.add_memset(buf_a, UInt8(0x01), dependencies=[])

        # Two producer nodes added inside the scope. The work is a named
        # capturing def, passed as a callback to the scope method. A node
        # from the enclosing scope (`pre_scope`) cannot be named inside the
        # callback because the callback is generic over the scope origin, so
        # it is injected as a scope-level predecessor via `dependencies=`
        # instead; every node the callback adds runs after it.
        def add_producers(mut b: DeviceGraphBuilder) raises {read}:
            _ = b.add_memset(buf_a, UInt8(0xAA), dependencies=[])
            _ = b.add_memset(buf_b, UInt8(0xBB), dependencies=[])

        var producers_join = builder.region(
            add_producers, dependencies=[pre_scope]
        )

        # Use the join node as the sole dependency of a memset on buf_c. The
        # final state of buf_c being 0xCC confirms that consumer ran; the
        # transitive order through the empty node enforces that the producers
        # have completed by then.
        _ = builder.add_memset(
            buf_c, UInt8(0xCC), dependencies=[producers_join]
        )

    var graph = ctx.create_device_graph(build)
    graph.replay()
    ctx.synchronize()

    with buf_a.map_to_host() as host_a:
        for i in range(length):
            assert_equal(host_a[i], UInt8(0xAA))
    with buf_b.map_to_host() as host_b:
        for i in range(length):
            assert_equal(host_b[i], UInt8(0xBB))
    with buf_c.map_to_host() as host_c:
        for i in range(length):
            assert_equal(host_c[i], UInt8(0xCC))


def test_region_empty(ctx: DeviceContext) raises:
    print(
        "Test region still returns a usable join node"
        " when the scope adds no nodes (empty node becomes a graph root)."
    )
    comptime length = 64
    var buf = ctx.enqueue_create_buffer[DType.uint8](length)

    def build(mut builder: DeviceGraphBuilder) raises {read}:
        def add_nothing(mut b: DeviceGraphBuilder) raises {read}:
            return

        var join = builder.region(add_nothing)

        # Hang a memset off the (rootless) empty node and verify the graph
        # still instantiates and replays successfully.
        _ = builder.add_memset(buf, UInt8(0xEE), dependencies=[join])

    var graph = ctx.create_device_graph(build)
    graph.replay()
    ctx.synchronize()

    with buf.map_to_host() as host:
        for i in range(length):
            assert_equal(host[i], UInt8(0xEE))


def test_region_with_dependencies(ctx: DeviceContext) raises:
    print(
        "Test region(dependencies=...) injects predecessors so"
        " a consumer scope runs after a producer scope (RAW on one buffer)."
    )
    comptime length = 1024
    comptime block_dim = 256
    comptime grid_dim = ceildiv(length, block_dim)

    var buf = ctx.enqueue_create_buffer[DType.float32](length)

    var fill = ctx.compile_function[fill_constant]()
    var incr = ctx.compile_function[add_in_place]()

    def build(mut builder: DeviceGraphBuilder) raises {read}:
        # Producer scope: fill `buf` with 5 (single kernel node, a graph root).
        def producer(mut b: DeviceGraphBuilder) raises {read}:
            _ = b.add_function(
                fill,
                buf,
                5,
                length,
                grid_dim=grid_dim,
                block_dim=block_dim,
                dependencies=[],
            )

        var join_a = builder.region(producer)

        # Consumer scope: increment `buf` by 10. Passing dependencies=[join_a]
        # injects join_a as an ambient predecessor of the incr node, so it
        # runs strictly after the producer. Final value must be 15, not 10.
        def consumer(mut b: DeviceGraphBuilder) raises {read}:
            _ = b.add_function(
                incr,
                buf,
                10,
                length,
                grid_dim=grid_dim,
                block_dim=block_dim,
                dependencies=[],
            )

        _ = builder.region(consumer, dependencies=[join_a])

    var graph = ctx.create_device_graph(build)
    graph.replay()
    ctx.synchronize()

    with buf.map_to_host() as host:
        for i in range(length):
            assert_equal(host[i], Float32(15))


def test_region_passthrough_dependencies(
    ctx: DeviceContext,
) raises:
    print(
        "Test region returns a join that still gates on"
        " `dependencies` when the scope adds no nodes (zero-node fallback)."
    )
    comptime length = 1024
    comptime block_dim = 256
    comptime grid_dim = ceildiv(length, block_dim)

    var buf = ctx.enqueue_create_buffer[DType.float32](length)

    var fill = ctx.compile_function[fill_constant]()
    var incr = ctx.compile_function[add_in_place]()

    def build(mut builder: DeviceGraphBuilder) raises {read}:
        # Producer scope: fill `buf` with 5.
        def producer(mut b: DeviceGraphBuilder) raises {read}:
            _ = b.add_function(
                fill,
                buf,
                5,
                length,
                grid_dim=grid_dim,
                block_dim=block_dim,
                dependencies=[],
            )

        var join_a = builder.region(producer)

        # Empty scope depending on join_a: adds no nodes, so its returned join
        # falls back to depending on join_a directly (it must chain the
        # barrier).
        def add_nothing(mut b: DeviceGraphBuilder) raises {read}:
            return

        var passthrough = builder.region(add_nothing, dependencies=[join_a])

        # Increment by 10, gated on the passthrough join. Final value must be
        # 15, proving the empty scope still ordered the incr after the
        # producer.
        _ = builder.add_function(
            incr,
            buf,
            10,
            length,
            grid_dim=grid_dim,
            block_dim=block_dim,
            dependencies=[passthrough],
        )

    var graph = ctx.create_device_graph(build)
    graph.replay()
    ctx.synchronize()

    with buf.map_to_host() as host:
        for i in range(length):
            assert_equal(host[i], Float32(15))


def main() raises:
    with DeviceContext() as ctx:
        test_vec_add_kernel_node(ctx)
        test_parameterized_kernel_node(ctx)
        test_closure_node(ctx)
        test_add_copy_to_device(ctx)
        test_add_copy_from_device(ctx)
        test_add_copy_device_to_device(ctx)
        test_add_memset(ctx)
        test_add_function_with_dependencies(ctx)
        test_add_memset_with_dependencies(ctx)
        test_add_copy_with_dependencies(ctx)
        test_region(ctx)
        test_region_empty(ctx)
        test_region_with_dependencies(ctx)
        test_region_passthrough_dependencies(ctx)
