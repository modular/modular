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
from max.gpu.host import DeviceContext
from std.testing import (
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_true,
)

from max.gpu.host import (
    DeviceGraph,
    DeviceGraphBuilder,
    DeviceGraphCache,
    DeviceGraphInput,
)
from max.runtime.async_value import AnyAsyncValueRef


def vec_add(
    output: UnsafePointer[Float32, MutAnyOrigin],
    in0: UnsafePointer[Float32, ImmutAnyOrigin],
    in1: UnsafePointer[Float32, ImmutAnyOrigin],
    length_dev: Int32,
):
    var length = Int(length_dev)
    var tid = global_idx.x
    if tid >= length:
        return
    output[tid] = in0[tid] + in1[tid]


def fill_constant(
    output: UnsafePointer[Float32, MutAnyOrigin],
    val_dev: Int32,
    length_dev: Int32,
):
    var val = Int(val_dev)
    var length = Int(length_dev)
    var tid = global_idx.x
    if tid >= length:
        return
    output[tid] = Float32(val)


def add_in_place(
    buf: UnsafePointer[Float32, MutAnyOrigin],
    delta_dev: Int32,
    length_dev: Int32,
):
    var delta = Int(delta_dev)
    var length = Int(length_dev)
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

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        _ = builder.add_function(
            func,
            out_dev,
            in0_dev,
            in1_dev,
            Int32(length),
            grid_dim=ceildiv(length, block_dim),
            block_dim=block_dim,
        )

    var graph = DeviceGraph.create(ctx, build)
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
    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        _ = builder.add_function[vec_add](
            out_dev,
            in0_dev,
            in1_dev,
            Int32(length),
            grid_dim=ceildiv(length, block_dim),
            block_dim=block_dim,
        )

    var graph = DeviceGraph.create(ctx, build)
    graph.replay()

    with out_dev.map_to_host() as out_host:
        for i in range(length):
            assert_equal(out_host[i], Float32(length))


def test_capturing_parameterized_kernel_node(ctx: DeviceContext) raises:
    print(
        "Test add_function compiling a capturing kernel passed as a parameter"
        " with runtime arguments."
    )
    comptime length = 1024
    comptime block_dim = 256
    var scale = Float32(3.0)

    var in0_dev = ctx.enqueue_create_buffer[DType.float32](length)
    var in1_dev = ctx.enqueue_create_buffer[DType.float32](length)
    var out_dev = ctx.enqueue_create_buffer[DType.float32](length)

    with in0_dev.map_to_host() as in0_host, in1_dev.map_to_host() as in1_host:
        for i in range(length):
            in0_host[i] = Float32(i)
            in1_host[i] = Float32(length - i)

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        # Captures `scale` from the enclosing scope while also taking runtime
        # arguments, exercising the capturing parameter-based overload.
        @parameter
        @__copy_capture(scale)
        def scaled_vec_add(
            output: UnsafePointer[Float32, MutAnyOrigin],
            in0: UnsafePointer[Float32, ImmutAnyOrigin],
            in1: UnsafePointer[Float32, ImmutAnyOrigin],
            length_dev: Int32,
        ):
            var length = Int(length_dev)
            var tid = global_idx.x
            if tid >= length:
                return
            output[tid] = (in0[tid] + in1[tid]) * scale

        _ = builder.add_function[scaled_vec_add](
            out_dev,
            in0_dev,
            in1_dev,
            Int32(length),
            grid_dim=ceildiv(length, block_dim),
            block_dim=block_dim,
        )

    var graph = DeviceGraph.create(ctx, build)
    graph.replay()

    with out_dev.map_to_host() as out_host:
        for i in range(length):
            assert_equal(out_host[i], Float32(length) * scale)


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

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        _ = builder.add_function(
            scaled_vec_add,
            grid_dim=ceildiv(length, block_dim),
            block_dim=block_dim,
        )

    var graph = DeviceGraph.create(ctx, build)
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

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        _ = builder.add_copy(dev_buf, host_src)

    var graph = DeviceGraph.create(ctx, build)
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

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        _ = builder.add_copy(host_dst, dev_buf)

    var graph = DeviceGraph.create(ctx, build)
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

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        _ = builder.add_copy(dst_dev, src_dev)

    var graph = DeviceGraph.create(ctx, build)
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

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        # The four memsets target disjoint buffers, so each can be an
        # independent graph root.
        _ = builder.add_memset(buf_u8, UInt8(123))
        _ = builder.add_memset(buf_u16, UInt16(0xBEEF))
        _ = builder.add_memset(buf_u32, UInt32(0xDEADBEEF))
        # Symmetric high/low halves so the graph builder can express it as a
        # single node.
        _ = builder.add_memset(buf_u64, UInt64(0x0101010101010101))

    var graph = DeviceGraph.create(ctx, build)
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


def test_add_output(ctx: DeviceContext) raises:
    print("Test registering a graph output alongside a kernel node.")
    comptime length = 1024
    comptime block_dim = 256

    var in0_dev = ctx.enqueue_create_buffer[DType.float32](length)
    var in1_dev = ctx.enqueue_create_buffer[DType.float32](length)
    var out_dev = ctx.enqueue_create_buffer[DType.float32](length)

    with in0_dev.map_to_host() as in0_host, in1_dev.map_to_host() as in1_host:
        for i in range(length):
            in0_host[i] = Float32(i)
            in1_host[i] = Float32(length - i)

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        _ = builder.add_function[vec_add](
            out_dev,
            in0_dev,
            in1_dev,
            Int32(length),
            grid_dim=ceildiv(length, block_dim),
            block_dim=block_dim,
        )
        # Register the result buffer as a graph output; the graph must still
        # instantiate and replay correctly. Copy the buffer handle so `out_dev`
        # remains valid for the host readback below.
        assert_equal(builder.num_outputs(), 0)
        builder.add_output(AnyAsyncValueRef(storage_buf=out_dev.copy()))
        assert_equal(builder.num_outputs(), 1)

    var graph = DeviceGraph.create(ctx, build)
    graph.replay()
    ctx.synchronize()

    with out_dev.map_to_host() as out_host:
        for i in range(length):
            assert_equal(out_host[i], Float32(length))


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

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        # Sequence A on `buf_a`: write 1, then 2, then 3, internally chained
        # via explicit dependencies. The first node is rooted with an empty
        # deps list; downstream nodes name their predecessor explicitly.
        var a0 = builder.add_function(
            func,
            buf_a,
            Int32(1),
            Int32(length),
            grid_dim=grid_dim,
            block_dim=block_dim,
        )
        var a1 = builder.add_function(
            func,
            buf_a,
            Int32(2),
            Int32(length),
            grid_dim=grid_dim,
            block_dim=block_dim,
            dependencies=[a0],
        )
        _ = builder.add_function(
            func,
            buf_a,
            Int32(3),
            Int32(length),
            grid_dim=grid_dim,
            block_dim=block_dim,
            dependencies=[a1],
        )

        # Sequence B on `buf_b`: write 4, then 5, then 6. Independent of
        # sequence A — also rooted explicitly.
        var b0 = builder.add_function(
            func,
            buf_b,
            Int32(4),
            Int32(length),
            grid_dim=grid_dim,
            block_dim=block_dim,
        )
        var b1 = builder.add_function(
            func,
            buf_b,
            Int32(5),
            Int32(length),
            grid_dim=grid_dim,
            block_dim=block_dim,
            dependencies=[b0],
        )
        _ = builder.add_function(
            func,
            buf_b,
            Int32(6),
            Int32(length),
            grid_dim=grid_dim,
            block_dim=block_dim,
            dependencies=[b1],
        )

    var graph = DeviceGraph.create(ctx, build)
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

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
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

    var graph = DeviceGraph.create(ctx, build)
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

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        # Sequence A: HtoD(host_a1) -> HtoD(host_a2). Final state of `buf_a`
        # is the second copy's payload (host_a2). First node rooted
        # explicitly.
        var a0 = builder.add_copy(buf_a, host_a1, dependencies=[])
        _ = builder.add_copy(buf_a, host_a2, dependencies=[a0])

        # Sequence B: HtoD(host_b1) -> HtoD(host_b2). Independent of seq A.
        var b0 = builder.add_copy(buf_b, host_b1, dependencies=[])
        _ = builder.add_copy(buf_b, host_b2, dependencies=[b0])

    var graph = DeviceGraph.create(ctx, build)
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

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        # Pre-existing root node added before the scope. It must NOT be a
        # predecessor of the join node returned by the scope.
        var pre_scope = builder.add_memset(buf_a, UInt8(0x01), dependencies=[])

        # Two producer nodes added inside the scope. The work is a named
        # capturing def, passed as a callback to the scope method. A node
        # from the enclosing scope (`pre_scope`) cannot be named inside the
        # callback because the callback is generic over the scope origin, so
        # it is injected as a scope-level predecessor via `dependencies=`
        # instead; every node the callback adds runs after it.
        def add_producers(mut b: DeviceGraphBuilder) raises {imm}:
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

    var graph = DeviceGraph.create(ctx, build)
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

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        def add_nothing(mut b: DeviceGraphBuilder) raises {imm}:
            return

        var join = builder.region(add_nothing)

        # Hang a memset off the (rootless) empty node and verify the graph
        # still instantiates and replays successfully.
        _ = builder.add_memset(buf, UInt8(0xEE), dependencies=[join])

    var graph = DeviceGraph.create(ctx, build)
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

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        # Producer scope: fill `buf` with 5 (single kernel node, a graph root).
        def producer(mut b: DeviceGraphBuilder) raises {imm}:
            _ = b.add_function(
                fill,
                buf,
                Int32(5),
                Int32(length),
                grid_dim=grid_dim,
                block_dim=block_dim,
                dependencies=[],
            )

        var join_a = builder.region(producer)

        # Consumer scope: increment `buf` by 10. Passing dependencies=[join_a]
        # injects join_a as an ambient predecessor of the incr node, so it
        # runs strictly after the producer. Final value must be 15, not 10.
        def consumer(mut b: DeviceGraphBuilder) raises {imm}:
            _ = b.add_function(
                incr,
                buf,
                Int32(10),
                Int32(length),
                grid_dim=grid_dim,
                block_dim=block_dim,
                dependencies=[],
            )

        _ = builder.region(consumer, dependencies=[join_a])

    var graph = DeviceGraph.create(ctx, build)
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

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        # Producer scope: fill `buf` with 5.
        def producer(mut b: DeviceGraphBuilder) raises {imm}:
            _ = b.add_function(
                fill,
                buf,
                Int32(5),
                Int32(length),
                grid_dim=grid_dim,
                block_dim=block_dim,
                dependencies=[],
            )

        var join_a = builder.region(producer)

        # Empty scope depending on join_a: adds no nodes, so its returned join
        # falls back to depending on join_a directly (it must chain the
        # barrier).
        def add_nothing(mut b: DeviceGraphBuilder) raises {imm}:
            return

        var passthrough = builder.region(add_nothing, dependencies=[join_a])

        # Increment by 10, gated on the passthrough join. Final value must be
        # 15, proving the empty scope still ordered the incr after the
        # producer.
        _ = builder.add_function(
            incr,
            buf,
            Int32(10),
            Int32(length),
            grid_dim=grid_dim,
            block_dim=block_dim,
            dependencies=[passthrough],
        )

    var graph = DeviceGraph.create(ctx, build)
    graph.replay()
    ctx.synchronize()

    with buf.map_to_host() as host:
        for i in range(length):
            assert_equal(host[i], Float32(15))


def test_as_context_kernel_chain(ctx: DeviceContext) raises:
    print(
        "Test recording a two-kernel chain through DeviceGraphBuilder"
        ".recording_context() using the ordinary enqueue_function API."
    )
    comptime length = 1024
    comptime block_dim = 256
    comptime grid_dim = ceildiv(length, block_dim)

    var buf = ctx.enqueue_create_buffer[DType.float32](length)

    var fill = ctx.compile_function[fill_constant]()
    var incr = ctx.compile_function[add_in_place]()

    # Record fill(=5) then incr(+10) on the same buffer through the recording
    # context. The facade chains each enqueue after the previous one, so a
    # correct final value of 15 proves the launches recorded AND stayed ordered
    # (incr must observe fill's write, which on CUDA/HIP relies on the recorded
    # dependency edge rather than any implicit stream FIFO).
    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        with builder.recording_context() as rec:
            rec.enqueue_function(
                fill,
                buf,
                Int32(5),
                Int32(length),
                grid_dim=grid_dim,
                block_dim=block_dim,
            )
            rec.enqueue_function(
                incr,
                buf,
                Int32(10),
                Int32(length),
                grid_dim=grid_dim,
                block_dim=block_dim,
            )

    var graph = DeviceGraph.create(ctx, build)

    graph.replay()
    ctx.synchronize()
    with buf.map_to_host() as host:
        for i in range(length):
            assert_equal(host[i], Float32(15))
            # Zero out so the second replay must recompute from scratch.
            host[i] = 0.0

    graph.replay()
    ctx.synchronize()
    with buf.map_to_host() as host:
        for i in range(length):
            assert_equal(host[i], Float32(15))


def test_create_buffer(ctx: DeviceContext) raises:
    print("Test allocating graph-owned device and host buffers.")
    comptime length = 1024

    var host_dst = ctx.enqueue_create_host_buffer[DType.uint8](length)
    for i in range(length):
        host_dst[i] = 0

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        # A device allocation is graph-scoped, so the graph copies it out to
        # storage the test owns rather than handing back the buffer itself.
        var dev_buf = builder.create_buffer[DType.uint8](length, is_host=False)
        var memset = builder.add_memset(dev_buf, UInt8(0x5A))
        _ = builder.add_copy(host_dst, dev_buf, dependencies=[memset])

        # A host allocation comes from the pool's other memory manager: writing
        # through it here would fault if it had been served as device memory.
        var pool_host = builder.create_buffer[DType.uint8](length, is_host=True)
        var ptr = pool_host.unsafe_ptr()

        # Slightly questionable way to check that this is a host allocation.
        for i in range(length):
            ptr[i] = UInt8(i % 251)

        for i in range(length):
            assert_equal(ptr[i], UInt8(i % 251))

    var graph = DeviceGraph.create(ctx, build)
    graph.replay()
    ctx.synchronize()

    for i in range(length):
        assert_equal(host_dst[i], UInt8(0x5A))


@fieldwise_init
struct _TaggedInput(DeviceGraphInput, ImplicitlyCopyable, Movable):
    """A graph input whose cache key is just its tag.

    Deliberately writes a bare, undelimited integer: framing is
    `make_key`'s job, so this is the adversarial case for it.
    """

    var tag: Int

    def write_graph_key(self, mut writer: Some[Writer]):
        writer.write(self.tag)


def test_cache_key_separates_inputs() raises:
    print("Test cache keys keep adjacent input contributions apart.")

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        return

    # Undelimited, these two would both spell "...123".
    var a = DeviceGraphCache.make_key(build, _TaggedInput(1), _TaggedInput(23))
    var b = DeviceGraphCache.make_key(build, _TaggedInput(12), _TaggedInput(3))
    assert_not_equal(a, b)

    # Equal inputs still agree, or nothing would ever hit.
    assert_equal(
        a, DeviceGraphCache.make_key(build, _TaggedInput(1), _TaggedInput(23))
    )

    # Arity is part of the key too: one input must not look like two.
    assert_not_equal(a, DeviceGraphCache.make_key(build, _TaggedInput(1)))

    # A different work function keys a different graph even with equal inputs.
    def other_build(mut builder: DeviceGraphBuilder) raises {imm}:
        return

    assert_not_equal(
        a,
        DeviceGraphCache.make_key(
            other_build, _TaggedInput(1), _TaggedInput(23)
        ),
    )


def test_cache_reuses_graph(ctx: DeviceContext) raises:
    print("Test a cache hit returns the prior graph without rebuilding it.")
    comptime length = 64

    var buf = ctx.enqueue_create_buffer[DType.uint8](length)
    var cache = DeviceGraphCache()

    # Counted through a pointer so the closure can stay `imm`-capturing.
    var build_count = 0
    var count = Pointer(to=build_count)

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        count[] += 1
        _ = builder.add_memset(buf, UInt8(0x5A))

    var first = DeviceGraph.create(ctx, build, cache=Pointer(to=cache))
    assert_equal(build_count, 1)

    # Same closure and same (empty) input list, so the key matches and the build
    # body must not run a second time.
    var second = DeviceGraph.create(ctx, build, cache=Pointer(to=cache))
    assert_equal(build_count, 1)

    # The graph handed back on a hit is a fully usable graph, not a husk.
    second.replay()
    ctx.synchronize()

    with buf.map_to_host() as host:
        for i in range(length):
            assert_equal(host[i], UInt8(0x5A))


def test_cache_distinguishes_inputs(ctx: DeviceContext) raises:
    print("Test inputs writing different cache keys do not share a graph.")
    comptime length = 64

    var buf = ctx.enqueue_create_buffer[DType.uint8](length)
    var cache = DeviceGraphCache()

    var build_count = 0
    var count = Pointer(to=build_count)

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        count[] += 1
        _ = builder.add_memset(buf, UInt8(0x11))

    _ = DeviceGraph.create(ctx, build, _TaggedInput(1), cache=Pointer(to=cache))
    assert_equal(build_count, 1)

    # A different key must miss and rebuild.
    _ = DeviceGraph.create(ctx, build, _TaggedInput(2), cache=Pointer(to=cache))
    assert_equal(build_count, 2)

    # Returning to the first key must hit the entry stored by the first call.
    _ = DeviceGraph.create(ctx, build, _TaggedInput(1), cache=Pointer(to=cache))
    assert_equal(build_count, 2)


def test_cache_without_cache_always_builds(ctx: DeviceContext) raises:
    print("Test omitting the cache rebuilds the graph on every call.")
    comptime length = 64

    var buf = ctx.enqueue_create_buffer[DType.uint8](length)

    var build_count = 0
    var count = Pointer(to=build_count)

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        count[] += 1
        _ = builder.add_memset(buf, UInt8(0x22))

    _ = DeviceGraph.create(ctx, build)
    _ = DeviceGraph.create(ctx, build)
    assert_equal(build_count, 2)


def test_cache_lookup_and_add(ctx: DeviceContext) raises:
    print("Test DeviceGraphCache lookup/add semantics directly.")
    comptime length = 64

    var buf = ctx.enqueue_create_buffer[DType.uint8](length)
    var cache = DeviceGraphCache()

    assert_false(Bool(cache.lookup("absent")))

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        _ = builder.add_memset(buf, UInt8(0x33))

    _ = cache.cache("key", DeviceGraph.create(ctx, build))

    var found = cache.lookup("key")
    assert_true(Bool(found))

    # The looked-up graph is an independent handle on the same graph, so it
    # replays even though the cache still holds its own reference.
    found.take().replay()
    ctx.synchronize()

    with buf.map_to_host() as host:
        for i in range(length):
            assert_equal(host[i], UInt8(0x33))

    # A second add under the same key replaces the entry rather than failing.
    _ = cache.cache("key", DeviceGraph.create(ctx, build))
    assert_true(Bool(cache.lookup("key")))
    assert_false(Bool(cache.lookup("other")))


def main() raises:
    with DeviceContext() as ctx:
        test_vec_add_kernel_node(ctx)
        test_as_context_kernel_chain(ctx)
        test_parameterized_kernel_node(ctx)
        test_capturing_parameterized_kernel_node(ctx)
        test_closure_node(ctx)
        test_add_copy_to_device(ctx)
        test_add_copy_from_device(ctx)
        test_add_copy_device_to_device(ctx)
        test_add_memset(ctx)
        test_add_output(ctx)
        test_add_function_with_dependencies(ctx)
        test_add_memset_with_dependencies(ctx)
        test_add_copy_with_dependencies(ctx)
        test_region(ctx)
        test_region_empty(ctx)
        test_region_with_dependencies(ctx)
        test_region_passthrough_dependencies(ctx)
        test_create_buffer(ctx)
        test_cache_reuses_graph(ctx)
        test_cache_distinguishes_inputs(ctx)
        test_cache_without_cache_always_builds(ctx)
        test_cache_lookup_and_add(ctx)
        test_cache_key_separates_inputs()
