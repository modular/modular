# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo-no-debug -D MODULAR_ASYNCRT_DEVICE_CONTEXT_V2=cpu %s
# RUN: %mojo-no-debug -D MODULAR_ASYNCRT_DEVICE_CONTEXT_V2=gpu %s

from asyncrt_test_utils import create_test_device_context, expect_eq
from gpu.host import DeviceBuffer, DeviceContext


fn _run_memset[
    type: DType
](ctx: DeviceContext, length: Int, val: Scalar[type]) raises:
    print("-")
    print("_run_memset(", length, ", ", val, ")")

    var in_host = ctx.malloc_host[Scalar[type]](length)
    var out_host = ctx.malloc_host[Scalar[type]](length)
    var on_dev = ctx.create_buffer_sync[type](length)

    # Initialize the input and outputs with known values.
    for i in range(length):
        in_host[i] = i
        out_host[i] = length + i

    # Copy to and from device buffers.
    ctx.copy(on_dev, in_host)
    ctx.memset_sync(on_dev, val)
    ctx.copy(out_host, on_dev)

    for i in range(length):
        if i < 10:
            print("at index", i, "the value is", out_host[i])
        expect_eq(
            out_host[i], val, "at index ", i, " the value is ", out_host[i]
        )

    ctx.free_host(out_host)
    ctx.free_host(in_host)


fn _run_memset_async[
    type: DType
](ctx: DeviceContext, length: Int, val: Scalar[type]) raises:
    print("-")
    print("_run_memset_async(", length, ", ", val, ")")

    var in_host = ctx.enqueue_create_host_buffer[type](length)
    var out_host = ctx.enqueue_create_host_buffer[type](length)
    var on_dev = ctx.enqueue_create_buffer[type](length)

    # Initialize the input and outputs with known values.
    for i in range(length):
        in_host[i] = i
        out_host[i] = length + i

    # Copy to and from device buffers.
    in_host.enqueue_copy_to(on_dev)
    ctx.memset(on_dev, val)  # Using old-style name here.
    on_dev.enqueue_copy_to(out_host)

    # Wait for the copies to be completed.
    ctx.synchronize()

    for i in range(length):
        if i < 10:
            print("at index", i, "the value is", out_host[i])
        expect_eq(
            out_host[i], val, "at index ", i, " the value is ", out_host[i]
        )


fn _run_memset_cascade[
    type: DType
](ctx: DeviceContext, length: Int, val: Scalar[type]) raises:
    print("-")
    print("_run_memset_cascade(", length, ", ", val, ")")

    var buf = ctx.enqueue_create_buffer[type](length).enqueue_fill(val)

    with buf.map_to_host() as buf:
        for i in range(length):
            if i < 10:
                print("buf[", i, "] = ", buf[i])
            expect_eq(buf[i], val, "at index ", i, " the value is ", buf[i])


fn main() raises:
    var ctx = create_test_device_context()

    print("-------")
    print("Running test_memset(" + ctx.name() + "):")

    alias one_mb = 1024 * 1024

    _run_memset[DType.uint8](ctx, 64, 15)
    _run_memset[DType.uint8](ctx, one_mb, 16)

    _run_memset_async[DType.uint8](ctx, 64, 12)
    _run_memset_async[DType.uint8](ctx, one_mb, 13)
    _run_memset_cascade[DType.uint8](ctx, one_mb, 14)

    _run_memset[DType.float16](ctx, 64, -2.125)
    _run_memset_async[DType.float16](ctx, 64, 1.75)
    _run_memset_cascade[DType.float16](ctx, 64, 2.5)

    _run_memset[DType.int32](ctx, 64, -2311503)
    _run_memset_async[DType.float32](ctx, 64, 2.3)
    _run_memset_cascade[DType.float32](ctx, 512, 25.125)

    _run_memset[DType.float64](ctx, 64, 0)
    _run_memset_async[DType.float64](ctx, 64, 0)
    _run_memset[DType.float64](ctx, 64, 2.71828182846)
    _run_memset_async[DType.float64](ctx, 64, 1.618033988749)
    _run_memset_cascade[DType.int64](ctx, one_mb, 1234567890)

    print("Done.")
