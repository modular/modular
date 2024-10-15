# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo-no-debug -D MODULAR_ASYNCRT_DEVICE_CONTEXT_V1 %s
# RUN: %mojo-no-debug -D MODULAR_ASYNCRT_DEVICE_CONTEXT_V2=cpu %s
# RUN: %mojo-no-debug -D MODULAR_ASYNCRT_DEVICE_CONTEXT_V2=cuda %s

from test_utils import create_test_device_context, expect_eq
from gpu.host import DeviceContext, DeviceBuffer


fn _run_memset[
    type: DType
](ctx: DeviceContext, length: Int, val: Scalar[type]) raises:
    print("-")
    print("_run_memset(" + str(length) + ", " + str(val) + ")")

    var in_host = ctx.malloc_host[Scalar[type]](length)
    var out_host = ctx.malloc_host[Scalar[type]](length)
    var on_dev = ctx.create_buffer_sync[type](length)

    # Initialize the input and outputs with known values.
    for i in range(length):
        in_host[i] = i
        out_host[i] = length + i

    # Copy to and from device buffers.
    ctx.copy_to_device_sync(on_dev, in_host)
    ctx.memset_sync(on_dev, val)
    ctx.copy_from_device_sync(out_host, on_dev)

    for i in range(length):
        if i < 10:
            print("at index", i, "the value is", out_host[i])
        expect_eq(
            out_host[i],
            val,
            "at index " + str(i) + " the value is " + str(out_host[i]),
        )

    ctx.free_host(out_host)
    ctx.free_host(in_host)


fn _run_memset_async[
    type: DType
](ctx: DeviceContext, length: Int, val: Scalar[type]) raises:
    print("-")
    print("_run_memset_async(" + str(length) + ", " + str(val) + ")")

    var in_host = ctx.malloc_host[Scalar[type]](length)
    var out_host = ctx.malloc_host[Scalar[type]](length)
    var on_dev = ctx.enqueue_create_buffer[type](length)

    # Initialize the input and outputs with known values.
    for i in range(length):
        in_host[i] = i
        out_host[i] = length + i

    # Copy to and from device buffers.
    ctx.enqueue_copy_to_device(on_dev, in_host)
    ctx.memset(on_dev, val)  # Using old-style name here.
    ctx.enqueue_copy_from_device(out_host, on_dev)

    # Wait for the copies to be completed.
    ctx.synchronize()

    for i in range(length):
        if i < 10:
            print("at index", i, "the value is", out_host[i])
        expect_eq(
            out_host[i],
            val,
            "at index " + str(i) + " the value is " + str(out_host[i]),
        )

    ctx.free_host(out_host)
    ctx.free_host(in_host)


fn main() raises:
    var ctx = create_test_device_context()

    print("-------")
    print("Running test_memset(" + ctx.name() + "):")

    alias one_mb = 1024 * 1024

    _run_memset[DType.uint8](ctx, 64, 15)
    _run_memset[DType.uint8](ctx, one_mb, 15)

    _run_memset_async[DType.uint8](ctx, 64, 12)
    _run_memset_async[DType.uint8](ctx, one_mb, 12)

    _run_memset[DType.float16](ctx, 64, -2.125)
    _run_memset_async[DType.float16](ctx, 64, 1.75)

    _run_memset[DType.float16](ctx, 64, -2.125)
    _run_memset_async[DType.float16](ctx, 64, 1.75)

    _run_memset[DType.int32](ctx, 64, -2311503)
    _run_memset_async[DType.float32](ctx, 64, 2.3)

    print("Done.")
