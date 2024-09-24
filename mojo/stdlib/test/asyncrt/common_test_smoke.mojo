# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from gpu.host import DeviceContextVariant
from smoke_test_utils import expect_eq


fn test_smoke(ctx: DeviceContextVariant) raises:
    print("-------")
    print("Running test_smoke(" + ctx.name() + "):")

    alias length = 64

    var in_host = ctx.malloc_host[Float32](length)
    var out_host = ctx.malloc_host[Float32](length)
    var in_dev = ctx.create_buffer[DType.float32](length)
    var out_dev = ctx.create_buffer[DType.float32](length)

    # Initialize the input and outputs with known values.
    for i in range(length):
        in_host[i] = i
        out_host[i] = length + i

    # Copy to and from device buffers.
    ctx.enqueue_copy_to_device(in_dev, in_host)
    ctx.enqueue_copy_device_to_device(out_dev, in_dev)
    ctx.enqueue_copy_from_device(out_host, out_dev)

    # Wait for the copies to be completed.
    ctx.synchronize()

    for i in range(length):
        if i < 10:
            print("at index", i, "the value is", out_host[i])
        expect_eq(
            out_host[i],
            i,
            "at index " + str(i) + " the value is " + str(out_host[i]),
        )

    ctx.free_host(out_host)
    ctx.free_host(in_host)

    print("Done.")
