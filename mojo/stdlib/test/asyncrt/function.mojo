# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# COM: Note: CPU function compilation not supported
# COM: %mojo-no-debug -D MODULAR_ASYNCRT_DEVICE_CONTEXT_V2=cpu %s
# RUN: %mojo-no-debug -D MODULAR_ASYNCRT_DEVICE_CONTEXT_V2=gpu %s

from asyncrt_test_utils import create_test_device_context, expect_eq
from gpu import *
from gpu.host import DeviceContext


fn vec_func(
    in0: UnsafePointer[Float32],
    in1: UnsafePointer[Float32],
    out: UnsafePointer[Float32],
    len: Int,
):
    var tid = GlobalIdx.x
    if tid >= len:
        return
    out[tid] = in0[tid] + in1[tid]  # breakpoint1


fn test_function(ctx: DeviceContext) raises:
    print("-------")
    print("Running test_function(" + ctx.name() + "):")

    alias length = 1024

    var in0_host = ctx.malloc_host[Float32](length)
    var in1_host = ctx.malloc_host[Float32](length)
    var out_host = ctx.malloc_host[Float32](length)
    var in0_dev = ctx.enqueue_create_buffer[DType.float32](length)
    var in1_dev = ctx.enqueue_create_buffer[DType.float32](length)
    var out_dev = ctx.enqueue_create_buffer[DType.float32](length)

    # Initialize the input and outputs with known values.
    for i in range(length):
        in0_host[i] = i
        in1_host[i] = 2
        out_host[i] = length + i

    # Copy to device buffers.
    ctx.enqueue_copy_to_device(in0_dev, in0_host)
    ctx.enqueue_copy_to_device(in1_dev, in1_host)
    # Write known bad values to out_dev.
    ctx.enqueue_copy_to_device(out_dev, out_host)

    var func = ctx.compile_function[vec_func]()

    var block_dim = 32

    ctx.enqueue_function(
        func,
        in0_dev,
        in1_dev,
        out_dev,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )

    ctx.enqueue_copy_from_device(out_host, out_dev)

    # Wait for the copies to be completed.
    ctx.synchronize()

    for i in range(length):
        if i < 10:
            print("at index", i, "the value is", out_host[i])
        expect_eq(
            out_host[i],
            i + 2,
            "at index " + str(i) + " the value is " + str(out_host[i]),
        )

    ctx.free_host(out_host)
    ctx.free_host(in1_host)
    ctx.free_host(in0_host)

    print("Done.")


fn main() raises:
    var ctx = create_test_device_context()
    test_function(ctx)
