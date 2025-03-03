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
    var tid = global_idx.x
    if tid >= len:
        return
    out[tid] = in0[tid] + in1[tid]  # breakpoint1


fn test_function(ctx: DeviceContext) raises:
    print("-------")
    print("Running test_function(" + ctx.name() + "):")

    alias length = 1024
    alias block_dim = 32
    alias T = DType.float32

    # Initialize the input and outputs with known values.
    var in0 = ctx.enqueue_create_buffer[T](length)
    var out = ctx.enqueue_create_buffer[T](length)
    with in0.map_to_host() as in0_host, out.map_to_host() as out_host:
        for i in range(length):
            in0_host[i] = i
            out_host[i] = length + i
    var in1 = ctx.enqueue_create_buffer[T](length).enqueue_fill(2.0)

    ctx.enqueue_function[vec_func](
        in0,
        in1,
        out,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )

    with out.map_to_host() as out_host:
        for i in range(length):
            if i < 10:
                print("at index", i, "the value is", out_host[i])
            expect_eq(
                out_host[i],
                i + 2,
                "at index ",
                i,
                " the value is ",
                out_host[i],
            )

    print("Done.")


fn main() raises:
    var ctx = create_test_device_context()
    test_function(ctx)
