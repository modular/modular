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


fn vec_func[
    op: fn (Float32, Float32) capturing [_] -> Float32
](
    in0: UnsafePointer[Float32],
    in1: UnsafePointer[Float32],
    out: UnsafePointer[Float32],
    len: Int,
):
    var tid = global_idx.x
    if tid >= len:
        return
    out[tid] = op(in0[tid], in1[tid])


@no_inline
fn run_captured_func(ctx: DeviceContext, captured: Float32) raises:
    print("-\nrun_captured_func(", captured, "):")

    alias length = 1024

    var in0_host = ctx.enqueue_create_host_buffer[DType.float32](length)
    var in1_host = ctx.enqueue_create_host_buffer[DType.float32](length)
    var out_host = ctx.enqueue_create_host_buffer[DType.float32](length)
    var in0_dev = ctx.enqueue_create_buffer[DType.float32](length)
    var in1_dev = ctx.enqueue_create_buffer[DType.float32](length)
    var out_dev = ctx.enqueue_create_buffer[DType.float32](length)

    # Initialize the input and outputs with known values.
    for i in range(length):
        in0_host[i] = i
        in1_host[i] = 2
        out_host[i] = length + i

    # Copy to device buffers.
    in0_host.enqueue_copy_to(in0_dev)
    in1_host.enqueue_copy_to(in1_dev)
    # Write known bad values to out_dev.
    out_host.enqueue_copy_to(out_dev)

    @parameter
    fn add_with_captured(left: Float32, right: Float32) -> Float32:
        return left + right + captured

    var func = ctx.compile_function[vec_func[add_with_captured]]()

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

    out_dev.enqueue_copy_to(out_host)

    # Wait for the copies to be completed.
    ctx.synchronize()

    for i in range(length):
        if i < 10:
            print("at index", i, "the value is", out_host[i])
        expect_eq(
            out_host[i],
            i + 2 + captured,
            String("at index ", i, " the value is ", out_host[i]),
        )


fn main() raises:
    var ctx = create_test_device_context()
    print("-------")
    print("Running test_capture(" + ctx.name() + "):")

    run_captured_func(ctx, 2.5)
    run_captured_func(ctx, -1.5)

    print("Done.")
