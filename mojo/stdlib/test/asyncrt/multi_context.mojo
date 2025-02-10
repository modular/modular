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


fn test_multi_function(ctx1: DeviceContext, ctx2: DeviceContext) raises:
    print("-------")
    print("Running test_function(" + ctx1.name() + ", " + ctx2.name() + "):")

    alias length = 1024

    var in0_host1 = ctx1.enqueue_create_host_buffer[DType.float32](length)
    var in0_host2 = ctx2.enqueue_create_host_buffer[DType.float32](length)
    var in1_host1 = ctx1.enqueue_create_host_buffer[DType.float32](length)
    var in1_host2 = ctx2.enqueue_create_host_buffer[DType.float32](length)
    var out_host1 = ctx1.enqueue_create_host_buffer[DType.float32](length)
    var out_host2 = ctx2.enqueue_create_host_buffer[DType.float32](length)
    var in0_dev1 = ctx1.enqueue_create_buffer[DType.float32](length)
    var in0_dev2 = ctx2.enqueue_create_buffer[DType.float32](length)
    var in1_dev1 = ctx1.enqueue_create_buffer[DType.float32](length)
    var in1_dev2 = ctx2.enqueue_create_buffer[DType.float32](length)
    var out_dev1 = ctx1.enqueue_create_buffer[DType.float32](length)
    var out_dev2 = ctx2.enqueue_create_buffer[DType.float32](length)

    ctx1.enqueue_memset(in0_dev1, 1.0)
    ctx2.enqueue_memset(in0_dev2, 2.0)

    # Initialize the input and outputs with known values.
    for i in range(length):
        in0_host1[i] = i
        in0_host2[i] = i
        in1_host1[i] = 2
        in1_host2[i] = 2
        out_host1[i] = length + i
        out_host2[i] = length + i

    # Copy to device buffers.
    in0_host1.enqueue_copy_to(in0_dev1)
    in0_host2.enqueue_copy_to(in0_dev2)
    in1_host1.enqueue_copy_to(in1_dev1)
    in1_host2.enqueue_copy_to(in1_dev2)
    # Write known bad values to out_dev.
    out_host1.enqueue_copy_to(out_dev1)
    out_host2.enqueue_copy_to(out_dev2)

    var func1 = ctx1.compile_function[vec_func]()
    var func2 = ctx2.compile_function[vec_func]()

    var block_dim = 32

    ctx1.enqueue_function(
        func1,
        in0_dev1,
        in1_dev1,
        out_dev1,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )
    print("4")
    ctx2.enqueue_function(
        func2,
        in0_dev2,
        in1_dev2,
        out_dev2,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )

    out_dev1.enqueue_copy_to(out_host1)
    out_dev2.enqueue_copy_to(out_host2)

    # Wait for the copies to be completed.
    ctx1.synchronize()
    ctx2.synchronize()

    for i in range(length):
        if i < 10:
            print("at index", i, "the value is", out_host1[i])
            print("at index", i, "the value is", out_host2[i])
        expect_eq(
            out_host1[i], i + 2, "at index ", i, " the value is ", out_host1[i]
        )
        expect_eq(
            out_host2[i], i + 2, "at index ", i, " the value is ", out_host2[i]
        )

    print("Done.")


fn main() raises:
    var ctx1 = create_test_device_context()
    var ctx2 = create_test_device_context()
    test_multi_function(ctx1, ctx2)
