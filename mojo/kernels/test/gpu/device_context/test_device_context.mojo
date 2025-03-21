# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from gpu import *
from gpu.host import DeviceBuffer, DeviceContext, DeviceFunction
from math import iota
from testing import assert_equal


# A Simple Kernel performing the sum of two arrays
fn vec_func(
    in0: UnsafePointer[Float32],
    in1: UnsafePointer[Float32],
    out: UnsafePointer[Float32],
    len: Int,
    supplement: Int,
):
    var tid = global_idx.x
    if tid >= len:
        return
    out[tid] = in0[tid] + in1[tid] + supplement


fn test(ctx: DeviceContext) raises:
    alias length = 1024

    # Host memory buffers for input and output data
    var in0_host = UnsafePointer[Float32].alloc(length)
    var in1_host = UnsafePointer[Float32].alloc(length)
    var out_host = UnsafePointer[Float32].alloc(length)

    # Initialize inputs
    for i in range(length):
        in0_host[i] = i
        in1_host[i] = 2

    # Device memory buffers for the kernel input and output
    var in0_device = ctx.enqueue_create_buffer[DType.float32](length)
    var in1_device = ctx.enqueue_create_buffer[DType.float32](length)
    var out_device = ctx.enqueue_create_buffer[DType.float32](length)

    # Copy the input data from the Host to the Device memory
    ctx.enqueue_copy(in0_device, in0_host)
    ctx.enqueue_copy(in1_device, in1_host)

    var block_dim = 32
    var supplement = 5

    # Execute the kernel on the device.
    #  - notice the simple function call like invocation
    ctx.enqueue_function[vec_func](
        in0_device,
        in1_device,
        out_device,
        length,
        supplement,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )

    # Copy the results back from the device to the host
    ctx.enqueue_copy(out_host, out_device)

    # Wait for the computation to be completed
    ctx.synchronize()

    # CHECK: at index 0 the value is 7.0
    # CHECK: at index 1 the value is 8.0
    # CHECK: at index 2 the value is 9.0
    # CHECK: at index 3 the value is 10.0
    # CHECK: at index 4 the value is 11.0
    # CHECK: at index 5 the value is 12.0
    # CHECK: at index 6 the value is 13.0
    # CHECK: at index 7 the value is 14.0
    # CHECK: at index 8 the value is 15.0
    # CHECK: at index 9 the value is 16.0
    for i in range(10):
        print("at index", i, "the value is", out_host[i])

    # Release the Host buffers
    in0_host.free()
    in1_host.free()
    out_host.free()


def test_move(ctx: DeviceContext):
    var b = ctx
    var c = b^
    c.synchronize()


def test_id(ctx: DeviceContext):
    # CPU always gets id 0 so test for that.
    assert_equal(ctx.id(), 0)


def test_print(ctx: DeviceContext):
    alias size = 15

    var host_buffer = ctx.enqueue_create_host_buffer[DType.uint16](size)
    ctx.synchronize()

    iota(host_buffer.unsafe_ptr(), size)

    # CHECK: DeviceBuffer([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    print(host_buffer)

    var dev_buffer = ctx.enqueue_create_buffer[DType.uint16](size)
    host_buffer.enqueue_copy_to(dev_buffer)
    ctx.synchronize()

    # CHECK: DeviceBuffer([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    print(dev_buffer)

    alias large_size = 1001
    var large_buffer = ctx.enqueue_create_host_buffer[DType.float32](large_size)
    ctx.synchronize()

    iota(large_buffer.unsafe_ptr(), large_size)

    # CHECK: DeviceBuffer([0.0, 1.0, 2.0, ..., 998.0, 999.0, 1000.0])
    print(large_buffer)


def main():
    # Create an instance of the DeviceContext
    with DeviceContext() as ctx:
        # Execute our test with the context
        test(ctx)
        test_move(ctx)
        test_id(ctx)
        test_print(ctx)
