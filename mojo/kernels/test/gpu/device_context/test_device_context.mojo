# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s | FileCheck %s

from gpu.host.device_context import DeviceContext, DeviceBuffer, DeviceFunction
from gpu import *


# A Simple Kernel performing the sum of two arrays
fn vec_func(
    in0: DTypePointer[DType.float32],
    in1: DTypePointer[DType.float32],
    out: DTypePointer[DType.float32],
    len: Int,
    supplement: Int,
):
    var tid = ThreadIdx.x() + BlockDim.x() * BlockIdx.x()
    if tid >= len:
        return
    out[tid] = in0[tid] + in1[tid] + supplement


fn test(ctx: DeviceContext) raises:
    alias length = 1024

    # Host memory buffers for input and output data
    var in0_host = Pointer[Float32].alloc(length)
    var in1_host = Pointer[Float32].alloc(length)
    var out_host = Pointer[Float32].alloc(length)

    # Initialize inputs
    for i in range(length):
        in0_host[i] = i
        in1_host[i] = 2

    # Device memory buffers for the kernel input and output
    var in0_device = ctx.create_buffer[Float32](length)
    var in1_device = ctx.create_buffer[Float32](length)
    var out_device = ctx.create_buffer[Float32](length)

    # Copy the input data from the Host to the Device memory
    ctx.enqueue_copy_to_device(in0_device, in0_host)
    ctx.enqueue_copy_to_device(in1_device, in1_host)

    # Compile the kernel for the device
    var func = ctx.compile_function[vec_func]()

    var block_dim = 32
    var supplement = 5

    # Execute the kernel on the device.
    #  - notice the simple function call like invocation
    ctx.enqueue_function(
        func,
        in0_device,
        in1_device,
        out_device,
        length,
        supplement,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )

    # Copy the results back from the device to the host
    ctx.enqueue_copy_from_device(out_host, out_device)

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
        print("at index", i, "the value is", out_host.load(i))

    # Release the Host buffers
    in0_host.free()
    in1_host.free()
    out_host.free()


def main():
    # Create an instance of the DeviceContext
    var ctx = DeviceContext()
    # Execute our test with the context
    test(ctx)
