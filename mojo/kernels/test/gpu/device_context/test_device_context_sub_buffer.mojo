# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from gpu import *
from gpu.host import DeviceBuffer, DeviceContext, DeviceFunction


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

    # Allocate the input buffers as sub buffers of a bigger one
    var in_host = UnsafePointer[Float32].alloc(2 * length)
    var out_host = UnsafePointer[Float32].alloc(length)

    for i in range(length):
        in_host[i] = i
        in_host[i + length] = 2

    var in_device = ctx.enqueue_create_buffer[DType.float32](2 * length)
    var in0_device = in_device.create_sub_buffer[DType.float32](0, length)
    var in1_device = in_device.create_sub_buffer[DType.float32](length, length)

    var out_device = ctx.enqueue_create_buffer[DType.float32](length)

    ctx.enqueue_copy(in_device, in_host)

    var block_dim = 32
    var supplement = 5

    ctx.enqueue_function[vec_func](
        in0_device,
        in1_device,
        out_device,
        length,
        supplement,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )

    # Make sure our main input device tensor doesn't disappear
    _ = in_device

    ctx.enqueue_copy(out_host, out_device)

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

    in_host.free()
    out_host.free()


def main():
    with DeviceContext() as ctx:
        test(ctx)
