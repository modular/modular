# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s | FileCheck %s

from gpu.host.device_context import DeviceContext, DeviceBuffer, DeviceFunction
from gpu import *
from gpu.host import Context, Dim, Function, Stream
from utils.variant import Variant


fn vec_func(
    in0: DTypePointer[DType.float32],
    in1: DTypePointer[DType.float32],
    out: DTypePointer[DType.float32],
    len: Int,
    addition: Int,
):
    var tid = ThreadIdx.x() + BlockDim.x() * BlockIdx.x()
    if tid >= len:
        return
    out[tid] = in0[tid] + in1[tid] + addition


fn test(ctx: DeviceContext) raises:
    alias length = 1024

    var in0_host = Pointer[Float32].alloc(length)
    var in1_host = Pointer[Float32].alloc(length)
    var out_host = Pointer[Float32].alloc(length)

    for i in range(length):
        in0_host[i] = i
        in1_host[i] = 2

    var in0_device = ctx.create_buffer[Float32](length)
    var in1_device = ctx.create_buffer[Float32](length)
    var out_device = ctx.create_buffer[Float32](length)

    ctx.enqueue_copy_to_device(in0_device, in0_host)
    ctx.enqueue_copy_to_device(in1_device, in1_host)

    var func = ctx.compile_function[vec_func]()

    var block_dim = 32
    var addition = 5

    ctx.enqueue_function(
        func,
        in0_device,
        in1_device,
        out_device,
        length,
        addition,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )

    ctx.enqueue_copy_from_device(out_host, out_device)

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

    in0_host.free()
    in1_host.free()
    out_host.free()


def main():
    var ctx = DeviceContext()
    test(ctx)
