# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s | FileCheck %s

from pathlib import Path
from gpu import *
from gpu.host import DeviceContext, Dim
from memory import UnsafePointer


fn vec_func(
    in0: DTypePointer[DType.float32],
    in1: DTypePointer[DType.float32],
    out: DTypePointer[DType.float32],
    len: Int,
):
    var tid = ThreadIdx.x() + BlockDim.x() * BlockIdx.x()
    if tid >= len:
        return
    out[tid] = in0[tid] + in1[tid]


# CHECK-LABEL: run_vec_add
# COM: Force the capture to be captured instead of inlined away.
@no_inline
fn run_vec_add(ctx: DeviceContext) raises:
    print("== run_vec_add")

    alias length = 1024

    var in0_host = UnsafePointer[Float32].alloc(length)
    var in1_host = UnsafePointer[Float32].alloc(length)
    var out_host = UnsafePointer[Float32].alloc(length)

    for i in range(length):
        in0_host[i] = i
        in1_host[i] = 2

    var in0_device = ctx.create_buffer[DType.float32](length)
    var in1_device = ctx.create_buffer[DType.float32](length)
    var out_device = ctx.create_buffer[DType.float32](length)

    ctx.enqueue_copy_to_device(in0_device, in0_host.address)
    ctx.enqueue_copy_to_device(in1_device, in1_host.address)

    var func = ctx.compile_function[vec_func]()

    var block_dim = 32

    ctx.enqueue_function(
        func,
        in0_device,
        in1_device,
        out_device,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )

    ctx.enqueue_copy_from_device(out_host.address, out_device)

    # CHECK: at index 0 the value is 2.0
    # CHECK: at index 1 the value is 3.0
    # CHECK: at index 2 the value is 4.0
    # CHECK: at index 3 the value is 5.0
    # CHECK: at index 4 the value is 6.0
    # CHECK: at index 5 the value is 7.0
    # CHECK: at index 6 the value is 8.0
    # CHECK: at index 7 the value is 9.0
    # CHECK: at index 8 the value is 10.0
    # CHECK: at index 9 the value is 11.0
    for i in range(10):
        print("at index", i, "the value is", out_host[i])

    _ = in0_device
    _ = in1_device
    _ = out_device

    in0_host.free()
    in1_host.free()
    out_host.free()

    _ = func^


# CHECK-NOT: CUDA_ERROR
def main():
    try:
        with DeviceContext() as ctx:
            run_vec_add(ctx)
    except e:
        print("CUDA_ERROR:", e)
