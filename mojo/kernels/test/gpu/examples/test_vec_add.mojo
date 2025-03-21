# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: NVIDIA-GPU

# FIXME: KERN-1468
# UNSUPPORTED: H100-GPU

# RUN: %if !debugging-test %{ %mojo-no-debug %s | FileCheck %s %}

# ===----------------------------------------------------------------------=== #
# Debugging tests:
# Run them with `./bazelw test Kernels/test-gpu-debugging/test_vec_add.mojo.test`

# compile:
# RUN: %if debugging-test %{ %mojo-build-no-debug -debug-level=line-tables -O0 %s %}

# execute:
# GDB-COMMAND: b %breakpoint1:location
# GDB-COMMAND: c
# GDB-COMMAND: cuda thread 0
# GDB-COMMAND: info locals
# GDB-COMMAND: info args

# RUN: %if debugging-test %{ %mojo-debug-cuda %s | FileCheck %s --check-prefix=CHECK-GDB %}

# CHECK-GDB: hit Breakpoint
# CHECK-GDB: tid = 0
# CHECK-GDB: in0 = 0x
# CHECK-GDB: in1 = 0x
# CHECK-GDB: out = 0x
# CHECK-GDB: len = 1024
# ===----------------------------------------------------------------------=== #

from pathlib import Path

from gpu import *
from gpu.host import DeviceContext, Dim
from memory import UnsafePointer


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

    var in0_device = ctx.enqueue_create_buffer[DType.float32](length)
    var in1_device = ctx.enqueue_create_buffer[DType.float32](length)
    var out_device = ctx.enqueue_create_buffer[DType.float32](length)

    ctx.enqueue_copy(in0_device, in0_host)
    ctx.enqueue_copy(in1_device, in1_host)

    var block_dim = 32

    ctx.enqueue_function[vec_func](
        in0_device,
        in1_device,
        out_device,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )

    ctx.enqueue_copy(out_host, out_device)

    ctx.synchronize()

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


def main():
    with DeviceContext() as ctx:
        run_vec_add(ctx)
