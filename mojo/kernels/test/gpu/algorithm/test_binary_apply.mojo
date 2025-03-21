# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from pathlib import Path
from sys.info import is_nvidia_gpu

from gpu import *
from gpu.host import DeviceContext, Dim


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


# CHECK-LABEL: run_binary_add
# COM: Force the capture to be captured instead of inlined away.
@no_inline
fn run_binary_add(ctx: DeviceContext, capture: Float32) raises:
    print("== run_binary_add")

    alias length = 1024

    var in0 = ctx.enqueue_create_buffer[DType.float32](length)
    var in1 = ctx.enqueue_create_buffer[DType.float32](length)
    var out = ctx.enqueue_create_buffer[DType.float32](length)

    with in0.map_to_host() as in0_host, in1.map_to_host() as in1_host:
        for i in range(length):
            in0_host[i] = i
            in1_host[i] = 2

    @parameter
    fn add(lhs: Float32, rhs: Float32) -> Float32:
        return capture + lhs + rhs

    var block_dim = 32
    ctx.enqueue_function[vec_func[add]](
        in0,
        in1,
        out,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )
    # CHECK: number of captures: 1
    print(
        "number of captures:",
        ctx.compile_function[vec_func[add]]()._func_impl.num_captures,
    )
    ctx.synchronize()

    with out.map_to_host() as out_host:
        # CHECK: at index 0 the value is 4.5
        # CHECK: at index 1 the value is 5.5
        # CHECK: at index 2 the value is 6.5
        # CHECK: at index 3 the value is 7.5
        # CHECK: at index 4 the value is 8.5
        # CHECK: at index 5 the value is 9.5
        # CHECK: at index 6 the value is 10.5
        # CHECK: at index 7 the value is 11.5
        # CHECK: at index 8 the value is 12.5
        # CHECK: at index 9 the value is 13.5
        for i in range(10):
            print("at index", i, "the value is", out_host[i])


def main():
    with DeviceContext() as ctx:
        run_binary_add(ctx, 2.5)
