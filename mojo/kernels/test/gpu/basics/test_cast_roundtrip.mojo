# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug-no-assert %s | FileCheck %s


from gpu import *
from gpu.host import DeviceContext, Dim
from memory import UnsafePointer

from utils.numerics import inf, nan, neg_inf


fn id(
    input: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    len: Int,
):
    var tid = global_idx.x
    if tid >= len:
        return
    output[tid] = Float32(BFloat16(input[tid]))


# CHECK-LABEL: run_vec_add
@no_inline
fn run_vec_add(ctx: DeviceContext) raises:
    print("== run_vec_add")

    alias length = 1024

    var in_host = UnsafePointer[Float32].alloc(length)

    for i in range(length):
        in_host[i] = Float32(i)

    in_host[4] = nan[DType.float32]()
    in_host[5] = inf[DType.float32]()
    in_host[6] = neg_inf[DType.float32]()
    in_host[7] = -0.0

    var in_device = ctx.enqueue_create_buffer[DType.float32](length)
    var out_device = ctx.enqueue_create_buffer[DType.float32](length)

    in_device.enqueue_copy_from(in_host)

    var block_dim = 32

    ctx.enqueue_function[id](
        in_device,
        out_device,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )

    with out_device.map_to_host() as out_host:
        # CHECK: at index 0 the value is 0.0
        # CHECK: at index 1 the value is 1.0
        # CHECK: at index 2 the value is 2.0
        # CHECK: at index 3 the value is 3.0
        # CHECK: at index 4 the value is nan
        # CHECK: at index 5 the value is inf
        # CHECK: at index 6 the value is -inf
        # CHECK: at index 7 the value is -0.0
        # CHECK: at index 8 the value is 8.0
        # CHECK: at index 9 the value is 9.0
        for i in range(10):
            print("at index", i, "the value is", out_host[i])

    _ = in_device

    in_host.free()


def main():
    with DeviceContext() as ctx:
        run_vec_add(ctx)
