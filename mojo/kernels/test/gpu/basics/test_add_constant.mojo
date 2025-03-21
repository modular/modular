# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from gpu import *
from gpu.host import DeviceContext, Dim
from testing import *


fn add_constant_fn(
    out: UnsafePointer[Float32],
    input: UnsafePointer[Float32],
    constant: Float32,
    len: Int,
):
    var tid = global_idx.x
    if tid >= len:
        return
    out[tid] = input[tid] + constant


def run_add_constant(ctx: DeviceContext):
    alias length = 1024

    var in_device = ctx.enqueue_create_buffer[DType.float32](length)
    var out_device = ctx.enqueue_create_buffer[DType.float32](length)

    with in_device.map_to_host() as in_host:
        for i in range(length):
            in_host[i] = i

    var block_dim = 32
    alias constant = Float32(33)

    ctx.enqueue_function[add_constant_fn](
        out_device,
        in_device,
        constant,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )

    with out_device.map_to_host() as out_host:
        for i in range(10):
            assert_equal(out_host[i], i + constant)


def main():
    with DeviceContext() as ctx:
        run_add_constant(ctx)
