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
    var tid = ThreadIdx.x() + BlockDim.x() * BlockIdx.x()
    if tid >= len:
        return
    out[tid] = input[tid] + constant


def run_add_constant(ctx: DeviceContext):
    alias length = 1024

    var in_host = UnsafePointer[Float32].alloc(length)
    var out_host = UnsafePointer[Float32].alloc(length)

    for i in range(length):
        in_host[i] = i

    var in_device = ctx.create_buffer[DType.float32](length)
    var out_device = ctx.create_buffer[DType.float32](length)

    ctx.enqueue_copy_to_device(in_device, in_host)

    var func = ctx.compile_function[add_constant_fn]()

    var block_dim = 32
    # FIXME: why did this have FloatLiteral here?
    alias constant = Float32(33)

    ctx.enqueue_function(
        func,
        out_device,
        in_device,
        constant,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )

    ctx.enqueue_copy_from_device(out_host, out_device)

    for i in range(10):
        assert_equal(out_host[i], i + constant)

    _ = in_device
    _ = out_device

    in_host.free()
    out_host.free()

    _ = func^


def main():
    with DeviceContext() as ctx:
        run_add_constant(ctx)
