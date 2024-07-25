# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s

from math import ceildiv
from os.atomic import Atomic

from buffer import DimList, NDBuffer
from gpu import *
from gpu.host import DeviceContext
from gpu.host.memory import (
    _memset,
)
from memory import UnsafePointer
from testing import assert_equal


fn reduce(
    res_add: UnsafePointer[Float32],
    vec: UnsafePointer[Float32],
    len: Int,
):
    var tid = BlockIdx.x() * BlockDim.x() + ThreadIdx.x()

    if tid < len:
        _ = Atomic._fetch_add(res_add, vec[tid])


# CHECK-LABEL: run_reduce
fn run_reduce(ctx: DeviceContext) raises:
    alias BLOCK_SIZE = 32
    alias n = 1024

    var vec_host = NDBuffer[DType.float32, 1, DimList(n)].stack_allocation()

    for i in range(n):
        vec_host[i] = 1

    var vec_device = ctx.create_buffer[DType.float32](n)
    var res_add_device = ctx.create_buffer[DType.float32](1)

    ctx.enqueue_copy_to_device(vec_device, vec_host.data)
    ctx.memset(res_add_device, 0, 1)

    var func = ctx.compile_function[reduce](verbose=True, dump_ptx=True)

    ctx.enqueue_function(
        func,
        res_add_device,
        vec_device,
        n,
        grid_dim=ceildiv(n, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )

    var res = Float32(0)
    ctx.enqueue_copy_from_device(UnsafePointer.address_of(res), res_add_device)

    assert_equal(res, 1024)

    _ = vec_device

    _ = func^


def main():
    with DeviceContext() as ctx:
        run_reduce(ctx)
