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
    res_min: UnsafePointer[Scalar[DType.float32]],
    res_max: UnsafePointer[Scalar[DType.float32]],
    vec: UnsafePointer[Float32],
    len: Int,
):
    var tid = BlockIdx.x() * BlockDim.x() + ThreadIdx.x()

    if tid >= len:
        return

    _ = Atomic._fetch_add(res_add, vec[tid])

    Atomic.min(res_min, vec[tid])
    Atomic.max(res_max, vec[tid])


# CHECK-LABEL: run_reduce
fn run_reduce(ctx: DeviceContext) raises:
    alias BLOCK_SIZE = 32
    alias n = 1024

    var vec_host = NDBuffer[DType.float32, 1, DimList(n)].stack_allocation()

    for i in range(n):
        vec_host[i] = i

    var vec_device = ctx.create_buffer[DType.float32](n)
    var res_add_device = ctx.create_buffer[DType.float32](1)

    ctx.enqueue_copy_to_device(vec_device, vec_host.data)
    ctx.memset(res_add_device, 0, 1)

    var func = ctx.compile_function[reduce](verbose=True)

    var res_min_device = ctx.create_buffer[DType.float32](1)
    ctx.memset(res_min_device, 0, 1)

    var res_max_device = ctx.create_buffer[DType.float32](1)
    ctx.memset(res_max_device, 0, 1)

    ctx.enqueue_function(
        func,
        res_add_device,
        res_min_device,
        res_max_device,
        vec_device,
        n,
        grid_dim=ceildiv(n, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )

    var res = Float32(0)
    ctx.enqueue_copy_from_device(UnsafePointer.address_of(res), res_add_device)

    var res_min = Float32(0)
    ctx.enqueue_copy_from_device(
        UnsafePointer.address_of(res_min), res_min_device
    )

    var res_max = Float32(0)
    ctx.enqueue_copy_from_device(
        UnsafePointer.address_of(res_max), res_max_device
    )

    assert_equal(res, n * (n - 1) // 2)

    assert_equal(res_min, 0)

    assert_equal(res_max, n - 1)

    _ = vec_device

    _ = func^


def main():
    with DeviceContext() as ctx:
        run_reduce(ctx)
