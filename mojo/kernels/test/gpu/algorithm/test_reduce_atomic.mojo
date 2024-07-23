# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s | FileCheck %s

from math import ceildiv
from os.atomic import Atomic

from buffer import DimList, NDBuffer
from gpu import *
from gpu.host import DeviceContext
from gpu.host.memory import (
    _memset,
)
from memory import UnsafePointer


fn reduce(
    res: UnsafePointer[Float32],
    vec: UnsafePointer[Float32],
    len: Int,
):
    var tid = BlockIdx.x() * BlockDim.x() + ThreadIdx.x()

    if tid < len:
        _ = Atomic._fetch_add(res, Scalar.load(vec, tid))


# CHECK-LABEL: run_reduce
fn run_reduce(ctx: DeviceContext) raises:
    print("== run_reduce")

    alias BLOCK_SIZE = 32
    alias n = 1024

    var vec_host = NDBuffer[DType.float32, 1, DimList(n)].stack_allocation()

    for i in range(n):
        vec_host[i] = 1

    var vec_device = ctx.create_buffer[DType.float32](n)
    var res_device = ctx.create_buffer[DType.float32](1)

    ctx.enqueue_copy_to_device(vec_device, vec_host.data)
    ctx.memset(res_device, 0, 1)

    var func = ctx.compile_function[reduce](verbose=True)

    ctx.enqueue_function(
        func,
        res_device,
        vec_device,
        n,
        grid_dim=(ceildiv(n, BLOCK_SIZE),),
        block_dim=(BLOCK_SIZE,),
    )

    var res = Float32(0)
    ctx.enqueue_copy_from_device(UnsafePointer.address_of(res), res_device)

    # CHECK: res =  1024.0
    print("res = ", res)

    _ = vec_device

    _ = func^


# CHECK-NOT: CUDA_ERROR
def main():
    try:
        with DeviceContext() as ctx:
            run_reduce(ctx)
    except e:
        print("CUDA_ERROR:", e)
