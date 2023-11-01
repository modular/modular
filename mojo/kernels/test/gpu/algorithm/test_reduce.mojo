# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo %s | FileCheck %s

from math import div_ceil
from pathlib import Path
from sys.info import triple_is_nvidia_cuda

from builtin.io import _printf
from gpu import *
from gpu.host import Context, Dim, Function, Stream
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
    _memset,
)
from tensor import Tensor

from utils.index import Index


@always_inline
fn _floorlog2[n: Int]() -> Int:
    return 0 if n <= 1 else 1 + _floorlog2[n >> 1]()


@always_inline
fn _static_log2[n: Int]() -> Int:
    return 0 if n <= 1 else _floorlog2[n - 1]() + 1


@always_inline
fn warp_sum_reduce(val: Float32) -> Float32:
    var res = val

    alias limit = _static_log2[WARP_SIZE]()

    for mask in range(limit - 1, -1, -1):
        res += shuffle_down[DType.float32](res, 1 << mask)

    return res


@always_inline
fn block_reduce[BLOCK_SIZE: Int](val: Float32) -> Float32:
    let shared = stack_allocation[
        BLOCK_SIZE // WARP_SIZE, DType.float32, AddressSpace.SHARED
    ]()

    let lane = lane_id()
    let warp = warp_id()

    let warp_sum = warp_sum_reduce(val)

    if lane == 0:
        shared.store(warp, warp_sum)

    barrier()

    # return warp_sum_reduce(
    #     shared.load(lane) if ThreadIdx.x() < (BlockDim.x() // WARP_SIZE) else 0
    # )

    if warp == 0:
        var accum = Float32()
        for i in range(BLOCK_SIZE // WARP_SIZE):
            accum += shared.load(i)
        return accum
    return 0


fn reduce[
    BLOCK_SIZE: Int
](res: Pointer[Float32], vec: DTypePointer[DType.float32], len: Int):
    @parameter
    if not triple_is_nvidia_cuda():
        return

    let tid = BlockIdx.x() * BlockDim.x() + ThreadIdx.x()

    let val = block_reduce[BLOCK_SIZE](vec.load(tid))

    if ThreadIdx.x() == 0:
        _ = Atomic._fetch_add(res, val)


# CHECK-LABEL: run_reduce
fn run_reduce() raises:
    print("== run_reduce")

    alias BLOCK_SIZE = 128
    alias n = 1024

    let stream = Stream()

    var vec_host = Tensor[DType.float32](n)

    for i in range(n):
        vec_host[i] = i

    let vec_device = _malloc[Float32](n)
    let res_device = _malloc[Float32](1)

    _copy_host_to_device(vec_device, vec_host.data(), n)
    _memset(res_device, 0, 1)

    let func = Function[
        fn (
            Pointer[Float32], DTypePointer[DType.float32], Int
        ) -> None, reduce[BLOCK_SIZE=BLOCK_SIZE]
    ](verbose=True, dump_ptx=False)

    func(
        (div_ceil(n, BLOCK_SIZE),),
        (BLOCK_SIZE,),
        res_device,
        vec_device,
        n,
        stream=stream,
    )

    var res = SIMD[DType.float32, 1](0)
    _copy_device_to_host(Pointer.address_of(res), res_device, 1)

    # CHECK: res =  523776.0
    print("res = ", res)

    _free(vec_device)

    _ = vec_host

    _ = func ^
    _ = stream ^


# CHECK-NOT: CUDA_ERROR
def main():
    try:
        with Context() as ctx:
            run_reduce()
    except e:
        print("CUDA_ERROR:", e)
