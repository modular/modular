# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s | FileCheck %s

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


fn reduce(
    res: Pointer[Float32],
    vec: DTypePointer[DType.float32],
    len: Int,
):
    var tid = BlockIdx.x() * BlockDim.x() + ThreadIdx.x()

    if tid < len:
        _ = Atomic._fetch_add(res, vec.load(tid))


# CHECK-LABEL: run_reduce
fn run_reduce() raises:
    print("== run_reduce")

    alias BLOCK_SIZE = 32
    alias n = 1024

    var stream = Stream()

    var vec_host = Tensor[DType.float32](n)

    for i in range(n):
        vec_host[i] = 1

    var vec_device = _malloc[Float32](n)
    var res_device = _malloc[Float32](1)

    _copy_host_to_device(vec_device, vec_host.data(), n)
    _memset(res_device, 0, 1)

    var func = Function[__type_of(reduce), reduce](verbose=True)

    func(
        res_device,
        vec_device,
        n,
        grid_dim=(div_ceil(n, BLOCK_SIZE),),
        block_dim=(BLOCK_SIZE,),
        stream=stream,
    )

    var res = Float32(0)
    _copy_device_to_host(Pointer.address_of(res), res_device, 1)

    # CHECK: res =  1024.0
    print("res = ", res)

    _free(vec_device)

    _ = vec_host

    _ = func^
    _ = stream^


# CHECK-NOT: CUDA_ERROR
def main():
    try:
        with Context() as ctx:
            run_reduce()
    except e:
        print("CUDA_ERROR:", e)
