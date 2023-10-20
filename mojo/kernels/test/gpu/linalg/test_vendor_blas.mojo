# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: nvptx_backend
# REQUIRES: has_cuda_device
# RUN: %mojo -debug-level full %s | FileCheck %s

from math import div_ceil
from pathlib import Path
from sys.info import triple_is_nvidia_cuda
from sys.param_env import env_get_string

from builtin.io import _printf
from gpu import AddressSpace, BlockDim, BlockIdx, ThreadIdx
from gpu.host import Context, Dim, Function, Stream, synchronize
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
)
from memory import memset_zero
from memory.buffer import NDBuffer
from memory.unsafe import DTypePointer
from tensor import Tensor

from utils.index import Index
from utils.list import DimList

alias BLOCK_DIM = 8


fn matmul(
    a_ptr: DTypePointer[DType.float32],
    b_ptr: DTypePointer[DType.float32],
    c_ptr: DTypePointer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
):
    @parameter
    if not triple_is_nvidia_cuda():
        return

    let x = BlockIdx.x() * BlockDim.x() + ThreadIdx.x()
    let y = BlockIdx.y() * BlockDim.y() + ThreadIdx.y()

    if x >= m or y >= n:
        return

    let a = NDBuffer[2, DimList.create_unknown[2](), DType.float32](
        a_ptr, Index(m, k)
    )
    let b = NDBuffer[2, DimList.create_unknown[2](), DType.float32](
        b_ptr, Index(k, n)
    )
    let c = NDBuffer[2, DimList.create_unknown[2](), DType.float32](
        c_ptr, Index(m, n)
    )

    var accum = Float32(0)
    for i in range(k):
        accum = a[x, i] * b[i, y] + accum
    c[Index(x, y)] = accum


# CHECK-LABEL: run_matmul
fn run_matmul() raises:
    print("== run_matmul")

    alias m = 64
    alias n = 64
    alias k = 64

    let stream = Stream[False]()

    var a_host = Tensor[DType.float32](m, k)
    var b_host = Tensor[DType.float32](k, n)
    var c_host = Tensor[DType.float32](m, n)

    for i in range(m):
        for j in range(k):
            a_host[Index(i, j)] = 0.1

    for i in range(k):
        for j in range(n):
            b_host[Index(i, j)] = 1

    for i in range(m):
        for j in range(n):
            c_host[Index(i, j)] = 0

    let a_device = _malloc[Float32](m * k)
    let b_device = _malloc[Float32](k * n)
    let c_device = _malloc[Float32](m * n)

    _copy_host_to_device(a_device, a_host.data(), m * k)
    _copy_host_to_device(b_device, b_host.data(), k * n)

    let func = Function[
        fn (
            DTypePointer[DType.float32],
            DTypePointer[DType.float32],
            DTypePointer[DType.float32],
            Int,
            Int,
            Int,
        ) -> None, matmul
    ](debug=True)

    func(
        (div_ceil(m, BLOCK_DIM), div_ceil(n, BLOCK_DIM)),
        (BLOCK_DIM, BLOCK_DIM),
        a_device,
        b_device,
        c_device,
        m,
        n,
        k,
        stream=stream,
    )
    synchronize()

    _copy_device_to_host(c_host.data(), c_device, m * n)

    for i in range(BLOCK_DIM):
        for j in range(BLOCK_DIM):
            print(
                "at index = [",
                i,
                ",",
                j,
                "] the value is",
                c_host[i, j],
            )

    _free(a_device)
    _free(b_device)
    _free(c_device)

    _ = a_host
    _ = b_host
    _ = c_host

    _ = func ^
    _ = stream ^


# CHECK-NOT: CUDA_ERROR
def main():
    try:
        with Context() as ctx:
            run_matmul()
    except e:
        print("CUDA_ERROR:", e)
