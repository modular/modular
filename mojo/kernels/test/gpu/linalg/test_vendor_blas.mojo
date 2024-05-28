# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: DISABLED
# TODO(#31429): Restore `--debug-level full` here
# RUN: %mojo-no-debug %s | FileCheck %s

from math import ceildiv

from buffer import NDBuffer, DimList
from gpu import BlockDim, BlockIdx, ThreadIdx
from gpu.host import Context, Function, Stream, synchronize
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
)
from gpu.cublas.cublas import *
from memory.unsafe import DTypePointer
from testing import *
from utils.index import Index


# CHECK-LABEL: run_matmul
fn run_matmul() raises:
    print("== run_matmul")

    alias m = 64
    alias n = 64
    alias k = 64

    var stream = Stream()

    var a_host = NDBuffer[DType.float32, 2, DimList(m, k)].stack_allocation()
    var b_host = NDBuffer[DType.float32, 2, DimList(k, n)].stack_allocation()
    var c_host = NDBuffer[DType.float32, 2, DimList(m, n)].stack_allocation()

    for i in range(m):
        for j in range(k):
            a_host[Index(i, j)] = 0.1

    for i in range(k):
        for j in range(n):
            b_host[Index(i, j)] = 1

    for i in range(m):
        for j in range(n):
            c_host[Index(i, j)] = 0

    var a_device = _malloc[Float32](m * k)
    var b_device = _malloc[Float32](k * n)
    var c_device = _malloc[Float32](m * n)

    _copy_host_to_device(a_device, a_host.data, m * k)
    _copy_host_to_device(b_device, b_host.data, k * n)

    synchronize()

    var handle = Pointer[cublasContext]()
    var alpha = Scalar[DType.float32](1.0)
    var beta = Scalar[DType.float32](0.0)
    var res0 = cublasCreate(Pointer.address_of(handle))
    print(res0)
    var res1 = cublasGemmEx(
        handle,
        cublasOperation_t.CUBLAS_OP_N,
        cublasOperation_t.CUBLAS_OP_N,
        m,
        n,
        k,
        Pointer.address_of(alpha).bitcast[NoneType](),
        a_device.bitcast[NoneType](),
        DataType.R_32F,
        m,
        b_device.bitcast[NoneType](),
        DataType.R_32F,
        k,
        Pointer.address_of(beta).bitcast[NoneType](),
        c_device.bitcast[NoneType](),
        DataType.R_32F,
        m,
        ComputeType.COMPUTE_32F,
        cublasGemmAlgo_t.CUBLAS_GEMM_DEFAULT,
    )
    print(res1)
    print(cublasDestroy(handle))

    _copy_device_to_host(c_host.data, c_device, m * n)

    for i in range(m):
        for j in range(n):
            assert_equal(c_host[i, j], 0.1 * k)

    _free(a_device)
    _free(b_device)
    _free(c_device)

    _ = stream^


# CHECK-NOT: CUDA_ERROR
def main():
    try:
        with Context() as ctx:
            run_matmul()
    except e:
        print("CUDA_ERROR:", e)
