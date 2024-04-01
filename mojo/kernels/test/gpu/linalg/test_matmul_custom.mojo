# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo %s | FileCheck %s

from math import div_ceil, isclose

from buffer import NDBuffer
from buffer.list import DimList
from gpu import WARP_SIZE, BlockDim, BlockIdx, GridDim, ThreadIdx, barrier
from gpu.host import Context, Dim, Function, Stream, synchronize
from gpu.host.event import time_function
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
)
from gpu.sync import syncwarp
from Matmul import (
    matmul_kernel,
    matmul_kernel_naive,
)
from memory.unsafe import DTypePointer, bitcast

from utils.index import Index
from random import random_float64
from testing import *


fn run_matmul(M: Int, N: Int, K: Int) raises:
    print("== run_matmul kernel")

    var stream = Stream()
    var a_host = Pointer[BFloat16].alloc(M * K)
    var b_host = Pointer[BFloat16].alloc(K * N)
    var c_host = Pointer[BFloat16].alloc(M * N)
    var a_host_n = Pointer[Float32].alloc(M * K)
    var b_host_n = Pointer[Float32].alloc(K * N)
    var c_host_n = Pointer[Float32].alloc(M * N)

    var rand_min = -1.0
    var rand_max = 1.0

    for i in range(M * K):
        var val = random_float64(rand_min, rand_max)
        a_host[i] = val.cast[DType.bfloat16]()
        a_host_n[i] = a_host[i].cast[DType.float32]()

    for i in range(K * N):
        var val = random_float64(rand_min, rand_max)
        b_host[i] = val.cast[DType.bfloat16]()
        b_host_n[i] = b_host[i].cast[DType.float32]()

    for i in range(M * N):
        c_host[i] = 0
        c_host_n[i] = 0

    var a_device = _malloc[BFloat16](M * K)
    var b_device = _malloc[BFloat16](K * N)
    var c_device = _malloc[BFloat16](M * N)
    var a_device_n = _malloc[Float32](M * K)
    var b_device_n = _malloc[Float32](K * N)
    var c_device_n = _malloc[Float32](M * N)

    _copy_host_to_device(a_device, a_host, M * K)
    _copy_host_to_device(b_device, b_host, K * N)

    alias BLOCK_DIM = 16
    var func_gemm_bf16 = Function[
        fn (
            DTypePointer[DType.bfloat16],
            DTypePointer[DType.bfloat16],
            DTypePointer[DType.bfloat16],
            Int,
            Int,
            Int,
        ) capturing -> None, matmul_kernel_naive[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            BLOCK_DIM,
            DType.float32,
        ]
    ]()

    @always_inline
    @__copy_capture(func_gemm_bf16, c_device, a_device, b_device)
    @parameter
    fn run_func_bf16(stream: Stream) raises:
        func_gemm_bf16(
            c_device,
            a_device,
            b_device,
            M,
            N,
            K,
            grid_dim=(div_ceil(M, BLOCK_DIM), div_ceil(N, BLOCK_DIM)),
            block_dim=(BLOCK_DIM, BLOCK_DIM),
            stream=stream,
        )

    run_func_bf16(stream)
    stream.synchronize()

    _copy_device_to_host(c_host, c_device, M * N)

    # running naive
    _copy_host_to_device(a_device_n, a_host_n, M * K)
    _copy_host_to_device(b_device_n, b_host_n, K * N)

    var func_gemm_fp32 = Function[
        fn (
            DTypePointer[DType.float32],
            DTypePointer[DType.float32],
            DTypePointer[DType.float32],
            Int,
            Int,
            Int,
        ) capturing -> None, matmul_kernel_naive[
            DType.float32,
            DType.float32,
            DType.float32,
            BLOCK_DIM,
        ]
    ]()

    @always_inline
    @__copy_capture(func_gemm_fp32, c_device_n, a_device_n, b_device_n)
    @parameter
    fn run_func_fp32(stream: Stream) raises:
        func_gemm_fp32(
            c_device_n,
            a_device_n,
            b_device_n,
            M,
            N,
            K,
            grid_dim=(div_ceil(M, BLOCK_DIM), div_ceil(N, BLOCK_DIM)),
            block_dim=(BLOCK_DIM, BLOCK_DIM),
            stream=stream,
        )

    run_func_fp32(stream)
    stream.synchronize()

    _copy_device_to_host(c_host_n, c_device_n, M * N)

    for i in range(M * N):
        var out_val = c_host.load(i)
        var out_ref = c_host_n.load(i).cast[DType.bfloat16]()
        testing.assert_true(math.isclose(out_val, out_ref))

    _free(a_device)
    _free(b_device)
    _free(c_device)

    _free(a_device_n)
    _free(b_device_n)
    _free(c_device_n)

    _ = a_host
    _ = b_host
    _ = c_host

    _ = a_host_n
    _ = b_host_n
    _ = c_host_n

    _ = func_gemm_bf16^
    _ = func_gemm_fp32^
    _ = stream^


# CHECK-NOT: CUDA_ERROR
def main():
    try:
        with Context() as ctx:
            run_matmul(1024, 1024, 1024)

    except e:
        print("CUDA_ERROR:", e)
