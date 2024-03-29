# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo %s | FileCheck %s

from math import ceildiv, max, min

from benchmark import *
from buffer import NDBuffer
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
    gemv_kernel,
    gevm_kernel,
    gemv_tc_kernel,
    matmul_kernel,
    matmul_kernel_naive,
)
from memory.unsafe import DTypePointer, bitcast

from utils.index import Index
from utils.list import DimList
from random import random_float64


fn bench_gemv_tc(inout m: Bench, spec: GemvSpec) raises:
    try:

        @parameter
        @always_inline
        fn bench_gemv_tc_wrapper(inout b: Bencher, concrete_spec: GemvSpec):
            try:
                bench_gemv_tc(b, concrete_spec)
            except:
                print("kernel failed")

        m.bench_with_input[GemvSpec, bench_gemv_tc_wrapper](
            BenchId("gemv_tc", str(spec)), spec
        )
    except:
        print("launch failed")


fn bench_gemv_ws(inout m: Bench, spec: GemvSpec) raises:
    try:

        @parameter
        @always_inline
        fn bench_gemv_ws_wrapper(inout b: Bencher, concrete_spec: GemvSpec):
            try:
                bench_gemv_ws(b, concrete_spec)
            except:
                print("kernel failed")

        m.bench_with_input[GemvSpec, bench_gemv_ws_wrapper](
            BenchId("gemv_ws", str(spec)), spec
        )
    except:
        print("launch failed")


fn bench_gemv_naive(inout m: Bench, spec: GemvSpec) raises:
    try:

        @parameter
        @always_inline
        fn bench_gemv_naive_wrapper(inout b: Bencher, concrete_spec: GemvSpec):
            try:
                bench_gemv_naive(b, concrete_spec)
            except:
                print("kernel failed")

        m.bench_with_input[GemvSpec, bench_gemv_naive_wrapper](
            BenchId("gemv_naive", str(spec)), spec
        )
    except:
        print("launch failed")


fn bench_gemv_tc(inout bencher: Bencher, spec: GemvSpec) raises:
    var M = spec.m
    var N = spec.n
    var K = spec.k
    var stream = Stream()
    var a_host = Pointer[BFloat16].alloc(M * K)
    var b_host = Pointer[BFloat16].alloc(K * N)
    var c_host = Pointer[Float32].alloc(M * N)
    var rand_min = -1000
    var rand_max = 1000
    for i in range(M * K):
        a_host[i] = random_float64(rand_min, rand_max)

    for i in range(K * N):
        b_host[i] = random_float64(rand_min, rand_max)

    for i in range(M * N):
        c_host[i] = 0

    var a_device = _malloc[BFloat16](M * K)
    var b_device = _malloc[BFloat16](K * N)
    var c_device = _malloc[Float32](M * N)

    _copy_host_to_device(a_device, a_host, M * K)
    _copy_host_to_device(b_device, b_host, K * N)

    alias WARPS_PER_BLOCK = 32
    var func_gemv = Function[
        fn (
            DTypePointer[DType.float32],
            DTypePointer[DType.bfloat16],
            DTypePointer[DType.bfloat16],
            Int,
            Int,
            Int,
        ) capturing -> None, gemv_tc_kernel[
            DType.float32,
            DType.bfloat16,
            DType.bfloat16,
        ]
    ]()

    @always_inline
    @parameter
    fn bench_fn() raises:
        func_gemv(
            c_device,
            a_device,
            b_device,
            M,
            N,
            K,
            grid_dim=ceildiv(M, WARPS_PER_BLOCK),
            block_dim=WARP_SIZE * WARPS_PER_BLOCK,
            stream=stream,
        )

    bencher.iter[bench_fn]()
    stream.synchronize()

    _free(a_device)
    _free(b_device)
    _free(c_device)

    _ = a_host
    _ = b_host
    _ = c_host

    _ = func_gemv^
    _ = stream^


fn bench_gemv_ws(inout bencher: Bencher, spec: GemvSpec) raises:
    var M = spec.m
    var N = spec.n
    var K = spec.k
    var stream = Stream()
    var a_host = Pointer[BFloat16].alloc(M * K)
    var b_host = Pointer[BFloat16].alloc(K * N)
    var c_host = Pointer[Float32].alloc(M * N)
    var rand_min = -1000
    var rand_max = 1000
    for i in range(M * K):
        a_host[i] = random_float64(rand_min, rand_max)

    for i in range(K * N):
        b_host[i] = random_float64(rand_min, rand_max)

    for i in range(M * N):
        c_host[i] = 0

    var a_device = _malloc[BFloat16](M * K)
    var b_device = _malloc[BFloat16](K * N)
    var c_device = _malloc[Float32](M * N)

    _copy_host_to_device(a_device, a_host, M * K)
    _copy_host_to_device(b_device, b_host, K * N)

    alias WARPS_PER_BLOCK = 32
    var func_gemv = Function[
        fn (
            DTypePointer[DType.float32],
            DTypePointer[DType.bfloat16],
            DTypePointer[DType.bfloat16],
            Int,
            Int,
            Int,
        ) capturing -> None, gemv_kernel[
            DType.float32,
            DType.bfloat16,
            DType.bfloat16,
        ]
    ]()

    @always_inline
    @parameter
    fn bench_fn() raises:
        func_gemv(
            c_device,
            a_device,
            b_device,
            M,
            N,
            K,
            grid_dim=ceildiv(M, WARPS_PER_BLOCK),
            block_dim=WARP_SIZE * WARPS_PER_BLOCK,
            stream=stream,
        )

    bencher.iter[bench_fn]()
    stream.synchronize()

    _free(a_device)
    _free(b_device)
    _free(c_device)

    _ = a_host
    _ = b_host
    _ = c_host

    _ = func_gemv^
    _ = stream^


fn bench_gemv_naive(inout bencher: Bencher, spec: GemvSpec) raises:
    var M = spec.m
    var N = spec.n
    var K = spec.k
    var stream = Stream()
    var a_host = Pointer[Float32].alloc(M * K)
    var b_host = Pointer[Float32].alloc(K * N)
    var c_host = Pointer[Float32].alloc(M * N)
    var rand_min = -1000
    var rand_max = 1000
    for i in range(M * K):
        a_host[i] = random_float64(rand_min, rand_max)

    for i in range(K * N):
        b_host[i] = random_float64(rand_min, rand_max)

    for i in range(M * N):
        c_host[i] = 0

    var a_device = _malloc[Float32](M * K)
    var b_device = _malloc[Float32](K * N)
    var c_device = _malloc[Float32](M * N)

    _copy_host_to_device(a_device, a_host, M * K)
    _copy_host_to_device(b_device, b_host, K * N)

    alias BLOCK_DIM = 16
    alias WARPS_PER_BLOCK = 32
    var func_gemv = Function[
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
    @parameter
    fn bench_fn() raises:
        func_gemv(
            c_device,
            a_device,
            b_device,
            M,
            N,
            K,
            grid_dim=(ceildiv(M, WARPS_PER_BLOCK), ceildiv(N, BLOCK_DIM)),
            block_dim=(BLOCK_DIM, BLOCK_DIM),
            stream=stream,
        )

    bencher.iter[bench_fn]()
    stream.synchronize()

    _free(a_device)
    _free(b_device)
    _free(c_device)

    _ = a_host
    _ = b_host
    _ = c_host

    _ = func_gemv^
    _ = stream^


@value
struct GemvSpec(Stringable):
    var m: Int
    var n: Int
    var k: Int

    fn __str__(self) -> String:
        return "m=" + str(self.m) + ";n=" + str(self.n) + ";k=" + str(self.k)

    fn flops(self) -> Int:
        return 2 * self.m * self.n * self.k


def main():
    with Context() as ctx:
        var m = Bench(BenchConfig(num_repetitions=1))

        bench_gemv_tc(m, GemvSpec(m=4096, n=1, k=4096))
        bench_gemv_ws(m, GemvSpec(m=4096, n=1, k=4096))
        bench_gemv_naive(m, GemvSpec(m=4096, n=1, k=4096))

        bench_gemv_tc(m, GemvSpec(m=8192, n=1, k=8192))
        bench_gemv_ws(m, GemvSpec(m=8192, n=1, k=8192))
        bench_gemv_naive(m, GemvSpec(m=8192, n=1, k=8192))

        bench_gemv_tc(m, GemvSpec(m=16384, n=1, k=16384))
        bench_gemv_ws(m, GemvSpec(m=16384, n=1, k=16384))
        bench_gemv_naive(m, GemvSpec(m=16384, n=1, k=16384))

        m.dump_report()
