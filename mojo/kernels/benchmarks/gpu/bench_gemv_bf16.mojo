# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s -t | FileCheck %s
# CHECK: Benchmark results

from math import ceildiv

from benchmark import *
from gpu import WARP_SIZE
from gpu.host import Context, Function, Stream, synchronize
from gpu.host.memory import (
    _copy_host_to_device,
    _free,
    _malloc,
)
from linalg.matmul_gpu import (
    gemv_kernel,
    gemv_tc_kernel,
    matmul_kernel_naive,
)

from memory.unsafe import DTypePointer
from memory import memset

from random import randn


@parameter
fn no_raise[
    func: fn (inout Bencher, GemvSpec) raises -> None
](inout m: Bencher, spec: GemvSpec):
    try:
        func(m, spec)
    except e:
        abort(e)


fn bench_gemv_tc(inout bencher: Bencher, spec: GemvSpec) raises:
    var M = spec.m
    var N = spec.n
    var K = spec.k
    var stream = Stream()
    var a_host = DTypePointer[DType.bfloat16].alloc(M * K)
    var b_host = DTypePointer[DType.bfloat16].alloc(K * N)
    var c_host = DTypePointer[DType.float32].alloc(M * N)
    var rand_min = -1000
    var rand_max = 1000
    randn(a_host, M * K)
    randn(b_host, K * N)
    memset(c_host, 0, M * N)

    var a_device = _malloc[BFloat16](M * K)
    var b_device = _malloc[BFloat16](K * N)
    var c_device = _malloc[Float32](M * N)

    _copy_host_to_device(a_device, a_host, M * K)
    _copy_host_to_device(b_device, b_host, K * N)

    alias WARPS_PER_BLOCK = 32
    var func_gemv = Function[
        gemv_tc_kernel[
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
    var a_host = DTypePointer[DType.bfloat16].alloc(M * K)
    var b_host = DTypePointer[DType.bfloat16].alloc(K * N)
    var c_host = DTypePointer[DType.float32].alloc(M * N)
    var rand_min = -1000
    var rand_max = 1000
    randn(a_host, M * K)
    randn(b_host, K * N)
    memset(c_host, 0, M * N)

    var a_device = _malloc[BFloat16](M * K)
    var b_device = _malloc[BFloat16](K * N)
    var c_device = _malloc[Float32](M * N)

    _copy_host_to_device(a_device, a_host, M * K)
    _copy_host_to_device(b_device, b_host, K * N)

    alias WARPS_PER_BLOCK = 32
    var func_gemv = Function[
        gemv_kernel[
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
    var a_host = DTypePointer[DType.float32].alloc(M * K)
    var b_host = DTypePointer[DType.float32].alloc(K * N)
    var c_host = DTypePointer[DType.float32].alloc(M * N)
    var rand_min = -1000
    var rand_max = 1000
    randn(a_host, M * K)
    randn(b_host, K * N)
    memset(c_host, 0, M * N)

    var a_device = _malloc[Float32](M * K)
    var b_device = _malloc[Float32](K * N)
    var c_device = _malloc[Float32](M * N)

    _copy_host_to_device(a_device, a_host, M * K)
    _copy_host_to_device(b_device, b_host, K * N)

    alias BLOCK_DIM = 16
    alias WARPS_PER_BLOCK = 32
    var func_gemv = Function[
        matmul_kernel_naive[
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

        fn bench_gemv_impls(spec: GemvSpec) raises:
            m.bench_with_input[GemvSpec, no_raise[bench_gemv_tc]](
                BenchId("gemv_tc", str(spec)), spec
            )
            m.bench_with_input[GemvSpec, no_raise[bench_gemv_ws]](
                BenchId("gemv_ws", str(spec)), spec
            )
            m.bench_with_input[GemvSpec, no_raise[bench_gemv_naive]](
                BenchId("gemv_naive", str(spec)), spec
            )

        bench_gemv_impls(GemvSpec(m=4096, n=1, k=4096))
        # TODO: bench_gemv_tc, bench_gemv_ws, and bench_gemv_naive
        # all do really large allocations and scalar fills of the same buffers
        # Needs be merged into a common function before uncommenting below
        # bench_gemv_impls(GemvSpec(m=8192, n=1, k=8192))
        # bench_gemv_impls(GemvSpec(m=16834, n=1, k=16834))

        m.dump_report()
