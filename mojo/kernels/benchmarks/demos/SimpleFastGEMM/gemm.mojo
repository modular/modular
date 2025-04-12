# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Meant to be run on an AVX512 system

from math import align_up
from sys import alignof, prefetch, simdwidthof
from sys.intrinsics import PrefetchOptions

import benchmark
from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from linalg.utils import (
    get_matmul_kernel_shape,
    get_matmul_prefetch_b_distance_k,
)
from memory import UnsafePointer

from utils.index import Index

alias dtype = DType.float32
alias simd_size = simdwidthof[dtype]()
alias alignment = alignof[SIMD[dtype, simd_size]]()

alias kernel_shape = get_matmul_kernel_shape[dtype, dtype, dtype, False]()
alias MR = kernel_shape.simd_rows
alias NR = kernel_shape.simd_cols * simd_size

# AVX512 values
# alias MR = 6
# alias NR = 64

alias prefetch_distance = get_matmul_prefetch_b_distance_k()


fn print_mat(a_ptr: UnsafePointer[Scalar[dtype]], m: Int, n: Int):
    var a = NDBuffer[dtype, 2](a_ptr, Index(m, n))
    for i in range(m):
        for j in range(n):
            print(a[i, j], end=" ")
        print("")


fn gemm_naive(
    a: NDBuffer[dtype, 2],
    b: NDBuffer[dtype, 2],
    c: NDBuffer[dtype, 2],
    m: Int,
    n: Int,
    k: Int,
):
    for i in range(m):
        for p in range(k):
            for j in range(n):
                c[i, j] += a[i, p] * b[p, j]


fn kernel(
    a_ptr: UnsafePointer[Scalar[dtype]],
    b_ptr: UnsafePointer[Scalar[dtype]],
    c_ptr: UnsafePointer[Scalar[dtype]],
    n: Int,
    k: Int,
    kc: Int,
):
    var a = NDBuffer[dtype, 1](a_ptr, MR * k)
    var b = NDBuffer[dtype, 1](b_ptr, k * NR)
    var c = NDBuffer[dtype, 1](c_ptr, MR * n)

    var c_local = NDBuffer[
        dtype, 1, MutableAnyOrigin, MR * NR
    ]().stack_allocation[alignment=alignment]()

    alias NR2 = NR // simd_size

    @parameter
    for idx0 in range(MR):
        for idx1 in range(NR2):
            var cv = c.load[width=simd_size](n * idx0 + simd_size * idx1)
            c_local.store(NR * idx0 + simd_size * idx1, cv)

    for pr in range(kc):

        @parameter
        for i in range(NR2):
            prefetch[
                PrefetchOptions().for_read().high_locality().to_data_cache()
            ](b_ptr.offset(NR * pr + simd_size * (i + 16)))

        @parameter
        for idx0 in range(MR):
            for idx1 in range(NR2):
                var av = a[idx0 * k + pr].cast[dtype]()
                var bv = b.load[width=simd_size](NR * pr + simd_size * idx1)
                var cv = c_local.load[width=simd_size](
                    NR * idx0 + simd_size * idx1
                )
                cv += av * bv
                c_local.store(NR * idx0 + simd_size * idx1, cv)

    @parameter
    for idx0 in range(MR):
        for idx1 in range(NR2):
            var cv = c_local.load[width=simd_size](NR * idx0 + simd_size * idx1)
            c.store(n * idx0 + simd_size * idx1, cv)


fn pack_B(
    b_ptr: UnsafePointer[Scalar[dtype]],
    b2_ptr: UnsafePointer[Scalar[dtype]],
    k: Int,
    n: Int,
    kc: Int,
    nc: Int,
):
    var b = NDBuffer[dtype, 1](b_ptr, k * n)
    var bc = NDBuffer[dtype, 1](b2_ptr, k * n)
    for pr in range(kc):
        for ir in range(nc // NR):
            for v in range(NR):
                bc[NR * (pr + kc * ir) + v] = b[pr * n + NR * ir + v]


fn prepack_B(
    b_ptr: UnsafePointer[Scalar[dtype]],
    b2_ptr: UnsafePointer[Scalar[dtype]],
    k: Int,
    n: Int,
    kc: Int,
    nc: Int,
):
    for pc in range(0, k, kc):
        for jc in range(0, n, nc):
            pack_B(b_ptr + pc * n + jc, b2_ptr + n * pc + jc * kc, k, n, kc, nc)


fn gemm(
    a_ptr: UnsafePointer[Scalar[dtype]],
    b_ptr: UnsafePointer[Scalar[dtype]],
    c_ptr: UnsafePointer[Scalar[dtype]],
    m: Int,
    n: Int,
    k: Int,
    mc: Int,
    nc: Int,
    kc: Int,
):
    for ic in range(0, m, mc):
        for pc in range(0, k, kc):
            for jc in range(0, n, nc):
                for ir in range(0, mc, MR):
                    for jr in range(0, nc, NR):
                        kernel(
                            a_ptr + (ic + ir) * k + pc,
                            b_ptr + n * pc + jc * kc + jr * kc,
                            c_ptr + (ic + ir) * n + jc + jr,
                            n,
                            k,
                            kc,
                        )


fn main() raises:
    var m = align_up(1024, MR)
    var n = align_up(1024, NR)
    var k: Int = 1024
    var mc: Int = m
    var nc: Int = NR
    var kc: Int = k
    if m % MR != 0:
        print("m must be multiple of 6")
        return
    if n % NR != 0:
        print("n must be a multiple of 64")
        return

    print(m, end="")
    print("x", end="")
    print(n, end="")
    print("x", end="")
    print(k)

    var a_ptr = UnsafePointer[Scalar[dtype], alignment=alignment].alloc(m * k)
    var b_ptr = UnsafePointer[Scalar[dtype], alignment=alignment].alloc(k * n)
    var b2_ptr = UnsafePointer[Scalar[dtype], alignment=alignment].alloc(k * n)
    var c_ptr = UnsafePointer[Scalar[dtype], alignment=alignment].alloc(m * n)
    var c2_ptr = UnsafePointer[Scalar[dtype], alignment=alignment].alloc(m * n)
    var a = NDBuffer[dtype, 1](a_ptr, m * k)
    var b = NDBuffer[dtype, 1](b_ptr, k * n)
    var b2 = NDBuffer[dtype, 1](b2_ptr, k * n)
    var c = NDBuffer[dtype, 1](c_ptr, m * n)
    var c2 = NDBuffer[dtype, 1](c2_ptr, m * n)

    var am = NDBuffer[dtype, 2](a_ptr, Index(m, k))
    var bm = NDBuffer[dtype, 2](b_ptr, Index(k, n))
    var cm = NDBuffer[dtype, 2](c_ptr, Index(m, n))

    for i in range(m * k):
        a[i] = i
    for i in range(k * n):
        b[i] = i
        b2[i] = i
    for i in range(m * n):
        c[i] = i
        c2[i] = i

    prepack_B(b.data, b2.data, k, n, kc, nc)

    gemm_naive(am, bm, cm, m, n, k)
    gemm(a.data, b2.data, c2.data, m, n, k, mc, nc, kc)
    var errors: Int = 0
    for i in range(m * n):
        if c[i] != c2[i]:
            errors += 1
    print(errors, end="")
    print("/", end="")
    print(m * n, end="")
    print(" errors")

    @parameter
    fn bench_gemm():
        gemm(a.data, b2.data, c2.data, m, n, k, mc, nc, kc)

    var num_warmup: Int = 1
    var time = benchmark.run[bench_gemm](num_warmup).mean()
    var flops = 2.0 * m * n * k / time / 1e9
    print(time, end="")
    print(" seconds")
    print(flops, end="")
    print(" GFLOPS")

    # assume turbo is disabled and the frequency set to 2.9 GHz
    var rpeak = flops / (2.9 * 64)
    print(rpeak, end="")
    print(" measured/peak FLOPS assuming 2.9 GHz")

    a_ptr.free()
    b_ptr.free()
    b2_ptr.free()
    c_ptr.free()
    c2_ptr.free()
