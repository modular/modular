# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Meant to be run on an AVX512 system

from math import align_up
from sys.intrinsics import PrefetchOptions

import benchmark
from buffer import Buffer, NDBuffer
from buffer.list import Dim, DimList
from linalg.utils import (
    get_matmul_kernel_shape,
    get_matmul_prefetch_b_distance_k,
)
from memory.unsafe import DTypePointer

from utils.index import Index
from utils.loop import unroll

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


fn print_mat(a_ptr: DTypePointer[dtype], m: Int, n: Int):
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
                c[(i, j)] += a[i, p] * b[p, j]


fn kernel(
    a_ptr: DTypePointer[dtype],
    b_ptr: DTypePointer[dtype],
    c_ptr: DTypePointer[dtype],
    n: Int,
    k: Int,
    kc: Int,
):
    var a = Buffer[dtype](a_ptr, MR * k)
    var b = Buffer[dtype](b_ptr, k * NR)
    var c = Buffer[dtype](c_ptr, MR * n)

    var c_local = Buffer[dtype, size = MR * NR]().stack_allocation[
        alignment=alignment
    ]()

    alias NR2 = NR // simd_size

    @parameter
    @always_inline
    fn loadc[idx0: Int, idx1: Int]():
        var cv = c.load[width=simd_size](n * idx0 + simd_size * idx1)
        c_local.store[width=simd_size](NR * idx0 + simd_size * idx1, cv)

    unroll[loadc, MR, NR2]()

    for pr in range(kc):

        @parameter
        @always_inline
        fn prefetch[idx0: Int]():
            SIMD.prefetch[
                PrefetchOptions().for_read().high_locality().to_data_cache()
            ](b_ptr.offset(NR * pr + simd_size * (idx0 + 16)))

        unroll[prefetch, NR2]()

        @parameter
        @always_inline
        fn calc[idx0: Int, idx1: Int]():
            var av = a[idx0 * k + pr].cast[dtype]()
            var bv = b.load[width=simd_size](NR * pr + simd_size * idx1)
            var cv = c_local.load[width=simd_size](NR * idx0 + simd_size * idx1)
            cv += av * bv
            c_local.store[width=simd_size](NR * idx0 + simd_size * idx1, cv)

        unroll[calc, MR, NR2]()

    @parameter
    @always_inline
    fn storec[idx0: Int, idx1: Int]():
        var cv = c_local.load[width=simd_size](NR * idx0 + simd_size * idx1)
        c.store[width=simd_size](n * idx0 + simd_size * idx1, cv)

    unroll[storec, MR, NR2]()


fn pack_B(
    b_ptr: DTypePointer[dtype],
    b2_ptr: DTypePointer[dtype],
    k: Int,
    n: Int,
    kc: Int,
    nc: Int,
):
    var b = Buffer[dtype](b_ptr, k * n)
    var bc = Buffer[dtype](b2_ptr, k * n)
    for pr in range(kc):
        for ir in range(nc // NR):
            for v in range(NR):
                bc[NR * (pr + kc * ir) + v] = b[pr * n + NR * ir + v]


fn prepack_B(
    b_ptr: DTypePointer[dtype],
    b2_ptr: DTypePointer[dtype],
    k: Int,
    n: Int,
    kc: Int,
    nc: Int,
):
    for pc in range(0, k, kc):
        for jc in range(0, n, nc):
            pack_B(b_ptr + pc * n + jc, b2_ptr + n * pc + jc * kc, k, n, kc, nc)


fn gemm(
    a_ptr: DTypePointer[dtype],
    b_ptr: DTypePointer[dtype],
    c_ptr: DTypePointer[dtype],
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


fn main():
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

    var a_ptr = DTypePointer[dtype].alloc(m * k, alignment=alignment)
    var b_ptr = DTypePointer[dtype].alloc(k * n, alignment=alignment)
    var b2_ptr = DTypePointer[dtype].alloc(k * n, alignment=alignment)
    var c_ptr = DTypePointer[dtype].alloc(m * n, alignment=alignment)
    var c2_ptr = DTypePointer[dtype].alloc(m * n, alignment=alignment)
    var a = Buffer[dtype](a_ptr, m * k)
    var b = Buffer[dtype](b_ptr, k * n)
    var b2 = Buffer[dtype](b2_ptr, k * n)
    var c = Buffer[dtype](c_ptr, m * n)
    var c2 = Buffer[dtype](c2_ptr, m * n)

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
