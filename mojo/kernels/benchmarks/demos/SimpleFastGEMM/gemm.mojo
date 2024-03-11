# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Meant to be run on an AVX512 system

from sys.intrinsics import PrefetchOptions

import benchmark
from utils import unroll
from memory.buffer import Buffer, NDBuffer
from memory.unsafe import DTypePointer

from utils.index import Index
from utils.list import Dim, DimList

alias mr = 6
alias nr = 64

alias simd_size = 16
alias alignment = 64
alias accum_type = DType.float32


fn print_mat(a_ptr: DTypePointer[DType.float32], m: Int, n: Int):
    var a = NDBuffer[DType.float32, 2, DimList.create_unknown[2]()](
        a_ptr, Index(m, n)
    )
    for i in range(m):
        for j in range(n):
            print_no_newline(a[i, j])
            print_no_newline(" ")
        print("")


fn gemm_naive(
    a: NDBuffer[DType.float32, 2, shape = DimList.create_unknown[2]()],
    b: NDBuffer[DType.float32, 2, shape = DimList.create_unknown[2]()],
    c: NDBuffer[DType.float32, 2, shape = DimList.create_unknown[2]()],
    m: Int,
    n: Int,
    k: Int,
):
    for i in range(m):
        for p in range(k):
            for j in range(n):
                c[(i, j)] += a[i, p] * b[p, j]


fn kernel(
    a_ptr: DTypePointer[DType.float32],
    b_ptr: DTypePointer[DType.float32],
    c_ptr: DTypePointer[DType.float32],
    n: Int,
    k: Int,
    kc: Int,
):
    var a = Buffer[DType.float32, size = Dim()](a_ptr, mr * k)
    var b = Buffer[DType.float32, size = Dim()](b_ptr, k * nr)
    var c = Buffer[DType.float32, size = Dim()](c_ptr, mr * n)

    var c_local = Buffer[
        DType.float32, size = mr * nr
    ]().aligned_stack_allocation[alignment]()

    alias nr2 = nr // simd_size

    @parameter
    @always_inline
    fn loadc[idx0: Int, idx1: Int]():
        var cv = c.simd_load[simd_size](n * idx0 + simd_size * idx1)
        c_local.simd_store[simd_size](nr * idx0 + simd_size * idx1, cv)

    unroll[loadc, mr, nr2]()

    for pr in range(kc):

        @parameter
        @always_inline
        fn prefetch[idx0: Int]():
            b_ptr.offset(nr * pr + simd_size * (idx0 + 16)).prefetch[
                PrefetchOptions().for_read().high_locality().to_data_cache()
            ]()

        unroll[prefetch, nr2]()

        @parameter
        @always_inline
        fn calc[idx0: Int, idx1: Int]():
            var av = a.simd_load[1](idx0 * k + pr).cast[accum_type]()
            var bv = b.simd_load[simd_size](nr * pr + simd_size * idx1)
            var cv = c_local.simd_load[simd_size](nr * idx0 + simd_size * idx1)
            cv += av * bv
            c_local.simd_store[simd_size](nr * idx0 + simd_size * idx1, cv)

        unroll[calc, mr, nr2]()

    @parameter
    @always_inline
    fn storec[idx0: Int, idx1: Int]():
        var cv = c_local.simd_load[simd_size](nr * idx0 + simd_size * idx1)
        c.simd_store[simd_size](n * idx0 + simd_size * idx1, cv)

    unroll[storec, mr, nr2]()


fn pack_B(
    b_ptr: DTypePointer[DType.float32],
    b2_ptr: DTypePointer[DType.float32],
    k: Int,
    n: Int,
    kc: Int,
    nc: Int,
):
    var b = Buffer[DType.float32, size = Dim()](b_ptr, k * n)
    var bc = Buffer[DType.float32, size = Dim()](b2_ptr, k * n)
    for pr in range(kc):
        for ir in range(nc // nr):
            for v in range(nr):
                bc[nr * (pr + kc * ir) + v] = b[pr * n + nr * ir + v]


fn prepack_B(
    b_ptr: DTypePointer[DType.float32],
    b2_ptr: DTypePointer[DType.float32],
    k: Int,
    n: Int,
    kc: Int,
    nc: Int,
):
    for pc in range(0, k, kc):
        for jc in range(0, n, nc):
            pack_B(b_ptr + pc * n + jc, b2_ptr + n * pc + jc * kc, k, n, kc, nc)


fn gemm(
    a_ptr: DTypePointer[DType.float32],
    b_ptr: DTypePointer[DType.float32],
    c_ptr: DTypePointer[DType.float32],
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
                for ir in range(0, mc, mr):
                    for jr in range(0, nc, nr):
                        kernel(
                            a_ptr + (ic + ir) * k + pc,
                            b_ptr + n * pc + jc * kc + jr * kc,
                            c_ptr + (ic + ir) * n + jc + jr,
                            n,
                            k,
                            kc,
                        )


fn main():
    var m: Int = 960
    var n: Int = 1024
    var k: Int = 1024
    var mc: Int = m
    var nc: Int = 64
    var kc: Int = k
    if m % 6 != 0:
        print("m must be multiple of 6")
        return
    if n % 64 != 0:
        print("n must be a multiple of 64")
        return

    print_no_newline(m)
    print_no_newline("x")
    print_no_newline(n)
    print_no_newline("x")
    print(k)

    var a_ptr = DTypePointer[DType.float32].alloc(m * k, alignment=alignment)
    var b_ptr = DTypePointer[DType.float32].alloc(k * n, alignment=alignment)
    var b2_ptr = DTypePointer[DType.float32].alloc(k * n, alignment=alignment)
    var c_ptr = DTypePointer[DType.float32].alloc(m * n, alignment=alignment)
    var c2_ptr = DTypePointer[DType.float32].alloc(m * n, alignment=alignment)
    var a = Buffer[DType.float32, size = Dim()](a_ptr, m * k)
    var b = Buffer[DType.float32, size = Dim()](b_ptr, k * n)
    var b2 = Buffer[DType.float32, size = Dim()](b2_ptr, k * n)
    var c = Buffer[DType.float32, size = Dim()](c_ptr, m * n)
    var c2 = Buffer[DType.float32, size = Dim()](c2_ptr, m * n)

    var am = NDBuffer[DType.float32, 2, DimList.create_unknown[2]()](
        a_ptr, Index(m, k)
    )
    var bm = NDBuffer[DType.float32, 2, DimList.create_unknown[2]()](
        b_ptr, Index(k, n)
    )
    var cm = NDBuffer[DType.float32, 2, DimList.create_unknown[2]()](
        c_ptr, Index(m, n)
    )

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
    print_no_newline(errors)
    print_no_newline("/")
    print_no_newline(m * n)
    print(" errors")

    @parameter
    fn bench_gemm():
        gemm(a.data, b2.data, c2.data, m, n, k, mc, nc, kc)

    var num_warmup: Int = 1
    var time = benchmark.run[bench_gemm](num_warmup).mean()
    var flops = 2.0 * m * n * k / time / 1e9
    print_no_newline(time)
    print(" seconds")
    print_no_newline(flops)
    print(" GFLOPS")

    # assume turbo is disabled and the frequency set to 2.9 GHz
    var rpeak = flops / (2.9 * 64)
    print_no_newline(rpeak)
    print(" measured/peak FLOPS assuming 2.9 GHz")

    a_ptr.free()
    b_ptr.free()
    b2_ptr.free()
    c_ptr.free()
    c2_ptr.free()
