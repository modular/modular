# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Meant to be run on an AVX512 system

from Range import range
from IO import print, print_no_newline
from Pointer import DTypePointer
from List import Dim, DimList
from Buffer import Buffer, NDBuffer
from Index import Index
from Matrix import Matrix
from Benchmark import Benchmark
from Intrinsics import PrefetchOptions
from Functional import unroll

alias mr = 6
alias nr = 64

alias simd_size = 16
alias alignment = 64
alias accum_type = DType.float32


fn print_mat(a_ptr: DTypePointer[DType.float32], m: Int, n: Int):
    let a = Matrix[DimList.create_unknown[2](), DType.float32, False](
        a_ptr, Index(m, n), DType.float32
    )
    for i in range(m):
        for j in range(n):
            print_no_newline(a[i, j])
            print_no_newline(" ")
        print("")


fn gemm_naive(
    a: Matrix[DimList.create_unknown[2](), DType.float32, False],
    b: Matrix[DimList.create_unknown[2](), DType.float32, False],
    c: Matrix[DimList.create_unknown[2](), DType.float32, False],
    m: Int,
    n: Int,
    k: Int,
):
    for i in range(m):
        for p in range(k):
            for j in range(n):
                c[i, j] += a[i, p] * b[p, j]


fn kernel(
    a_ptr: DTypePointer[DType.float32],
    b_ptr: DTypePointer[DType.float32],
    c_ptr: DTypePointer[DType.float32],
    n: Int,
    k: Int,
    kc: Int,
):
    let a = Buffer[Dim(), DType.float32](a_ptr, mr * k)
    let b = Buffer[Dim(), DType.float32](b_ptr, kc * nr)
    let c = Buffer[Dim(), DType.float32](c_ptr, mr * n)

    let c_local = Buffer[mr * nr, DType.float32]().aligned_stack_allocation[
        alignment
    ]()

    alias nr2 = nr // simd_size

    @parameter
    @always_inline
    fn loadc[idx0: Int, idx1: Int]():
        let cv = c.simd_load[simd_size](n * idx0 + simd_size * idx1)
        c_local.simd_store[simd_size](nr * idx0 + simd_size * idx1, cv)

    unroll[mr, nr2, loadc]()

    for pr in range(kc):

        @parameter
        @always_inline
        fn prefetch[idx0: Int]():
            b_ptr.offset(nr * pr + simd_size * (idx0 + 16)).prefetch[
                PrefetchOptions().for_read().high_locality().to_data_cache()
            ]()

        unroll[nr2, prefetch]()

        @parameter
        @always_inline
        fn calc[idx0: Int, idx1: Int]():
            let av = a.simd_load[1](idx0 * k + pr).cast[accum_type]()
            let bv = b.simd_load[simd_size](nr * pr + simd_size * idx1)
            var cv = c_local.simd_load[simd_size](nr * idx0 + simd_size * idx1)
            cv += av * bv
            c_local.simd_store[simd_size](nr * idx0 + simd_size * idx1, cv)

        unroll[mr, nr2, calc]()

    @parameter
    @always_inline
    fn storec[idx0: Int, idx1: Int]():
        let cv = c_local.simd_load[simd_size](nr * idx0 + simd_size * idx1)
        c.simd_store[simd_size](n * idx0 + simd_size * idx1, cv)

    unroll[mr, nr2, storec]()


fn pack_B(
    b_ptr: DTypePointer[DType.float32],
    b2_ptr: DTypePointer[DType.float32],
    k: Int,
    n: Int,
    kc: Int,
    nc: Int,
):
    let b = Buffer[Dim(), DType.float32](b_ptr, kc * n)
    let bc = Buffer[Dim(), DType.float32](b2_ptr, kc * n)
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
    let m: Int = 960
    let n: Int = 1024
    let k: Int = 1024
    let mc: Int = m
    let nc: Int = 64
    let kc: Int = k
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

    let a_ptr = DTypePointer[DType.float32].aligned_alloc(alignment, m * k)
    let b_ptr = DTypePointer[DType.float32].aligned_alloc(alignment, k * n)
    let b2_ptr = DTypePointer[DType.float32].aligned_alloc(alignment, k * n)
    let c_ptr = DTypePointer[DType.float32].aligned_alloc(alignment, m * n)
    let c2_ptr = DTypePointer[DType.float32].aligned_alloc(alignment, m * n)
    let a = Buffer[Dim(), DType.float32](a_ptr, m * k)
    let b = Buffer[Dim(), DType.float32](b_ptr, k * n)
    let b2 = Buffer[Dim(), DType.float32](b2_ptr, k * n)
    let c = Buffer[Dim(), DType.float32](c_ptr, m * n)
    let c2 = Buffer[Dim(), DType.float32](c2_ptr, m * n)

    let am = Matrix[DimList.create_unknown[2](), DType.float32, False](
        a_ptr, Index(m, k), DType.float32
    )
    let bm = Matrix[DimList.create_unknown[2](), DType.float32, False](
        b_ptr, Index(k, n), DType.float32
    )
    let cm = Matrix[DimList.create_unknown[2](), DType.float32, False](
        c_ptr, Index(m, n), DType.float32
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

    var time: Float32
    let ns_per_second: Int = 1_000_000_000
    var num_warmup: Int = 1
    time = Benchmark(num_warmup).run[bench_gemm]()
    time = time / ns_per_second
    let flops = 2.0 * m * n * k / time / 1e9
    print_no_newline(time)
    print(" seconds")
    print_no_newline(flops)
    print(" GFLOPS")

    # assume turbo is disabled and the frequency set to 2.9 GHz
    let rpeak = flops / (2.9 * 64)
    print_no_newline(rpeak)
    print(" measured/peak FLOPS assuming 2.9 GHz")

    a_ptr.free()
    b_ptr.free()
    b2_ptr.free()
    c_ptr.free()
    c2_ptr.free()
