# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Meant to be run on an AVX512 system

import benchmark
from algorithm import sync_parallelize
from utils import unroll

from kernel_utils.layout import Layout
from kernel_utils.layout_tensor import LayoutTensor, TensorBuilder

alias MR = 6
alias NR = 64

alias dtype = DType.float32
alias simd_size = simdwidthof[dtype]()
alias alignment = 64


fn gemm_naive[
    layout_c: Layout,
    layout_a: Layout,
    layout_b: Layout,
](
    c: LayoutTensor[layout_c, dtype],  # M x N
    a: LayoutTensor[layout_a, dtype],  # M x K
    b: LayoutTensor[layout_b, dtype],  # N x K
):
    alias M = c.dim[0]()
    alias N = c.dim[1]()
    alias K = a.dim[1]()

    for mm in range(M):
        for kk in range(K):
            for nn in range(N):
                c[mm, nn] += a[mm, kk] * b[kk, nn]


fn kernel[
    layout_c: Layout,
    layout_a: Layout,
    layout_b: Layout,
](
    c: LayoutTensor[layout_c, dtype],  # MR, NR
    a: LayoutTensor[layout_a, dtype],  # MR, K
    b_packed: LayoutTensor[layout_b, dtype],  # 1, K * NR
):
    alias K = a.dim[1]()

    var c_cache = TensorBuilder[MR, NR, dtype].OnStackAligned[alignment]()

    @parameter
    @always_inline
    fn loadc[m: Int, n: Int]():
        c_cache.aligned_store[NR](m, NR * n, c.load[NR](m, NR * n))

    unroll[loadc, MR, 1]()

    for pr in range(K // NR):
        var a_tile = a.tile[MR, NR](0, pr)
        var b_row = b_packed.tile[1, NR * NR](0, pr)

        for k in range(NR):
            var b_next_tile = b_row.tile[1, NR](0, k + 4)

            @unroll
            for n in range(0, NR, simd_size):
                b_next_tile.prefetch(0, n)

            var b_tile = b_row.tile[1, NR](0, k)

            @unroll
            for m in range(MR):
                var av = a_tile[m, k]

                c_cache.aligned_store[NR](
                    m,
                    0,
                    av * b_tile.load[NR](0, 0) + c_cache.aligned_load[NR](m, 0),
                )

    @parameter
    @always_inline
    fn storec[m: Int, n: Int]():
        c.store[NR](
            m,
            NR * n,
            c_cache.aligned_load[NR](m, NR * n),
        )

    unroll[storec, MR, 1]()


fn pack_b[
    layout_b: Layout,
    layout_packed: Layout,
](
    b: LayoutTensor[layout_b, dtype],  # K x N
    packed: LayoutTensor[layout_packed, dtype],  # N // NR x K * NR
):
    alias K = b.dim[0]()
    alias N = b.dim[1]()

    for jc in range(N // NR):
        for pr in range(K // NR):
            var b_tile = b.tile[NR, NR](pr, jc)
            var packed_row = packed.tile[1, NR * NR](jc, pr)

            for k in range(NR):
                var packed_tile = packed_row.tile[1, NR](0, k)
                for n in range(NR):
                    packed_tile[0, n] = b_tile[k, n]


fn gemm[
    layout_c: Layout,
    layout_a: Layout,
    layout_b: Layout,
](
    c: LayoutTensor[layout_c, dtype],  # M x N
    a: LayoutTensor[layout_a, dtype],  # M x K
    b_packed: LayoutTensor[layout_b, dtype],  # (N // NR) x (K * NR)
):
    alias M = c.dim[0]()
    alias N = c.dim[1]()
    alias K = a.dim[1]()

    for jc in range(N // NR):
        var b_tile = b_packed.tile[1, K * NR](jc, 0)

        # @parameter
        # fn process_row(ir: Int):
        @unroll
        for ir in range(M // MR):
            var c_tile = c.tile[MR, NR](ir, jc)
            var a_tile = a.tile[MR, K](ir, 0)

            kernel(c_tile, a_tile, b_tile)

        # sync_parallelize[process_row](M // MR)


fn main():
    alias M: Int = 960
    alias N: Int = 1024
    alias K: Int = 1024

    if M % MR != 0:
        print("M must be multiple of", MR)
        return
    if N % NR != 0:
        print("N must be a multiple of", NR)
        return

    print_no_newline(M)
    print_no_newline("x")
    print_no_newline(N)
    print_no_newline("x")
    print(K)

    # FIXME: Something causes sporadic crashes on intel with TensorBuilder.Build()
    var a_ptr = DTypePointer[DType.float32].alloc(M * K, alignment=alignment)
    var b_ptr = DTypePointer[DType.float32].alloc(K * N, alignment=alignment)
    var b_packed_ptr = DTypePointer[DType.float32].alloc(
        K * N, alignment=alignment
    )
    var c_ptr = DTypePointer[DType.float32].alloc(M * N, alignment=alignment)
    var c2_ptr = DTypePointer[DType.float32].alloc(M * N, alignment=alignment)

    var a = TensorBuilder[M, K, dtype].Wrap(a_ptr)
    var b = TensorBuilder[K, N, dtype].Wrap(b_ptr)
    var b_packed = TensorBuilder[N // NR, K * NR, dtype].Wrap(b_packed_ptr)
    var c = TensorBuilder[M, N, dtype].Wrap(c_ptr)
    var c2 = TensorBuilder[M, N, dtype].Wrap(c2_ptr)

    for j in range(M):
        for i in range(K):
            a[j, i] = K * j + i

    for j in range(K):
        for i in range(N):
            b[j, i] = N * j + i

    for j in range(M):
        for i in range(N):
            c[j, i] = c2[j, i] = 0

    pack_b(b, b_packed)

    gemm_naive(c, a, b)
    gemm(c2, a, b_packed)
    var errors: Int = 0
    for j in range(M):
        for i in range(N):
            if c[j, i] != c2[j, i]:
                errors += 1

    print_no_newline(errors)
    print_no_newline("/")
    print_no_newline(M * N)
    print(" errors")

    @parameter
    fn bench_gemm():
        gemm(c2, a, b_packed)

    var num_warmup: Int = 1
    var time = benchmark.run[bench_gemm](num_warmup).mean()
    var flops = 2.0 * M * N * K / time / 1e9
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
    b_packed_ptr.free()
    c_ptr.free()
    c2_ptr.free()
