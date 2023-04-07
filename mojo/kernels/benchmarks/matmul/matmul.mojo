# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# This demonstrates incremental improvements to a naive matmul.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s -execute | FileCheck %s

from Benchmark import Benchmark
from DType import DType
from SIMD import F32
from Int import Int
from IO import print
from List import create_dim_list
from Matrix import Matrix
from Pointer import DTypePointer
from Range import range
from TargetInfo import dtype_sizeof

alias f32 = DType.f32

alias M = 128
alias N = 128
alias K = 128


@always_inline
fn naive_matmul(
    C: Matrix[create_dim_list(M, N), f32, False],
    A: Matrix[create_dim_list(M, K), f32, False],
    B: Matrix[create_dim_list(K, N), f32, False],
):
    for m in range(M):
        for n in range(N):
            for k in range(K):
                C[m, n] += A[m, k] * B[k, n]


fn benchmark_naive_matmul():
    let a_ptr = DTypePointer[f32].alloc(M * K)
    let b_ptr = DTypePointer[f32].alloc(K * N)
    let c_ptr = DTypePointer[f32].alloc(M * N)
    let A = Matrix[create_dim_list(M, K), f32, False](a_ptr)
    let B = Matrix[create_dim_list(K, N), f32, False](b_ptr)
    let C = Matrix[create_dim_list(M, N), f32, False](c_ptr)

    @always_inline
    fn benchmark_fn():
        naive_matmul(A, B, C)

    print(F32(Benchmark().run[benchmark_fn]()) / F32(1_000_000_000))

    a_ptr.free()
    b_ptr.free()
    c_ptr.free()


fn main():
    benchmark_naive_matmul()
