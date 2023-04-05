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
from F32 import F32
from Int import Int
from IO import print
from List import create_dim_list
from Matrix import Matrix
from Memory import _aligned_alloc, _aligned_free
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
    let a_ptr: DTypePointer[f32] = _aligned_alloc[
        __mlir_type[`!pop.scalar<`, f32.value, `>`]
    ](M * K * dtype_sizeof[f32]()).address
    let b_ptr: DTypePointer[f32] = _aligned_alloc[
        __mlir_type[`!pop.scalar<`, f32.value, `>`]
    ](K * N * dtype_sizeof[f32]()).address
    let c_ptr: DTypePointer[f32] = _aligned_alloc[
        __mlir_type[`!pop.scalar<`, f32.value, `>`]
    ](M * N * dtype_sizeof[f32]()).address
    let A = Matrix[create_dim_list(M, K), f32, False](a_ptr)
    let B = Matrix[create_dim_list(K, N), f32, False](b_ptr)
    let C = Matrix[create_dim_list(M, N), f32, False](c_ptr)

    @always_inline
    fn benchmark_fn():
        naive_matmul(A, B, C)

    print(F32(Benchmark().run[benchmark_fn]()) / F32(1_000_000_000))

    _aligned_free(a_ptr)
    _aligned_free(b_ptr)
    _aligned_free(c_ptr)


fn main():
    benchmark_naive_matmul()
