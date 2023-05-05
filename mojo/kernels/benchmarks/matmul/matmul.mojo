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
from IO import print
from Pointer import DTypePointer
from Range import range


alias M = 128
alias N = 128
alias K = 128


struct Matrix:
    var data: DTypePointer[DType.f32]
    var rows: Int
    var cols: Int

    fn __init__(inout self, rows: Int, cols: Int):
        self.data = DTypePointer[DType.f32].alloc(rows * cols)
        self.rows = rows
        self.cols = cols

    fn __del__(owned self):
        self.data.free()

    fn __getitem__(self, row: Int, col: Int) -> F32:
        return self.data.load(row * self.cols + col)

    fn __setitem__(inout self, row: Int, col: Int, val: F32):
        self.data.store(row * self.cols + col, val)


@always_inline
fn naive_matmul(C&: Matrix, A&: Matrix, B&: Matrix):
    for m in range(M):
        for n in range(N):
            for k in range(K):
                C[m, n] += A[m, k] * B[k, n]


fn benchmark_naive_matmul():
    var A = Matrix(M, K)
    var B = Matrix(K, N)
    var C = Matrix(M, N)

    @always_inline
    fn benchmark_fn():
        naive_matmul(A, B, C)

    print(F32(Benchmark().run[benchmark_fn]()) / F32(1_000_000_000))


fn main():
    benchmark_naive_matmul()
