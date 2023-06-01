# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Range import range
from Object import object
from Random import random_float64
from IO import print
from SIMD import Float64
from Benchmark import Benchmark


def main():
    benchmark_matmul()


def benchmark_matmul():
    var C: object = []
    var A: object = []
    var B: object = []
    var c: object
    var b: object
    var a: object
    for i in range(128):
        c = []
        b = []
        a = []
        for j in range(128):
            c.append(0)
            b.append(random_float64(-5, 5))
            a.append(random_float64(-5, 5))
        C.append(c)
        B.append(b)
        A.append(a)

    @always_inline
    fn test_fn():
        try:
            matmul(C, A, B, 128, 128, 128)
        except:
            pass

    print(Float64(Benchmark(2).run[test_fn]()) / 1000000000)


def matmul(C, A, B, M, N, K):
    for m in range(M):
        for n in range(N):
            for k in range(K):
                C[m][n] += A[m][k] * B[k][n]
