# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import timeit

import numpy as np

M = N = K = 128
A = B = C = [[]]


def setup():
    global A, B, C
    A = list(np.random.rand(M, K))
    B = list(np.random.rand(K, N))
    C = list(np.random.rand(M, N))


def matmul():
    global A, B, C
    for m in range(M):
        for n in range(N):
            for k in range(K):
                C[m][n] += A[m][k] * B[k][n]


def main():
    times = timeit.repeat(matmul, setup=setup, number=100, repeat=1)
    print(round(min(times), 2))


if __name__ == "__main__":
    main()
