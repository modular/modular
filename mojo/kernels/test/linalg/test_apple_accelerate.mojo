# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s

from sys.info import os_is_macos

from buffer import NDBuffer
from buffer.list import DimList
from LinAlg.apple_accelerate import matmul
from utils.index import Index, StaticIntTuple
from testing import *

alias alignment = 64


alias a_type = DType.float32
alias b_type = DType.float32
alias c_type = DType.float32


fn gemm_naive(
    c: NDBuffer,
    a: NDBuffer,
    b: NDBuffer,
    m: Int,
    n: Int,
    k: Int,
):
    for i in range(m):
        for p in range(k):
            for j in range(n):
                var a_val = a[i, p].cast[c.type]()
                var b_val = b[p, j].cast[c.type]()
                c[(i, j)] += a_val * b_val


def test_matmul(c: NDBuffer, a: NDBuffer, b: NDBuffer, m: Int, n: Int, k: Int):
    var golden_ptr = DTypePointer[c.type].alloc(m * n, alignment=alignment)
    var golden = NDBuffer[c.type, 2](golden_ptr, Index(m, n))

    for i in range(m):
        for j in range(k):
            a[(i, j)] = 1  # i * 0.001 + j * 0.001

    for i in range(k):
        for j in range(n):
            b[(i, j)] = 1  # i * 0.001 + k * 0.001

    for i in range(m):
        for j in range(n):
            c[(i, j)] = 0
            golden[(i, j)] = 0

    matmul(c, a, b)
    gemm_naive(golden, a, b, m, n, k)

    var errors: Int = 0
    for i in range(m):
        for j in range(n):
            if c[i, j] != golden[i, j]:
                if errors < 10:
                    print(c[i, j] - golden[i, j])
                errors += 1

    assert_true(
        errors == 0,
        "num of errors must be 0, but got "
        + str(errors)
        + " for dimensions M="
        + str(m)
        + ", N="
        + str(n)
        + ", K="
        + str(k),
    )

    golden_ptr.free()


def test_matmul(m: Int, n: Int, k: Int):
    var c_ptr = DTypePointer[c_type].alloc(m * n, alignment=alignment)
    var a_ptr = DTypePointer[a_type].alloc(m * k, alignment=alignment)
    var b_ptr = DTypePointer[b_type].alloc(k * n, alignment=alignment)

    var c = NDBuffer[c_type, 2](c_ptr, Index(m, n))
    var a = NDBuffer[a_type, 2](a_ptr, Index(m, k))
    var b = NDBuffer[b_type, 2](b_ptr, Index(k, n))

    test_matmul(c, a, b, m, n, k)

    c_ptr.free()
    b_ptr.free()
    a_ptr.free()


def test_shapes():
    test_matmul(256, 1024, 4096)
    test_matmul(4, 5, 6)
    test_matmul(15, 16, 17)
    test_matmul(24, 32, 64)
    test_matmul(61, 73, 79)
    test_matmul(123, 456, 321)
    test_matmul(256, 256, 256)
    test_matmul(2, 65, 1200)


def main():
    @parameter
    if not os_is_macos():
        return

    test_shapes()
