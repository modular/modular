# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from sys.info import os_is_macos

from buffer import NDBuffer
from buffer.dimlist import DimList
from linalg.apple_accelerate import apple_batched_matmul, apple_matmul
from memory import UnsafePointer
from testing import *

from utils.index import Index, IndexList

alias alignment = 64


alias a_type = DType.float32
alias b_type = DType.float32
alias c_type = DType.float32


fn gemm_naive(
    c: NDBuffer[mut=True, *_],
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
                c[i, j] += a_val * b_val


def test_matmul(
    c: NDBuffer[mut=True, *_],
    a: NDBuffer[mut=True, *_],
    b: NDBuffer[mut=True, *_],
    m: Int,
    n: Int,
    k: Int,
):
    @parameter
    if not os_is_macos():
        return

    var golden_ptr = UnsafePointer[Scalar[c.type], alignment=alignment].alloc(
        m * n
    )
    var golden = NDBuffer[c.type, 2](golden_ptr, Index(m, n))

    for i in range(m):
        for j in range(k):
            a[i, j] = (i + j) * Scalar[a.type](0.001)

    for i in range(k):
        for j in range(n):
            b[i, j] = (i + k) * Scalar[b.type](0.001)

    for i in range(m):
        for j in range(n):
            c[i, j] = 0
            golden[i, j] = 0

    apple_matmul(c, a, b)
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
        String(
            "num of errors must be 0, but got ",
            errors,
            " for dimensions M=",
            m,
            ", N=",
            n,
            ", K=",
            k,
        ),
    )

    golden_ptr.free()


def test_matmul(m: Int, n: Int, k: Int):
    var c_ptr = UnsafePointer[Scalar[c_type], alignment=alignment].alloc(m * n)
    var a_ptr = UnsafePointer[Scalar[a_type], alignment=alignment].alloc(m * k)
    var b_ptr = UnsafePointer[Scalar[b_type], alignment=alignment].alloc(k * n)

    var c = NDBuffer[c_type, 2](c_ptr, Index(m, n))
    var a = NDBuffer[a_type, 2](a_ptr, Index(m, k))
    var b = NDBuffer[b_type, 2](b_ptr, Index(k, n))

    test_matmul(c, a, b, m, n, k)

    c_ptr.free()
    b_ptr.free()
    a_ptr.free()


def test_matmul():
    test_matmul(256, 1024, 4096)
    test_matmul(4, 5, 6)
    test_matmul(15, 16, 17)
    test_matmul(24, 32, 64)
    test_matmul(61, 73, 79)
    test_matmul(123, 456, 321)
    test_matmul(256, 256, 256)
    test_matmul(2, 65, 1200)


fn bmm_naive(
    c: NDBuffer[mut=True, *_],
    a: NDBuffer,
    b: NDBuffer,
    batches: Int,
    m: Int,
    n: Int,
    k: Int,
):
    for batch in range(batches):
        for i in range(m):
            for p in range(k):
                for j in range(n):
                    var a_val = a[batch, i, p].cast[c.type]()
                    var b_val = b[batch, p, j].cast[c.type]()
                    c[batch, i, j] += a_val * b_val


def test_batched_matmul(
    c: NDBuffer[mut=True, *_],
    a: NDBuffer[mut=True, *_],
    b: NDBuffer[mut=True, *_],
    batches: Int,
    m: Int,
    n: Int,
    k: Int,
):
    @parameter
    if not os_is_macos():
        return

    var golden_ptr = UnsafePointer[Scalar[c.type], alignment=alignment].alloc(
        batches * m * n
    )
    var golden = NDBuffer[c.type, 3](golden_ptr, Index(batches, m, n))

    for batch in range(batches):
        for i in range(m):
            for j in range(k):
                a[batch, i, j] = (i + j) * Scalar[a.type](0.001)

    for batch in range(batches):
        for i in range(k):
            for j in range(n):
                b[batch, i, j] = (i + k) * Scalar[b.type](0.001)

    for batch in range(batches):
        for i in range(m):
            for j in range(n):
                c[batch, i, j] = 0
                golden[batch, i, j] = 0

    apple_batched_matmul(c, a, b)
    bmm_naive(golden, a, b, batches, m, n, k)

    var errors: Int = 0
    for batch in range(batches):
        for i in range(m):
            for j in range(n):
                if c[batch, i, j] != golden[batch, i, j]:
                    if errors < 10:
                        print(
                            c[batch, i, j] - golden[batch, i, j],
                            "at",
                            batch,
                            i,
                            j,
                        )
                    errors += 1

    assert_true(
        errors == 0,
        String(
            "num of errors must be 0, but got ",
            errors,
            " for dimensions Batch=",
            batches,
            " M=",
            m,
            ", N=",
            n,
            ", K=",
            k,
        ),
    )

    golden_ptr.free()


def test_batched_matmul(batch: Int, m: Int, n: Int, k: Int):
    var c_ptr = UnsafePointer[Scalar[c_type], alignment=alignment].alloc(
        batch * m * n
    )
    var a_ptr = UnsafePointer[Scalar[a_type], alignment=alignment].alloc(
        batch * m * k
    )
    var b_ptr = UnsafePointer[Scalar[b_type], alignment=alignment].alloc(
        batch * k * n
    )

    var c = NDBuffer[c_type, 3](c_ptr, Index(batch, m, n))
    var a = NDBuffer[a_type, 3](a_ptr, Index(batch, m, k))
    var b = NDBuffer[b_type, 3](b_ptr, Index(batch, k, n))

    test_batched_matmul(c, a, b, batch, m, n, k)

    c_ptr.free()
    b_ptr.free()
    a_ptr.free()


def test_batched_matmul():
    for batch_ref in List[Int](1, 2, 4, 9, 12):
        var batch = batch_ref[]
        test_batched_matmul(batch, 256, 1024, 4096)
        test_batched_matmul(batch, 4, 5, 6)
        test_batched_matmul(batch, 15, 16, 17)
        test_batched_matmul(batch, 24, 32, 64)
        test_batched_matmul(batch, 61, 73, 79)
        test_batched_matmul(batch, 123, 456, 321)
        test_batched_matmul(batch, 256, 256, 256)
        test_batched_matmul(batch, 2, 65, 1200)


def main():
    @parameter
    if not os_is_macos():
        return

    test_matmul()
    test_batched_matmul()
