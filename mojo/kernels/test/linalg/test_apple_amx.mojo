# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# This file only tests the Apple AMX matmul functionality which is defined as a
# A^T.B where A and B are 16x16 Float32 matrices.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from sys.info import is_apple_silicon, sizeof

from buffer import NDBuffer
from buffer.dimlist import DimList
from linalg.apple_amx_intrinsics import *
from testing import *

from utils.index import IndexList


fn fill_a(buf: NDBuffer[mut=True, *_]):
    # Fills the A matrix with the following values row + 2*col
    for i in range(buf.dim[0]()):
        for j in range(buf.dim[1]()):
            buf[i, j] = Scalar[buf.type](i // (j + 1) + j)


fn fill_b(buf: NDBuffer[mut=True, *_]):
    # Fills the A matrix with the following values row/(col + 1) + col
    for i in range(buf.dim[0]()):
        for j in range(buf.dim[1]()):
            buf[i, j] = Scalar[buf.type](i // (j + 1) + j)


fn clear_c(buf: NDBuffer):
    buf.zero()


def test_dot_at_b[type: DType, shape: Tuple[Int, Int]]():
    var a_matrix = NDBuffer[
        type, 2, MutableAnyOrigin, shape=shape
    ].stack_allocation()
    var b_matrix = NDBuffer[
        type, 2, MutableAnyOrigin, shape=shape
    ].stack_allocation()
    var c_matrix = NDBuffer[
        type, 2, MutableAnyOrigin, shape=shape
    ].stack_allocation()

    fill_a(a_matrix)
    fill_b(b_matrix)
    clear_c(c_matrix)

    dot_at_b(c_matrix, a_matrix, b_matrix)

    for m in range(c_matrix.dim[0]()):
        for n in range(c_matrix.dim[1]()):
            var golden = Scalar[type](0)
            for k in range(a_matrix.dim[1]()):
                golden += a_matrix[k, m] * b_matrix[k, n]
            assert_almost_equal(
                c_matrix[m, n],
                golden,
                msg=String("invalid value at m=", m, ",n=", n),
            )


def main():
    @parameter
    if is_apple_silicon():
        test_dot_at_b[DType.float32, (16, 16)]()
        test_dot_at_b[DType.float16, (32, 32)]()
