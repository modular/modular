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
# RUN: %mojo -debug-level full %s

from sys.info import is_apple_silicon, sizeof

from AppleAMX import *
from memory.buffer import NDBuffer
from testing import *

from utils.index import StaticIntTuple
from utils.list import DimList


fn fill_a(buf: NDBuffer[DType.float32, _, _]):
    # Fills the A matrix with the following values row + 2*col
    var rows = 16
    var cols = 16
    for i in range(rows):
        for j in range(cols):
            buf[(i, j)] = Float32(i // (j + 1) + j)


fn fill_b(buf: NDBuffer[DType.float32, _, _]):
    # Fills the A matrix with the following values row/(col + 1) + col
    var rows = 16
    var cols = 16
    for i in range(rows):
        for j in range(cols):
            buf[(i, j)] = Float32(i // (j + 1) + j)


fn clear_c(buf: NDBuffer[DType.float32, _, _]):
    buf.zero()


def test_dot_at_b():
    var a_matrix = NDBuffer[
        DType.float32,
        2,
        DimList(16, 16),
    ].stack_allocation()
    var b_matrix = NDBuffer[
        DType.float32,
        2,
        DimList(16, 16),
    ].stack_allocation()
    var c_matrix = NDBuffer[
        DType.float32,
        2,
        DimList(16, 16),
    ].stack_allocation()

    fill_a(a_matrix)
    fill_b(b_matrix)
    clear_c(c_matrix)

    dot_at_b(c_matrix, a_matrix, b_matrix)

    for m in range(c_matrix.dim[0]()):
        for n in range(c_matrix.dim[1]()):
            var golden = Float32(0)
            for k in range(a_matrix.dim[1]()):
                golden += a_matrix[k, m] * b_matrix[k, n]
            assert_almost_equal(
                c_matrix[m, n],
                golden,
                msg="invalid value at m=" + str(m) + ",n=" + str(n),
            )


def main():
    @parameter
    if is_apple_silicon():
        test_dot_at_b()
