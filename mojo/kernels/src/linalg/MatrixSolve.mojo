# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""The module implements Matrix Solve functions."""

from Assert import assert_param
from Buffer import NDBuffer
from DType import DType
from Index import Index
from List import DimList
from Range import range
from Index import StaticIntTuple


fn matrix_solve_tiny[
    type: DType, M: Int, N: Int, K: Int
](
    X: NDBuffer[2, DimList(K, N), type],
    A: NDBuffer[2, DimList(M, K), type],
    B: NDBuffer[2, DimList(M, N), type],
):
    """Solve A*X = B, where A is a 3x3 matrix, B and X are 3x2."""
    assert_param[M == 3]()
    assert_param[N == 2]()
    assert_param[K == 3]()

    # A matrix
    let A00 = A[0, 0]
    let A01 = A[0, 1]
    let A02 = A[0, 2]
    let A10 = A[1, 0]
    let A11 = A[1, 1]
    let A12 = A[1, 2]
    let A20 = A[2, 0]
    let A21 = A[2, 1]
    let A22 = A[2, 2]

    # A inverse
    # fmt: off
    let det_A = A02 * A11 * A20 - A01 * A12 * A20 - A02 * A10 * A21 \
              + A00 * A12 * A21 + A01 * A10 * A22 - A00 * A11 * A22
    # fmt: on
    let rdet_A = 1.0 / det_A
    let A_inv00 = A12 * A21 - A11 * A22
    let A_inv01 = A01 * A22 - A02 * A21
    let A_inv02 = A02 * A11 - A01 * A12
    let A_inv10 = A10 * A22 - A12 * A20
    let A_inv11 = A02 * A20 - A00 * A22
    let A_inv12 = A00 * A12 - A02 * A10
    let A_inv20 = A11 * A20 - A10 * A21
    let A_inv21 = A00 * A21 - A01 * A20
    let A_inv22 = A01 * A10 - A00 * A11

    # Rows in B.
    let B0 = B.simd_load[N](Index(0, 0))
    let B1 = B.simd_load[N](Index(1, 0))
    let B2 = B.simd_load[N](Index(2, 0))

    # Update solution.
    X.simd_store[N](
        Index(0, 0), rdet_A * B2.fma(A_inv02, B1.fma(A_inv01, A_inv00 * B0))
    )
    X.simd_store[N](
        Index(1, 0), rdet_A * B2.fma(A_inv12, B1.fma(A_inv11, A_inv10 * B0))
    )
    X.simd_store[N](
        Index(2, 0), rdet_A * B2.fma(A_inv22, B1.fma(A_inv21, A_inv20 * B0))
    )


fn batch_matrix_solve[
    type: DType, x_rank: Int, a_rank: Int, b_rank: Int
](
    x: NDBuffer[x_rank, DimList.create_unknown[x_rank](), type],
    a: NDBuffer[a_rank, DimList.create_unknown[a_rank](), type],
    b: NDBuffer[b_rank, DimList.create_unknown[b_rank](), type],
):
    """
    A specialized matrix solver for batch_sizex3x3 matrix LHS
    and batch_sizex3x2 RHS.
    """
    assert_param[a_rank == b_rank]()
    assert_param[a_rank == x_rank]()

    alias row_dim = a_rank - 2
    alias col_dim = a_rank - 1

    var batch_size = 1
    for i in range(row_dim):
        batch_size *= a.dim(i)

    for batch in range(batch_size):
        # Get a 2D view of the Tensor.
        let x_view = NDBuffer[2, DimList(3, 2), type](
            x.data.offset(batch * 3 * 2),
            StaticIntTuple[2](3, 2),
            type,
        )
        let a_view = NDBuffer[2, DimList(3, 3), type](
            a.data.offset(batch * 3 * 3),
            StaticIntTuple[2](3, 3),
            type,
        )
        let b_view = NDBuffer[2, DimList(3, 2), type](
            b.data.offset(batch * 3 * 2),
            StaticIntTuple[2](3, 2),
            type,
        )
        matrix_solve_tiny[type, 3, 2, 3](x_view, a_view, b_view)
