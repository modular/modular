# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""The module implements Matrix Solve functions."""

from memory.buffer import NDBuffer
from runtime.tracing import TraceLevel, Trace

from utils.index import StaticIntTuple
from utils.list import DimList


@always_inline
fn matrix_solve_tiny[
    type: DType, M: Int, N: Int, K: Int
](
    X: NDBuffer[2, DimList(K, N), type],
    A: NDBuffer[2, DimList(M, K), type],
    B: NDBuffer[2, DimList(M, N), type],
):
    """Solve A*X = B, where A is a 3x3 matrix, B and X are 3x2."""
    constrained[M == 3]()
    constrained[N == 2]()
    constrained[K == 3]()

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
    let B0 = B.simd_load[N]((0, 0))
    let B1 = B.simd_load[N]((1, 0))
    let B2 = B.simd_load[N]((2, 0))

    # Update solution.
    X.simd_store[N](
        (0, 0), rdet_A * B2.fma(A_inv02, B1.fma(A_inv01, A_inv00 * B0))
    )
    X.simd_store[N](
        (1, 0), rdet_A * B2.fma(A_inv12, B1.fma(A_inv11, A_inv10 * B0))
    )
    X.simd_store[N](
        (2, 0), rdet_A * B2.fma(A_inv22, B1.fma(A_inv21, A_inv20 * B0))
    )


@always_inline
fn matrix_solve[
    type: DType,
    x_rank: Int,
    a_rank: Int,
    b_rank: Int,
    single_thread_blocking_override: Bool,
](
    a: NDBuffer[a_rank, DimList.create_unknown[a_rank](), type],
    b: NDBuffer[b_rank, DimList.create_unknown[b_rank](), type],
    x: NDBuffer[x_rank, DimList.create_unknown[x_rank](), type],
) raises:
    """
    A specialized matrix solver for batch_sizex3x3 matrix LHS
    and batch_sizex3x2 RHS.
    """
    constrained[a_rank == b_rank]()
    constrained[a_rank == x_rank]()

    @parameter
    if not type.is_floating_point():
        raise Error("Only floating point types are supported.")

    @parameter
    if a_rank > 2:
        if a.dim(0) != b.dim(0) or b.dim(0) != x.dim(0):
            raise Error("input and output batch sizes must match")

    alias row_dim = a_rank - 2
    alias col_dim = a_rank - 1

    if a.dim(row_dim) != 3 or a.dim(col_dim) != 3:
        raise Error("The a matrix's shape must be (3,3)")

    if x.dim(row_dim) != 3 or x.dim(col_dim) != 2:
        raise Error("The x matrix's shape must be (3,2)")
    if b.dim(row_dim) != 3 or b.dim(col_dim) != 2:
        raise Error("The b matrix's shape must be (3,2)")

    with Trace[TraceLevel.OP]("mojo.matrix_solve") as t:
        var batch_size = 1
        for i in range(row_dim):
            batch_size *= a.dim(i)

        for batch in range(batch_size):
            # Get a 2D view of the Tensor.
            let x_view = NDBuffer[2, DimList(3, 2), type](
                x.data.offset(batch * 3 * 2), (3, 2)
            )
            let a_view = NDBuffer[2, DimList(3, 3), type](
                a.data.offset(batch * 3 * 3), (3, 3)
            )
            let b_view = NDBuffer[2, DimList(3, 2), type](
                b.data.offset(batch * 3 * 2), (3, 2)
            )
            matrix_solve_tiny[type, 3, 2, 3](x_view, a_view, b_view)


@always_inline
fn matrix_solve_shape[
    rank: Int,
    type: DType,
    single_thread_blocking_override: Bool,
](
    a_buff: NDBuffer[rank, DimList.create_unknown[rank](), type],
    b_buff: NDBuffer[rank, DimList.create_unknown[rank](), type],
) -> StaticIntTuple[rank]:
    """
    Compute the output shape of a matrix solve operation (i.e., given A and B
    from AX = B, compute X), and assert the inputs are compatible.

    Parameters:
        rank: Rank of the input and output tensors.
        type: Type of the input and output tensors.
        single_thread_blocking_override: If True, then the operation is run
          synchronously using a single thread.

    Args:
        a_buff: The input tensor A.
        b_buff: The input tensor B.

    Returns:
        The output shape.
    """

    # TODO(#17512)
    debug_assert(rank >= 2, "matrix-solve requires rank >= 2")

    # TODO(#17512)
    debug_assert(
        a_buff.dim(rank - 1) == a_buff.dim(rank - 2),
        "matrix-solve requires first input to be a (batch of) square matrix",
    )

    # TODO(#17512)
    debug_assert(
        a_buff.dim(rank - 2) == b_buff.dim(rank - 2),
        "matrix-solve input outter dimensions must match",
    )

    @unroll
    for i in range(rank - 2):
        # TODO(#17512)
        debug_assert(
            a_buff.dim(i) == b_buff.dim(i),
            "matrix-solve batch dimensions must match",
        )

    return b_buff.get_shape()
