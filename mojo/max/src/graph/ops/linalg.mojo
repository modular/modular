# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Ops that perform linear algebra."""

from .casting import reshape
from ..error import error
from ..type import Dim, TensorType
from max.tensor import Tensor, TensorShape


def outer(lhs: Symbol, rhs: Symbol) -> Symbol:
    """Computes the outer product of two symbolic vectors.

    Args:
        lhs: The left side of the product. Whatever its shape,
            it will be flattened to a rank-1 vector.
        rhs: The right side of the product. Whatever its shape,
            it will be flattened to a rank-1 vector. Must have the
            same number of elements as `lhs`.

    Returns:
        A symbolic tensor representing the
        [outer product](https://en.wikipedia.org/wiki/Outer_product)
        of the two input vectors. It will have rank 2, with the dimension
        sizes being the number of elements of `lhs` and `rhs` respectively.
    """
    return lhs.reshape(-1, 1) * rhs.reshape(1, -1)


def matmul_broadcast(lhs: Symbol, rhs: Symbol) -> List[Symbol]:
    """Computes the broadcasting of two symbolic tensors for a matmul.

    Args:
        lhs: The left side of the matmul.
        rhs: The right side of the matmul.

    Returns:
        A pair of symbolic tensors corresponding to the `lhs` and `rhs`
        respectively, after being broadcast to the right shapes to perform
        a matmul between them. All but the final two dimensions are broadcasted.
    """
    var g = lhs.graph()
    var lhs_type = lhs.tensor_type()
    var rhs_type = rhs.tensor_type()

    var lhs_rank = lhs_type.rank()
    var rhs_rank = rhs_type.rank()

    var broadcast_rank = max(lhs_rank, rhs_rank)
    var lhs_shape = shape_of(lhs)
    var rhs_shape = shape_of(rhs)

    var lhs_broadcast_dims = lhs_shape[: lhs_rank - 2]
    var lhs_matrix_dims = lhs_shape[lhs_rank - 2 : lhs_rank]

    var rhs_broadcast_dims = rhs_shape[: rhs_rank - 2]
    var rhs_matrix_dims = rhs_shape[rhs_rank - 2 : rhs_rank]

    var broadcast_dims_shape = g.op(
        "rmo.mo.broadcast_shape",
        List[Symbol](lhs_broadcast_dims, rhs_broadcast_dims),
        TensorType(DType.int64, broadcast_rank - 2),
    )

    var lhs_final_dims = List[Dim]()
    var rhs_final_dims = List[Dim]()
    for _ in range(broadcast_rank - 2):
        lhs_final_dims.append(Dim.dynamic())
        rhs_final_dims.append(Dim.dynamic())
    lhs_final_dims.append(lhs_type.dim(-2))
    lhs_final_dims.append(lhs_type.dim(-1))
    rhs_final_dims.append(rhs_type.dim(-2))
    rhs_final_dims.append(rhs_type.dim(-1))

    var lhs_broadcast_shape = concat(
        List[Symbol](broadcast_dims_shape, lhs_matrix_dims)
    )

    var broadcast_lhs = g.op(
        "rmo.mo.broadcast_to",
        List[Symbol](lhs, lhs_broadcast_shape),
        TensorType(lhs_type.dtype, lhs_final_dims),
    )

    var rhs_broadcast_shape = concat(
        List[Symbol](broadcast_dims_shape, rhs_matrix_dims)
    )

    var broadcast_rhs = g.op(
        "rmo.mo.broadcast_to",
        List[Symbol](rhs, rhs_broadcast_shape),
        TensorType(rhs_type.dtype, rhs_final_dims),
    )

    return List[Symbol](broadcast_lhs, broadcast_rhs)


def matmul(lhs: Symbol, rhs: Symbol) -> Symbol:
    """Computes the matrix multiplication of two symbolic tensors.

    The last two dimensions of each tensor are treated as matricies and multiplied,
    and the remaining dimensions are broadcast dimensions. If `rhs` is rank 2,
    this delegates to `matmul_by_matrix()` for better performance.

    Args:
        lhs: The left-hand-side of the matmul.
        rhs: The right-hand-side of the matmul.

    Returns:
        A symbolic tensor representing he result of broadcasting the two
        matricies together according to `matmul_broadcast` and then performing
        a matrix multiply along the last two dimension of each tensor.
    """
    var g = lhs.graph()
    var lhs_type = lhs.tensor_type()
    var rhs_type = rhs.tensor_type()
    var lk = lhs_type.dims[-1]
    var rk = rhs_type.dims[-2]
    if rhs_type.rank() < 2:
        raise error(
            g, "right hand side of matrix multiply must have rank at least 2"
        )
    if lk != rk and not (lk.is_dynamic() or rk.is_dynamic()):
        raise error(
            g,
            str("matrix multiply K dimensions don't match: ")
            + str(lk)
            + " != "
            + str(rk),
        )
    if rhs_type.rank() > 2:
        return batch_matmul(lhs, rhs)
    else:
        return matmul_by_matrix(lhs, rhs)


def batch_matmul(lhs: Symbol, rhs: Symbol) -> Symbol:
    """Computes the matrix multiplication of two symbolic tensors.

    The last two dimensions of each tensor are treated as matricies and
    multiplied, and the remaining dimensions are broadcast dimensions.

    This supports arbitrary-rank `rhs` inputs, but may be less performant than
    `matmul_by_matrix` if `rhs` is rank 2.

    Args:
        lhs: The left-hand-side of the matmul.
        rhs: The right-hand-side of the matmul.

    Returns:
        A symbolic tensor representing he result of broadcasting the two
        matricies together according to `matmul_broadcast` and then performing
        a matrix multiply along the last two dimension of each tensor.
    """
    var g = lhs.graph()
    var broadcast_pair = matmul_broadcast(lhs, rhs)
    var broadcast_lhs = broadcast_pair[0]
    var broadcast_rhs = broadcast_pair[1]

    var lhs_type = broadcast_lhs.tensor_type()
    var rhs_type = broadcast_rhs.tensor_type()
    var dims = List[Dim]()
    for i in range(lhs_type.rank() - 1):
        dims.append(lhs_type.dims[i])
    dims.append(rhs_type.dim(-1))
    var out_type = TensorType(lhs_type.dtype, dims)

    return g.op(
        "rmo.mo.batch_matmul",
        List[Symbol](broadcast_lhs, broadcast_rhs),
        out_type,
    )


def matmul_by_matrix(lhs: Symbol, rhs: Symbol) -> Symbol:
    """Computes the matrix multiplication of two symbolic tensors.

    The last two dimensions in `lhs` are treated as matricies and multiplied
    by `rhs` (which must be a 2D tensor). Any remaining dimensions in `lhs`
    are broadcast dimensions.

    Args:
        lhs: The left-hand-side of the matmul.
        rhs: The right-hand-side of the matmul.
             Must be rank 2 (a 2D tensor/matrix).

    Returns:
        A symbolic tensor representing he result of broadcasting the two
        matricies together according to `matmul_broadcast` and then performing
        a matrix multiply along the last two dimension of each tensor.
    """
    var g = lhs.graph()
    var lhs_type = lhs.tensor_type()
    var rhs_type = rhs.tensor_type()
    if rhs_type.rank() != 2:
        raise error(g, "rhs must be a matrix")

    var lhs_shape = shape_of(lhs)
    var rhs_shape = shape_of(rhs)
    last_lhs_axis = lhs_type.rank() - 1
    var reshape_shape = stack(
        List[Symbol](g.scalar(Int64(-1)), lhs_shape[last_lhs_axis])
    )
    var final_shape = concat(
        List[Symbol](lhs_shape[:last_lhs_axis], rhs_shape[1:2])
    )

    var final_dims = List[Dim]()
    for i in range(lhs_type.rank() - 1):
        final_dims.append(lhs_type.dim(i))
    final_dims.append(rhs_type.dim(-1))

    var matmul_dims = List[Dim]()
    matmul_dims.append(Dim.dynamic())
    matmul_dims.append(lhs_type.dim(-1))
    var matmul_out = g.op(
        "rmo.mo.matmul",
        List[Symbol](reshape(lhs, reshape_shape, matmul_dims), rhs),
        TensorType(lhs_type.dtype, Dim.dynamic(), rhs_type.dim(-1)),
    )

    return reshape(matmul_out, final_shape, final_dims)


def band_part(
    input: Symbol, num_lower: Symbol, num_upper: Symbol, exclude: Bool = False
) -> Symbol:
    """Masks out everything except a diagonal band of an input matrix.

    Copies a tensor setting everything outside the central diagonal band of the
    matricies to zero, where all but the last two axes are effectively batches,
    and the last two axes define sub matricies.

    Assumes the input has dimensions [I, J, ..., M, N], then the output tensor
    has the same shape as the input, and the values are given by

    ```
    out[i, j, ..., m, n] = in_band(m, n) * input[i, j,  ..., m, n].
    ```

    with the indicator function:

    ```
    in_band(m, n) = ((num_lower < 0 || (m - n) <= num_lower)) &&
                     (num_upper < 0 || (n - m) <= num_upper))
    ```

    Args:
        input: The input to mask out.
        num_lower: The number of diagonal bands to include below the central
            diagonal. If -1, include the entire lower triangle.
        num_upper: The number of diagonal bands to include above the central
            diagonal. If -1, include the entire upper triangle.
        exclude: If true, invert the selection of elements to mask. Elements
            in the band are set to zero.

    Returns:
        A symbolic tensor value with the configured selection masked out
        to 0 values, and the remaining values copied from the input tensor.
    """
    var g = input.graph()
    return g.op(
        "rmo.mo.linalg.band_part",
        List[Symbol](
            input,
            num_lower.reshape(),
            num_upper.reshape(),
            g.scalar[DType.bool](exclude),
        ),
        input.type(),
    )


def layer_norm[
    dtype: DType
](input: Symbol, gamma: Symbol, beta: Symbol, epsilon: Scalar[dtype]) -> Symbol:
    """Performs layer normalization.

    Args:
        input: The input tensor to normalize.
        gamma: The gamma parameter of the normalization.
        beta: The beta parameter of the normalization.
        epsilon: The epsilon parameter of the normalization.

    Returns:
        A symbolic tensor value with the normalization applied.
    """
    g = input.graph()
    epsilon_constant = g.constant(Tensor[dtype](TensorShape(1), epsilon))

    return g.op(
        "mo.layer_norm",
        List[Symbol](
            input,
            gamma,
            beta,
            epsilon_constant,
        ),
        input.type(),
    )


def range_fill(start: Symbol, limit: Symbol, step: Symbol) -> Symbol:
    """Creates a sequence of numbers. The sequence goes from `start` with
    increments of size `step` up to (but not including) `limit`. All arguments
    are mandatory and must have the same element type.

    Note the following restrictions on input values:
    1. `step` must be non-zero
    2. `limit - start` must be zero or have the same sign as `step`

    Args:
        start: The start of the range to generate.
        limit: The range will be generated up to, but not including, this value.
        step: The step size for the range.

    Returns:
        A symbolic tensor value containing the defined range of values.
    """
    g = limit.graph()

    return g.op(
        "rmo.mo.range",
        List[Symbol](
            start,
            limit,
            step,
        ),
        TensorType(limit.tensor_type().dtype, Dim.dynamic()),
    )


def tile(input: Symbol, repeats: List[Int64]) -> Symbol:
    """Returns a new Tensor as the result of copying the input tensor N_i times
    on each dimension, where N_i = tiles[i].

    The i-th dimension of output shape will be the ith dimension of input shape
    multiplied by N_i.

    Args:
        input: The input tensor to tile.
        repeats: A list containing the number of repeats to perform across
                each dimension. The length of the list must be the same as the
                rank of the input tensor.

    Returns:
        A symbolic tensor value containing input repeated across the specified
        dimensions.
    """
    g = input.graph()
    output_dims = List[Dim]()
    input_type = input.tensor_type()
    if not len(repeats) == len(input_type.dims):
        raise error(
            g,
            "the rank of the input must match the dimensions to repeat across",
        )
    for i in range(len(repeats)):
        input_dimension = input_type.dims[i]
        if input_dimension.is_symbolic() and repeats[i] == 1:
            output_dims.append(input_dimension)
        elif input_dimension.is_dynamic() or input_dimension.is_symbolic():
            output_dims.append(Dim.dynamic())
        else:
            output_dims.append(
                Dim.static(repeats[i] * input_type.dims[i].num_elements())
            )

    return g.op(
        "rmo.mo.tile",
        List[Symbol](
            input,
            g.vector(repeats),
        ),
        TensorType(input.tensor_type().dtype, output_dims),
    )
