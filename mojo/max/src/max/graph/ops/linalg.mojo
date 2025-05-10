# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Ops that perform linear algebra."""

from collections import Optional

from builtin._location import __call_location, _SourceLocation
from max.graph import Symbol
from max.tensor import Tensor, TensorShape

from ..error import error
from ..type import Dim, TensorType
from .casting import reshape


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


@always_inline
def matmul(
    lhs: Symbol,
    rhs: Symbol,
    location: Optional[_SourceLocation] = None,
) -> Symbol:
    """Computes the matrix multiplication of two symbolic tensors.

    Performs general matrix multiplication with broadcasting.

    If the lhs is 1d, it will be reshaped to `1xD`.
    If the rhs is 1d, it will be reshaped to `Dx1`.
    In both cases, the addition `1` dimensions will be removed from the output shape.

    For the multiplication, the innermost (rightmost) 2 dimensions are treated as a maxtrix.
    The lhs matrix will have the shape `MxK`.
    The rhs matrix will have the shape `KxN`.
    The output will have the shape `MxN`
    The `K` dimensions must be equivalent in both matrices.

    The remaining outer dimensions will be broadcasted.

    Args:
        lhs: The left-hand-side of the matmul.
        rhs: The right-hand-side of the matmul.
        location: An optional location for a more specific error message.

    Returns:
        A symbolic tensor representing he result of broadcasting the two
        matricies together and then performing a matrix multiply
        along the innermost two dimension of each tensor.
    """

    var g = lhs.graph()
    try:
        return g.op("rmo.matmul", List(lhs, rhs))
    except e:
        raise error(g, e, location=location or __call_location())


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

    # We need input to be rank2 since the GPU kernel only supports this case right now
    var input_shape = input.shape()
    var input_rank2 = input.reshape(-1, input_shape[len(input_shape) - 1])
    var result = g.op(
        "mo.layer_norm",
        List[Symbol](
            input_rank2,
            gamma,
            beta,
            epsilon_constant,
        ),
        input_rank2.type(),
    )

    # reshape back to the old shape
    return reshape_like(result, input)


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
