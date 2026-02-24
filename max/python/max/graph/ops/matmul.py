# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
"""Op implementation for matmul."""

from max.mlir.dialects import rmo

from ..graph import Graph
from ..value import TensorValue, TensorValueLike
from .validation import assert_same_device


def matmul(lhs: TensorValueLike, rhs: TensorValueLike) -> TensorValue:
    """Computes the matrix multiplication of two tensor graph values.

    Performs general matrix multiplication with broadcasting. Matrix multiplication
    is fundamental to neural networks, used for linear transformations, attention
    mechanisms, and fully connected layers.

    .. code-block:: python

        from max.experimental.tensor import Tensor

        # Create two 2x2 matrices
        x = Tensor.constant([[1.0, 2.0], [3.0, 4.0]])  # Shape: (2, 2)
        w = Tensor.constant([[5.0, 6.0], [7.0, 8.0]])  # Shape: (2, 2)

        # Matrix multiply using @ operator (uses matmul internally)
        result = x @ w
        print("Matrix multiplication result:")
        print(result)
        # Output: [[19.0, 22.0],
        #          [43.0, 50.0]]
        # Computed as: result[i,j] = sum(x[i,k] * w[k,j])

        # Can also call directly via functional API
        import max.experimental.functional as F
        result2 = F.matmul(x, w)
        # Same result as x @ w

    If the lhs is 1D, it will be reshaped to ``1xD``.
    If the rhs is 1D, it will be reshaped to ``Dx1``.
    In both cases, the additional `1` dimensions will be removed from the
    output shape.

    For the multiplication, the innermost (rightmost) 2 dimensions are treated
    as a matrix.
    The lhs matrix will have the shape ``MxK``.
    The rhs matrix will have the shape ``KxN``.
    The output will have the shape `MxN`
    The ``K`` dimensions must be equivalent in both matrices.

    The remaining outer dimensions will be broadcasted.

    Args:
        lhs: The left-hand side input tensor.
        rhs: The right-hand side input tensor.
        location: An optional location for a more specific error message.

    Returns:
        A tensor graph value representing the matrix product of ``lhs`` and ``rhs``.
        For 2D inputs, the output shape is ``(M, N)`` where ``lhs`` is ``(M, K)``
        and ``rhs`` is ``(K, N)``. For higher-dimensional inputs, batch
        dimensions are preserved and the operation is applied to the last two
        dimensions of each input.
    """
    lhs = TensorValue(lhs)
    rhs = TensorValue(rhs)
    assert_same_device(lhs=lhs, rhs=rhs)
    return Graph.current._add_op(rmo.matmul, lhs, rhs)[0].tensor
