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
"""Op implementation for concat."""

from collections.abc import Iterable

from max import mlir
from max.mlir.dialects import rmo

from ..graph import Graph
from ..value import TensorValue, TensorValueLike
from .validation import assert_same_device


def concat(
    original_vals: Iterable[TensorValueLike],
    axis: int = 0,
) -> TensorValue:
    """Concatenates a list of symbolic tensors along an axis.

    Joins multiple tensors along a specified dimension. This operation requires
    the functional API since it operates on multiple tensors. All input tensors
    must have the same rank and the same size in all dimensions except the
    concatenation axis.

    .. code-block:: python

        import max.experimental.functional as F
        from max.experimental.tensor import Tensor

        # Create two 2x2 matrices
        a = Tensor.constant([[1, 2], [3, 4]])
        b = Tensor.constant([[5, 6], [7, 8]])

        # Concatenate along axis 0 (rows) - stacks vertically
        vertical = F.concat([a, b], axis=0)
        print(f"Concatenated along axis 0: {vertical.shape}")
        # Output: Concatenated along axis 0: [Dim(4), Dim(2)]
        print(vertical)
        # [[1, 2],
        #  [3, 4],
        #  [5, 6],
        #  [7, 8]]

        # Concatenate along axis 1 (columns) - joins horizontally
        horizontal = F.concat([a, b], axis=1)
        print(f"Concatenated along axis 1: {horizontal.shape}")
        # Output: Concatenated along axis 1: [Dim(2), Dim(4)]
        print(horizontal)
        # [[1, 2, 5, 6],
        #  [3, 4, 7, 8]]

    Args:
        original_vals: The list of symbolic tensor values to concatenate. Each tensor must have the same
            dtype and rank, and must have the same dimension size for each
            dimension other than ``axis``.
        axis: The axis to concatenate along. If negative, indexes relative
            to the end of the tensor shape. For instance, ``concat(vs, -1)``
            will concatenate along the last dimension.

    Returns:
        A new symbolic tensor representing the concatenation result. It will
        have the same rank as each input tensor, and its dimensions will be the same
        as each input tensor's for each dimension other than `axis`, which will
        have size equal to the sum of all tensor's size for that dimension.
    """
    vals = [TensorValue(v) for v in original_vals]

    if not vals:
        raise ValueError("Must provide at least one value to concat.")
    if not all(val.rank == vals[0].rank for val in vals):
        raise ValueError(f"Concat inputs must all have the same rank. {vals=}")
    if not -vals[0].rank <= axis < vals[0].rank:
        raise IndexError(f"Axis out of range {axis=}, {vals=}")
    for i, dim in enumerate(vals[0].shape):
        if i in (axis, axis + vals[0].rank):
            continue
        if not all(val.shape[i] == dim for val in vals):
            raise ValueError(
                f"Concat inputs differ on non-concat axis {i}: {vals=}"
            )
    assert_same_device(*vals)

    axis_attr = mlir.IntegerAttr.get(mlir.IndexType.get(), axis)

    result = Graph.current._add_op(rmo.concat, vals, axis=axis_attr)[0].tensor

    return result
