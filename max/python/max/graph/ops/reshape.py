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
"""Op implementation for reshape."""

import operator
from collections.abc import Iterable
from functools import reduce

from max.mlir.dialects import rmo

from ..dim import Dim, StaticDim
from ..graph import Graph
from ..type import Shape, ShapeLike
from ..value import TensorValue, TensorValueLike


def _dim_prod(dims: Iterable[Dim]) -> Dim:
    # 1 is the multiplicative identity.
    return reduce(operator.mul, dims, Dim(1))


def reshape(x: TensorValueLike, shape: ShapeLike) -> TensorValue:
    """Reshapes a symbolic tensor.

    The number and order of the elements in the tensor is unchanged.
    In other words, if you were to iterate over elements in the tensor
    by major dimension to minor dimension, the iteration order would stay
    the same.

    If a value of -1 is present in the shape, that dimension becomes
    an automatically calculated dimension collecting all unspecified dimensions.
    Its length becomes the number of elements in the original tensor
    divided by the product of elements of the reshape.

    Args:
        x: The input symbolic tensor to reshape.
        shape: The new shape as a list of dimensions.
               A single dimension may be `-1`.

    Returns:
        A symbolic tensor with the same elements as the original tensor, but
        in a new shape. Its symbolic shape is the same as :code:`shape`.

    Raises:
        ValueError: if input and target shapes' number of elements mismatch.
    """
    v = TensorValue(x)
    shape = Shape(shape)

    # Find the single -1 dimension (if any).
    neg_indices = [
        i
        for i, d in enumerate(shape)
        if isinstance(d, StaticDim) and int(d) == -1
    ]
    if len(neg_indices) > 1:
        raise ValueError("reshape(): at most one -1 dimension is allowed")

    neg_idx: int | None = neg_indices[0] if neg_indices else None

    if neg_idx is not None:
        # Disallow inferring -1 if another requested dim is 0.
        has_zero_elsewhere = any(
            i != neg_idx and isinstance(d, StaticDim) and int(d) == 0
            for i, d in enumerate(shape)
        )
        if has_zero_elsewhere:
            raise ValueError(
                "reshape(): cannot infer -1 dimension when another dimension is 0"
            )

        # total = product(input dims)
        total = _dim_prod(v.shape)

        # known = product(all requested dims except the -1)
        known = _dim_prod(d for i, d in enumerate(shape) if i != neg_idx)

        # missing = total // known  (symbolic; folds when possible)
        shape[neg_idx] = total // known

    return Graph.current._add_op(
        rmo.reshape,
        v,
        new_shape=shape.to_mlir(),
    )[0].tensor
