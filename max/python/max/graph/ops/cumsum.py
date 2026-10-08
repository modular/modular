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
"""Op implementation for cumsum."""

from max._core.dialects import builtin, kgen, rmo

from ..graph import Graph
from ..value import TensorValue, TensorValueLike


def cumsum(
    x: TensorValueLike,
    axis: int = -1,
    exclusive: bool = False,
    reverse: bool = False,
) -> TensorValue:
    """Computes the cumulative sum of the input tensor along the given axis.

    Args:
        x: The input tensor to sum over.
        axis: The axis along which to compute the sum. If negative,
            indexes from the last dimension. For example, a value of -1 will
            compute the sum along the last dimension.
        exclusive: If set, start at 0 and exclude the final element.
            Otherwise, start with the first element. Said another way, cumsum
            computes `[sum(x[..., :i, ...]) for i in range(x.shape[axis])]`.
            If exclusive is set, the bounds are instead `range(1, x.shape[axis])`.
        reverse: If set, start from the end. In other words, the first element
            will be the total sum, with each element following counting
            downwards; or `[sum(x[..., i:, ...]) for i in range(x.shape[axis])]`.

    Returns:
        A symbolic tensor representing the result of the cumsum operation.
        The tensor will have the same type as the input tensor. The computed
        values will be the cumulative sum of the values along the given axis,
        according to the specified parameters:

        - if `exclusive` is set, the first value will be 0, and the last
          value will be excluded from the sum
        - if `reverse` is set, the sum will be computed starting at the
          back of the axis back to the front, rather than front-to-back

    Raises:
        ValueError: If ``axis`` is out of range for the rank of ``x``.
    """
    x = TensorValue(x)

    if axis < 0:
        axis += x.rank
    if not 0 <= axis < x.rank:
        raise ValueError(f"Invalid {axis=} for input {x.rank=}")

    index_type = builtin.IndexType()
    return Graph.current._add_op_generated(
        rmo.MoCumsumOp,
        result=x.type,
        input=x,
        axis=builtin.IntegerAttr(index_type, axis),
        exclusive=builtin.IntegerAttr(index_type, int(exclusive)),
        reverse=builtin.IntegerAttr(index_type, int(reverse)),
        output_param_decls=kgen.ParamDeclArrayAttr([]),
    )[0].tensor
