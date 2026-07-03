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
"""Reduction ops."""

from __future__ import annotations

from max._core import Operation
from max._core.dialects import builtin, kgen, rmo
from max.dtype import DType

from ..dim import Dim
from ..graph import Graph
from ..shape import Shape
from ..type import TensorType
from ..value import TensorValue, TensorValueLike


def sum(x: TensorValueLike, axis: int = -1) -> TensorValue:
    """Reduces a symbolic tensor using a sum operation.

    Args:
        x: The input tensor for the operation.
        axis: The axis along which to compute the reduction. If negative,
            indexes from the last dimension. For example, a value of ``-1`` will
            compute the reduction along the last dimension.

    Returns:
        A symbolic tensor representing the result of the sum operation.
        The tensor will have the same rank as the input tensor, and the same
        shape except along the ``axis`` dimension which will have size ``1``.
    """
    return _reduce(rmo.MoReduceAddOp, x, axis=axis)


def mean(x: TensorValueLike, axis: int = -1) -> TensorValue:
    """Reduces a symbolic tensor using a mean operation.

    Args:
        x: The input tensor for the operation.
        axis: The axis along which to compute the reduction. If negative,
            indexes from the last dimension. For example, a value of ``-1`` will
            compute the reduction along the last dimension.

    Returns:
        A symbolic tensor representing the result of the mean operation.
        The tensor will have the same rank as the input tensor, and the same
        shape except along the ``axis`` dimension which will have size ``1``.
    """
    return _reduce(rmo.MoReduceMeanOp, x, axis=axis)


def min(x: TensorValueLike, axis: int = -1) -> TensorValue:
    """Computes the minimum value along a specified axis.

    This operation is useful for finding the smallest values in data,
    implementing certain loss functions, or analyzing numerical ranges in
    tensors.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("min_example") as graph:
            x = ops.constant(
                [[1.2, 3.5, 2.1, 0.8], [2.3, 1.9, 4.2, 3.1]],
                DType.float32,
                device=device,
            )
            graph.output(
                ops.min(x, axis=-1),  # row_min, shape (2, 1)
                ops.min(x, axis=0),  # col_min, shape (1, 4)
            )

        model = InferenceSession().load(graph)
        row_min, col_min = model.execute()

    .. invisible-code-block: python

        import numpy as np

        assert np.allclose(row_min.to_numpy(), [[0.8], [1.9]])
        assert np.allclose(col_min.to_numpy(), [[1.2, 1.9, 2.1, 0.8]])

    Args:
        x: The input tensor for the operation.
        axis: The axis along which to compute the reduction. If negative,
            indexes from the last dimension. For example, a value of ``-1`` will
            compute the reduction along the last dimension.

    Returns:
        A symbolic tensor that has the same rank as the input tensor and the same
        shape except along the ``axis`` dimension which will have size ``1``.
    """
    return _reduce(rmo.MoReduceMinOp, x, axis=axis)


def max(x: TensorValueLike, axis: int = -1) -> TensorValue:
    """Reduces a symbolic tensor using a max operation.

    Args:
        x: The input tensor for the operation.
        axis: The axis along which to compute the reduction. If negative,
            indexes from the last dimension. For example, a value of ``-1`` will
            compute the reduction along the last dimension.

    Returns:
        A symbolic tensor representing the result of the max operation.
        The tensor will have the same rank as the input tensor, and the same
        shape except along the ``axis`` dimension which will have size ``1``.
    """
    return _reduce(rmo.MoReduceMaxOp, x, axis=axis)


def prod(x: TensorValueLike, axis: int = -1) -> TensorValue:
    """Reduces a symbolic tensor using a product operation.

    Computes the product of elements along a specified axis.

    Args:
        x: The input tensor for the operation.
        axis: The axis along which to compute the reduction. If negative,
            indexes from the last dimension. For example, a value of ``-1`` will
            compute the reduction along the last dimension.

    Returns:
        A symbolic tensor representing the result of the product operation.
        The tensor will have the same rank as the input tensor, and the same
        shape except along the ``axis`` dimension which will have size ``1``.
    """
    return _reduce(rmo.MoReduceMulOp, x, axis=axis)


def _reduce(
    op_type: type[Operation],
    x: TensorValueLike,
    axis: int = -1,
    out_dtype: DType | None = None,
) -> TensorValue:
    """Reduces a symbolic tensor using a reduction operation.

    Args:
        op_type: The reduction op class (for example,
            :class:`~max._core.dialects.rmo.MoReduceAddOp`).
        x: The input tensor for the operation.
        axis: The axis along which to compute the reduction. If negative,
            indexes from the last dimension. For example, a value of ``-1`` will
            compute the reduction along the last dimension.
        out_dtype: The dtype of the result. Defaults to the dtype of ``x``.

    Returns:
        A symbolic tensor representing the result of the reduction operation.
        The tensor will have the same rank as the input tensor, and the same
        shape except along the ``axis`` dimension which will have size ``1``.
    """
    x = TensorValue(x)

    if axis < 0:
        axis += x.rank
    if not 0 <= axis < x.rank:
        raise ValueError(f"Invalid {axis=} for input {x.rank=}")

    shape = Shape(x.shape)
    shape[axis] = Dim(1)
    result_type = TensorType(out_dtype or x.dtype, shape, x.device)
    return Graph.current._add_op_generated(
        op_type,
        result=result_type,
        input=x,
        axis=builtin.IntegerAttr(builtin.IndexType(), axis),
        output_param_decls=kgen.ParamDeclArrayAttr([]),
    )[0].tensor


def argmin(x: TensorValueLike, axis: int = -1) -> TensorValue:
    """Reduces a symbolic tensor using an argmin operation.

    When provided with a tensor with all identical elements,
    on CPU this will return the first element index in the tensor,
    on GPU this will return an arbitrary index.

    Args:
        x: The input tensor for the operation.
        axis: The axis along which to compute the reduction. If negative,
            indexes from the last dimension. For example, a value of ``-1`` will
            compute the reduction along the last dimension.

    Returns:
        A symbolic tensor representing the result of the argmin operation.
        The tensor will have the same rank as the input tensor, and the same
        shape except along the ``axis`` dimension which will have size ``1``.
    """
    return _reduce(rmo.MoReduceArgMinOp, x, axis, out_dtype=DType.int64)


def argmax(x: TensorValueLike, axis: int = -1) -> TensorValue:
    """Reduces a symbolic tensor using an argmax operation.

    When provided with a tensor with all identical elements,
    on CPU this will return the first element index in the tensor,
    on GPU this will return an arbitrary index.

    Args:
        x: The input tensor for the operation.
        axis: The axis along which to compute the reduction. If negative,
            indexes from the last dimension. For example, a value of ``-1`` will
            compute the reduction along the last dimension.

    Returns:
        A symbolic tensor representing the result of the argmax operation.
        The tensor will have the same rank as the input tensor, and the same
        shape except along the ``axis`` dimension which will have size ``1``.
    """
    return _reduce(rmo.MoReduceArgMaxOp, x, axis, out_dtype=DType.int64)
