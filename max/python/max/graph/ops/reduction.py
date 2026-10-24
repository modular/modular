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

import math

from max._core import Operation
from max._core.dialects import builtin, kgen, rmo
from max.dtype import DType

from ..dim import Dim, StaticDim
from ..graph import Graph
from ..shape import Shape
from ..type import TensorType
from ..value import TensorValue, TensorValueLike
from .cast import cast
from .constant import constant
from .elementwise import max as elementwise_max
from .reshape import reshape
from .shape_to_tensor import shape_to_tensor
from .transfer_to import transfer_to


def sum(x: TensorValueLike, axis: int = -1) -> TensorValue:
    """Computes the sum of elements along a specified axis.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("sum_example") as graph:
            x = ops.constant(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                DType.float32,
                device=device,
            )
            graph.output(ops.sum(x, axis=-1))  # shape (2, 1): [[6.0], [15.0]]

        model = InferenceSession().load(graph)
        result = model.execute()[0]

    Args:
        x: The input tensor for the operation.
        axis: The axis along which to compute the reduction. If negative,
            indexes from the last dimension. For example, a value of ``-1``
            computes the reduction along the last dimension. Defaults to
            ``-1``.

    Returns:
        A ``TensorValue`` representing the sum along ``axis``. It has the same
        rank as ``x``, with the ``axis`` dimension reduced to size ``1``.

    Raises:
        ValueError: If ``axis`` is out of range for the input's rank.
    """
    return _reduce(rmo.MoReduceAddOp, x, axis=axis)


def mean(x: TensorValueLike, axis: int = -1) -> TensorValue:
    """Computes the mean of elements along a specified axis.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("mean_example") as graph:
            x = ops.constant(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                DType.float32,
                device=device,
            )
            graph.output(ops.mean(x, axis=-1))  # shape (2, 1): [[2.0], [5.0]]

        model = InferenceSession().load(graph)
        result = model.execute()[0]

    Args:
        x: The input tensor for the operation.
        axis: The axis along which to compute the reduction. If negative,
            indexes from the last dimension. For example, a value of ``-1``
            computes the reduction along the last dimension. Defaults to
            ``-1``.

    Returns:
        A ``TensorValue`` representing the mean along ``axis``. It has the same
        rank as ``x``, with the ``axis`` dimension reduced to size ``1``.

    Raises:
        ValueError: If ``axis`` is out of range for the input's rank.
    """
    x = TensorValue(x)
    total = sum(x, axis=axis)
    if not x.dtype.is_float():
        # The mean of integers is an integer.
        return Graph.current._add_op_generated(
            rmo.DivOp, input_x=total, input_y=_axis_length(x, axis)
        )[0].tensor
    # A half-precision type can't hold a long axis's length, but it holds the
    # length's reciprocal.
    return Graph.current._add_op_generated(
        rmo.MulOp, input_x=total, input_y=_inverse_axis_length(x, axis)
    )[0].tensor


def _axis_length(x: TensorValue, axis: int) -> TensorValue:
    """Returns the length of ``axis`` of an integer ``x`` in ``x``'s dtype.

    The length is at least 1, so the mean of an empty axis is 0 instead of a
    division by zero.
    """
    dim = x.shape[axis]
    if isinstance(dim, StaticDim):
        length = constant(int(dim) or 1, DType.int64, device=x.device)
    else:
        length = elementwise_max(
            transfer_to(reshape(shape_to_tensor([dim]), []), x.device),
            constant(1, DType.int64, device=x.device),
        )
    return cast(length, x.dtype)


def _inverse_axis_length(x: TensorValue, axis: int) -> TensorValue:
    """Returns ``1 / length`` of ``axis`` of a float ``x`` in ``x``'s dtype.

    It's computed in at least float32 and rounded once. An empty axis gives
    ``inf``, so its mean is NaN.
    """
    dim = x.shape[axis]
    if isinstance(dim, StaticDim):
        n = int(dim)
        return constant(1.0 / n if n else math.inf, x.dtype, device=x.device)
    wide = DType.float64 if x.dtype == DType.float64 else DType.float32
    length = cast(
        transfer_to(reshape(shape_to_tensor([dim]), []), x.device), wide
    )
    inverse = Graph.current._add_op_generated(
        rmo.DivOp,
        input_x=constant(1.0, wide, device=x.device),
        input_y=length,
    )[0].tensor
    return cast(inverse, x.dtype)


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
            indexes from the last dimension. For example, a value of ``-1``
            computes the reduction along the last dimension. Defaults to
            ``-1``.

    Returns:
        A ``TensorValue`` representing the minimum along ``axis``. It has the
        same rank as ``x``, with the ``axis`` dimension reduced to size ``1``.

    Raises:
        ValueError: If ``axis`` is out of range for the input's rank.
    """
    return _reduce(rmo.MoReduceMinOp, x, axis=axis)


def max(x: TensorValueLike, axis: int = -1) -> TensorValue:
    """Computes the maximum value along a specified axis.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("max_example") as graph:
            x = ops.constant(
                [[1.2, 3.5, 2.1, 0.8], [2.3, 1.9, 4.2, 3.1]],
                DType.float32,
                device=device,
            )
            graph.output(ops.max(x, axis=-1))  # shape (2, 1): [[3.5], [4.2]]

        model = InferenceSession().load(graph)
        result = model.execute()[0]

    Args:
        x: The input tensor for the operation.
        axis: The axis along which to compute the reduction. If negative,
            indexes from the last dimension. For example, a value of ``-1``
            computes the reduction along the last dimension. Defaults to
            ``-1``.

    Returns:
        A ``TensorValue`` representing the maximum along ``axis``. It has the
        same rank as ``x``, with the ``axis`` dimension reduced to size ``1``.

    Raises:
        ValueError: If ``axis`` is out of range for the input's rank.
    """
    return _reduce(rmo.MoReduceMaxOp, x, axis=axis)


def prod(x: TensorValueLike, axis: int = -1) -> TensorValue:
    """Computes the product of elements along a specified axis.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("prod_example") as graph:
            x = ops.constant(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                DType.float32,
                device=device,
            )
            graph.output(ops.prod(x, axis=-1))  # shape (2, 1): [[6.0], [120.0]]

        model = InferenceSession().load(graph)
        result = model.execute()[0]

    Args:
        x: The input tensor for the operation.
        axis: The axis along which to compute the reduction. If negative,
            indexes from the last dimension. For example, a value of ``-1``
            computes the reduction along the last dimension. Defaults to
            ``-1``.

    Returns:
        A ``TensorValue`` representing the product along ``axis``. It has the
        same rank as ``x``, with the ``axis`` dimension reduced to size ``1``.

    Raises:
        ValueError: If ``axis`` is out of range for the input's rank.
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
    """Returns the indices of the minimum values along an axis.

    When the input contains ties (identical minimum values), behavior
    depends on the device: CPU returns the first matching index, while
    GPU may return any of them.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("argmin_example") as graph:
            x = ops.constant(
                [[1.2, 3.5, 2.1, 0.8], [2.3, 1.9, 4.2, 3.1]],
                DType.float32,
                device=device,
            )
            graph.output(ops.argmin(x, axis=-1))  # shape (2, 1): [[3], [1]]

        model = InferenceSession().load(graph)
        result = model.execute()[0]

    Args:
        x: The input tensor for the operation.
        axis: The axis along which to compute the reduction. If negative,
            indexes from the last dimension. For example, a value of ``-1``
            computes the reduction along the last dimension. Defaults to
            ``-1``.

    Returns:
        A ``TensorValue`` with ``int64`` dtype representing the indices of the
        minimum values along ``axis``. The result has the same rank as ``x``,
        with the ``axis`` dimension reduced to size ``1``.

    Raises:
        ValueError: If ``axis`` is out of range for the input's rank.
    """
    return _reduce(rmo.MoReduceArgMinOp, x, axis, out_dtype=DType.int64)


def argmax(x: TensorValueLike, axis: int = -1) -> TensorValue:
    """Returns the indices of the maximum values along an axis.

    It's useful for finding the position of the largest element along a
    given dimension, such as determining predicted classes in
    classification.

    When the input contains ties (identical maximum values), behavior
    depends on the device: CPU returns the first matching index, while
    GPU may return any of them.

    .. code-block:: python

        import numpy as np
        from max.driver import Accelerator, CPU, accelerator_count
        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = Accelerator() if accelerator_count() > 0 else CPU()
        device_ref = DeviceRef.from_device(device)

        with Graph("argmax", input_types=[]) as graph:
            x = ops.constant(
                [[1.2, 3.5, 2.1, 0.8], [2.3, 1.9, 4.2, 3.1]],
                DType.float32,
                device=device_ref,
            )
            indices = ops.argmax(x, axis=-1)
            # indices has shape (2, 1): [[1], [2]]
            graph.output(indices)

        model = InferenceSession(devices=[device]).load(graph)
        result = model.execute()[0]

    .. invisible-code-block: python

        np.testing.assert_array_equal(result.to_numpy(), [[1], [2]])

    Args:
        x: The input tensor.
        axis: The axis along which to compute the argmax. Negative values
            index from the last dimension. Defaults to ``-1``.

    Returns:
        A ``TensorValue`` with ``int64`` dtype representing the indices of the
        maximum values along ``axis``. The result has the same rank as
        ``x``, with the ``axis`` dimension reduced to size ``1``.

    Raises:
        ValueError: If ``axis`` is out of range for the input's rank.
    """
    return _reduce(rmo.MoReduceArgMaxOp, x, axis, out_dtype=DType.int64)
