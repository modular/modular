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
"""Elementwise ops."""

from collections.abc import Callable

from max.dtype import DType
from max.mlir.dialects import rmo

from .. import dtype_promotion
from ..graph import Graph
from ..type import TensorType
from ..value import TensorValue, TensorValueLike
from .cast import cast

# ===----------------------------------------------------------------------=== #
# Utilities
# ===----------------------------------------------------------------------=== #


# This implementation needs to be in sync with the mojo implementation found in
# stdlib/utils/numerics.mojo
def _accum_type(x: TensorValue, preferred_type: DType = DType.float32) -> DType:
    dtype = x.dtype
    if dtype.is_float8():
        return (
            DType.float32 if preferred_type == DType.float32 else DType.bfloat16
        )
    if dtype == DType.float16:
        return (
            DType.float32 if preferred_type == DType.float32 else DType.float16
        )
    if dtype == DType.bfloat16:
        return DType.float32
    return dtype


# ===----------------------------------------------------------------------=== #
# Binary Ops
# ===----------------------------------------------------------------------=== #
# Note: Keep alphabetized.


def _elementwise_binary(op):  # noqa: ANN001, ANN202
    def elementwise_op(
        lhs: TensorValueLike, rhs: TensorValueLike
    ) -> TensorValue:
        lhs, rhs = dtype_promotion._promote_weak_dtypes(lhs, rhs)
        return Graph.current._add_op(op, lhs, rhs)[0].tensor

    elementwise_op.__name__ = op.__name__
    return elementwise_op


add = _elementwise_binary(rmo.add)
"""
Adds two symbolic tensors.

Creates a new op node to compute the addition of two symbol tensor values
and adds it to the graph, returning the symbolic result.

The following shows an example of the `add()` function with two inputs:

.. code-block:: python

    def add_graph():
        input_type = TensorType(dtype=DType.float32, shape=(2,), device=DeviceRef.CPU())

        with Graph("add_graph", input_types=(input_type, input_type)) as graph:
            x = graph.inputs[0]
            y = graph.inputs[1]

            out = ops.add(x, y)
            graph.output(out)

-
    - If ``lhs`` and ``rhs`` have different dtypes, they will be promoted according
        to the dtype promotion rules before the operation.
    - If ``lhs`` and ``rhs`` have different shapes, they will be broadcast to the
        same shape according to broadcasting rules` before the operation.

Args:
    lhs: The symbol to use as left side of the addition.
    rhs: The symbol to use as right side of the addition.
    location: An optional location for a more specific error message.

Returns:
    A symbolic tensor value representing the output of the addition.
    The result will have:
    - the same dtype as the type-promotion of the two input dtypes
    - the same shape as the broadcast of the two input shapes.

Raises:
    Error: If the input values' shapes are not compatible for broadcasting.
    Error: If one of the input values has an unsupported dtype.
    Error: If the two symbols are parts of different graphs.
"""


def div(lhs: TensorValueLike, rhs: TensorValueLike) -> TensorValue:
    """
    Divides two symbolic tensors using true division (Python operator `/`).

    For integer operands, this performs true division by promoting to float,
    matching Python's `/` operator behavior. For floating-point operands,
    this performs standard floating-point division.

    Creates a new op node to compute the division of two symbol tensor values
    and adds it to the graph, returning the symbolic result.

    Args:
        lhs: The symbol to use as left side of the division.
        rhs: The symbol to use as right side of the division.

    Returns:
        A symbolic tensor value representing the output of the division. The
          result will have:
            - floating-point dtype for integer operands, promoted dtype for mixed types
            - the same shape as the broadcast of the two input shapes.

    Raises:
        Error: If the input values' shapes are not compatible for broadcasting.
        Error: If one of the input values has an unsupported dtype.
        Error: If the two symbols are parts of different graphs.
    """
    lhs, rhs = dtype_promotion._promote_weak_dtypes(lhs, rhs)

    if lhs.dtype.is_integral() and rhs.dtype.is_integral():
        float_dtype = DType.float64  # Use double precision for accuracy
        lhs = cast(lhs, float_dtype)
        rhs = cast(rhs, float_dtype)

    return Graph.current._add_op(rmo.div, lhs, rhs)[0].tensor


max = _elementwise_binary(rmo.max)
"""
Computes the elementwise maximum of two symbolic tensors.

Creates a new op node to compute the maximum of two symbol tensor values
and adds it to the graph, returning the symbolic result.

.. code-block:: python

    def maximum_graph():
        input_type = TensorType(dtype=DType.float32, shape=(2,), device=DeviceRef.CPU())

        with Graph("maximum_graph", input_types=(input_type, input_type)) as graph:
            out = ops.max(graph.inputs[0], graph.inputs[1])
            graph.output(out)

-
    - If ``lhs`` and ``rhs`` have different dtypes, they will be promoted
      according to the dtype promotion rules before the operation.
    - If ``lhs`` and ``rhs`` have different shapes, they will be broadcast to
      the same shape according to broadcasting rules` before the operation.

Args:
    lhs: The symbol to use as left side of the maximum operation.
    rhs: The symbol to use as right side of the maximum operation.

Returns:
    A symbolic tensor value representing the output of the maximum operation.
    The result will have:
    - the same dtype as the type-promotion of the two input dtypes
    - the same shape as the broadcast of the two input shapes.

Raises:
    Error: If the input values' shapes are not compatible for broadcasting.
    Error: If one of the input values has an unsupported dtype.
    Error: If the two symbols are parts of different graphs.
"""
min = _elementwise_binary(rmo.min)
"""
Computes the elementwise minimum of two symbolic tensors.

Creates a new op node to compute the minimum of two symbol tensor values
and adds it to the graph, returning the symbolic result.

.. code-block:: python

    def min_graph():
        input_type = TensorType(dtype=DType.float32, shape=(2,), device=DeviceRef.CPU())

        with Graph("min_graph", input_types=(input_type, input_type)) as graph:
            out = ops.min(graph.inputs[0], graph.inputs[1])
            graph.output(out)

-
    - If ``lhs`` and ``rhs`` have different dtypes, they will be promoted
      according to the dtype promotion rules before the operation.
    - If ``lhs`` and ``rhs`` have different shapes, they will be broadcast to
      the same shape according to broadcasting rules` before the operation.

Args:
    lhs: The symbol to use as left side of the minimum operation.
    rhs: The symbol to use as right side of the minimum operation.

Returns:
    A symbolic tensor value representing the output of the minimum operation.
    The result will have:
    - the same dtype as the type-promotion of the two input dtypes
    - the same shape as the broadcast of the two input shapes.

Raises:
    Error: If the input values' shapes are not compatible for broadcasting.
    Error: If one of the input values has an unsupported dtype.
    Error: If the two symbols are parts of different graphs.
"""
mod = _elementwise_binary(rmo.mod)
"""
Computes the elementwise modulus of two symbolic tensors.

Creates a new op node to compute the modulus of two symbol tensor values
and adds it to the graph, returning the symbolic result.

-
    - If ``lhs`` and ``rhs`` have different dtypes, they will be promoted
      according to the dtype promotion rules before the operation.
    - If ``lhs`` and ``rhs`` have different shapes, they will be broadcast to
      the same shape according to broadcasting rules` before the operation.

Args:
    lhs: The symbol to use as left side of the modulus operation.
    rhs: The symbol to use as right side of the modulus operation.

Returns:
    A symbolic tensor value representing the output of the modulus operation.
    The result will have:
    - the same dtype as the type-promotion of the two input dtypes
    - the same shape as the broadcast of the two input shapes.

Raises:
    Error: If the input values' shapes are not compatible for broadcasting.
    Error: If one of the input values has an unsupported dtype.
    Error: If the two symbols are parts of different graphs.
"""
mul = _elementwise_binary(rmo.mul)
"""
Computes the elementwise multiplication of two symbolic tensors.

Creates a new op node to compute the multiplication of two symbol tensor values
and adds it to the graph, returning the symbolic result.

-
    - If ``lhs`` and ``rhs`` have different dtypes, they will be promoted
      according to the dtype promotion rules before the operation.
    - If ``lhs`` and ``rhs`` have different shapes, they will be broadcast to
      the same shape according to broadcasting rules` before the operation.

Args:
    lhs: The symbol to use as left side of the multiplication.
    rhs: The symbol to use as right side of the multiplication.

Returns:
    A symbolic tensor value representing the output of the multiplication.
    The result will have:
    - the same dtype as the type-promotion of the two input dtypes
    - the same shape as the broadcast of the two input shapes.

Raises:
    Error: If the input values' shapes are not compatible for broadcasting.
    Error: If one of the input values has an unsupported dtype.
    Error: If the two symbols are parts of different graphs.
"""
pow = _elementwise_binary(rmo.pow)
"""
Computes the elementwise exponentiation of two symbolic tensors.

Creates a new op node to compute the exponentiation of two symbol tensor values
and adds it to the graph, returning the symbolic result.

-
    - If ``lhs`` and ``rhs`` have different dtypes, they will be promoted
      according to the dtype promotion rules before the operation.
    - If ``lhs`` and ``rhs`` have different shapes, they will be broadcast to
      the same shape according to broadcasting rules` before the operation.

Args:
    lhs: The symbol to use as left side of the exponentiation.
    rhs: The symbol to use as right side of the exponentiation.

Returns:
    A symbolic tensor value representing the output of the exponentiation.
    The result will have:
    - the same dtype as the type-promotion of the two input dtypes
    - the same shape as the broadcast of the two input shapes.

Raises:
    Error: If the input values' shapes are not compatible for broadcasting.
    Error: If one of the input values has an unsupported dtype.
    Error: If the two symbols are parts of different graphs.
"""
sub = _elementwise_binary(rmo.sub)
"""
Computes the elementwise subtraction of two symbolic tensors.

Creates a new op node to compute the subtraction of two symbol tensor values
and adds it to the graph, returning the symbolic result.

.. code-block:: python

    def sub_graph():
        input_type = TensorType(dtype=DType.float32, shape=(2,), device=DeviceRef.CPU())

        with Graph("sub_graph", input_types=(input_type, input_type)) as graph:
            x = graph.inputs[0]  # Minuend (number being subtracted from)
            y = graph.inputs[1]  # Subtrahend (number being subtracted)

            out = ops.sub(x, y)
            graph.output(out)

-
    - If ``lhs`` and ``rhs`` have different dtypes, they will be promoted
      according to the dtype promotion rules before the operation.
    - If ``lhs`` and ``rhs`` have different shapes, they will be broadcast to
      the same shape according to broadcasting rules` before the operation.

Args:
    lhs: The symbol to use as left side of the subtraction.
    rhs: The symbol to use as right side of the subtraction.

Returns:
    A symbolic tensor value representing the output of the subtraction.
    The result will have:
    - the same dtype as the type-promotion of the two input dtypes
    - the same shape as the broadcast of the two input shapes.

Raises:
    Error: If the input values' shapes are not compatible for broadcasting.
    Error: If one of the input values has an unsupported dtype.
    Error: If the two symbols are parts of different graphs.
"""
equal = _elementwise_binary(rmo.equal)
"""
Computes the elementwise equality comparison between two symbolic tensors.

Creates a new op node to compute the equality comparison of two symbol
tensor values and adds it to the graph, returning the symbolic result.

.. code-block:: python

    def equal_graph():
        input_type = TensorType(dtype=DType.float32, shape=(3,), device=DeviceRef.CPU())

        with Graph("equal_graph", input_types=(input_type, input_type)) as graph:
            x = graph.inputs[0]  # First input
            y = graph.inputs[1]  # Second input

            out = ops.equal(x, y)
            graph.output(out)

-
    - If ``lhs`` and ``rhs`` have different dtypes, they will be promoted
      according to the dtype promotion rules before the operation.
    - If ``lhs`` and ``rhs`` have different shapes, they will be broadcast to
      the same shape according to broadcasting rules` before the operation.

Args:
    lhs: The symbol to use as left side of the equality comparison.
    rhs: The symbol to use as right side of the equality comparison.

Returns:
    A symbolic tensor value representing the output of the equality comparison.
    The result will have:
    - the same dtype as the type promotion of the two input dtypes
    - the same shape as the broadcast of the two input shapes.

Raises:
    Error: If the input values' shapes are not compatible for broadcasting.
    Error: If one of the input values has an unsupported dtype.
    Error: If the two symbols are parts of different graphs.
"""
greater = _elementwise_binary(rmo.greater)
"""
Computes the elementwise greater than comparison between two symbolic tensors.

Creates a new op node to compute the greater than comparison of two symbol
tensor values and adds it to the graph, returning the symbolic result.


.. code-block:: python

    def greater_than_graph():
        input_type = TensorType(dtype=DType.float32, shape=(2,), device=DeviceRef.CPU())

        with Graph("greater_graph", input_types=(input_type, input_type)) as graph:
            x = graph.inputs[0]  # Left hand side
            y = graph.inputs[1]  # Right hand side

            out = ops.greater(x, y)
            graph.output(out)

-
    - If ``lhs`` and ``rhs`` have different dtypes, they will be promoted
      according to the dtype promotion rules before the operation.
    - If ``lhs`` and ``rhs`` have different shapes, they will be broadcast to
      the same shape according to broadcasting rules` before the operation.

Args:
    lhs: The symbol to use as left side of the greater than comparison.
    rhs: The symbol to use as right side of the greater than comparison.

Returns:
    A symbolic tensor value representing the output of the greater than comparison.
    The result will have:
    - the same dtype as the type-promotion of the two input dtypes
    - the same shape as the broadcast of the two input shapes.

Raises:
    Error: If the input values' shapes are not compatible for broadcasting.
    Error: If one of the input values has an unsupported dtype.
    Error: If the two symbols are parts of different graphs.
"""
greater_equal = _elementwise_binary(rmo.greater_equal)
"""
Computes the elementwise greater-or-equal comparison between two symbolic tensors.

Creates a new op node to compute the equality comparison of two symbol
tensor values and adds it to the graph, returning the symbolic result.

-
    - If ``lhs`` and ``rhs`` have different dtypes, they will be promoted
      according to the dtype promotion rules before the operation.
    - If ``lhs`` and ``rhs`` have different shapes, they will be broadcast to
      the same shape according to broadcasting rules` before the operation.

Args:
    lhs: The symbol to use as left side of the greater-or-equal comparison.
    rhs: The symbol to use as right side of the greater-or-equal comparison.

Returns:
    A symbolic tensor value representing the output of the greater-or-equal comparison.
    The result will have:
    - the same dtype as the type-promotion of the two input dtypes
    - the same shape as the broadcast of the two input shapes.

Raises:
    Error: If the input values' shapes are not compatible for broadcasting.
    Error: If one of the input values has an unsupported dtype.
    Error: If the two symbols are parts of different graphs.
"""
not_equal = _elementwise_binary(rmo.not_equal)
"""
Computes the elementwise inequality comparison between two symbolic tensors.

Creates a new op node to compute the inequality comparison of two symbol
tensor values and adds it to the graph, returning the symbolic result.


.. code-block:: python

    def not_equal_graph():
        input_type = TensorType(dtype=DType.float32, shape=(2,), device=DeviceRef.CPU())

        with Graph("not_equal_graph", input_types=(input_type, input_type)) as graph:
            x = graph.inputs[0]  # Left hand side
            y = graph.inputs[1]  # Right hand side

            out = ops.not_equal(x, y)
            graph.output(out)

-
    - If ``lhs`` and ``rhs`` have different dtypes, they will be promoted
      according to the dtype promotion rules before the operation.
    - If ``lhs`` and ``rhs`` have different shapes, they will be broadcast to
      the same shape according to broadcasting rules` before the operation.

Args:
    lhs: The symbol to use as left side of the inequality comparison.
    rhs: The symbol to use as right side of the inequality comparison.

Returns:
    A symbolic tensor value representing the output of the inequality comparison.
    The result will have:
    - the same dtype as the type-promotion of the two input dtypes
    - the same shape as the broadcast of the two input shapes.

Raises:
    Error: If the input values' shapes are not compatible for broadcasting.
    Error: If one of the input values has an unsupported dtype.
    Error: If the two symbols are parts of different graphs.
"""

logical_and = _elementwise_binary(rmo.and_)
"""
Computes the logical and between two symbolic tensors.

Only supports boolean inputs. If ``lhs`` and ``rhs`` have different shapes,
they will be broadcast to the same shape according to broadcasting rules
before the operation.

Args:
    lhs: The symbol to use as left side of the logical and.
    rhs: The symbol to use as right side of the logical and.

Returns:
    A symbolic tensor value representing the output of the and operation.
    The result will be a boolean tensor with the shape determined by
    broadcasting the two input shapes.

Raises:
    Error: If the input values' shapes are not compatible for broadcasting.
    Error: If one of the input values has an unsupported dtype.
    Error: If the two symbols are parts of different graphs.
"""

logical_or = _elementwise_binary(rmo.or_)
"""
Computes the logical or between two symbolic tensors.

Only supports boolean inputs. If ``lhs`` and ``rhs`` have different shapes,
they will be broadcast to the same shape according to broadcasting rules
before the operation.

Args:
    lhs: The symbol to use as left side of the logical or.
    rhs: The symbol to use as right side of the logical or.

Returns:
    A symbolic tensor value representing the output of the or operation.
    The result will be a boolean tensor with the shape determined by
    broadcasting the two input shapes.

Raises:
    Error: If the input values' shapes are not compatible for broadcasting.
    Error: If one of the input values has an unsupported dtype.
    Error: If the two symbols are parts of different graphs.
"""

logical_xor = _elementwise_binary(rmo.xor)
"""
Computes the logical xor between two symbolic tensors.

Only supports boolean inputs. If ``lhs`` and ``rhs`` have different shapes,
they will be broadcast to the same shape according to broadcasting rules
before the operation.

Args:
    lhs: The symbol to use as left side of the logical xor.
    rhs: The symbol to use as right side of the logical xor.

Returns:
    A symbolic tensor value representing the output of the xor operation.
    The result will be a boolean tensor with the shape determined by
    broadcasting the two input shapes.

Raises:
    Error: If the input values' shapes are not compatible for broadcasting.
    Error: If one of the input values has an unsupported dtype.
    Error: If the two symbols are parts of different graphs.
"""


# ===----------------------------------------------------------------------=== #
# Unary Ops
# ===----------------------------------------------------------------------=== #
# Note: Keep alphabetized.


def _elementwise_unary(op):  # noqa: ANN001, ANN202
    def elementwise_op(x: TensorValueLike) -> TensorValue:
        x = dtype_promotion._restrict_to_strong_dtypes(x)
        return Graph.current._add_op(op, x._mlir_value.type, x)[0].tensor

    elementwise_op.__name__ = op.__name__
    return elementwise_op


def _elementwise_unary_predicate(
    op,  # noqa: ANN001
) -> Callable[[TensorValueLike], TensorValue]:
    def elementwise_op(x: TensorValueLike) -> TensorValue:
        x = dtype_promotion._restrict_to_strong_dtypes(x)
        return Graph.current._add_op(
            op,
            TensorType(
                dtype=DType.bool, shape=x.shape, device=x.device
            ).to_mlir(),
            x,
        )[0].tensor

    elementwise_op.__name__ = op.__name__
    return elementwise_op


abs = _elementwise_unary(rmo.mo_abs)
"""
Computes the elementwise absolute value of a symbolic tensor.

Creates a new op node to compute the elementwise absolute value of a
symbolic tensor and adds it to the graph, returning the symbolic result.

The following demonstrates how to compute the absolute value using the :obj:`abs()` function:

.. code-block:: python

    def abs_graph():
        input_type = TensorType(dtype=DType.float32, shape=(2,), device=DeviceRef.CPU())

        with Graph("abs_graph", input_types=(input_type,)) as graph:
            x = graph.inputs[0]

            out = ops.abs(x)
            graph.output(out)

Args:
    value: The symbolic tensor to use as the input to the absolute value
        computation.

Returns:
    A new symbolic tensor value representing the output of the absolute
        value computation.

Raises:
    Error: If the symbol doesn't represent a tensor value.
"""
exp = _elementwise_unary(rmo.mo_exp)
"""
Computes the elementwise exp function of a symbolic tensor.

Creates a new op node to compute the elementwise exp function of a
symbolic tensor and adds it to the graph, returning the symbolic result.

``exp`` is defined as ``exp(x) = e^x``, where ``e`` is Euler's number.

Args:
    value: The symbolic tensor to use as the input to the exp function
        computation.

Returns:
    A new symbolic tensor value representing the output of the exp
        value computation.

Raises:
    Error: If the symbol doesn't represent a tensor value.
"""
erf = _elementwise_unary(rmo.mo_erf)
"""
Computes the elementwise error function of a symbolic tensor.

Creates a new op node to compute the elementwise error function of a
symbolic tensor and adds it to the graph, returning the symbolic result.

The error function ``erf`` is defined as the probability that a randomly
sampled normal distribution falls within a given range.

Args:
    value: The symbolic tensor to use as the input to the error function
           computation.

Returns:
    A new symbolic tensor value representing the output of the error function
    value computation.

Raises:
    Error: If the symbol doesn't represent a tensor value.
"""

_gelu_exact = _elementwise_unary(rmo.mo_gelu)
"""
Computes the elementwise gelu function of a symbolic tensor.

Creates a new op node to compute the elementwise gelu function of a
symbolic tensor and adds it to the graph, returning the symbolic result.

``gelu`` is defined as ``$$gelu(x) = x \\Phi(x)$$`` where ``$$\\Phi$$`` is the
cumulative distribution function of the Gaussian distribution.

Args:
    value: The symbolic tensor to use as the input to the gelu function
        computation.

Returns:
    A new symbolic tensor value representing the output of the gelu computation.

Raises:
    Error: If the symbol doesn't represent a tensor value.
"""


def _gelu_quick(x: TensorValue):  # noqa: ANN202
    """
    Computes the elementwise quick gelu of a symbolic tensor.

    Creates a new op node to compute the elementwise quick gelu of a
    symbolic tensor and adds it to the graph, returning the symbolic result.

    ``quick gelu`` is defined as ``gelu_quick(x) = sigmoid(1.702 * x) * x``.

    References:
        - https://github.com/hendrycks/GELUs
        - https://arxiv.org/abs/1606.08415

    Args:
        value: The symbolic tensor to use as the input to the quick gelu
            computation.

    Returns:
        A new symbolic tensor value representing the output of the quick gelu
            ccomputation.

    Raises:
        Error: If the symbol doesn't represent a tensor value.
    """
    x_cast = x.cast(_accum_type(x))
    return (x_cast * sigmoid(x_cast * 1.702)).cast(x.dtype)


def _gelu_tanh(x: TensorValue):  # noqa: ANN202
    """
    Computes the elementwise gelu of a symbolic tensor.

    Creates a new op node to compute the elementwise gelu of a
    symbolic tensor and adds it to the graph, returning the symbolic result.

    Args:
        value: The symbolic tensor to use as the input to the gelu
            computation.

    Returns:
        A new symbolic tensor value representing the output of the tanh
            value computation.

    Raises:
        Error: If the symbol doesn't represent a tensor value.
    """
    x_cast = x.cast(_accum_type(x))
    return (
        x_cast
        * 0.5
        * (1.0 + tanh(0.7978845608028654 * (x_cast + 0.044715 * x_cast**3)))
    ).cast(x.dtype)


def gelu(x: TensorValue, approximate: str = "none"):  # noqa: ANN201
    """
    Computes the elementwise gelu of a symbolic tensor.

    Creates a new op node to compute the elementwise gelu of a
    symbolic tensor and adds it to the graph, returning the symbolic result.

    For ``approximate == "none"``, the exact gelu function is computed.

    For ``approximate == "tanh"``, the approximation:

    .. math::

        gelu(x) = 0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x**3)))

    is used.

    For ``approximate == "quick"``, the approximation:


    .. math::

        gelu(x) = sigmoid(1.702 * x) * x

    is used.

    Args:
        value: The symbolic tensor to use as the input to the gelu
            computation.

    Returns:
        A new symbolic tensor value representing the output of the gelu
            value computation.

    Raises:
        Error: If the symbol doesn't represent a tensor value.
        ValueError: If the approximation method is invalid.
    """
    if approximate == "none":
        return _gelu_exact(x)
    if approximate == "tanh":
        return _gelu_tanh(x)
    if approximate == "quick":
        return _gelu_quick(x)

    raise ValueError(f"Invalid approximation method: {approximate}")


log = _elementwise_unary(rmo.mo_log)
"""
Computes the elementwise natural logarithm of a symbolic tensor.

Creates a new op node to compute the elementwise natural logarithm of a
symbolic tensor and adds it to the graph, returning the symbolic result.

The natural logarithm function ``log`` is defined as the inverse of the
exponential function ``exp()``. In other words, it computes the value ``y`` in
the equation ``x = e^y`` where ``e`` is Euler's number.

``log(x)`` is undefined for ``x <= 0`` for real numbers. Complex numbers
are currently unsupported.

Args:
    value: The symbolic tensor to use as the input to the natural logarithm
        computation.

Returns:
    A new symbolic tensor value representing the output of the natural logarithm
        value computation.

Raises:
    Error: If the symbol doesn't represent a tensor value.
"""

log1p = _elementwise_unary(rmo.mo_log1p)
"""
Computes the elementwise logarithm of 1 plus a symbolic tensor.

Creates a new op node to compute the elementwise log1p of a
symbolic tensor and adds it to the graph, returning the symbolic result.

The ``log1p`` function is defined as ``log1p(x) = log(1 + x)``, where ``log()``
is the natural logarithm.

Using ``log1p(x)`` rather than computing ``log(1 + x)`` can give greater
numerical precision results.

``log(x)`` is undefined for ``x <= 0`` for real numbers. Complex numbers
are currently unsupported.

Args:
    value: The symbolic tensor to use as the input to the log1p
        computation.

Returns:
    A new symbolic tensor value representing the output of the log1p
        value computation.

Raises:
    Error: If the symbol doesn't represent a tensor value.
"""

logsoftmax = _elementwise_unary(rmo.mo_logsoftmax)
"""
Computes the elementwise logsoftmax of a symbolic tensor.

Creates a new op node to compute the elementwise logsoftmax of a
symbolic tensor and adds it to the graph, returning the symbolic result.

Args:
    value: The symbolic tensor to use as the input to the logsoftmax
        computation.

Returns:
    A new symbolic tensor value representing the output of the logsoftmax
        value computation.

Raises:
    Error: If the symbol doesn't represent a tensor value.
"""

relu = _elementwise_unary(rmo.mo_relu)
"""
Computes the elementwise relu of a symbolic tensor.

Creates a new op node to compute the elementwise relu of a
symbolic tensor and adds it to the graph, returning the symbolic result.

Args:
    value: The symbolic tensor to use as the input to the relu
        computation.

Returns:
    A new symbolic tensor value representing the output of the relu
        value computation.

Raises:
    Error: If the symbol doesn't represent a tensor value.
"""

sigmoid = _elementwise_unary(rmo.sigmoid)
"""
Computes the elementwise sigmoid of a symbolic tensor.

Creates a new op node to compute the elementwise sigmoid of a
symbolic tensor and adds it to the graph, returning the symbolic result.

Args:
    value: The symbolic tensor to use as the input to the sigmoid
        computation.

Returns:
    A new symbolic tensor value representing the output of the sigmoid
        value computation.

Raises:
    Error: If the symbol doesn't represent a tensor value.
"""


def silu(x: TensorValue):  # noqa: ANN201
    """
    Computes the elementwise silu of a symbolic tensor.

    Creates a new op node to compute the elementwise silu of a
    symbolic tensor and adds it to the graph, returning the symbolic result.

    ``silu`` is defined as ``silu(x) = x * sigmoid(x)``.

    Args:
        value: The symbolic tensor to use as the input to the silu
            computation.

    Returns:
        A new symbolic tensor value representing the output of the silu
            value computation.

    Raises:
        Error: If the symbol doesn't represent a tensor value.
    """
    x_cast = x.cast(_accum_type(x))
    return mul(x_cast, sigmoid(x_cast)).cast(x.dtype)


softmax = _elementwise_unary(rmo.mo_softmax)
"""
Computes the elementwise softmax of a symbolic tensor.

Creates a new op node to compute the elementwise softmax of a
symbolic tensor and adds it to the graph, returning the symbolic result.

Args:
    value: The symbolic tensor to use as the input to the softmax
        computation.

Returns:
    A new symbolic tensor value representing the output of the softmax
        value computation.

Raises:
    Error: If the symbol doesn't represent a tensor value.
"""

cos = _elementwise_unary(rmo.mo_cos)
"""
Computes the elementwise cosine of a symbolic tensor.

Creates a new op node to compute the elementwise cosine of a
symbolic tensor and adds it to the graph, returning the symbolic result.

Args:
    value: The symbolic tensor to use as the input to the cos
           computation. If it's not a floating-point DType, an exception will be
           raised.

Returns:
    A new symbolic tensor value representing the output of the cosine
    value computation.

Raises:
    Error: If the symbol doesn't represent a tensor value.
"""

floor = _elementwise_unary(rmo.mo_floor)
"""
Computes the elementwise floor of a symbolic tensor.

Creates a new op node to compute the elementwise floor of a
symbolic tensor and adds it to the graph, returning the symbolic result.

Args:
    value: The symbolic tensor to use as the input to the floor
           computation. If it's not a floating-point DType, an exception will be
           raised.

Returns:
    A new symbolic tensor value representing the output of the floor
    value computation.

Raises:
    Error: If the symbol doesn't represent a tensor value.
"""

round = _elementwise_unary(rmo.mo_round)
"""
Computes the elementwise round of a symbolic tensor.


Creates a new op node to compute the elementwise round of a
symbolic tensor and adds it to the graph, returning the symbolic result.
Rounding is done with ties towards the nearest even number.

For example, if the model has one input tensor:

.. code-block:: python

    def round_graph():
        input_type = TensorType(dtype=DType.float32, shape=(4,), device=DeviceRef.CPU())

        with Graph("round_graph_example", input_types=(input_type,)) as graph:
            x = graph.inputs[0]
            out = ops.round(x)
            graph.output(out)

Args:
    value: The symbolic tensor to use as the input to the round
           computation. If it's not a floating-point DType, an exception will be
           raised.

Returns:
    A new symbolic tensor value representing the output of the round
    value computation.

Raises:
    Error: If the symbol doesn't represent a tensor value.
"""

rsqrt = _elementwise_unary(rmo.mo_isqrt)
"""
Computes the elementwise inverse-square-root of a symbolic tensor.

Creates a new op node to compute the elementwise rsqrt of a
symbolic tensor and adds it to the graph, returning the symbolic result.

Args:
    value: The symbolic tensor to use as the input to the rsqrt
        computation. If it's not a floating-point DType, an exception will be raised.

Returns:
    A new symbolic tensor value representing the output of the rsqrt
        value computation.

Raises:
    Error: If the symbol doesn't represent a tensor value.
"""

sqrt = _elementwise_unary(rmo.mo_sqrt)
"""
Computes the elementwise sqrt of a symbolic tensor.

Creates a new op node to compute the elementwise sqrt of a
symbolic tensor and adds it to the graph, returning the symbolic result.

Args:
    value: The symbolic tensor to use as the input to the sqrt
        computation. If it's not a floating-point DType, an exception will be raised.

Returns:
    A new symbolic tensor value representing the output of the sqrt
        value computation.

Raises:
    Error: If the symbol doesn't represent a tensor value.
"""

sin = _elementwise_unary(rmo.mo_sin)
"""
Computes the elementwise sine of a symbolic tensor.

Creates a new op node to compute the elementwise sine of a
symbolic tensor and adds it to the graph, returning the symbolic result.

Args:
    value: The symbolic tensor to use as the input to the sin
        computation. If it's not a floating-point DType, an exception will be raised.

Returns:
    A new symbolic tensor value representing the output of the sin
        value computation.

Raises:
    Error: If the symbol doesn't represent a tensor value.
"""
tanh = _elementwise_unary(rmo.mo_tanh)
"""
Computes the elementwise tanh of a symbolic tensor.

Creates a new op node to compute the elementwise tanh of a
symbolic tensor and adds it to the graph, returning the symbolic result.

Args:
    value: The symbolic tensor to use as the input to the tanh
        computation. If it's not a floating-point DType, an exception will be raised.

Returns:
    A new symbolic tensor value representing the output of the tanh
        value computation.

Raises:
    Error: If the symbol doesn't represent a tensor value.
"""

atanh = _elementwise_unary(rmo.mo_atanh)
"""
Computes the elementwise atanh of a symbolic tensor.

Creates a new op node to compute the elementwise tanh of a
symbolic tensor and adds it to the graph, returning the symbolic result.

Args:
    value: The symbolic tensor to use as the input to the tanh
        computation. If it's not a floating-point DType, an exception will be raised.

Returns:
    A new symbolic tensor value representing the output of the tanh
        value computation.

Raises:
    Error: If the symbol doesn't represent a tensor value.
"""

trunc = _elementwise_unary(rmo.mo_trunc)
"""
Computes the elementwise truncation of a symbolic tensor.

Creates a new op node to compute the elementwise truncation of a
symbolic tensor and adds it to the graph, returning the symbolic result.

Args:
    value: The symbolic tensor to use as the input to the truncation
        computation. If it's not a floating-point DType, an exception will be
        raised.

Returns:
    A new symbolic tensor value representing the output of the truncation
        value computation.

Raises:
    Error: If the symbol doesn't represent a tensor value.
"""

is_nan = _elementwise_unary_predicate(rmo.mo_is_nan)
"""
Computes the elementwise is_nan of a symbolic tensor.

Creates a new op node to compute the elementwise is_nan of a
symbolic tensor and adds it to the graph, returning the symbolic result.

Args:
    value: The symbolic tensor to use as the input to the is_nan
        computation.

Returns:
    The result will have:
        - element type ``bool``, true if the element at a given position
            is NaN, false otherwise
        - the same shape as the input value.

Raises:
    Error: If the symbol doesn't represent a tensor value.
"""


is_inf = _elementwise_unary_predicate(rmo.mo_is_inf)
"""
Computes the elementwise :obj:`is_inf()` of a symbolic tensor.

Creates a new op node to compute the elementwise :obj:`is_inf()` of a
symbolic tensor and adds it to the graph, returning the symbolic result.

Args:
    value: The symbolic tensor to use as the input to the :obj:`is_inf()`
        computation.

Returns:
    The result will have:
        - element type ``bool``, true if the element at a given position
            is plus or minus infinity, false otherwise
        - the same shape as the input value.

Raises:
    Raises: If the symbol doesn't represent a tensor value.
"""

logical_not = _elementwise_unary(rmo.mo_not)
"""
Computes the elementwise logical_not of a symbolic tensor.

Creates a new op node to compute the elementwise logical_not of a
symbolic tensor and adds it to the graph, returning the symbolic result.

Args:
    value: The symbolic tensor to use as the input to the logical_not
        computation.

Returns:
    The result will have:
        - element type ``bool``, true if the element at a given position
            is plus or minus infinity, false otherwise
        - the same shape as the input value.

Raises:
    Error: If the symbol doesn't represent a tensor value.
"""

negate = _elementwise_unary(rmo.mo_negative)
"""
Computes the elementwise negation of a symbolic tensor.

Creates a new op node to compute the elementwise negation of a
symbolic tensor and adds it to the graph, returning the symbolic result.

Args:
    value: The symbolic tensor to use as the input to the negation
        computation.

Returns:
    The result will have:
        - element type ``bool``, true if the element at a given position
            is plus or minus infinity, false otherwise
        - the same shape as the input value.

Raises:
    Error: If the symbol doesn't represent a tensor value.
"""
