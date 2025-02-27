# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Elementwise ops."""

from max.mlir.dialects import rmo

from .. import dtype_promotion
from ..graph import Graph
from ..value import TensorValue, TensorValueLike

# ===----------------------------------------------------------------------=== #
# Binary Ops
# ===----------------------------------------------------------------------=== #
# Note: Keep alphabetized.


def _elementwise_binary(op):
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
        input_type = TensorType(dtype=DType.float32, shape=(2,))

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
div = _elementwise_binary(rmo.div)
"""
Divides two symbolic tensors.

Creates a new op node to compute the division of two symbol tensor values
and adds it to the graph, returning the symbolic result.

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
max = _elementwise_binary(rmo.max)
"""
Computes the elementwise maximum of two symbolic tensors.

Creates a new op node to compute the maximum of two symbol tensor values
and adds it to the graph, returning the symbolic result.

.. code-block:: python

    def maximum_graph():
        input_type = TensorType(dtype=DType.float32, shape=(2,))

        with Graph("maximum_graph", input_types=(input_type, input_type)) as graph:
            out = ops.max(graph.inputs[0], graph.inputs[1])
            graph.output(out)

-
    - If ``lhs`` and ``rhs`` have different dtypes, they will be promoted
      according to the dtype promotion rules before the operation.
    - If ``lhs`` and ``rhs`` have different shapes, they will be broadcast to
      the same shape according to broadcasting rules` before the operation.

Args:
    lhs: The symbol to use as left side of the addition.
    rhs: The symbol to use as right side of the addition.

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
min = _elementwise_binary(rmo.min)
"""
Computes the elementwise minimum of two symbolic tensors.

Creates a new op node to compute the minimum of two symbol tensor values
and adds it to the graph, returning the symbolic result.

.. code-block:: python

    def min_graph():
        input_type = TensorType(dtype=DType.float32, shape=(2,))

        with Graph("min_graph", input_types=(input_type, input_type)) as graph:
            out = ops.min(graph.inputs[0], graph.inputs[1])
            graph.output(out)

-
    - If ``lhs`` and ``rhs`` have different dtypes, they will be promoted
      according to the dtype promotion rules before the operation.
    - If ``lhs`` and ``rhs`` have different shapes, they will be broadcast to
      the same shape according to broadcasting rules` before the operation.

Args:
    lhs: The symbol to use as left side of the addition.
    rhs: The symbol to use as right side of the addition.

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
mod = _elementwise_binary(rmo.mod)
"""
Computes the elementwise maximum of two symbolic tensors.

Creates a new op node to compute the maximum of two symbol tensor values
and adds it to the graph, returning the symbolic result.

-
    - If ``lhs`` and ``rhs`` have different dtypes, they will be promoted
      according to the dtype promotion rules before the operation.
    - If ``lhs`` and ``rhs`` have different shapes, they will be broadcast to
      the same shape according to broadcasting rules` before the operation.

Args:
    lhs: The symbol to use as left side of the addition.
    rhs: The symbol to use as right side of the addition.

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
    lhs: The symbol to use as left side of the addition.
    rhs: The symbol to use as right side of the addition.

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
    lhs: The symbol to use as left side of the addition.
    rhs: The symbol to use as right side of the addition.

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
sub = _elementwise_binary(rmo.sub)
"""
Computes the elementwise subtraction of two symbolic tensors.

Creates a new op node to compute the subtraction of two symbol tensor values
and adds it to the graph, returning the symbolic result.

.. code-block:: python

    def sub_graph():
        input_type = TensorType(dtype=DType.float32, shape=(2,))

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
    lhs: The symbol to use as left side of the addition.
    rhs: The symbol to use as right side of the addition.

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
equal = _elementwise_binary(rmo.equal)
"""
Computes the elementwise equality comparison between two symbolic tensors.

Creates a new op node to compute the equality comparison of two symbol
tensor values and adds it to the graph, returning the symbolic result.

.. code-block:: python

    def equal_graph():
        input_type = TensorType(dtype=DType.float32, shape=(3,))

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
    lhs: The symbol to use as left side of the addition.
    rhs: The symbol to use as right side of the addition.

Returns:
    A symbolic tensor value representing the output of the addition.
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
        input_type = TensorType(dtype=DType.float32, shape=(2,))

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
    lhs: The symbol to use as left side of the addition.
    rhs: The symbol to use as right side of the addition.

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
    lhs: The symbol to use as left side of the addition.
    rhs: The symbol to use as right side of the addition.

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
not_equal = _elementwise_binary(rmo.not_equal)
"""
Computes the elementwise inequality comparison between two symbolic tensors.

Creates a new op node to compute the inequality comparison of two symbol
tensor values and adds it to the graph, returning the symbolic result.


.. code-block:: python

    def not_equal_graph():
        input_type = TensorType(dtype=DType.float32, shape=(2,))

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
    lhs: The symbol to use as left side of the addition.
    rhs: The symbol to use as right side of the addition.

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
# Unary  Ops
# ===----------------------------------------------------------------------=== #
# Note: Keep alphabetized.


def _elementwise_unary(op):
    def elementwise_op(x: TensorValueLike) -> TensorValue:
        x = dtype_promotion._restrict_to_strong_dtypes(x)
        return Graph.current._add_op(op, x._mlir_value.type, x)[0].tensor

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
        input_type = TensorType(dtype=DType.float32, shape=(2,))

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
    A new symbolic tensor value representing the output of the absolute
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
    A new symbolic tensor value representing the output of the absolute
    value computation.

Raises:
    Error: If the symbol doesn't represent a tensor value.
"""
gelu = _elementwise_unary(rmo.mo_gelu)
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
    A new symbolic tensor value representing the output of the absolute
        value computation.

Raises:
    Error: If the symbol doesn't represent a tensor value.
"""
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
    A new symbolic tensor value representing the output of the absolute
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
    A new symbolic tensor value representing the output of the absolute
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
    A new symbolic tensor value representing the output of the absolute
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
    A new symbolic tensor value representing the output of the absolute
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
    A new symbolic tensor value representing the output of the absolute
        value computation.

Raises:
    Error: If the symbol doesn't represent a tensor value.
"""


def silu(x: TensorValue):
    """
    Computes the elementwise silu of a symbolic tensor.

    Creates a new op node to compute the elementwise silu of a
    symbolic tensor and adds it to the graph, returning the symbolic result.

    ``silu`` is defined as ``silu(x) = x * sigmoid(x)``.

    Args:
        value: The symbolic tensor to use as the input to the silu
            computation.

    Returns:
        A new symbolic tensor value representing the output of the absolute
            value computation.

    Raises:
        Error: If the symbol doesn't represent a tensor value.
    """
    return mul(x, sigmoid(x))


def gelu_quick(x: TensorValue):
    """
    Computes the elementwise quick gelu of a symbolic tensor.

    Creates a new op node to compute the elementwise quick gelu of a
    symbolic tensor and adds it to the graph, returning the symbolic result.

    ``quick gelu`` is defined as ``gelu_quick(x) = sigmoid(1.702 * x) * x``.

    See github.com/hendrycks/GELUs/blob/master/README.md.

    Args:
        value: The symbolic tensor to use as the input to the quick gelu
            computation.

    Returns:
        A new symbolic tensor value representing the output of the absolute
            value computation.

    Raises:
        Error: If the symbol doesn't represent a tensor value.
    """
    QUICK_GELU_SCALING_FACTOR = 1.702
    return x * sigmoid(QUICK_GELU_SCALING_FACTOR * x)


softmax = _elementwise_unary(rmo.mo_softmax)
"""
Computes the elementwise softmax of a symbolic tensor.

Creates a new op node to compute the elementwise softmax of a
symbolic tensor and adds it to the graph, returning the symbolic result.

Args:
    value: The symbolic tensor to use as the input to the softmax
        computation.

Returns:
    A new symbolic tensor value representing the output of the absolute
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
    A new symbolic tensor value representing the output of the absolute
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
    A new symbolic tensor value representing the output of the absolute
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
        input_type = TensorType(dtype=DType.float32, shape=(4,))

        with Graph("round_graph_example", input_types=(input_type,)) as graph:
            x = graph.inputs[0]
            out = ops.round(x)
            graph.output(out)

Args:
    value: The symbolic tensor to use as the input to the round
           computation. If it's not a floating-point DType, an exception will be
           raised.

Returns:
    A new symbolic tensor value representing the output of the absolute
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
    A new symbolic tensor value representing the output of the absolute
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
    A new symbolic tensor value representing the output of the absolute
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
    A new symbolic tensor value representing the output of the absolute
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
    A new symbolic tensor value representing the output of the absolute
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
    A new symbolic tensor value representing the output of the absolute
        value computation.

Raises:
    Error: If the symbol doesn't represent a tensor value.
"""

is_nan = _elementwise_unary(rmo.mo_is_nan)
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

is_inf = _elementwise_unary(rmo.mo_is_inf)
"""
Computes the elementwise is_inf of a symbolic tensor.

Creates a new op node to compute the elementwise is_inf of a
symbolic tensor and adds it to the graph, returning the symbolic result.

Args:
    value: The symbolic tensor to use as the input to the is_inf
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
Computes the elementwise negate of a symbolic tensor.

Creates a new op node to compute the elementwise negate of a
symbolic tensor and adds it to the graph, returning the symbolic result.

Args:
    value: The symbolic tensor to use as the input to the negate
        computation.

Returns:
    The result will have:
        - element type ``bool``, true if the element at a given position
            is plus or minus infinity, false otherwise
        - the same shape as the input value.

Raises:
    Error: If the symbol doesn't represent a tensor value.
"""
