# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Elementwise operations on graph tensor `Symbol`s.

Operations in this library are split into two main kinds:

*Unary operations*

Elementwise-unary-operations all have the following properties:
- They operate on a single symbolic tensor value of any shape
- Their output is a single symbolic tensor value with the same shape as their
    input
- The computation they represent will be itemwise-independent, in other words
    the output value in any position of the output tensor at computation time
    will depend only on the input value at that same position, and no others.

*Binary operations*

Elementwise-binary-operations all have the following properties:
- They operate on two symbolic tensor values, a `left` value and a `right`
    value.
- The input tensor types must be compatible according to the
    _elementwise_broadcast()` broadcasting rules. `broadcasting` documentation
    for more details.
- If the input tensor types have different element types, they will each
    be "promoted" to some dtype according to `type_promotion.promote()`
    _before_ executing the operation. This may involve a cast that changes
    the representation (including precision) of the data values.
    See the `type_promotion` documentation for more details.
- Their output is a single symbolic tensor value with
    - dtype depending on the op and the _promoted_ dtype, ie. `promote(lhs, rhs)`
    - shape equal to the result of `_elementwise_broadcast(lhs, rhs)`
- The computation they represent will be itemwise-independent, in other words
    _after broadcasting_ the input values to the same shape, the output value in
    any position of the output tensor at computation time will depend only on
    the input position at the two broadcast input values at that same
    position, and no others.
"""

from math import max as math_max

from max.graph.type import Dim, ElementType, MOTensor
from max.graph.type_promotion import implicit_cast_type, implicit_cast


# ===----------------------------------------------------------------------=== #
# Helpers
# ===----------------------------------------------------------------------=== #


def _elementwise_broadcast(lhs: Symbol, rhs: Symbol) -> SymbolTuple:
    var g = lhs.graph()
    var lhs_type = lhs.tensor_type()
    var rhs_type = rhs.tensor_type()

    if lhs_type == rhs_type and lhs_type.is_static():
        return (lhs, rhs)

    var lhs_rank = lhs_type.rank()
    var rhs_rank = rhs_type.rank()
    var bcast_rank = math_max(lhs_rank, rhs_rank)

    var lhs_shape = shape_of(lhs)
    var rhs_shape = shape_of(rhs)
    var broadcast_shape = g.op(
        "mo.broadcast_shape",
        (lhs_shape, rhs_shape),
        MOTensor(DType.int64, bcast_rank),
    )

    # This follows NumPy broadcasting semantics:
    #   1. The smaller shape is filled with 1 from the left
    #   2. Dimensions are promoted by the rule 1 -> N -> dynamic
    # TODO: Raise error if static dumensions don't match and can't be promoted.
    var broadcast_dims = List[Dim]()
    var larger = lhs_type if lhs_rank > rhs_rank else rhs_type
    var smaller = rhs_type if lhs_rank > rhs_rank else lhs_type
    var offset = larger.rank() - smaller.rank()
    for i in range(offset):
        broadcast_dims.append(larger.dims[i])
    for i in range(offset, bcast_rank):
        var d1 = larger.dims[i]
        var d2 = smaller.dims[i - offset]
        broadcast_dims.append(
            d1 if d1 == d2 or d2 == 1 else (d2 if d1 == 1 else Dim.dynamic())
        )

    var broadcast_lhs = g.op(
        "mo.broadcast_to",
        (lhs, broadcast_shape),
        MOTensor(lhs_type.dtype, broadcast_dims),
    )
    var broadcast_rhs = g.op(
        "mo.broadcast_to",
        (rhs, broadcast_shape),
        MOTensor(rhs_type.dtype, broadcast_dims),
    )
    return (broadcast_lhs, broadcast_rhs)


# ===----------------------------------------------------------------------=== #
# Binary Ops
# ===----------------------------------------------------------------------=== #
# Note: Keep alphabetized.


def _binary_op[op_name: StringLiteral](lhs: Symbol, rhs: Symbol) -> Symbol:
    var broadcast_operands = _elementwise_broadcast(lhs, rhs)
    var operands = implicit_cast(broadcast_operands[0], broadcast_operands[1])
    return lhs.graph().op(op_name, operands, operands[0].tensor_type())


def add(lhs: Symbol, rhs: Symbol) -> Symbol:
    """Adds two symbolic tensors.

    Creates a new op node to compute the addition of two symbol tensor values
    and adds it to the graph, returning the symbolic result.

    - If `lhs` and `rhs` have different dtypes, they will be promoted
        according to `type_promotion` before the operation.
    - If `lhs` and `rhs` have different shapes, they will be broadcast
        to the same shape according to `_elementwise_broadcast()` before
        the operation.

    Args:
        lhs: The symbol to use as left side of the addition.
        rhs: The symbol to use as left side of the addition.

    Returns:
        A symbolic tensor value representing the output of the addition.
        The result will have:
            - the same dtype as the type-promotion of the two input dtypes
            - the same shape as the broadcast of the two input shapes.

    Raises:
        - If the input values' shapes are not compatible for broadcasting.
            See `_elementwise_broadcast()` for more.
        - If one of the input values has an unsupported dtype.
        - If the two symbols are parts of different graphs.
    """
    return _binary_op["mo.add"](lhs, rhs)


def div(lhs: Symbol, rhs: Symbol) -> Symbol:
    """Divides two symbolic tensors.

    Creates a new op node to compute the division of two symbol tensor values
    and adds it to the graph, returning the symbolic result.

    - If `lhs` and `rhs` have different dtypes, they will be promoted
        according to `type_promotion` before the operation.
    - If `lhs` and `rhs` have different shapes, they will be broadcast
        to the same shape according to `_elementwise_broadcast()` before
        the operation.

    Args:
        lhs: The symbol to use as left side of the division.
        rhs: The symbol to use as left side of the division.

    Returns:
        A symbolic tensor value representing the output of the division.
        The result will have:
            - the same dtype as the type-promotion of the two input dtypes
            - the same shape as the broadcast of the two input shapes.

    Raises:
        - If the input values' shapes are not compatible for broadcasting.
            See `_elementwise_broadcast()` for more.
        - If one of the input values has an unsupported dtype.
        - If the two symbols are parts of different graphs.
    """
    return _binary_op["mo.div"](lhs, rhs)


def max(lhs: Symbol, rhs: Symbol) -> Symbol:
    """Computes the elementwise maximum of two symbolic tensors.

    Creates a new op node to compute the maximum of two symbol tensor values
    and adds it to the graph, returning the symbolic result.

    - If `lhs` and `rhs` have different dtypes, they will be promoted
        according to `type_promotion` before the operation.
    - If `lhs` and `rhs` have different shapes, they will be broadcast
        to the same shape according to `_elementwise_broadcast()` before
        the operation.

    Args:
        lhs: The symbol to use as left side of the maximum.
        rhs: The symbol to use as left side of the maximum.

    Returns:
        A symbolic tensor value representing the output of the maximum.
        The result will have:
            - the same dtype as the type-promotion of the two input dtypes
            - the same shape as the broadcast of the two input shapes.

    Raises:
        - If the input values' shapes are not compatible for broadcasting.
            See `_elementwise_broadcast()` for more.
        - If one of the input values has an unsupported dtype.
        - If the two symbols are parts of different graphs.
    """
    return _binary_op["mo.max"](lhs, rhs)


def min(lhs: Symbol, rhs: Symbol) -> Symbol:
    """Computes the elementwise minimum of two symbolic tensors.

    Creates a new op node to compute the minimum of two symbol tensor values
    and adds it to the graph, returning the symbolic result.

    - If `lhs` and `rhs` have different dtypes, they will be promoted
        according to `type_promotion` before the operation.
    - If `lhs` and `rhs` have different shapes, they will be broadcast
        to the same shape according to `_elementwise_broadcast()` before
        the operation.

    Args:
        lhs: The symbol to use as left side of the minimum.
        rhs: The symbol to use as left side of the minimum.

    Returns:
        A symbolic tensor value representing the output of the minimum.
        The result will have:
            - the same dtype as the type-promotion of the two input dtypes
            - the same shape as the broadcast of the two input shapes.

    Raises:
        - If the input values' shapes are not compatible for broadcasting.
            See `_elementwise_broadcast()` for more.
        - If one of the input values has an unsupported dtype.
        - If the two symbols are parts of different graphs.
    """
    return _binary_op["mo.min"](lhs, rhs)


def mod(lhs: Symbol, rhs: Symbol) -> Symbol:
    """Computes the elementwise maximum of two symbolic tensors.

    Creates a new op node to compute the maximum of two symbol tensor values
    and adds it to the graph, returning the symbolic result.

    - If `lhs` and `rhs` have different dtypes, they will be promoted
        according to `type_promotion` before the operation.
    - If `lhs` and `rhs` have different shapes, they will be broadcast
        to the same shape according to `_elementwise_broadcast()` before
        the operation.

    Args:
        lhs: The symbol to use as left side of the maximum.
        rhs: The symbol to use as left side of the maximum.

    Returns:
        A symbolic tensor value representing the output of the maximum.
        The result will have:
            - the same dtype as the type-promotion of the two input dtypes
            - the same shape as the broadcast of the two input shapes.

    Raises:
        - If the input values' shapes are not compatible for broadcasting.
            See `_elementwise_broadcast()` for more.
        - If one of the input values has an unsupported dtype.
        - If the two symbols are parts of different graphs.
    """
    return _binary_op["mo.mod"](lhs, rhs)


def mul(lhs: Symbol, rhs: Symbol) -> Symbol:
    """Computes the elementwise multiplication of two symbolic tensors.

    Creates a new op node to compute the multiplication of two symbol tensor values
    and adds it to the graph, returning the symbolic result.

    - If `lhs` and `rhs` have different dtypes, they will be promoted
        according to `type_promotion` before the operation.
    - If `lhs` and `rhs` have different shapes, they will be broadcast
        to the same shape according to `_elementwise_broadcast()` before
        the operation.

    Args:
        lhs: The symbol to use as left side of the multiplication.
        rhs: The symbol to use as left side of the multiplication.

    Returns:
        A symbolic tensor value representing the output of the multiplication.
        The result will have:
            - the same dtype as the type-promotion of the two input dtypes
            - the same shape as the broadcast of the two input shapes.

    Raises:
        - If the input values' shapes are not compatible for broadcasting.
            See `_elementwise_broadcast()` for more.
        - If one of the input values has an unsupported dtype.
        - If the two symbols are parts of different graphs.
    """
    return _binary_op["mo.mul"](lhs, rhs)


def pow(lhs: Symbol, rhs: Symbol) -> Symbol:
    """Computes the elementwise exponentiation of two symbolic tensors.

    Creates a new op node to compute the exponentiation of two symbol tensor values
    and adds it to the graph, returning the symbolic result.

    - If `lhs` and `rhs` have different dtypes, they will be promoted
        according to `type_promotion` before the operation.
    - If `lhs` and `rhs` have different shapes, they will be broadcast
        to the same shape according to `_elementwise_broadcast()` before
        the operation.

    Args:
        lhs: The symbol to use as left side of the exponentiation.
        rhs: The symbol to use as left side of the exponentiation.

    Returns:
        A symbolic tensor value representing the output of the exponentiation.
        The result will have:
            - the same dtype as the type-promotion of the two input dtypes
            - the same shape as the broadcast of the two input shapes.

    Raises:
        - If the input values' shapes are not compatible for broadcasting.
            See `_elementwise_broadcast()` for more.
        - If one of the input values has an unsupported dtype.
        - If the two symbols are parts of different graphs.
    """
    return _binary_op["mo.pow"](lhs, rhs)


def sub(lhs: Symbol, rhs: Symbol) -> Symbol:
    """Computes the elementwise subtraction of two symbolic tensors.

    Creates a new op node to compute the subtraction of two symbol tensor values
    and adds it to the graph, returning the symbolic result.

    - If `lhs` and `rhs` have different dtypes, they will be promoted
        according to `type_promotion` before the operation.
    - If `lhs` and `rhs` have different shapes, they will be broadcast
        to the same shape according to `_elementwise_broadcast()` before
        the operation.

    Args:
        lhs: The symbol to use as left side of the subtraction.
        rhs: The symbol to use as left side of the subtraction.

    Returns:
        A symbolic tensor value representing the output of the subtraction.
        The result will have:
            - the same dtype as the type-promotion of the two input dtypes
            - the same shape as the broadcast of the two input shapes.

    Raises:
        - If the input values' shapes are not compatible for broadcasting.
            See `_elementwise_broadcast()` for more.
        - If one of the input values has an unsupported dtype.
        - If the two symbols are parts of different graphs.
    """
    return _binary_op["mo.sub"](lhs, rhs)


def _binary_comparison_op[
    op_name: StringLiteral
](lhs: Symbol, rhs: Symbol) -> Symbol:
    var operands = _elementwise_broadcast(lhs, rhs)
    var result_type = operands[0].tensor_type().cast(DType.bool)
    return lhs.graph().op(op_name, operands, result_type)


def equal(lhs: Symbol, rhs: Symbol) -> Symbol:
    """Computes the elementwise equality comparison between two symbolic tensors.

    Creates a new op node to compute the equality comparison of two symbol
    tensor values and adds it to the graph, returning the symbolic result.

    - If `lhs` and `rhs` have different dtypes, they will be promoted
        according to `type_promotion` before the operation.
    - If `lhs` and `rhs` have different shapes, they will be broadcast
        to the same shape according to `_elementwise_broadcast()` before
        the operation.

    Args:
        lhs: The symbol to use as left side of the equality comparison.
        rhs: The symbol to use as left side of the equality comparison.

    Returns:
        A symbolic tensor value representing the output of the equality
        comparison.
        The result will have:
            - element type `bool`, true if the left-hand-side value at a given
                position is equal to the right-hand-side value at that same
                position, and false otherwise.
            - the same shape as the broadcast of the two input shapes.

    Raises:
        - If the input values' shapes are not compatible for broadcasting.
            See `_elementwise_broadcast()` for more.
        - If one of the input values has an unsupported dtype.
        - If the two symbols are parts of different graphs.
    """
    return _binary_comparison_op["mo.equal"](lhs, rhs)


def greater(lhs: Symbol, rhs: Symbol) -> Symbol:
    """Computes the elementwise greater than comparison between two symbolictensors.

    Creates a new op node to compute the greater than comparison of two symbol
    tensor values and adds it to the graph, returning the symbolic result.

    - If `lhs` and `rhs` have different dtypes, they will be promoted
        according to `type_promotion` before the operation.
    - If `lhs` and `rhs` have different shapes, they will be broadcast
        to the same shape according to `_elementwise_broadcast()` before
        the operation.

    Args:
        lhs: The symbol to use as left side of the greater than comparison.
        rhs: The symbol to use as left side of the greater than comparison.

    Returns:
        A symbolic tensor value representing the output of the greater than
        comparison.
        The result will have:
            - element type `bool`, true if the left-hand-side value at
                a given position is strictly greater than (not equal to)
                the right-hand-side at that same position, and false otherwise.
            - the same shape as the broadcast of the two input shapes.

    Raises:
        - If the input values' shapes are not compatible for broadcasting.
            See `_elementwise_broadcast()` for more.
        - If one of the input values has an unsupported dtype.
        - If the two symbols are parts of different graphs.
    """
    return _binary_comparison_op["mo.greater"](lhs, rhs)


def greater_equal(lhs: Symbol, rhs: Symbol) -> Symbol:
    """Computes the elementwise greater-or-equal comparison between two symbolic tensors.

    Creates a new op node to compute the equality comparison of two symbol
    tensor values and adds it to the graph, returning the symbolic result.

    - If `lhs` and `rhs` have different dtypes, they will be promoted
        according to `type_promotion` before the operation.
    - If `lhs` and `rhs` have different shapes, they will be broadcast
        to the same shape according to `_elementwise_broadcast()` before
        the operation.

    Args:
        lhs: The symbol to use as left side of the equality comparison.
        rhs: The symbol to use as left side of the equality comparison.

    Returns:
        A symbolic tensor value representing the output of the equality
        comparison.
        The result will have:
            - element type `bool`, true if the left-hand-side value at
                a given position is greater than or equal to the right hand
                side at that same position, and false otherwise.
            - the same shape as the broadcast of the two input shapes.

    Raises:
        - If the input values' shapes are not compatible for broadcasting.
            See `_elementwise_broadcast()` for more.
        - If one of the input values has an unsupported dtype.
        - If the two symbols are parts of different graphs.
    """
    return _binary_comparison_op["mo.greater_equal"](lhs, rhs)


def not_equal(lhs: Symbol, rhs: Symbol) -> Symbol:
    """Computes the elementwise inequality comparison between two symbolic tensors.

    Creates a new op node to compute the inequality comparison of two symbol
    tensor values and adds it to the graph, returning the symbolic result.

    - If `lhs` and `rhs` have different dtypes, they will be promoted
        according to `type_promotion` before the operation.
    - If `lhs` and `rhs` have different shapes, they will be broadcast
        to the same shape according to `_elementwise_broadcast()` before
        the operation.

    Args:
        lhs: The symbol to use as left side of the inequality comparison.
        rhs: The symbol to use as left side of the inequality comparison.

    Returns:
        A symbolic tensor value representing the output of the inequality
        comparison.
        The result will have:
            - element type `bool`, true if the elements at
                a given position are _not_ equal and false otherwise.
            - the same shape as the broadcast of the two input shapes.

    Raises:
        - If the input values' shapes are not compatible for broadcasting.
            See `_elementwise_broadcast()` for more.
        - If one of the input values has an unsupported dtype.
        - If the two symbols are parts of different graphs.
    """
    return _binary_comparison_op["mo.not_equal"](lhs, rhs)


# ===----------------------------------------------------------------------=== #
# Unary Ops
# ===----------------------------------------------------------------------=== #
# Note: Keep alphabetized.


def _unary_op[op_name: StringLiteral](value: Symbol) -> Symbol:
    return value.graph().op(op_name, value, value.tensor_type())


def _unary_float_op[op_name: StringLiteral](value: Symbol) -> Symbol:
    var dtype = value.tensor_type().dtype.dtype
    # TODO: Don't cast implicitly here, we don't know what to cast to.
    var float_v = cast(value, implicit_cast_type(dtype, DType.float16))
    return value.graph().op(op_name, float_v, float_v.tensor_type())


def _unary_comparison_op[op_name: StringLiteral](value: Symbol) -> Symbol:
    var result_type = value.tensor_type().cast(DType.bool)
    return value.graph().op(op_name, value, result_type)


def abs(value: Symbol) -> Symbol:
    """Computes the elementwise absolute value of a symbolic tensor.

    Creates a new op node to compute the elementwise absolute value of a
    symbolic tensor and adds it to the graph, returning the symbolic result.

    Args:
        value: The symbolic tensor to use as the input to the absolute value
            computation.

    Returns:
        A new symbolic tensor value representing the output of the absolute
            value computation.

    Raises:
        If the symbol doesn't represent a tensor value.
    """
    return _unary_op["mo.abs"](value)


def exp(value: Symbol) -> Symbol:
    """Computes the elementwise exp function of a symbolic tensor.

    Creates a new op node to compute the elementwise exp function of a
    symbolic tensor and adds it to the graph, returning the symbolic result.

    `exp` is defined as `exp(x) = e^x`, where `e` is
    [Euler's number](https://en.wikipedia.org/wiki/E_(mathematical_constant)).

    Args:
        value: The symbolic tensor to use as the input to the exp function
            computation.

    Returns:
        A new symbolic tensor value representing the output of the absolute
            value computation.

    Raises:
        If the symbol doesn't represent a tensor value.
    """
    return _unary_op["mo.exp"](value)


def erf(value: Symbol) -> Symbol:
    """Computes the elementwise error function of a symbolic tensor.

    Creates a new op node to compute the elementwise error function of a
    symbolic tensor and adds it to the graph, returning the symbolic result.

    The error function `erf` is defined as the probability that a randomly
    sampled normal distribution falls within a given range. See
    [Error function](https://en.wikipedia.org/wiki/Error_function) for more
    details.

    Args:
        value: The symbolic tensor to use as the input to the error function
            computation.

    Returns:
        A new symbolic tensor value representing the output of the absolute
            value computation.

    Raises:
        If the symbol doesn't represent a tensor value.
    """
    return _unary_op["mo.erf"](value)


def gelu(value: Symbol) -> Symbol:
    """Computes the elementwise gelu function of a symbolic tensor.

    Creates a new op node to compute the elementwise gelu function of a
    symbolic tensor and adds it to the graph, returning the symbolic result.

    `gelu` is defined as $$gelu(x) = x \\Phi(x)$$ where $$\\Phi$$ is the
    [cumulative distribution function of the Gaussian distribution](https://en.wikipedia.org/wiki/Normal_distribution#Cumulative_distribution_function).
    See [the paper](https://arxiv.org/pdf/1606.08415.pdf) for more details.

    Args:
        value: The symbolic tensor to use as the input to the gelu function
            computation.

    Returns:
        A new symbolic tensor value representing the output of the absolute
            value computation.

    Raises:
        If the symbol doesn't represent a tensor value.
    """
    return _unary_op["mo.gelu"](value)


def log(value: Symbol) -> Symbol:
    """Computes the elementwise natural logarithm of a symbolic tensor.

    Creates a new op node to compute the elementwise natural logarithm of a
    symbolic tensor and adds it to the graph, returning the symbolic result.

    The [natural logarithm](https://en.wikipedia.org/wiki/Natural_logarithm)
    function `log` is defined as the inverse of the exponential function `exp()`,
    ie. it computes the value `y` in the equation `x = e^y` where `e` is
    [Euler's number](https://en.wikipedia.org/wiki/E_(mathematical_constant)).

    `log(x)` is undefined for `x <= 0` for real numbers. Complex numbers
    are currently unsupported.

    Args:
        value: The symbolic tensor to use as the input to the natural logarithm
            computation.

    Returns:
        A new symbolic tensor value representing the output of the absolute
            value computation.

    Raises:
        If the symbol doesn't represent a tensor value.
    """
    return _unary_op["mo.log"](value)


def log1p(value: Symbol) -> Symbol:
    """Computes the elementwise logarithm of 1 plus a symbolic tensor.

    Creates a new op node to compute the elementwise log1p of a
    symbolic tensor and adds it to the graph, returning the symbolic result.

    The `log1p` function is defined as `log1p(x) = log(1 + x)`, where `log()`
    is the [natural logarithm](https://en.wikipedia.org/wiki/Natural_logarithm).

    Using `log1p(x)` rather than computing `log(1 + x)` can give greater
    numerical precision results.

    `log(x)` is undefined for `x <= 0` for real numbers. Complex numbers
    are currently unsupported.

    Args:
        value: The symbolic tensor to use as the input to the log1p
            computation.

    Returns:
        A new symbolic tensor value representing the output of the absolute
            value computation.

    Raises:
        If the symbol doesn't represent a tensor value.
    """
    return _unary_op["mo.log1p"](value)


def logsoftmax(value: Symbol) -> Symbol:
    """Computes the elementwise logsoftmax of a symbolic tensor.

    Creates a new op node to compute the elementwise logsoftmax of a
    symbolic tensor and adds it to the graph, returning the symbolic result.

    Args:
        value: The symbolic tensor to use as the input to the logsoftmax
            computation.

    Returns:
        A new symbolic tensor value representing the output of the absolute
            value computation.

    Raises:
        If the symbol doesn't represent a tensor value.
    """
    return _unary_op["mo.logsoftmax"](value)


def relu(value: Symbol) -> Symbol:
    """Computes the elementwise relu of a symbolic tensor.

    Creates a new op node to compute the elementwise relu of a
    symbolic tensor and adds it to the graph, returning the symbolic result.

    Args:
        value: The symbolic tensor to use as the input to the relu
            computation.

    Returns:
        A new symbolic tensor value representing the output of the absolute
            value computation.

    Raises:
        If the symbol doesn't represent a tensor value.
    """
    return _unary_op["mo.relu"](value)


def sigmoid(value: Symbol) -> Symbol:
    """Computes the elementwise sigmoid of a symbolic tensor.

    Creates a new op node to compute the elementwise sigmoid of a
    symbolic tensor and adds it to the graph, returning the symbolic result.

    Args:
        value: The symbolic tensor to use as the input to the sigmoid
            computation.

    Returns:
        A new symbolic tensor value representing the output of the absolute
            value computation.

    Raises:
        If the symbol doesn't represent a tensor value.
    """
    return _unary_op["mo.sigmoid"](value)


def silu(value: Symbol) -> Symbol:
    """Computes the elementwise silu of a symbolic tensor.

    Creates a new op node to compute the elementwise silu of a
    symbolic tensor and adds it to the graph, returning the symbolic result.

    `silu` is defined as `silu(x) = x * [sigmoid](/engine/reference/mojo/graph/ops/elementwise.html#sigmoid)(x)`.

    Args:
        value: The symbolic tensor to use as the input to the silu
            computation.

    Returns:
        A new symbolic tensor value representing the output of the absolute
            value computation.

    Raises:
        If the symbol doesn't represent a tensor value.
    """
    return mul(value, sigmoid(value))


def softmax(value: Symbol) -> Symbol:
    """Computes the elementwise softmax of a symbolic tensor.

    Creates a new op node to compute the elementwise softmax of a
    symbolic tensor and adds it to the graph, returning the symbolic result.

    Args:
        value: The symbolic tensor to use as the input to the softmax
            computation.

    Returns:
        A new symbolic tensor value representing the output of the absolute
            value computation.

    Raises:
        If the symbol doesn't represent a tensor value.
    """
    return _unary_op["mo.softmax"](value)


def cos(value: Symbol) -> Symbol:
    """Computes the elementwise cosine of a symbolic tensor.

    Creates a new op node to compute the elementwise cosine of a
    symbolic tensor and adds it to the graph, returning the symbolic result.

    Args:
        value: The symbolic tensor to use as the input to the trunc
            computation. If it's not a floating-point DType, it will be promoted
            to one according to [type promotion rules](/engine/reference/mojo/graph/type_promotion)
            before computation.

    Returns:
        A new symbolic tensor value representing the output of the absolute
            value computation.

    Raises:
        If the symbol doesn't represent a tensor value.
    """
    return _unary_float_op["mo.cos"](value)


def floor(value: Symbol) -> Symbol:
    """Computes the elementwise floor of a symbolic tensor.

    Creates a new op node to compute the elementwise floor of a
    symbolic tensor and adds it to the graph, returning the symbolic result.

    Args:
        value: The symbolic tensor to use as the input to the trunc
            computation. If it's not a floating-point DType, it will be promoted
            to one according to [type promotion rules](/engine/reference/mojo/graph/type_promotion)
            before computation.

    Returns:
        A new symbolic tensor value representing the output of the absolute
            value computation.

    Raises:
        If the symbol doesn't represent a tensor value.
    """
    return _unary_float_op["mo.floor"](value)


def round(value: Symbol) -> Symbol:
    """Computes the elementwise round of a symbolic tensor.

    Creates a new op node to compute the elementwise round of a
    symbolic tensor and adds it to the graph, returning the symbolic result.

    Args:
        value: The symbolic tensor to use as the input to the trunc
            computation. If it's not a floating-point DType, it will be promoted
            to one according to [type promotion rules](/engine/reference/mojo/graph/type_promotion)
            before computation.

    Returns:
        A new symbolic tensor value representing the output of the absolute
            value computation.

    Raises:
        If the symbol doesn't represent a tensor value.
    """
    return _unary_float_op["mo.round"](value)


def roundeven(value: Symbol) -> Symbol:
    """Computes the elementwise roundeven of a symbolic tensor.

    Creates a new op node to compute the elementwise roundeven of a
    symbolic tensor and adds it to the graph, returning the symbolic result.

    Args:
        value: The symbolic tensor to use as the input to the trunc
            computation. If it's not a floating-point DType, it will be promoted
            to one according to [type promotion rules](/engine/reference/mojo/graph/type_promotion)
            before computation.

    Returns:
        A new symbolic tensor value representing the output of the absolute
            value computation.

    Raises:
        If the symbol doesn't represent a tensor value.
    """
    return _unary_float_op["mo.roundeven"](value)


def rsqrt(value: Symbol) -> Symbol:
    """Computes the elementwise inverse-square-root of a symbolic tensor.

    Creates a new op node to compute the elementwise rsqrt of a
    symbolic tensor and adds it to the graph, returning the symbolic result.

    Args:
        value: The symbolic tensor to use as the input to the trunc
            computation. If it's not a floating-point DType, it will be promoted
            to one according to [type promotion rules](/engine/reference/mojo/graph/type_promotion)
            before computation.

    Returns:
        A new symbolic tensor value representing the output of the absolute
            value computation.

    Raises:
        If the symbol doesn't represent a tensor value.
    """
    return _unary_float_op["mo.rsqrt"](value)


def sqrt(value: Symbol) -> Symbol:
    """Computes the elementwise sqrt of a symbolic tensor.

    Creates a new op node to compute the elementwise sqrt of a
    symbolic tensor and adds it to the graph, returning the symbolic result.

    Args:
        value: The symbolic tensor to use as the input to the trunc
            computation. If it's not a floating-point DType, it will be promoted
            to one according to [type promotion rules](/engine/reference/mojo/graph/type_promotion)
            before computation.

    Returns:
        A new symbolic tensor value representing the output of the absolute
            value computation.

    Raises:
        If the symbol doesn't represent a tensor value.
    """
    return _unary_float_op["mo.sqrt"](value)


def sin(value: Symbol) -> Symbol:
    """Computes the elementwise sine of a symbolic tensor.

    Creates a new op node to compute the elementwise sine of a
    symbolic tensor and adds it to the graph, returning the symbolic result.

    Args:
        value: The symbolic tensor to use as the input to the trunc
            computation. If it's not a floating-point DType, it will be promoted
            to one according to [type promotion rules](/engine/reference/mojo/graph/type_promotion)
            before computation.

    Returns:
        A new symbolic tensor value representing the output of the absolute
            value computation.

    Raises:
        If the symbol doesn't represent a tensor value.
    """
    return _unary_float_op["mo.sin"](value)


def tanh(value: Symbol) -> Symbol:
    """Computes the elementwise tanh of a symbolic tensor.

    Creates a new op node to compute the elementwise tanh of a
    symbolic tensor and adds it to the graph, returning the symbolic result.

    Args:
        value: The symbolic tensor to use as the input to the trunc
            computation. If it's not a floating-point DType, it will be promoted
            to one according to [type promotion rules](/engine/reference/mojo/graph/type_promotion)
            before computation.

    Returns:
        A new symbolic tensor value representing the output of the absolute
            value computation.

    Raises:
        If the symbol doesn't represent a tensor value.
    """
    return _unary_float_op["mo.tanh"](value)


def trunc(value: Symbol) -> Symbol:
    """Computes the elementwise truncation of a symbolic tensor.

    Creates a new op node to compute the elementwise trunc of a
    symbolic tensor and adds it to the graph, returning the symbolic result.

    Args:
        value: The symbolic tensor to use as the input to the trunc
            computation. If it's not a floating-point DType, it will be promoted
            to one according to [type promotion rules](/engine/reference/mojo/graph/type_promotion)
            before computation.

    Returns:
        A new symbolic tensor value representing the output of the absolute
            value computation.

    Raises:
        If the symbol doesn't represent a tensor value.
    """
    return _unary_float_op["mo.trunc"](value)


def is_nan(value: Symbol) -> Symbol:
    """Computes the elementwise is_nan of a symbolic tensor.

    Creates a new op node to compute the elementwise is_nan of a
    symbolic tensor and adds it to the graph, returning the symbolic result.

    Args:
        value: The symbolic tensor to use as the input to the is_nan
            computation.

    Returns:
        The result will have:
            - element type `bool`, true if the element at a given position
                is NaN, false otherwise
            - the same shape as the input value.

    Raises:
        If the symbol doesn't represent a tensor value.
    """
    return _unary_comparison_op["mo.is_nan"](value)


def is_inf(value: Symbol) -> Symbol:
    """Computes the elementwise is_inf of a symbolic tensor.

    Creates a new op node to compute the elementwise is_inf of a
    symbolic tensor and adds it to the graph, returning the symbolic result.

    Args:
        value: The symbolic tensor to use as the input to the is_inf
            computation.

    Returns:
        The result will have:
            - element type `bool`, true if the element at a given position
                is plus or minus infinity, false otherwise
            - the same shape as the input value.

    Raises:
        If the symbol doesn't represent a tensor value.
    """
    return _unary_comparison_op["mo.is_inf"](value)
