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
"""Ops that perform element-wise computations/comparisons.

Operations in this module are split into either unary or binary operations.

<h3 id="unary_operations">Unary Operations</h3>

Elementwise-unary-operations all have the following properties:

- They operate on a single symbolic tensor value of any shape
- Their output is a single symbolic tensor value with the same shape as their
    input
- The computation they represent will be itemwise-independent, in other words
    the output value in any position of the output tensor at computation time
    will depend only on the input value at that same position, and no others.

<h3 id="binary_operations">Binary Operations</h3>

Elementwise-binary-operations all have the following properties:

- They operate on two symbolic tensor values, a `left` value and a `right`
    value.
- The input tensor types must be compatible according to the
    [broadcasting rules](#broadcasting_rules).
- If the input tensor types have different element types, they will each
    be _promoted_ to the same dtype according to the
    [dtype promotion rules](#dtype_promotion_rules) _before_ executing
    the operation. This may involve a cast that changes
    the representation (including precision) of the data values.
- Their output is a single symbolic tensor value with
    - dtype depending on the op and the _promoted_ dtype, ie. `promote(lhs, rhs)`.
    - shape equal to the result of `broadcast(lhs, rhs)`
- The computation they represent will be itemwise-independent, in other words
    _after broadcasting_ the input values to the same shape, the output value in
    any position of the output tensor at computation time will depend only on
    the input position at the two broadcast input values at that same
    position, and no others.

<h3 id="dtype_promotion_rules">DType Promotion Rules</h3>

The Graph API splits dtype promotion into two pieces: bit width and category.
Bit width is simply the number of bits that are needed to represent a dtype.
Category is an order hierarchy: `bool < unsigned int < signed int < float`.

A promotion candidate is calculated between two dtypes (`a` and `b`) as:
`(max(category(a), category(b)), max(bitwidth(a), bitwidth(b)))`.

An exception will be raised if a either input dtype might contain a value that is
unrepresentable by the promotion candidate (e.g `u32 -> i32` or `i32 -> f32`).

An exception will be raised if the input has the same bit width but a different format
than the promotion candidate (e.g. `f16 -> bf16` or `f32 -> tf32`).

If no exception is raised, the promotion candidate is accepted.
All inputs will be cast to the promotion candidate before the underlying operation is run.

<h3 id="broadcasting_rules">Broadcasting Rules</h3>

Given two input tensor shapes, broadcasting works as following:

1. Prepend static 1 dimensions onto the tensor with lower rank to make it so that both tensors have the same rank.
2. If a dimension is a static 1 dimension, it will broadcast to the size of the dimension in the other tensor.
3. All other dimensions will be asserted to be equivalent. If they are not, an exception will be raised.
"""

from collections import Optional
from collections.string.string_slice import StaticString

from builtin._location import __call_location, _SourceLocation

from ..error import error

# ===----------------------------------------------------------------------=== #
# Binary Ops
# ===----------------------------------------------------------------------=== #
# Note: Keep alphabetized.


def _binary_op[op_name: StaticString](lhs: Symbol, rhs: Symbol) -> Symbol:
    return lhs.graph().op(
        String(op_name),
        List[Symbol](lhs, rhs),
    )


@always_inline
def add(
    lhs: Symbol, rhs: Symbol, location: Optional[_SourceLocation] = None
) -> Symbol:
    """Adds two symbolic tensors.

    Creates a new op node to compute the addition of two symbol tensor values
    and adds it to the graph, returning the symbolic result.

    - If `lhs` and `rhs` have different dtypes, they will be promoted according
        to the [dtype promotion
        rules](/max/api/mojo/graph/ops/elementwise/#dtype_promotion_rules)
        before the operation.
    - If `lhs` and `rhs` have different shapes, they will be broadcast to the
        same shape according to [broadcasting
        rules](/max/api/mojo/graph/ops/elementwise/#broadcasting_rules)
        before the operation.

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
        - If the input values' shapes are not compatible for broadcasting.
            See `_elementwise_broadcast()` for more.
        - If one of the input values has an unsupported dtype.
        - If the two symbols are parts of different graphs.
    """
    return _op_impl["rmo.add"](lhs, rhs, location, __call_location())


@always_inline
def div(
    lhs: Symbol, rhs: Symbol, location: Optional[_SourceLocation] = None
) -> Symbol:
    """Divides two symbolic tensors.

    Creates a new op node to compute the division of two symbol tensor values
    and adds it to the graph, returning the symbolic result.

    - If `lhs` and `rhs` have different dtypes, they will be promoted according
        to the [dtype promotion
        rules](/max/api/mojo/graph/ops/elementwise/#dtype_promotion_rules)
        before the operation.
    - If `lhs` and `rhs` have different shapes, they will be broadcast to the
        same shape according to [broadcasting
        rules](/max/api/mojo/graph/ops/elementwise/#broadcasting_rules)
        before the operation.

    Args:
        lhs: The symbol to use as left side of the division.
        rhs: The symbol to use as right side of the division.
        location: An optional location for a more specific error message.

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
    return _op_impl["rmo.div"](lhs, rhs, location, __call_location())


@always_inline
def max(
    lhs: Symbol, rhs: Symbol, location: Optional[_SourceLocation] = None
) -> Symbol:
    """Computes the elementwise maximum of two symbolic tensors.

    Creates a new op node to compute the maximum of two symbol tensor values
    and adds it to the graph, returning the symbolic result.

    - If `lhs` and `rhs` have different dtypes, they will be promoted according
        to the [dtype promotion
        rules](/max/api/mojo/graph/ops/elementwise/#dtype_promotion_rules)
        before the operation.
    - If `lhs` and `rhs` have different shapes, they will be broadcast to the
        same shape according to [broadcasting
        rules](/max/api/mojo/graph/ops/elementwise/#broadcasting_rules)
        before the operation.

    Args:
        lhs: The symbol to use as left side of the maximum.
        rhs: The symbol to use as right side of the maximum.
        location: An optional location for a more specific error message.

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
    return _op_impl["rmo.max"](lhs, rhs, location, __call_location())


@always_inline
def min(
    lhs: Symbol, rhs: Symbol, location: Optional[_SourceLocation] = None
) -> Symbol:
    """Computes the elementwise minimum of two symbolic tensors.

    Creates a new op node to compute the minimum of two symbol tensor values
    and adds it to the graph, returning the symbolic result.

    - If `lhs` and `rhs` have different dtypes, they will be promoted according
        to the [dtype promotion
        rules](/max/api/mojo/graph/ops/elementwise/#dtype_promotion_rules)
        before the operation.
    - If `lhs` and `rhs` have different shapes, they will be broadcast to the
        same shape according to [broadcasting
        rules](/max/api/mojo/graph/ops/elementwise/#broadcasting_rules)
        before the operation.

    Args:
        lhs: The symbol to use as left side of the minimum.
        rhs: The symbol to use as right side of the minimum.
        location: An optional location for a more specific error message.

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
    return _op_impl["rmo.min"](lhs, rhs, location, __call_location())


@always_inline
def mod(
    lhs: Symbol, rhs: Symbol, location: Optional[_SourceLocation] = None
) -> Symbol:
    """Computes the elementwise maximum of two symbolic tensors.

    Creates a new op node to compute the maximum of two symbol tensor values
    and adds it to the graph, returning the symbolic result.

    - If `lhs` and `rhs` have different dtypes, they will be promoted according
        to the [dtype promotion
        rules](/max/api/mojo/graph/ops/elementwise/#dtype_promotion_rules)
        before the operation.
    - If `lhs` and `rhs` have different shapes, they will be broadcast to the
        same shape according to [broadcasting
        rules](/max/api/mojo/graph/ops/elementwise/#broadcasting_rules)
        before the operation.

    Args:
        lhs: The symbol to use as left side of the maximum.
        rhs: The symbol to use as right side of the maximum.
        location: An optional location for a more specific error message.

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
    return _op_impl["rmo.mod"](lhs, rhs, location, __call_location())


@always_inline
def mul(
    lhs: Symbol, rhs: Symbol, location: Optional[_SourceLocation] = None
) -> Symbol:
    """Computes the elementwise multiplication of two symbolic tensors.

    Creates a new op node to compute the multiplication of two symbol tensor values
    and adds it to the graph, returning the symbolic result.

    - If `lhs` and `rhs` have different dtypes, they will be promoted according
        to the [dtype promotion
        rules](/max/api/mojo/graph/ops/elementwise/#dtype_promotion_rules)
        before the operation.
    - If `lhs` and `rhs` have different shapes, they will be broadcast to the
        same shape according to [broadcasting
        rules](/max/api/mojo/graph/ops/elementwise/#broadcasting_rules)
        before the operation.

    Args:
        lhs: The symbol to use as left side of the multiplication.
        rhs: The symbol to use as right side of the multiplication.
        location: An optional location for a more specific error message.

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
    return _op_impl["rmo.mul"](lhs, rhs, location, __call_location())


@always_inline
def pow(
    lhs: Symbol, rhs: Symbol, location: Optional[_SourceLocation] = None
) -> Symbol:
    """Computes the elementwise exponentiation of two symbolic tensors.

    Creates a new op node to compute the exponentiation of two symbol tensor values
    and adds it to the graph, returning the symbolic result.

    - If `lhs` and `rhs` have different dtypes, they will be promoted according
        to the [dtype promotion
        rules](/max/api/mojo/graph/ops/elementwise/#dtype_promotion_rules)
        before the operation.
    - If `lhs` and `rhs` have different shapes, they will be broadcast to the
        same shape according to [broadcasting
        rules](/max/api/mojo/graph/ops/elementwise/#broadcasting_rules)
        before the operation.

    Args:
        lhs: The symbol to use as left side of the exponentiation.
        rhs: The symbol to use as right side of the exponentiation.
        location: An optional location for a more specific error message.

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
    return _op_impl["rmo.pow"](lhs, rhs, location, __call_location())


@always_inline
def sub(
    lhs: Symbol, rhs: Symbol, location: Optional[_SourceLocation] = None
) -> Symbol:
    """Computes the elementwise subtraction of two symbolic tensors.

    Creates a new op node to compute the subtraction of two symbol tensor values
    and adds it to the graph, returning the symbolic result.

    - If `lhs` and `rhs` have different dtypes, they will be promoted according
        to the [dtype promotion
        rules](/max/api/mojo/graph/ops/elementwise/#dtype_promotion_rules)
        before the operation.
    - If `lhs` and `rhs` have different shapes, they will be broadcast to the
        same shape according to [broadcasting
        rules](/max/api/mojo/graph/ops/elementwise/#broadcasting_rules)
        before the operation.

    Args:
        lhs: The symbol to use as left side of the subtraction.
        rhs: The symbol to use as right side of the subtraction.
        location: An optional location for a more specific error message.

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
    return _op_impl["rmo.sub"](lhs, rhs, location, __call_location())


@always_inline
def equal(
    lhs: Symbol, rhs: Symbol, location: Optional[_SourceLocation] = None
) -> Symbol:
    """Computes the elementwise equality comparison between two symbolic tensors.

    Creates a new op node to compute the equality comparison of two symbol
    tensor values and adds it to the graph, returning the symbolic result.

    - If `lhs` and `rhs` have different dtypes, they will be promoted according
        to the
        [dtype promotion rules](/max/api/mojo/graph/ops/elementwise/#dtype_promotion_rules)
        before the operation.
    - If `lhs` and `rhs` have different shapes, they will be broadcast to the
        same shape according to
        [broadcasting rules](/max/api/mojo/graph/ops/elementwise/#broadcasting_rules)
        before the operation.

    Args:
        lhs: The symbol to use as left side of the equality comparison.
        rhs: The symbol to use as right side of the equality comparison.
        location: An optional location for a more specific error message.

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
    return _op_impl["rmo.equal"](lhs, rhs, location, __call_location())


@always_inline
def greater(
    lhs: Symbol, rhs: Symbol, location: Optional[_SourceLocation] = None
) -> Symbol:
    """Computes the elementwise greater than comparison between two symbolic tensors.

    Creates a new op node to compute the greater than comparison of two symbol
    tensor values and adds it to the graph, returning the symbolic result.

    - If `lhs` and `rhs` have different dtypes, they will be promoted according
        to the
        [dtype promotion rules](/max/api/mojo/graph/ops/elementwise/#dtype_promotion_rules)
        before the operation.
    - If `lhs` and `rhs` have different shapes, they will be broadcast to the
        same shape according to
        [broadcasting rules](/max/api/mojo/graph/ops/elementwise/#broadcasting_rules)
        before the operation.

    Args:
        lhs: The symbol to use as left side of the greater than comparison.
        rhs: The symbol to use as right side of the greater than comparison.
        location: An optional location for a more specific error message.

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
    return _op_impl["rmo.greater"](lhs, rhs, location, __call_location())


@always_inline
def greater_equal(
    lhs: Symbol, rhs: Symbol, location: Optional[_SourceLocation] = None
) -> Symbol:
    """Computes the elementwise greater-or-equal comparison between two symbolic tensors.

    Creates a new op node to compute the equality comparison of two symbol
    tensor values and adds it to the graph, returning the symbolic result.

    - If `lhs` and `rhs` have different dtypes, they will be promoted according
        to the
        [dtype promotion rules](/max/api/mojo/graph/ops/elementwise/#dtype_promotion_rules)
        before the operation.
    - If `lhs` and `rhs` have different shapes, they will be broadcast to the
        same shape according to
        [broadcasting rules](/max/api/mojo/graph/ops/elementwise/#broadcasting_rules)
        before the operation.

    Args:
        lhs: The symbol to use as left side of the equality comparison.
        rhs: The symbol to use as right side of the equality comparison.
        location: An optional location for a more specific error message.

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
    return _op_impl["rmo.greater_equal"](lhs, rhs, location, __call_location())


@always_inline
def not_equal(
    lhs: Symbol, rhs: Symbol, location: Optional[_SourceLocation] = None
) -> Symbol:
    """Computes the elementwise inequality comparison between two symbolic tensors.

    Creates a new op node to compute the inequality comparison of two symbol
    tensor values and adds it to the graph, returning the symbolic result.

    - If `lhs` and `rhs` have different dtypes, they will be promoted according
        to the
        [dtype promotion rules](/max/api/mojo/graph/ops/elementwise/#dtype_promotion_rules)
        before the operation.
    - If `lhs` and `rhs` have different shapes, they will be broadcast to the
        same shape according to
        [broadcasting rules](/max/api/mojo/graph/ops/elementwise/#broadcasting_rules)
        before the operation.

    Args:
        lhs: The symbol to use as left side of the inequality comparison.
        rhs: The symbol to use as right side of the inequality comparison.
        location: An optional location for a more specific error message.

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
    return _op_impl["rmo.not_equal"](lhs, rhs, location, __call_location())


fn _op_impl[
    op_name: StaticString
](
    lhs: Symbol,
    rhs: Symbol,
    location: Optional[_SourceLocation],
    call_loc: _SourceLocation,
) raises -> Symbol:
    try:
        return _binary_op[op_name](lhs, rhs)
    except e:
        raise error(lhs.graph(), e, location=location or call_loc)


# ===----------------------------------------------------------------------=== #
# Unary Ops
# ===----------------------------------------------------------------------=== #
# Note: Keep alphabetized.


def _unary_op[op_name: StaticString](value: Symbol) -> Symbol:
    return value.graph().op(String(op_name), value, value.tensor_type())


def _unary_float_op[op_name: StaticString](value: Symbol) -> Symbol:
    var dtype = value.tensor_type().dtype
    if not dtype.is_floating_point():
        raise error(
            value.graph(),
            op_name,
            " only supports floating point inputs. Please explicitly cast to",
            " your desired float type first.",
        )
    return value.graph().op(String(op_name), value, value.tensor_type())


def _unary_comparison_op[op_name: StaticString](value: Symbol) -> Symbol:
    var result_type = value.tensor_type().cast(DType.bool)
    return value.graph().op(String(op_name), value, result_type)


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
    return _unary_op["rmo.mo.abs"](value)


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
    return _unary_op["rmo.mo.exp"](value)


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
    return _unary_op["rmo.mo.erf"](value)


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
    return _unary_op["rmo.mo.gelu"](value)


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
    return _unary_op["rmo.mo.log"](value)


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
    return _unary_op["rmo.mo.log1p"](value)


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
    return _unary_op["rmo.mo.logsoftmax"](value)


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
    return _unary_op["rmo.mo.relu"](value)


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
    return _unary_op["rmo.sigmoid"](value)


def silu(value: Symbol) -> Symbol:
    """Computes the elementwise silu of a symbolic tensor.

    Creates a new op node to compute the elementwise silu of a
    symbolic tensor and adds it to the graph, returning the symbolic result.

    `silu` is defined as `silu(x) = x * [sigmoid](#sigmoid)(x)`.

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
    return _unary_op["rmo.mo.softmax"](value)


def cos(value: Symbol) -> Symbol:
    """Computes the elementwise cosine of a symbolic tensor.

    Creates a new op node to compute the elementwise cosine of a
    symbolic tensor and adds it to the graph, returning the symbolic result.

    Args:
        value: The symbolic tensor to use as the input to the cos
            computation. If it's not a floating-point DType, an exception will be raised.

    Returns:
        A new symbolic tensor value representing the output of the absolute
            value computation.

    Raises:
        If the symbol doesn't represent a tensor value.
    """
    return _unary_float_op["rmo.mo.cos"](value)


def floor(value: Symbol) -> Symbol:
    """Computes the elementwise floor of a symbolic tensor.

    Creates a new op node to compute the elementwise floor of a
    symbolic tensor and adds it to the graph, returning the symbolic result.

    Args:
        value: The symbolic tensor to use as the input to the floor
            computation. If it's not a floating-point DType, an exception will be raised.

    Returns:
        A new symbolic tensor value representing the output of the absolute
            value computation.

    Raises:
        If the symbol doesn't represent a tensor value.
    """
    return _unary_float_op["rmo.mo.floor"](value)


def round(value: Symbol) -> Symbol:
    """Computes the elementwise round of a symbolic tensor.

    Creates a new op node to compute the elementwise round of a
    symbolic tensor and adds it to the graph, returning the symbolic result.

    Args:
        value: The symbolic tensor to use as the input to the round
            computation. If it's not a floating-point DType, an exception will be raised.

    Returns:
        A new symbolic tensor value representing the output of the absolute
            value computation.

    Raises:
        If the symbol doesn't represent a tensor value.
    """
    return _unary_float_op["rmo.mo.round"](value)


def rsqrt(value: Symbol) -> Symbol:
    """Computes the elementwise inverse-square-root of a symbolic tensor.

    Creates a new op node to compute the elementwise rsqrt of a
    symbolic tensor and adds it to the graph, returning the symbolic result.

    Args:
        value: The symbolic tensor to use as the input to the rsqrt
            computation. If it's not a floating-point DType, an exception will be raised.

    Returns:
        A new symbolic tensor value representing the output of the absolute
            value computation.

    Raises:
        If the symbol doesn't represent a tensor value.
    """
    return _unary_float_op["rmo.mo.isqrt"](value)


def sqrt(value: Symbol) -> Symbol:
    """Computes the elementwise sqrt of a symbolic tensor.

    Creates a new op node to compute the elementwise sqrt of a
    symbolic tensor and adds it to the graph, returning the symbolic result.

    Args:
        value: The symbolic tensor to use as the input to the sqrt
            computation. If it's not a floating-point DType, an exception will be raised.

    Returns:
        A new symbolic tensor value representing the output of the absolute
            value computation.

    Raises:
        If the symbol doesn't represent a tensor value.
    """
    return _unary_float_op["rmo.mo.sqrt"](value)


def sin(value: Symbol) -> Symbol:
    """Computes the elementwise sine of a symbolic tensor.

    Creates a new op node to compute the elementwise sine of a
    symbolic tensor and adds it to the graph, returning the symbolic result.

    Args:
        value: The symbolic tensor to use as the input to the sin
            computation. If it's not a floating-point DType, an exception will be raised.

    Returns:
        A new symbolic tensor value representing the output of the absolute
            value computation.

    Raises:
        If the symbol doesn't represent a tensor value.
    """
    return _unary_float_op["rmo.mo.sin"](value)


def tanh(value: Symbol) -> Symbol:
    """Computes the elementwise tanh of a symbolic tensor.

    Creates a new op node to compute the elementwise tanh of a
    symbolic tensor and adds it to the graph, returning the symbolic result.

    Args:
        value: The symbolic tensor to use as the input to the tanh
            computation. If it's not a floating-point DType, an exception will be raised.

    Returns:
        A new symbolic tensor value representing the output of the absolute
            value computation.

    Raises:
        If the symbol doesn't represent a tensor value.
    """
    return _unary_float_op["rmo.mo.tanh"](value)


def trunc(value: Symbol) -> Symbol:
    """Computes the elementwise truncation of a symbolic tensor.

    Creates a new op node to compute the elementwise trunc of a
    symbolic tensor and adds it to the graph, returning the symbolic result.

    Args:
        value: The symbolic tensor to use as the input to the trunc
            computation. If it's not a floating-point DType, an exception will be raised.

    Returns:
        A new symbolic tensor value representing the output of the absolute
            value computation.

    Raises:
        If the symbol doesn't represent a tensor value.
    """
    return _unary_float_op["rmo.mo.trunc"](value)


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
    return _unary_comparison_op["rmo.mo.is_nan"](value)


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
    return _unary_comparison_op["rmo.mo.is_inf"](value)
