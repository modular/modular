# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Ops that modify the shape or data type of a symbolic tensor."""
from _mlir.ir import NamedAttribute, Identifier
from _mlir.builtin_attributes import StringAttr

from .._attributes import _shape_attr
from ..error import error
from ..type import Dim

from tensor import Tensor, TensorShape

# TODO: Add checks or extend to unranked support, where static shapes assumed.


# ===----------------------------------------------------------------------=== #
# Shape accessors
# ===----------------------------------------------------------------------=== #


def shape_of(v: Symbol) -> Symbol:
    """Gets the shape of a symbolic tensor as a rank-1 symbolic tensor.

    Args:
        v: The symbolic tensor whose shape is returned.

    Returns:
        A symbolic rank-1 tensor representing the input's shape.
    """
    var g = v.graph()
    return g.op(
        "rmo.mo.shape_of", v, TensorType(DType.int64, v.tensor_type().rank())
    )


# ===----------------------------------------------------------------------=== #
# Casters
# ===----------------------------------------------------------------------=== #


def cast(v: Symbol, dtype: DType) -> Symbol:
    """Casts a symbolic tensor to a different data type.

    Args:
        v: The input tensor to cast.
        dtype: The target dtype to which the tensor is cast.

    Returns:
        A new symbolic tensor with the same shape as the input and the
        specified dtype.
    """
    return v.graph().op("rmo.mo.cast", v, v.tensor_type().cast(dtype))


# ===----------------------------------------------------------------------=== #
# Rebind
# ===----------------------------------------------------------------------=== #


fn rebind(v: Symbol, out_dims: List[Dim], message: String) raises -> Symbol:
    """Rebinds a symbolic tensor to a specified set of dimensions.

    This does not mutate the symbolic tensor passed in, but instead adds a
    runtime assert that the input symbolic shape is equivalent to `out_dims`
    shape. For example, if the input tensor shape has dynamic/unknown sizes,
    this will assert a fixed sizes that may be required for a subsequent
    operation.

    Args:
        v: The input symbolic tensor to rebind.
        out_dims: The symbolic shape to assert for `v`, as a list of
                  [`Dim`](/max/api/mojo/graph/type/Dim) values.
        message: The message printed if the rebind fails at runtime.

    Returns:
        A symbolic tensor with the same elements and shape as the given
        tensor, but with the symbolic shape asserted to `out_dims`.

    """
    var g = v.graph()
    if v.tensor_type().rank() != len(out_dims):
        raise error(
            g, "rebind out_dims length must match the rank of the input shape"
        )

    var ctx = g._context()
    return g.op(
        "rmo.rebind_tensor_shape",
        (v),
        TensorType(v.tensor_type().dtype, out_dims),
        attrs=List[NamedAttribute](
            NamedAttribute(
                name=Identifier(ctx, "message"),
                attr=StringAttr(ctx, message),
            )
        ),
    )


# ===----------------------------------------------------------------------=== #
# Reshapes
# ===----------------------------------------------------------------------=== #


def squeeze(v: Symbol, axis: Int) -> Symbol:
    """Removes a size-1 dimension from a symbolic tensor.

    Args:
        v: The input symbolic tensor to squeeze.
        axis: The dimension to remove from the input's shape. If negative, this
            indexes from the end of the tensor. For example, `squeeze(v, -1)`
            squeezes the last dimension.

    Returns:
        A symbolic tensor with the same number of elements as the input tensor,
        and whose rank is 1 less than the rank of the input tensor.
    """
    var g = v.graph()
    var v_type = v.tensor_type()
    var rank = v_type.rank()
    if axis < 0:
        axis += rank

    var new_shape = g.op(
        "rmo.mo.squeeze_shape",
        List[Symbol](shape_of(v), g.scalar(Int64(axis), rank=1)),
        TensorType(DType.int64, rank - 1),
    )

    var squeezed_dims = List[Dim]()
    for i in range(rank):
        if i != axis:
            squeezed_dims.append(v_type.dims[i])

    return reshape(v, new_shape, squeezed_dims)


def unsqueeze(v: Symbol, axis: Int) -> Symbol:
    """Inserts a size-1 dimension into a symbolic tensor.

    Args:
        v: The input symbolic tensor to unsqueeze.
        axis: The index at which to insert a new dimension into the input's
            shape. Elements at that index or higher are shifted back.
            If negative, it indexes relative _1 plus_ the rank of the tensor.
            For example, `unsqueeze(v, -1)` adds a new dimension at the end,
            and `unsqueeze(v, -2)` inserts the dimension immediately before
            the last dimension.

    Returns:
        A symbolic tensor with the same muber of elements as the input tensor,
        whose rank is 1 larger than the rank of the input tensor. The result's
        shape at the `axis` dimension is a static dimension of size 1.
    """
    var g = v.graph()
    var type = v.tensor_type()
    var rank = type.rank()
    # Negative values add an extra 1, as -1 adds a new dim at the _end_.
    if axis < 0:
        axis += rank + 1
    if axis < 0 or axis > rank:
        raise error(
            g,
            "unsqueeze axis out of bounds: axis="
            + str(axis)
            + ", rank="
            + str(rank),
        )

    # TODO: Bug - passing v_type.rank() + 1 into a variadic Int64 corrupts it.
    var new_shape = g.op(
        "rmo.mo.unsqueeze_shape",
        List[Symbol](shape_of(v), g.scalar(Int64(axis), rank=1)),
        TensorType(DType.int64, rank + 1),
    )

    var dims = List[Dim]()
    for i in range(rank):
        if i == axis:
            dims.append(1)
        dims.append(type.dims[i])
    if axis == rank:
        dims.append(1)

    return reshape(v, new_shape, dims)


# TODO(GRA-578): Remove old reshape apis once we have dim expressions and remove dynamic dimensions.
# Only this version should be needed in the future.
fn reshape(v: Symbol, shape: List[Dim]) raises -> Symbol:
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
        v: The input symbolic tensor to reshape.
            This tensor may not contain any dynamic dimensions.
        shape: The new shape as a list of dimensions.
            Dynamic dimensions are not allowed.
            A single dimension my be `-1`.

    Returns:
        A symbolic tensor with the same elements as the original tensor, but
        in a new shape. Its symbolic shape is the same as `shape`.
    """
    var g = v.graph()

    var ctx = g._context()
    var newShapeAttr = _shape_attr(ctx, "newShape", shape)
    return g.nvop(
        "rmo.reshape",
        List[Symbol](v),
        attrs=List[NamedAttribute](newShapeAttr),
        enable_result_type_inference=True,
    )[0]


fn reshape(v: Symbol, shape: Symbol, out_dims: List[Dim]) raises -> Symbol:
    """Reshapes a symbolic tensor.

    The number and order of the elements in the tensor is unchanged.
    In other words, if you were to iterate over elements in the tensor
    by major dimension to minor dimension, the iteration order would stay
    the same.

    If a value of -1 is present in the shape, that dimension becomes
    a dynamic dimension collecting all unspecified dimensions.
    Its length becomes the number of elements in the original tensor
    divided by the product of elements of the reshape.

    Args:
        v: The input symbolic tensor to reshape.
        shape: The new shape as a symbolic rank-1 tensor.
            The input must have integer dtype, and must either be all
            non-negative elements or allows a single element of -1.
        out_dims: A type hint for the tensor's symbolic shape in the graph.

    Returns:
        A symbolic tensor with the same elements as the original tensor, but
        in a new shape. Its symbolic shape is the same as `out_dims`.
    """
    var g = v.graph()
    var dtype = shape.tensor_type().dtype
    if not (dtype is DType.int64 or dtype is DType.int32):
        raise error(g, "reshape shape must be int32 or int64")
    if shape.tensor_type().rank() != 1:
        raise error(g, "reshape shape must be rank 1")
    return g.op(
        "rmo.mo.reshape",
        List[Symbol](v, shape),
        TensorType(v.tensor_type().dtype, out_dims),
    )


fn reshape(v: Symbol, shape: List[Symbol]) raises -> Symbol:
    """Reshapes a symbolic tensor.

    The number and order of the elements in the tensor is unchanged.
    In other words, if you were to iterate over elements in the tensor
    by major dimension to minor dimension, the iteration order would stay
    the same.

    If a value of -1 is present in the shape, that dimension becomes
    a dynamic dimension collecting all unspecified dimensions.
    Its length becomes the number of elements in the original tensor
    divided by the product of elements of the reshape.

    Args:
        v: The input symbolic tensor to reshape.
        shape: The new shape as a list of rank-0 symbolic tensors.
            The inputs must have integer dtype, and must either be all
            non-negative elements or allows a single element of -1.

    Returns:
        A symbolic tensor with the same elements as the original tensor, but
        in a new shape. It has a rank equal to the length of `shape`,
        but every symbolic dimension of the result is a dynamic size.
    """
    var g = v.graph()

    if len(shape) == 0:  # Can't `stack` an empty tuple
        var dims = List[Dim]()
        return reshape(v, g.constant(Tensor[DType.int64](TensorShape(0))), dims)

    for i in range(len(shape)):
        if shape[i].tensor_type().rank() != 0:
            print(shape[i])
            raise error(g, "reshape requires 0-rank dims")

    return reshape(v, stack(shape))


fn reshape(v: Symbol, shape: Symbol) raises -> Symbol:
    """Reshapes a symbolic tensor.

    The number and order of the elements in the tensor is unchanged.
    In other words, if you were to iterate over elements in the tensor
    by major dimension to minor dimension, the iteration order would stay
    the same.

    If a value of -1 is present in the shape, that dimension becomes
    a dynamic dimension collecting all unspecified dimensions.
    Its length becomes the number of elements in the original tensor
    divided by the product of elements of the reshape.

    Args:
        v: The input symbolic tensor to reshape.
        shape: The new shape as a symbolic rank-1 tensor.
            The input must have integer dtype, and must either be all
            non-negative elements or allows a single element of -1.

    Returns:
        A symbolic tensor with the same elements as the original tensor, but
        in a new shape. It has a rank equal to the static size of `shape`,
        but every symbolic dimension of the result is a dynamic size.
    """
    var g = v.graph()

    var shape_t = shape.tensor_type()
    if (shape_t.rank() != 1) or (not shape_t.dims[0].is_static()):
        raise error(g, "reshape shape requires static shape shape")
    var out_dims = List[Dim]()
    for _ in range(shape_t.dims[0].num_elements()):
        out_dims.append(Dim.dynamic())

    return reshape(v, shape, out_dims)


fn reshape_like(v: Symbol, like: Symbol) raises -> Symbol:
    """Reshapes a symbolic tensor to the same shape as another symbolic tensor.

    The number and order of the elements in the tensor is unchanged.
    In other words, if you were to iterate over elements in the tensor
    by major dimension to minor dimension, the iteration order would stay
    the same.

    Args:
        v: The input symbolic tensor to reshape.
        like: A symbolic tensor whose shape should be used as the reshape.

    Returns:
        A symbolic tensor with the same elements as the original tensor, but
        in a new shape. The shape of the new tensor is the same as
        the shape of the `like` tensor.
    """
    return reshape(v, shape_of(like), like.tensor_type().dims)


# ===----------------------------------------------------------------------=== #
# Transpositions
# ===----------------------------------------------------------------------=== #


def transpose(input: Symbol, x: Int, y: Int) -> Symbol:
    """Transposes two dimensions of a symbolic tensor.

    Args:
        input: The input symbolic tensor to transpose.
        x: One of the two dimensions to transpose. If negative, this indexes
            from the end of the tensor. For example,  `transpose(v, -1, -2)`
            transposes the last two dimensions.
        y: The other dimension to transpose. May also be negative to index from
            the end of the tensor.

    Returns:
        A new symbolic tensor with the two specified dimensions transposed.
        It has the same elements and dtype, but the order of the elements
        is different according to the transposition.
    """
    var g = input.graph()
    var input_type = input.tensor_type()
    if input_type.rank() < 2:
        raise "transpose input must have rank >= 2"
    if x < 0:
        x += input_type.rank()
    if y < 0:
        y += input_type.rank()
    if x < 0 or x >= input_type.rank() or y < 0 or y >= input_type.rank():
        raise "transpose dim outside range"

    var dims = List[Dim]()
    var ptr = DTypePointer[DType.int64].alloc(input_type.rank())
    for i in range(input_type.rank()):
        dims.append(input_type.dims[i])
        Scalar.store(ptr, i, i)

    dims[x] = input_type.dims[y]
    dims[y] = input_type.dims[x]
    Scalar.store(ptr, x, y)
    Scalar.store(ptr, y, x)

    var transpose_indices = g.constant(
        Tensor[DType.int64](TensorShape(input_type.rank()), ptr)
    )

    return g.op(
        "rmo.mo.transpose",
        List[Symbol](input, transpose_indices),
        TensorType(input_type.dtype, dims),
    )


def transpose_matrix(matrix: Symbol) -> Symbol:
    """Transposes the last two dimensions of a symbolic tensor.

    Args:
        matrix: The symbolic tensor to transpose.

    Returns:
        A new symbolic tensor with its last two dimensions transposed.
        It has the same elements and dtype, but the order of the elements
        is different according to the transposition.
    """
    return transpose(matrix, -1, -2)
