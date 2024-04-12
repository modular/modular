# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Operations which operate on the shape or dtype of symbolic tensor."""

from tensor import Tensor, TensorShape
from utils.variant import Variant

from max.graph.type import Dim, ElementType


# TODO: Add checks or extend to unranked support, where static shapes assumed.


# ===----------------------------------------------------------------------=== #
# Shape accessors
# ===----------------------------------------------------------------------=== #


def shape_of(v: Symbol) -> Symbol:
    """Gets the shape of an existing tensor as a rank-1 symbolic tensor.

    Args:
        v: The symbolic tensor whose shape is returned.

    Returns:
        A symbolic rank-1 tensor representing the input's shape.
    """
    var g = v.graph()
    return g.op("mo.shape_of", v, MOTensor(DType.int64, v.tensor_type().rank()))


# ===----------------------------------------------------------------------=== #
# Casters
# ===----------------------------------------------------------------------=== #


def cast(v: Symbol, dtype: ElementType) -> Symbol:
    """Casts a symbolic tensor to a different dtype.

    Args:
        v: The input tensor to cast.
        dtype: The target dtype to which the tensor is cast.

    Returns:
        A new symbolic tensor with the same shape as the input and the
        specified dtype.
    """
    return v.graph().op("mo.cast", v, v.tensor_type().cast(dtype))


# ===----------------------------------------------------------------------=== #
# Rebind
# ===----------------------------------------------------------------------=== #


fn rebind(v: Symbol, out_dims: List[Dim]) raises -> Symbol:
    """Rebinds a symbolic tensor to a specified set of Dims.

    Args:
        v: The input symbolic tensor to rebind.
        out_dims: The new symbolic shape in the graph.

    Returns:
        A symbolic tensor with the same elements and shape as the original tensor.

        Its symbolic shape will be changed to `out_dims`.
        A runtime assert will be added that the original symbolic shape is equivalent to the new symbolic shape.
    """
    var g = v.graph()
    if v.tensor_type().rank() != len(out_dims):
        raise "rebind out_dims length must match the rank of the input shape"

    return g.op(
        "rmo.rebind_tensor_shape",
        (v),
        MOTensor(v.tensor_type().dtype, out_dims),
    )


# ===----------------------------------------------------------------------=== #
# Reshapes
# ===----------------------------------------------------------------------=== #


def squeeze(v: Symbol, axis: Int) -> Symbol:
    """Removes a size-1 dimension from a symbolic tensor.

    Args:
        v: The input symbolic tensor to squeeze.
        axis: The dimension to remove from the input's shape. If negative,
            indexes from the end of the tensor, eg. `squeeze(v, -1)` will
            squeeze the last dimension.

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
        "mo.squeeze_shape",
        (shape_of(v), g.scalar(Int64(axis), rank=1)),
        MOTensor(DType.int64, rank - 1),
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
            shape. Dimensions with that index or higher will be shifted back.
            If negative, it indexes relative _1 plus_ the rank of the tensor,
            in other words `unsqueeze(v, -1)` will add the new dimension at the
            end, `unsqueeze(v, -2)` will insert the dimension immediately before
            the last dimension, etc.

    Returns:
        A symbolic tensor with the same muber of elements as the input tensor,
        whose rank is 1 larger than the rank of the input tensor. The result's
        shape at the `axis` dimension will be a static dimension of size 1.
    """
    var g = v.graph()
    var type = v.tensor_type()
    var rank = type.rank()
    # Negative values add an extra 1, as -1 adds a new dim at the _end_.
    if axis < 0:
        axis += rank + 1
    if axis < 0 or axis > rank:
        raise (
            "unsqueeze axis out of bounds: axis="
            + str(axis)
            + ", rank="
            + str(rank)
        )

    # TODO: Bug - passing v_type.rank() + 1 into a variadic Int64 corrupts it.
    var new_shape = g.op(
        "mo.unsqueeze_shape",
        (shape_of(v), g.scalar(Int64(axis), rank=1)),
        MOTensor(DType.int64, rank + 1),
    )

    var dims = List[Dim]()
    for i in range(rank):
        if i == axis:
            dims.append(1)
        dims.append(type.dims[i])
    if axis == rank:
        dims.append(1)

    return reshape(v, new_shape, dims)


fn reshape(v: Symbol, shape: Symbol, out_dims: List[Dim]) raises -> Symbol:
    """Reshapes a symbolic tensor to a specified shape.

    Args:
        v: The input symbolic tensor to reshape.
        shape: The shape to reshape to as a symbolic rank-1 tensor.
            The input must have integer dtype, and must either be all
            non-negative elements or allows a single element of -1.
        out_dims: A type hint for the tensor's symbolic shape in the graph.

    Returns:
        A symbolic tensor with the same elements as the original tensor
        in a new shape. Its symbolic shape will be the same as `out_dims`.

        The number and order of the elements in the tensor is unchanged.
        In other words, if you were to iterate over elements in the tensor
        by major dimension to minor dimension, the iteration order would stay
        the same.

        If a value of -1 is present in the shape, that dimension will become
        a dynamic dimension collecting all unspecified dimensions.
        Its length will be the number of elements in the original tensor
        divided by the product of elements of the reshape.
    """
    var g = v.graph()
    var dtype = shape.tensor_type().dtype.dtype
    if not (dtype == DType.int64 or dtype == DType.int32):
        raise "reshape shape must be int32 or int64"
    if shape.tensor_type().rank() != 1:
        raise "reshape shape must be rank 1"
    return g.op(
        "mo.reshape", (v, shape), MOTensor(v.tensor_type().dtype, out_dims)
    )


fn reshape(v: Symbol, shape: SymbolTuple) raises -> Symbol:
    """Reshapes a symbolic tensor to a specified shape.

    Args:
        v: The input symbolic tensor to reshape.
        shape: A list of rank-0 symbolic tensors to reshape to.
            The inputs must have integer dtype, and must either be all
            non-negative elements or allows a single element of -1.

    Returns:
        A symbolic tensor with the same elements as the original tensor
        in a new shape. It will have rank equal to the length of `shape`,
        but every symbolic dimension of the result will be dynamic size.

        The number and order of the elements in the tensor is unchanged.
        In other words, if you were to iterate over elements in the tensor
        by major dimension to minor dimension, the iteration order would stay
        the same.

        If a value of -1 is present in the shape, that dimension will become
        a dynamic dimension collecting all unspecified dimensions.
        Its length will be the number of elements in the original tensor
        divided by the product of elements of the reshape.
    """
    var g = v.graph()

    if len(shape) == 0:  # Can't `stack` an empty tuple
        var dims = List[Dim]()
        return reshape(v, g.constant(Tensor[DType.int64](TensorShape(0))), dims)

    for i in range(len(shape)):
        if shape[i].tensor_type().rank() != 0:
            print(shape[i])
            raise "reshape requires 0-rank dims"

    return reshape(v, stack(shape))


fn reshape(v: Symbol, shape: Symbol) raises -> Symbol:
    """Reshapes a symbolic tensor to a specified shape.

    Args:
        v: The input symbolic tensor to reshape.
        shape: The shape to reshape to as a symbolic rank-1 tensor.
            The input must have integer dtype, and must either be all
            non-negative elements or allows a single element of -1.

    Returns:
        A symbolic tensor with the same elements as the original tensor
        in a new shape. It will have rank equal to the static size of `shape`,
        but every symbolic dimension of the result will be dynamic size.

        The number and order of the elements in the tensor is unchanged.
        In other words, if you were to iterate over elements in the tensor
        by major dimension to minor dimension, the iteration order would stay
        the same.

        If a value of -1 is present in the shape, that dimension will become
        a dynamic dimension collecting all unspecified dimensions.
        Its length will be the number of elements in the original tensor
        divided by the product of elements of the reshape.
    """
    var g = v.graph()

    var shape_t = shape.tensor_type()
    if (shape_t.rank() != 1) or (not shape_t.dims[0].is_static()):
        raise "reshape shape requires static shape shape"
    var out_dims = List[Dim]()
    for _ in range(shape_t.dims[0].num_elements()):
        out_dims.append(Dim.dynamic())

    return reshape(v, shape, out_dims)


fn reshape_like(v: Symbol, like: Symbol) raises -> Symbol:
    """Reshapes a symbolic tensor to the same shape as another tensor.

    Args:
        v: The input symbolic tensor to reshape.
        like: A symbolic tensor whose shape should be used as the reshape.

    Returns:
        A symbolic tensor with the same elements as the original tensor
        in a new shape. The shape of the new tensor will be the same as
        the shape of the `like` tensor.

        The number and order of the elements in the tensor is unchanged.
        In other words, if you were to iterate over elements in the tensor
        by major dimension to minor dimension, the iteration order would stay
        the same.
    """
    return reshape(v, shape_of(like), like.tensor_type().dims)


# ===----------------------------------------------------------------------=== #
# Transpositions
# ===----------------------------------------------------------------------=== #


def transpose(input: Symbol, x: Int, y: Int) -> Symbol:
    """Transposes two dimensions of a symbolic tensor.

    Args:
        input: The input symbolic tensor to transpose.
        x: One of the two dimensions to transpose. If negative,
            indexes from the end of the tensor, eg. `transpose(v, -1, -2)` will
            transpose the last two dimensions.
        y: The other dimension to transpose. May also be negative to index from
            the end of the tensor.

    Returns:
        A new symbolic tensor with the two specified dimensions transposed.
        It will have the same elements and dtype, but the order of the elements
        will be different according to the transposition.
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
        ptr.store(i, i)

    dims[x] = input_type.dims[y]
    dims[y] = input_type.dims[x]
    ptr.store(x, y)
    ptr.store(y, x)

    var transpose_indices = g.constant(
        Tensor[DType.int64](TensorShape(input_type.rank()), ptr)
    )

    return g.op(
        "mo.transpose",
        (input, transpose_indices),
        MOTensor(input_type.dtype, dims),
    )


def transpose_matrix(matrix: Symbol) -> Symbol:
    """Transposes the last two dimensions of a symbolic tensor.

    Args:
        matrix: The symbolic tensor to transpose.

    Returns:
        A new symbolic tensor with its last two dimensions transposed.
        It will have the same elements and dtype, but the order of the elements
        will be different according to the transposition.
    """
    return transpose(matrix, -1, -2)
