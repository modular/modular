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
"""Ops that modify the shape or data type of a symbolic tensor."""
from collections import Dict, Optional

from _mlir.builtin_attributes import StringAttr
from _mlir.ir import Identifier, NamedAttribute
from builtin._location import __call_location, _SourceLocation
from max.tensor import Tensor, TensorShape
from memory import UnsafePointer

from .._attributes import _shape_attr
from ..error import error
from ..type import Dim

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
    g = v.graph()
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
    if v.tensor_type().dtype == dtype:
        return v
    return v.graph().op("rmo.mo.cast", v, v.tensor_type().cast(dtype))


# ===----------------------------------------------------------------------=== #
# Rebind
# ===----------------------------------------------------------------------=== #


def rebind(v: Symbol, out_dims: List[Dim], message: String) -> Symbol:
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
    g = v.graph()
    if v.tensor_type().rank() != len(out_dims):
        raise error(
            g, "rebind out_dims length must match the rank of the input shape"
        )

    known_dims = Dict[String, Int64]()

    def try_unify_symbolic_to_static(x: Dim, y: Dim):
        """Fills `known_dims` with mappings from symbolic dims to static dims.

        Will raise if a symbolic dim is defined twice.
        """
        if not (x.is_symbolic() and y.is_static()):
            return

        x_str = String(x)
        y_int = y.num_elements()

        if x_str not in known_dims:
            known_dims[x_str] = y_int
            return

        known_dim = known_dims[x_str]
        if known_dim != y_int:
            raise error(
                g,
                "rebind out_dims statically known to be incorrect.",
                ' Dimension (value "',
                x_str,
                '") rebinds to two different constants: ',
                known_dim,
                " and ",
                y_int,
            )

    # Build mapping from symbolic to known statically known values.
    for i in range(len(out_dims)):
        src_dim = v.shape()[i]
        dst_dim = out_dims[i]

        try_unify_symbolic_to_static(src_dim, dst_dim)
        try_unify_symbolic_to_static(dst_dim, src_dim)

    def known_dim_size(d: Dim) -> Optional[Int64]:
        """Loads the length of a dim if known.

        The length is known in two cases:
          1. The dim is static.
          2. The dim is symbolic and in `known_dims`.
        """
        if d.is_static():
            return d.num_elements()

        if d.is_symbolic() and String(d) in known_dims:
            return known_dims[String(d)]

        return None

    # Ensure all statically known dims are equivalent.
    for i in range(len(out_dims)):
        src_dim = v.shape()[i]
        dst_dim = out_dims[i]

        src_size = known_dim_size(src_dim)
        dst_size = known_dim_size(dst_dim)
        if src_size and dst_size and src_size.value() != dst_size.value():
            raise Error(
                g,
                "rebind out_dims statically known to be incorrect. Dimension",
                ' (name: "',
                src_dim,
                ", value: ",
                src_size.value(),
                '") cannot rebind to Dimension (name: ',
                dst_dim,
                ", value: ",
                dst_size.value(),
                ")",
            )

    ctx = g._context()
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
    g = v.graph()
    v_type = v.tensor_type()
    rank = v_type.rank()
    if axis < 0:
        axis += rank

    new_shape = g.op(
        "rmo.mo.squeeze_shape",
        List[Symbol](shape_of(v), g.scalar(Int64(axis), rank=1)),
        TensorType(DType.int64, rank - 1),
    )

    squeezed_dims = List[Dim]()
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
    g = v.graph()
    type = v.tensor_type()
    rank = type.rank()
    # Negative values add an extra 1, as -1 adds a new dim at the _end_.
    if axis < 0:
        axis += rank + 1
    if axis < 0 or axis > rank:
        raise Error(
            g, "unsqueeze axis out of bounds: axis=", axis, ", rank=", rank
        )

    # Short circuit to handle scalars with less ops.
    if rank == 0:
        return v.reshape(1)

    # TODO: Bug - passing v_type.rank() + 1 into a variadic Int64 corrupts it.
    new_shape = g.op(
        "rmo.mo.unsqueeze_shape",
        List[Symbol](shape_of(v), g.scalar(Int64(axis), rank=1)),
        TensorType(DType.int64, rank + 1),
    )

    dims = List[Dim]()
    for i in range(rank):
        if i == axis:
            dims.append(1)
        dims.append(type.dims[i])
    if axis == rank:
        dims.append(1)

    return reshape(v, new_shape, dims)


# TODO(GEX-578): Remove old reshape apis once we have dim expressions and remove dynamic dimensions.
# Only this version should be needed in the future.
def reshape(v: Symbol, shape: List[Dim]) -> Symbol:
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
            A single dimension may be `-1`.

    Returns:
        A symbolic tensor with the same elements as the original tensor, but
        in a new shape. Its symbolic shape is the same as `shape`.
    """
    g = v.graph()

    ctx = g._context()
    newShapeAttr = _shape_attr(ctx, "newShape", shape)
    return g.op(
        "rmo.reshape",
        List[Symbol](v),
        attrs=List[NamedAttribute](newShapeAttr),
    )


def reshape(v: Symbol, shape: Symbol, out_dims: List[Dim]) -> Symbol:
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
    g = v.graph()
    dtype = shape.tensor_type().dtype
    if not (dtype is DType.int64 or dtype is DType.int32):
        raise error(g, "reshape shape must be int32 or int64")
    if shape.tensor_type().rank() != 1:
        raise error(g, "reshape shape must be rank 1")
    return g.op(
        "rmo.mo.reshape",
        List[Symbol](v, shape),
        TensorType(v.tensor_type().dtype, out_dims),
    )


def reshape(v: Symbol, shape: List[Symbol]) -> Symbol:
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
    g = v.graph()

    if len(shape) == 0:  # Can't `stack` an empty tuple
        dims = List[Dim]()
        return reshape(v, g.constant(Tensor[DType.int64](TensorShape(0))), dims)

    for i in range(len(shape)):
        if shape[i].tensor_type().rank() != 0:
            print(shape[i])
            raise error(g, "reshape requires 0-rank dims")

    return reshape(v, stack(shape))


def reshape(v: Symbol, shape: Symbol) -> Symbol:
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
    g = v.graph()

    shape_t = shape.tensor_type()
    if (shape_t.rank() != 1) or (not shape_t.dims[0].is_static()):
        raise error(g, "reshape shape requires static shape shape")
    out_dims = List[Dim]()
    for _ in range(shape_t.dims[0].num_elements()):
        out_dims.append(Dim.dynamic())

    return reshape(v, shape, out_dims)


def reshape_like(v: Symbol, like: Symbol) -> Symbol:
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
# Broadcasts
# ===----------------------------------------------------------------------=== #


@always_inline
def broadcast_to(
    v: Symbol, shape: List[Dim], location: Optional[_SourceLocation] = None
) -> Symbol:
    """Broadcasts a symbolic tensor.

    Broadcasts the input tensor to the specified shape.
    Dimensions in the input must be one or match the target dimension.

    Args:
        v: The input symbolic tensor to broadcast.
            This tensor may not contain any dynamic dimensions.
        shape: The new shape as a list of dimensions.
            Dynamic dimensions are not allowed.
        location: An optional location for a more specific error message.

    Returns:
        A symbolic tensor with the same elements as the original tensor, but
        in a new shape. Its symbolic shape is the same as `shape`.
    """
    g = v.graph()

    ctx = g._context()
    newShapeAttr = _shape_attr(ctx, "newShape", shape)
    try:
        return g.op(
            "rmo.broadcast_to",
            List[Symbol](v),
            attrs=List[NamedAttribute](newShapeAttr),
        )
    except e:
        raise error(g, e, location=location or __call_location())


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
    g = input.graph()
    input_type = input.tensor_type()
    if input_type.rank() < 2:
        raise "transpose input must have rank >= 2"
    if x < 0:
        x += input_type.rank()
    if y < 0:
        y += input_type.rank()
    if x < 0 or x >= input_type.rank() or y < 0 or y >= input_type.rank():
        raise "transpose dim outside range"

    dims = List[Dim]()
    ptr = UnsafePointer[Int64].alloc(input_type.rank())
    for i in range(input_type.rank()):
        dims.append(input_type.dims[i])
        ptr.store(i, i)

    dims[x] = input_type.dims[y]
    dims[y] = input_type.dims[x]
    ptr.store(x, y)
    ptr.store(y, x)

    transpose_indices = g.constant(
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
