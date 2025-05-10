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
"""Attribute primitives.

`Attribute`s are key-value pairs that can be attached to a `Node`, `Graph` and
other elements. Attributes are similar to inputs, except they are constant -
their value doesn't change at runtime. The attribute name is always a string.

For exmple, `mo.constant` has a `value` attribute, representing the value
of the constant it holds.

`Attribute`s can hold various types of values, including primitive values,
lists, tensors, etc.
"""

import _mlir
from max.tensor import Tensor

import ._c
from .type import TensorType

# ===------------------------------------------------------------------=== #
# Attribute factories
# ===------------------------------------------------------------------=== #


fn _tensor_attr[
    dtype: DType
](
    ctx: _mlir.Context, name: String, owned value: Tensor[dtype]
) -> _mlir.NamedAttribute:
    """Creates a new `Tensor`-valued `Attribute`.

    The value of this attribute will have the type `TensorType` with the same
    shape and dtype as `value`.
    This method takes ownership of `value` and is suitable for use with
    very large `Tensor` values (such as model weights).

    Parameters:
        dtype: The attribute tensor's element type.

    Args:
        ctx: The MLIR context.
        name: The `Attribute` name.
        value: The `Attribute` value.

    Returns:
        An internal representation of an `Attribute`.
    """
    var t = TensorType(value.spec()).to_mlir(ctx)
    return _c.attr_new_tensor(
        name,
        value._steal_ptr().bitcast[NoneType](),
        t,
        is_owned=True,
    )


fn _tensor_resource_attr(
    ctx: _mlir.Context, name: String, file_name: String, type: TensorType
) -> _mlir.NamedAttribute:
    """Creates a new `Tensor` `Attribute` from an external file.

    The value of this constant will have the type `type`.
    The file must contain the `Tensor`s raw data, as returned by
    `Tensor.data`. No endianness transformation is performed.

    Args:
        ctx: The MLIR context.
        name: The `Attribute` name.
        file_name: The file name to load from.
        type: The `Tensor` type (element type, shape).

    Returns:
        An internal representation of an `Attribute`.
    """
    return _c.attr_new_tensor_from_file(name, file_name, type.to_mlir(ctx))


fn _vector_attr[
    dtype: DType
](
    ctx: _mlir.Context, name: String, values: List[Scalar[dtype]]
) -> _mlir.NamedAttribute:
    """Creates a new `Tensor`-valued `Attribute`.

    The value of this attribute will have the type `TensorType` with 1D shape,
    consistent with the size of `values`.

    Parameters:
        dtype: The attribute tensor's element type.

    Args:
        ctx: The MLIR context.
        name: The `Attribute` name.
        values: A vector representing the attribute's value.

    Returns:
        An internal representation of an `Attribute`.
    """
    return _c.attr_new_tensor(
        name,
        values,
        TensorType(dtype, len(values)).to_mlir(ctx),
        is_owned=False,
    )


fn _scalar_attr[
    dtype: DType
](
    ctx: _mlir.Context, name: String, value: Scalar[dtype], rank: Int = 0
) raises -> _mlir.NamedAttribute:
    """Creates a new `Tensor`-valued `Attribute`.

    The `Tensor` is considered to contain a single element, and its shape
    be of the specified rank (for example, `rank=0` denotes a scalar).

    Parameters:
        dtype: The attribute tensor's element type.

    Args:
        ctx: The MLIR context.
        name: The `Attribute` name.
        value: The `Attribute` value.
        rank: The attribute tensor's rank.

    Returns:
        An internal representation of an `Attribute`.
    """
    # Note: while this could generalize to something like splat, MO doesn't
    # really make use of those.
    var shape = List[Int, hint_trivial_type=True](capacity=rank)
    for _ in range(rank):
        shape.append(1)
    return _tensor_attr[dtype](ctx, name, Tensor(shape, value))


fn _string_attr(
    ctx: _mlir.Context, name: String, value: String
) -> _mlir.NamedAttribute:
    """Creates a new `String`-valued `Attribute`.

    Args:
        ctx: The MLIR context.
        name: The `Attribute` name.
        value: The `Attribute` value.

    Returns:
        An internal representation of an `Attribute`.
    """
    return _mlir.NamedAttribute(
        name=_mlir.Identifier(ctx, name),
        attr=_mlir.builtin_attributes.StringAttr(ctx, value),
    )


fn _shape_attr(
    ctx: _mlir.Context, name: String, shape: List[Dim]
) -> _mlir.NamedAttribute:
    """Creates a new `Shape`-valued `Attribute`.

    Args:
        ctx: The mlir.Context in which to create the type.
        name: The `Attribute` name.
        shape: The dimensions that make up the shape.

    Returns:
        An _mlir.Type in the specified Context.
    """
    var dims = List[_mlir.Attribute](capacity=len(shape))
    for i in range(len(shape)):
        dims.append(shape[i].to_mlir(ctx))
    return _mlir.NamedAttribute(
        name=_mlir.Identifier(ctx, name),
        attr=_c.attr_new_shape(ctx, dims),
    )
