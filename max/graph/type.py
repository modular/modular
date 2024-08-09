# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Library for graph value types."""

from __future__ import annotations

import math
import re
import typing
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    NewType,
    TypeGuard,
    Union,
)

from max import _graph, mlir

from .dtype import DType

# Hypothesis registration wants Dim to be a NewType, which
# allows us to register generation strategies against it;
# however MyPy really doesn't like NewTypes of types
# that aren't subclassable.
if TYPE_CHECKING:
    Dim = Union[int, str, mlir.Attribute]
else:
    Dim = NewType("Dim", Union[int, str, mlir.Attribute])  # type: ignore


def _dim_from_mlir(dim_attr: mlir.Attribute) -> Dim:
    """Constructs a dimension from an mlir.Attribute.

    Args:
        dim_attr: The MLIR Attribute object to parse into a dimension.

    Returns:
        The dimension represented by the MLIR Attr value.
    """
    if _graph.is_static_dim(dim_attr):
        return _graph.static_dim_value(dim_attr)
    elif _graph.is_symbolic_dim(dim_attr):
        return _graph.symbolic_dim_name(dim_attr)
    elif _graph.is_symbolic_expression_dim(dim_attr):
        return dim_attr
        # raise ValueError(
        #     "graph api does not support algebraic expression dimensions"
        # )
    else:
        raise ValueError("graph api does not support unknown dimensions")


def assert_legal_dim(dim: Dim):
    # TODO(MSDK-695): less restrictive names
    if isinstance(dim, str) and not re.match(r"^[a-zA-Z_]\w*$", dim):
        raise ValueError("Invalid name for symbolic dimension")
    elif isinstance(dim, int) and not -1 <= dim < 2**63:
        raise ValueError("Dim value must be -1 <= dim < 2**63")
    if not isinstance(dim, (int, str, mlir.Attribute)):
        typing.assert_never()
        raise TypeError(f"Invalid dim type: {dim!r}")


def _dim_to_mlir(dim: Dim) -> mlir.Attribute:
    """Creates an mlir.Attribute representing this dimension.

    This is used internally when constructing tensor MLIR types.

    Returns:
        An mlir.Attribute in the context representing the dimension.
    """
    assert_legal_dim(dim)
    if not mlir.Context.current:
        raise RuntimeError("No active mlir Context.")
    if isinstance(dim, int):
        return _graph.static_dim(mlir.Context.current, dim)
    elif isinstance(dim, str):
        return _graph.symbolic_dim(mlir.Context.current, dim)
    elif isinstance(dim, mlir.Attribute):
        return dim
    typing.assert_never()


Shape = list[Dim]
StaticShape = list[int]

ShapeLike = Iterable[Dim]


def shape(shape_like: ShapeLike) -> Shape:
    return list(shape_like)


def is_static_shape(dims: Shape) -> TypeGuard[StaticShape]:
    return all(isinstance(dim, int) for dim in dims)


@dataclass
class Type:
    """Represents any possible type for Graph values.

    Every GraphValue has a Type, and that type is represented by an Type.
    This type may be inspected to get finer-grained types and learn more
    about an individual Value.
    """

    def to_mlir(self) -> mlir.Type:
        """Converts to an mlir.Type instance.

        Returns:
            An mlir.Type in the specified Context.
        """
        raise NotImplementedError

    @staticmethod
    def from_mlir(t: mlir.Type) -> Type:
        """Constructs a type from an MLIR type.

        Args:
            t: The MLIR Type object to parse into a type.

        Returns:
            The type represented by the MLIR Type value.
        """
        raise NotImplementedError


@dataclass
class TensorType(Type):
    """A symbolic tensor type.

    This is _not_ an eager tensor type! This contains no actual data, but
    instead represents the type of a value at some point in time during model
    execution.

    Most internal values in a model will be tensors. This type represents
    their element type (dtype) and dimensions (dims) at a specific point during
    model computation. It allows us to do some optimistic optimizations and
    shape inference during graph construction, and to provide more detailed
    shape information to the compiler for further optimization passes.

    It can also represent a fully dynamic rank tensor. The presence of dynamic
    rank tensors in a graph will often degrade performance dramatically and
    prevents many classes of optimizations.
    """

    dtype: DType
    """The element type of the tensor value."""
    shape: Shape
    """The dimensions of the tensor value."""

    def __init__(self, dtype: DType, shape: ShapeLike) -> None:
        """Constructs a tensor type.

        Args:
            dtype: The element type of the tensor data.
            dims: The shape dimensions of the tensor. The number of dims
                  is the rank of the tensor.
        """
        self.dtype = dtype
        self.shape = list(shape)

    def to_mlir(self) -> mlir.Type:
        """Converts to an mlir.Type instance.

        Returns:
            An mlir.Type in the specified Context.
        """
        if not mlir.Context.current:
            raise RuntimeError("No active mlir Context.")
        return _graph.tensor_type(
            mlir.Context.current,
            _graph.dtype_type(mlir.Context.current, self.dtype._mlir),
            [_dim_to_mlir(dim) for dim in self.shape],
        )

    @staticmethod
    def from_mlir(t: mlir.Type) -> TensorType:
        """Constructs a tensor type from an MLIR type.

        Args:
            t: The MLIR Type object to parse into a tensor type.

        Returns:
            The tensor type represented by the MLIR Type value.
        """
        if not _graph.type_is_tensor(t):
            raise TypeError(f"Expected TensorType, got: {t}")

        dtype = _graph.tensor_type_get_dtype(t)
        rank = _graph.tensor_type_get_rank(t)
        shape = [
            _dim_from_mlir(_graph.tensor_type_get_dim(t, i))
            for i in range(rank)
        ]

        return TensorType(DType(dtype), shape)

    # ===------------------------------------------------------------------=== #
    # Basic accessors
    # ===------------------------------------------------------------------=== #

    def is_static(self) -> bool:
        """Checks whether the tensor type has a fully static shape or not.

        A tensor must have all of its dimensions be `static` (or be 0-dimensional)
        in order to be `static`.

        Returns:
            True if the tensor has a fully static shape, False otherwise.
        """
        return is_static_shape(self.shape)

    @property
    def rank(self) -> int:
        """Gets the rank of the tensor type.

        Returns:
            The tensor's static rank.
        """
        return len(self.shape)

    def dim(self, pos: int) -> Dim:
        """Gets the pos'th dimension of the tensor type.

        Supports negative-indexing, ie. `t.dim(-1)` will give the last
        dimension.

        Args:
            pos: The dimension index to retrieve.

        Returns:
            The dimension value at dimension `pos`.

        Raises:
            If the dimension is out-of-bounds.
        """
        return self.shape[pos + (self.rank if pos < 0 else 0)]

    def __eq__(self, other: Any) -> bool:
        """Checks whether the two tensors have the same rank, type, and shape.

        Args:
            other: The other tensor to check equality against.

        Returns:
            True if the tensors have identical element type and shape,
            False otherwise.
        """
        return (
            isinstance(other, TensorType)
            and (self.dtype == other.dtype)
            and (self.rank == other.rank)
            and all(d == d_other for d, d_other in zip(self.shape, other.shape))
        )

    # ===------------------------------------------------------------------=== #
    # Utilities
    # ===------------------------------------------------------------------=== #

    def num_elements(self) -> int:
        """Counts the total number of elements in the tensor type.

        For a static tensor, returns the product of all static dimensions.
        This is the number of elements the tensor will hold _during execution_,
        TensorType doesn't actually hold any element values at all.

        For any non-static tensor, in other words a tensor having any symbolic
        dimensions, the return value will be meaningless.

        Returns:
            The number of elements the tensor contains.
        """
        if not is_static_shape(self.shape):
            raise RuntimeError(
                "can't find num elements since tensor has symbolic dims"
            )

        return math.prod(self.shape)

    def cast(self, dtype: DType) -> TensorType:
        """Constructs a new tensor type of the same shape with the new dtype.

        Args:
            dtype: The new element type for the tensor.

        Returns:
            A new tensor type with the same shape, and the new element type.
        """
        return TensorType(dtype, self.shape)


@dataclass
class _OpaqueType(Type):
    """A type representing an opaque type."""

    name: str
    """Identifier for the opaque type."""

    def to_mlir(self) -> mlir.Type:
        """Converts to an mlir.Type instance.

        Returns:
            An mlir.Type in the specified Context.
        """
        if not mlir.Context.current:
            raise RuntimeError("No active mlir Context.")
        return _graph.opaque_type(mlir.Context.current, self.name)

    @staticmethod
    def from_mlir(t: mlir.Type) -> _OpaqueType:
        """Constructs an opaque type from an MLIR type.

        Args:
            t: The MLIR Type object to parse into an opaque type.

        Returns:
            The opaque type represented by the MLIR Type value.
        """
        return _OpaqueType(_graph.opaque_type_name(t))
