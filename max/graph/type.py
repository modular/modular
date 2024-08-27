# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Library for graph value types."""

from __future__ import annotations

import functools
import math
import re
from dataclasses import dataclass
import sys
from typing import Any, Iterable, Union

if sys.version_info >= (3, 10):
    from typing import TypeGuard
else:
    from typing_extensions import TypeGuard

import numpy as np
from max import _graph, mlir
from max.dtype import DType


class Dim:
    """A tensor dimension.

    Tensor dimensions can be:

      - Static, aka known size
      - Dynamic, aka unknown size
      - Symbolic, aka unknown size but named

    In most cases you don't need to work with a `Dim` directly, but can rely
    on conversion constructors, for instance you can specify a tensor type as

    .. code-block:: python

        from max.graph import Dim, TensorType
        tensor_type = TensorType(DType.int64, ("batch", 10))

    will create a tensor type with 3 dimensions: a symbolic "batch" dimension,
    a static dimension of size 10, and a dynamic dimension.

    You can still construct dimensions explicitly via helpers, eg.

    .. code-block:: python

        some_dims = [
            Dim.symbolic("batch"),
            Dim.static(5),
        ]

    Constraining tensor dimensions is one important way to improve model
    performance. If tensors have unknown dimensions, we can't optimize them
    as aggressively. Symbolic tensors allow the compiler to learn constraints
    on a specific dimension (eg. if 2 inputs have the same `batch` dimension)
    which can be an important improvement over dynamic dimensions, but static
    dims are the easiest to optimize and therefore the easiest to create
    and work with.
    """

    def __new__(cls, value: DimLike):
        """Converts valid input values to Dim."""
        # There has to be a better pattern for this
        if isinstance(value, Dim):
            return super().__new__(type(value))
        elif isinstance(value, (int, np.integer)):
            return super().__new__(StaticDim)
        elif isinstance(value, str):
            return super().__new__(SymbolicDim)
        raise TypeError(f"Unsupported dimension type {value} ({type(value)})")

    def is_static(self) -> bool:
        """Checks whether or not the dimension is a static dimension.

        Returns:
            True if the dimension is static, False otherwise.
        """
        raise NotImplementedError

    def is_symbolic(self) -> bool:
        """Whether or not the dimension is a symbolic dimension.

        Returns:
            True if the dimension is symbolic, False otherwise.
        """
        raise NotImplementedError

    def __eq__(self, other: Any) -> bool:
        """Checks whether two dimensions are equal.

        Dimensions are equal if they are the same dimension type
        (symbolic, static). Additionally, static dimensions
        are only equal if their dimension is the same size, and symbolic
        dimensions are only equal if they have the same name.

        Args:
            other: The other dimension to check equality against.

        Returns:
            True if the dimensions are equal, False otherwise.
        """
        raise NotImplementedError

    def __ne__(self, other: Any) -> bool:
        """Checks whether two dimensions are not equal.

        The inverse of __eq__.

        Args:
            other: The other dimension to check inequality against.

        Returns:
            False if the dimensions are equal, True otherwise.
        """
        return not self == other

    def to_mlir(self) -> mlir.Attribute:
        """Creates an mlir.Attribute representing this dimension.

        This is used internally when constructing tensor MLIR types.

        Returns:
            An mlir.Attribute in the context representing the dimension.
        """
        raise NotImplementedError

    @staticmethod
    def from_mlir(dim_attr: mlir.Attribute) -> Dim:
        """Constructs a dimension from an mlir.Attribute.

        Args:
            dim_attr: The MLIR Attribute object to parse into a dimension.

        Returns:
            The dimension represented by the MLIR Attr value.
        """
        if _graph.is_static_dim(dim_attr):
            return StaticDim.from_mlir(dim_attr)
        elif _graph.is_symbolic_dim(dim_attr):
            return SymbolicDim.from_mlir(dim_attr)
        elif _graph.is_symbolic_expression_dim(dim_attr):
            raise ValueError(
                """
    Graph API currently doesn't support directly materializing algebraic dimensions.
    Please rebind the dimension before using it.

    For example:
    >>> x.reshape(["batch", -1]).rebind(["batch", "new_dim"])
    """
            )
        else:
            raise ValueError("graph api does not support unknown dimensions")


@dataclass(frozen=True)
class SymbolicDim(Dim):
    """A symbolic tensor dimension.

    `SymbolicDims`s have a name and are printed as their name on MO types, eg.
    `!mo.tensor<[batch, x, 10], si32]>` the first and second dimensions are
    named "batch" and "x" respectively.

    Symbolic dimensions don't have a static value, but they allow a readable
    name to understand what's going on in the model IR better, and they also
    allow users to hint to the compiler that two dimensions will have the same
    value, which can often allow important speedups.

    Create a symbolic dimension via `SymbolicDim("name")`, for example:
    `TensorType(DType.bool, ("batch", Dim.dynamic(), 10))`.
    """

    name: str
    """The name of the dimension."""

    def __init__(self, name: str | SymbolicDim):
        # Can't assign directly to frozen dataclasses.
        super().__setattr__("name", str(name))
        # TODO(MSDK-695): less restrictive names
        if not re.match(r"^[a-zA-Z_]\w*$", self.name):
            raise ValueError("Invalid name for symbolic dimension")

    def __str__(self) -> str:
        return self.name

    def is_static(self) -> bool:
        """Checks whether or not the dimension is a static dimension.

        Returns:
            True if the dimension is static, False otherwise.
        """
        return False

    def is_symbolic(self) -> bool:
        """Whether or not the dimension is a symbolic dimension.

        Returns:
            True if the dimension is symbolic, False otherwise.
        """
        return True

    def __eq__(self, other: Any) -> bool:
        """Whether the dimension is the same as another symbolic dimension.

        Symbolic dimensions with the same name are interpreted as the same
        dimensionality! If you use Symbolic dimensions, make sure you're naming
        them consistently, your model will likely fail to compile if you name
        two actually different dimensions the same name.

        Args:
            other: The other dimension to check equality against.

        Returns:
            True if the dimensions have the same name, False otherwise.
        """
        return self.name == other or (
            isinstance(other, SymbolicDim) and self.name == other.name
        )

    def to_mlir(self) -> mlir.Attribute:
        """Creates an mlir.Attribute representing this dimension.

        This is used internally when constructing tensor MLIR types.

        Returns:
            An mlir.Attribute in the context representing the dimension.
        """
        if not mlir.Context.current:
            raise RuntimeError("No active mlir Context.")
        return _graph.symbolic_dim(mlir.Context.current, self.name)

    @staticmethod
    def from_mlir(dim_attr: mlir.Attribute) -> Dim:
        """Constructs a dimension from an mlir.Attribute.

        Args:
            dim_attr: The MLIR Attribute object to parse into a dimension.

        Returns:
            The dimension represented by the MLIR Attr value.
        """
        return SymbolicDim(_graph.symbolic_dim_name(dim_attr))


@functools.total_ordering
@dataclass(frozen=True)
class StaticDim(Dim):
    """A static tensor dimension.

    Static tensor dimensions will always have exactly the same value,
    and are key to good model performance.

    Static dimensions can be created implicitly in most cases:
    `TensorType(DType.int64, (4, 5))` is a tensor with 2 static dimensions:
    `4` and `5` respectively.
    """

    dim: int
    """The size of the static dimension."""

    def __init__(self, dim: int | StaticDim):
        # Can't assign directly to frozen dataclasses.
        super().__setattr__("dim", int(dim))
        if not -1 <= self.dim < 2**63:
            raise ValueError("Dim value must be -1 <= dim < 2**63")

    def __int__(self) -> int:
        return self.dim

    def is_static(self) -> bool:
        """Checks whether or not the dimension is a static dimension.

        Returns:
            True if the dimension is static, False otherwise.
        """
        return True

    def is_symbolic(self) -> bool:
        """Whether or not the dimension is a symbolic dimension.

        Returns:
            True if the dimension is symbolic, False otherwise.
        """
        return False

    def __eq__(self, other: Any) -> bool:
        """Whether the dimension has the same size as another dimension.

        Args:
            other: The other dimension to check equality against.

        Returns:
            True if both dimensions have the same static size, False otherwise.
        """
        return self.dim == other or (
            isinstance(other, StaticDim) and self.dim == other.dim
        )

    def __lt__(self, other: Union[int, StaticDim]):
        return self.dim < (other.dim if isinstance(other, StaticDim) else other)

    def __hash__(self):
        return hash(self.dim)

    def to_mlir(self) -> mlir.Attribute:
        """Creates an mlir.Attribute representing this dimension.

        This is used internally when constructing tensor MLIR types.

        Returns:
            An mlir.Attribute in the context representing the dimension.
        """
        if not mlir.Context.current:
            raise RuntimeError("No active mlir Context.")
        return _graph.static_dim(mlir.Context.current, self.dim)

    @staticmethod
    def from_mlir(dim_attr: mlir.Attribute) -> Dim:
        """Constructs a dimension from an mlir.Attribute.

        Args:
            dim_attr: The MLIR Attribute object to parse into a dimension.

        Returns:
            The dimension represented by the MLIR Attr value.
        """
        return StaticDim(_graph.static_dim_value(dim_attr))


class Shape(list[Dim]):
    def __init__(self, dims: ShapeLike = ()):
        super().__init__(Dim(dim) for dim in dims)

    @property
    def rank(self):
        return len(self)

    def to_mlir(self) -> mlir.Attribute:
        return _graph.shape_attr(
            mlir.Context.current, [dim.to_mlir() for dim in self]
        )


StaticShape = list[StaticDim]

DimLike = Union[int, str, Dim, np.integer]
ShapeLike = Iterable[DimLike]


def _is_static_shape(dims: Shape) -> TypeGuard[StaticShape]:
    return all(dim.is_static() for dim in dims)


class Type:
    """Represents any possible type for Graph values.

    Every Value in the Graph has a Type, and that type is represented by an Type.
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
        self.shape = Shape(shape)

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
            [dim.to_mlir() for dim in self.shape],
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
            Dim.from_mlir(_graph.tensor_type_get_dim(t, i)) for i in range(rank)
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
        return all(d.is_static() for d in self.shape)

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
        This is the number of elements the tensor will hold *during execution*,
        TensorType doesn't actually hold any element values at all.

        For any non-static tensor, in other words a tensor having any symbolic
        dimensions, the return value will be meaningless.

        Returns:
            The number of elements the tensor contains.
        """
        if not _is_static_shape(self.shape):
            raise RuntimeError(
                "can't find num elements since tensor has symbolic dims"
            )

        return math.prod(dim.dim for dim in self.shape)

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
