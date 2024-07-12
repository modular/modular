# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Library for graph value types."""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable, Union

import max.graph.core as _c
from max.graph import mlir
from max.graph.dtype import DType


@dataclass
class Dim:
    """A tensor dimension.

    Tensor dimensions can be
    - Static, aka known size
    - Dynamic, aka unknown size
    - Symbolic, aka unknown size but named

    In most cases you don't need to work with a `Dim` directly, but can rely
    on conversion constructors, for instance you can specify a tensor type as

    ```mojo
    from max.graph import Dim, TensorType
    tensor_type = TensorType(DType.int64, "batch", 10)
    ```
    will create a tensor type with 3 dimensions: a symbolic "batch" dimension,
    a static dimension of size 10, and a dynamic dimension.
    ```

    You can still construct dimensions explicitly via helpers, eg.

    ```python
    some_dims = [
        Dim.symbolic("batch"),
        Dim.static(5),
    ]
    ```

    Constraining tensor dimensions is one important way to improve model
    performance. If tensors have unknown dimensions, we can't optimize them
    as aggressively. Symoblic tensors allow the compiler to learn constraints
    on a specific dimension (eg. if 2 inputs have the same `batch` dimension)
    which can be an important improvement over dynamic dimensions, but static
    dims are the easiest to optimize and therefore the easiest to create
    and work with.
    """


@dataclass
class Type:
    """Represents any possible type for Graph values.

    Every GraphValue has a Type, and that type is represented by an Type.
    This type may be inspected to get finer-grained types and learn more
    about an individual Value.
    """

    def to_mlir(self, ctx: mlir.ir.Context) -> mlir.ir.Type:
        """Converts to an mlir.ir.Type instance.

        Args:
            ctx: The mlir.ir.Context in which to create the type.

        Returns:
            An mlir.ir.Type in the specified Context.
        """
        raise NotImplementedError

    @staticmethod
    def from_mlir(t: mlir.ir.Type) -> Type:
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
    shape information to the compiler for further optimizatino passes.

    It can also represent a fully dynamic rank tensor. The presence of dynamic
    rank tensors in a graph will often degrade performance dramatically and
    prevents many classes of optimizations.
    """

    dtype: DType
    """The element type of the tensor value."""
    dims: list[Union[int, Dim]]
    """The dimensions of the tensor value."""

    def __init__(self, dtype: DType, dims: Iterable[Union[int, Dim]]) -> None:
        """Constructs a tensor type.

        Args:
            dtype: The element type of the tensor data.
            dims: The shape dimensions of the tensor. The number of dims
                  is the rank of the tensor.
        """
        self.dtype = dtype
        self.dims = list(dims)

    def to_mlir(self, ctx: mlir.ir.Context) -> mlir.ir.Type:
        """Converts to an _mlir.Type instance.

        Args:
            ctx: The mlir.ir.Context in which to create the type.

        Returns:
            An _mlir.Type in the specified Context.
        """
        return _c.tensor_type_new(
            ctx,
            _c.dtype_new(ctx, self.dtype),
            dims=[mlir.ir.Attribute(d.to_mlir(ctx)) for d in self.dims],
            ranked=True,
        )

    @staticmethod
    def from_mlir(t: mlir.ir.Type) -> TensorType:
        """Constructs a tensor type from an MLIR type.

        Args:
            t: The MLIR Type object to parse into a tensor type.

        Returns:
            The tensor type represented by the MLIR Type value.
        """
        dtype = _c.tensor_type_get_dtype(t)
        rank = _c.tensor_type_get_rank(t)
        dims = [
            Dim.from_mlir(_c.tensor_type_get_dim(t, i)) for i in range(rank)
        ]

        return TensorType(dtype, dims)

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
        return not any(d.is_dynamic() for d in self.dims)

    def rank(self) -> int:
        """Gets the rank of the tensor type.

        Returns:
            The tensor's static rank.
        """
        return len(self.dims)

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
        return self.dims[pos + (self.rank() if pos < 0 else 0)]

    def __eq__(self, other: TensorType) -> bool:
        """Checks whether the two tensors have the same rank, type, and shape.

        Args:
            other: The other tensor to check equality against.

        Returns:
            True if the tensors have identical element type and shape,
            False otherwise.
        """
        return (
            (self.dtype == other.dtype)
            and (self.rank() == other.rank())
            and all(d == d_other for d, d_other in zip(self.dims, other.dims))
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
        if not self.is_static():
            raise "can't find num elements since tensor has symbolic dims"

        return math.prod(self.dims)

    def cast(self, dtype: DType) -> TensorType:
        """Constructs a new tensor type of the same shape with the new dtype.

        Args:
            dtype: The new element type for the tensor.

        Returns:
            A new tensor type with the same shape, and the new element type.
        """
        return TensorType(dtype, self.dims)


@dataclass
class _OpaqueType(Type):
    """A type representing an opaque type."""

    name: str
    """Identifier for the opaque type."""

    def to_mlir(self, ctx: mlir.ir.Context) -> mlir.ir.Type:
        """Converts to an mlir.ir.Type instance.

        Args:
            ctx: The mlir.ir.Context in which to create the type.

        Returns:
            An mlir.ir.Type in the specified Context.
        """
        return _c.opaque_type_new(ctx, self.name)

    @staticmethod
    def from_mlir(t: mlir.ir.Type) -> _OpaqueType:
        """Constructs an opaque type from an MLIR type.

        Args:
            t: The MLIR Type object to parse into an opaque type.

        Returns:
            The opaque type represented by the MLIR Type value.
        """
        return _OpaqueType(_c.opaque_type_name(t))
