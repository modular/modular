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
import sys
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Iterable, Optional, Union

if sys.version_info >= (3, 10):
    from typing import TypeGuard
else:
    from typing_extensions import TypeGuard

import numpy as np
from max import mlir
from max._core import graph as _graph
from max.dtype import DType


class Dim:
    """A tensor dimension.

    Tensor dimensions can be one of three types:

    - **Static**: Known size
    - **Symbolic**: Unknown size but named
    - **Algebraic**: Unknown size has an algebraic expression


    In most cases, you don't need to work with a ``Dim`` directly.
    Instead, use conversion constructors:


    .. code-block:: python

        from max.graph import Dim, TensorType

        tensor_type = TensorType(DType.int64, ("batch", 10))

    This creates a tensor type with three dimensions:

    - A symbolic "batch" dimension
    - A static dimension of size 10

    For explicit dimension construction, use the following helpers:

    .. code-block:: python

        from max.graph import Dim

        some_dims = [
            SymbolicDim("batch"),
            StaticDim(5),
            AlgebraicDim(Dim("batch") + 1),
        ]

    Constraining tensor dimensions is one important way to improve model
    performance. If tensors have unknown dimensions, we can't optimize them
    as aggressively. Symbolic tensors allow the compiler to learn constraints
    on a specific dimension (eg. if 2 inputs have the same `batch` dimension),
    but static dims are the easiest to optimize and therefore the easiest to
    create and work with.
    """

    def __new__(cls, value: DimLike):
        """Converts valid input values to Dim."""
        if cls is not Dim:
            # Create subclass if given instead of redirecting to Dim.
            return super().__new__(cls)

        if isinstance(value, Dim):
            # Directly return existing Dim instance.
            return value
        elif isinstance(value, (int, np.integer)):
            return super().__new__(StaticDim)
        elif isinstance(value, str):
            return super().__new__(SymbolicDim)
        elif isinstance(value, mlir.Attribute):
            return super().__new__(AlgebraicDim)

        msg = f"Unsupported dimension type {value} ({type(value)})"
        raise TypeError(msg)

    def __index__(self) -> int:
        """Converts this dim to an index as used by indexing and slicing.

        This raises and suggests explicitly converting to int, so that we only
        support implicit slicing operations on TensorValues.
        Types such as list and np.ndarray call __index__ on inputs to their
        __getitem__ special methods to convert those inputs to int.

        This also prevents a MyPy false positive error: Slice index must be an
        integer or None.
        Related MyPy discussion: https://github.com/python/mypy/issues/2410
        """
        msg = (
            "when using dims to index into a list or NumPy array, explicitly "
            "convert to int with int(dim)"
        )
        raise TypeError(msg)

    def __int__(self) -> int:
        """Converts this dim to an int by casting to StaticDim."""
        return int(StaticDim(self))

    def __eq__(self, other: Any) -> bool:
        """Checks whether two dimensions are equal.

        Dimensions are equal if they are the same dimension type
        (symbolic, static). Additionally, static dimensions
        are only equal if their dimension is the same size, and symbolic
        dimensions are only equal if they have the same name.

        Args:
            other: The other dimension to check equality against.

        Returns:
            True if the dimensions are equal, false otherwise.
        """
        raise NotImplementedError

    def __ne__(self, other: Any) -> bool:
        """Checks whether two dimensions are not equal.

        The inverse of __eq__.

        Args:
            other: The other dimension to check inequality against.

        Returns:
            False if the dimensions are equal, true otherwise.
        """
        return not self == other

    @staticmethod
    def _algebraic_op(op: AlgebraicDim._Opcode, lhs: Dim, rhs: Dim) -> Dim:
        if not mlir.Context.current:
            raise RuntimeError("No active mlir Context.")
        return Dim.from_mlir(
            _graph.algebraic_dim(
                mlir.Context.current,
                op,
                lhs.to_mlir(),
                rhs.to_mlir(),
            )
        )

    def __add__(self, rhs: DimLike) -> Dim:
        rhs = Dim(rhs)
        return Dim._algebraic_op(AlgebraicDim._Opcode.Add, self, rhs)

    # hitting https://github.com/python/mypy/issues/11595 which causes mypy to fail to typecheck.
    def __radd__(self, lhs: DimLike) -> Dim:  # type: ignore
        lhs = Dim(lhs)
        return lhs + self

    def __mul__(self, rhs: DimLike) -> Dim:
        rhs = Dim(rhs)
        return Dim._algebraic_op(AlgebraicDim._Opcode.MulNuw, self, rhs)

    # hitting https://github.com/python/mypy/issues/11595 which causes mypy to fail to typecheck.
    def __rmul__(self, lhs: DimLike) -> Dim:  # type: ignore
        lhs = Dim(lhs)
        return lhs * self

    def __neg__(self) -> Dim:
        return -1 * self

    def __sub__(self, rhs: DimLike) -> Dim:
        rhs = Dim(rhs)
        return self + -rhs

    def __rsub__(self, lhs: DimLike) -> Dim:
        lhs = Dim(lhs)
        return lhs + -self

    def __floordiv__(self, rhs: DimLike) -> Dim:
        rhs = Dim(rhs)
        return Dim._algebraic_op(AlgebraicDim._Opcode.Div, self, rhs)

    def __rfloordiv__(self, lhs: DimLike) -> Dim:
        lhs = Dim(lhs)
        return lhs // self

    def to_mlir(self) -> mlir.Attribute:
        """Creates an ``mlir.Attribute`` representing this dimension.

        This is used internally when constructing tensor MLIR types.

        Returns:
            An ``mlir.Attribute`` in the context representing the dimension.
        """
        raise NotImplementedError

    @staticmethod
    def from_mlir(dim_attr: mlir.Attribute) -> Dim:
        """Constructs a dimension from an ``mlir.Attribute``.

        Args:
            dim_attr: The MLIR Attribute object to parse into a dimension.

        Returns:
            Dim: The dimension represented by the MLIR Attr value.
        """
        if _graph.is_static_dim(dim_attr):
            return StaticDim.from_mlir(dim_attr)
        elif _graph.is_symbolic_dim(dim_attr):
            return SymbolicDim.from_mlir(dim_attr)
        elif _graph.is_algebraic_dim(dim_attr):
            return AlgebraicDim.from_mlir(dim_attr)
        else:
            raise ValueError("graph api does not support unknown dimensions")


@dataclass(frozen=True)
class SymbolicDim(Dim):
    """A symbolic tensor dimension.

    Symbolic dimensions represent named dimensions in MO tensor types.

    Symbolic dimensions don't have a static value, but they allow a readable
    name to understand what's going on in the model IR better, and they also
    allow users to hint to the compiler that two dimensions will have the same
    value, which can often allow important speedups.

    In tensor type notation:

    .. code-block:: python

        !mo.tensor<[batch, x, 10], si32]>

    The first and second dimensions are named ``batch`` and ``x`` respectively.

    Creating a ``SymbolicDim``:

    .. code-block:: python

        dim = SymbolicDim("name")

    Using ``SymbolicDim`` in a :obj:`TensorType`:

    .. code-block:: python

        tensor_type = TensorType(DType.bool, (SymbolicDim("batch"), Dim.dynamic(), 10))
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

    def __repr__(self):
        return f"Dim({repr(self.name)})"

    def __eq__(self, other: Any) -> bool:
        """Whether the dimension is the same as another symbolic dimension.

        Symbolic dimensions with the same name are interpreted as the same
        dimensionality! If you use Symbolic dimensions, make sure you're naming
        them consistently, your model will likely fail to compile if you name
        two actually different dimensions the same name.

        Args:
            other: The other dimension to check equality against.
        Returns:
            True if the dimensions have the same name, false otherwise.
        """
        return self.name == other or (
            isinstance(other, SymbolicDim) and self.name == other.name
        )

    def to_mlir(self) -> mlir.Attribute:
        """Creates an ``mlir.Attribute`` representing this dimension.

        This is used internally when constructing tensor MLIR types.

        Returns:
            An ``mlir.Attribute`` in the context representing the dimension.
        """
        if not mlir.Context.current:
            raise RuntimeError("No active mlir Context.")
        return _graph.symbolic_dim(mlir.Context.current, self.name)

    @staticmethod
    def from_mlir(dim_attr: mlir.Attribute) -> Dim:
        """Constructs a dimension from an ``mlir.Attribute``.

        Args:
            dim_attr: The MLIR Attribute object to parse into a dimension.

        Returns:
            Dim: The dimension represented by the MLIR Attr value.
        """
        return SymbolicDim(_graph.symbolic_dim_name(dim_attr))


@dataclass(frozen=True)
class AlgebraicDim(Dim):
    """An algebraic tensor dimension to enable expressions over symbolic
    dimensions.

    That is, any expression over a symbolic dimension returns ``AlgebraicDim``.
    Furthermore, algebraic dimensions automatically simplify into a canonical
    form.

    For example:

        >>> from max.graph import AlgebraicDim, Dim
        >>> isinstance(Dim("batch") * 5, AlgebraicDim)
        True
        >>> print(Dim("batch") * 5)
        batch * 5
        >>> -Dim("x") - 4 == -(Dim("x") + 4)
        True
    """

    attr: mlir.Attribute

    class _Opcode(IntEnum):
        # This is only the part of KGEN::POC that seems relevant to python.
        # On top of that, this is starting extra slim to keep things simple.
        # We can expand at any time if needed.
        Add = 0
        # Mul = 1 : We probably never want Mul, instead we want MulNuw which blocks overflow.
        MulNuw = 2
        # And = 3
        # Or = 4
        # Xor = 5
        # Max = 6
        # Min = 7
        # Shl = 8
        # Shr = 9
        Div = 10
        # Mod = 11

    def __init__(self, attr: mlir.Attribute | AlgebraicDim):
        if isinstance(attr, mlir.Attribute) and not _graph.is_algebraic_dim(
            attr
        ):
            raise TypeError(
                f"Cannot create AlgebraicDim from mlir attribute: {attr}"
            )

        # Can't assign directly to frozen dataclasses.
        super().__setattr__(
            "attr", attr.attr if isinstance(attr, AlgebraicDim) else attr
        )

    def __str__(self):
        return self.to_str(False)

    def __repr__(self):
        return self.to_str(True)

    def to_str(self, use_repr: bool):
        opcode = _graph.algebraic_dim_opcode(self.attr)
        operands = _graph.algebraic_dim_operands(self.attr)
        operand_dims = (Dim.from_mlir(operand) for operand in operands)

        # Make sure to wrap nested expression.
        def str_fn(x) -> str:
            if use_repr:
                return repr(x)
            else:
                return str(x)

        operand_reprs = (
            f"({str_fn(dim)})" if isinstance(dim, AlgebraicDim) else str_fn(dim)
            for dim in operand_dims
        )

        # For the opcodes we support in the graph api, print with python math.
        if opcode == AlgebraicDim._Opcode.Add:
            return " + ".join(operand_reprs)
        elif opcode == AlgebraicDim._Opcode.MulNuw:
            return " * ".join(operand_reprs)
        elif opcode == AlgebraicDim._Opcode.Div:
            return " // ".join(operand_reprs)
        else:
            return str_fn(self.attr)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, AlgebraicDim) and self.attr == other.attr

    def to_mlir(self) -> mlir.Attribute:
        """Creates an mlir.Attribute representing this dimension.
        This is used internally when constructing tensor MLIR types.

        Returns:
            An mlir.Attribute in the context representing the dimension.
        """
        return self.attr

    @staticmethod
    def from_mlir(dim_attr: mlir.Attribute) -> Dim:
        return AlgebraicDim(dim_attr)


@functools.total_ordering
@dataclass(frozen=True)
class StaticDim(Dim):
    """A static tensor dimension.

    Static tensor dimensions will always have exactly the same value,
    and are key to good model performance.

    Static dimensions can be created implicitly in most cases:

    ``TensorType(DType.int64, (4, 5))`` is a tensor with 2 static dimensions:
    ``4`` and ``5`` respectively.
    """

    dim: int
    """The size of the static dimension."""

    def __init__(self, dim: int | Dim):
        if not isinstance(dim, (StaticDim, int)):
            msg = "expected statically known dim"
            raise TypeError(msg)

        # Can't assign directly to frozen dataclasses.
        super().__setattr__("dim", int(dim))
        if not -(2**63) <= self.dim < 2**63:
            raise ValueError("Dim value must be -2**63 <= dim < 2**63")

    def __str__(self):
        return str(self.dim)

    def __repr__(self):
        return f"Dim({repr(self.dim)})"

    def __int__(self) -> int:
        return self.dim

    def __eq__(self, other: Any) -> bool:
        """Whether the dimension has the same size as another dimension.

        Args:
            other: The other dimension to check equality against.

        Returns:
            True if both dimensions have the same static size, false otherwise.
        """
        return self.dim == other or (
            isinstance(other, StaticDim) and self.dim == other.dim
        )

    def __lt__(self, other: Union[int, StaticDim]):
        return self.dim < (other.dim if isinstance(other, StaticDim) else other)

    def __hash__(self):
        return hash(self.dim)

    def to_mlir(self) -> mlir.Attribute:
        """Creates an ``mlir.Attribute`` representing this dimension.

        This is used internally when constructing tensor MLIR types.

        Returns:
            An ``mlir.Attribute`` in the context representing the dimension.
        """
        if not mlir.Context.current:
            raise RuntimeError("No active mlir Context.")
        return _graph.static_dim(mlir.Context.current, self.dim)

    @staticmethod
    def from_mlir(dim_attr: mlir.Attribute) -> Dim:
        """Constructs a dimension from an ``mlir.Attribute``.

        Args:
            dim_attr: The MLIR Attribute object to parse into a dimension.

        Returns:
            The dimension represented by the MLIR Attr value.
        """
        return StaticDim(_graph.static_dim_value(dim_attr))


def _is_static_shape(dims: Shape) -> TypeGuard[StaticShape]:
    return all(isinstance(dim, StaticDim) and dim.dim >= 0 for dim in dims)


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

    @property
    def static_dims(self) -> list[int]:
        """Returns all static dims in the shape as a list of integers."""
        return [StaticDim(d).dim for d in self if isinstance(d, StaticDim)]


@dataclass(frozen=True)
class DeviceKind(str, Enum):
    """A device type representation."""

    CPU = "cpu"
    GPU = "gpu"

    def __str__(self) -> str:
        return self.value


class DeviceRef:
    """A symbolic device representation.

    DeviceRef type representation consists of a DeviceKind and an id. This is a direct
    representation of the device attribute in mlir.
    """

    device_type: DeviceKind
    id: int

    @staticmethod
    def CPU(id: int = 0) -> DeviceRef:
        """Static Method for creating a CPU device."""
        return DeviceRef(DeviceKind.CPU, id)

    @staticmethod
    def GPU(id: int = 0) -> DeviceRef:
        """Static Method for creating a GPU device."""
        return DeviceRef(DeviceKind.GPU, id)

    def __init__(self, device_type: Union[DeviceKind, str], id: int = 0):
        if isinstance(device_type, DeviceKind):
            self.device_type = device_type
        else:
            self.device_type = DeviceKind(device_type)
        if id < 0:
            id = 0
        self.id = id

    def __str__(self) -> str:
        return str(self.device_type) + ":" + str(self.id)

    def __repr__(self):
        return str(self)

    def __eq__(self, other: Any) -> bool:
        """Returns true if devices are equal."""
        return self.device_type is other.device_type and self.id == other.id

    def to_mlir(self) -> mlir.Attribute:
        """Returns a mlir attribute representing device."""
        return _graph.device_attr(
            mlir.Context.current, str(self.device_type), self.id
        )

    @staticmethod
    def from_mlir(device_attr: mlir.Attribute) -> DeviceRef:
        """Returns a device from mlir attribute"""
        return DeviceRef(
            device_type=DeviceKind(_graph.device_attr_get_label(device_attr)),
            id=_graph.device_attr_get_id(device_attr),
        )


StaticShape = list[StaticDim]

DimLike = Union[int, str, Dim, np.integer]
ShapeLike = Iterable[DimLike]


class Type:
    """Represents any possible type for Graph values.

    Every Value in the Graph has a Type, and that type is represented by an Type.
    This type may be inspected to get finer-grained types and learn more
    about an individual Value.
    """

    def to_mlir(self) -> mlir.Type:
        """Converts to an ``mlir.Type`` instance.

        Returns:
            An ``mlir.Type`` in the specified Context.
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
    """A symbolic :obj:`TensorType`.

    This is not an eager tensor type! This contains no actual data, but
    instead represents the type of a value at some point in time during model
    execution.

    Most internal values in a model will be tensors. This type represents
    their element type (``dtype``) and dimensions (``dims``) at a specific point during
    model computation. It allows us to do some optimistic optimizations and
    shape inference during graph construction, and to provide more detailed
    shape information to the compiler for further optimization passes.

    It can also represent a fully dynamic rank tensor. The presence of dynamic
    rank tensors in a graph will often degrade performance dramatically and
    prevents many classes of optimizations.

    An optional device (``device``) can also be provided to indicate the explicit
    device the tensor is associated with.
    """

    dtype: DType
    """The element type of the tensor value."""
    shape: Shape
    """The dimensions of the tensor value."""
    device: Optional[DeviceRef]
    """The device of the tensor value."""

    def __init__(
        self,
        dtype: DType,
        shape: ShapeLike,
        device: Optional[DeviceRef] = None,
    ) -> None:
        """Constructs a tensor type.

        Args:
            dtype: The element type of the tensor data.
            dims: The shape dimensions of the tensor. The number of dims
                  is the rank of the tensor.
        """
        self.dtype = dtype
        self.shape = Shape(shape)
        self.device = device

    def to_mlir(self) -> mlir.Type:
        """Converts to an ``mlir.Type`` instance.

        Returns:
            An ``mlir.Type`` in the specified Context.
        """
        if not mlir.Context.current:
            raise RuntimeError("No active mlir Context.")
        if self.device:
            return _graph.tensor_type_with_device(
                mlir.Context.current,
                _graph.dtype_type(mlir.Context.current, self.dtype._mlir),
                [dim.to_mlir() for dim in self.shape],
                self.device.to_mlir(),
            )
        else:
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
        mlir_device = _graph.tensor_type_get_device(t)
        device = DeviceRef.from_mlir(mlir_device) if mlir_device else None
        return TensorType(DType(dtype), shape, device)

    # ===------------------------------------------------------------------=== #
    # Basic accessors
    # ===------------------------------------------------------------------=== #

    @property
    def rank(self) -> int:
        """Gets the rank of the tensor type.

        Returns:
            The tensor's static rank.
        """
        return len(self.shape)

    def dim(self, pos: int) -> Dim:
        """Gets the ``pos``'th dimension of the tensor type.

        Supports negative-indexing, ie. ``t.dim(-1)`` will give the last
        dimension.

        Args:
            pos: The dimension index to retrieve.

        Returns:
            The dimension value at dimension ``pos``.

        Raises:
            RuntimeError: If the dimension is out-of-bounds.
        """
        return self.shape[pos + (self.rank if pos < 0 else 0)]

    def __eq__(self, other: Any) -> bool:
        """Checks whether the two tensors have the same rank, type, and shape.

        Args:
            other: The other tensor to check equality against.

        Returns:
            True if the tensors have identical element type and shape,
            false otherwise.
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

    def as_buffer(self) -> BufferType:
        """Returns the analogous buffer type."""
        return BufferType(self.dtype, self.shape, self.device)

    def num_elements(self) -> int:
        """Counts the total number of elements in the tensor type.

        For a static tensor, returns the product of all static dimensions.
        This is the number of elements the tensor will hold **during execution**,
        :obj:`TensorType` doesn't actually hold any element values at all.

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
        """Constructs a new tensor type of the same shape with the new `dtype`.

        Args:
            dtype: The new element type for the tensor.

        Returns:
            A new tensor type with the same shape, device, and the new element type.
        """
        return TensorType(dtype, self.shape, self.device)


@dataclass
class BufferType(Type):
    """A symbolic buffer type.

    This is a reference to a tensor that can be mutated in place.
    """

    dtype: DType
    """The element type of the buffer value."""
    shape: Shape
    """The dimensions of the buffer value."""
    device: Optional[DeviceRef]
    """The device of the tensor value."""

    def __init__(
        self,
        dtype: DType,
        shape: ShapeLike,
        device: Optional[DeviceRef] = None,
    ) -> None:
        """Constructs a buffer type.

        Args:
            dtype: The element type of the buffer data.
            dims: The shape dimensions of the buffer. The number of dims
                  is the rank of the buffer.
        """
        self.dtype = dtype
        self.shape = Shape(shape)
        self.device = device

    def to_mlir(self) -> mlir.Type:
        """Converts to an ``mlir.Type`` instance.

        Returns:
            An ``mlir.Type`` in the specified Context.
        """
        if not mlir.Context.current:
            raise RuntimeError("No active mlir Context.")
        if self.device:
            return _graph.buffer_type_with_device(
                mlir.Context.current,
                _graph.dtype_type(mlir.Context.current, self.dtype._mlir),
                [dim.to_mlir() for dim in self.shape],
                self.device.to_mlir(),
            )
        else:
            return _graph.buffer_type(
                mlir.Context.current,
                _graph.dtype_type(mlir.Context.current, self.dtype._mlir),
                [dim.to_mlir() for dim in self.shape],
            )

    @staticmethod
    def from_mlir(t: mlir.Type) -> BufferType:
        """Constructs a buffer type from an MLIR type.

        Args:
            t: The MLIR Type object to parse into a buffer type.

        Returns:
            The buffer type represented by the MLIR Type value.
        """
        if not _graph.type_is_buffer(t):
            raise TypeError(f"Expected BufferType, got: {t}")

        dtype = _graph.buffer_type_get_dtype(t)
        rank = _graph.buffer_type_get_rank(t)
        shape = [
            Dim.from_mlir(_graph.buffer_type_get_dim(t, i)) for i in range(rank)
        ]
        mlir_device = _graph.buffer_type_get_device(t)
        device = DeviceRef.from_mlir(mlir_device) if mlir_device else None

        return BufferType(DType(dtype), shape, device)

    # ===------------------------------------------------------------------=== #
    # Basic accessors
    # ===------------------------------------------------------------------=== #

    @property
    def rank(self) -> int:
        """Gets the rank of the buffer type.

        Returns:
            The buffer's static rank.
        """
        return len(self.shape)

    def dim(self, pos: int) -> Dim:
        """Gets the pos'th dimension of the buffer type.

        Supports negative-indexing, ie. ``t.dim(-1)`` will give the last
        dimension.

        Args:
            pos: The dimension index to retrieve.

        Returns:
            The dimension value at dimension ``pos``.

        Raises:
            If the dimension is out-of-bounds.
        """
        return self.shape[pos + (self.rank if pos < 0 else 0)]

    def __eq__(self, other: Any) -> bool:
        """Checks whether the two buffers have the same rank, type, and shape.

        Args:
            other: The other buffer to check equality against.

        Returns:
            True if the buffers have identical element type and shape,
            false otherwise.
        """
        return (
            isinstance(other, BufferType)
            and (self.dtype == other.dtype)
            and (self.rank == other.rank)
            and all(d == d_other for d, d_other in zip(self.shape, other.shape))
        )

    # ===------------------------------------------------------------------=== #
    # Utilities
    # ===------------------------------------------------------------------=== #

    def as_tensor(self) -> TensorType:
        """Returns the analogous tensor type."""
        return TensorType(self.dtype, self.shape, self.device)

    def num_elements(self) -> int:
        """Counts the total number of elements in the buffer type.

        For a static buffer, returns the product of all static dimensions.
        This is the number of elements the buffer will hold **during execution**,
        BufferType doesn't actually hold any element values at all.

        For any non-static buffer, in other words a buffer having any symbolic
        dimensions, the return value will be meaningless.

        Returns:
            The number of elements the buffer contains.
        """
        if not _is_static_shape(self.shape):
            raise RuntimeError(
                "can't find num elements since buffer has symbolic dims"
            )

        return math.prod(dim.dim for dim in self.shape)

    def cast(self, dtype: DType) -> BufferType:
        """Constructs a new buffer type of the same shape with the new dtype.

        Args:
            dtype: The new element type for the buffer.

        Returns:
            A new buffer type with the same shape, and the new element type.
        """
        return BufferType(dtype, self.shape)


@dataclass(frozen=True)
class _OpaqueType(Type):
    """A type representing an opaque type."""

    name: str
    """Identifier for the opaque type."""

    def to_mlir(self) -> mlir.Type:
        """Converts to an ``mlir.Type`` instance.

        Returns:
            An ``mlir.Type`` in the specified Context.
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


@dataclass
class _ChainType(Type):
    """A chain type.

    Used in order to sequence operations that have side-effects.

    As a user you should never need to directly interact with this type.
    """

    def to_mlir(self) -> mlir.Type:
        """Converts to an mlir.Type instance.

        Returns:
            An mlir.Type in the specified Context.
        """
        if not mlir.Context.current:
            raise RuntimeError("No active mlir Context.")
        return _graph.chain_type(mlir.Context.current)

    @staticmethod
    def from_mlir(t: mlir.Type) -> _ChainType:
        """Constructs an opaque type from an MLIR type.

        Args:
            t: The MLIR Type object to parse into an opaque type.

        Returns:
            The opaque type represented by the MLIR Type value.
        """
        if not _graph.type_is_chain(t):
            raise TypeError(f"Expected _ChainType, got: {t}")
        return _ChainType()
