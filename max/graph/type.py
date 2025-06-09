# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Library for graph value types."""

from __future__ import annotations

import enum
import functools
import math
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Generic, TypeVar, Union

if sys.version_info >= (3, 10):
    from typing import TypeGuard
else:
    from typing_extensions import TypeGuard

if sys.version_info >= (3, 11):
    from typing import Self
else:
    Self = TypeVar("Self")

import numpy as np
from max._core import NamedAttribute
from max._core import Type as _Type
from max._core.dialects import builtin, kgen, m, mo, mosh
from max.driver import CPU, Accelerator, Device
from max.dtype import DType

MlirType = TypeVar("MlirType", bound=_Type)


class FilterLayout(enum.Enum):
    RSCF = "RSCF"
    QRSCF = "QRSCF"
    FCRS = "FCRS"
    FCQRS = "FCQRS"
    CFRS = "CFRS"

    def to_mlir(self) -> mo.LayoutAttr:
        """Returns an mlir Attribute representing this Layout.
        This attribute is used in tensor type metadata for certain ops.

        Returns:
            An Attribute representing the layout.
        """
        return mo.LayoutAttr(format_str=str(self.value))

    @staticmethod
    def from_mlir(attr: mo.LayoutAttr) -> FilterLayout:
        """Constructs a layout from an attribute.

        Args:
            attr: The MLIR Attribute object to parse into a layout.

        Returns:
            The FilterLayout represented by the Attribute value.
        """
        return FilterLayout(attr.format.value)


class ConvInputLayout(enum.Enum):
    # TODO(GEX-2302): We need to differentiate between 2D - 3D layouts.
    # TODO(GEX-2302): Another Layout type should be used instead of StringAttr.
    # Simpler implementation to quickly support CUDNN input formats.
    NHWC = "NHWC"
    NCHW = "NCHW"

    def to_mlir(self) -> builtin.StringAttr:
        """Returns an mlir Attribute representing this Layout.
        This attribute is used for certain convolution ops.

        Returns:
            An Attribute representing the layout.
        """
        return builtin.StringAttr(self.value)

    @staticmethod
    def from_mlir(attr: builtin.StringAttr) -> ConvInputLayout:
        """Constructs a layout from an attribute.

        Args:
            attr: The MLIR Attribute object to parse into a layout.

        Returns:
            The FilterLayout represented by the Attribute value.
        """
        return ConvInputLayout(attr.value)


class Dim:
    """A tensor dimension.

    Tensor dimensions can be one of three types:

    - **Static**: Known size
    - **Symbolic**: Unknown size but named
    - **Algebraic**: Unknown size has an algebraic expression


    In most cases, you don't need to work with a ``Dim`` directly.
    Instead, use conversion constructors:


    .. code-block:: python

        from max.graph import Dim, TensorType, DeviceRef

        tensor_type = TensorType(DType.int64, ("batch", 10), device=DeviceRef.CPU())

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
        elif isinstance(value, kgen.ParamOperatorAttr):
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
        """Conversion to an int only supported for static dims."""
        raise TypeError(
            f"int({self!r}): Int conversions only supported for static dims"
        )

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

    def __add__(self, rhs: DimLike) -> Dim:
        return AlgebraicDim.apply(kgen.POC.add, self, rhs)

    # hitting https://github.com/python/mypy/issues/11595 which causes mypy to fail to typecheck.
    def __radd__(self, lhs: DimLike) -> Dim:  # type: ignore
        return Dim(lhs) + self

    def __mul__(self, rhs: DimLike) -> Dim:
        return AlgebraicDim.apply(kgen.POC.mul_nuw, self, rhs)

    # hitting https://github.com/python/mypy/issues/11595 which causes mypy to fail to typecheck.
    def __rmul__(self, lhs: DimLike) -> Dim:  # type: ignore
        return Dim(lhs) * self

    def __neg__(self) -> Dim:
        return -1 * self

    def __sub__(self, rhs: DimLike) -> Dim:
        return self + -Dim(rhs)

    def __rsub__(self, lhs: DimLike) -> Dim:
        return lhs + -self

    def __floordiv__(self, rhs: DimLike) -> Dim:
        if isinstance(rhs, (int, StaticDim)) and int(rhs) == 0:
            raise ZeroDivisionError
        return AlgebraicDim.apply(kgen.POC.div, self, rhs)

    def __rfloordiv__(self, lhs: DimLike) -> Dim:
        return lhs // self

    def to_mlir(self) -> builtin.TypedAttr:
        """Creates an ``mlir.Attribute`` representing this dimension.

        This is used internally when constructing tensor MLIR types.

        Returns:
            An ``mlir.Attribute`` in the context representing the dimension.
        """
        raise NotImplementedError

    @staticmethod
    def from_mlir(attr: builtin.TypedAttr) -> Dim:
        """Constructs a dimension from an ``mlir.Attribute``.

        Args:
            dim_attr: The MLIR Attribute object to parse into a dimension.

        Returns:
            Dim: The dimension represented by the MLIR Attr value.
        """
        if isinstance(attr, builtin.IntegerAttr):
            return StaticDim.from_mlir(attr)
        elif isinstance(attr, kgen.ParamDeclRefAttr):
            return SymbolicDim.from_mlir(attr)
        elif isinstance(attr, kgen.ParamOperatorAttr):
            return AlgebraicDim.from_mlir(attr)
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

    .. code-block::

        !mo.tensor<[batch, x, 10], si32]>

    The first and second dimensions are named ``batch`` and ``x`` respectively.

    Creating a ``SymbolicDim``:

    .. code-block:: python

        dim = SymbolicDim("name")

    Using ``SymbolicDim`` in a :obj:`TensorType`:

    .. code-block:: python

        tensor_type = TensorType(DType.bool, (SymbolicDim("batch"), SymbolicDim("x"), 10))
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
        return f"Dim({self.name!r})"

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

    def to_mlir(self) -> kgen.ParamDeclRefAttr:
        """Creates an ``mlir.Attribute`` representing this dimension.

        This is used internally when constructing tensor MLIR types.

        Returns:
            An ``mlir.Attribute`` in the context representing the dimension.
        """
        si64 = builtin.IntegerType(64, builtin.SignednessSemantics.signed)
        return kgen.ParamDeclRefAttr(self.name, si64)

    @staticmethod
    def from_mlir(attr: builtin.TypedAttr) -> Dim:
        """Constructs a dimension from an ``mlir.Attribute``.

        Args:
            dim_attr: The MLIR Attribute object to parse into a dimension.

        Returns:
            Dim: The dimension represented by the MLIR Attr value.
        """
        if not isinstance(attr, kgen.ParamDeclRefAttr):
            raise TypeError(f"Attr is not a symbolic dim: {attr}")
        return SymbolicDim(attr.name.value)


@dataclass(frozen=True)
class AlgebraicDim(Dim):
    """An algebraic tensor dimension to enable expressions over symbolic
    dimensions.

    That is, any expression over a symbolic dimension returns ``AlgebraicDim``.
    Furthermore, algebraic dimensions automatically simplify into a canonical
    form.

    The following example demonstrates how to create and use algebraic dimensions with symbolic values:

    .. code-block:: python

        from max.graph import AlgebraicDim, Dim
        isinstance(Dim("batch") * 5, AlgebraicDim)  # Returns True
        print(Dim("batch") * 5)  # Outputs: batch * 5
        -Dim("x") - 4 == -(Dim("x") + 4)  # Returns True
    """

    attr: kgen.ParamOperatorAttr

    def __init__(self, attr: kgen.ParamOperatorAttr | AlgebraicDim):
        super().__setattr__(
            "attr", attr.attr if isinstance(attr, AlgebraicDim) else attr
        )

    @classmethod
    def apply(cls, op: kgen.POC, *operands: DimLike):
        # kgen.ParamOperatorAttr eagerly folds on construction!
        #  - this can return static or symbolic dims
        #  - let Dim decide what type to returtn
        attr = kgen.ParamOperatorAttr(
            op, [Dim(operand).to_mlir() for operand in operands]
        )
        return Dim.from_mlir(attr)

    def __format__(self, format_spec: str):
        formatters: Mapping[str, Callable[[Any], str]] = {
            "str": str,
            "repr": repr,
        }
        formatter = formatters[format_spec or "str"]

        def format(dim: Dim):
            formatted = formatter(dim)
            return (
                f"({formatted})" if isinstance(dim, AlgebraicDim) else formatted
            )

        # For the opcodes we support in the graph api, print with python math.
        opcodes = {
            kgen.POC.add: "+",
            kgen.POC.mul_nuw: "*",
            kgen.POC.div: "//",
        }
        opcode = self.attr.opcode
        dims = [Dim.from_mlir(operand) for operand in self.attr.operands]
        if opcode in opcodes:
            # Wrap algebraic sub-expressions in parens
            return f" {opcodes[opcode]} ".join(map(format, dims))
        return formatter(self.attr)

    def __str__(self):
        return f"{self:str}"

    def __repr__(self):
        return f"{self:repr}"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, AlgebraicDim) and self.attr == other.attr

    def to_mlir(self) -> kgen.ParamOperatorAttr:
        """Creates an mlir.Attribute representing this dimension.
        This is used internally when constructing tensor MLIR types.

        Returns:
            An mlir.Attribute in the context representing the dimension.
        """
        return self.attr

    @staticmethod
    def from_mlir(attr: builtin.TypedAttr) -> Dim:
        if not isinstance(attr, kgen.ParamOperatorAttr):
            raise TypeError(f"Attribute is not an algebraic dimension: {attr}")
        return AlgebraicDim(attr)


@functools.total_ordering
@dataclass(frozen=True)
class StaticDim(Dim):
    """A static tensor dimension.

    Static tensor dimensions will always have exactly the same value,
    and are key to good model performance.

    The following example shows how static dimensions can be created implicitly:

    .. code-block:: python

        from max.graph import TensorType
        from max.dtype import DType
        tensor = TensorType(DType.int64, (4, 5))
        # This creates a tensor with 2 static dimensions: 4 and 5 respectively
    """

    dim: int
    """The size of the static dimension."""

    def __init__(self, dim: int | StaticDim):
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

    def to_mlir(self) -> builtin.IntegerAttr:
        """Creates an ``mlir.Attribute`` representing this dimension.

        This is used internally when constructing tensor MLIR types.

        Returns:
            An ``mlir.Attribute`` in the context representing the dimension.
        """
        si64 = builtin.IntegerType(64, builtin.SignednessSemantics.signed)
        return builtin.IntegerAttr(si64, self.dim)

    @staticmethod
    def from_mlir(attr: builtin.TypedAttr) -> Dim:
        """Constructs a dimension from an ``mlir.Attribute``.

        Args:
            dim_attr: The MLIR Attribute object to parse into a dimension.

        Returns:
            The dimension represented by the MLIR Attr value.
        """
        if not isinstance(attr, builtin.IntegerAttr):
            raise TypeError(f"Attribute is not a static dimension: {attr}")
        return StaticDim(attr.value)


def _is_static_shape(dims: Shape) -> TypeGuard[StaticShape]:
    return all(isinstance(dim, StaticDim) and dim.dim >= 0 for dim in dims)


class Shape(list[Dim]):
    def __init__(self, dims: ShapeLike = ()):
        super().__init__(Dim(dim) for dim in dims)

    @property
    def rank(self):
        return len(self)

    def to_mlir(self) -> mosh.ShapeAttr:
        shape_type = mosh.ShapeType()
        return mosh.ShapeAttr([dim.to_mlir() for dim in self], shape_type)

    @classmethod
    def from_mlir(cls, attr: builtin.TypedAttr) -> Shape:
        if not isinstance(attr, mosh.ShapeAttr):
            raise TypeError(
                f"Shape.from_mlir only supported for mosh.ShapeAttr, got {attr}"
            )
        return cls([Dim.from_mlir(dim) for dim in attr.values])

    @property
    def static_dims(self) -> list[int]:
        """Returns all static dims in the shape as a list of integers."""
        return [d.dim for d in self if isinstance(d, StaticDim)]


@dataclass(frozen=True)
class DeviceKind(str, Enum):
    """A device type representation."""

    CPU = "cpu"
    GPU = "gpu"

    def __str__(self) -> str:
        return self.value

    @staticmethod
    def from_string(txt) -> DeviceKind:
        if txt == str(DeviceKind.CPU):
            return DeviceKind.CPU
        elif txt == str(DeviceKind.GPU):
            return DeviceKind.GPU
        else:
            raise ValueError(f"Unknown device kind {txt}")


class DeviceRef:
    """A symbolic device representation.

    DeviceRef type representation consists of a DeviceKind and an id. This is a direct
    representation of the device attribute in mlir.

    The following example demonstrates how to create and use device references:

    .. code-block:: python

        from max.graph import DeviceRef
        # Create a GPU device reference (default id=0)
        gpu_device = DeviceRef.GPU()
        print(gpu_device)  # Outputs: gpu:0
        # Create a CPU device with specific id
        cpu_device = DeviceRef.CPU(id=1)
        print(cpu_device)  # Outputs: cpu:1
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
        if not isinstance(other, DeviceRef):
            return False
        return self.device_type is other.device_type and self.id == other.id

    def to_mlir(self) -> m.DeviceRefAttr:
        """Returns a mlir attribute representing device."""
        return m.DeviceRefAttr(str(self.device_type), self.id)

    def to_device(self) -> Device:
        """Convert device reference to a concrete driver Device."""
        if self.device_type is DeviceKind.CPU:
            return CPU(self.id)
        elif self.device_type is DeviceKind.GPU:
            return Accelerator(self.id)
        else:
            raise ValueError(f"Unsupported device type: {self.device_type}")

    def is_cpu(self) -> bool:
        """Returns true if the device is a CPU device."""
        return self.device_type is DeviceKind.CPU

    def is_gpu(self) -> bool:
        """Returns true if the device is a GPU device."""
        return self.device_type is DeviceKind.GPU

    @staticmethod
    def from_mlir(attr: m.DeviceRefAttr) -> DeviceRef:
        """Returns a device from mlir attribute"""
        return DeviceRef(device_type=DeviceKind(attr.label), id=attr.id)

    @staticmethod
    def from_device(device: Device) -> DeviceRef:
        return DeviceRef(DeviceKind(device.label), device.id)


StaticShape = list[StaticDim]

DimLike = Union[int, str, Dim, np.integer]
ShapeLike = Iterable[DimLike]


class Type(Generic[MlirType]):
    """Represents any possible type for Graph values.

    Every Value in the Graph has a Type, and that type is represented by an Type.
    This type may be inspected to get finer-grained types and learn more
    about an individual Value.

    The following example shows how to work with types in a graph:

    .. code-block:: python

        from max.graph import Graph, TensorType
        from max.dtype import DType
        with Graph() as g:
            # Create a tensor constant with a specific type
            tensor_type = TensorType(DType.float32, [2, 3])
            # The type can be inspected to get information about the value
            print(f"Tensor element type: {tensor_type.dtype}")  # Outputs: DType.float32
            print(f"Tensor shape: {tensor_type.shape}")  # Outputs: [2, 3]
    """

    def to_mlir(self) -> MlirType:
        """Converts to an ``mlir.Type`` instance.

        Returns:
            An ``mlir.Type`` in the specified Context.
        """
        raise NotImplementedError

    @staticmethod
    def from_mlir(t: MlirType) -> Type:
        """Constructs a type from an MLIR type.

        Args:
            t: The MLIR Type object to parse into a type.

        Returns:
            The type represented by the MLIR Type value.
        """
        if isinstance(t, mo.TensorType):
            return TensorType.from_mlir(t)
        elif isinstance(t, mo.BufferType):
            return BufferType.from_mlir(t)
        elif isinstance(t, mo.ChainType):
            return _ChainType.from_mlir(t)
        elif isinstance(t, mo.OpaqueType):
            return _OpaqueType.from_mlir(t)
        raise TypeError(f"No type found for MLIR type: {t}")


@dataclass
class _TensorTypeBase(Type[MlirType]):
    dtype: DType
    """The element type of the tensor value."""
    shape: Shape
    """The dimensions of the tensor value."""
    device: DeviceRef
    """The device of the tensor value."""

    def __init__(
        self, dtype: DType, shape: ShapeLike, device: DeviceRef
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

        if any(dim < 0 for dim in self.shape.static_dims):
            raise TypeError(
                f"Static tensor dimensions must be non-negative; got {shape=}"
            )

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

    def __eq__(self, other: Any) -> bool:
        """Checks whether the two tensors have the same rank, type, and shape.

        Args:
            other: The other tensor to check equality against.

        Returns:
            True if the tensors have identical element type and shape,
            false otherwise.
        """
        return (
            isinstance(other, type(self))
            and (self.dtype == other.dtype)
            and (self.shape == other.shape)
        )

    # ===------------------------------------------------------------------=== #
    # Utilities
    # ===------------------------------------------------------------------=== #

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

    def cast(self, dtype: DType):
        """Constructs a new tensor type of the same shape with the new `dtype`.

        Args:
            dtype: The new element type for the tensor.

        Returns:
            A new tensor type with the same shape, device, and the new element type.
        """
        return type(self)(dtype, self.shape, self.device)


@dataclass
class TensorType(_TensorTypeBase[mo.TensorType]):
    """A symbolic :obj:`TensorType`.

    This is not an eager tensor type! This contains no actual data, but
    instead represents the type of a value at some point in time during model
    execution.

    Most internal values in a model will be tensors. This type represents
    their element type (``dtype``) and dimensions (``dims``) at a specific point during
    model computation. It allows us to do some optimistic optimizations and
    shape inference during graph construction, and to provide more detailed
    shape information to the compiler for further optimization passes.

    The following example shows how to create a tensor type with static dimensions and access its properties:

    .. code-block:: python

        from max.graph import TensorType
        from max.dtype import DType
        # Create a tensor type with float32 elements and static dimensions 2x3
        tensor_type = TensorType(DType.float32, (2, 3))
        print(tensor_type.dtype)  # Outputs: DType.float32
        print(tensor_type.shape)  # Outputs: [2, 3]

    It can also represent a fully dynamic rank tensor. The presence of dynamic
    rank tensors in a graph will often degrade performance dramatically and
    prevents many classes of optimizations.

    An optional device (``device``) can also be provided to indicate the explicit
    device the tensor is associated with.
    """

    _layout: FilterLayout | None = field(
        default=None, compare=False, repr=False
    )

    __init__ = _TensorTypeBase.__init__

    @classmethod
    def from_mlir(cls, type: mo.TensorType) -> TensorType:
        """Constructs a tensor type from an MLIR type.

        Args:
            t: The MLIR Type object to parse into a tensor type.

        Returns:
            The tensor type represented by the MLIR Type value.
        """
        device_ref = DeviceRef.from_mlir(type.device_ref)
        self = cls(type.dtype, Shape.from_mlir(type.shape_attr), device_ref)
        for name, attr in type.metadata.value:
            if name == "layout":
                assert isinstance(attr, mo.LayoutAttr)
                self._layout = FilterLayout.from_mlir(attr)
        return self

    def to_mlir(self) -> mo.TensorType:
        """Converts to an ``mlir.Type`` instance.

        Returns:
            An ``mlir.Type`` in the specified Context.
        """
        metadata = []
        if self._layout:
            metadata.append(NamedAttribute("layout", self._layout.to_mlir()))
        return mo.TensorType(
            self.shape.to_mlir(),
            self.dtype,
            self.device.to_mlir(),
            metadata=builtin.DictionaryAttr(metadata),
        )

    def as_buffer(self) -> BufferType:
        """Returns the analogous buffer type."""
        return BufferType(self.dtype, self.shape, self.device)


class BufferType(_TensorTypeBase[mo.BufferType]):
    """A symbolic buffer type.

    This is a reference to a tensor that can be mutated in place.
    """

    @classmethod
    def from_mlir(cls, type: mo.BufferType) -> BufferType:
        """Constructs a buffer type from an MLIR type.

        Args:
            t: The MLIR Type object to parse into a buffer type.

        Returns:
            The buffer type represented by the MLIR Type value.
        """
        device_ref = DeviceRef.from_mlir(type.device_ref)
        return cls(type.dtype, Shape.from_mlir(type.shape_attr), device_ref)

    def to_mlir(self) -> mo.BufferType:
        """Converts to an ``mlir.Type`` instance.

        Returns:
            An ``mlir.Type`` in the specified Context.
        """
        return mo.BufferType(
            self.shape.to_mlir(), self.dtype, self.device.to_mlir()
        )

    def as_tensor(self) -> TensorType:
        """Returns the analogous tensor type."""
        return TensorType(self.dtype, self.shape, self.device)


@dataclass(frozen=True)
class _OpaqueType(Type[mo.OpaqueType]):
    """A type representing an opaque type."""

    name: str
    """Identifier for the opaque type."""

    def to_mlir(self) -> mo.OpaqueType:
        """Converts to an ``mlir.Type`` instance.

        Returns:
            An ``mlir.Type`` in the specified Context.
        """
        return mo.OpaqueType(
            builtin.StringAttr(self.name), parameters=builtin.DictionaryAttr([])
        )

    @staticmethod
    def from_mlir(t: mo.OpaqueType) -> _OpaqueType:
        """Constructs an opaque type from an MLIR type.

        Args:
            t: The MLIR Type object to parse into an opaque type.

        Returns:
            The opaque type represented by the MLIR Type value.
        """
        return _OpaqueType(t.symbol.value)


@dataclass
class _ChainType(Type[mo.ChainType]):
    """A chain type.

    Used in order to sequence operations that have side-effects.

    As a user you should never need to directly interact with this type.
    """

    def to_mlir(self) -> mo.ChainType:
        """Converts to an mlir.Type instance.

        Returns:
            An mlir.Type in the specified Context.
        """
        return mo.ChainType()

    @staticmethod
    def from_mlir(t: mo.ChainType) -> _ChainType:
        """Constructs an opaque type from an MLIR type.

        Args:
            t: The MLIR Type object to parse into an opaque type.

        Returns:
            The opaque type represented by the MLIR Type value.
        """
        return _ChainType()
