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
"""Library for graph value types."""

from __future__ import annotations

import enum
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

from max._core import Attribute, NamedAttribute
from max._core import Type as _Type
from max._core import graph as _graph
from max._core.dialects import builtin, m, mo
from max.driver import CPU, Accelerator, Device
from max.dtype import DType

from .dim import SymbolicDim
from .shape import Shape, ShapeLike

MlirType = TypeVar("MlirType", bound=_Type)
OpaqueParameter = bool | int | str | DType


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


@dataclass(frozen=True)
class DeviceKind(str, Enum):
    """A device type representation."""

    CPU = "cpu"
    GPU = "gpu"

    def __str__(self) -> str:
        return self.value

    @staticmethod
    def from_string(txt: str) -> DeviceKind:
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

    def __init__(self, device_type: DeviceKind | str, id: int = 0) -> None:
        if isinstance(device_type, DeviceKind):
            self.device_type = device_type
        else:
            self.device_type = DeviceKind(device_type)
        if id < 0:
            id = 0
        self.id = id

    def __str__(self) -> str:
        return str(self.device_type) + ":" + str(self.id)

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: Any) -> bool:
        """Returns true if devices are equal."""
        if not isinstance(other, DeviceRef):
            return False
        return self.device_type is other.device_type and self.id == other.id

    def __hash__(self) -> int:
        """Hashes based on the immutable identity (device_type, id)."""
        return hash((self.device_type, self.id))

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
    def from_device(device: Device | DeviceRef) -> DeviceRef:
        if isinstance(device, DeviceRef):
            return device
        return DeviceRef(DeviceKind(device.label), device.id)


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
    def from_mlir(t: MlirType) -> Type[Any]:
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
            and (self.device == other.device)
        )

    def __hash__(self) -> int:
        return hash((self.dtype, tuple(self.shape), self.device))

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
        if not Shape.is_static(self.shape):
            raise RuntimeError(
                "can't find num elements since tensor has symbolic dims"
            )

        return math.prod(int(dim) for dim in self.shape)

    def cast(self, dtype: DType):  # noqa: ANN202
        """Constructs a new tensor type of the same shape with the new `dtype`.

        Args:
            dtype: The new element type for the tensor.

        Returns:
            A new tensor type with the same shape, device, and the new element type.
        """
        return type(self)(dtype, self.shape, self.device)

    @property
    def parameters(self) -> Iterable[SymbolicDim]:
        """Lists the symbolic dimension names on which this type depends."""
        return self.shape.parameters


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

    _layout: FilterLayout | None = field(default=None, repr=False)

    def __init__(
        self,
        dtype: DType,
        shape: ShapeLike,
        device: Device | DeviceRef,
        _layout: FilterLayout | None = None,
    ) -> None:
        super().__init__(dtype, shape, DeviceRef.from_device(device))
        self._layout = _layout

    def __hash__(self) -> int:
        return hash((self.dtype, tuple(self.shape), self.device, self._layout))

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


def _value_to_attribute(param: OpaqueParameter) -> Attribute:
    """Converts a native Python value to an MLIR attribute to parametrize a
    kernel or opaque type.
    """
    if isinstance(param, bool):
        return builtin.BoolAttr(param)

    if isinstance(param, int):
        signedness = builtin.SignednessSemantics.signed
        return builtin.IntegerAttr(builtin.IntegerType(64, signedness), param)

    if isinstance(param, str):
        return builtin.StringAttr(param)

    if isinstance(param, DType):
        # Wrap the MLIR type corresponding to dtype in a TypeAttr,
        # which MOToKGENLowering expects.
        dtype = _graph.dtype_to_type(param)
        return builtin.TypeAttr(dtype)

    raise TypeError(f"unsupported parameter type {type(param)} for custom op")


def _attribute_to_value(value: Attribute) -> OpaqueParameter:
    """Converts an MLIR attribute representing a Mojo parameter to the
    corresponding Python type.

    This function is the inverse of _value_to_attribute.
    """
    if isinstance(value, builtin.BoolAttr | builtin.StringAttr):
        return value.value

    if isinstance(value, builtin.IntegerAttr):
        type = value.type
        if isinstance(type, builtin.IntegerType) and type.width == 1:
            return bool(value.value)

        return value.value

    if isinstance(value, builtin.TypeAttr):
        val = value.value
        assert val is not None
        return _graph.type_to_dtype(val)

    raise TypeError(f"unsupported attribute type {value}")


@dataclass(frozen=True)
class _OpaqueType(Type[mo.OpaqueType]):
    """A type representing an opaque type."""

    name: str
    """Identifier for the opaque type."""

    parameters: dict[str, OpaqueParameter] = field(default_factory=dict)
    """Dictionary of Mojo parameter assignments.

    Valid parameter types are ``bool``, ``int``, ``str``, and ``DType``.

    Example:

    .. code-block:: python

        # Create an opaque type with parameters
        custom_type = _OpaqueType(
            name="MyCustomType",
            parameters={
                "rank": 3,
                "mut": True,
                "type": DType.float32,
                "address_space": "global"
            }
        )
    """

    def to_mlir(self) -> mo.OpaqueType:
        """Converts to an ``mlir.Type`` instance.

        Returns:
            An ``mlir.Type`` in the specified Context.
        """
        # Convert Python dict to MLIR DictionaryAttr

        mlir_params = [
            NamedAttribute(key, _value_to_attribute(value))
            for key, value in self.parameters.items()
        ]

        return mo.OpaqueType(
            builtin.StringAttr(self.name),
            parameters=builtin.DictionaryAttr(mlir_params),
        )

    @staticmethod
    def from_mlir(t: mo.OpaqueType) -> _OpaqueType:
        """Constructs an opaque type from an MLIR type.

        Args:
            t: The MLIR Type object to parse into an opaque type.

        Returns:
            The opaque type represented by the MLIR Type value.
        """
        params: dict[str, OpaqueParameter] = {
            name: _attribute_to_value(attr) for name, attr in t.parameters.value
        }

        return _OpaqueType(t.symbol.value, params)


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
