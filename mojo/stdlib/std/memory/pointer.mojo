# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
"""Implements the Pointer type.

You can import these APIs from the `memory` package. For example:

```mojo
from memory import Pointer
```
"""

from format._utils import FormatStruct, Named, TypeNames

# ===-----------------------------------------------------------------------===#
# AddressSpace
# ===-----------------------------------------------------------------------===#


struct AddressSpace(
    Equatable,
    ImplicitlyCopyable,
    Intable,
    TrivialRegisterPassable,
    Writable,
):
    """Address space of the pointer.

    This type represents memory address spaces for both CPU and GPU targets.
    On CPUs, typically only GENERIC is used. On GPUs (NVIDIA/AMD), various
    address spaces provide access to different memory regions with different
    performance characteristics.
    """

    var _value: Int

    # CPU address space
    comptime GENERIC = AddressSpace(0)
    """Generic address space. Used for CPU memory and default GPU memory."""

    # GPU address spaces
    # See https://docs.nvidia.com/cuda/nvvm-ir-spec/#address-space
    # And https://llvm.org/docs/AMDGPUUsage.html#address-spaces
    comptime GLOBAL = AddressSpace(1)
    """Global GPU memory address space."""
    comptime SHARED = AddressSpace(3)
    """Shared GPU memory address space (per thread block/workgroup)."""
    comptime CONSTANT = AddressSpace(4)
    """Constant GPU memory address space (read-only)."""
    comptime LOCAL = AddressSpace(5)
    """Local GPU memory address space (per thread, private)."""
    comptime SHARED_CLUSTER = AddressSpace(7)
    """Shared cluster GPU memory address space (NVIDIA-specific)."""
    comptime BUFFER_RESOURCE = AddressSpace(8)
    """Buffer resource GPU memory address space (AMD-specific)."""

    @always_inline("builtin")
    fn __init__(out self, value: Int):
        """Initializes the address space from the underlying integral value.

        Args:
          value: The address space value.
        """
        self._value = value

    @always_inline("builtin")
    fn value(self) -> Int:
        """The integral value of the address space.

        Returns:
          The integral value of the address space.
        """
        return self._value

    @always_inline("builtin")
    fn __int__(self) -> Int:
        """The integral value of the address space.

        Returns:
          The integral value of the address space.
        """
        return self._value

    @always_inline("builtin")
    fn __eq__(self, other: Self) -> Bool:
        """Checks if the two address spaces are equal.

        Args:
          other: The other address space value.

        Returns:
          True if the two address spaces are equal and False otherwise.
        """
        return self._value == other._value

    @deprecated("Stringable is deprecated. Use Writable instead.")
    @always_inline("nodebug")
    fn __str__(self) -> String:
        """Gets a string representation of the AddressSpace.

        Returns:
            The string representation of the AddressSpace.
        """
        return String.write(self)

    @always_inline("nodebug")
    fn write_to(self, mut writer: Some[Writer]):
        """Formats the address space to the provided Writer.

        Args:
            writer: The object to write to.
        """
        if self == AddressSpace.GENERIC:
            writer.write("AddressSpace.GENERIC")
        elif self == AddressSpace.GLOBAL:
            writer.write("AddressSpace.GLOBAL")
        elif self == AddressSpace.SHARED:
            writer.write("AddressSpace.SHARED")
        elif self == AddressSpace.CONSTANT:
            writer.write("AddressSpace.CONSTANT")
        elif self == AddressSpace.LOCAL:
            writer.write("AddressSpace.LOCAL")
        elif self == AddressSpace.SHARED_CLUSTER:
            writer.write("AddressSpace.SHARED_CLUSTER")
        else:
            writer.write("AddressSpace(", self.value(), ")")

    fn write_repr_to(self, mut writer: Some[Writer]):
        """Write the string representation of the AddressSpace.

        Args:
            writer: The object to write to.
        """
        self.write_to(writer)


# ===-----------------------------------------------------------------------===#
# Deprecated aliases for backward compatibility
# ===-----------------------------------------------------------------------===#

comptime _GPUAddressSpace = AddressSpace
"""Deprecated: Use `AddressSpace` instead. This alias is provided for backward
compatibility and will be removed in a future release."""

comptime GPUAddressSpace = AddressSpace
"""Deprecated: Use `AddressSpace` instead. This alias is provided for backward
compatibility and will be removed in a future release."""


# ===-----------------------------------------------------------------------===#
# Pointer aliases
# ===-----------------------------------------------------------------------===#


comptime MutPointer[
    type: AnyType,
    origin: MutOrigin,
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
] = Pointer[type, origin, address_space=address_space]
"""A mutable pointer.

Parameters:
    type: The pointee type.
    origin: The origin of the pointer.
    address_space: The address space of the pointer.
"""

comptime ImmutPointer[
    type: AnyType,
    origin: ImmutOrigin,
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
] = Pointer[type, origin, address_space=address_space]
"""An immutable pointer.

Parameters:
    type: The pointee type.
    origin: The origin of the pointer.
    address_space: The address space of the pointer.
"""


# ===-----------------------------------------------------------------------===#
# Pointer
# ===-----------------------------------------------------------------------===#


struct Pointer[
    mut: Bool,
    //,
    type: AnyType,
    origin: Origin[mut=mut],
    address_space: AddressSpace = AddressSpace.GENERIC,
](TrivialRegisterPassable, Writable):
    """Defines a non-nullable safe pointer.

    For a comparison with other pointer types, see [Intro to
    pointers](/mojo/manual/pointers/) in the Mojo Manual.

    Parameters:
        mut: Whether the pointee data may be mutated through this.
        type: Type of the underlying data.
        origin: The origin of the pointer.
        address_space: The address space of the pointee data.
    """

    # Aliases
    comptime _mlir_type = __mlir_type[
        `!lit.ref<`,
        Self.type,
        `, `,
        Self.origin._mlir_origin,
        `, `,
        Self.address_space._value._mlir_value,
        `>`,
    ]
    comptime _with_origin = Pointer[Self.type, _, Self.address_space]

    comptime Immutable = Self._with_origin[ImmutOrigin(Self.origin)]
    """The immutable version of the `Pointer`."""
    # Fields
    var _value: Self._mlir_type
    """The underlying MLIR representation."""

    # ===------------------------------------------------------------------===#
    # Initializers
    # ===------------------------------------------------------------------===#

    @doc_private
    @implicit
    @always_inline("nodebug")
    fn __init__(
        other: Pointer,
        out self: Pointer[
            other.type,
            ImmutOrigin(other.origin),
            address_space = other.address_space,
        ],
    ):
        """Implicitly cast the mutable origin of self to an immutable one.

        Args:
            other: The `Pointer` to cast.
        """
        self = {_mlir_value = other._value}

    @doc_private
    @always_inline("nodebug")
    fn __init__(out self, *, _mlir_value: Self._mlir_type):
        """Constructs a Pointer from its MLIR prepresentation.

        Args:
             _mlir_value: The MLIR representation of the pointer.
        """
        self._value = _mlir_value

    @always_inline("nodebug")
    fn __init__(
        out self,
        *,
        ref[Self.origin, Self.address_space._value._mlir_value] to: Self.type,
    ):
        """Constructs a Pointer from a reference to a value.

        Args:
            to: The value to construct a pointer to.
        """
        self = Self(_mlir_value=__get_mvalue_as_litref(to))

    @always_inline
    fn get_immutable(self) -> Self.Immutable:
        """Constructs a new Pointer with the same underlying target
        and an ImmutOrigin.

        Returns:
            A new Pointer with the same target as self and an ImmutOrigin.

        Notes:
            This does **not** copy the underlying data.
        """
        return rebind[Self.Immutable](self)

    # ===------------------------------------------------------------------===#
    # Operator dunders
    # ===------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __getitem__(self) -> ref[Self.origin, Self.address_space] Self.type:
        """Enable subscript syntax `ptr[]` to access the element.

        Returns:
            A reference to the underlying value in memory.
        """
        return __get_litref_as_mvalue(self._value)

    # This decorator informs the compiler that indirect address spaces are not
    # dereferenced by the method.
    # TODO: replace with a safe model that checks the body of the method for
    # accesses to the origin.
    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __eq__(self, rhs: Pointer[Self.type, _, Self.address_space]) -> Bool:
        """Returns True if the two pointers are equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are equal and False otherwise.
        """
        return UnsafePointer(to=self[]) == UnsafePointer(to=rhs[])

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __ne__(self, rhs: Pointer[Self.type, _, Self.address_space]) -> Bool:
        """Returns True if the two pointers are not equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are not equal and False otherwise.
        """
        return not (self == rhs)

    @deprecated("Stringable is deprecated. Use Writable instead.")
    @no_inline
    fn __str__(self) -> String:
        """Gets a string representation of the Pointer.

        Returns:
            The string representation of the Pointer.
        """
        return String(UnsafePointer(to=self[]))

    fn write_to(self, mut writer: Some[Writer]):
        """Formats this pointer address to the provided Writer.

        Args:
            writer: The object to write to.
        """
        UnsafePointer(to=self[]).write_to(writer)

    fn write_repr_to(self, mut writer: Some[Writer]):
        """Write the string representation of the Pointer.

        Args:
            writer: The object to write to.
        """
        FormatStruct(writer, "Pointer").params(
            Named("mut", Self.mut),
            TypeNames[Self.type](),
            Named("address_space", Self.address_space),
        ).fields(self)

    @always_inline("nodebug")
    fn __merge_with__[
        other_type: type_of(Pointer[Self.type, _, Self.address_space]),
    ](
        self,
        out result: Pointer[
            mut = Self.mut & other_type.origin.mut,
            type = Self.type,
            origin = origin_of(Self.origin, other_type.origin),
            address_space = Self.address_space,
        ],
    ):
        """Returns a pointer merged with the specified `other_type`.

        Parameters:
            other_type: The type of the pointer to merge with.

        Returns:
            A pointer merged with the specified `other_type`.
        """
        return {_mlir_value = self._value}  # allow lit.ref to convert.
