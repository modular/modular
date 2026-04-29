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
"""A temporary `UnsafeNullablePointer` type used to help transition users of
`UnsafePointer` to `Optional[UnsafePointer]` for explicit nullability.
"""

from std.sys import align_of, is_gpu, is_nvidia_gpu, size_of
from std.sys.intrinsics import (
    gather,
    scatter,
    strided_load,
    strided_store,
    unlikely,
)

from std.builtin.device_passable import DevicePassable
from std.builtin.rebind import downcast
from std.builtin.format_int import _write_int
from std.builtin.simd import _simd_construction_checks
from std.collections import OptionalReg
from std.format._utils import FormatStruct, Named, TypeNames
from std.reflection import reflect
from std.memory import memcpy
from std.memory.memory import _free, _malloc
from std.memory import UnsafeMaybeUninit
from std.memory._poison import _check_not_poison, _check_not_poison_masked
from std.os import abort
from std.python import PythonObject
from std.utils._nicheable import (
    UnsafeSingleNicheable,
    UnsafeCustomNicheStorage,
    NicheStorageTraits,
)


@always_inline
def _default_invariant[mut: Bool]() -> Bool:
    return is_gpu() and mut == False


struct UnsafeNullablePointer[
    mut: Bool,
    //,
    type: AnyType,
    origin: Origin[mut=mut],
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
](
    Boolable,
    Comparable,
    Defaultable,
    DevicePassable,
    ImplicitlyCopyable,
    Intable,
    TrivialRegisterPassable,
    Writable,
):
    """A temporary `UnsafeNullablePointer` type used to help transition users of
    `UnsafePointer` to `Optional[UnsafePointer]` for explicit nullability.

    Parameters:
        mut: Whether the origin is mutable.
        type: The type the pointer points to.
        origin: The origin of the memory being addressed.
        address_space: The address space associated with the pointer's allocated memory.
    """

    # ===-------------------------------------------------------------------===#
    # Aliases
    # ===-------------------------------------------------------------------===#

    comptime _mlir_type = __mlir_type[
        `!kgen.pointer<`,
        Self.type,
        `, `,
        Self.address_space._value._mlir_value,
        `>`,
    ]
    """The underlying pointer type."""

    comptime _with_origin[
        with_mut: Bool, //, with_origin: Origin[mut=with_mut]
    ] = UnsafeNullablePointer[
        mut=with_mut,
        Self.type,
        with_origin,
        address_space=Self.address_space,
    ]

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var address: Self._mlir_type
    """The underlying pointer."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    def __init__(out self):
        """Create a null pointer."""
        self.address = __mlir_attr[`#interp.pointer<0> : `, Self._mlir_type]

    @doc_hidden
    @always_inline("nodebug")
    @implicit
    def __init__(out self, value: Self._mlir_type):
        """Create a pointer from a low-level pointer primitive.

        Args:
            value: The MLIR value of the pointer to construct with.
        """
        self.address = value

    @always_inline
    def __init__(out self, *, unsafe_from_address: Int):
        """Create a pointer from a raw address.

        Args:
            unsafe_from_address: The raw address to create a pointer from.

        Safety:
            Creating a pointer from a raw address is inherently unsafe as the
            caller must ensure the address is valid before writing to it, and
            that the memory is initialized before reading from it. The caller
            must also ensure the pointer's origin and mutability is valid for
            the address, failure to do may result in undefined behavior.
        """
        comptime assert (
            size_of[type_of(self)]() == size_of[Int]()
        ), "Pointer/Int size mismatch"
        self = UnsafeNullablePointer(to=unsafe_from_address).bitcast[
            type_of(self)
        ]()[]

    @always_inline("nodebug")
    def __init__(
        out self,
        *,
        ref[Self.origin, Self.address_space._value._mlir_value] to: Self.type,
    ):
        """Constructs a Pointer from a reference to a value.

        Args:
            to: The value to construct a pointer to.
        """
        self = Self(__mlir_op.`lit.ref.to_pointer`(__get_mvalue_as_litref(to)))

    @always_inline("nodebug")
    @implicit
    def __init__[
        disambig2: Int = 0
    ](
        other: UnsafeNullablePointer,
        out self: UnsafeNullablePointer[
            other.type,
            ImmutOrigin(other.origin),
            address_space=other.address_space,
        ],
    ):
        """Implicitly casts a mutable pointer to immutable.

        Args:
            other: The mutable pointer to cast from.

        Parameters:
            disambig2: Ignored. Works around name mangling conflict.
        """
        self.address = __mlir_op.`pop.pointer.bitcast`[
            _type=type_of(self)._mlir_type
        ](other.address)

    @always_inline("nodebug")
    @implicit
    def __init__[
        disambig: Int = 0  # FIXME: Work around name mangling conflict.
    ](
        other: UnsafeNullablePointer[mut=True, ...],
        out self: UnsafeNullablePointer[
            other.type,
            MutAnyOrigin,
            address_space=other.address_space,
        ],
    ):
        """Implicitly casts a mutable pointer to `MutAnyOrigin`.

        Args:
            other: The mutable pointer to cast from.

        Parameters:
            disambig: Ignored. Works around name mangling conflict.
        """
        self.address = __mlir_op.`pop.pointer.bitcast`[
            _type=type_of(self)._mlir_type
        ](other.address)

    @always_inline("nodebug")
    @implicit
    def __init__(
        other: UnsafeNullablePointer[...],
        out self: UnsafeNullablePointer[
            other.type,
            ImmutAnyOrigin,
            address_space=other.address_space,
        ],
    ):
        """Implicitly casts a pointer to `ImmutAnyOrigin`.

        Args:
            other: The pointer to cast from.
        """
        self.address = __mlir_op.`pop.pointer.bitcast`[
            _type=type_of(self)._mlir_type
        ](other.address)

    # ===-------------------------------------------------------------------===#
    # UnsafePointer conversion
    # ===-------------------------------------------------------------------===#

    @always_inline
    @implicit
    def __init__(
        other: UnsafePointer,
        out self: UnsafeNullablePointer[
            other.type,
            other.origin,
            address_space=other.address_space,
        ],
    ):
        """Implicitly casts an `UnsafePointer` to `UnsafeNullablePointer`.

        Args:
            other: The mutable pointer to cast from.
        """
        self.address = __mlir_op.`pop.pointer.bitcast`[
            _type=type_of(self)._mlir_type
        ](other.address)

    @always_inline
    @implicit
    def __init__[
        disambig2: Int = 0
    ](
        other: UnsafePointer,
        out self: UnsafeNullablePointer[
            other.type,
            ImmutOrigin(other.origin),
            address_space=other.address_space,
        ],
    ):
        """Implicitly casts an `UnsafePointer` to `UnsafeNullablePointer`.

        Args:
            other: The mutable pointer to cast from.

        Parameters:
            disambig2: Ignored. Works around name mangling conflict.
        """
        self.address = __mlir_op.`pop.pointer.bitcast`[
            _type=type_of(self)._mlir_type
        ](other.address)

    @always_inline
    @implicit
    def __init__[
        disambig: Int = 0  # FIXME: Work around name mangling conflict.
    ](
        other: UnsafePointer[mut=True, ...],
        out self: UnsafeNullablePointer[
            other.type,
            MutAnyOrigin,
            address_space=other.address_space,
        ],
    ):
        """Implicitly casts an `UnsafePointer` to `UnsafeNullablePointer`.

        Args:
            other: The mutable pointer to cast from.

        Parameters:
            disambig: Ignored. Works around name mangling conflict.
        """
        self.address = __mlir_op.`pop.pointer.bitcast`[
            _type=type_of(self)._mlir_type
        ](other.address)

    @always_inline
    @implicit
    def __init__[
        T: AnyType, other_origin: Origin, other_address_space: AddressSpace, //
    ](
        other: UnsafePointer[
            T, other_origin, address_space=other_address_space
        ],
        out self: UnsafeNullablePointer[
            T,
            ImmutAnyOrigin,
            address_space=other_address_space,
        ],
    ):
        """Implicitly casts an `UnsafePointer` to `UnsafeNullablePointer`.

        Args:
            other: The pointer to cast from.
        """
        self.address = __mlir_op.`pop.pointer.bitcast`[
            _type=type_of(self)._mlir_type
        ](other.address)

    def __init__[
        T: ImplicitlyDestructible, //
    ](
        out self: UnsafeNullablePointer[T, Self.origin],
        *,
        ref[Self.origin] unchecked_downcast_value: PythonObject,
    ):
        """Downcast a `PythonObject` known to contain a Mojo object to a pointer.

        This operation is only valid if the provided Python object contains
        an initialized Mojo object of matching type.

        Parameters:
            T: Pointee type that can be destroyed implicitly (without
              deinitializer arguments).

        Args:
            unchecked_downcast_value: The Python object to downcast from.
        """
        self = unchecked_downcast_value.unchecked_downcast_value_ptr[T]()

    # ===-------------------------------------------------------------------===#
    # UnsafePointer conversion
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    def to_unsafe_pointer_unchecked(
        self,
        out result: UnsafePointer[
            Self.type, Self.origin, address_space=Self.address_space
        ],
    ):
        """Transforms the `UnsafeNullablePointer` into a non-null `UnsfaePointer`.

        This method does not check if `self` is null or not, that is up to the user.

        Returns:
            An `UnsafePointer` pointer to the same value as `self`.
        """
        result = UnsafePointer(to=self).bitcast[type_of(result)]()[]

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    def __getitem__(self) -> ref[Self.origin, Self.address_space] Self.type:
        """Please refer to `UnsafePointer.__getitem__`.

        Returns:
            A reference to the value.
        """
        return self.to_unsafe_pointer_unchecked().__getitem__()

    @always_inline("nodebug")
    def __getitem__[
        I: Indexer, //
    ](self, offset: I) -> ref[Self.origin, Self.address_space] Self.type:
        """Please refer to `UnsafePointer.__getitem__`.

        Parameters:
            I: A type that can be used as an index.

        Args:
            offset: The offset index.

        Returns:
            An offset reference.
        """
        return self.to_unsafe_pointer_unchecked().__getitem__(offset)

    @always_inline("nodebug")
    def __add__[I: Indexer, //](self, offset: I) -> Self:
        """Please refer to `UnsafePointer.__add__`.

        Parameters:
            I: A type that can be used as an index.

        Args:
            offset: The offset index.

        Returns:
            An offset pointer.
        """
        return (self.to_unsafe_pointer_unchecked() + offset).address

    @always_inline
    def __sub__[I: Indexer, //](self, offset: I) -> Self:
        """Please refer to `UnsafePointer.__sub__`.

        Parameters:
            I: A type that can be used as an index.

        Args:
            offset: The offset index.

        Returns:
            An offset pointer.
        """
        return (self.to_unsafe_pointer_unchecked() - offset).address

    @always_inline
    def __iadd__[I: Indexer, //](mut self, offset: I):
        """Please refer to `UnsafePointer.__iadd__`.

        Parameters:
            I: A type that can be used as an index.

        Args:
            offset: The offset index.
        """
        self = (self.to_unsafe_pointer_unchecked() + offset).address

    @always_inline
    def __isub__[I: Indexer, //](mut self, offset: I):
        """Please refer to `UnsafePointer.__isub__`.

        Parameters:
            I: A type that can be used as an index.

        Args:
            offset: The offset index.
        """
        self = (self.to_unsafe_pointer_unchecked() - offset).address

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    def __eq__(
        self,
        rhs: UnsafeNullablePointer[
            Self.type, address_space=Self.address_space, ...
        ],
    ) -> Bool:
        """Please refer to `UnsafePointer.__eq__`.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are equal and False otherwise.
        """
        return (
            self.to_unsafe_pointer_unchecked()
            == rhs.to_unsafe_pointer_unchecked()
        )

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    def __eq__(self, rhs: Self) -> Bool:
        """Please refer to `UnsafePointer.__eq__`.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are equal and False otherwise.
        """
        return (
            self.to_unsafe_pointer_unchecked()
            == rhs.to_unsafe_pointer_unchecked()
        )

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    def __ne__(
        self,
        rhs: UnsafeNullablePointer[
            Self.type, address_space=Self.address_space, ...
        ],
    ) -> Bool:
        """Please refer to `UnsafePointer.__ne__`.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are not equal and False otherwise.
        """
        return (
            self.to_unsafe_pointer_unchecked()
            != rhs.to_unsafe_pointer_unchecked()
        )

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    def __ne__(self, rhs: Self) -> Bool:
        """Please refer to `UnsafePointer.__ne__`.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are not equal and False otherwise.
        """
        return (
            self.to_unsafe_pointer_unchecked()
            != rhs.to_unsafe_pointer_unchecked()
        )

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    def __lt__(
        self,
        rhs: UnsafeNullablePointer[
            Self.type, address_space=Self.address_space, ...
        ],
    ) -> Bool:
        """Please refer to `UnsafePointer.__lt__`.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a lower address and False otherwise.
        """
        return (
            self.to_unsafe_pointer_unchecked()
            < rhs.to_unsafe_pointer_unchecked()
        )

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    def __lt__(self, rhs: Self) -> Bool:
        """Please refer to `UnsafePointer.__lt__`.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a lower address and False otherwise.
        """
        return (
            self.to_unsafe_pointer_unchecked()
            < rhs.to_unsafe_pointer_unchecked()
        )

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    def __le__(
        self,
        rhs: UnsafeNullablePointer[
            Self.type, address_space=Self.address_space, ...
        ],
    ) -> Bool:
        """Please refer to `UnsafePointer.__le__`.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a lower address and False otherwise.
        """
        return (
            self.to_unsafe_pointer_unchecked()
            <= rhs.to_unsafe_pointer_unchecked()
        )

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    def __le__(self, rhs: Self) -> Bool:
        """Please refer to `UnsafePointer.__le__`.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a lower address and False otherwise.
        """
        return (
            self.to_unsafe_pointer_unchecked()
            <= rhs.to_unsafe_pointer_unchecked()
        )

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    def __gt__(
        self,
        rhs: UnsafeNullablePointer[
            Self.type, address_space=Self.address_space, ...
        ],
    ) -> Bool:
        """Please refer to `UnsafePointer.__gt__`.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a higher than or equal address and
            False otherwise.
        """
        return (
            self.to_unsafe_pointer_unchecked()
            > rhs.to_unsafe_pointer_unchecked()
        )

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    def __gt__(self, rhs: Self) -> Bool:
        """Please refer to `UnsafePointer.__gt__`.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a higher than or equal address and
            False otherwise.
        """
        return (
            self.to_unsafe_pointer_unchecked()
            > rhs.to_unsafe_pointer_unchecked()
        )

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    def __ge__(
        self,
        rhs: UnsafeNullablePointer[
            Self.type, address_space=Self.address_space, ...
        ],
    ) -> Bool:
        """Please refer to `UnsafePointer.__ge__`.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a higher than or equal address and
            False otherwise.
        """
        return (
            self.to_unsafe_pointer_unchecked()
            >= rhs.to_unsafe_pointer_unchecked()
        )

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    def __ge__(self, rhs: Self) -> Bool:
        """Please refer to `UnsafePointer.__ge__`.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a higher than or equal address and
            False otherwise.
        """
        return (
            self.to_unsafe_pointer_unchecked()
            >= rhs.to_unsafe_pointer_unchecked()
        )

    @always_inline("nodebug")
    def __merge_with__[
        other_type: type_of(
            UnsafeNullablePointer[
                Self.type,
                origin=_,
                address_space=Self.address_space,
            ]
        ),
    ](self) -> UnsafeNullablePointer[
        type=Self.type,
        origin=origin_of(Self.origin, other_type.origin),
        address_space=Self.address_space,
    ]:
        """Please refer to `UnsafePointer.__merge_with__`.

        Parameters:
            other_type: The type of the pointer to merge with.

        Returns:
            A pointer merged with the specified `other_type`.
        """
        return (
            self.to_unsafe_pointer_unchecked().address
        )  # allow kgen.pointer to convert.

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline
    def __bool__(self) -> Bool:
        """Return true if the pointer is non-null.

        Returns:
            Whether the pointer is null.
        """
        return Int(self) != 0

    @always_inline
    def __int__(self) -> Int:
        """Please refer to `UnsafePointer.__int__`.

        Returns:
          The address of the pointer as an Int.
        """
        return Int(self.to_unsafe_pointer_unchecked())

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        """Please refer to `UnsafePointer.write_to`.

        Args:
            writer: The object to write to.
        """
        self.to_unsafe_pointer_unchecked().write_to(writer)

    @no_inline
    def write_repr_to(self, mut writer: Some[Writer]):
        """Write the string representation of the UnsafeNullablePointer.

        Args:
            writer: The object to write to.
        """
        FormatStruct(writer, "UnsafeNullablePointer").params(
            Named("mut", Self.mut),
            TypeNames[Self.type](),
            Named("address_space", Self.address_space),
        ).fields(self)

    # ===-------------------------------------------------------------------===#
    # DevicePassable
    # ===-------------------------------------------------------------------===#

    # Implementation of `DevicePassable`
    comptime device_type: AnyType = Self
    """DeviceBuffer dtypes are remapped to UnsafeNullablePointer when passed to accelerator devices."""

    def _to_device_type(self, target: MutOpaquePointer[_]):
        """Device dtype mapping from DeviceBuffer to the device's UnsafeNullablePointer.
        """
        # TODO: Allow the low-level DeviceContext implementation to intercept
        # these translations.
        target.bitcast[Self.device_type]()[] = self.address

    @staticmethod
    def get_type_name() -> String:
        """
        Gets this type name, for use in error messages when handing arguments
        to kernels.
        TODO: This will go away soon, when we get better error messages for
        kernel calls.

        Returns:
            This name of the type.
        """
        return String(
            "UnsafeNullablePointer[",
            reflect[Self.type]().name(),
            ", mut=",
            Self.mut,
            ", address_space=",
            Self.address_space,
            "]",
        )

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    def swap_pointees[
        U: Movable
    ](
        self: UnsafeNullablePointer[mut=True, U, _],
        other: UnsafeNullablePointer[mut=True, U, _],
    ):
        """Please refer to `UnsafePointer.swap_pointees`.

        Parameters:
            U: The type the pointers point to, which must be `Movable`.

        Args:
            other: The other pointer to swap with.
        """
        self.to_unsafe_pointer_unchecked().swap_pointees(
            other.to_unsafe_pointer_unchecked()
        )

    @always_inline("nodebug")
    def as_noalias_ptr(self) -> Self:
        """Please refer to `UnsafePointer.as_noalias_ptr`.

        Returns:
            A noalias pointer.
        """
        return self.to_unsafe_pointer_unchecked().as_noalias_ptr()

    @always_inline("nodebug")
    def load[
        dtype: DType,
        //,
        width: Int = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
        invariant: Bool = _default_invariant[Self.mut](),
        non_temporal: Bool = False,
    ](self: UnsafeNullablePointer[Scalar[dtype], ...]) -> SIMD[dtype, width]:
        """Please refer to `UnsafePointer.load`.

        Parameters:
            dtype: The data type of the SIMD vector.
            width: The number of elements to load.
            alignment: The minimal alignment (bytes) of the address.
            volatile: Whether the operation is volatile.
            invariant: Whether the load is from invariant memory.
            non_temporal: Whether the load has no temporal locality (streaming).

        Returns:
            The loaded SIMD vector.
        """
        return self.to_unsafe_pointer_unchecked().load[
            width=width,
            alignment=alignment,
            volatile=volatile,
            invariant=invariant,
            non_temporal=non_temporal,
        ]()

    @always_inline("nodebug")
    def load[
        dtype: DType,
        //,
        width: Int = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
        invariant: Bool = _default_invariant[Self.mut](),
        non_temporal: Bool = False,
    ](
        self: UnsafeNullablePointer[Scalar[dtype], ...],
        offset: Scalar,
    ) -> SIMD[
        dtype, width
    ]:
        """Please refer to `UnsafePointer.load`.

        Parameters:
            dtype: The data type of SIMD vector elements.
            width: The size of the SIMD vector.
            alignment: The minimal alignment of the address.
            volatile: Whether the operation is volatile or not.
            invariant: Whether the memory is load invariant.
            non_temporal: Whether the load has no temporal locality (streaming).

        Args:
            offset: The offset to load from.

        Returns:
            The loaded value.
        """
        return self.to_unsafe_pointer_unchecked().load[
            width=width,
            alignment=alignment,
            volatile=volatile,
            invariant=invariant,
            non_temporal=non_temporal,
        ](offset)

    @always_inline("nodebug")
    def load[
        I: Indexer,
        dtype: DType,
        //,
        width: Int = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
        invariant: Bool = _default_invariant[Self.mut](),
        non_temporal: Bool = False,
    ](
        self: UnsafeNullablePointer[Scalar[dtype], ...],
        offset: I,
    ) -> SIMD[
        dtype, width
    ]:
        """Please refer to `UnsafePointer.load`.

        Parameters:
            I: A type that can be used as an index.
            dtype: The data type of SIMD vector elements.
            width: The size of the SIMD vector.
            alignment: The minimal alignment of the address.
            volatile: Whether the operation is volatile or not.
            invariant: Whether the memory is load invariant.
            non_temporal: Whether the load has no temporal locality (streaming).

        Args:
            offset: The offset to load from.

        Returns:
            The loaded value.
        """
        return self.to_unsafe_pointer_unchecked().load[
            width=width,
            alignment=alignment,
            volatile=volatile,
            invariant=invariant,
            non_temporal=non_temporal,
        ](offset)

    @always_inline("nodebug")
    def store[
        I: Indexer,
        dtype: DType,
        //,
        width: SIMDSize = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
        non_temporal: Bool = False,
    ](
        self: UnsafeNullablePointer[mut=True, Scalar[dtype], ...],
        offset: I,
        val: SIMD[dtype, width],
    ):
        """Please refer to `UnsafePointer.store`.

        Parameters:
            I: A type that can be used as an index.
            dtype: The data type of SIMD vector elements.
            width: The size of the SIMD vector.
            alignment: The minimal alignment of the address.
            volatile: Whether the operation is volatile or not.
            non_temporal: Whether the store has no temporal locality (streaming).

        Args:
            offset: The offset to store to.
            val: The value to store.
        """
        self.to_unsafe_pointer_unchecked().store[
            alignment=alignment, volatile=volatile, non_temporal=non_temporal
        ](offset, val)

    @always_inline("nodebug")
    def store[
        dtype: DType,
        offset_type: DType,
        //,
        width: Int = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
        non_temporal: Bool = False,
    ](
        self: UnsafeNullablePointer[mut=True, Scalar[dtype], ...],
        offset: Scalar[offset_type],
        val: SIMD[dtype, width],
    ):
        """Please refer to `UnsafePointer.store`.

        Parameters:
            dtype: The data type of SIMD vector elements.
            offset_type: The data type of the offset value.
            width: The size of the SIMD vector.
            alignment: The minimal alignment of the address.
            volatile: Whether the operation is volatile or not.
            non_temporal: Whether the store has no temporal locality (streaming).

        Args:
            offset: The offset to store to.
            val: The value to store.
        """
        self.to_unsafe_pointer_unchecked().store[
            alignment=alignment, volatile=volatile, non_temporal=non_temporal
        ](offset, val)

    @always_inline("nodebug")
    def store[
        dtype: DType,
        //,
        width: SIMDSize = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
        non_temporal: Bool = False,
    ](
        self: UnsafeNullablePointer[mut=True, Scalar[dtype], ...],
        val: SIMD[dtype, width],
    ):
        """Please refer to `UnsafePointer.store`.

        Parameters:
            dtype: The data type of SIMD vector elements.
            width: The number of elements to store.
            alignment: The minimal alignment (bytes) of the address.
            volatile: Whether the operation is volatile.
            non_temporal: Whether the store has no temporal locality (streaming).

        Args:
            val: The SIMD value to store.
        """
        self.to_unsafe_pointer_unchecked().store[
            alignment=alignment, volatile=volatile, non_temporal=non_temporal
        ](val)

    @always_inline("nodebug")
    def _store[
        dtype: DType,
        width: SIMDSize,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
        non_temporal: Bool = False,
    ](
        self: UnsafeNullablePointer[mut=True, Scalar[dtype], ...],
        val: SIMD[dtype, width],
    ):
        self.to_unsafe_pointer_unchecked()._store[
            alignment=alignment, volatile=volatile, non_temporal=non_temporal
        ](val)

    @always_inline("nodebug")
    def strided_load[
        dtype: DType, T: Intable, //, width: Int
    ](
        self: UnsafeNullablePointer[Scalar[dtype], ...],
        stride: T,
    ) -> SIMD[
        dtype, width
    ]:
        """Please refer to `UnsafePointer.strided_load`.

        Parameters:
            dtype: DType of returned SIMD value.
            T: The Intable type of the stride.
            width: The SIMD width.

        Args:
            stride: The stride between loads.

        Returns:
            A vector which is stride loaded.
        """
        return self.to_unsafe_pointer_unchecked().strided_load[width=width](
            stride
        )

    @always_inline("nodebug")
    def strided_store[
        dtype: DType,
        T: Intable,
        //,
        width: SIMDSize = 1,
    ](
        self: UnsafeNullablePointer[mut=True, Scalar[dtype], ...],
        val: SIMD[dtype, width],
        stride: T,
    ):
        """Please refer to `UnsafePointer.strided_store`.

        Parameters:
            dtype: DType of `val`, the SIMD value to store.
            T: The Intable type of the stride.
            width: The SIMD width.

        Args:
            val: The SIMD value to store.
            stride: The stride between stores.
        """
        self.to_unsafe_pointer_unchecked().strided_store(val, stride)

    @always_inline("nodebug")
    def gather[
        dtype: DType,
        //,
        *,
        width: SIMDSize = 1,
        alignment: Int = align_of[dtype](),
    ](
        self: UnsafeNullablePointer[Scalar[dtype], ...],
        offset: SIMD[_, width],
        mask: SIMD[DType.bool, width] = SIMD[DType.bool, width](fill=True),
        default: SIMD[dtype, width] = 0,
    ) -> SIMD[dtype, width]:
        """Please refer to `UnsafePointer.gather`.

        Parameters:
            dtype: DType of the return SIMD.
            width: The SIMD width.
            alignment: The minimal alignment of the address.

        Args:
            offset: The SIMD vector of offsets to gather from.
            mask: The SIMD vector of boolean values, indicating for each
                element whether to load from memory or to take from the
                `default` SIMD vector.
            default: The SIMD vector providing default values to be taken
                where the `mask` SIMD vector is `False`.

        Returns:
            The SIMD vector containing the gathered values.
        """
        return self.to_unsafe_pointer_unchecked().gather[
            width=width, alignment=alignment
        ](offset, mask, default)

    @always_inline("nodebug")
    def scatter[
        dtype: DType,
        //,
        *,
        width: SIMDSize = 1,
        alignment: Int = align_of[dtype](),
    ](
        self: UnsafeNullablePointer[mut=True, Scalar[dtype], ...],
        offset: SIMD[_, width],
        val: SIMD[dtype, width],
        mask: SIMD[DType.bool, width] = SIMD[DType.bool, width](fill=True),
    ):
        """Please refer to `UnsafePointer.scatter`.

        Parameters:
            dtype: DType of `value`, the result SIMD buffer.
            width: The SIMD width.
            alignment: The minimal alignment of the address.

        Args:
            offset: The SIMD vector of offsets to scatter into.
            val: The SIMD vector containing the values to be scattered.
            mask: The SIMD vector of boolean values, indicating for each
                element whether to store at memory or not.
        """
        self.to_unsafe_pointer_unchecked().scatter[
            width=width, alignment=alignment
        ](offset, val, mask)

    @always_inline
    def free(self: UnsafeNullablePointer[mut=True, Self.type, ...]):
        """Please refer to `UnsafePointer.free`."""
        self.to_unsafe_pointer_unchecked().free()

    @always_inline("nodebug")
    def bitcast[
        T: AnyType
    ](self) -> UnsafeNullablePointer[
        T,
        Self.origin,
        address_space=Self.address_space,
    ]:
        """Please refer to `UnsafePointer.bitcast`.

        Parameters:
            T: The target type.

        Returns:
            A new UnsafeNullablePointer object with the specified type and the same address,
            as the original UnsafeNullablePointer.
        """
        return self.to_unsafe_pointer_unchecked().bitcast[T]().address

    comptime _OriginCastType[
        target_mut: Bool, //, target_origin: Origin[mut=target_mut]
    ] = UnsafeNullablePointer[
        Self.type,
        target_origin,
        address_space=Self.address_space,
    ]

    @always_inline("nodebug")
    def mut_cast[
        target_mut: Bool
    ](self) -> Self._OriginCastType[Self.origin.unsafe_mut_cast[target_mut]()]:
        """Please refer to `UnsafePointer.mut_cast`.

        Parameters:
            target_mut: Mutability of the destination pointer.

        Returns:
            A pointer with the same type, origin and address space as the
            original pointer, but with the newly specified mutability.
        """
        return self.to_unsafe_pointer_unchecked().mut_cast[target_mut]().address

    @always_inline("nodebug")
    def unsafe_mut_cast[
        target_mut: Bool
    ](self) -> Self._OriginCastType[Self.origin.unsafe_mut_cast[target_mut]()]:
        """Please refer to `UnsafePointer.unsafe_mut_cast`.

        Parameters:
            target_mut: Mutability of the destination pointer.

        Returns:
            A pointer with the same type, origin and address space as the
            original pointer, but with the newly specified mutability.
        """
        return (
            self.to_unsafe_pointer_unchecked()
            .unsafe_mut_cast[target_mut]()
            .address
        )

    @always_inline("nodebug")
    def unsafe_origin_cast[
        target_origin: Origin[mut=Self.mut]
    ](self) -> Self._OriginCastType[target_origin]:
        """Please refer to `UnsafePointer.unsafe_origin_cast`.

        Parameters:
            target_origin: Origin of the destination pointer.

        Returns:
            A pointer with the same type, mutability and address space as the
            original pointer, but with the newly specified origin.
        """
        return (
            self.to_unsafe_pointer_unchecked()
            .unsafe_origin_cast[target_origin]()
            .address
        )

    @always_inline("nodebug")
    def as_immutable(
        self,
    ) -> Self._OriginCastType[ImmutOrigin(Self.origin)]:
        """Please refer to `UnsafePointer.as_immutable`.

        Returns:
            A pointer with the mutability set to immutable.
        """
        return self.to_unsafe_pointer_unchecked().as_immutable().address

    @always_inline("nodebug")
    def as_any_origin(
        self,
    ) -> UnsafeNullablePointer[
        Self.type,
        AnyOrigin[mut=Self.mut],
        address_space=Self.address_space,
    ]:
        """Please refer to `UnsafePointer.as_any_origin`.

        Returns:
            A pointer with the origin set to `AnyOrigin`.
        """
        return self.to_unsafe_pointer_unchecked().as_any_origin().address

    @always_inline("nodebug")
    def address_space_cast[
        target_address_space: AddressSpace = Self.address_space,
    ](self) -> UnsafeNullablePointer[
        Self.type,
        Self.origin,
        address_space=target_address_space,
    ]:
        """Please refer to `UnsafePointer.address_space_cast`.

        Parameters:
            target_address_space: The address space of the result.

        Returns:
            A new UnsafeNullablePointer object with the same type and the same address,
            as the original UnsafeNullablePointer and the new address space.
        """
        return (
            self.to_unsafe_pointer_unchecked()
            .address_space_cast[target_address_space]()
            .address
        )

    @always_inline
    def destroy_pointee[
        T: ImplicitlyDestructible, //
    ](self: UnsafeNullablePointer[T, _]) where type_of(self).mut:
        """Please refer to `UnsafePointer.destroy_pointee`.

        Parameters:
            T: Pointee type that can be destroyed implicitly (without
              deinitializer arguments).

        """
        self.to_unsafe_pointer_unchecked().destroy_pointee()

    # TODO(MOCO-2367): Use a `unified` closure parameter here instead.
    @always_inline
    def destroy_pointee_with(
        self: UnsafeNullablePointer[
            Self.type,
            _,
            address_space=AddressSpace.GENERIC,
        ],
        destroy_func: def(var Self.type) thin,
    ) where type_of(self).mut:
        """Please refer to `UnsafePointer.destroy_pointee_with`.

        Args:
            destroy_func: A function that takes ownership of the pointee value
                for the purpose of deinitializing it.
        """
        self.to_unsafe_pointer_unchecked().destroy_pointee_with(destroy_func)

    @always_inline
    def take_pointee[
        T: Movable,
        //,
    ](self: UnsafeNullablePointer[T, _]) -> T where type_of(self).mut:
        """Please refer to `UnsafePointer.take_pointee`.

        Parameters:
            T: The type the pointer points to, which must be `Movable`.

        Returns:
            The value at the pointer.
        """
        return self.to_unsafe_pointer_unchecked().take_pointee()

    @always_inline
    def init_pointee_move[
        T: Movable,
        //,
    ](self: UnsafeNullablePointer[T, _], var value: T) where type_of(self).mut:
        """Please refer to `UnsafePointer.init_pointee_move`.

        Parameters:
            T: The type the pointer points to, which must be `Movable`.

        Args:
            value: The value to emplace.
        """
        self.to_unsafe_pointer_unchecked().init_pointee_move(value^)

    @always_inline
    def init_pointee_copy[
        T: Copyable,
        //,
    ](self: UnsafeNullablePointer[T, _], value: T) where type_of(self).mut:
        """Please refer to `UnsafePointer.init_pointee_copy`.

        Parameters:
            T: The type the pointer points to, which must be `Copyable`.

        Args:
            value: The value to emplace.
        """
        self.to_unsafe_pointer_unchecked().init_pointee_copy(value)

    @always_inline
    def init_pointee_move_from[
        T: Movable,
        //,
    ](
        self: UnsafeNullablePointer[T, _], src: UnsafeNullablePointer[T, _]
    ) where (type_of(self).mut) and (type_of(src).mut):
        """Please refer to `UnsafePointer.init_pointee_move_from`.

        Parameters:
            T: The type the pointer points to, which must be `Movable`.

        Args:
            src: Source pointer that the value will be moved from.
        """
        self.to_unsafe_pointer_unchecked().init_pointee_move_from(
            src.to_unsafe_pointer_unchecked()
        )
