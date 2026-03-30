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
"""Implements a non-null pointer type that guarantees a valid address.

This module provides `NonNullUnsafePointer`, a pointer type with the same API as
`UnsafePointer` except that it has no default constructor (preventing null
initialization) and no `__bool__` conversion (since the pointer is always
non-null, a boolean check is meaningless).
"""

from std.builtin.format_int import _write_int
from std.compile import get_type_name
from std.format._utils import FormatStruct, Named, TypeNames
from std.sys import align_of, size_of
from std.utils._nicheable import UnsafeSingleNicheable


struct _Null[address_space: AddressSpace = AddressSpace.GENERIC](
    Defaultable, Intable, TrivialRegisterPassable
):
    comptime _mlir_type = __mlir_type[
        `!kgen.pointer<`,
        NoneType,
        `, `,
        Self.address_space._value._mlir_value,
        `>`,
    ]

    var address: Self._mlir_type

    @always_inline("builtin")
    def __init__(out self):
        self.address = __mlir_attr[`#interp.pointer<0> : `, Self._mlir_type]

    @always_inline("nodebug")
    def __int__(self) -> Int:
        return Int(mlir_value=__mlir_op.`pop.pointer_to_index`(self.address))


struct NonNullUnsafePointer[
    mut: Bool,
    //,
    type: AnyType,
    origin: Origin[mut=mut],
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
](
    Comparable,
    ImplicitlyCopyable,
    Intable,
    TrivialRegisterPassable,
    UnsafeSingleNicheable,
    Writable,
):
    """`NonNullUnsafePointer` is a pointer type that is guaranteed to be non-null.

    It has almost the same API as `UnsafePointer`, with two key differences:

    - There is no default constructor, so a `NonNullUnsafePointer` cannot be
      null-initialized.
    - There is no `__bool__` method, since the pointer is always non-null and a
      boolean check would be meaningless.

    Like `UnsafePointer`, this pointer is unsafe. No bounds checks are
    performed, and reading before writing is undefined behavior. It does not
    own existing memory.

    Parameters:
        mut: Whether the origin is mutable.
        type: The type the pointer points to.
        origin: The origin of the memory being addressed.
        address_space: The address space associated with the pointer.
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

    comptime _OriginCastType[
        target_mut: Bool, //, target_origin: Origin[mut=target_mut]
    ] = NonNullUnsafePointer[
        Self.type,
        target_origin,
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

    @doc_hidden
    @always_inline("builtin")
    def __init__(out self, *, _mlir_value: Self._mlir_type):
        """Create a pointer from a low-level pointer primitive.

        Args:
            _mlir_value: The MLIR value of the pointer to construct with.
        """
        self.address = _mlir_value

    @always_inline
    def __init__(out self, *, unsafe_from_address: Int):
        """Create a pointer from a raw address.

        The caller is responsible for ensuring the address is valid and
        non-null.

        Args:
            unsafe_from_address: The raw address to create a pointer from.

        Safety:
            Creating a pointer from a raw address is inherently unsafe as the
            caller must ensure the address is valid before writing to it, and
            that the memory is initialized before reading from it. The caller
            must also ensure the pointer's origin and mutability is valid for
            the address, and that the address is non-null.
        """
        comptime assert (
            size_of[type_of(self)]() == size_of[Int]()
        ), "Pointer/Int size mismatch"
        assert unsafe_from_address != Int(
            _Null[Self.address_space]()
        ), "cannot create a non-null pointer from the null address"
        self = NonNullUnsafePointer(to=unsafe_from_address).bitcast[
            type_of(self)
        ]()[]

    @always_inline("nodebug")
    def __init__(
        out self,
        *,
        ref[Self.origin, Self.address_space._value._mlir_value] to: Self.type,
    ):
        """Constructs a NonNullUnsafePointer from a reference to a value.

        Args:
            to: The value to construct a pointer to.
        """
        self = {
            _mlir_value = __mlir_op.`lit.ref.to_pointer`(
                __get_mvalue_as_litref(to)
            )
        }

    @always_inline("builtin")
    def __init__(
        *,
        unsafe_from_nullable: UnsafePointer[
            Self.type,
            origin=Self.origin,
            address_space=Self.address_space,
        ],
        out self,
    ):
        """Constructs a NonNullUnsafePointer from an UnsafePointer.

        The caller is responsible for ensuring the UnsafePointer is non-null.

        Args:
            unsafe_from_nullable: The UnsafePointer to construct from.

        Safety:
            The provided pointer must not be null.
        """
        self.address = __mlir_op.`pop.pointer.bitcast`[_type=Self._mlir_type](
            unsafe_from_nullable.address
        )

    @always_inline("builtin")
    @implicit
    def __init__[
        disambig2: Int = 0
    ](
        other: NonNullUnsafePointer,
        out self: NonNullUnsafePointer[
            other.type,
            ImmutOrigin(other.origin),
            address_space=other.address_space,
        ],
    ):
        """Implicitly casts a mutable NonNullUnsafePointer to immutable.

        Args:
            other: The mutable pointer to cast from.

        Parameters:
            disambig2: Ignored. Works around name mangling conflict.
        """
        self.address = __mlir_op.`pop.pointer.bitcast`[
            _type=type_of(self)._mlir_type
        ](other.address)

    @always_inline("builtin")
    @implicit
    def __init__[
        disambig: Int = 0
    ](
        other: NonNullUnsafePointer[mut=True, ...],
        out self: NonNullUnsafePointer[
            other.type,
            MutAnyOrigin,
            address_space=other.address_space,
        ],
    ):
        """Implicitly casts a mutable NonNullUnsafePointer to `MutAnyOrigin`.

        Args:
            other: The mutable pointer to cast from.

        Parameters:
            disambig: Ignored. Works around name mangling conflict.
        """
        self.address = __mlir_op.`pop.pointer.bitcast`[
            _type=type_of(self)._mlir_type
        ](other.address)

    @always_inline("builtin")
    @implicit
    def __init__(
        other: NonNullUnsafePointer[...],
        out self: NonNullUnsafePointer[
            other.type,
            ImmutAnyOrigin,
            address_space=other.address_space,
        ],
    ):
        """Implicitly casts a NonNullUnsafePointer to `ImmutAnyOrigin`.

        Args:
            other: The pointer to cast from.
        """
        self.address = __mlir_op.`pop.pointer.bitcast`[
            _type=type_of(self)._mlir_type
        ](other.address)

    @always_inline
    @staticmethod
    def dangling() -> Self:
        """Creates a new `NonNullUnsafePointer` that is dangling, but well-aligned.

        This is useful for initializing types which lazily allocate.

        Note that the address of the returned pointer may potentially be that
        of a valid pointer, which means this must not be used as a "not yet
        initialized" sentinel value. Types that lazily allocate must track
        initialization by some other means.

        Returns:
            A dangling but well-aligned `NonNullUnsafePointer`.

        Example:

        ```mojo
        var ptr = NonNullUnsafePointer[Int, MutExternalOrigin].dangling()
        # Important: don't try to access the value of `ptr` without
        # initializing it first! The pointer is not null but isn't valid either!
        ```
        """
        comptime alignment = align_of[Self.type]()
        return Self(unsafe_from_address=alignment)

    @always_inline
    def free(self: NonNullUnsafePointer[Self.type, ExternalOrigin[mut=True]]):
        """Free the memory referenced by the pointer."""
        self._as_unsafe_pointer().free()

    # ===-------------------------------------------------------------------===#
    # Conversion to/from UnsafePointer
    # ===-------------------------------------------------------------------===#

    @always_inline("builtin")
    def _as_unsafe_pointer(
        self,
        out result: UnsafePointer[
            Self.type,
            origin=Self.origin,
            address_space=Self.address_space,
        ],
    ):
        result = self.address

    # ===------------------------------------------------------------------===#
    # UnsafeNicheable
    # ===------------------------------------------------------------------===#

    @staticmethod
    @always_inline
    @doc_hidden
    def write_niche(
        memory: UnsafePointer[mut=True, UnsafeMaybeUninit[Self], _]
    ):
        memory.bitcast[_Null[Self.address_space]]().init_pointee_move({})

    @staticmethod
    @always_inline
    @doc_hidden
    def isa_niche(
        memory: UnsafePointer[mut=False, UnsafeMaybeUninit[Self], _]
    ) -> Bool:
        comptime NullType = _Null[Self.address_space]
        return Int(memory.bitcast[NullType]()[]) == Int(NullType())

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    def __getitem__(self) -> ref[Self.origin, Self.address_space] Self.type:
        """Return a reference to the underlying data.

        Returns:
            A reference to the value.

        Safety:
            The pointer must point to initialized memory.
        """
        return self._as_unsafe_pointer()[]

    @always_inline("nodebug")
    def __getitem__[
        I: Indexer, //
    ](self, offset: I) -> ref[Self.origin, Self.address_space] Self.type:
        """Return a reference to the underlying data, offset by the given index.

        Parameters:
            I: A type that can be used as an index.

        Args:
            offset: The offset index.

        Returns:
            An offset reference.
        """
        return self._as_unsafe_pointer()[offset]

    @always_inline("nodebug")
    def __add__[I: Indexer, //](self, offset: I) -> Self:
        """Return a pointer at an offset from the current one.

        Parameters:
            I: A type that can be used as an index.

        Args:
            offset: The offset index.

        Returns:
            An offset pointer.
        """
        return {
            _mlir_value = __mlir_op.`pop.offset`(
                self.address, index(offset)._mlir_value
            )
        }

    @always_inline
    def __sub__[I: Indexer, //](self, offset: I) -> Self:
        """Return a pointer at a negative offset from the current one.

        Parameters:
            I: A type that can be used as an index.

        Args:
            offset: The offset index.

        Returns:
            An offset pointer.
        """
        return self + (-1 * index(offset))

    @always_inline
    def __iadd__[I: Indexer, //](mut self, offset: I):
        """Add an offset to this pointer.

        Parameters:
            I: A type that can be used as an index.

        Args:
            offset: The offset index.
        """
        self = self + offset

    @always_inline
    def __isub__[I: Indexer, //](mut self, offset: I):
        """Subtract an offset from this pointer.

        Parameters:
            I: A type that can be used as an index.

        Args:
            offset: The offset index.
        """
        self = self - offset

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    def __eq__(
        self,
        rhs: NonNullUnsafePointer[
            Self.type, address_space=Self.address_space, ...
        ],
    ) -> Bool:
        """Returns True if the two pointers are equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are equal and False otherwise.
        """
        return Int(self) == Int(rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    def __eq__(self, rhs: Self) -> Bool:
        """Returns True if the two pointers are equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are equal and False otherwise.
        """
        return Int(self) == Int(rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    def __ne__(
        self,
        rhs: NonNullUnsafePointer[
            Self.type, address_space=Self.address_space, ...
        ],
    ) -> Bool:
        """Returns True if the two pointers are not equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are not equal and False otherwise.
        """
        return not (self == rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    def __ne__(self, rhs: Self) -> Bool:
        """Returns True if the two pointers are not equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are not equal and False otherwise.
        """
        return not (self == rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    def __lt__(
        self,
        rhs: NonNullUnsafePointer[
            Self.type, address_space=Self.address_space, ...
        ],
    ) -> Bool:
        """Returns True if this pointer represents a lower address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a lower address and False otherwise.
        """
        return Int(self) < Int(rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    def __lt__(self, rhs: Self) -> Bool:
        """Returns True if this pointer represents a lower address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a lower address and False otherwise.
        """
        return Int(self) < Int(rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    def __le__(
        self,
        rhs: NonNullUnsafePointer[
            Self.type, address_space=Self.address_space, ...
        ],
    ) -> Bool:
        """Returns True if this pointer represents a lower than or equal
           address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a lower or equal address and False
            otherwise.
        """
        return Int(self) <= Int(rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    def __le__(self, rhs: Self) -> Bool:
        """Returns True if this pointer represents a lower than or equal
           address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a lower or equal address and False
            otherwise.
        """
        return Int(self) <= Int(rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    def __gt__(
        self,
        rhs: NonNullUnsafePointer[
            Self.type, address_space=Self.address_space, ...
        ],
    ) -> Bool:
        """Returns True if this pointer represents a higher address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a higher address and False otherwise.
        """
        return Int(self) > Int(rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    def __gt__(self, rhs: Self) -> Bool:
        """Returns True if this pointer represents a higher address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a higher address and False otherwise.
        """
        return Int(self) > Int(rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    def __ge__(
        self,
        rhs: NonNullUnsafePointer[
            Self.type, address_space=Self.address_space, ...
        ],
    ) -> Bool:
        """Returns True if this pointer represents a higher than or equal
           address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a higher or equal address and False
            otherwise.
        """
        return Int(self) >= Int(rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    def __ge__(self, rhs: Self) -> Bool:
        """Returns True if this pointer represents a higher than or equal
           address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a higher or equal address and False
            otherwise.
        """
        return Int(self) >= Int(rhs)

    @always_inline("builtin")
    def __merge_with__[
        other_type: type_of(
            NonNullUnsafePointer[
                Self.type,
                origin=_,
                address_space=Self.address_space,
            ]
        ),
    ](self) -> NonNullUnsafePointer[
        type=Self.type,
        origin=origin_of(Self.origin, other_type.origin),
        address_space=Self.address_space,
    ]:
        """Returns a pointer merged with the specified `other_type`.

        Parameters:
            other_type: The type of the pointer to merge with.

        Returns:
            A pointer merged with the specified `other_type`.
        """
        return {_mlir_value = self.address}

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline
    def __int__(self) -> Int:
        """Returns the pointer address as an integer.

        Returns:
            The address of the pointer as an Int.
        """
        return self._as_unsafe_pointer().__int__()

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        """Formats this pointer address to the provided Writer.

        Args:
            writer: The object to write to.
        """
        _write_int[radix=16](writer, Scalar[DType.int](Int(self)), prefix="0x")

    @no_inline
    def write_repr_to(self, mut writer: Some[Writer]):
        """Write the string representation of the NonNullUnsafePointer.

        Args:
            writer: The object to write to.
        """
        FormatStruct(writer, "NonNullUnsafePointer").params(
            Named("mut", Self.mut),
            TypeNames[Self.type](),
            Named("address_space", Self.address_space),
        ).fields(self)

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    def swap_pointees[
        U: Movable, //
    ](
        self: NonNullUnsafePointer[mut=True, U, _],
        other: NonNullUnsafePointer[mut=True, U, _],
    ):
        """Swap the values at the pointers.

        This function assumes that `self` and `other` _may_ overlap in memory.
        If that is not the case, or when references are available, you should
        use `builtin.swap` instead.

        Parameters:
            U: The type the pointers point to, which must be `Movable`.

        Args:
            other: The other pointer to swap with.

        Safety:
            - `self` and `other` must both point to valid, initialized instances
              of `T`.
        """
        self._as_unsafe_pointer().swap_pointees(other._as_unsafe_pointer())

    @always_inline("nodebug")
    def as_noalias_ptr(self) -> Self:
        """Cast the pointer to a new pointer that is known not to locally alias
        any other pointer.

        This information is relayed to the optimizer. If the pointer does
        locally alias another memory value, the behaviour is undefined.

        Returns:
            A noalias pointer.
        """
        return {
            unsafe_from_nullable = self._as_unsafe_pointer().as_noalias_ptr()
        }

    @always_inline("builtin")
    def bitcast[
        T: AnyType
    ](self) -> NonNullUnsafePointer[
        T,
        Self.origin,
        address_space=Self.address_space,
    ]:
        """Bitcasts a NonNullUnsafePointer to a different type.

        Parameters:
            T: The target type.

        Returns:
            A new pointer object with the specified type and the same
            address, mutability, and origin as the original pointer.
        """
        return {unsafe_from_nullable = self._as_unsafe_pointer().bitcast[T]()}

    @always_inline("builtin")
    def unsafe_mut_cast[
        target_mut: Bool
    ](self) -> Self._OriginCastType[Self.origin.unsafe_mut_cast[target_mut]()]:
        """Changes the mutability of a pointer.

        Parameters:
            target_mut: Mutability of the destination pointer.

        Returns:
            A pointer with the same type, origin and address space as the
            original pointer, but with the newly specified mutability.

        Safety:
            Casting the mutability of a pointer is inherently very unsafe.
            Improper usage can lead to undefined behavior.
        """
        return {
            unsafe_from_nullable = self._as_unsafe_pointer().unsafe_mut_cast[
                target_mut
            ]()
        }

    @always_inline("builtin")
    def unsafe_origin_cast[
        target_origin: Origin[mut=Self.mut]
    ](self) -> Self._OriginCastType[target_origin]:
        """Changes the origin of a pointer.

        Parameters:
            target_origin: Origin of the destination pointer.

        Returns:
            A pointer with the same type, mutability and address space as the
            original pointer, but with the newly specified origin.

        Safety:
            Casting the origin of a pointer is inherently very unsafe.
            Improper usage can lead to undefined behavior or unexpected variable
            destruction.
        """
        return {
            unsafe_from_nullable = self._as_unsafe_pointer().unsafe_origin_cast[
                target_origin
            ]()
        }

    @always_inline("builtin")
    def as_immut(
        self,
    ) -> Self._OriginCastType[ImmutOrigin(Self.origin)]:
        """Changes the mutability of a pointer to immutable.

        Unlike `unsafe_mut_cast`, this function is always safe to use as casting
        from (im)mutable to immutable is always safe.

        Returns:
            A pointer with the mutability set to immutable.
        """
        return {unsafe_from_nullable = self._as_unsafe_pointer().as_immutable()}

    @always_inline("builtin")
    def as_any_origin(
        self,
    ) -> NonNullUnsafePointer[
        Self.type,
        AnyOrigin[mut=Self.mut],
        address_space=Self.address_space,
    ]:
        """Casts the origin of a pointer to `AnyOrigin`.

        Returns:
            A pointer with the origin set to `AnyOrigin`.
        """
        return {
            _mlir_value = __mlir_op.`pop.pointer.bitcast`[
                _type=NonNullUnsafePointer[
                    Self.type,
                    AnyOrigin[mut=Self.mut],
                    address_space=Self.address_space,
                ]._mlir_type,
            ](self.address)
        }

    @always_inline("builtin")
    def address_space_cast[
        target_address_space: AddressSpace = Self.address_space,
    ](self) -> NonNullUnsafePointer[
        Self.type,
        Self.origin,
        address_space=target_address_space,
    ]:
        """Casts this pointer to a different address space.

        Parameters:
            target_address_space: The address space of the result.

        Returns:
            A new pointer object with the same type and the same address,
            as the original pointer and the new address space.
        """
        return {
            unsafe_from_nullable = self._as_unsafe_pointer().address_space_cast[
                target_address_space
            ]()
        }

    @always_inline
    def destroy_pointee[
        T: ImplicitlyDestructible, //
    ](self: NonNullUnsafePointer[T, _]) where type_of(self).mut:
        """Destroy the pointed-to value.

        The pointer memory location is assumed to contain a valid initialized
        instance of `type`. This is equivalent to `_ = self.take_pointee()`
        but doesn't require `Movable` and is more efficient because it doesn't
        invoke a move constructor.

        Parameters:
            T: Pointee type that can be destroyed implicitly (without
              deinitializer arguments).
        """
        _ = __get_address_as_owned_value(self.address)

    @always_inline
    def destroy_pointee_with(
        self: NonNullUnsafePointer[
            Self.type,
            _,
            address_space=AddressSpace.GENERIC,
        ],
        destroy_func: def(var Self.type),
    ) where type_of(self).mut:
        """Destroy the pointed-to value using a user-provided destructor function.

        This can be used to destroy non-`ImplicitlyDestructible` values in-place
        without moving.

        Args:
            destroy_func: A function that takes ownership of the pointee value
                for the purpose of deinitializing it.
        """
        destroy_func(__get_address_as_owned_value(self.address))

    @always_inline
    def take_pointee[
        T: Movable,
        //,
    ](self: NonNullUnsafePointer[T, _]) -> T where type_of(self).mut:
        """Move the value at the pointer out, leaving it uninitialized.

        The pointer memory location is assumed to contain a valid initialized
        instance of `T`.

        This performs a _consuming_ move, ending the origin of the value stored
        in this pointer memory location. Subsequent reads of this pointer are
        not valid. If a new valid value is stored using `init_pointee_move()`,
        then reading from this pointer becomes valid again.

        Parameters:
            T: The type the pointer points to, which must be `Movable`.

        Returns:
            The value at the pointer.
        """
        return __get_address_as_owned_value(self.address)

    @always_inline
    def init_pointee_move[
        T: Movable,
        //,
    ](self: NonNullUnsafePointer[T, _], var value: T) where type_of(self).mut:
        """Emplace a new value into the pointer location, moving from `value`.

        The pointer memory location is assumed to contain uninitialized data,
        and consequently the current contents of this pointer are not destructed
        before writing `value`. Similarly, ownership of `value` is logically
        transferred into the pointer location.

        Parameters:
            T: The type the pointer points to, which must be `Movable`.

        Args:
            value: The value to emplace.
        """
        __get_address_as_uninit_lvalue(self.address) = value^

    @always_inline
    def init_pointee_copy[
        T: Copyable,
        //,
    ](self: NonNullUnsafePointer[T, _], value: T) where type_of(self).mut:
        """Emplace a copy of `value` into the pointer location.

        The pointer memory location is assumed to contain uninitialized data,
        and consequently the current contents of this pointer are not destructed
        before writing `value`. Similarly, ownership of `value` is logically
        transferred into the pointer location.

        Parameters:
            T: The type the pointer points to, which must be `Copyable`.

        Args:
            value: The value to emplace.
        """
        __get_address_as_uninit_lvalue(self.address) = value.copy()

    @always_inline
    def init_pointee_move_from[
        T: Movable,
        //,
    ](self: NonNullUnsafePointer[T, _], src: NonNullUnsafePointer[T, _]) where (
        type_of(self).mut
    ) and (type_of(src).mut):
        """Moves the value `src` points to into the memory location pointed to
        by `self`.

        The `self` pointer memory location is assumed to contain uninitialized
        data prior to this assignment, and consequently the current contents of
        this pointer are not destructed before writing the value from the `src`
        pointer.

        Ownership of the value is logically transferred from `src` into `self`'s
        pointer location.

        After this call, the `src` pointee value should be treated as
        uninitialized data. Subsequent reads of or destructor calls on the `src`
        pointee value are invalid, unless and until a new valid value has been
        moved into the `src` pointer's memory location using an
        `init_pointee_*()` operation.

        Parameters:
            T: The type the pointer points to, which must be `Movable`.

        Args:
            src: Source pointer that the value will be moved from.
        """
        __get_address_as_uninit_lvalue(
            self.address
        ) = __get_address_as_owned_value(src.address)


# ===-----------------------------------------------------------------------===#
# bitcast
# ===-----------------------------------------------------------------------===#


@always_inline
def bitcast[
    From: AnyType,
    origin: Origin,
    //,
    To: AnyType,
](pointer: Optional[NonNullUnsafePointer[From, origin]]) -> Optional[
    NonNullUnsafePointer[To, origin]
]:
    """Bitcasts an `Optional[NonNullUnsafePointer]` to point to a different type.

    Parameters:
        From: The source pointee type.
        origin: The origin of the pointer.
        To: The target pointee type.

    Args:
        pointer: The optional pointer to bitcast.

    Returns:
        An optional pointer to `To` with the same address and origin.
    """
    return NonNullUnsafePointer(to=pointer).bitcast[
        Optional[NonNullUnsafePointer[To, origin]]
    ]()[]


@always_inline
def unsafe_origin_cast[
    T: AnyType,
    from_origin: Origin,
    address_space: AddressSpace,
    //,
    to_origin: Origin,
](
    pointer: Optional[
        NonNullUnsafePointer[T, from_origin, address_space=address_space]
    ]
) -> Optional[NonNullUnsafePointer[T, to_origin, address_space=address_space]]:
    try:
        return (
            pointer[]
            .unsafe_mut_cast[to_origin.mut]()
            .unsafe_origin_cast[to_origin]()
        )
    except:
        return {}


@always_inline
def address_of[
    T: AnyType,
    origin: Origin,
    //,
](pointer: Optional[NonNullUnsafePointer[T, origin]]) -> Int:
    try:
        return Int(pointer[])
    except:
        return Int(_Null())
