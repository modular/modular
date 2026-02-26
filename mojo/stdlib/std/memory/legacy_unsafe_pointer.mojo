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
"""Implement a generic unsafe pointer type.

These APIs are imported automatically, just like builtins.
"""

from sys import align_of, is_gpu, is_nvidia_gpu, size_of
from sys.intrinsics import gather, scatter, strided_load, strided_store

from builtin.type_aliases import _lit_origin_type_of_mut
from builtin.rebind import downcast
from builtin.simd import _simd_construction_checks
from builtin.variadics import Variadic
from format._utils import (
    FormatStruct,
    Named,
    TypeNames,
    constrained_conforms_to_writable,
)
from memory import memcpy
from memory.memory import _free, _malloc
from memory import UnsafeMaybeUninit
from os import abort
from python import PythonObject

from builtin.device_passable import DevicePassable
from compile import get_type_name

# ===----------------------------------------------------------------------=== #
# LegacyUnsafePointer
# ===----------------------------------------------------------------------=== #


@always_inline
fn _default_invariant[mut: Bool]() -> Bool:
    return is_gpu() and mut == False


struct LegacyUnsafePointer[
    mut: Bool,
    //,
    type: AnyType,
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
    _mlir_origin: _lit_origin_type_of_mut[mut] = AnyOrigin[
        mut=mut
    ]._mlir_origin,
    origin: Origin[mut=mut, _mlir_origin=_mlir_origin] = Origin[
        _mlir_origin=_mlir_origin
    ](),
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
    """`LegacyUnsafePointer` is a deprecated pointer type that is replaced by
    the new `UnsafePointer`. It is kept for backwards compatibility and will
    be removed in a future version of Mojo.

    In the interim, you can implicitly convert between a `LegacyUnsafePointer`
    and an `UnsafePointer`. This will allow you to pass a legacy pointer to an
    `UnsafePointer` function, and vice versa.

    Parameters:
        mut: Whether the origin is mutable.
        type: The type the pointer points to.
        address_space: The address space associated with the pointer's allocated memory.
        _mlir_origin: The raw MLIR origin value.
        origin: The origin of the memory being addressed.
    """

    # ===-------------------------------------------------------------------===#
    # Aliases
    # ===-------------------------------------------------------------------===#

    comptime _UnsafePointerType = UnsafePointer[
        Self.type,
        Self.origin,
        address_space = Self.address_space,
    ]

    # Fields
    comptime _mlir_type = __mlir_type[
        `!kgen.pointer<`,
        Self.type,
        `, `,
        Self.address_space._value._mlir_value,
        `>`,
    ]
    """The underlying pointer type."""

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var address: Self._mlir_type
    """The underlying pointer."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline("builtin")
    fn __init__(out self):
        """Create a null pointer."""
        self.address = __mlir_attr[`#interp.pointer<0> : `, Self._mlir_type]

    @doc_private
    @always_inline("builtin")
    @implicit
    fn __init__(out self, value: Self._mlir_type):
        """Create a pointer from a low-level pointer primitive.

        Args:
            value: The MLIR value of the pointer to construct with.
        """
        self.address = value

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
        self = Self(__mlir_op.`lit.ref.to_pointer`(__get_mvalue_as_litref(to)))

    @always_inline("builtin")
    @implicit
    fn __init__(
        out self,
        other: LegacyUnsafePointer[
            Self.type, address_space = Self.address_space, ...
        ],
    ):
        """Exclusivity parameter cast a pointer.

        Args:
            other: Pointer to cast.
        """
        self.address = __mlir_op.`pop.pointer.bitcast`[_type = Self._mlir_type](
            other.address
        )

    @always_inline("builtin")
    @implicit
    fn __init__(
        out self,
        other: UnsafePointer[
            Self.type,
            Self.origin,
            address_space = Self.address_space,
        ],
    ):
        """Implicitly cast an `UnsafePointer` to a `LegacyUnsafePointer`.

        Args:
            other: The `UnsafePointer` to cast from.
        """
        self.address = __mlir_op.`pop.pointer.bitcast`[_type = Self._mlir_type](
            other.address
        )

    @always_inline("builtin")
    @implicit
    fn __init__(
        out self,
        other: UnsafePointer[
            Self.type, address_space = Self.address_space, ...
        ],
    ):
        """Implicitly cast an `UnsafePointer` to a `LegacyUnsafePointer`.

        Args:
            other: The `UnsafePointer` to cast from.
        """
        self.address = __mlir_op.`pop.pointer.bitcast`[_type = Self._mlir_type](
            other.address
        )

    fn __init__[
        T: ImplicitlyDestructible, //
    ](
        out self: LegacyUnsafePointer[T, origin = Self.origin],
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
    # Factory methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    @always_inline
    fn alloc(
        count: Int, *, alignment: Int = align_of[Self.type]()
    ) -> LegacyUnsafePointer[
        Self.type,
        address_space = AddressSpace.GENERIC,
        origin=MutExternalOrigin,
    ]:
        """Allocates contiguous storage for `count` elements of `type`
        with compile-time alignment `alignment`.

        - The returned memory is uninitialized; reading before writing is undefined.
        - The returned pointer has an empty mutable origin; you must call `free()`
          to release it.
        - `count` must be positive and `size_of[type]()` must be > 0.

        Example:

        ```mojo
        var p = LegacyUnsafePointer[Int32].alloc(4)
        p.store(0, Int32(42))
        p.store(1, Int32(7))
        p.store(2, Int32(9))
        var a = p.load(0)
        print(a[0], p.load(1)[0], p.load(2)[0])
        p.free()
        ```

        Args:
            count: Number of elements to allocate.
            alignment: The alignment of the allocation.

        Returns:
            Pointer to the newly allocated uninitialized array.
        """
        comptime size_of_t = size_of[Self.type]()
        comptime assert size_of_t > 0, "size must be greater than zero"
        return _malloc[Self.type](size_of_t * count, alignment=alignment)

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __getitem__(self) -> ref[Self.origin, Self.address_space] Self.type:
        """Return a reference to the underlying data.

        Returns:
            A reference to the value.
        """

        # We're unsafe, so we can have unsafe things.
        comptime _ref_type = Pointer[Self.type, Self.origin, Self.address_space]
        return __get_litref_as_mvalue(
            __mlir_op.`lit.ref.from_pointer`[_type = _ref_type._mlir_type](
                self.address
            )
        )

    @always_inline("nodebug")
    fn __getitem__[
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
        return (self + offset)[]

    @always_inline("nodebug")
    fn __add__[I: Indexer, //](self, offset: I) -> Self:
        """Return a pointer at an offset from the current one.

        Parameters:
            I: A type that can be used as an index.

        Args:
            offset: The offset index.

        Returns:
            An offset pointer.
        """
        return __mlir_op.`pop.offset`(self.address, index(offset)._mlir_value)

    @always_inline
    fn __sub__[I: Indexer, //](self, offset: I) -> Self:
        """Return a pointer at an offset from the current one.

        Parameters:
            I: A type that can be used as an index.

        Args:
            offset: The offset index.

        Returns:
            An offset pointer.
        """
        return self + (-1 * index(offset))

    @always_inline
    fn __iadd__[I: Indexer, //](mut self, offset: I):
        """Add an offset to this pointer.

        Parameters:
            I: A type that can be used as an index.

        Args:
            offset: The offset index.
        """
        self = self + offset

    @always_inline
    fn __isub__[I: Indexer, //](mut self, offset: I):
        """Subtract an offset from this pointer.

        Parameters:
            I: A type that can be used as an index.

        Args:
            offset: The offset index.
        """
        self = self - offset

    # This decorator informs the compiler that indirect address spaces are not
    # dereferenced by the method.
    # TODO: replace with a safe model that checks the body of the method for
    # accesses to the origin.
    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __eq__(self, rhs: Self) -> Bool:
        """Returns True if the two pointers are equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are equal and False otherwise.
        """
        return Int(self) == Int(rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __ne__(self, rhs: Self) -> Bool:
        """Returns True if the two pointers are not equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are not equal and False otherwise.
        """
        return not (self == rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __lt__(self, rhs: Self) -> Bool:
        """Returns True if this pointer represents a lower address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a lower address and False otherwise.
        """
        return Int(self) < Int(rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __le__(self, rhs: Self) -> Bool:
        """Returns True if this pointer represents a lower than or equal
           address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a lower address and False otherwise.
        """
        return Int(self) <= Int(rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __gt__(self, rhs: Self) -> Bool:
        """Returns True if this pointer represents a higher address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a higher than or equal address and
            False otherwise.
        """
        return Int(self) > Int(rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __ge__(self, rhs: Self) -> Bool:
        """Returns True if this pointer represents a higher than or equal
           address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a higher than or equal address and
            False otherwise.
        """
        return Int(self) >= Int(rhs)

    @always_inline("builtin")
    fn __merge_with__[
        other_type: type_of(
            LegacyUnsafePointer[
                Self.type,
                address_space = Self.address_space,
                origin=_,
            ]
        ),
    ](self) -> LegacyUnsafePointer[
        type = Self.type,
        origin = origin_of(Self.origin, other_type.origin),
        address_space = Self.address_space,
    ]:
        """Returns a pointer merged with the specified `other_type`.

        Parameters:
            other_type: The type of the pointer to merge with.

        Returns:
            A pointer merged with the specified `other_type`.
        """
        return self.address  # allow kgen.pointer to convert.

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __bool__(self) -> Bool:
        """Return true if the pointer is non-null.

        Returns:
            Whether the pointer is null.
        """
        return Int(self) != 0

    @always_inline
    fn __int__(self) -> Int:
        """Returns the pointer address as an integer.

        Returns:
          The address of the pointer as an Int.
        """
        return Int(mlir_value=__mlir_op.`pop.pointer_to_index`(self.address))

    @deprecated("Stringable is deprecated. Use Writable instead.")
    @no_inline
    fn __str__(self) -> String:
        """Gets a string representation of the pointer.

        Returns:
            The string representation of the pointer.
        """
        return hex(Int(self))

    @no_inline
    fn write_to(self, mut writer: Some[Writer]):
        """
        Formats this pointer address to the provided Writer.

        Args:
            writer: The object to write to.
        """
        self.as_unsafe_pointer().write_to(writer)

    @no_inline
    fn write_repr_to(self, mut writer: Some[Writer]):
        """Write the string representation of the LegacyUnsafePointer.

        Args:
            writer: The object to write to.
        """
        FormatStruct(writer, "LegacyUnsafePointer").params(
            Named("mut", Self.mut),
            TypeNames[Self.type](),
            Named("address_space", Self.address_space),
        ).fields(self)

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    # Implementation of `DevicePassable`
    comptime device_type: AnyType = Self
    """DeviceBuffer dtypes are remapped to UnsafePointer when passed to accelerator devices."""

    @staticmethod
    fn _is_convertible_to_device_type[T: AnyType]() -> Bool:
        comptime if Self.mut:
            return Variadic.contains[
                T,
                Variadic.types[
                    T=AnyType,
                    Self,
                    Self._OriginCastType[MutAnyOrigin],
                    Self._OriginCastType[MutExternalOrigin],
                    Self._OriginCastType[ImmutAnyOrigin],
                    Self._OriginCastType[ImmutExternalOrigin],
                    Self._UnsafePointerType,
                    Self._UnsafePointerType._OriginCastType[MutAnyOrigin],
                    Self._UnsafePointerType._OriginCastType[MutExternalOrigin],
                    Self._UnsafePointerType._OriginCastType[ImmutAnyOrigin],
                    Self._UnsafePointerType._OriginCastType[
                        ImmutExternalOrigin
                    ],
                ],
            ]
        else:
            return Variadic.contains[
                T,
                Variadic.types[
                    T=AnyType,
                    Self,
                    Self._OriginCastType[ImmutAnyOrigin],
                    Self._OriginCastType[ImmutExternalOrigin],
                    Self._UnsafePointerType,
                    Self._UnsafePointerType._OriginCastType[ImmutAnyOrigin],
                    Self._UnsafePointerType._OriginCastType[
                        ImmutExternalOrigin
                    ],
                ],
            ]

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        """Device dtype mapping from DeviceBuffer to the device's UnsafePointer.
        """
        # TODO: Allow the low-level DeviceContext implementation to intercept
        # these translations.
        target.bitcast[Self.device_type]()[] = self.address

    @staticmethod
    fn get_type_name() -> String:
        """
        Gets this type name, for use in error messages when handing arguments
        to kernels.
        TODO: This will go away soon, when we get better error messages for
        kernel calls.

        Returns:
            This name of the type.
        """
        return String(
            "LegacyUnsafePointer[",
            get_type_name[Self.type](),
            ", address_space=",
            Self.address_space,
            ", mut=",
            Self.mut,
            "]",
        )

    @always_inline("builtin")
    fn as_unsafe_pointer(
        self,
        out result: UnsafePointer[
            Self.type, Self.origin, address_space = Self.address_space
        ],
    ):
        """Explicitly cast this legacy pointer to an `UnsafePointer`.

        Returns:
            An `UnsafePointer` with the same type, mutability, origin, and
            address space as the original pointer.
        """
        result = type_of(result)(self.address)

    @always_inline("nodebug")
    fn swap_pointees[
        U: Movable
    ](
        self: LegacyUnsafePointer[mut=True, U, origin=_],
        other: LegacyUnsafePointer[mut=True, U, origin=_],
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

        comptime if U.__move_ctor_is_trivial:
            # If `moveinit` is trivial, we can avoid the branch introduced from
            # checking if the pointers are equal by using temporary stack
            # values.
            #
            # Since `lhs` may overlap with `rhs` we need two temporary stack
            # values since we cannot call `memcpy` with the potentially
            # overlapping pointers.
            #
            # Even if they are not overlapping, this also produces better llvm
            # code with only 2 loads and 2 stores. Whereas with only 1 temporary
            # and a memcpy between the pointers it produces 3 load and 3 stores.

            var self_tmp = UnsafeMaybeUninit[U]()
            var other_tmp = UnsafeMaybeUninit[U]()
            memcpy(dest=self_tmp.unsafe_ptr(), src=UnsafePointer(self), count=1)
            memcpy(
                dest=other_tmp.unsafe_ptr(), src=UnsafePointer(other), count=1
            )

            memcpy(dest=self, src=other_tmp.unsafe_ptr(), count=1)
            memcpy(dest=other, src=self_tmp.unsafe_ptr(), count=1)
        else:
            # If `moveinit` is NOT trivial, we need to check if the pointers are
            # the same to avoid undefined behavior when moving from rhs to lhs.
            if self == other:
                return
            var tmp = self.take_pointee()
            self.init_pointee_move_from(other)
            other.init_pointee_move(tmp^)

    @always_inline("nodebug")
    fn as_noalias_ptr(self) -> Self:
        """Cast the pointer to a new pointer that is known not to locally alias
        any other pointer. In other words, the pointer transitively does not
        alias any other memory value declared in the local function context.

        This information is relayed to the optimizer. If the pointer does
        locally alias another memory value, the behaviour is undefined.

        Returns:
            A noalias pointer.
        """
        return __mlir_op.`pop.noalias_pointer_cast`(self.address)

    @always_inline("nodebug")
    fn load[
        dtype: DType,
        //,
        width: Int = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
        invariant: Bool = _default_invariant[Self.mut](),
    ](self: LegacyUnsafePointer[Scalar[dtype], ...]) -> SIMD[dtype, width]:
        """Loads `width` elements from the value the pointer points to.

        Use `alignment` to specify minimal known alignment in bytes; pass a
        smaller value (such as 1) if loading from packed/unaligned memory. The
        `volatile`/`invariant` flags control reordering and common-subexpression
        elimination semantics for special cases.

        Example:

        ```mojo
        var p = LegacyUnsafePointer[Int32].alloc(8)
        p.store(0, SIMD[DType.int32, 4](1, 2, 3, 4))
        var v = p.load[width=4]()
        print(v)  # => [1, 2, 3, 4]
        p.free()
        ```

        Constraints:
            The width and alignment must be positive integer values.

        Parameters:
            dtype: The data type of the SIMD vector.
            width: The number of elements to load.
            alignment: The minimal alignment (bytes) of the address.
            volatile: Whether the operation is volatile.
            invariant: Whether the load is from invariant memory.

        Returns:
            The loaded SIMD vector.
        """
        _simd_construction_checks[dtype, width]()
        comptime assert (
            alignment > 0
        ), "alignment must be a positive integer value"
        comptime assert (
            not volatile or volatile ^ invariant
        ), "both volatile and invariant cannot be set at the same time"

        comptime if is_nvidia_gpu() and size_of[
            dtype
        ]() == 1 and alignment == 1:
            # LLVM lowering to PTX incorrectly vectorizes loads for 1-byte types
            # regardless of the alignment that is passed. This causes issues if
            # this method is called on an unaligned pointer.
            # TODO #37823 We can make this smarter when we add an `aligned`
            # trait to the pointer class.
            var v = SIMD[dtype, width]()

            # intentionally don't unroll, otherwise the compiler vectorizes
            for i in range(width):
                v[i] = __mlir_op.`pop.load`[
                    alignment = alignment._mlir_value,
                    isVolatile = volatile._mlir_value,
                    isInvariant = invariant._mlir_value,
                ]((self + i).address)
            return v
        elif dtype == DType.bool and width > 1:
            # Bool (i1) is sub-byte, so a vector load of SIMD[bool, N]
            # packs bits. Load as uint8 and convert to bool so each
            # element occupies its own byte boundary.
            return rebind[SIMD[dtype, width]](
                self.bitcast[Scalar[DType.uint8]]()
                .load[
                    width=width,
                    alignment=alignment,
                    volatile=volatile,
                    invariant=invariant,
                ]()
                .cast[DType.bool]()
            )

        var address = self.bitcast[SIMD[dtype, width]]().address

        return __mlir_op.`pop.load`[
            alignment = alignment._mlir_value,
            isVolatile = volatile._mlir_value,
            isInvariant = invariant._mlir_value,
        ](address)

    @always_inline("nodebug")
    fn load[
        dtype: DType,
        //,
        width: Int = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
        invariant: Bool = _default_invariant[Self.mut](),
    ](self: LegacyUnsafePointer[Scalar[dtype], ...], offset: Scalar) -> SIMD[
        dtype, width
    ]:
        """Loads the value the pointer points to with the given offset.

        Constraints:
            The width and alignment must be positive integer values.
            The offset must be integer.

        Parameters:
            dtype: The data type of SIMD vector elements.
            width: The size of the SIMD vector.
            alignment: The minimal alignment of the address.
            volatile: Whether the operation is volatile or not.
            invariant: Whether the memory is load invariant.

        Args:
            offset: The offset to load from.

        Returns:
            The loaded value.
        """
        comptime assert offset.dtype.is_integral(), "offset must be integer"
        return (self + Int(offset)).load[
            width=width,
            alignment=alignment,
            volatile=volatile,
            invariant=invariant,
        ]()

    @always_inline("nodebug")
    fn load[
        I: Indexer,
        dtype: DType,
        //,
        width: Int = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
        invariant: Bool = _default_invariant[Self.mut](),
    ](self: LegacyUnsafePointer[Scalar[dtype], ...], offset: I) -> SIMD[
        dtype, width
    ]:
        """Loads the value the pointer points to with the given offset.

        Constraints:
            The width and alignment must be positive integer values.

        Parameters:
            I: A type that can be used as an index.
            dtype: The data type of SIMD vector elements.
            width: The size of the SIMD vector.
            alignment: The minimal alignment of the address.
            volatile: Whether the operation is volatile or not.
            invariant: Whether the memory is load invariant.

        Args:
            offset: The offset to load from.

        Returns:
            The loaded value.
        """
        return (self + offset).load[
            width=width,
            alignment=alignment,
            volatile=volatile,
            invariant=invariant,
        ]()

    @always_inline("nodebug")
    fn store[
        I: Indexer,
        dtype: DType,
        //,
        width: Int = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
    ](
        self: LegacyUnsafePointer[mut=True, Scalar[dtype], ...],
        offset: I,
        val: SIMD[dtype, width],
    ):
        """Stores a single element value at the given offset.

        Constraints:
            The width and alignment must be positive integer values.
            The offset must be integer.

        Parameters:
            I: A type that can be used as an index.
            dtype: The data type of SIMD vector elements.
            width: The size of the SIMD vector.
            alignment: The minimal alignment of the address.
            volatile: Whether the operation is volatile or not.

        Args:
            offset: The offset to store to.
            val: The value to store.
        """
        (self + offset).store[alignment=alignment, volatile=volatile](val)

    @always_inline("nodebug")
    fn store[
        dtype: DType,
        offset_type: DType,
        //,
        width: Int = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
    ](
        self: LegacyUnsafePointer[mut=True, Scalar[dtype], ...],
        offset: Scalar[offset_type],
        val: SIMD[dtype, width],
    ):
        """Stores a single element value at the given offset.

        Constraints:
            The width and alignment must be positive integer values.

        Parameters:
            dtype: The data type of SIMD vector elements.
            offset_type: The data type of the offset value.
            width: The size of the SIMD vector.
            alignment: The minimal alignment of the address.
            volatile: Whether the operation is volatile or not.

        Args:
            offset: The offset to store to.
            val: The value to store.
        """
        comptime assert offset_type.is_integral(), "offset must be integer"
        (self + Int(offset))._store[alignment=alignment, volatile=volatile](val)

    @always_inline("nodebug")
    fn store[
        dtype: DType,
        //,
        width: Int = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
    ](
        self: LegacyUnsafePointer[mut=True, Scalar[dtype], ...],
        val: SIMD[dtype, width],
    ):
        """Stores a single element value `val` at element offset 0.

        Specify `alignment` when writing to packed/unaligned memory. Requires a
        mutable pointer. For writing at an element offset, use the overloads
        that accept an index or scalar offset.

        Example:

        ```mojo
        var p = LegacyUnsafePointer[Float32].alloc(4)
        var vec = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
        p.store(vec)
        var out = p.load[width=4]()
        print(out)  # => [1.0, 2.0, 3.0, 4.0]
        p.free()
        ```

        Constraints:
            The width and alignment must be positive integer values.

        Parameters:
            dtype: The data type of SIMD vector elements.
            width: The number of elements to store.
            alignment: The minimal alignment (bytes) of the address.
            volatile: Whether the operation is volatile.

        Args:
            val: The SIMD value to store.
        """
        self._store[alignment=alignment, volatile=volatile](val)

    @always_inline("nodebug")
    fn _store[
        dtype: DType,
        width: Int,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
    ](
        self: LegacyUnsafePointer[mut=True, Scalar[dtype], ...],
        val: SIMD[dtype, width],
    ):
        comptime assert width > 0, "width must be a positive integer value"
        comptime assert (
            alignment > 0
        ), "alignment must be a positive integer value"

        comptime if dtype == DType.bool and width > 1:
            # Bool (i1) is sub-byte, so a vector store of SIMD[bool, N]
            # packs bits. Cast to uint8 and store so each element
            # occupies its own byte boundary.
            self.bitcast[Scalar[DType.uint8]]()._store[
                alignment=alignment, volatile=volatile
            ](val.cast[DType.uint8]())
        else:
            __mlir_op.`pop.store`[
                alignment = alignment._mlir_value,
                isVolatile = volatile._mlir_value,
            ](val, self.bitcast[SIMD[dtype, width]]().address)

    @always_inline("nodebug")
    fn strided_load[
        dtype: DType, T: Intable, //, width: Int
    ](self: LegacyUnsafePointer[Scalar[dtype], ...], stride: T) -> SIMD[
        dtype, width
    ]:
        """Performs a strided load of the SIMD vector.

        Parameters:
            dtype: DType of returned SIMD value.
            T: The Intable type of the stride.
            width: The SIMD width.

        Args:
            stride: The stride between loads.

        Returns:
            A vector which is stride loaded.
        """
        return strided_load(
            UnsafePointer(self), Int(stride), SIMD[DType.bool, width](fill=True)
        )

    @always_inline("nodebug")
    fn strided_store[
        dtype: DType,
        T: Intable,
        //,
        width: Int = 1,
    ](
        self: LegacyUnsafePointer[mut=True, Scalar[dtype], ...],
        val: SIMD[dtype, width],
        stride: T,
    ):
        """Performs a strided store of the SIMD vector.

        Parameters:
            dtype: DType of `val`, the SIMD value to store.
            T: The Intable type of the stride.
            width: The SIMD width.

        Args:
            val: The SIMD value to store.
            stride: The stride between stores.
        """
        strided_store(
            val, self, Int(stride), SIMD[DType.bool, width](fill=True)
        )

    @always_inline("nodebug")
    fn gather[
        dtype: DType,
        //,
        *,
        width: Int = 1,
        alignment: Int = align_of[dtype](),
    ](
        self: LegacyUnsafePointer[Scalar[dtype], ...],
        offset: SIMD[_, width],
        mask: SIMD[DType.bool, width] = SIMD[DType.bool, width](fill=True),
        default: SIMD[dtype, width] = 0,
    ) -> SIMD[dtype, width]:
        """Gathers a SIMD vector from offsets of the current pointer.

        This method loads from memory addresses calculated by appropriately
        shifting the current pointer according to the `offset` SIMD vector,
        or takes from the `default` SIMD vector, depending on the values of
        the `mask` SIMD vector.

        If a mask element is `True`, the respective result element is given
        by the current pointer and the `offset` SIMD vector; otherwise, the
        result element is taken from the `default` SIMD vector.

        Constraints:
            The offset type must be an integral type.
            The alignment must be a power of two integer value.

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
        comptime assert (
            offset.dtype.is_integral()
        ), "offset type must be an integral type"
        comptime assert (
            alignment.is_power_of_two()
        ), "alignment must be a power of two integer value"

        var base = offset.cast[DType.int]().fma(
            SIMD[DType.int, width](size_of[dtype]()),
            SIMD[DType.int, width](Int(self)),
        )
        return gather[alignment=alignment](base, mask, default)

    @always_inline("nodebug")
    fn scatter[
        dtype: DType,
        //,
        *,
        width: Int = 1,
        alignment: Int = align_of[dtype](),
    ](
        self: LegacyUnsafePointer[mut=True, Scalar[dtype], ...],
        offset: SIMD[_, width],
        val: SIMD[dtype, width],
        mask: SIMD[DType.bool, width] = SIMD[DType.bool, width](fill=True),
    ):
        """Scatters a SIMD vector into offsets of the current pointer.

        This method stores at memory addresses calculated by appropriately
        shifting the current pointer according to the `offset` SIMD vector,
        depending on the values of the `mask` SIMD vector.

        If a mask element is `True`, the respective element in the `val` SIMD
        vector is stored at the memory address defined by the current pointer
        and the `offset` SIMD vector; otherwise, no action is taken for that
        element in `val`.

        If the same offset is targeted multiple times, the values are stored
        in the order they appear in the `val` SIMD vector, from the first to
        the last element.

        Constraints:
            The offset type must be an integral type.
            The alignment must be a power of two integer value.

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
        comptime assert (
            offset.dtype.is_integral()
        ), "offset type must be an integral type"
        comptime assert (
            alignment.is_power_of_two()
        ), "alignment must be a power of two integer value"

        var base = offset.cast[DType.int]().fma(
            SIMD[DType.int, width](size_of[dtype]()),
            SIMD[DType.int, width](Int(self)),
        )
        scatter[alignment=alignment](val, base, mask)

    @always_inline
    fn free(
        self: LegacyUnsafePointer[
            mut=True, _, address_space = AddressSpace.GENERIC, ...
        ]
    ):
        """Free the memory referenced by the pointer."""
        _free(self)

    @always_inline("builtin")
    fn bitcast[
        T: AnyType = Self.type
    ](self) -> LegacyUnsafePointer[
        T,
        address_space = Self.address_space,
        origin = Self.origin,
    ]:
        """Bitcasts a LegacyUnsafePointer to a different type.

        Parameters:
            T: The target type.

        Returns:
            A new LegacyUnsafePointer object with the specified type and the same address,
            as the original LegacyUnsafePointer.
        """
        return __mlir_op.`pop.pointer.bitcast`[
            _type = LegacyUnsafePointer[
                T,
                address_space = Self.address_space,
            ]._mlir_type,
        ](self.address)

    comptime _OriginCastType[
        target_mut: Bool, //, target_origin: Origin[mut=target_mut]
    ] = LegacyUnsafePointer[
        Self.type,
        address_space = Self.address_space,
        origin=target_origin,
    ]

    @always_inline("nodebug")
    fn mut_cast[
        target_mut: Bool
    ](self) -> Self._OriginCastType[
        Origin[mut=target_mut](unsafe_mut_cast=Self.origin)
    ]:
        """Changes the mutability of a pointer.

        This is a safe way to change the mutability of a pointer with an
        unbounded mutability. This function will emit a compile time error if
        you try to cast an immutable pointer to mutable.

        Parameters:
            target_mut: Mutability of the destination pointer.

        Returns:
            A pointer with the same type, origin and address space as the
            original pointer, but with the newly specified mutability.
        """
        comptime assert (
            target_mut == False or target_mut == Self.mut
        ), "Cannot safely cast an immutable pointer to mutable"
        return self.unsafe_mut_cast[target_mut]()

    @always_inline("builtin")
    fn unsafe_mut_cast[
        target_mut: Bool
    ](self) -> Self._OriginCastType[
        Origin[mut=target_mut](unsafe_mut_cast=Self.origin)
    ]:
        """Changes the mutability of a pointer.

        Parameters:
            target_mut: Mutability of the destination pointer.

        Returns:
            A pointer with the same type, origin and address space as the
            original pointer, but with the newly specified mutability.

        If you are unconditionally casting the mutability to `False`, use
        `as_immutable` instead.
        If you are casting to mutable or a parameterized mutability, prefer
        using the safe `mut_cast` method instead.

        Safety:
            Casting the mutability of a pointer is inherently very unsafe.
            Improper usage can lead to undefined behavior. Consider restricting
            types to their proper mutability at the function signature level.
            For example, taking an `LegacyUnsafePointer[T, mut=True, ...]` as an
            argument over an unbound `LegacyUnsafePointer[T, ...]` is preferred.
        """
        return __mlir_op.`pop.pointer.bitcast`[
            _type = Self._OriginCastType[
                Origin[mut=target_mut](unsafe_mut_cast=Self.origin)
            ]._mlir_type,
        ](self.address)

    @always_inline("builtin")
    fn unsafe_origin_cast[
        target_origin: Origin[mut = Self.mut]
    ](self) -> Self._OriginCastType[target_origin]:
        """Changes the origin of a pointer.

        Parameters:
            target_origin: Origin of the destination pointer.

        Returns:
            A pointer with the same type, mutability and address space as the
            original pointer, but with the newly specified origin.

        If you are unconditionally casting the origin to an `AnyOrigin`, use
        `as_any_origin` instead.

        Safety:
            Casting the origin of a pointer is inherently very unsafe.
            Improper usage can lead to undefined behavior or unexpected variable
            destruction. Considering parameterizing the origin at the function
            level to avoid unnecessary casts.
        """
        return __mlir_op.`pop.pointer.bitcast`[
            _type = Self._OriginCastType[target_origin]._mlir_type,
        ](self.address)

    @always_inline("builtin")
    fn as_immutable(
        self,
    ) -> Self._OriginCastType[ImmutOrigin(Self.origin)]:
        """Changes the mutability of a pointer to immutable.

        Unlike `unsafe_mut_cast`, this function is always safe to use as casting
        from (im)mutable to immutable is always safe.

        Returns:
            A pointer with the mutability set to immutable.
        """
        return self.unsafe_mut_cast[False]()

    @always_inline("builtin")
    fn as_any_origin(
        self,
    ) -> LegacyUnsafePointer[
        Self.type,
        address_space = Self.address_space,
        origin = AnyOrigin[mut = Self.mut],
    ]:
        """Casts the origin of a pointer to `AnyOrigin`.

        Returns:
            A pointer with the origin set to `AnyOrigin`.

        It is usually preferred to maintain concrete origin values instead of
        using `AnyOrigin`. However, if it is needed, keep in mind that
        `AnyOrigin` can alias any memory value, so Mojo's ASAP
        destruction will not apply during the lifetime of the pointer.
        """
        # TODO: compiler error if using self.unsafe_origin_cast
        return __mlir_op.`pop.pointer.bitcast`[
            _type = LegacyUnsafePointer[
                Self.type,
                address_space = Self.address_space,
                origin = AnyOrigin[mut = Self.mut],
            ]._mlir_type,
        ](self.address)

    @always_inline("builtin")
    fn address_space_cast[
        target_address_space: AddressSpace = Self.address_space,
    ](self) -> LegacyUnsafePointer[
        Self.type,
        address_space=target_address_space,
        origin = Self.origin,
    ]:
        """Casts an LegacyUnsafePointer to a different address space.

        Parameters:
            target_address_space: The address space of the result.

        Returns:
            A new LegacyUnsafePointer object with the same type and the same address,
            as the original LegacyUnsafePointer and the new address space.
        """
        return __mlir_op.`pop.pointer.bitcast`[
            _type = LegacyUnsafePointer[
                Self.type,
                address_space=target_address_space,
            ]._mlir_type,
        ](self.address)

    @always_inline
    fn destroy_pointee[
        T: ImplicitlyDestructible, //
    ](
        self: LegacyUnsafePointer[
            mut=True, T, address_space = AddressSpace.GENERIC, ...
        ]
    ):
        """Destroy the pointed-to value.

        The pointer must not be null, and the pointer memory location is assumed
        to contain a valid initialized instance of `type`.  This is equivalent to
        `_ = self.take_pointee()` but doesn't require `Movable` and is
        more efficient because it doesn't invoke the move constructor.

        Parameters:
            T: Pointee type that can be destroyed implicitly (without
              deinitializer arguments).

        """
        _ = __get_address_as_owned_value(self.address)

    @always_inline
    fn take_pointee[
        T: Movable,
        //,
    ](
        self: LegacyUnsafePointer[
            mut=True, T, address_space = AddressSpace.GENERIC, ...
        ]
    ) -> T:
        """Move the value at the pointer out, leaving it uninitialized.

        The pointer must not be null, and the pointer memory location is assumed
        to contain a valid initialized instance of `T`.

        This performs a _consuming_ move, ending the origin of the value stored
        in this pointer memory location. Subsequent reads of this pointer are
        not valid. If a new valid value is stored using `init_pointee_move()`, then
        reading from this pointer becomes valid again.

        Parameters:
            T: The type the pointer points to, which must be `Movable`.

        Returns:
            The value at the pointer.
        """
        return __get_address_as_owned_value(self.address)

    # TODO: Allow overloading on more specific traits
    @always_inline
    fn init_pointee_move[
        T: Movable,
        //,
    ](
        self: LegacyUnsafePointer[
            mut=True, T, address_space = AddressSpace.GENERIC, ...
        ],
        var value: T,
    ):
        """Emplace a new value into the pointer location, moving from `value`.

        The pointer memory location is assumed to contain uninitialized data,
        and consequently the current contents of this pointer are not destructed
        before writing `value`. Similarly, ownership of `value` is logically
        transferred into the pointer location.

        When compared to `init_pointee_copy`, this avoids an extra copy on
        the caller side when the value is an `owned` rvalue.

        Parameters:
            T: The type the pointer points to, which must be `Movable`.

        Args:
            value: The value to emplace.
        """
        __get_address_as_uninit_lvalue(self.address) = value^

    @always_inline
    fn init_pointee_copy[
        T: Copyable,
        //,
    ](
        self: LegacyUnsafePointer[
            mut=True, T, address_space = AddressSpace.GENERIC, ...
        ],
        value: T,
    ):
        """Emplace a copy of `value` into the pointer location.

        The pointer memory location is assumed to contain uninitialized data,
        and consequently the current contents of this pointer are not destructed
        before writing `value`. Similarly, ownership of `value` is logically
        transferred into the pointer location.

        When compared to `init_pointee_move`, this avoids an extra move on
        the callee side when the value must be copied.

        Parameters:
            T: The type the pointer points to, which must be `Copyable`.

        Args:
            value: The value to emplace.
        """
        __get_address_as_uninit_lvalue(self.address) = value.copy()

    @always_inline
    fn init_pointee_move_from[
        T: Movable,
        //,
    ](
        self: LegacyUnsafePointer[
            mut=True, T, address_space = AddressSpace.GENERIC, ...
        ],
        src: LegacyUnsafePointer[
            mut=True, T, address_space = AddressSpace.GENERIC, ...
        ],
    ):
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

        This transfers the value out of `src` and into `self` using at most one
        move constructor call.

        ### Example

        ```mojo
        var a_ptr = LegacyUnsafePointer.alloc[String](1)
        var b_ptr = LegacyUnsafePointer.alloc[String](2)

        # Initialize A pointee
        a_ptr.init_pointee_move("foo")

        # Perform the move
        b_ptr.init_pointee_move_from(a_ptr)

        # Clean up
        b_ptr.destroy_pointee()
        a_ptr.free()
        b_ptr.free()
        ```

        ### Safety

        * `self` and `src` must be non-null
        * `src` must contain a valid, initialized instance of `T`
        * The pointee contents of `self` should be uninitialized. If `self` was
          previously written with a valid value, that value will be be
          overwritten and its destructor will NOT be run.

        Parameters:
            T: The type the pointer points to, which must be `Movable`.

        Args:
            src: Source pointer that the value will be moved from.
        """
        __get_address_as_uninit_lvalue(
            self.address
        ) = __get_address_as_owned_value(src.address)


comptime LegacyOpaquePointer = LegacyUnsafePointer[
    NoneType, origin=MutAnyOrigin
]
"""An opaque pointer, equivalent to the C `void*` type.

Parameters:
    origin: The origin of the pointer.
"""
