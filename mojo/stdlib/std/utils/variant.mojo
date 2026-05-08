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
"""Defines a Variant type."""

from std.builtin.rebind import downcast
from std.format._utils import (
    FormatStruct,
    TypeNames,
)
from std.memory import UnsafeMaybeUninit
from std.hashlib.hasher import Hasher
from std.reflection import call_location
from std.reflection.traits import (
    AllCopyable,
    AllEquatable,
    AllHashable,
    AllImplicitlyCopyable,
    AllRegisterPassable,
    AllWritable,
)
from ._nicheable import (
    UnsafeNicheable,
    NicheIndex,
    NicheStorageTraits,
    UnsafeCustomNicheStorage,
)
from std.os import abort
from std.sys import align_of, size_of
from std.sys.intrinsics import _type_is_eq
from std.utils.type_functions import ConditionalType

# ===----------------------------------------------------------------------=== #
# Variant Storages
# ===----------------------------------------------------------------------=== #

comptime _InvalidTypeIndex: Int = -1


@always_inline
def _get_type_index[T: AnyType, *Ts: AnyType]() -> Int:
    comptime for i in range(Ts.size):
        comptime if _type_is_eq[Ts[i], T]():
            return i
    return _InvalidTypeIndex


trait _VariantStorage(Copyable, ImplicitlyDestructible):
    """Internal storage backend for `Variant`.

    This trait abstracts over the two concrete storage strategies:

    - `_DefaultVariantStorage`: general discriminated-union storage backed by
      an MLIR `kgen.variant` allocation with an explicit integer discriminant.
    - `_NichedOptionalStorage`: niche-optimized storage for two-type variants
      where one type is `UnsafeNicheable` and the other is zero-sized; encodes
      the active type in an invalid bit pattern rather than a separate tag byte.
    """

    def __init__[U: Movable](out self, var value: U):
        """Initialize storage with a value of type `U`."""
        ...

    def take[U: Movable](deinit self) -> U:
        """Consume this storage and return the held value as type `U`."""
        return self.unsafe_ptr[U]().take_pointee()

    def isa[U: AnyType](self) -> Bool:
        """Return `True` if the currently active type is `U`."""
        ...

    def unsafe_ptr[U: AnyType](ref self) -> UnsafePointer[U, origin_of(self)]:
        """Return a raw pointer to the stored data interpreted as type `U`.

        Safety: the caller must ensure `U` matches the active type."""
        ...


trait _NicheStorage(Defaultable, ImplicitlyCopyable, ImplicitlyDestructible):
    """Internal abstraction over niche backing storage backends."""

    def as_uninit[
        T: AnyType
    ](ref self) -> UnsafePointer[UnsafeMaybeUninit[T], origin_of(self)]:
        ...


struct _DefaultNicheStorage[T: AnyType](Defaultable, _NicheStorage):
    """Default niche backing: stores the value in `UnsafeMaybeUninit[T]`
    (lowers to `pop.array<1, T>`)."""

    var _memory: UnsafeMaybeUninit[Self.T]

    @always_inline
    def __init__(out self):
        self._memory = {}

    @always_inline
    def as_uninit[
        U: AnyType
    ](ref self) -> UnsafePointer[UnsafeMaybeUninit[U], origin_of(self)]:
        comptime assert _type_is_eq[Self.T, U]()
        return (
            UnsafePointer(to=self._memory)
            .bitcast[UnsafeMaybeUninit[U]]()
            .unsafe_origin_cast[origin_of(self)]()
        )


struct _CustomNicheStorage[Storage: UnsafeCustomNicheStorage](
    Defaultable, _NicheStorage
):
    """Niche backing that delegates to the user-provided `Storage` type,
    allowing the nicheable type to control what MLIR type the storage lowers
    to."""

    var _memory: Self.Storage.NicheStorage

    @always_inline
    def __init__(out self):
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))

    @always_inline
    def as_uninit[
        T: AnyType
    ](ref self) -> UnsafePointer[UnsafeMaybeUninit[T], origin_of(self)]:
        comptime assert (
            size_of[Self.Storage.NicheStorage]()
            == size_of[UnsafeMaybeUninit[T]]()
        ), "Custom storage must be the same size as Self"
        comptime assert (
            align_of[Self.Storage.NicheStorage]()
            == align_of[UnsafeMaybeUninit[T]]()
        ), "Custom storage must have the the same alignment as Self"
        return (
            UnsafePointer(to=self._memory)
            .bitcast[UnsafeMaybeUninit[T]]()
            .unsafe_origin_cast[origin_of(self)]()
        )


comptime _NicheStorageFor[T: AnyType] = ConditionalType[
    Trait=_NicheStorage,
    If=conforms_to(T, UnsafeCustomNicheStorage),
    Then=_CustomNicheStorage[downcast[T, UnsafeCustomNicheStorage]],
    Else=_DefaultNicheStorage[T],
]


struct _NichedOptionalStorage[
    T: UnsafeNicheable, EmptyType: TrivialRegisterPassable
](
    Copyable,
    RegisterPassable where conforms_to(T, RegisterPassable),
    _VariantStorage,
):
    """Optimized storage for two-type variants where one type is `UnsafeNicheable`
    and the other is zero-sized & `TrivialRegisterPassable` (e.g. `NoneType`).

    Instead of storing a discriminant tag, the niche of `T` (an invalid bit
    pattern) is repurposed to encode the "empty" state, eliminating the extra
    byte of overhead that `_DefaultVariantStorage` would require."""

    comptime __del__is_trivial = _all_trivial_del[Self.T]()
    comptime __copy_ctor_is_trivial = _all_trivial_copyinit[Self.T]()
    comptime __move_ctor_is_trivial = _all_trivial_moveinit[Self.T]()

    var _memory: _NicheStorageFor[Self.T]

    @staticmethod
    def _check[U: AnyType]():
        comptime assert (
            _type_is_eq[U, Self.T]() or _type_is_eq[U, Self.EmptyType]()
        ), "unexpected type"

    @always_inline
    def __init__(out self):
        comptime assert (
            Self.T.niche_count() > 0
        ), "UnsafeNicheable must specify at least 1 invalid bit pattern"
        self._memory = {}
        Self.T.write_niche[index=0](self._memory.as_uninit[Self.T]())

    @always_inline
    def __init__[U: Movable](out self, var value: U):
        Self._check[U]()
        comptime if _type_is_eq[U, Self.T]():
            self._memory = {}
            self._memory.as_uninit[U]()[].init_from(value^)
        else:
            # This is the empty "none" type.
            comptime assert conforms_to(U, TrivialRegisterPassable)
            _ = rebind_var[downcast[U, TrivialRegisterPassable]](value^)
            self = Self()

    @always_inline
    def __init__(out self, *, deinit take: Self):
        comptime assert conforms_to(Self.T, Movable)
        if take.isa[Self.T]():
            self = Self(
                take.unsafe_ptr[downcast[Self.T, Movable]]().take_pointee()
            )
        else:
            self = Self()

    @always_inline
    def __init__(out self, *, copy: Self):
        comptime assert conforms_to(Self.T, Copyable)
        if copy.isa[Self.T]():
            self = Self(
                trait_downcast[Copyable](copy.unsafe_ptr[Self.T]()[]).copy()
            )
        else:
            self = Self()

    @always_inline
    def __del__(deinit self):
        comptime assert conforms_to(Self.T, ImplicitlyDestructible)
        if self.isa[Self.T]():
            rebind[UnsafeMaybeUninit[downcast[Self.T, ImplicitlyDestructible]]](
                self._memory.as_uninit[Self.T]()[]
            ).unsafe_assume_init_destroy()

    @always_inline
    def isa[U: AnyType](self) -> Bool:
        Self._check[U]()
        var niche = Self.T.classify_niche(self._memory.as_uninit[Self.T]())
        var is_some = niche == NicheIndex.NotANiche
        comptime if _type_is_eq[U, Self.T]():
            return is_some
        else:
            return not is_some

    @always_inline
    def unsafe_ptr[U: AnyType](ref self) -> UnsafePointer[U, origin_of(self)]:
        Self._check[U]()
        return (
            self._memory.as_uninit[U]()
            .bitcast[U]()
            .unsafe_origin_cast[origin_of(self)]()
        )


struct _DefaultVariantStorage[*Ts: AnyType](
    Copyable,
    RegisterPassable where AllRegisterPassable[*Ts],
    _VariantStorage,
):
    """General-purpose discriminated-union storage for `Variant`.

    Stores all possible types in a single MLIR `kgen.variant` allocation and
    tracks the active type via an integer discriminant. Used whenever the
    variant types do not qualify for the niche-optimized path."""

    comptime __del__is_trivial = _all_trivial_del[*Self.Ts]()
    comptime __copy_ctor_is_trivial = _all_trivial_copyinit[*Self.Ts]()
    comptime __move_ctor_is_trivial = _all_trivial_moveinit[*Self.Ts]()

    comptime _mlir_type = __mlir_type[
        `!kgen.variant<[rebind(:`,
        type_of(Self.Ts.values),
        ` `,
        Self.Ts.values,
        `)]>`,
    ]
    var _impl: Self._mlir_type

    @always_inline
    def __init__(out self, *, unsafe_uninitialized: ()):
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))

    @always_inline
    def __init__[T: Movable](out self, var value: T):
        self = Self(unsafe_uninitialized=())
        self.get_discriminant() = UInt8(_get_type_index[T, *Self.Ts]())
        self.unsafe_ptr[T]().init_pointee_move(value^)

    @always_inline
    def __init__(out self, *, copy: Self):
        self = Self(unsafe_uninitialized=())
        self.get_discriminant() = copy.get_discriminant()

        comptime for i in range(Self.Ts.size):
            comptime TUnknown = Self.Ts[i]
            comptime assert conforms_to(TUnknown, Copyable)
            comptime T = downcast[TUnknown, Copyable]

            if self.get_discriminant() == UInt8(i):
                self.unsafe_ptr[T]().init_pointee_copy(copy.unsafe_ptr[T]()[])
                return

    @always_inline
    def __init__(out self, *, deinit take: Self):
        self = Self(unsafe_uninitialized=())
        self.get_discriminant() = take.get_discriminant()

        comptime for i in range(Self.Ts.size):
            comptime TUnknown = Self.Ts[i]
            comptime assert conforms_to(TUnknown, Movable)
            comptime T = downcast[TUnknown, Movable]

            if self.get_discriminant() == UInt8(i):
                self.unsafe_ptr[T]().init_pointee_move_from(
                    take.unsafe_ptr[T]()
                )
                return

    @always_inline
    def __del__(deinit self):
        comptime for i in range(Self.Ts.size):
            comptime TUnknown = Self.Ts[i]
            comptime assert conforms_to(TUnknown, ImplicitlyDestructible)
            comptime T = downcast[TUnknown, ImplicitlyDestructible]

            if self.get_discriminant() == UInt8(i):
                self.unsafe_ptr[T]().destroy_pointee()
                return

    @always_inline("nodebug")
    def get_discriminant(ref self) -> ref[self] UInt8:
        var discr_ptr = __mlir_op.`pop.variant.discr_gep`[
            _type=__mlir_type.`!kgen.pointer<scalar<ui8>>`
        ](UnsafePointer(to=self._impl).address)
        return UnsafePointer[_, origin_of(self)](discr_ptr).bitcast[UInt8]()[]

    @always_inline("nodebug")
    def isa[T: AnyType](self) -> Bool:
        comptime discriminant = UInt8(_get_type_index[T, *Self.Ts]())
        return self.get_discriminant() == discriminant

    @always_inline("nodebug")
    def unsafe_ptr[T: AnyType](ref self) -> UnsafePointer[T, origin_of(self)]:
        comptime idx = _get_type_index[T, *Self.Ts]()
        return __mlir_op.`pop.variant.bitcast`[
            _type=UnsafePointer[T, origin_of(self)]._mlir_type,
            index=idx._int_mlir_index(),
        ](UnsafePointer(to=self._impl).address)


# TODO(MOCO-3653): size_of[T]() == 0 does not work correctly in some cases when
# an `Optional` is used as a comptime parameter's field.
comptime _IsEmptyType[T: AnyType]: Bool = reflect[
    T
].field_count() == 0 and conforms_to(T, TrivialRegisterPassable)
"""True if `T` is a zero-sized, trivially passable type (i.e. carries no state,
like `NoneType`). Used to identify the "empty" arm of a niche-optimized variant."""

comptime _IsNicheablePair[T: AnyType, U: AnyType]: Bool = conforms_to(
    T, UnsafeNicheable
) and _IsEmptyType[U]
"""True if `T` is `UnsafeNicheable` and `U` is an empty type. Called twice with
swapped args by `_IsNicheEligible` to handle either ordering."""

comptime _IsNicheEligible[*Ts: AnyType]: Bool = (Ts.size == 2) and (
    _IsNicheablePair[Ts[0], Ts[1]] or _IsNicheablePair[Ts[1], Ts[0]]
)
"""True if `Ts` qualifies for niche-optimized storage: exactly two types
where one is `UnsafeNicheable` and the other is an empty type."""

comptime _NichedStorageFor[*Ts: AnyType] = ConditionalType[
    Trait=_VariantStorage,
    If=conforms_to(Ts[0], UnsafeNicheable),
    Then=_NichedOptionalStorage[
        downcast[Ts[0], UnsafeNicheable],
        downcast[Ts[1], TrivialRegisterPassable],
    ],
    Else=_NichedOptionalStorage[
        downcast[Ts[1], UnsafeNicheable],
        downcast[Ts[0], TrivialRegisterPassable],
    ],
]
"""Resolves to the concrete `_NichedOptionalStorage[T]` for the eligible type,
regardless of which position the `UnsafeNicheable` type occupies in `Ts`."""

comptime _VariantStorageFor[*Ts: AnyType] = ConditionalType[
    Trait=_VariantStorage,
    If=_IsNicheEligible[*Ts],
    Then=_NichedStorageFor[*Ts],
    Else=_DefaultVariantStorage[*Ts],
]
"""Selects the storage strategy for `Variant[*Ts]`: niche-optimized storage
when eligible, falling back to the general discriminant-tagged storage."""

# ===----------------------------------------------------------------------=== #
# Variant
# ===----------------------------------------------------------------------=== #


struct Variant[*Ts: Movable](
    Copyable where AllCopyable[*Ts],
    Equatable where AllEquatable[*Ts],
    Hashable where AllHashable[*Ts],
    # TODO(MOCO-3421): AllImplicitlyCopyable implies AllCopyable since
    # ImplicitlyCopyable refines Copyable, but the compiler can't infer
    # parent trait constraints from derived ones yet. Remove AllCopyable
    # from this where clause once that's fixed.
    ImplicitlyCopyable where AllImplicitlyCopyable[*Ts] and AllCopyable[*Ts],
    ImplicitlyDestructible,
    Movable,
    RegisterPassable where AllRegisterPassable[*Ts],
    Writable where AllWritable[*Ts],
):
    """A union that can hold a runtime-variant value from a set of predefined
    types.

    `Variant` is a discriminated union type, similar to `std::variant` in C++
    or `enum` in Rust. It can store exactly one value that can be any of the
    specified types, determined at runtime.

    The key feature is that the actual type stored in a `Variant` is determined
    at runtime, not compile time. This allows you to change what type a variant
    holds during program execution. Memory-wise, a variant only uses the space
    needed for the largest possible type plus a small discriminant field to
    track which type is currently active.

    Tips:

    - use `isa[T]()` to check what type a variant is
    - use `unsafe_take[T]()` to take a value from the variant
    - use `[T]` to get a value out of a variant
        - This currently does an extra copy/move until we have origins
        - It also temporarily requires the value to be mutable
    - use `set[T](var new_value: T)` to reset the variant to a new value
    - use `is_type_supported[T]` to check if the variant permits the type `T`

    **Note**: Currently, variant operations require the variant to be
    mutable (`mut`), even for read operations.

    Example:

    ```mojo
    from std.utils import Variant
    import std.random as random

    comptime IntOrString = Variant[Int, String]

    def to_string(mut x: IntOrString) -> String:
        if x.isa[String]():
            return x[String]
        return String(x[Int])

    var an_int = IntOrString(4)
    var a_string = IntOrString("I'm a string!")
    var who_knows = IntOrString(0)
    # Randomly change who_knows to a string
    random.seed()
    if random.random_ui64(0, 1):
        who_knows.set[String]("I'm also a string!")

    print(a_string[String])      # => I'm a string!
    print(an_int[Int])           # => 4
    print(to_string(who_knows))  # Either 0 or "I'm also a string!"

    if who_knows.isa[String]():
        print("It's a String!")
    ```

    Example usage for error handling:

    ```mojo
    comptime Result = Variant[String, Error]

    def process_data(data: String) -> Result:
        if data.byte_length() == 0:
            return Result(Error("Empty data"))
        return Result(String("Processed: ", data))

    var result = process_data("Hello")
    if result.isa[String]():
        print("Success:", result[String])
    else:
        print("Error:", result[Error])
    ```

    Example usage in a `List` to create a heterogeneous list:

    ```mojo
    comptime MixedType = Variant[Int, Float64, String, Bool]

    var mixed_list = List[MixedType]()
    mixed_list.append(MixedType(42))
    mixed_list.append(MixedType(3.14))
    mixed_list.append(MixedType("hello"))
    mixed_list.append(MixedType(True))

    for item in mixed_list:
        if item.isa[String]():
            print("String:", item[String])
        elif item.isa[Int]():
            print("Integer:", item[Int])
        elif item.isa[Float64]():
            print("Float:", item[Float64])
        elif item.isa[Bool]():
            print("Boolean:", item[Bool])
    ```

    ## Layout

    The layout of `Variant` is not guaranteed and may change at any time. The
    implementation may apply niche optimizations (for example, encoding the
    discriminant inside spare bits of one of the types in `Ts`) that alter the
    resulting layout. Do not rely on `size_of[Variant[...]]()` or
    `align_of[Variant[...]]()` being stable across language versions. The only
    guarantee is that the size and alignment will be at least as large as those
    of the largest type among `Ts`.

    If you need to inspect the current size or alignment, use `size_of` and
    `align_of`, but treat the results as non-stable implementation details.

    Parameters:
        Ts: The possible types that this variant can hold. Types that
            implement `Copyable` enable copy semantics for the variant.
    """

    comptime _Storage: _VariantStorage = _VariantStorageFor[*Self.Ts]

    comptime __del__is_trivial = Self._Storage.__del__is_trivial
    comptime __copy_ctor_is_trivial = Self._Storage.__copy_ctor_is_trivial
    comptime __move_ctor_is_trivial = Self._Storage.__move_ctor_is_trivial

    # Fields
    var _storage: Self._Storage

    @staticmethod
    def _check[T: AnyType]():
        comptime idx = _get_type_index[T, *Self.Ts]()
        comptime assert (
            idx != _InvalidTypeIndex
        ), "Type does not exist in Variant."

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @implicit
    def __init__[T: Movable](out self, var value: T):
        """Create a variant with one of the types.

        Parameters:
            T: The type to initialize the variant to. Generally this should
                be able to be inferred from the call type, eg. `Variant[Int, String](4)`.

        Args:
            value: The value to initialize the variant with.
        """
        Self._check[T]()
        self._storage = Self._Storage(value^)

    def __init__(out self, *, copy: Self):
        """Copy-initialize this variant from another variant of the same type.

        Args:
            copy: The variant to copy from.
        """
        # TODO(MOCO-3640): This should be a `where AllCopyable[*Self.Ts]`
        # constraint, but the compiler can't propagate evidence through
        # variadic conformance checks (e.g. Optional calling this with
        # Variant[_NoneType, T] can't prove AllCopyable from conforms_to(T,
        # Copyable)). Using comptime assert as a workaround.
        comptime assert AllCopyable[
            *Self.Ts
        ], "Cannot copy Variant with non-copyable types"
        self._storage = Self._Storage(copy=copy._storage)

    def __init__(out self, *, deinit take: Self):
        """Move-initialize this variant from another variant of the same type.

        Args:
            take: The variant to move from.
        """
        comptime assert _all_movable[
            *Self.Ts
        ](), "Cannot move Variant with non-movable types"
        self._storage = Self._Storage(take=take._storage^)

    def __del__(deinit self):
        """Destroy the variant, running the destructor of the currently held value.

        Constraints:
            All types in `Ts` must conform to `ImplicitlyDestructible`.
        """
        comptime assert _all_implicitly_destructible[
            *Self.Ts
        ](), "Cannot call __del__ on Variant with explicitly destroyed types"
        self._storage^.__del__()

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline
    def __getitem_param__[T: AnyType](ref self) -> ref[self] T:
        """Get the value out of the variant as a type-checked type.

        This explicitly check that your value is of that type!
        If you haven't verified the type correctness at runtime, the program
        will abort!

        For now this has the limitations that it
            - requires the variant value to be mutable

        Parameters:
            T: The type of the value to get out.

        Returns:
            A reference to the internal data.
        """
        if not self.isa[T]():
            abort("get: wrong variant type", location=call_location())

        return self.unsafe_get[T]()

    @always_inline
    def __eq__(self, other: Self) -> Bool where AllEquatable[*Self.Ts]:
        """Compares two variants for equality.

        Two variants are equal if they hold the same type and the held
        values are equal.

        Args:
            other: The other variant to compare against.

        Returns:
            True if the variants hold the same type and equal values.
        """
        comptime for i in range(Self.Ts.size):
            comptime T = Self.Ts[i]
            if self.isa[T]():
                if not other.isa[T]():
                    return False
                return trait_downcast[Equatable](
                    self.unsafe_get[T]()
                ) == trait_downcast[Equatable](other.unsafe_get[T]())
        return False

    @always_inline
    def __ne__(self, other: Self) -> Bool where AllEquatable[*Self.Ts]:
        """Compares two variants for inequality.

        Args:
            other: The other variant to compare against.

        Returns:
            True if the variants hold different types or unequal values.
        """
        return not self == other

    def __hash__(self, mut hasher: Some[Hasher]) where AllHashable[*Self.Ts]:
        """Hashes the variant using the given hasher.

        The hash incorporates both the type discriminant and the held
        value's hash, so variants holding different types are unlikely to
        collide.

        Args:
            hasher: The hasher instance.
        """
        comptime for i in range(Self.Ts.size):
            comptime T = Self.Ts[i]
            if self.isa[T]():
                hasher.update(UInt8(i))
                trait_downcast[Hashable](self.unsafe_get[T]()).__hash__(hasher)
                return

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    def _write_value_to[
        *, is_repr: Bool
    ](self, mut writer: Some[Writer]) where AllWritable[*Self.Ts]:
        comptime for i in range(Self.Ts.size):
            comptime T = Self.Ts[i]
            if self.isa[T]():
                ref value = trait_downcast[Writable](self.unsafe_get[T]())

                comptime if is_repr:
                    value.write_repr_to(writer)
                else:
                    value.write_to(writer)

                return

    @no_inline
    def write_to(self, mut writer: Some[Writer]) where AllWritable[*Self.Ts]:
        """Writes the currently held variant value to the provided Writer.

        Args:
            writer: The object to write to.
        """
        self._write_value_to[is_repr=False](writer)

    @no_inline
    def write_repr_to(
        self, mut writer: Some[Writer]
    ) where AllWritable[*Self.Ts]:
        """Write the string representation of the Variant.

        Args:
            writer: The object to write to.
        """

        @parameter
        def write_field(mut w: Some[Writer]):
            self._write_value_to[is_repr=True](w)

        FormatStruct(writer, "Variant").params(TypeNames[*Self.Ts]()).fields[
            FieldsFn=write_field
        ]()

    @always_inline
    def take[T: Movable](deinit self) -> T:
        """Take the current value of the variant with the provided type.

        The caller takes ownership of the underlying value.

        This explicitly check that your value is of that type!
        If you haven't verified the type correctness at runtime, the program
        will abort!

        Parameters:
            T: The type to take out.

        Returns:
            The underlying data to be taken out as an owned value.
        """
        # TODO(MOCO-3336): Remove isa/storage hack.
        # Explicitly destroyed types don't play nicely with abort
        var isa = self.isa[T]()
        var storage = self._storage^
        if not isa:
            std.memory.forget_deinit(storage^)
            abort("taking the wrong type!")

        return storage^.take[T]()

    @always_inline
    def unsafe_take[T: Movable](deinit self) -> T:
        """Unsafely take the current value of the variant with the provided type.

        The caller takes ownership of the underlying value.

        This doesn't explicitly check that your value is of that type!
        If you haven't verified the type correctness at runtime, you'll get
        a type that _looks_ like your type, but has potentially unsafe
        and garbage member data.

        Parameters:
            T: The type to take out.

        Returns:
            The underlying data to be taken out as an owned value.
        """
        Self._check[T]()
        assert self.isa[T](), "taking wrong type"
        return self._storage^.take[T]()

    @always_inline
    def replace[
        Tin: Movable & ImplicitlyDestructible,
        Tout: Movable,
    ](mut self, var value: Tin) -> Tout:
        """Replace the current value of the variant with the provided type.

        The caller takes ownership of the underlying value.

        This explicitly check that your value is of that type!
        If you haven't verified the type correctness at runtime, the program
        will abort!

        Parameters:
            Tin: The type to put in.
            Tout: The type to take out.

        Args:
            value: The value to put in.

        Returns:
            The underlying data to be taken out as an owned value.
        """
        if not self.isa[Tout]():
            abort("taking out the wrong type!")

        return self.unsafe_replace[Tin, Tout](value^)

    @always_inline
    def unsafe_replace[
        Tin: Movable, Tout: Movable
    ](mut self, var value: Tin) -> Tout:
        """Unsafely replace the current value of the variant with the provided type.

        The caller takes ownership of the underlying value.

        This doesn't explicitly check that your value is of that type!
        If you haven't verified the type correctness at runtime, you'll get
        a type that _looks_ like your type, but has potentially unsafe
        and garbage member data.

        Parameters:
            Tin: The type to put in.
            Tout: The type to take out.

        Args:
            value: The value to put in.

        Returns:
            The underlying data to be taken out as an owned value.
        """
        assert self.isa[Tout](), "taking out the wrong type!"

        var x = self^.unsafe_take[Tout]()
        self = Self(value^)
        return x^

    def set[T: Movable](mut self, var value: T):
        """Set the variant value.

        This will call the destructor on the old value, and update the variant's
        internal type and data to the new value.

        Parameters:
            T: The new variant type. Must be one of the Variant's type arguments.

        Args:
            value: The new value to set the variant to.
        """
        self = Self(value^)

    def isa[T: AnyType](self) -> Bool:
        """Check if the variant contains the required type.

        Parameters:
            T: The type to check.

        Returns:
            True if the variant contains the requested type.
        """
        Self._check[T]()
        return self._storage.isa[T]()

    def unsafe_get[T: AnyType](ref self) -> ref[self] T:
        """Get the value out of the variant as a type-checked type.

        This doesn't explicitly check that your value is of that type!
        If you haven't verified the type correctness at runtime, you'll get
        a type that _looks_ like your type, but has potentially unsafe
        and garbage member data.

        For now this has the limitations that it
            - requires the variant value to be mutable

        Parameters:
            T: The type of the value to get out.

        Returns:
            The internal data represented as a `Pointer[T]`.
        """
        Self._check[T]()
        assert self.isa[T](), "get: wrong variant type"
        return self._storage.unsafe_ptr[T]().unsafe_origin_cast[
            origin_of(self)
        ]()[]

    @staticmethod
    def is_type_supported[T: Movable]() -> Bool:
        """Check if a type can be used by the `Variant`.

        Parameters:
            T: The type of the value to check support for.

        Returns:
            `True` if type `T` is supported by the `Variant`.

        Example:

        ```mojo
        from std.utils import Variant

        def takes_variant(mut arg: Variant) raises:
            if arg.is_type_supported[Float64]():
                arg = Float64(1.5)

        def main() raises:
            var x = Variant[Int, Float64](1)
            takes_variant(x)
            if x.isa[Float64]():
                print(x[Float64]) # 1.5
        ```

        For example, the `Variant[Int, Bool]` permits `Int` and `Bool`.
        """
        return Self.Ts.contains[T]()

    def destroy_with[T: Movable, F: def(var T)](deinit self, destroy_func: F):
        """Destroy a value contained in this Variant in-place using a caller
        provided destructor function.

        This method can be used to destroy types marked `@explicit_destroy`
        in a `Variant` in-place, without requiring that they be
        `ImplicitlyDestructible`.

        This method will abort if this variant does not current contain an
        element of the specified type `T`.

        Parameters:
            T: The element type the variant is expected to currently contain,
                and which will be destroyed by `destroy_func`.
            F: The type of the caller-provided destructor function.

        Args:
            destroy_func: Caller-provided destructor function for destroying
                an instance of `T`.
        """
        # TODO(MOCO-3336): Remove isa/storage hack.
        # Explicitly destroyed types don't play nicely with abort
        var isa = self.isa[T]()
        var storage = self._storage^
        if not isa:
            std.memory.forget_deinit(storage^)
            abort("Variant.destroy_with: wrong variant type")

        destroy_func(storage^.take[T]())


# ===-------------------------------------------------------------------===#
# Helper functions
# ===-------------------------------------------------------------------===#


def _all_implicitly_destructible[*Ts: AnyType]() -> Bool:
    comptime for i in range(Ts.size):
        comptime T = Ts[i]
        if not conforms_to(T, ImplicitlyDestructible):
            return False
    return True


def _all_movable[*Ts: AnyType]() -> Bool:
    comptime for i in range(Ts.size):
        comptime T = Ts[i]
        if not conforms_to(T, Movable):
            return False
    return True


def _all_trivial_del[*Ts: AnyType]() -> Bool:
    comptime for i in range(Ts.size):
        comptime if conforms_to(Ts[i], ImplicitlyDestructible):
            if not downcast[Ts[i], ImplicitlyDestructible].__del__is_trivial:
                return False
        else:
            return False
    return True


def _all_trivial_copyinit[*Ts: AnyType]() -> Bool:
    comptime for i in range(Ts.size):
        comptime if conforms_to(Ts[i], Copyable):
            if not downcast[Ts[i], Copyable].__copy_ctor_is_trivial:
                return False
        else:
            return False

    return True


def _all_trivial_moveinit[*Ts: AnyType]() -> Bool:
    comptime for i in range(Ts.size):
        comptime if conforms_to(Ts[i], Movable):
            if not downcast[Ts[i], Movable].__move_ctor_is_trivial:
                return False
        else:
            return False
    return True
