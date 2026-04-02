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
"""Defines Optional, a type modeling a value which may or may not be present.

Optional values can be thought of as a type-safe nullable pattern.
Your value can take on a value or `None`, and you need to check
and explicitly extract the value to get it out.

Examples:

```mojo
var a = Optional(1)
var b = Optional[Int](None)
if a:
    print(a.value())  # prints 1
if b:  # Bool(b) is False, so no print
    print(b.value())
var c = a.or_else(2)
var d = b.or_else(2)
print(c)  # prints 1
print(d)  # prints 2
```
"""

from std.os import abort

from std.utils import Variant

from std.builtin.device_passable import DevicePassable
from std.compile import get_type_name
from std.format._utils import FormatStruct, TypeNames, write_to, write_repr_to
from std.hashlib import Hasher
from std.memory._nonnull import NonNullUnsafePointer, unsafe_origin_cast
from std.reflection import call_location


@fieldwise_init
struct _NoneType(TrivialRegisterPassable):
    pass


@fieldwise_init
struct EmptyOptionalError[T: AnyType](
    ImplicitlyCopyable, RegisterPassable, Writable
):
    """An error type for when an empty `Optional` is accessed.

    Parameters:
        T: The type of the value that was accessed in the `Optional`.
    """

    def write_to(self, mut writer: Some[Writer]):
        """Write the error to a `Writer`.

        Args:
            writer: The `Writer` to write to.
        """
        FormatStruct(writer, "EmptyOptionalError").params(
            TypeNames[Self.T]()
        ).fields()

    def write_repr_to(self, mut writer: Some[Writer]):
        """Write the error to a `Writer`.

        Args:
            writer: The `Writer` to write to.
        """
        self.write_to(writer)


# ===-----------------------------------------------------------------------===#
# Optional
# ===-----------------------------------------------------------------------===#


struct Optional[T: Movable](
    Boolable,
    Copyable where conforms_to(T, Copyable),
    Defaultable,
    DevicePassable where conforms_to(T, DevicePassable) and conforms_to(
        T, Copyable
    ),
    Equatable where conforms_to(T, Equatable),
    Hashable where conforms_to(T, Hashable),
    ImplicitlyCopyable where conforms_to(T, ImplicitlyCopyable),
    Iterable,
    Iterator,
    Movable,
    RegisterPassable where conforms_to(T, RegisterPassable),
    Writable where conforms_to(T, Writable),
):
    """A type modeling a value which may or may not be present.

    Parameters:
        T: The type of value stored in the `Optional`.

    Optional values can be thought of as a type-safe nullable pattern.
    Your value can take on a value or `None`, and you need to check
    and explicitly extract the value to get it out.

    ## Layout

    The layout of `Optional` is not guaranteed and may change at any time.
    The implementation may apply niche optimizations (for example, storing the
    `None` sentinel inside spare bits of `T`) that alter the resulting layout.
    Do not rely on `size_of[Optional[T]]()` or `align_of[Optional[T]]()` being
    stable across compiler versions. The only guarantee is that the size and
    alignment will be at least as large as those of `T` itself.

    If you need to inspect the current size or alignment, use `size_of` and
    `align_of`, but treat the results as non-stable implementation details.

    Examples:

    ```mojo
    var a = Optional(1)
    var b = Optional[Int](None)
    if a:
        print(a.value())  # prints 1
    if b:  # Bool(b) is False, so no print
        print(b.value())
    var c = a.or_else(2)
    var d = b.or_else(2)
    print(c)  # prints 1
    print(d)  # prints 2
    ```
    """

    # Iterator aliases
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    """The iterator type for this optional.

    Parameters:
        iterable_mut: Whether the iterable is mutable.
        iterable_origin: The origin of the iterable.
    """

    comptime Element = Self.T
    """The element type of this optional."""

    comptime device_type: AnyType = Self
    """The device-side type for this optional."""

    comptime _type = Variant[_NoneType, Self.T]
    var _value: Self._type

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    def __init__(out self):
        """Construct an empty `Optional`.

        Examples:

        ```mojo
        instance = Optional[String]()
        print(instance) # Output: None
        ```
        """
        self._value = Self._type(_NoneType())

    @implicit
    def __init__(out self, var value: Self.T):
        """Construct an `Optional` containing a value.

        Args:
            value: The value to store in the `Optional`.

        Examples:

        ```mojo
        instance = Optional[String]("Hello")
        print(instance) # Output: 'Hello'
        ```
        """
        self._value = Self._type(value^)

    # TODO(MSTDL-715):
    #   This initializer should not be necessary, we should need
    #   only the initializer from a `NoneType`.
    @doc_hidden
    @implicit
    def __init__(out self, value: NoneType._mlir_type):
        """Construct an empty `Optional`.

        Args:
            value: Must be exactly `None`.

        Examples:

        ```mojo
        instance = Optional[String](None)
        print(instance) # Output: None
        ```
        """
        self = Self(value=NoneType(value))

    @implicit
    def __init__(out self, value: NoneType):
        """Construct an empty `Optional`.

        Args:
            value: Must be exactly `None`.

        Examples:

        ```mojo
        instance = Optional[String](None)
        print(instance) # Output: None
        ```
        """
        self = Self()

    @implicit
    @doc_hidden
    @always_inline
    def __init__[
        U: AnyType, origin: Origin, address_space: AddressSpace, //
    ](
        out self: Optional[
            NonNullUnsafePointer[U, origin, address_space=address_space]
        ],
        nullable: UnsafePointer[U, origin, address_space=address_space],
    ):
        self = nullable.as_nonnull()

    @always_inline
    @implicit
    @doc_hidden
    def __init__(
        nullable: UnsafePointer[...],
        out self: Optional[
            NonNullUnsafePointer[
                nullable.type,
                AnyOrigin[mut=False],
                address_space=nullable.address_space,
            ]
        ],
    ):
        self = unsafe_origin_cast[AnyOrigin[mut=False]](nullable.as_nonnull())

    @always_inline
    @implicit
    @doc_hidden
    def __init__(
        nullable: UnsafePointer[mut=True, ...],
        out self: Optional[
            NonNullUnsafePointer[
                nullable.type,
                AnyOrigin[mut=True],
                address_space=nullable.address_space,
            ]
        ],
    ):
        self = unsafe_origin_cast[AnyOrigin[mut=True]](nullable.as_nonnull())

    # TODO(MOCO-3640): Remove once the compiler can synthesize copy
    # constructors through variadic conditional conformances
    # (AllCopyable[_NoneType, T] when T: Copyable).
    @always_inline
    def __init__(out self, *, copy: Self):
        """Copy-initialize an `Optional`.

        Args:
            copy: The `Optional` to copy from.
        """
        self._value = Self._type(copy=copy._value)

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    def __is__(self, other: NoneType) -> Bool:
        """Return `True` if the Optional has no value.

        Args:
            other: The value to compare to (None).

        Returns:
            True if the Optional has no value and False otherwise.

        Notes:
            It allows you to use the following syntax:
            `if my_optional is None:`.
        """
        return not self.__bool__()

    def __isnot__(self, other: NoneType) -> Bool:
        """Return `True` if the Optional has a value.

        Args:
            other: The value to compare to (None).

        Returns:
            True if the Optional has a value and False otherwise.

        Notes:
            It allows you to use the following syntax:
            `if my_optional is not None:`.
        """
        return self.__bool__()

    def __eq__(self, rhs: type_of(None)) -> Bool:
        """Return `True` if a value is not present.

        Args:
            rhs: The `None` value to compare to.

        Returns:
            `True` if a value is not present, `False` otherwise.
        """
        return self is None

    def __eq__(self, rhs: Self) -> Bool where conforms_to(Self.T, Equatable):
        """Return `True` if this is the same as another `Optional` value,
        meaning both are absent, or both are present and have the same
        underlying value.

        Args:
            rhs: The value to compare to.

        Returns:
            True if the values are the same.
        """
        if self:
            if rhs:
                return trait_downcast[Equatable](
                    self.unsafe_value()
                ) == trait_downcast[Equatable](rhs.unsafe_value())
            return False
        return not rhs

    def __ne__(self, rhs: type_of(None)) -> Bool:
        """Return `True` if a value is present.

        Args:
            rhs: The `None` value to compare to.

        Returns:
            `False` if a value is not present, `True` otherwise.
        """
        return self is not None

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        """Iterate over the Optional's possibly contained value.

        Optionals act as a collection of size 0 or 1.

        Returns:
            An iterator over the Optional's value (if present).

        Examples:

        ```mojo
        instance = Optional("Hello")
        for value in instance:
            print(value) # Output: Hello
        instance = None
        for value in instance:
            print(value) # Does not reach line
        ```
        """
        comptime assert conforms_to(
            Self.T, Copyable
        ), "Cannot iterate over non-copyable Optional."
        return self.copy()

    @always_inline
    def __next__(mut self) raises StopIteration -> Self.Element:
        """Return the contained value of the Optional.

        Returns:
            The value contained in the Optional.

        Raises:
            `StopIteration` if the iterator has been exhausted.
        """
        if not self.__bool__():
            raise StopIteration()
        return self.take()

    @always_inline
    def bounds(self) -> Tuple[Int, Optional[Int]]:
        """Return the bounds of the `Optional`, which is 0 or 1.

        Returns:
            A tuple containing the length (0 or 1) and an `Optional` containing the length.

        Examples:

        ```mojo
        def bounds():
            empty_instance = Optional[Int]()
            populated_instance = Optional[Int](50)

            # Bounds returns a tuple: (`bounds`, `Optional` version of `bounds`)
            # with the length of the `Optional`.
            print(empty_instance.bounds()[0])     # 0
            print(populated_instance.bounds()[0]) # 1
            print(empty_instance.bounds()[1])     # 0
            print(populated_instance.bounds()[1]) # 1
        ```
        """
        var len = 1 if self else 0
        return (len, {len})

    @always_inline
    def __bool__(self) -> Bool:
        """Return true if the Optional has a value.

        Returns:
            True if the `Optional` has a value and False otherwise.
        """
        return not self._value.isa[_NoneType]()

    @always_inline
    def __invert__(self) -> Bool:
        """Return False if the `Optional` has a value.

        Returns:
            False if the `Optional` has a value and True otherwise.
        """
        return not self

    @always_inline
    def __getitem__(
        ref self,
    ) raises EmptyOptionalError[Self.T] -> ref[self._value] Self.T:
        """Retrieve a reference to the value inside the `Optional`.

        Returns:
            A reference to the value inside the `Optional`.

        Raises:
            On empty `Optional`.
        """
        if not self:
            raise EmptyOptionalError[Self.T]()
        return self.unsafe_value()

    @always_inline("nodebug")
    def __merge_with__[
        other_type: type_of(Bool),
    ](self) -> Bool:
        """Merge with other bools in an expression.

        Parameters:
            other_type: The type of the bool to merge with.

        Returns:
            A Bool after merging with the specified `other_type`.
        """
        return self.__bool__()

    def _write_to[
        *, is_repr: Bool
    ](self: Self, mut writer: Some[Writer]) where conforms_to(Self.T, Writable):
        if self:
            comptime if is_repr:
                trait_downcast[Writable](self.value()).write_repr_to(writer)
            else:
                trait_downcast[Writable](self.value()).write_to(writer)
        else:
            writer.write_string("None")

    def write_to(
        self: Self, mut writer: Some[Writer]
    ) where conforms_to(Self.T, Writable):
        """Write this `Optional` to a `Writer`.

        Args:
            writer: The object to write to.
        """
        self._write_to[is_repr=False](writer)

    def write_repr_to(
        self: Self, mut writer: Some[Writer]
    ) where conforms_to(Self.T, Writable):
        """Write this `Optional`'s representation to a `Writer`.

        Args:
            writer: The object to write to.
        """

        @parameter
        def fields(mut w: Some[Writer]):
            self._write_to[is_repr=True](w)

        FormatStruct(writer, "Optional").params(TypeNames[Self.T]()).fields[
            FieldsFn=fields
        ]()

    def __hash__[
        H: Hasher
    ](self, mut hasher: H) where conforms_to(Self.T, Hashable):
        """Updates hasher with the hash of the contained value, if present.

        A `None` optional hashes differently from any present value.

        Parameters:
            H: The hasher type.

        Args:
            hasher: The hasher instance.
        """
        if self:
            # Tag the hash so that hash(T) != hash(Optional[T](..)).
            hasher.update(UInt8(1))
            trait_downcast[Hashable](self.value()).__hash__(hasher)
        else:
            hasher.update(UInt8(0))

    def _to_device_type(
        self, target: MutOpaquePointer[_]
    ) where conforms_to(Self.T, DevicePassable) and conforms_to(
        Self.T, Copyable
    ):
        """Convert to device type and store at the target address.

        Args:
            target: The target pointer to store the device type.
        """
        target.bitcast[Self]().init_pointee_copy(self)

    @staticmethod
    def get_type_name() -> (
        String
    ) where conforms_to(Self.T, DevicePassable) and conforms_to(
        Self.T, Copyable
    ):
        """Get the human-readable type name for this `Optional` type.

        Returns:
            A string representation of the type, e.g. `Optional[Int]`.
        """
        return String(t"Optional[{get_type_name[Self.T]()}]")

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    def value(ref self) -> ref[self._value] Self.T:
        """Retrieve a reference to the value of the `Optional`.

        Returns:
            A reference to the contained data of the `Optional` as a reference.

        Notes:
            This will abort on empty `Optional`.

        Examples:

        ```mojo
        instance = Optional("Hello")
        x = instance.value()
        print(x) # Hello
        # instance = Optional[String]() # Uncomment both lines to crash
        # print(instance.value())       # Attempts to take value from `None`
        ```
        """
        if not self.__bool__():
            abort(
                (
                    "`Optional.value()` called on empty `Optional`. Consider"
                    " using `if optional:` to check whether the `Optional` is"
                    " empty before calling `.value()`, or use `.or_else()` to"
                    " provide a default value."
                ),
                location=call_location(),
            )

        return self.unsafe_value()

    @always_inline
    def unsafe_value(ref self) -> ref[self._value] Self.T:
        """Unsafely retrieve a reference to the value of the `Optional`.

        Returns:
            A reference to the contained data of the `Optional` as a reference.

        Notes:
            This will **not** abort on empty `Optional`.

        Examples:

        ```mojo
        instance = Optional("Hello")
        x = instance.unsafe_value()
        print(x) # Hello
        instance = Optional[String](None)

        # Best practice:
        if instance:
            y = instance.unsafe_value() # Will not reach this line
            print(y)

        # In debug builds, this will deterministically abort:
        y = instance.unsafe_value()
        print(y)
        ```
        """
        assert self.__bool__(), "`.value()` on empty `Optional`"
        return self._value.unsafe_get[Self.T]()

    def take(mut self) -> Self.T:
        """Move the value out of the `Optional`.

        Returns:
            The contained data of the `Optional` as an owned T value.

        Notes:
            This will abort on empty `Optional`.

        Examples:

        ```mojo
        instance = Optional("Hello")
        print(instance.bounds()[0])  # Output: 1
        x = instance.take() # Moves value from `instance` to `x`
        print(x)  # Output: Hello

        # `instance` is now `Optional(None)`
        print(instance.bounds()[0])  # Output: 0
        print(instance)  # Output: None

        # Best practice
        if instance:
            y = instance.take()  # Won't reach this line
            print(y)

        # Used directly
        # y = instance.take()         # ABORT: `Optional.take()` called on empty `Optional` (via runtime `abort`)
        # print(y)                    # Does not reach this line
        ```
        """
        if not self.__bool__():
            abort(
                "`Optional.take()` called on empty `Optional`. Consider using"
                " `if optional:` to check whether the `Optional` is empty"
                " before calling `.take()`, or use `.or_else()` to provide a"
                " default value."
            )
        return self.unsafe_take()

    def unsafe_take(mut self) -> Self.T:
        """Unsafely move the value out of the `Optional`.

        Returns:
            The contained data of the `Optional` as an owned T value.

        Notes:
            This will **not** abort on empty `Optional`.

        Examples:

        ```mojo
        instance = Optional("Hello")
        print(instance.bounds()[0]) # Output: 1
        x = instance.unsafe_take()  # Moves value from `instance` to `x`
        print(x)                    # Output: Hello

        # `instance` is now `Optional(None)`
        print(instance.bounds()[0]) # Output: 0
        print(instance)             # Output: None

        # Best practice:
        if instance:
            y = instance.unsafe_take() # Won't reach this line
            print(y)

        # In debug builds, this will deterministically abort:
        y = instance.unsafe_take()  # ABORT: `Optional.take()` called on empty `Optional` (via `debug_assert`)
        print(y)                    # Does not reach this line
        ```
        """
        assert self.__bool__(), "`.unsafe_take()` on empty `Optional`"
        return self._value.unsafe_replace[_NoneType, Self.T](_NoneType())

    def or_else[
        _T: Movable & ImplicitlyDestructible, //
    ](deinit self: Optional[_T], var default: _T) -> _T:
        """Return the underlying value contained in the `Optional` or a default
        value if the `Optional`'s underlying value is not present.

        Parameters:
            _T: Type of the optional element, which must conform to
                `ImplicitlyDestructible`.

        Args:
            default: The new value to use if no value was present.

        Returns:
            The underlying value contained in the `Optional` or a default value.

        Examples:

        ```mojo
        instance = Optional("Hello")
        print(instance)                  # Output: 'Hello'
        print(instance.or_else("Bye"))   # Output: Hello
        instance = None
        print(instance)                  # Output: None
        print(instance.or_else("Bye"))   # Output: Bye
        ```
        """
        if self:
            return self._value^.unsafe_take[_T]()
        return default^

    def copied[
        mut: Bool,
        origin: Origin[mut=mut],
        //,
        _T: Copyable,
    ](self: Optional[Pointer[_T, origin]]) -> Optional[_T]:
        """Converts an `Optional` containing a Pointer to an `Optional` of an
        owned value by copying.

        Parameters:
            mut: Mutability of the pointee origin.
            origin: Origin of the contained `Pointer`.
            _T: Type of the owned result value.

        Returns:
            An `Optional` containing an owned copy of the pointee value.

        Examples:

        Copy the value of an `Optional[Pointer[_]]`

        ```mojo
        var data = "foo"
        var opt = Optional(Pointer(to=data))
        var opt_owned: Optional[String] = opt.copied()
        ```

        Notes:
            If `self` is an empty `Optional`, the returned `Optional` will be
            empty as well.
        """
        if self:
            # SAFETY: We just checked that `self` is populated.
            # Perform an implicit copy
            return self.unsafe_value()[].copy()
        else:
            return None


# ===-----------------------------------------------------------------------===#
# OptionalReg
# ===-----------------------------------------------------------------------===#


struct OptionalReg[T: TrivialRegisterPassable](
    Boolable, Defaultable, DevicePassable, TrivialRegisterPassable
):
    """A register-passable optional type.

    This struct optionally contains a value. It only works with trivial register
    passable types at the moment.

    Parameters:
        T: The type of value stored in the Optional.
    """

    # Fields
    comptime _mlir_type = __mlir_type[`!kgen.variant<`, Self.T, `, i1>`]
    var _value: Self._mlir_type

    comptime device_type: AnyType = Self
    """The device-side type for this optional register."""

    def _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    def get_type_name() -> String:
        """Get the human-readable type name for this `OptionalReg` type.

        Returns:
            A string representation of the type, e.g. `OptionalReg[Int]`.
        """
        return String(t"OptionalReg[{get_type_name[Self.T]()}]")

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline("builtin")
    def __init__(out self):
        """Create an optional with a value of None."""
        self = Self(None)

    @always_inline("builtin")
    @implicit
    def __init__(out self, value: Self.T):
        """Create an optional with a value.

        Args:
            value: The value.
        """
        self._value = __mlir_op.`kgen.variant.create`[
            _type=Self._mlir_type, index=Int(0)._mlir_value
        ](value)

    # TODO(MSTDL-715):
    #   This initializer should not be necessary, we should need
    #   only the initializer from a `NoneType`.
    @doc_hidden
    @always_inline("builtin")
    @implicit
    def __init__(out self, value: NoneType._mlir_type):
        """Construct an empty Optional.

        Args:
            value: Must be exactly `None`.
        """
        self = Self(value=NoneType(value))

    @always_inline("builtin")
    @implicit
    def __init__(out self, value: NoneType):
        """Create an optional without a value from a None literal.

        Args:
            value: The None value.
        """
        self._value = __mlir_op.`kgen.variant.create`[
            _type=Self._mlir_type, index=Int(1)._mlir_value
        ](__mlir_attr.false)

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    def __is__(self, other: NoneType) -> Bool:
        """Return `True` if the Optional has no value.

        It allows you to use the following syntax: `if my_optional is None:`

        Args:
            other: The value to compare to (None).

        Returns:
            True if the Optional has no value and False otherwise.
        """
        return not self.__bool__()

    def __isnot__(self, other: NoneType) -> Bool:
        """Return `True` if the Optional has a value.

        It allows you to use the following syntax: `if my_optional is not None:`

        Args:
            other: The value to compare to (None).

        Returns:
            True if the Optional has a value and False otherwise.
        """
        return self.__bool__()

    @always_inline("nodebug")
    def __merge_with__[
        other_type: type_of(Bool),
    ](self) -> Bool:
        """Merge with other bools in an expression.

        Parameters:
            other_type: The type of the bool to merge with.

        Returns:
            A Bool after merging with the specified `other_type`.
        """
        return self.__bool__()

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    def __bool__(self) -> Bool:
        """Return true if the optional has a value.

        Returns:
            True if the optional has a value and False otherwise.
        """
        return __mlir_op.`kgen.variant.is`[index=Int(0)._mlir_value](
            self._value
        )

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    def value(self) -> Self.T:
        """Get the optional value.

        Returns:
            The contained value.
        """
        return __mlir_op.`kgen.variant.get`[index=Int(0)._mlir_value](
            self._value
        )

    def or_else(var self, var default: Self.T) -> Self.T:
        """Return the underlying value contained in the Optional or a default
        value if the Optional's underlying value is not present.

        Args:
            default: The new value to use if no value was present.

        Returns:
            The underlying value contained in the Optional or a default value.
        """
        if self:
            return self.value()
        return default
