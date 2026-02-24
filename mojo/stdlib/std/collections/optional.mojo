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

from os import abort

from utils import Variant

from builtin.constrained import _constrained_conforms_to
from builtin.device_passable import DevicePassable
from compile import get_type_name
from format._utils import FormatStruct, TypeNames, write_to, write_repr_to


# TODO(27780): NoneType can't currently conform to traits
@fieldwise_init
struct _NoneType(ImplicitlyCopyable):
    pass


@fieldwise_init
struct EmptyOptionalError[T: AnyType](
    ImplicitlyCopyable, RegisterPassable, Writable
):
    """An error type for when an empty `Optional` is accessed.

    Parameters:
        T: The type of the value that was accessed in the `Optional`.
    """

    fn write_to(self, mut writer: Some[Writer]):
        """Write the error to a `Writer`.

        Args:
            writer: The `Writer` to write to.
        """
        FormatStruct(writer, "EmptyOptionalError").params(
            TypeNames[Self.T]()
        ).fields()

    fn write_repr_to(self, mut writer: Some[Writer]):
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
    ImplicitlyCopyable where conforms_to(T, ImplicitlyCopyable),
    Iterable,
    Iterator,
    Representable where conforms_to(T, Representable),
    Stringable where conforms_to(T, Stringable),
    Writable where conforms_to(T, Writable),
):
    """A type modeling a value which may or may not be present.

    Parameters:
        T: The type of value stored in the `Optional`.

    Optional values can be thought of as a type-safe nullable pattern.
    Your value can take on a value or `None`, and you need to check
    and explicitly extract the value to get it out.

    Currently T is required to be a `Copyable` so we can implement
    copy/move for Optional and allow it to be used in collections itself.

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

    # Fields
    # _NoneType comes first so its index is 0.
    # This means that Optionals that are 0-initialized will be None.
    comptime _type = Variant[_NoneType, Self.T]
    var _value: Self._type

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __init__(out self):
        """Construct an empty `Optional`.

        Examples:

        ```mojo
        instance = Optional[String]()
        print(instance) # Output: None
        ```
        """
        self._value = Self._type(_NoneType())

    @implicit
    fn __init__(out self, var value: Self.T):
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
    @doc_private
    @implicit
    fn __init__(out self, value: NoneType._mlir_type):
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
    fn __init__(out self, value: NoneType):
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

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    fn __is__(self, other: NoneType) -> Bool:
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

    fn __isnot__(self, other: NoneType) -> Bool:
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

    fn __eq__(self, rhs: NoneType) -> Bool:
        """Return `True` if a value is not present.

        Args:
            rhs: The `None` value to compare to.

        Returns:
            `True` if a value is not present, `False` otherwise.
        """
        return self is None

    fn __eq__[
        _T: Equatable & Copyable
    ](self: Optional[_T], rhs: Optional[_T]) -> Bool:
        """Return `True` if this is the same as another `Optional` value,
        meaning both are absent, or both are present and have the same
        underlying value.

        Parameters:
            _T: The type of the elements in the list. Must implement the
                traits `Copyable` and `Equatable`.

        Args:
            rhs: The value to compare to.

        Returns:
            True if the values are the same.
        """
        if self:
            if rhs:
                return self.value() == rhs.value()
            return False
        return not rhs

    fn __ne__(self, rhs: NoneType) -> Bool:
        """Return `True` if a value is present.

        Args:
            rhs: The `None` value to compare to.

        Returns:
            `False` if a value is not present, `True` otherwise.
        """
        return self is not None

    fn __ne__[
        _T: Equatable & Copyable, //
    ](self: Optional[_T], rhs: Optional[_T]) -> Bool:
        """Return `False` if this is the same as another `Optional` value,
        meaning both are absent, or both are present and have the same
        underlying value.

        Parameters:
            _T: The type of the elements in the list. Must implement the
                traits `Copyable` and `Equatable`.

        Args:
            rhs: The value to compare to.

        Returns:
            False if the values are the same.
        """
        return not (self == rhs)

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
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
        return self.copy()

    @always_inline
    fn __next__(mut self) raises StopIteration -> Self.Element:
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
    fn bounds(self) -> Tuple[Int, Optional[Int]]:
        """Return the bounds of the `Optional`, which is 0 or 1.

        Returns:
            A tuple containing the length (0 or 1) and an `Optional` containing the length.

        Examples:

        ```mojo
        fn bounds():
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
    fn __bool__(self) -> Bool:
        """Return true if the Optional has a value.

        Returns:
            True if the `Optional` has a value and False otherwise.
        """
        return not self._value.isa[_NoneType]()

    @always_inline
    fn __invert__(self) -> Bool:
        """Return False if the `Optional` has a value.

        Returns:
            False if the `Optional` has a value and True otherwise.
        """
        return not self

    @always_inline
    fn __getitem__(
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

    fn __str__(self: Self) -> String where conforms_to(Self.T, Stringable):
        """Return the string representation of the value of the `Optional`.

        Returns:
            A string representation of the `Optional`.
        """
        if self:
            return trait_downcast[Stringable](self.value()).__str__()
        else:
            return "None"

    fn __repr__(self: Self) -> String where conforms_to(Self.T, Representable):
        """Returns the verbose string representation of the `Optional`.

        Returns:
            A verbose string representation of the `Optional`.
        """
        var output = String()
        output.write("Optional(", self, ")")
        return output^

    @always_inline("nodebug")
    fn __merge_with__[
        other_type: type_of(Bool),
    ](self) -> Bool:
        """Merge with other bools in an expression.

        Parameters:
            other_type: The type of the bool to merge with.

        Returns:
            A Bool after merging with the specified `other_type`.
        """
        return self.__bool__()

    fn _write_to[*, is_repr: Bool](self: Self, mut writer: Some[Writer]):
        _constrained_conforms_to[
            conforms_to(Self.T, Writable),
            Parent=Self,
            Element = Self.T,
            ParentConformsTo="Writable",
        ]()

        if self:
            comptime if is_repr:
                trait_downcast[Writable](self.value()).write_repr_to(writer)
            else:
                trait_downcast[Writable](self.value()).write_to(writer)
        else:
            writer.write_string("None")

    fn write_to(
        self: Self, mut writer: Some[Writer]
    ) where conforms_to(Self.T, Writable):
        """Write this `Optional` to a `Writer`.

        Args:
            writer: The object to write to.
        """
        self._write_to[is_repr=False](writer)

    fn write_repr_to(
        self: Self, mut writer: Some[Writer]
    ) where conforms_to(Self.T, Writable):
        """Write this `Optional`'s representation to a `Writer`.

        Args:
            writer: The object to write to.
        """

        @parameter
        fn fields(mut w: Some[Writer]):
            self._write_to[is_repr=True](w)

        FormatStruct(writer, "Optional").params(TypeNames[Self.T]()).fields[
            FieldsFn=fields
        ]()

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn value(ref self) -> ref[self._value] Self.T:
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
                "`Optional.value()` called on empty `Optional`. Consider using"
                " `if optional:` to check whether the `Optional` is empty"
                " before calling `.value()`, or use `.or_else()` to provide a"
                " default value."
            )

        return self.unsafe_value()

    @always_inline
    fn unsafe_value(ref self) -> ref[self._value] Self.T:
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
        debug_assert(self.__bool__(), "`.value()` on empty `Optional`")
        return self._value.unsafe_get[Self.T]()

    fn take(mut self) -> Self.T:
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

    fn unsafe_take(mut self) -> Self.T:
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
        debug_assert(self.__bool__(), "`.unsafe_take()` on empty `Optional`")
        return self._value.unsafe_replace[_NoneType, Self.T](_NoneType())

    fn or_else[
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
            return self.unsafe_take()
        return default^

    fn copied[
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


struct OptionalReg[T: __TypeOfAllTypes](
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

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        """Get the human-readable type name for this `OptionalReg` type.

        Returns:
            A string representation of the type, e.g. `OptionalReg[Int]`.
        """
        return String("OptionalReg[", get_type_name[Self.T](), "]")

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline("builtin")
    fn __init__(out self):
        """Create an optional with a value of None."""
        self = Self(None)

    @always_inline("builtin")
    @implicit
    fn __init__(out self, value: Self.T):
        """Create an optional with a value.

        Args:
            value: The value.
        """
        self._value = __mlir_op.`kgen.variant.create`[
            _type = Self._mlir_type, index = Int(0)._mlir_value
        ](value)

    # TODO(MSTDL-715):
    #   This initializer should not be necessary, we should need
    #   only the initializer from a `NoneType`.
    @doc_private
    @always_inline("builtin")
    @implicit
    fn __init__(out self, value: NoneType._mlir_type):
        """Construct an empty Optional.

        Args:
            value: Must be exactly `None`.
        """
        self = Self(value=NoneType(value))

    @always_inline("builtin")
    @implicit
    fn __init__(out self, value: NoneType):
        """Create an optional without a value from a None literal.

        Args:
            value: The None value.
        """
        self._value = __mlir_op.`kgen.variant.create`[
            _type = Self._mlir_type, index = Int(1)._mlir_value
        ](__mlir_attr.false)

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    fn __is__(self, other: NoneType) -> Bool:
        """Return `True` if the Optional has no value.

        It allows you to use the following syntax: `if my_optional is None:`

        Args:
            other: The value to compare to (None).

        Returns:
            True if the Optional has no value and False otherwise.
        """
        return not self.__bool__()

    fn __isnot__(self, other: NoneType) -> Bool:
        """Return `True` if the Optional has a value.

        It allows you to use the following syntax: `if my_optional is not None:`

        Args:
            other: The value to compare to (None).

        Returns:
            True if the Optional has a value and False otherwise.
        """
        return self.__bool__()

    @always_inline("nodebug")
    fn __merge_with__[
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

    fn __bool__(self) -> Bool:
        """Return true if the optional has a value.

        Returns:
            True if the optional has a value and False otherwise.
        """
        return __mlir_op.`kgen.variant.is`[index = Int(0)._mlir_value](
            self._value
        )

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn value(self) -> Self.T:
        """Get the optional value.

        Returns:
            The contained value.
        """
        return __mlir_op.`kgen.variant.get`[index = Int(0)._mlir_value](
            self._value
        )

    fn or_else(var self, var default: Self.T) -> Self.T:
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
