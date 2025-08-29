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


# TODO(27780): NoneType can't currently conform to traits
@fieldwise_init
struct _NoneType(Copyable, ExplicitlyCopyable, Movable):
    fn __copyinit__(out self, other: Self):
        pass


# ===-----------------------------------------------------------------------===#
# Optional
# ===-----------------------------------------------------------------------===#


struct Optional[T: ExplicitlyCopyable & Movable](
    Boolable, Copyable, Defaultable, ExplicitlyCopyable, Movable
):
    """A type modeling a value which may or may not be present.

    Parameters:
        T: The type of value stored in the `Optional`.

    Optional values can be thought of as a type-safe nullable pattern.
    Your value can take on a value or `None`, and you need to check
    and explicitly extract the value to get it out.

    Currently T is required to be a `Copyable & Movable` so we can implement
    copy/move for Optional and allow it to be used in collections itself.

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

    # Fields
    # _NoneType comes first so its index is 0.
    # This means that Optionals that are 0-initialized will be None.
    alias _type = Variant[_NoneType, T]
    var _value: Self._type

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __init__(out self):
        """Construct an empty `Optional`."""
        self._value = Self._type(_NoneType())

    @implicit
    fn __init__(out self, var value: T):
        """Construct an `Optional` containing a value.

        Args:
            value: The value to store in the `Optional`.
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
        """
        self = Self(value=NoneType(value))

    @implicit
    fn __init__(out self, value: NoneType):
        """Construct an empty `Optional`.

        Args:
            value: Must be exactly `None`.
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
        T: EqualityComparable & Copyable & Movable
    ](self: Optional[T], rhs: Optional[T]) -> Bool:
        """Return `True` if this is the same as another `Optional` value,
        meaning both are absent, or both are present and have the same
        underlying value.

        Parameters:
            T: The type of the elements in the list. Must implement the
                traits `Copyable`, `Movable` and `EqualityComparable`.

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
        T: EqualityComparable & Copyable & Movable, //
    ](self: Optional[T], rhs: Optional[T]) -> Bool:
        """Return `False` if this is the same as another `Optional` value,
        meaning both are absent, or both are present and have the same
        underlying value.

        Parameters:
            T: The type of the elements in the list. Must implement the
                traits `Copyable`, `Movable` and `EqualityComparable`.

        Args:
            rhs: The value to compare to.

        Returns:
            False if the values are the same.
        """
        return not (self == rhs)

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    fn __bool__(self) -> Bool:
        """Return true if the Optional has a value.

        Returns:
            True if the `Optional` has a value and False otherwise.
        """
        return not self._value.isa[_NoneType]()

    fn __invert__(self) -> Bool:
        """Return False if the `Optional` has a value.

        Returns:
            False if the `Optional` has a value and True otherwise.
        """
        return not self

    @always_inline
    fn __getitem__(ref self) raises -> ref [self._value] T:
        """Retrieve a reference to the value inside the `Optional`.

        Returns:
            A reference to the value inside the `Optional`.

        Raises:
            On empty `Optional`.
        """
        if not self:
            raise Error(".value() on empty Optional")
        return self.unsafe_value()

    fn __str__[
        U: Copyable & Movable & Representable, //
    ](self: Optional[U]) -> String:
        """Return the string representation of the value of the `Optional`.

        Parameters:
            U: The type of the elements in the list. Must implement the
                traits `Representable`, `Copyable` and `Movable`.

        Returns:
            A string representation of the `Optional`.
        """
        var output = String()
        self.write_to(output)
        return output

    # TODO: Include the Parameter type in the string as well.
    fn __repr__[
        U: Representable & Copyable & Movable, //
    ](self: Optional[U]) -> String:
        """Returns the verbose string representation of the `Optional`.

        Parameters:
            U: The type of the elements in the list. Must implement the
                traits `Representable`, `Copyable` and `Movable`.

        Returns:
            A verbose string representation of the `Optional`.
        """
        var output = String()
        output.write("Optional(")
        self.write_to(output)
        output.write(")")
        return output

    @always_inline("nodebug")
    fn __merge_with__[
        other_type: __type_of(Bool),
    ](self) -> Bool:
        """Merge with other bools in an expression.

        Parameters:
            other_type: The type of the bool to merge with.

        Returns:
            A Bool after merging with the specified `other_type`.
        """
        return self.__bool__()

    fn write_to[
        U: Representable & Copyable & Movable, //
    ](self: Optional[U], mut writer: Some[Writer]):
        """Write `Optional` string representation to a `Writer`.

        Parameters:
            U: The type of the elements in the list. Must implement the
                traits `Representable`, `Copyable` and `Movable`.

        Args:
            writer: The object to write to.
        """
        if self:
            writer.write(repr(self.value()))
        else:
            writer.write("None")

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn value(ref self) -> ref [self._value] T:
        """Retrieve a reference to the value of the `Optional`.

        Returns:
            A reference to the contained data of the `Optional` as a reference.

        Notes:
            This will abort on empty `Optional`.
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
    fn unsafe_value(ref self) -> ref [self._value] T:
        """Unsafely retrieve a reference to the value of the `Optional`.

        Returns:
            A reference to the contained data of the `Optional` as a reference.

        Notes:
            This will **not** abort on empty `Optional`.
        """
        debug_assert(self.__bool__(), "`.value()` on empty `Optional`")
        return self._value.unsafe_get[T]()

    fn take(mut self) -> T:
        """Move the value out of the `Optional`.

        Returns:
            The contained data of the `Optional` as an owned T value.

        Notes:
            This will abort on empty `Optional`.
        """
        if not self.__bool__():
            abort(
                "`Optional.take()` called on empty `Optional`. Consider using"
                " `if optional:` to check whether the `Optional` is empty"
                " before calling `.take()`, or use `.or_else()` to provide a"
                " default value."
            )
        return self.unsafe_take()

    fn unsafe_take(mut self) -> T:
        """Unsafely move the value out of the `Optional`.

        Returns:
            The contained data of the `Optional` as an owned T value.

        Notes:
            This will **not** abort on empty `Optional`.
        """
        debug_assert(self.__bool__(), "`.unsafe_take()` on empty `Optional`")
        return self._value.unsafe_replace[_NoneType, T](_NoneType())

    fn or_else(self, default: T) -> T:
        """Return the underlying value contained in the `Optional` or a default
        value if the `Optional`'s underlying value is not present.

        Args:
            default: The new value to use if no value was present.

        Returns:
            The underlying value contained in the `Optional` or a default value.
        """
        if self.__bool__():
            return self._value[T].copy()
        return default.copy()

    fn copied[
        mut: Bool,
        origin: Origin[mut], //,
        T: ExplicitlyCopyable & Movable,
    ](self: Optional[Pointer[T, origin]]) -> Optional[T]:
        """Converts an `Optional` containing a Pointer to an `Optional` of an
        owned value by copying.

        Parameters:
            mut: Mutability of the pointee origin.
            origin: Origin of the contained `Pointer`.
            T: Type of the owned result value.

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


@register_passable("trivial")
struct OptionalReg[T: AnyTrivialRegType](Boolable, Defaultable):
    """A register-passable optional type.

    This struct optionally contains a value. It only works with trivial register
    passable types at the moment.

    Parameters:
        T: The type of value stored in the Optional.
    """

    # Fields
    alias _mlir_type = __mlir_type[`!kgen.variant<`, T, `, i1>`]
    var _value: Self._mlir_type

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline("builtin")
    fn __init__(out self):
        """Create an optional with a value of None."""
        self = Self(None)

    @always_inline("builtin")
    @implicit
    fn __init__(out self, value: T):
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
        other_type: __type_of(Bool),
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
    fn value(self) -> T:
        """Get the optional value.

        Returns:
            The contained value.
        """
        return __mlir_op.`kgen.variant.get`[index = Int(0)._mlir_value](
            self._value
        )

    fn or_else(var self, var default: T) -> T:
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
