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
"""Defines Result, a type modeling a value which may or may not be present.
With an Error in the case of failure.
Result values can be thought of as a type-safe nullable pattern.
Your value can take on a value or `None`, and you need to check
and explicitly extract the value to get it out.
Examples:
```mojo
from collections.result import Result
var a = Result(1)
var b = Result[Int]()
if a:
    print(a.value())  # prints 1
if b:  # bool(b) is False, so no print
    print(b.value())
var c = a.or_else(2)
var d = b.or_else(2)
print(c)  # prints 1
print(d)  # prints 2
```
And if more information about the returned Error is wanted it is available.
```mojo
from collections.result import Result
var a = Result(1)
var b = Result[Int](err=Error("something went wrong"))
var c = Result[Int](None, Error("error 1"))
var d = Result[Int](err=Error("error 2"))
if a:
    print(a.err)  # prints ""
if not b:
    print(b.err) # prints "something went wrong"
if c.err:
    print("c had an error")
# TODO: pattern matching
if String(d.err) == "error 1":
    print("d had error 1")
elif String(d.err) == "error 2":
    print("d had error 2")
```
A Result with an Error can also be retuned early:
```mojo
fn func_that_can_err[A: Copyable]() -> Result[A]:
    ...

fn return_early_if_err[T: Copyable, A: Copyable]() -> Result[T]:
    var result: Result[A] = func_that_can_err[A]()
    if not result:
        # the internal err gets transferred to a Result[T]
        return result
        # its also possible to do:
        # return None, Error("func_that_can_err failed")
    var val = result.value()
    var final_result: T
    ...
    return final_result
```
A Result can be unwrapped within a raising function:
```mojo
fn func_that_can_err[A: Copyable]() -> Result[A]:
    ...

fn return_early_if_err[T: Copyable, A: Copyable]() raises  -> T:
    var val = func_that_can_err[A]()[]
    var final_result: T
    ...
    return final_result
```
.
"""


from os import abort

from utils import Variant

from builtin.constrained import _constrained_conforms_to
from builtin.device_passable import DevicePassable
from compile import get_type_name
from memory import LegacyOpaquePointer as OpaquePointer

from sys.intrinsics import _type_is_eq, _type_is_eq_parse_time

# ===-----------------------------------------------------------------------===#
# Result
# ===-----------------------------------------------------------------------===#


struct Result[
    T: Copyable & Movable,
    E: Writable & Copyable & Movable & Defaultable = Error,
](
    Boolable,
    Defaultable,
    ImplicitlyCopyable,
    Iterable,
    Iterator,
    Movable,
    Representable,
    Stringable,
    Writable,
):
    """A type modeling a value which may or may not be present.
    With an Error in the case of failure.
    Result values can be thought of as a type-safe nullable pattern.
    Your value can take on a value or `None`, and you need to check
    and explicitly extract the value to get it out.
    Currently T is required to be a `Copyable` so we can implement
    copy/move for Result and allow it to be used in collections itself.

    Parameters:
        T: The type of value stored in the `Result`.
        E: The error type of the Result.
    """

    # Iterator aliases
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[iterable_mut]
    ]: Iterator = Self
    comptime Element = Self.T

    comptime _container_type = InlineArray[Self.T, 1]
    comptime _type = Variant[Self._container_type, Self.E]
    var _value: Self._type

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __init__(out self):
        """Construct an empty `Result`."""
        self._value = {Self._container_type(uninitialized=True)}

    @implicit
    fn __init__(out self, var value: Self.T):
        """Construct an `Result` containing a value.

        Args:
            value: The value to store in the `Result`.
        """
        self._value = {Self._container_type(value^)}

    @implicit
    fn __init__(out self, var err: Self.E):
        """Construct an `Result` containing a value.

        Args:
            err: The error to store in the `Result`.
        """
        self._value = {err^}

    # TODO(MSTDL-715):
    #   This initializer should not be necessary, we should need
    #   only the initializer from a `NoneType`.
    @doc_private
    @implicit
    fn __init__(out self, value: NoneType._mlir_type):
        """Construct an empty `Result`.

        Args:
            value: Must be exactly `None`.
        """
        self = Self()

    @implicit
    fn __init__(out self, value: NoneType):
        """Construct an empty `Result`.

        Args:
            value: Must be exactly `None`.
        """
        self = Self()

    @implicit
    fn __init__(out self, var value: Tuple[NoneType, Self.E]):
        """Construct an empty `Result`.

        Args:
            value: Must be exactly `None`.
        """
        self = Self(value[1])

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    fn __is__(self, other: NoneType) -> Bool:
        """Return `True` if the Result has no value.

        Args:
            other: The value to compare to (None).

        Returns:
            True if the Result has no value and False otherwise.

        Notes:
            It allows you to use the following syntax:
            `if my_Result is None:`.
        """
        return not self.__bool__()

    fn __isnot__(self, other: NoneType) -> Bool:
        """Return `True` if the Result has a value.

        Args:
            other: The value to compare to (None).

        Returns:
            True if the Result has a value and False otherwise.

        Notes:
            It allows you to use the following syntax:
            `if my_Result is not None:`.
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
        _T: Equatable & Copyable & Movable
    ](self: Result[_T], rhs: Result[_T]) -> Bool:
        """Return `True` if this is the same as another `Result` value,
        meaning both are absent, or both are present and have the same
        underlying value.

        Parameters:
            _T: The type of the elements in the list. Must implement the
                traits `Copyable`, `Movable` and `Equatable`.

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
        _T: Equatable & Copyable & Movable, //
    ](self: Result[_T], rhs: Result[_T]) -> Bool:
        """Return `False` if this is the same as another `Result` value,
        meaning both are absent, or both are present and have the same
        underlying value.

        Parameters:
            _T: The type of the elements in the list. Must implement the
                traits `Copyable`, `Movable` and `Equatable`.

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
        """Iterate over the Result's possibly contained value.

        Results act as a collection of size 0 or 1.

        Returns:
            An iterator over the Result's value (if present).
        """
        return self.copy()

    @always_inline
    fn __has_next__(self) -> Bool:
        """Return true if the Result has a value.

        Returns:
            True if the Result contains a value, False otherwise.
        """
        return self.__bool__()

    @always_inline
    fn __next__(mut self) -> Self.Element:
        """Return the contained value of the Result.

        Returns:
            The value contained in the Result.
        """
        return self.take()

    @always_inline
    fn bounds(self) -> Tuple[Int, Result[Int]]:
        """Return the bounds of the `Result`, which is 0 or 1.

        Returns:
            A tuple containing the length (0 or 1) and an `Result` containing the length.
        """
        var len = 1 if self else 0
        return (len, {len})

    fn __bool__(self) -> Bool:
        """Return true if the Result has a value.

        Returns:
            True if the `Result` has a value and False otherwise.
        """
        return not self._value.isa[Self.E]()

    fn __invert__(self) -> Bool:
        """Return False if the `Result` has a value.

        Returns:
            False if the `Result` has a value and True otherwise.
        """
        return not self

    @always_inline
    fn __getitem__(ref self) raises -> ref [self._value] Self.T:
        """Retrieve a reference to the value inside the `Result`.

        Returns:
            A reference to the value inside the `Result`.

        Raises:
            on `Result` with an error.
        """
        if not self:
            raise Error(".value() on empty Result")
        return self.unsafe_value()

    fn __str__(self: Self) -> String:
        """Return the string representation of the value of the `Result`.

        Returns:
            A string representation of the `Result`.
        """
        _constrained_conforms_to[
            conforms_to(Self.T, Stringable),
            Parent=Self,
            Element = Self.T,
            ParentConformsTo="Stringable",
        ]()

        if self:
            return trait_downcast[Stringable](self.value()).__str__()
        else:
            return "None"

    fn __repr__(self: Self) -> String:
        """Returns the verbose string representation of the `Result`.

        Returns:
            A verbose string representation of the `Result`.
        """
        _constrained_conforms_to[
            conforms_to(Self.T, Representable),
            Parent=Self,
            Element = Self.T,
            ParentConformsTo="Representable",
        ]()

        var output = String()
        output.write("Result(", self, ")")
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

    fn write_to(self: Self, mut writer: Some[Writer]):
        """Write `Result` string representation to a `Writer`.

        Args:
            writer: The object to write to.
        """
        _constrained_conforms_to[
            conforms_to(Self.T, Representable),
            Parent=Self,
            Element = Self.T,
            ParentConformsTo="Representable",
        ]()

        if self:
            writer.write(trait_downcast[Representable](self.value()).__repr__())
        else:
            writer.write("None")

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn value[
        V: Movable = Self.T
    ](ref self) -> ref [self._value] V where (
        _type_is_eq_parse_time[V, Self.T]()
        or _type_is_eq_parse_time[V, Self.E]()
    ):
        """Retrieve a reference to the value of the `Result`.

        Returns:
            A reference to the contained data of the `Result` as a reference.

        Notes:
            This will abort on `Result` with an error.
        """

        @parameter
        if _type_is_eq[V, Self.T]():
            if not self.__bool__():
                abort(
                    "`Result.value()` called on `Result` with an error."
                    " Consider using `if Result:` to check whether the `Result`"
                    " is empty before calling `.value()`, or use `.or_else()`"
                    " to provide a default value."
                )

        return self.unsafe_value[V]()

    @always_inline
    fn unsafe_value[
        V: Movable = Self.T
    ](ref self) -> ref [self._value] V where (
        _type_is_eq_parse_time[V, Self.T]()
        or _type_is_eq_parse_time[V, Self.E]()
    ):
        """Unsafely retrieve a reference to the value of the `Result`.

        Returns:
            A reference to the contained data of the `Result` as a reference.

        Notes:
            This will **not** abort on `Result` with an error.
        """

        @parameter
        if _type_is_eq[V, Self.T]():
            debug_assert(
                self.__bool__(), "`.value()` on `Result` with an error"
            )
            return rebind[V](self._value.unsafe_get[Self._container_type]()[0])
        else:
            return rebind[V](self._value.unsafe_get[Self._container_type]()[0])

    fn take[
        V: Movable = Self.T
    ](mut self) -> V where (
        _type_is_eq_parse_time[V, Self.T]()
        or _type_is_eq_parse_time[V, Self.E]()
    ):
        """Move the value out of the `Result`.

        Returns:
            The contained data of the `Result` as an owned T value.

        Notes:
            This will abort when trying to take a value from a `Result`
            with an error.
        """

        @parameter
        if _type_is_eq[V, Self.T]():
            if not self.__bool__():
                abort(
                    "`Result.take()` called trying to take a value from a"
                    "`Result` with an error. Consider"
                    " using `if Result:` to check whether the `Result` is empty"
                    " before calling `.take()`, or use `.or_else()` to provide"
                    " a default value."
                )
        return self.unsafe_take[V]()

    fn unsafe_take[
        V: Movable = Self.T
    ](mut self) -> V where (
        _type_is_eq_parse_time[V, Self.T]()
        or _type_is_eq_parse_time[V, Self.E]()
    ):
        """Unsafely move the value out of the `Result`.

        Returns:
            The contained data of the `Result` as an owned T value.

        Notes:
            This will **not** abort when trying to take a value from a `Result`
            with an error.
        """

        @parameter
        if _type_is_eq[V, Self.T]():
            debug_assert(
                self.__bool__(),
                "`.unsafe_take()` on a value from a `Result` with an error",
            )
            return rebind_var[V](
                UnsafePointer(
                    to=self._value.unsafe_replace[Self.E, Self._container_type](
                        {}
                    )
                )
                .unsafe_mut_cast[True]()
                .bitcast[Self.T]()
                .take_pointee()
            )
        else:
            return rebind_var[V](
                UnsafePointer(to=self._value.unsafe_replace[Self.E, Self.E]({}))
                .unsafe_mut_cast[True]()
                .bitcast[Self.T]()
                .take_pointee()
            )

    fn or_else(deinit self, var default: Self.T) -> Self.T:
        """Return the underlying value contained in the `Result` or a default
        value if the `Result`'s underlying value is not present.

        Args:
            default: The new value to use if no value was present.

        Returns:
            The underlying value contained in the `Result` or a default value.
        """
        if self:
            return self.unsafe_take()
        return default^

    fn copied[
        mut: Bool,
        origin: Origin[mut], //,
        _T: Copyable & Movable,
    ](self: Result[Pointer[_T, origin]]) -> Result[_T]:
        """Converts an `Result` containing a Pointer to an `Result` of an
        owned value by copying.

        Parameters:
            mut: Mutability of the pointee origin.
            origin: Origin of the contained `Pointer`.
            _T: Type of the owned result value.

        Returns:
            An `Result` containing an owned copy of the pointee value.

        Examples:

        Copy the value of an `Result[Pointer[_]]`

        ```mojo
        var data = "foo"
        var opt = Result(Pointer(to=data))
        var opt_owned: Result[String] = opt.copied()
        ```

        Notes:
            If `self` is an empty `Result`, the returned `Result` will be
            empty as well.
        """
        if self:
            # SAFETY: We just checked that `self` is populated.
            # Perform an implicit copy
            return self.unsafe_value()[].copy()
        else:
            return None
