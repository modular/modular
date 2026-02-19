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
"""Implements the Int class.

These are Mojo built-ins, so you don't need to import them.
"""

from collections.interval import IntervalElement
from collections.string.string import (
    _calc_initial_buffer_size_int32,
    _calc_initial_buffer_size_int64,
)
from hashlib.hasher import Hasher
from math import Ceilable, CeilDivable, Floorable, Truncable
from sys.info import is_32bit
from sys.info import bit_width_of

from builtin.device_passable import DevicePassable
from math import Absable, DivModable, Powable
from python import (
    ConvertibleFromPython,
    ConvertibleToPython,
    Python,
    PythonObject,
)

from utils._select import _select_register_value as select
from utils._visualizers import lldb_formatter_wrapping_type

# ===----------------------------------------------------------------------=== #
#  Indexer
# ===----------------------------------------------------------------------=== #


trait Indexer(ImplicitlyDestructible):
    """
    The `Indexer` trait is used for types that can index into a collection or
    pointer. The type returned is the underlying __mlir_type.index, enabling
    types like `SIMD` to not have to be converted to an `Int` first.
    """

    fn __mlir_index__(self) -> __mlir_type.index:
        """Convert to index.

        Returns:
            The corresponding __mlir_type.index value.
        """
        ...


# ===----------------------------------------------------------------------=== #
#  index
# ===----------------------------------------------------------------------=== #


@always_inline("nodebug")
fn index[T: Indexer](idx: T, /) -> Int:
    """Returns the value of `__mlir_index__` for the given value.

    Parameters:
        T: A type conforming to the `Indexer` trait.

    Args:
        idx: The value.

    Returns:
        An `__mlir_type` representing the index value.
    """
    return Int(mlir_value=idx.__mlir_index__())


# ===----------------------------------------------------------------------=== #
#  Intable
# ===----------------------------------------------------------------------=== #


trait Intable(ImplicitlyDestructible):
    """The `Intable` trait describes a type that can be converted to an Int.

    Any type that conforms to `Intable` or
    [`IntableRaising`](/mojo/std/builtin/int/IntableRaising) can construct an
    `Int`.

    This trait requires the type to implement the `__int__()` method. For
    example:

    ```mojo
    struct Foo(Intable):
        var i: Int

        fn __int__(self) -> Int:
            return self.i
    ```

    Now you can construct an `Int`:

    ```mojo
    %# from testing import assert_equal
    foo = Foo(42)
    assert_equal(Int(foo), 42)
    ```

    **Note:** If the `__int__()` method can raise an error, use the
    [`IntableRaising`](/mojo/std/builtin/int/intableraising) trait
    instead.
    """

    fn __int__(self) -> Int:
        """Get the integral representation of the value.

        Returns:
            The integral representation of the value.
        """
        ...


trait IntableRaising:
    """
    The `IntableRaising` trait describes a type can be converted to an Int, but
    the conversion might raise an error.

    Any type that conforms to [`Intable`](/mojo/std/builtin/int/Intable)
    or `IntableRaising` can construct an `Int`.

    This trait requires the type to implement the `__int__()` method, which can
    raise an error. For example:

    ```mojo
    struct Foo(IntableRaising):
        var i: Int

        fn __int__(self) raises -> Int:
            return self.i
    ```

    Now you can construct an `Int`:

    ```mojo
    %# from testing import assert_equal
    foo = Foo(42)
    assert_equal(Int(foo), 42)
    ```
    """

    fn __int__(self) raises -> Int:
        """Get the integral representation of the value.

        Returns:
            The integral representation of the type.

        Raises:
            If the type does not have an integral representation.
        """
        ...


@lldb_formatter_wrapping_type
struct Int(
    Absable,
    Boolable,
    CeilDivable,
    Ceilable,
    Comparable,
    ConvertibleFromPython,
    ConvertibleToPython,
    Defaultable,
    DevicePassable,
    DivModable,
    Floorable,
    Hashable,
    ImplicitlyCopyable,
    Indexer,
    Intable,
    IntervalElement,
    KeyElement,
    Powable,
    Representable,
    Roundable,
    Stringable,
    TrivialRegisterPassable,
    Truncable,
    Writable,
):
    """This type represents an integer value."""

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var _mlir_value: __mlir_type.index
    """The underlying storage for the integer value."""

    # ===-------------------------------------------------------------------===#
    # Aliases
    # ===-------------------------------------------------------------------===#

    comptime BITWIDTH: Int = bit_width_of[DType.int]()
    """The bit width of the integer type."""

    comptime MAX = Int(Scalar[DType.int].MAX)
    """Returns the maximum integer value."""

    comptime MIN = Int(Scalar[DType.int].MIN)
    """Returns the minimum value of type."""

    comptime device_type: AnyType = Self
    """Int is remapped to the same type when passed to accelerator devices."""

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        """Device type mapping is the identity function."""
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        """
        Gets this type's name, for use in error messages when handing arguments
        to kernels.
        TODO: This will go away soon, when we get better error messages for
        kernel calls.

        Returns:
            This type's name.
        """
        return "Int"

    # ===------------------------------------------------------------------=== #
    # Life cycle methods
    # ===------------------------------------------------------------------=== #

    @always_inline("builtin")
    fn __init__(out self):
        """Default constructor that produces zero."""
        self._mlir_value = __mlir_attr.`0 : index`

    @doc_private
    @always_inline("builtin")
    fn __init__(out self, *, mlir_value: __mlir_type.index):
        """Construct Int from the given index value.

        Args:
            mlir_value: The init value.
        """
        self._mlir_value = mlir_value

    @doc_private
    @always_inline("nodebug")
    fn __init__(out self, value: __mlir_type.`!pop.scalar<index>`):
        """Construct Int from the given Index value.

        Args:
            value: The init value.
        """
        self._mlir_value = __mlir_op.`pop.cast_to_builtin`[
            _type = __mlir_type.index
        ](value)

    @always_inline("builtin")
    @implicit
    fn __init__(out self, value: IntLiteral):
        """Construct Int from the given IntLiteral value.

        Args:
            value: The init value.
        """
        self = value.__int__()

    @always_inline("nodebug")
    fn __init__[T: Intable](out self, value: T):
        """Get the Int representation of the value.

        Parameters:
            T: The Intable type.

        Args:
            value: The object to get the integral representation of.
        """
        self = value.__int__()

    @always_inline("nodebug")
    fn __init__[T: IntableRaising](out self, value: T) raises:
        """Get the Int representation of the value.

        Parameters:
            T: The Intable type.

        Args:
            value: The object to get the integral representation of.

        Raises:
            If the type does not have an integral representation.
        """
        self = value.__int__()

    # ===------------------------------------------------------------------=== #
    # Operator dunders
    # ===------------------------------------------------------------------=== #

    @always_inline("builtin")
    fn __lt__(self, rhs: Int) -> Bool:
        """Compare this Int to the RHS using LT comparison.

        Args:
            rhs: The other Int to compare against.

        Returns:
            True if this Int is less-than the RHS Int and False otherwise.
        """
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate slt>`
        ](self._mlir_value, rhs._mlir_value)

    @always_inline("builtin")
    fn __le__(self, rhs: Int) -> Bool:
        """Compare this Int to the RHS using LE comparison.

        Args:
            rhs: The other Int to compare against.

        Returns:
            True if this Int is less-or-equal than the RHS Int and False
            otherwise.
        """
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate sle>`
        ](self._mlir_value, rhs._mlir_value)

    @always_inline("builtin")
    fn __eq__(self, rhs: Int) -> Bool:
        """Compare this Int to the RHS using EQ comparison.

        Args:
            rhs: The other Int to compare against.

        Returns:
            True if this Int is equal to the RHS Int and False otherwise.
        """
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate eq>`
        ](self._mlir_value, rhs._mlir_value)

    @always_inline("builtin")
    fn __ne__(self, rhs: Int) -> Bool:
        """Compare this Int to the RHS using NE comparison.

        Args:
            rhs: The other Int to compare against.

        Returns:
            True if this Int is non-equal to the RHS Int and False otherwise.
        """
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate ne>`
        ](self._mlir_value, rhs._mlir_value)

    @always_inline("builtin")
    fn __gt__(self, rhs: Int) -> Bool:
        """Compare this Int to the RHS using GT comparison.

        Args:
            rhs: The other Int to compare against.

        Returns:
            True if this Int is greater than the RHS Int and False otherwise.
        """
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate sgt>`
        ](self._mlir_value, rhs._mlir_value)

    @always_inline("builtin")
    fn __ge__(self, rhs: Int) -> Bool:
        """Compare this Int to the RHS using GE comparison.

        Args:
            rhs: The other Int to compare against.

        Returns:
            True if this Int is greater-or-equal than the RHS Int and False
            otherwise.
        """
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate sge>`
        ](self._mlir_value, rhs._mlir_value)

    @always_inline("builtin")
    fn __pos__(self) -> Int:
        """Return +self.

        Returns:
            The +self value.
        """
        return self

    @always_inline("builtin")
    fn __neg__(self) -> Int:
        """Return -self.

        Returns:
            The -self value.
        """
        return self * -1

    @always_inline("builtin")
    fn __invert__(self) -> Int:
        """Return ~self.

        Returns:
            The ~self value.
        """
        return self ^ -1

    @always_inline("builtin")
    fn __add__(self, rhs: Int) -> Int:
        """Return `self + rhs`.

        Args:
            rhs: The value to add.

        Returns:
            `self + rhs` value.
        """
        return Int(
            mlir_value=__mlir_op.`index.add`(self._mlir_value, rhs._mlir_value)
        )

    @always_inline("builtin")
    fn __sub__(self, rhs: Int) -> Int:
        """Return `self - rhs`.

        Args:
            rhs: The value to subtract.

        Returns:
            `self - rhs` value.
        """
        return Int(
            mlir_value=__mlir_op.`index.sub`(self._mlir_value, rhs._mlir_value)
        )

    @always_inline("builtin")
    fn __mul__(self, rhs: Int) -> Int:
        """Return `self * rhs`.

        Args:
            rhs: The value to multiply with.

        Returns:
            `self * rhs` value.
        """
        return Int(
            mlir_value=__mlir_op.`index.mul`(self._mlir_value, rhs._mlir_value)
        )

    @always_inline("builtin")
    fn __truediv__(self, rhs: Int) -> Self:
        """Return the result of the division of `self` and `rhs`.

        Performs truncating division (toward zero) for integers.

        Args:
            rhs: The value to divide on.

        Returns:
            `self / rhs` value.
        """
        return Int(
            mlir_value=__mlir_op.`index.divs`(self._mlir_value, rhs._mlir_value)
        )

    @always_inline("nodebug")
    fn __floordiv__(self, rhs: Int) -> Int:
        """Return the division of `self` and `rhs` rounded down to the nearest
        integer.

        Args:
            rhs: The value to divide on.

        Returns:
            `floor(self/rhs)` value.
        """
        # This should raise an exception
        var denom = select(rhs == 0, 1, rhs)
        var div = self / denom
        var rem = self._positive_rem(denom)
        var res = select(((rhs < 0) ^ (self < 0)) & (rem != 0), div - 1, div)
        return select(rhs == 0, 0, res)

    @always_inline("nodebug")
    fn __mod__(self, rhs: Int) -> Int:
        """Return the remainder of self divided by rhs.

        Args:
            rhs: The value to divide on.

        Returns:
            The remainder of dividing self by rhs.
        """
        # this should raise an exception
        var denom = select(rhs == 0, 1, rhs)
        var rem = self._positive_rem(denom)
        var res = select(((rhs < 0) ^ (self < 0)) & (rem != 0), rem + rhs, rem)
        return select(rhs == 0, 0, res)

    @always_inline("nodebug")
    fn __divmod__(self, rhs: Int) -> Tuple[Int, Int]:
        """Computes both the quotient and remainder using integer division.

        Args:
            rhs: The value to divide on.

        Returns:
            The quotient and remainder as a tuple `(self // rhs, self % rhs)`.
        """
        # this should raise an exception
        var denom = select(rhs == 0, 1, rhs)
        var div = self / denom
        var rem = self._positive_rem(denom)
        var neg = ((rhs < 0) ^ (self < 0)) & Bool(rem)
        div = select(neg, div - 1, div)
        mod = select(neg, rem + rhs, rem)
        return select(rhs == 0, 0, div), select(rhs == 0, 0, mod)

    @always_inline("nodebug")
    fn __pow__(self, exp: Self) -> Self:
        """Return the value raised to the power of the given exponent.

        Computes the power of an integer using the Russian Peasant Method.

        Args:
            exp: The exponent value.

        Returns:
            The value of `self` raised to the power of `exp`.
        """
        if exp < 0:
            # Not defined for Integers, this should raise an
            # exception.
            return 0
        var res: Int = 1
        var x = self
        var n = exp
        while n > 0:
            if n & 1 != 0:
                res *= x
            x *= x
            n >>= 1
        return res

    @always_inline("builtin")
    fn __lshift__(self, rhs: Int) -> Int:
        """Return `self << rhs`.

        Args:
            rhs: The value to shift with.

        Returns:
            `self << rhs`.
        """
        return select(
            rhs < 0,
            0,
            Int(
                mlir_value=__mlir_op.`index.shl`(
                    self._mlir_value, rhs._mlir_value
                )
            ),
        )

    @always_inline("builtin")
    fn __rshift__(self, rhs: Int) -> Int:
        """Return `self >> rhs`.

        Args:
            rhs: The value to shift with.

        Returns:
            `self >> rhs`.
        """
        return select(
            rhs < 0,
            0,
            Int(
                mlir_value=__mlir_op.`index.shrs`(
                    self._mlir_value, rhs._mlir_value
                )
            ),
        )

    @always_inline("builtin")
    fn __and__(self, rhs: Int) -> Int:
        """Return `self & rhs`.

        Args:
            rhs: The RHS value.

        Returns:
            `self & rhs`.
        """
        return Int(
            mlir_value=__mlir_op.`index.and`(self._mlir_value, rhs._mlir_value)
        )

    @always_inline("builtin")
    fn __xor__(self, rhs: Int) -> Int:
        """Return `self ^ rhs`.

        Args:
            rhs: The RHS value.

        Returns:
            `self ^ rhs`.
        """
        return Int(
            mlir_value=__mlir_op.`index.xor`(self._mlir_value, rhs._mlir_value)
        )

    @always_inline("builtin")
    fn __or__(self, rhs: Int) -> Int:
        """Return `self | rhs`.

        Args:
            rhs: The RHS value.

        Returns:
            `self | rhs`.
        """
        return Int(
            mlir_value=__mlir_op.`index.or`(self._mlir_value, rhs._mlir_value)
        )

    # ===-------------------------------------------------------------------===#
    # In place operations.
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __iadd__(mut self, rhs: Int):
        """Compute `self + rhs` and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self + rhs

    @always_inline("nodebug")
    fn __isub__(mut self, rhs: Int):
        """Compute `self - rhs` and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self - rhs

    @always_inline("nodebug")
    fn __imul__(mut self, rhs: Int):
        """Compute self*rhs and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self * rhs

    @always_inline("nodebug")
    fn __itruediv__(mut self, rhs: Int):
        """Compute `self / rhs`, convert to int, and save the result in self.

        Since `floor(self / rhs)` is equivalent to `self // rhs`, this yields
        the same as `__ifloordiv__`.

        Args:
            rhs: The RHS value.
        """
        self = self // rhs

    @always_inline("nodebug")
    fn __ifloordiv__(mut self, rhs: Int):
        """Compute `self // rhs` and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self // rhs

    @always_inline("nodebug")
    fn __imod__(mut self, rhs: Int):
        """Compute `self % rhs` and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self % rhs

    @always_inline("nodebug")
    fn __ipow__(mut self, rhs: Int):
        """Compute `pow(self, rhs)` and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self**rhs

    @always_inline("nodebug")
    fn __ilshift__(mut self, rhs: Int):
        """Compute `self << rhs` and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self << rhs

    @always_inline("nodebug")
    fn __irshift__(mut self, rhs: Int):
        """Compute `self >> rhs` and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self >> rhs

    @always_inline("nodebug")
    fn __iand__(mut self, rhs: Int):
        """Compute `self & rhs` and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self & rhs

    @always_inline("nodebug")
    fn __ixor__(mut self, rhs: Int):
        """Compute `self ^ rhs` and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self ^ rhs

    @always_inline("nodebug")
    fn __ior__(mut self, rhs: Int):
        """Compute self|rhs and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self | rhs

    # ===-------------------------------------------------------------------===#
    # Reversed operations
    # ===-------------------------------------------------------------------===#

    @always_inline("builtin")
    fn __radd__(self, value: Int) -> Int:
        """Return `value + self`.

        Args:
            value: The other value.

        Returns:
            `value + self`.
        """
        return self + value

    @always_inline("builtin")
    fn __rsub__(self, value: Int) -> Int:
        """Return `value - self`.

        Args:
            value: The other value.

        Returns:
            `value - self`.
        """
        return value - self

    @always_inline("builtin")
    fn __rmul__(self, value: Int) -> Int:
        """Return `value * self`.

        Args:
            value: The other value.

        Returns:
            `value * self`.
        """
        return self * value

    @always_inline("nodebug")
    fn __rfloordiv__(self, value: Int) -> Int:
        """Return `value // self`.

        Args:
            value: The other value.

        Returns:
            `value // self`.
        """
        return value // self

    @always_inline("nodebug")
    fn __rmod__(self, value: Int) -> Int:
        """Return `value % self`.

        Args:
            value: The other value.

        Returns:
            `value % self`.
        """
        return value % self

    @always_inline("nodebug")
    fn __rpow__(self, value: Int) -> Int:
        """Return `pow(value,self)`.

        Args:
            value: The other value.

        Returns:
            `pow(value,self)`.
        """
        return value**self

    @always_inline("builtin")
    fn __rlshift__(self, value: Int) -> Int:
        """Return `value << self`.

        Args:
            value: The other value.

        Returns:
            `value << self`.
        """
        return value << self

    @always_inline("builtin")
    fn __rrshift__(self, value: Int) -> Int:
        """Return `value >> self`.

        Args:
            value: The other value.

        Returns:
            `value >> self`.
        """
        return value >> self

    @always_inline("builtin")
    fn __rand__(self, value: Int) -> Int:
        """Return `value & self`.

        Args:
            value: The other value.

        Returns:
            `value & self`.
        """
        return value & self

    @always_inline("builtin")
    fn __ror__(self, value: Int) -> Int:
        """Return `value | self`.

        Args:
            value: The other value.

        Returns:
            `value | self`.
        """
        return value | self

    @always_inline("builtin")
    fn __rxor__(self, value: Int) -> Int:
        """Return `value ^ self`.

        Args:
            value: The other value.

        Returns:
            `value ^ self`.
        """
        return value ^ self

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline("builtin")
    fn __bool__(self) -> Bool:
        """Convert this Int to Bool.

        Returns:
            False Bool value if the value is equal to 0 and True otherwise.
        """
        return self != 0

    @always_inline("builtin")
    fn __mlir_index__(self) -> __mlir_type.index:
        """Convert to index.

        Returns:
            The corresponding __mlir_type.index value.
        """
        return self._mlir_value

    @always_inline("builtin")
    fn __int__(self) -> Int:
        """Gets the integral value (this is an identity function for Int).

        Returns:
            The value as an integer.
        """
        return self

    @always_inline("builtin")
    fn __abs__(self) -> Self:
        """Return the absolute value of the Int value.

        The absolute value of Int.MIN is Int.MIN.

        Returns:
            The absolute value.
        """
        return select(self < 0, -self, self)

    @always_inline("builtin")
    fn __ceil__(self) -> Self:
        """Return the ceiling of the Int value, which is itself.

        Returns:
            The Int value itself.
        """
        return self

    @always_inline("builtin")
    fn __floor__(self) -> Self:
        """Return the floor of the Int value, which is itself.

        Returns:
            The Int value itself.
        """
        return self

    @always_inline("builtin")
    fn __round__(self) -> Self:
        """Return the rounded value of the Int value, which is itself.

        Returns:
            The Int value itself.
        """
        return self

    @always_inline("nodebug")
    fn __round__(self, ndigits: Int) -> Self:
        """Return the rounded value of the Int value, which is itself.

        Args:
            ndigits: The number of digits to round to.

        Returns:
            The Int value itself if ndigits >= 0 else the rounded value.
        """
        if ndigits >= 0:
            return self
        return self - (self % 10 ** -(ndigits))

    @always_inline("builtin")
    fn __trunc__(self) -> Self:
        """Return the truncated Int value, which is itself.

        Returns:
            The Int value itself.
        """
        return self

    @always_inline("nodebug")
    fn __ceildiv__(self, denominator: Self) -> Self:
        """Return the rounded-up result of dividing self by denominator.


        Args:
            denominator: The denominator.

        Returns:
            The ceiling of dividing numerator by denominator.
        """
        return -(self // -denominator)

    @always_inline("builtin")
    fn is_power_of_two(self) -> Bool:
        """Check if the integer is a (non-zero) power of two.

        Returns:
            True if the integer is a power of two, False otherwise.
        """
        return (self & (self - 1) == 0) & (self > 0)

    fn write_to(self, mut writer: Some[Writer]):
        """Formats this integer to the provided Writer.

        Args:
            writer: The object to write to.
        """

        writer.write(Int64(self))

    fn write_repr_to(self, mut writer: Some[Writer]):
        """Write the string representation of the Int".

        Args:
            writer: The value to write to.
        """
        writer.write("Int(", self, ")")

    fn write_padded[W: Writer](self, mut writer: W, width: Int):
        """Write the int right-aligned to a set padding.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
            width: The amount to pad to the left.
        """
        var int_width = self._decimal_digit_count() + (1 if self < 0 else 0)

        # TODO: Assumes user wants right-aligned content.
        if int_width < width:
            for _ in range(width - int_width):
                writer.write(" ")

        writer.write(self)

    @no_inline
    fn __str__(self) -> String:
        """Get the integer as a string.

        Returns:
            A string representation.
        """

        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        """Get the integer as a string. Returns the same `String` as `__str__`.

        Returns:
            A string representation.
        """
        return String(self)

    fn __hash__[H: Hasher](self, mut hasher: H):
        """Updates hasher with this int value.

        Parameters:
            H: The hasher type.

        Args:
            hasher: The hasher instance.
        """
        hasher._update_with_simd(Int64(self))

    @doc_private
    fn __init__(out self, *, py: PythonObject) raises:
        """Construct an `Int` from a PythonObject.

        Args:
            py: The Python object to convert from.

        Raises:
            An error if the conversion failed.
        """
        self = Python.py_long_as_ssize_t(py.__int__())

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn to_python_object(var self) raises -> PythonObject:
        """Convert this value to a PythonObject.

        Returns:
            A PythonObject representing the value.

        Raises:
            If the Python runtime is not initialized or conversion fails.
        """
        return PythonObject(self)

    @always_inline("builtin")
    fn _positive_rem(self, rhs: Int) -> Int:
        """Return the modulus of `self` and `rhs` assuming that the arguments
        are both positive.

        Args:
            rhs: The value to divide on.

        Returns:
            The integer modulus of `self` and `rhs` .
        """
        return Int(
            mlir_value=__mlir_op.`index.rems`(self._mlir_value, rhs._mlir_value)
        )

    fn _decimal_digit_count(self) -> Int:
        """
        Returns the number of decimal digits required to display this integer.

        Note that if this integer is negative, the returned count does not
        include space to store a leading minus character.

        Returns:
            A count of the number of decimal digits required to display this integer.

        Examples:

        ```mojo
        %# from testing import assert_equal
        assert_equal(Int(10)._decimal_digit_count(), 2)
        assert_equal(Int(-10)._decimal_digit_count(), 2)
        ```
        """

        var n = abs(self)

        @parameter
        if is_32bit():
            return _calc_initial_buffer_size_int32(n)

        # The value only has low-bits.
        if n >> 32 == 0:
            return _calc_initial_buffer_size_int32(n)

        return _calc_initial_buffer_size_int64(UInt64(n))
