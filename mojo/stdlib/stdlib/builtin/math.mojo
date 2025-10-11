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
"""Defines basic math functions for use in the open source parts of the standard
library since the `math` package is currently closed source and cannot be
depended on in the open source parts of the standard library.

These are Mojo built-ins, so you don't need to import them.
"""

# ===----------------------------------------------------------------------=== #
# abs
# ===----------------------------------------------------------------------=== #


trait Absable:
    """
    The `Absable` trait describes a type that defines an absolute value
    operation.

    Types that conform to `Absable` will work with the builtin `abs` function.
    The absolute value operation always returns the same type as the input.

    For example:
    ```mojo
    struct Point(Absable):
        var x: Float64
        var y: Float64

        fn __abs__(self) -> Self:
            return sqrt(self.x * self.x + self.y * self.y)
    ```
    """

    # TODO(MOCO-333): Reconsider the signature when we have parametric traits or
    # associated types.
    fn __abs__(self) -> Self:
        """Get the absolute value of this instance.

        Returns:
            The absolute value of the instance.
        """
        ...


@always_inline
fn abs[T: Absable](value: T) -> T:
    """Get the absolute value of the given object.

    Parameters:
        T: The type conforming to Absable.

    Args:
        value: The object to get the absolute value of.

    Returns:
        The absolute value of the object.
    """
    return value.__abs__()


# ===----------------------------------------------------------------------=== #
# divmod
# ===----------------------------------------------------------------------=== #


trait DivModable(ImplicitlyCopyable, Movable):
    """
    The `DivModable` trait describes a type that defines division and
    modulo operations returning both quotient and remainder.

    Types that conform to `DivModable` will work with the builtin `divmod` function,
    which will return the same type as the inputs.

    For example:
    ```mojo
    @fieldwise_init
    struct Bytes(DivModable):
        var size: Int

        fn __divmod__(self, other: Self) -> Tuple[Self, Self]:
            var quotient_int = self.size // other.size
            var remainder_int = self.size % other.size
            return (Bytes(quotient_int), Bytes(remainder_int))
    ```
    """

    fn __divmod__(self, denominator: Self) -> Tuple[Self, Self]:
        """Performs division and returns the quotient and the remainder.

        Returns:
            A `Tuple` containing the quotient and the remainder.
        """
        ...


fn divmod[T: DivModable](numerator: T, denominator: T) -> Tuple[T, T]:
    """Performs division and returns the quotient and the remainder.

    Parameters:
        T: A type conforming to the `DivModable` trait.

    Args:
        numerator: The dividend.
        denominator: The divisor.

    Returns:
        A `Tuple` containing the quotient and the remainder.
    """
    return numerator.__divmod__(denominator)


# ===----------------------------------------------------------------------=== #
# max
# ===----------------------------------------------------------------------=== #


@always_inline("nodebug")
fn max(x: Int, y: Int, /) -> Int:
    """Gets the maximum of two integers.

    Args:
        x: Integer input to max.
        y: Integer input to max.

    Returns:
        Maximum of x and y.
    """
    return Int(mlir_value=__mlir_op.`index.maxs`(x._mlir_value, y._mlir_value))


@always_inline("nodebug")
fn max(x: UInt, y: UInt, /) -> UInt:
    """Gets the maximum of two integers.

    Args:
        x: Integer input to max.
        y: Integer input to max.

    Returns:
        Maximum of x and y.
    """
    return UInt(mlir_value=__mlir_op.`index.maxu`(x._mlir_value, y._mlir_value))


@always_inline("nodebug")
fn max[dtype: DType, //](x: SIMD[dtype, _], y: __type_of(x), /) -> __type_of(x):
    """Performs elementwise maximum of x and y.

    An element of the result SIMD vector will be the maximum of the
    corresponding elements in x and y.

    Constraints:
        The type of the inputs must be numeric or boolean.

    Parameters:
            dtype: The data type of the SIMD vector.

    Args:
        x: First SIMD vector.
        y: Second SIMD vector.

    Returns:
        A SIMD vector containing the elementwise maximum of x and y.
    """

    constrained[
        x.dtype is DType.bool or x.dtype.is_numeric(),
        "the SIMD type must be numeric or boolean",
    ]()

    return {mlir_value = __mlir_op.`pop.max`(x._mlir_value, y._mlir_value)}


@always_inline
fn max[T: Copyable & GreaterThanComparable](x: T, *ys: T) -> T:
    """Gets the maximum value from a sequence of values.

    Parameters:
        T: A type that is both copyable and comparable with greater than.

    Args:
        x: The first value to compare.
        ys: Zero or more additional values to compare.

    Returns:
        The maximum value from the input sequence.
    """
    var res = x.copy()
    for y in ys:
        if y > res:
            res = y.copy()
    return res.copy()


# ===----------------------------------------------------------------------=== #
# min
# ===----------------------------------------------------------------------=== #


@always_inline("nodebug")
fn min(x: Int, y: Int, /) -> Int:
    """Gets the minimum of two integers.

    Args:
        x: Integer input to min.
        y: Integer input to min.

    Returns:
        Minimum of x and y.
    """
    return Int(mlir_value=__mlir_op.`index.mins`(x._mlir_value, y._mlir_value))


@always_inline("nodebug")
fn min(x: UInt, y: UInt, /) -> UInt:
    """Gets the minimum of two integers.

    Args:
        x: Integer input to min.
        y: Integer input to min.

    Returns:
        Minimum of x and y.
    """
    return UInt(mlir_value=__mlir_op.`index.minu`(x._mlir_value, y._mlir_value))


@always_inline("nodebug")
fn min[dtype: DType, //](x: SIMD[dtype, _], y: __type_of(x), /) -> __type_of(x):
    """Gets the elementwise minimum of x and y.

    An element of the result SIMD vector will be the minimum of the
    corresponding elements in x and y.

    Constraints:
        The type of the inputs must be numeric or boolean.

    Parameters:
         dtype: The data type of the SIMD vector.

    Args:
        x: First SIMD vector.
        y: Second SIMD vector.

    Returns:
        A SIMD vector containing the elementwise minimum of x and y.
    """

    constrained[
        x.dtype is DType.bool or x.dtype.is_numeric(),
        "the SIMD type must be numeric or boolean",
    ]()

    return {mlir_value = __mlir_op.`pop.min`(x._mlir_value, y._mlir_value)}


@always_inline
fn min[T: Copyable & LessThanComparable](x: T, *ys: T) -> T:
    """Gets the minimum value from a sequence of values.

    Parameters:
        T: A type that is both copyable and comparable with less than.

    Args:
        x: The first value to compare.
        ys: Zero or more additional values to compare.

    Returns:
        The minimum value from the input sequence.
    """
    var res = x.copy()
    for y in ys:
        if y < res:
            res = y.copy()
    return res.copy()


# ===----------------------------------------------------------------------=== #
# pow
# ===----------------------------------------------------------------------=== #


trait Powable:
    """
    The `Powable` trait describes a type that defines a power operation (i.e.
    exponentiation) with the same base and exponent types.

    Types that conform to `Powable` will work with the builtin `pow` function,
    which will return the same type as the inputs.

    For example:
    ```mojo
    struct Rational(Powable):
        var numerator: Float64
        var denominator: Float64

        fn __init__(out self, numerator: Float64, denominator: Float64):
            self.numerator = numerator
            self.denominator = denominator

        fn __pow__(self, exp: Self)  -> Self:
            var exp_value = exp.numerator / exp.denominator
            return Self(pow(self.numerator, exp_value), pow(self.denominator, exp_value))
    ```

    You can now use the ** operator to exponentiate objects
    inside generic functions:

    ```mojo
    fn exponentiate[T: Powable](base: T, exp: T) -> T:
        return base ** exp

    var base = Rational(Float64(3.0), 5.0)
    var exp = Rational(Float64(1.0), 2.0)
    var res = exponentiate(base, exp)
    ```

    ```plaintext
    raising to power
    ```
    """

    # TODO(MOCO-333): Reconsider the signature when we have parametric traits or
    # associated types.
    fn __pow__(self, exp: Self) -> Self:
        """Return the value raised to the power of the given exponent.

        Args:
            exp: The exponent value.

        Returns:
            The value of `self` raised to the power of `exp`.
        """
        ...


fn pow[T: Powable](base: T, exp: T) -> T:
    """Computes the `base` raised to the power of the `exp`.

    Parameters:
        T: A type conforming to the `Powable` trait.

    Args:
        base: The base of the power operation.
        exp: The exponent of the power operation.

    Returns:
        The `base` raised to the power of the `exp`.
    """
    return base.__pow__(exp)


fn pow(base: SIMD, exp: Int) -> __type_of(base):
    """Computes elementwise value of a SIMD vector raised to the power of the
    given integer.

    Args:
        base: The first input argument.
        exp: The second input argument.

    Returns:
        The `base` elementwise raised raised to the power of `exp`.
    """
    return base.__pow__(exp)


# ===----------------------------------------------------------------------=== #
# round
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct RoundMode(Copyable, EqualityComparable, Movable):
    """The `RoundMode` to use for the operation."""

    alias Up = RoundMode(0)
    """Round towards positive infinity."""
    alias Down = RoundMode(1)
    """Round towards negative infinity."""
    alias ToEven = RoundMode(2)
    """Round towards the even number."""
    alias ToZero = RoundMode(3)
    """Round towards zero."""
    alias HalfUp = RoundMode(4)
    """When in the middle, round towards positive infinity."""
    alias HalfDown = RoundMode(5)
    """When in the middle, round towards negative infinity."""
    alias HalfToEven = RoundMode(6)
    """When in the middle, round towards the even number."""
    alias HalfToZero = RoundMode(7)
    """When in the middle, round towards zero."""

    var _value: UInt

    @always_inline("nodebug")
    fn __eq__(self, rhs: Self) -> Bool:
        """Compare this value to the RHS value using EQ comparison.

        Args:
            rhs: The other value to compare against.

        Returns:
            True if this value is equal to the RHS value.
        """
        return self._value == rhs._value

    @always_inline("nodebug")
    fn __ne__(self, rhs: Self) -> Bool:
        """Compare this value to the RHS value using NE comparison.

        Args:
            rhs: The other value to compare against.

        Returns:
            True if this value is not equal to the RHS value.
        """
        return self._value != rhs._value

    @always_inline("nodebug")
    fn __is__(self, rhs: Self) -> Bool:
        """Compare this value to the RHS value using EQ comparison.

        Args:
            rhs: The other value to compare against.

        Returns:
            True if this value is equal to the RHS value.
        """
        return self == rhs

    @always_inline("nodebug")
    fn __isnot__(self, rhs: Self) -> Bool:
        """Compare this value to the RHS value using NE comparison.

        Args:
            rhs: The other value to compare against.

        Returns:
            True if this value is not equal to the RHS value.
        """
        return self != rhs


trait Roundable:
    """The `Roundable` trait describes a type that defines a rounding operation.

    Types that conform to `Roundable` will work with the builtin `round`
    function. The round operation always returns the same type as the input.

    Examples:
    ```mojo
    @fieldwise_init
    struct Complex(Roundable):
        alias _RoundMultipleType = Float64
        alias _RoundMultipleDefault = Float64(1)
        alias _RoundModeDefault = RoundMode.HalfToEven

        var re: Float64
        var im: Float64

        fn __round__[
            mode: RoundMode, to_multiple_of: Self._RoundMultipleType
        ](self) -> Self:
            return Self(
                round[mode, to_multiple_of=to_multiple_of](self.re),
                round[mode, to_multiple_of=to_multiple_of](self.im),
            )

        fn __round__(self, ndigits: Int) -> Self:
            return Self(round(self.re, ndigits), round(self.im, ndigits))
    ```
    """

    # FIXME(#5276): The struct should be able to set the default round mode to
    # `RoundMode.HalfToEven``.
    alias _RoundMultipleType: AnyType
    alias _RoundMultipleDefault: Self._RoundMultipleType
    alias _RoundModeDefault: RoundMode

    fn __round__[
        mode: RoundMode, to_multiple_of: Self._RoundMultipleType
    ](self) -> Self:
        """Get a rounded value for the type.

        Parameters:
            mode: The `RoundMode` for the operation.
            to_multiple_of: The target base to which self is is rounded until
                reaching a multiple of.

        Returns:
            The rounded value.
        """
        ...

    fn __round__(self, ndigits: Int) -> Self:
        """Get a rounded value for the type.

        Args:
            ndigits: Number of digits after the decimal point.

        Returns:
            The rounded value.
        """
        ...


@always_inline
fn round[
    T: Roundable, //,
    mode: RoundMode = T._RoundModeDefault,
    *,
    to_multiple_of: T._RoundMultipleType = T._RoundMultipleDefault,
](number: T) -> T:
    """Get the rounded value of the given object.

    Parameters:
        T: The type conforming to Roundable.
        mode: The `RoundMode` for the operation.
        to_multiple_of: The target base to which self is is rounded until
            reaching a multiple of.

    Args:
        number: The object to get the rounded value of.

    Returns:
        The rounded value of the object.
    """
    return number.__round__[mode, to_multiple_of]()


@always_inline
fn round[T: Roundable, //](number: T, ndigits: Int) -> T:
    """Get the value of this object, rounded to a specified number of
    digits after the decimal point.

    Parameters:
        T: The type conforming to Roundable.

    Args:
        number: The object to get the rounded value of.
        ndigits: The number of digits to round to.

    Returns:
        The rounded value of the object.
    """
    return number.__round__(ndigits)
