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
"""Implements the IntLiteral class."""

from math import Ceilable, CeilDivable, Floorable, Truncable


@value
@nonmaterializable(Int)
@register_passable("trivial")
struct IntLiteral(
    Absable,
    Ceilable,
    CeilDivable,
    Comparable,
    Floorable,
    ImplicitlyBoolable,
    ImplicitlyIntable,
    Indexer,
    Stringable,
    Truncable,
):
    """This type represents a static integer literal value with
    infinite precision.  They can't be materialized at runtime and
    must be lowered to other integer types (like Int), but allow for
    compile-time operations that would overflow on Int and other fixed
    precision integer types.
    """

    # Fields
    alias _mlir_type = __mlir_type.`!kgen.int_literal`

    var value: Self._mlir_type
    """The underlying storage for the integer value."""

    alias _one = IntLiteral(
        __mlir_attr.`#kgen.int_literal<1> : !kgen.int_literal`
    )

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline("builtin")
    fn __init__(out self):
        """Default constructor."""
        self.value = __mlir_attr.`#kgen.int_literal<0> : !kgen.int_literal`

    @doc_private
    @always_inline("builtin")
    @implicit
    fn __init__(out self, value: __mlir_type.`!kgen.int_literal`):
        """Construct IntLiteral from the given mlir !kgen.int_literal value.

        Args:
            value: The init value.
        """
        self.value = value

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline("builtin")
    fn __lt__(self, rhs: Self) -> Bool:
        """Compare this IntLiteral to the RHS using LT comparison.

        Args:
            rhs: The other IntLiteral to compare against.

        Returns:
            True if this IntLiteral is less-than the RHS IntLiteral and False otherwise.
        """
        return __mlir_op.`kgen.int_literal.cmp`[
            pred = __mlir_attr.`#kgen<int_literal.cmp_pred lt>`
        ](self.value, rhs.value)

    @always_inline("builtin")
    fn __le__(self, rhs: Self) -> Bool:
        """Compare this IntLiteral to the RHS using LE comparison.

        Args:
            rhs: The other IntLiteral to compare against.

        Returns:
            True if this IntLiteral is less-or-equal than the RHS IntLiteral and False
            otherwise.
        """
        return __mlir_op.`kgen.int_literal.cmp`[
            pred = __mlir_attr.`#kgen<int_literal.cmp_pred le>`
        ](self.value, rhs.value)

    @always_inline("builtin")
    fn __eq__(self, rhs: Self) -> Bool:
        """Compare this IntLiteral to the RHS using EQ comparison.

        Args:
            rhs: The other IntLiteral to compare against.

        Returns:
            True if this IntLiteral is equal to the RHS IntLiteral and False otherwise.
        """
        return __mlir_op.`kgen.int_literal.cmp`[
            pred = __mlir_attr.`#kgen<int_literal.cmp_pred eq>`
        ](self.value, rhs.value)

    @always_inline("builtin")
    fn __ne__(self, rhs: Self) -> Bool:
        """Compare this IntLiteral to the RHS using NE comparison.

        Args:
            rhs: The other IntLiteral to compare against.

        Returns:
            True if this IntLiteral is non-equal to the RHS IntLiteral and False otherwise.
        """
        return __mlir_op.`kgen.int_literal.cmp`[
            pred = __mlir_attr.`#kgen<int_literal.cmp_pred ne>`
        ](self.value, rhs.value)

    @always_inline("builtin")
    fn __gt__(self, rhs: Self) -> Bool:
        """Compare this IntLiteral to the RHS using GT comparison.

        Args:
            rhs: The other IntLiteral to compare against.

        Returns:
            True if this IntLiteral is greater-than the RHS IntLiteral and False otherwise.
        """
        return __mlir_op.`kgen.int_literal.cmp`[
            pred = __mlir_attr.`#kgen<int_literal.cmp_pred gt>`
        ](self.value, rhs.value)

    @always_inline("builtin")
    fn __ge__(self, rhs: Self) -> Bool:
        """Compare this IntLiteral to the RHS using GE comparison.

        Args:
            rhs: The other IntLiteral to compare against.

        Returns:
            True if this IntLiteral is greater-or-equal than the RHS IntLiteral and False
            otherwise.
        """
        return __mlir_op.`kgen.int_literal.cmp`[
            pred = __mlir_attr.`#kgen<int_literal.cmp_pred ge>`
        ](self.value, rhs.value)

    @always_inline("builtin")
    fn __pos__(self) -> Self:
        """Return +self.

        Returns:
            The +self value.
        """
        return self

    @always_inline("builtin")
    fn __neg__(self) -> Self:
        """Return -self.

        Returns:
            The -self value.
        """
        return Self() - self

    @always_inline("builtin")
    fn __invert__(self) -> Self:
        """Return ~self.

        Returns:
            The ~self value.
        """
        return self ^ (Self() - Self._one)

    @always_inline("builtin")
    fn __add__(self, rhs: Self) -> Self:
        """Return `self + rhs`.

        Args:
            rhs: The value to add.

        Returns:
            `self + rhs` value.
        """
        return Self(
            __mlir_op.`kgen.int_literal.binop`[
                oper = __mlir_attr.`#kgen<int_literal.binop_kind add>`
            ](self.value, rhs.value)
        )

    @always_inline("builtin")
    fn __sub__(self, rhs: Self) -> Self:
        """Return `self - rhs`.

        Args:
            rhs: The value to subtract.

        Returns:
            `self - rhs` value.
        """
        return Self(
            __mlir_op.`kgen.int_literal.binop`[
                oper = __mlir_attr.`#kgen<int_literal.binop_kind sub>`
            ](self.value, rhs.value)
        )

    @always_inline("builtin")
    fn __mul__(self, rhs: Self) -> Self:
        """Return `self * rhs`.

        Args:
            rhs: The value to multiply with.

        Returns:
            `self * rhs` value.
        """
        return Self(
            __mlir_op.`kgen.int_literal.binop`[
                oper = __mlir_attr.`#kgen<int_literal.binop_kind mul>`
            ](self.value, rhs.value)
        )

    # TODO: implement __pow__

    @always_inline("builtin")
    fn __floordiv__(self, rhs: Self) -> Self:
        """Return `self // rhs`.

        Args:
            rhs: The value to divide with.

        Returns:
            `self // rhs` value.
        """
        # This handles the case where rhs is 0.
        return Self(
            __mlir_op.`kgen.int_literal.binop`[
                oper = __mlir_attr.`#kgen<int_literal.binop_kind floordiv>`
            ](self.value, rhs.value)
        )

    @always_inline("builtin")
    fn __mod__(self, rhs: Self) -> Self:
        """Return the remainder of self divided by rhs.

        Args:
            rhs: The value to divide on.

        Returns:
            The remainder of dividing self by rhs.
        """
        # This handles the case where rhs is 0.
        return Self(
            __mlir_op.`kgen.int_literal.binop`[
                oper = __mlir_attr.`#kgen<int_literal.binop_kind mod>`
            ](self.value, rhs.value)
        )

    @always_inline("builtin")
    fn __lshift__(self, rhs: Self) -> Self:
        """Return `self << rhs`.

        Args:
            rhs: The value to shift with.

        Returns:
            `self << rhs`.
        """
        # This handles the case where rhs is 0.
        return Self(
            __mlir_op.`kgen.int_literal.binop`[
                oper = __mlir_attr.`#kgen<int_literal.binop_kind lshift>`
            ](self.value, rhs.value)
        )

    @always_inline("builtin")
    fn __rshift__(self, rhs: Self) -> Self:
        """Return `self >> rhs`.

        Args:
            rhs: The value to shift with.

        Returns:
            `self >> rhs`.
        """
        # This handles the case where rhs is 0.
        return Self(
            __mlir_op.`kgen.int_literal.binop`[
                oper = __mlir_attr.`#kgen<int_literal.binop_kind rshift>`
            ](self.value, rhs.value)
        )

    @always_inline("builtin")
    fn __and__(self, rhs: Self) -> Self:
        """Return `self & rhs`.

        Args:
            rhs: The RHS value.

        Returns:
            `self & rhs`.
        """
        return Self(
            __mlir_op.`kgen.int_literal.binop`[
                oper = __mlir_attr.`#kgen<int_literal.binop_kind and>`
            ](self.value, rhs.value)
        )

    @always_inline("builtin")
    fn __xor__(self, rhs: Self) -> Self:
        """Return `self ^ rhs`.

        Args:
            rhs: The RHS value.

        Returns:
            `self ^ rhs`.
        """
        return Self(
            __mlir_op.`kgen.int_literal.binop`[
                oper = __mlir_attr.`#kgen<int_literal.binop_kind xor>`
            ](self.value, rhs.value)
        )

    @always_inline("builtin")
    fn __or__(self, rhs: Self) -> Self:
        """Return `self | rhs`.

        Args:
            rhs: The RHS value.

        Returns:
            `self | rhs`.
        """
        return Self(
            __mlir_op.`kgen.int_literal.binop`[
                oper = __mlir_attr.`#kgen<int_literal.binop_kind or>`
            ](self.value, rhs.value)
        )

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline("builtin")
    fn __bool__(self) -> Bool:
        """Convert this IntLiteral to Bool.

        Returns:
            False Bool value if the value is equal to 0 and True otherwise.
        """
        return self != Self()

    @always_inline("builtin")
    fn __as_bool__(self) -> Bool:
        """Convert this IntLiteral to Bool.

        Returns:
            False Bool value if the value is equal to 0 and True otherwise.
        """
        return self.__bool__()

    @always_inline("builtin")
    fn __int__(self) -> Int:
        """Convert from IntLiteral to Int.

        Returns:
            The value as an integer of platform-specific width.
        """
        return self.__index__()

    @always_inline("builtin")
    fn __as_int__(self) -> Int:
        """Implicitly convert to an Int.

        Returns:
            An integral value that represents this object.
        """
        return self.__int__()

    @always_inline("builtin")
    fn __uint__(self) -> UInt:
        """Convert from IntLiteral to UInt.

        Returns:
            The value as an unsigned integer of platform-specific width.
        """
        return __mlir_op.`kgen.int_literal.convert`[
            _type = __mlir_type.index, treatIndexAsUnsigned = __mlir_attr.unit
        ](self.value)

    @always_inline("nodebug")
    fn __abs__(self) -> Self:
        """Return the absolute value of the IntLiteral value.

        Returns:
            The absolute value.
        """
        if self >= 0:
            return self
        return -self

    @always_inline("builtin")
    fn __ceil__(self) -> Self:
        """Return the ceiling of the IntLiteral value, which is itself.

        Returns:
            The IntLiteral value itself.
        """
        return self

    @always_inline("builtin")
    fn __floor__(self) -> Self:
        """Return the floor of the IntLiteral value, which is itself.

        Returns:
            The IntLiteral value itself.
        """
        return self

    @always_inline("builtin")
    fn __trunc__(self) -> Self:
        """Return the truncated of the IntLiteral value, which is itself.

        Returns:
            The IntLiteral value itself.
        """
        return self

    @no_inline
    fn __str__(self) -> String:
        """Convert from IntLiteral to String.

        Returns:
            The value as a string.
        """
        return String(Int(self))

    @always_inline("builtin")
    fn __ceildiv__(self, denominator: Self) -> Self:
        """Return the rounded-up result of dividing self by denominator.


        Args:
            denominator: The denominator.

        Returns:
            The ceiling of dividing numerator by denominator.
        """
        return -(self // -denominator)

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @always_inline("builtin")
    fn __index__(self) -> __mlir_type.index:
        """Convert from IntLiteral to index.

        Returns:
            The corresponding __mlir_type.index value.
        """
        return __mlir_op.`kgen.int_literal.convert`[_type = __mlir_type.index](
            self.value
        )
