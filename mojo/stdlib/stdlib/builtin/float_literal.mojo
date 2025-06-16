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
"""Implements the FloatLiteral class.

These are Mojo built-ins, so you don't need to import them.
"""

# ===-----------------------------------------------------------------------===#
# FloatLiteral
# ===-----------------------------------------------------------------------===#


@nonmaterializable(Float64)
@register_passable("trivial")
struct FloatLiteral[value: __mlir_type.`!pop.float_literal`](
    Defaultable, Floatable, ImplicitlyBoolable, Intable, Stringable
):
    """Mojo floating point literal type.

    Parameters:
        value: The underlying infinite precision floating point value.
    """

    # ===------------------------------------------------------------------===#
    # Constructors
    # ===------------------------------------------------------------------===#

    @always_inline("builtin")
    fn __init__(out self):
        """Create a FloatLiteral for any parameter value."""
        pass

    @always_inline("builtin")
    @implicit
    fn __init__(
        value: IntLiteral[_],
    ) -> FloatLiteral[
        __mlir_attr[
            `#pop<int_to_float_literal<`,
            value.value,
            `>> : !pop.float_literal`,
        ]
    ]:
        """Convert an IntLiteral to a FloatLiteral value.

        Args:
            value: The IntLiteral value.
        """
        return {}

    alias nan = FloatLiteral[__mlir_attr.`#pop.float_literal<nan>`]()
    alias infinity = FloatLiteral[__mlir_attr.`#pop.float_literal<inf>`]()
    alias negative_infinity = FloatLiteral[
        __mlir_attr.`#pop.float_literal<neg_inf>`
    ]()
    alias negative_zero = FloatLiteral[
        __mlir_attr.`#pop.float_literal<neg_zero>`
    ]()

    @always_inline("builtin")
    fn is_nan(self) -> Bool:
        """Return whether the FloatLiteral is nan.

        Since `nan == nan` is False, this provides a way to check for nan-ness.

        Returns:
            True, if the value is nan, False otherwise.
        """
        return __mlir_attr[`#pop<float_literal_isa<nan `, self.value, `>> : i1`]

    @always_inline("builtin")
    fn is_neg_zero(self) -> Bool:
        """Return whether the FloatLiteral is negative zero.

        Since `FloatLiteral.negative_zero == 0.0` is True, this provides a way
        to check if the FloatLiteral is negative zero.

        Returns:
            True, if the value is negative zero, False otherwise.
        """
        return __mlir_attr[
            `#pop<float_literal_isa<neg_zero `, self.value, `>> : i1`
        ]

    @always_inline("builtin")
    fn _is_normal(self) -> Bool:
        """Return whether the FloatLiteral is a normal (i.e. not special) value.

        Returns:
            True, if the value is a normal float, False otherwise.
        """
        return __mlir_attr[
            `#pop<float_literal_isa<normal `, self.value, `>> : i1`
        ]

    # ===------------------------------------------------------------------===#
    # Conversion Operators
    # ===------------------------------------------------------------------===#

    @no_inline
    fn __str__(self) -> String:
        """Get the float as a string.

        Returns:
            A string representation.
        """
        return String(Float64(self))

    @always_inline("builtin")
    fn __int_literal__(
        self,
    ) -> IntLiteral[
        __mlir_attr[
            `#pop<float_to_int_literal<`,
            value,
            `>> : !pop.int_literal`,
        ]
    ]:
        """Casts the floating point value to an IntLiteral. If there is a
        fractional component, then the value is truncated towards zero.

        Eg. `(4.5).__int_literal__()` returns `4`, and `(-3.7).__int_literal__()`
        returns `-3`.

        Returns:
            The value as an integer.
        """
        return {}

    @always_inline("builtin")
    fn __int__(self) -> Int:
        """Converts the FloatLiteral value to an Int. If there is a fractional
        component, then the value is truncated towards zero.

        Eg. `(4.5).__int__()` returns `4`, and `(-3.7).__int__()` returns `-3`.

        Returns:
            The value as an integer.
        """
        return self.__int_literal__().__int__()

    @always_inline("nodebug")
    fn __float__(self) -> Float64:
        """Converts the FloatLiteral to a concrete Float64.

        Returns:
            The Float value.
        """
        return Float64(self)

    # ===------------------------------------------------------------------===#
    # Unary Operators
    # ===------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __bool__(self) -> Bool:
        """A FloatLiteral value is true if it is non-zero.

        Returns:
            True if non-zero.
        """
        return self != 0.0

    @always_inline("nodebug")
    fn __as_bool__(self) -> Bool:
        """A FloatLiteral value is true if it is non-zero.

        Returns:
            True if non-zero.
        """
        return self.__bool__()

    @always_inline("builtin")
    fn __neg__(self) -> __type_of(self * -1):
        """Return the negation of the FloatLiteral value.

        Returns:
            The negated FloatLiteral value.
        """
        return {}

    # ===------------------------------------------------------------------===#
    # Arithmetic Operators
    # ===------------------------------------------------------------------===#

    @always_inline("builtin")
    fn __add__(
        self, rhs: FloatLiteral
    ) -> FloatLiteral[
        __mlir_attr[
            `#pop<float_literal_bin<add `,
            value,
            `,`,
            rhs.value,
            `>> : !pop.float_literal`,
        ]
    ]:
        """Add two FloatLiterals.

        Args:
            rhs: The value to add.

        Returns:
            The sum of the two values.
        """
        return {}

    @always_inline("builtin")
    fn __sub__(
        self, rhs: FloatLiteral
    ) -> FloatLiteral[
        __mlir_attr[
            `#pop<float_literal_bin<sub `,
            value,
            `,`,
            rhs.value,
            `>> : !pop.float_literal`,
        ]
    ]:
        """Subtract two FloatLiterals.

        Args:
            rhs: The value to subtract.

        Returns:
            The difference of the two values.
        """
        return {}

    @always_inline("builtin")
    fn __mul__(
        self, rhs: FloatLiteral
    ) -> FloatLiteral[
        __mlir_attr[
            `#pop<float_literal_bin<mul `,
            value,
            `,`,
            rhs.value,
            `>> : !pop.float_literal`,
        ]
    ]:
        """Multiply two FloatLiterals.

        Args:
            rhs: The value to multiply.

        Returns:
            The product of the two values.
        """
        return {}

    @always_inline("builtin")
    fn __truediv__(
        self, rhs: FloatLiteral
    ) -> FloatLiteral[
        __mlir_attr[
            `#pop<float_literal_bin<truediv `,
            value,
            `,`,
            rhs.value,
            `>> : !pop.float_literal`,
        ]
    ]:
        """Divide two FloatLiterals.

        Args:
            rhs: The value to divide.

        Returns:
            The quotient of the two values.
        """
        # TODO - Python raises an error on divide by 0.0 or -0.0
        return {}

    @always_inline("builtin")
    fn __floordiv__(
        self, rhs: FloatLiteral
    ) -> FloatLiteral[
        __mlir_attr[
            `#pop<float_literal_bin<floordiv `,
            value,
            `,`,
            rhs.value,
            `>> : !pop.float_literal`,
        ]
    ]:
        """Returns self divided by rhs, rounded down to the nearest integer.

        Args:
            rhs: The divisor value.

        Returns:
            `floor(self / rhs)` value.
        """
        # TODO - Python raises an error on divide by 0.0 or -0.0
        return {}

    @always_inline("builtin")
    fn __mod__(
        self, rhs: FloatLiteral
    ) -> __type_of(self - (self.__floordiv__(rhs) * rhs)):
        """Return the remainder of self divided by rhs.

        Args:
            rhs: The value to divide on.

        Returns:
            The remainder of dividing self by rhs.
        """
        return {}

    @always_inline("builtin")
    fn __ceildiv__(
        self, denominator: FloatLiteral
    ) -> __type_of(-(self // -denominator)):
        """Return the rounded-up result of dividing self by denominator.

        Args:
            denominator: The denominator.

        Returns:
            The ceiling of dividing numerator by denominator.
        """
        return {}

    # TODO - maybe __pow__?

    # ===------------------------------------------------------------------===#
    # Reversed Operators, allowing things like "1 / 2.0" to work
    # ===------------------------------------------------------------------===#

    @always_inline("builtin")
    fn __radd__(self, rhs: FloatLiteral) -> __type_of(rhs + self):
        """Reversed addition operator.

        Args:
            rhs: The value to add.

        Returns:
            The sum of this and the given value.
        """
        return {}

    @always_inline("builtin")
    fn __rsub__(self, rhs: FloatLiteral) -> __type_of(rhs - self):
        """Reversed subtraction operator.

        Args:
            rhs: The value to subtract from.

        Returns:
            The result of subtracting this from the given value.
        """
        return {}

    @always_inline("builtin")
    fn __rmul__(self, rhs: FloatLiteral) -> __type_of(rhs * self):
        """Reversed multiplication operator.

        Args:
            rhs: The value to multiply.

        Returns:
            The product of the given number and this.
        """
        return {}

    @always_inline("builtin")
    fn __rmod__(self, rhs: FloatLiteral) -> __type_of(rhs.__mod__(self)):
        """Return the remainder of rhs divided by self.

        Args:
            rhs: The value to divide on.

        Returns:
            The remainder of dividing rhs by self.
        """
        return {}

    @always_inline("builtin")
    fn __rfloordiv__(self, rhs: FloatLiteral) -> __type_of(rhs // self):
        """Returns rhs divided by self, rounded down to the nearest integer.

        Args:
            rhs: The value to be divided by self.

        Returns:
            `floor(rhs / self)` value.
        """
        return {}

    @always_inline("builtin")
    fn __rtruediv__(self, rhs: FloatLiteral) -> __type_of(rhs / self):
        """Reversed division.

        Args:
            rhs: The value to be divided by this.

        Returns:
            The result of dividing the given value by this.
        """
        return {}

    # ===------------------------------------------------------------------===#
    # Comparison Operators
    # ===------------------------------------------------------------------===#

    @always_inline("builtin")
    fn __eq__(self, rhs: FloatLiteral) -> Bool:
        """Compare for equality.

        Args:
            rhs: The value to compare.

        Returns:
            True if they are equal.
        """
        return Bool(
            __mlir_attr[
                `#pop<float_literal_cmp<eq `,
                self.value,
                `,`,
                rhs.value,
                `>> : i1`,
            ]
        )

    @always_inline("builtin")
    fn __ne__(self, rhs: FloatLiteral) -> Bool:
        """Compare for inequality.

        Args:
            rhs: The value to compare.

        Returns:
            True if they are not equal.
        """
        return Bool(
            __mlir_attr[
                `#pop<float_literal_cmp<ne `,
                self.value,
                `,`,
                rhs.value,
                `>> : i1`,
            ]
        )

    @always_inline("builtin")
    fn __lt__(self, rhs: FloatLiteral) -> Bool:
        """Less than comparison.

        Args:
            rhs: The value to compare.

        Returns:
            True if this value is less than `rhs`.
        """
        return Bool(
            __mlir_attr[
                `#pop<float_literal_cmp<lt `,
                self.value,
                `,`,
                rhs.value,
                `>> : i1`,
            ]
        )

    @always_inline("builtin")
    fn __le__(self, rhs: FloatLiteral) -> Bool:
        """Less than or equal to comparison.

        Args:
            rhs: The value to compare.

        Returns:
            True if this value is less than or equal to `rhs`.
        """
        return Bool(
            __mlir_attr[
                `#pop<float_literal_cmp<le `,
                self.value,
                `,`,
                rhs.value,
                `>> : i1`,
            ]
        )

    @always_inline("builtin")
    fn __gt__(self, rhs: FloatLiteral) -> Bool:
        """Greater than comparison.

        Args:
            rhs: The value to compare.

        Returns:
            True if this value is greater than `rhs`.
        """
        return Bool(
            __mlir_attr[
                `#pop<float_literal_cmp<gt `,
                self.value,
                `,`,
                rhs.value,
                `>> : i1`,
            ]
        )

    @always_inline("builtin")
    fn __ge__(self, rhs: FloatLiteral) -> Bool:
        """Greater than or equal to comparison.

        Args:
            rhs: The value to compare.

        Returns:
            True if this value is greater than or equal to `rhs`.
        """
        return Bool(
            __mlir_attr[
                `#pop<float_literal_cmp<ge `,
                self.value,
                `,`,
                rhs.value,
                `>> : i1`,
            ]
        )
