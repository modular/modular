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
"""Implements the Complex dtype.

You can import these APIs from the `complex` package. For example:

```mojo
from complex import ComplexSIMD
```
"""

import math
from math.math import _Expable
from sys import llvm_intrinsic
from format._utils import FormatStruct

comptime ComplexScalar = ComplexSIMD[size=1]
"""Represents a scalar complex value."""
comptime ComplexFloat32 = ComplexScalar[DType.float32]
"""A complex number with 32-bit floating point components."""

comptime ComplexFloat64 = ComplexScalar[DType.float64]
"""A complex number with 64-bit floating point components."""


# ===-----------------------------------------------------------------------===#
# ComplexSIMD
# ===-----------------------------------------------------------------------===#


struct ComplexSIMD[dtype: DType, size: Int](
    Equatable, Stringable, TrivialRegisterPassable, Writable, _Expable
):
    """Represents a complex SIMD value.

    The class provides basic methods for manipulating complex values.

    Parameters:
        dtype: DType of the value.
        size: SIMD width of the value.
    """

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    comptime type = Self.dtype
    """The data type of the complex components."""

    comptime element_type = SIMD[Self.dtype, Self.size]
    """The SIMD type used for real and imaginary parts."""

    var re: Self.element_type
    """The real part of the complex SIMD value."""
    var im: Self.element_type
    """The imaginary part of the complex SIMD value."""

    # ===-------------------------------------------------------------------===#
    # Initialization
    # ===-------------------------------------------------------------------===#

    fn __init__(out self, re: Self.element_type, im: Self.element_type = 0):
        """Initializes a complex SIMD value.

        Args:
            re: The real part of the complex value.
            im: The imaginary part of the complex value.
        """
        self.re = re
        self.im = im

    fn __init__(out self, *, from_interleaved: SIMD[Self.dtype, 2 * Self.size]):
        """Initializes a complex SIMD value.

        Args:
            from_interleaved: An interleaved vector of complex values e.g.
                `[0, 1, 1, 0]` where the pattern is `[re0, im0, re1, im1]`.
        """
        comptime T = Tuple[Self.element_type, Self.element_type]
        self.re, self.im = rebind[T](from_interleaved.deinterleave())

    fn __init__(
        out self, *, from_deinterleaved: SIMD[Self.dtype, 2 * Self.size]
    ):
        """Initializes a complex SIMD value.

        Args:
            from_deinterleaved: A deinterleaved vector of complex values e.g.
                `[0, 1, 1, 0]` where the pattern is `[re0, re1, im0, im1]`.
        """
        comptime T = Self.element_type
        self.re = rebind[T](from_deinterleaved.slice[Self.size]())
        self.im = rebind[T](
            from_deinterleaved.slice[Self.size, offset = Self.size]()
        )

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @no_inline
    fn __str__(self) -> String:
        """Get the complex as a string.

        Returns:
            A string representation.
        """
        return String.write(self)

    fn write_to(self, mut writer: Some[Writer]):
        """
        Formats this complex value to the provided Writer.

        Args:
            writer: The object to write to.
        """

        # Print an opening `[`.
        comptime if Self.size > 1:
            writer.write_string("[")

        # Print each element.
        for i in range(Self.size):
            var re = self.re[i]
            var im = self.im[i]
            # Print separators between each element.
            if i != 0:
                writer.write_string(", ")

            writer.write(re)

            if im != 0:
                writer.write(" + ", im, "i")

        # Print a closing `]`.
        comptime if Self.size > 1:
            writer.write_string("]")

    @no_inline
    fn write_repr_to(self, mut writer: Some[Writer]):
        """Formats the complex value for debug representation.

        Args:
            writer: The Writer to write the representation to.
        """
        FormatStruct(writer, "ComplexSIMD").params(
            Self.dtype,
            Self.size,
        ).fields(self)

    @always_inline
    fn __abs__(self) -> SIMD[Self.dtype, Self.size]:
        """Returns the magnitude of the complex value.

        Returns:
            Value of `sqrt(re*re + im*im)`.
        """
        return self.norm()

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __add__(self, rhs: Self) -> Self:
        """Adds two complex values.

        Args:
            rhs: Complex value to add.

        Returns:
            A sum of this and RHS complex values.
        """
        return Self(self.re + rhs.re, self.im + rhs.im)

    @always_inline
    fn __mul__(self, rhs: Self) -> Self:
        """Multiplies two complex values.

        Args:
            rhs: Complex value to multiply with.

        Returns:
            A product of this and RHS complex values.
        """
        return Self(
            self.re.fma(rhs.re, -self.im * rhs.im),
            self.re.fma(rhs.im, self.im * rhs.re),
        )

    @always_inline
    fn __mul__(self, rhs: Scalar[Self.dtype]) -> Self:
        """Multiplies a complex value to a scalar.

        Args:
            rhs: Scalar value to multiply with.

        Returns:
            A product of self and rhs.
        """
        return Self(self.re * rhs, self.im * rhs)

    @always_inline
    fn __rmul__(self, lhs: Scalar[Self.dtype]) -> Self:
        """Multiplies a complex value to a scalar.

        Args:
            lhs: Scalar value to multiply with.

        Returns:
            A product of self and lhs.
        """
        return Self(self.re * lhs, self.im * lhs)

    @always_inline
    fn __imul__(mut self, rhs: Self):
        """Multiplies two complex values inplace.

        Args:
            rhs: Complex value to multiply with.
        """
        self = self * rhs

    @always_inline
    fn __imul__(mut self, rhs: Scalar[Self.dtype]):
        """Multiplies a complex value to a scalar inplace.

        Args:
            rhs: Scalar value to multiply with.
        """
        self.re *= rhs
        self.im *= rhs

    @always_inline
    fn __sub__(self, rhs: Self) -> Self:
        """Subtracts two complex values.

        Args:
            rhs: Complex value to subtract.

        Returns:
            A difference of this and RHS complex values.
        """
        return Self(self.re - rhs.re, self.im - rhs.im)

    @always_inline
    fn __truediv__(self, rhs: Self) -> Self:
        """Divides two complex values.

        Args:
            rhs: Complex value to divide by.

        Returns:
            A quotient of this and RHS complex values.
        """
        var denom = rhs.squared_norm()
        return Self(
            self.re.fma(rhs.re, self.im * rhs.im) / denom,
            self.im.fma(rhs.re, -(self.re * rhs.im)) / denom,
        )

    @always_inline
    fn __neg__(self) -> Self:
        """Negates the complex value.

        Returns:
            The negative of the complex value.
        """
        return ComplexSIMD(-self.re, -self.im)

    @always_inline
    fn __eq__(self, rhs: Self) -> Bool:
        """Compares two ComplexSIMD for equality.

        Args:
            rhs: The ComplexSIMD to compare with.

        Returns:
            True if all elements of the ComplexSIMD are equal, False otherwise.
        """
        return Bool((self.re.eq(rhs.re) & self.im.eq(rhs.im)).reduce_and())

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn conj(self) -> Self:
        """Return the complex conjugate of self.

        Returns:
            The complex conjugate of self.
        """
        return Self(self.re, -self.im)

    @always_inline
    fn norm(self) -> SIMD[Self.dtype, Self.size]:
        """Returns the magnitude of the complex value.

        Returns:
            Value of `sqrt(re*re + im*im)`.
        """
        return llvm_intrinsic["llvm.sqrt", SIMD[Self.dtype, Self.size]](
            self.squared_norm()
        )

    @always_inline
    fn squared_norm(self) -> SIMD[Self.dtype, Self.size]:
        """Returns the squared magnitude of the complex value.

        Returns:
            Value of `re*re + im*im`.
        """
        return self.re.fma(self.re, self.im * self.im)

    # fma(self, b, c)
    @always_inline
    fn fma(self, b: Self, c: Self) -> Self:
        """Computes FMA operation.

        Compute fused multiple-add with two other complex values:
        `result = self * b + c`

        Args:
            b: Multiplier complex value.
            c: Complex value to add.

        Returns:
            Computed `Self * B + C` complex value.
        """
        return Self(
            self.re.fma(b.re, -(self.im.fma(b.im, -c.re))),
            self.re.fma(b.im, self.im.fma(b.re, c.im)),
        )

    # fma(self, self, c)
    @always_inline
    fn squared_add(self, c: Self) -> Self:
        """Computes Square-Add operation.

        Compute `Self * Self + C`.

        Args:
            c: Complex value to add.

        Returns:
            Computed `Self * Self + C` complex value.
        """
        return Self(
            self.re.fma(self.re, self.im.fma(-self.im, c.re)),
            self.re.fma(self.im + self.im, c.im),
        )

    @always_inline
    fn __exp__(self) -> Self:
        """Computes the exponential of the complex value.

        Returns:
            The exponential of the complex value.
        """
        comptime assert (
            Self.dtype.is_floating_point()
        ), "dtype must be floating point"
        var exp_re = math.exp(self.re)
        return Self(exp_re * math.cos(self.im), exp_re * math.sin(self.im))


# TODO: we need this overload, because the Absable trait requires returning Self
# dtype. We could maybe get rid of this if we had associated dtypes?
@always_inline
fn abs(x: ComplexSIMD[...]) -> SIMD[x.dtype, x.size]:
    """Performs elementwise abs (norm) on each element of the complex value.

    Args:
        x: The complex vector to perform absolute value on.

    Returns:
        The elementwise abs of x.
    """
    return x.__abs__()
