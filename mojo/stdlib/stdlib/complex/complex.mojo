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
"""Implements the Complex dtype.

You can import these APIs from the `complex` package. For example:

```mojo
from complex import ComplexSIMD
```
"""

import math
from math.math import _Expable
from sys import llvm_intrinsic

alias ComplexScalar = ComplexSIMD[size=1]
"""Represents a scalar complex value."""
alias ComplexFloat32 = ComplexScalar[DType.float32]
alias ComplexFloat64 = ComplexScalar[DType.float64]


# ===-----------------------------------------------------------------------===#
# ComplexSIMD
# ===-----------------------------------------------------------------------===#


@register_passable("trivial")
struct ComplexSIMD[dtype: DType, size: Int](
    EqualityComparable, Stringable, Writable, _Expable
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

    alias type = dtype
    alias element_type = SIMD[dtype, size]
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

    fn __init__(out self, *, from_interleaved: SIMD[dtype, 2 * size]):
        """Initializes a complex SIMD value.

        Args:
            from_interleaved: An interleaved vector of complex values e.g.
                `[0, 1, 1, 0]` where the pattern is `[re0, im0, re1, im1]`.
        """
        alias T = Tuple[Self.element_type, Self.element_type]
        self.re, self.im = rebind[T](from_interleaved.deinterleave())

    fn __init__(out self, *, from_deinterleaved: SIMD[dtype, 2 * size]):
        """Initializes a complex SIMD value.

        Args:
            from_deinterleaved: A deinterleaved vector of complex values e.g.
                `[0, 1, 1, 0]` where the pattern is `[re0, re1, im0, im1]`.
        """
        alias T = Self.element_type
        self.re = rebind[T](from_deinterleaved.slice[size]())
        self.im = rebind[T](from_deinterleaved.slice[size, offset=size]())

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

        # TODO(MSTDL-700):
        #   Add a Writer.reserve() method, to afford writer implementations
        #   to request reservation of additional space from `Writer`
        #   implementations that support that. Then use the logic below to
        #   call that method here.

        # Reserve space for opening and closing brackets, plus each element and
        # its trailing commas.
        # var initial_buffer_size = 2
        # for i in range(size):
        #     initial_buffer_size += (
        #         _calc_initial_buffer_size(self.re[i])
        #         + _calc_initial_buffer_size(self.im[i])
        #         + 4  # for the ' + i' suffix on the imaginary
        #         + 2
        #     )
        # buf.reserve(initial_buffer_size)

        # Print an opening `[`.
        @parameter
        if size > 1:
            writer.write("[")

        # Print each element.
        for i in range(size):
            var re = self.re[i]
            var im = self.im[i]
            # Print separators between each element.
            if i != 0:
                writer.write(", ")

            writer.write(re)

            if im != 0:
                writer.write(" + ", im, "i")

        # Print a closing `]`.
        @parameter
        if size > 1:
            writer.write("]")

    @always_inline
    fn __abs__(self) -> SIMD[dtype, size]:
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
    fn __mul__(self, rhs: Scalar[dtype]) -> Self:
        """Multiplies a complex value to a scalar.

        Args:
            rhs: Scalar value to multiply with.

        Returns:
            A product of self and rhs.
        """
        return Self(self.re * rhs, self.im * rhs)

    @always_inline
    fn __rmul__(self, lhs: Scalar[dtype]) -> Self:
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
    fn __imul__(mut self, rhs: Scalar[dtype]):
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
            self.re.fma(rhs.im, -self.im * rhs.re) / denom,
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
    fn norm(self) -> SIMD[dtype, size]:
        """Returns the magnitude of the complex value.

        Returns:
            Value of `sqrt(re*re + im*im)`.
        """
        return llvm_intrinsic["llvm.sqrt", SIMD[dtype, size]](
            self.squared_norm()
        )

    @always_inline
    fn squared_norm(self) -> SIMD[dtype, size]:
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
        var exp_re = math.exp(self.re)
        return Self(exp_re * math.cos(self.im), exp_re * math.sin(self.im))


# TODO: we need this overload, because the Absable trait requires returning Self
# dtype. We could maybe get rid of this if we had associated dtypes?
@always_inline
fn abs(x: ComplexSIMD[*_]) -> SIMD[x.dtype, x.size]:
    """Performs elementwise abs (norm) on each element of the complex value.

    Args:
        x: The complex vector to perform absolute value on.

    Returns:
        The elementwise abs of x.
    """
    return x.__abs__()
