# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""The module provides implementation for Complex type."""

from DType import DType
from SIMD import SIMD


alias ComplexF32 = ComplexSIMD[DType.f32, 1]
alias ComplexF64 = ComplexSIMD[DType.f64, 1]


@register_passable("trivial")
struct ComplexSIMD[type: DType, size: Int]:
    """Represents a complex SIMD value.

    The class provides basic methods for manipulating complex values.

    Parameters:
        type: DType of the value.
        size: SIMD width of the value.
    """

    var re: SIMD[type, size]
    var im: SIMD[type, size]

    fn __init__(re: SIMD[type, size], im: SIMD[type, size]) -> Self:
        """Construct a complex value.

        Args:
            re: The real component.
            im: The imaginary component.

        Returns:
            The constructed Complex object.
        """
        return Self {re: re, im: im}

    fn __add__(self, rhs: Self) -> Self:
        """Add two complex values.

        Args:
            rhs: Complex value to add.

        Returns:
            A sum of this and RHS complex values.
        """
        return Self {re: self.re + rhs.re, im: self.im + rhs.im}

    fn __mul__(self, rhs: Self) -> Self:
        """Multiple two complex values.

        Args:
            rhs: Complex value to multiply with.

        Returns:
            A product of this and RHS complex values.
        """
        return Self(
            self.re.fma(rhs.re, -self.im * rhs.im),
            self.re.fma(rhs.im, self.im * rhs.re),
        )

    fn __neg__(self) -> Self:
        """Negates the complex value.

        Returns:
            The negative of the complex value.
        """
        return ComplexSIMD(-self.re, -self.im)

    # returns the squared magnitude
    fn norm(self) -> SIMD[type, size]:
        """Returns the squared magnitude of the complex value.

        Returns:
            Value of `re*re + im*im`.
        """
        return self.re.fma(self.re, self.im * self.im)

    # fma(self, b, c)
    fn fma(self, b: Self, c: Self) -> Self:
        """Compute FMA operation.

        Compute fused multiple-add with two other complex values:
        `result = self * b + c`

        Args:
            b: Multiplier complex value.
            c: Complex value to add.

        Returns:
            Computed `Self * B + C` complex value.
        """
        return Self(
            self.re.fma(b.re, -(self.im.fma(b.im, c.re))),
            self.re.fma(b.im, self.im.fma(b.re, c.im)),
        )

    # fma(self, self, c)
    fn sq_add(self, c: Self) -> Self:
        """Compute Square-Add operation.

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
