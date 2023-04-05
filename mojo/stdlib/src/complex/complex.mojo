# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""The module provides implementation for Complex type."""

from DType import DType
from SIMD import SIMD


@register_passable("trivial")
struct Complex[size: Int, type: DType]:
    """Represents a complex value.

    The class provides basic methods for manipulating complex values.

    Parameters:
        size: SIMD width of the value.
        type: DType of the value.
    """

    var re: SIMD[size, type.value]
    var im: SIMD[size, type.value]

    fn __init__(
        re: SIMD[size, type.value],
        im: SIMD[size, type.value],
    ) -> Complex[size, type]:
        """Construct a complex value.

        Args:
            re: The real component.
            im: The imaginary component.

        Returns:
            The constructed Complex object.
        """
        return Complex[size, type] {re: re, im: im}

    fn __add__(self, rhs: Complex[size, type]) -> Complex[size, type]:
        """Add two complex values.

        Args:
            rhs: Complex value to add.

        Returns:
            A sum of this and RHS complex values.
        """
        return Complex[size, type] {re: self.re + rhs.re, im: self.im + rhs.im}

    fn __mul__(self, rhs: Complex[size, type]) -> Complex[size, type]:
        """Multiple two complex values.

        Args:
            rhs: Complex value to multiply with.

        Returns:
            A product of this and RHS complex values.
        """
        return Complex[size, type] {
            re: self.re.fma(rhs.re, -self.im * rhs.im),
            im: self.re.fma(rhs.im, self.im * rhs.re),
        }

    # returns the squared magnitude
    fn norm(self) -> SIMD[size, type.value]:
        """Returns the squared magnitude of the complex value.

        Returns:
            Value of `re*re + im*im`.
        """
        return self.re.fma(self.re, self.im * self.im)

    # fma(self, b, c)
    fn fma(
        self, b: Complex[size, type], c: Complex[size, type]
    ) -> Complex[size, type]:
        """Compute FMA operation.

        Compute fused multiple-add with two other complex values:
        `result = self * b + c`

        Args:
            b: Multiplier complex value.
            c: Complex value to add.

        Returns:
            Computed `Self * B + C` complex value.
        """
        return Complex[size, type](
            self.re.fma(b.re, -self.im.fma(b.im, -c.re)),
            self.re.fma(b.im, self.im.fma(b.re, c.im)),
        )

    # fma(self, self, c)
    fn sq_add(self, c: Complex[size, type]) -> Complex[size, type]:
        """Compute Square-Add operation.

        Compute `Self * Self + C`.

        Args:
            c: Complex value to add.

        Returns:
            Computed `Self * Self + C` complex value.
        """
        return Complex[size, type](
            self.re.fma(self.re, self.im.fma(-self.im, c.re)),
            self.re.fma(self.im + self.im, c.im),
        )
