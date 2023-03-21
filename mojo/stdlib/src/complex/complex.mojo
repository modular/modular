# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from DType import DType
from SIMD import SIMD
from Int import Int


@register_passable
struct Complex[size: Int, type: DType]:
    var re: SIMD[size, type.value]
    var im: SIMD[size, type.value]

    fn __clone__(self) -> Self:
        return Self {re: self.re, im: self.im}

    fn __init__(
        re: SIMD[size, type.value],
        im: SIMD[size, type.value],
    ) -> Complex[size, type]:
        return Complex[size, type] {re: re, im: im}

    fn __add__(self, rhs: Complex[size, type]) -> Complex[size, type]:
        return Complex[size, type] {re: self.re + rhs.re, im: self.im + rhs.im}

    fn __mul__(self, rhs: Complex[size, type]) -> Complex[size, type]:
        return Complex[size, type] {
            re: self.re.fma(rhs.re, -self.im * rhs.im),
            im: self.re.fma(rhs.im, self.im * rhs.re),
        }

    # returns the squared magnitude
    fn norm(self) -> SIMD[size, type.value]:
        return self.re.fma(self.re, self.im * self.im)

    # fma(self, b, c)
    fn fma(
        self, b: Complex[size, type], c: Complex[size, type]
    ) -> Complex[size, type]:
        return Complex[size, type](
            self.re.fma(b.re, -self.im.fma(b.im, -c.re)),
            self.re.fma(b.im, self.im.fma(b.re, c.im)),
        )

    # fma(self, self, c)
    fn sq_add(self, c: Complex[size, type]) -> Complex[size, type]:
        return Complex[size, type](
            self.re.fma(self.re, self.im.fma(-self.im, c.re)),
            self.re.fma(self.im + self.im, c.im),
        )
