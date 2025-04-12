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

from math import sqrt


@value
@register_passable("trivial")
struct Complex(
    Boolable,
    EqualityComparable,
    Writable,
    Representable,
    Stringable,
):
    """Represents a complex value.

    The struct provides basic methods for manipulating complex values.
    """

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var re: Float64
    var im: Float64

    # ===-------------------------------------------------------------------===#
    # Initializers
    # ===-------------------------------------------------------------------===#

    @implicit
    fn __init__(out self, re: Float64, im: Float64 = 0.0):
        self.re = re
        self.im = im

    @implicit
    fn __init__(out self, re: IntLiteral):
        self = Self(Float64(re))

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    fn __repr__(self) -> String:
        return String("Complex(re = ", self.re, ", im = ", self.im, ")")

    fn __str__(self) -> String:
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("(", self.re)
        if self.im < 0:
            writer.write(" - ", -self.im)
        else:
            writer.write(" + ", self.im)
        writer.write("i)")

    fn __bool__(self) -> Bool:
        return self != 0

    # ===-------------------------------------------------------------------===#
    # Indexing
    # ===-------------------------------------------------------------------===#

    fn __getitem__(self, idx: Int) raises -> Float64:
        if idx == 0:
            return self.re
        elif idx == 1:
            return self.im
        else:
            raise "index out of bounds"

    fn __setitem__(mut self, idx: Int, value: Float64) raises:
        if idx == 0:
            self.re = value
        elif idx == 1:
            self.im = value
        else:
            raise "index out of bounds"

    # ===-------------------------------------------------------------------===#
    # Unary arithmetic operator dunders
    # ===-------------------------------------------------------------------===#

    fn __neg__(self) -> Self:
        return Self(-self.re, -self.im)

    fn __pos__(self) -> Self:
        return self

    # ===-------------------------------------------------------------------===#
    # Binary arithmetic operator dunders
    # ===-------------------------------------------------------------------===#

    fn __add__(self, rhs: Self) -> Self:
        return Self(self.re + rhs.re, self.im + rhs.im)

    fn __radd__(self, lhs: Float64) -> Self:
        return self + lhs

    fn __iadd__(mut self, rhs: Self):
        self = self + rhs

    fn __sub__(self, rhs: Self) -> Self:
        return Self(self.re - rhs.re, self.im - rhs.im)

    fn __rsub__(self, lhs: Float64) -> Self:
        return self - lhs

    fn __isub__(mut self, rhs: Self):
        self = self - rhs

    fn __mul__(self, rhs: Self) -> Self:
        return Self(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )

    fn __rmul__(self, lhs: Float64) -> Self:
        return self * lhs

    fn __imul__(mut self, rhs: Self):
        self = self * rhs

    fn __truediv__(self, rhs: Self) -> Self:
        denom = rhs.squared_norm()
        return Self(
            (self.re * rhs.re + self.im * rhs.im) / denom,
            (self.im * rhs.re - self.re * rhs.im) / denom,
        )

    fn __rtruediv__(self, lhs: Float64) -> Self:
        return Self(lhs) / self

    fn __itruediv__(mut self, rhs: Self):
        self = self / rhs

    # ===-------------------------------------------------------------------===#
    # Equality comparison operator dunders
    # ===-------------------------------------------------------------------===#

    fn __eq__(self, other: Self) -> Bool:
        return self.re == other.re and self.im == other.im

    fn __ne__(self, other: Self) -> Bool:
        return not self == other

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn squared_norm(self) -> Float64:
        return self.re * self.re + self.im * self.im

    fn norm(self) -> Float64:
        return sqrt(self.squared_norm())
