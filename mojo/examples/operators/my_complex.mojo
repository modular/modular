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

from std.math import sqrt


struct Complex(
    Boolable,
    Equatable,
    TrivialRegisterPassable,
    Writable,
):
    """Represents a complex value.

    The struct provides basic methods for manipulating complex values.
    """

    # ===------------------------------------------------------------------=== #
    # Fields
    # ===------------------------------------------------------------------=== #

    var re: Float64
    var im: Float64

    # ===------------------------------------------------------------------=== #
    # Initializers
    # ===------------------------------------------------------------------=== #

    @implicit
    def __init__(out self, re: Float64, im: Float64 = 0.0):
        self.re = re
        self.im = im

    @implicit
    def __init__(out self, re: IntLiteral):
        self = Self(Float64(re))

    # ===------------------------------------------------------------------=== #
    # Trait implementations
    # ===------------------------------------------------------------------=== #

    def write_to(self, mut writer: Some[Writer]):
        writer.write("(", self.re)
        if self.im < 0:
            writer.write(" - ", -self.im)
        else:
            writer.write(" + ", self.im)
        writer.write("i)")

    def write_repr_to(self, mut writer: Some[Writer]):
        t"Complex(re = {self.re}, im = {self.im})".write_to(writer)

    def __bool__(self) -> Bool:
        return self != 0

    # ===------------------------------------------------------------------=== #
    # Indexing
    # ===------------------------------------------------------------------=== #

    def __getitem_param__[idx: Int](ref self) -> ref[self] Float64:
        comptime assert idx in (0, 1), "idx must be 0 or 1"

        comptime if idx == 0:
            var p = UnsafePointer(to=self.re).unsafe_origin_cast[
                origin_of(self)
            ]()
            return p[]
        else:
            var p = UnsafePointer(to=self.im).unsafe_origin_cast[
                origin_of(self)
            ]()
            return p[]

    # ===------------------------------------------------------------------=== #
    # Unary arithmetic operator dunders
    # ===------------------------------------------------------------------=== #

    def __neg__(self) -> Self:
        return Self(-self.re, -self.im)

    def __pos__(self) -> Self:
        return self

    # ===------------------------------------------------------------------=== #
    # Binary arithmetic operator dunders
    # ===------------------------------------------------------------------=== #

    def __add__(self, rhs: Self) -> Self:
        return Self(self.re + rhs.re, self.im + rhs.im)

    def __radd__(self, lhs: Float64) -> Self:
        return self + lhs

    def __iadd__(mut self, rhs: Self):
        self = self + rhs

    def __sub__(self, rhs: Self) -> Self:
        return Self(self.re - rhs.re, self.im - rhs.im)

    def __rsub__(self, lhs: Float64) -> Self:
        return Self(lhs) - self

    def __isub__(mut self, rhs: Self):
        self = self - rhs

    def __mul__(self, rhs: Self) -> Self:
        return Self(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )

    def __rmul__(self, lhs: Float64) -> Self:
        return self * lhs

    def __imul__(mut self, rhs: Self):
        self = self * rhs

    def __truediv__(self, rhs: Self) -> Self:
        denom = rhs.squared_norm()
        return Self(
            (self.re * rhs.re + self.im * rhs.im) / denom,
            (self.im * rhs.re - self.re * rhs.im) / denom,
        )

    def __rtruediv__(self, lhs: Float64) -> Self:
        return Self(lhs) / self

    def __itruediv__(mut self, rhs: Self):
        self = self / rhs

    # ===------------------------------------------------------------------=== #
    # Equality comparison operator dunders
    # ===------------------------------------------------------------------=== #

    def __eq__(self, other: Self) -> Bool:
        return self.re == other.re and self.im == other.im

    def __ne__(self, other: Self) -> Bool:
        return not self == other

    # ===------------------------------------------------------------------=== #
    # Methods
    # ===------------------------------------------------------------------=== #

    def squared_norm(self) -> Float64:
        return self.re * self.re + self.im * self.im

    def norm(self) -> Float64:
        return sqrt(self.squared_norm())
