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

"""Implements the fast division algorithm.

This method replaces division by constants with a sequence of shifts and
multiplications, significantly optimizing division performance.
"""

from sys import bit_width_of
from bit import log2_ceil
from builtin.dtype import _uint_type_of_width
from gpu.intrinsics import mulhi


@register_passable("trivial")
struct FastDiv[dtype: DType](Stringable, Writable):
    """Implements fast division for a given type.

    This struct provides optimized division by a constant divisor,
    replacing the division operation with a series of shifts and
    multiplications. This approach significantly improves performance,
    especially in scenarios where division is a frequent operation.
    """

    alias uint_type = _uint_type_of_width[bit_width_of[dtype]()]()

    var _div: Scalar[Self.uint_type]
    var _mprime: Scalar[Self.uint_type]
    var _sh1: UInt8
    var _sh2: UInt8
    var _is_pow2: Bool
    var _log2_shift: UInt8

    @always_inline
    fn __init__(out self, divisor: Int = 1):
        """Initializes FastDiv with the divisor.

        Constraints:
            ConstraintError: If the bitwidth of the type is > 32.

        Args:
            divisor: The divisor to use for fast division.
                Defaults to 1.
        """
        constrained[
            bit_width_of[dtype]() <= 32,
            "larger types are not currently supported",
        ]()
        self._div = divisor

        self._is_pow2 = divisor.is_power_of_two()
        self._log2_shift = log2_ceil(Int32(divisor)).cast[DType.uint8]()

        # Only compute magic number parameters if not power of 2
        if not self._is_pow2:
            self._mprime = (
                (
                    (UInt64(1) << bit_width_of[dtype]())
                    * ((1 << self._log2_shift.cast[DType.uint64]()) - divisor)
                    / divisor
                )
            ).cast[Self.uint_type]() + 1
            self._sh1 = min(self._log2_shift, 1)
            self._sh2 = max(self._log2_shift - 1, 0)
        else:
            self._mprime = 0
            self._sh1 = 0
            self._sh2 = 0

    @always_inline
    fn __rdiv__(self, other: Scalar[Self.uint_type]) -> Scalar[Self.uint_type]:
        """Divides the other scalar by the divisor.

        Args:
            other: The dividend.

        Returns:
            The result of the division.
        """
        return other / self

    @always_inline
    fn __rtruediv__(
        self, other: Scalar[Self.uint_type]
    ) -> Scalar[Self.uint_type]:
        """Divides the other scalar by the divisor (true division).

        Uses the fast division algorithm, with optimized path for power-of-2 divisors.

        Args:
            other: The dividend.

        Returns:
            The result of the division.
        """
        if self._is_pow2:
            # For power-of-2 divisors, just use bit shift
            return other >> self._log2_shift.cast[Self.uint_type]()
        else:
            # FastDiv algorithm for non-power-of-2 divisors.
            var t = mulhi(
                self._mprime.cast[DType.uint32](), other.cast[DType.uint32]()
            ).cast[Self.uint_type]()
            return (
                t + ((other - t) >> self._sh1.cast[Self.uint_type]())
            ) >> self._sh2.cast[Self.uint_type]()

    @always_inline
    fn __rmod__(self, other: Scalar[Self.uint_type]) -> Scalar[Self.uint_type]:
        """Computes the remainder of division.

        Args:
            other: The dividend.

        Returns:
            The remainder.
        """
        var q = other / self
        return other - (q * self._div)

    @always_inline
    fn __divmod__(
        self, other: Scalar[Self.uint_type]
    ) -> (Scalar[Self.uint_type], Scalar[Self.uint_type]):
        """Computes both quotient and remainder.

        Args:
            other: The dividend.

        Returns:
            A tuple containing the quotient and remainder.
        """
        var q = other / self
        return q, (other - (q * self._div))

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        """Writes the FastDiv parameters to a writer.

        Args:
            writer: The writer to which the parameters are written.
        """
        writer.write("div: ", self._div, "\n")
        writer.write("mprime: ", self._mprime, "\n")
        writer.write("sh1: ", self._sh1, "\n")
        writer.write("sh2: ", self._sh2, "\n")
        writer.write("is_pow2: ", self._is_pow2, "\n")
        writer.write("log2_shift: ", self._log2_shift, "\n")

    @no_inline
    fn __str__(self) -> String:
        """Get the object as a string.

        Returns:
            A string representation.
        """
        return String.write(self)
