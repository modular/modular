# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Implements the fast division algorithm.

This method replaces division by constants with a sequence of shifts and
multiplications, significantly optimizing division performance.
"""

from sys import bitwidthof, is_nvidia_gpu

from builtin.dtype import _uint_type_of_width
from gpu.intrinsics import mulhi


@always_inline
fn _ceillog2(x: Scalar) -> Int32:
    """Computes ceil(log_2(d))."""

    @parameter
    for i in range(bitwidthof[x.dtype]()):
        if (__type_of(x)(1) << i) >= x:
            return i
    return bitwidthof[x.dtype]()


@register_passable("trivial")
struct FastDiv[type: DType]:
    """Implements fast division for a given type.

    This struct provides optimized division by a constant divisor,
    replacing the division operation with a series of shifts and
    multiplications. This approach significantly improves performance,
    especially in scenarios where division is a frequent operation.
    """

    alias uint_type = _uint_type_of_width[bitwidthof[type]()]()

    var _div: Scalar[Self.uint_type]
    var _mprime: Scalar[Self.uint_type]
    var _sh1: Int32
    var _sh2: Int32

    @always_inline
    @implicit
    fn __init__(out self, divisor: Int = 1):
        """Initializes FastDiv with the divisor.

        Constraints:
            ConstraintError: If the bitwidth of the type is > 32.

        Args:
            divisor: The divisor to use for fast division.
                Defaults to 1.
        """
        constrained[
            bitwidthof[type]() <= 32, "larger types are not currently supported"
        ]()
        self._div = divisor

        var cl = _ceillog2(UInt32(divisor))
        self._mprime = (
            (
                (UInt64(1) << bitwidthof[type]())
                * ((1 << cl.cast[DType.uint64]()) - divisor)
                / divisor
            )
        ).cast[Self.uint_type]() + 1
        self._sh1 = min(cl, 1)
        self._sh2 = max(cl - 1, 0)

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

        Uses the fast division algorithm.

        Args:
            other: The dividend.

        Returns:
            The result of the division.
        """
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
