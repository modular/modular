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
"""Utilities for doing unsigned division and modulo operations on `Int`.

These helpers allow performing unsigned division and modulo operations on Int
arguments. This can provide better performance, as these operations can be
slower when performed using signed integers on some accelerators, as correctly
handling negative values requires additional instructions.
"""


@always_inline
def ufloordiv(a: Int, b: Int) -> Int:
    """Perform unsigned floor division (`//`) on Int arguments.

    This function treats both arguments as unsigned values and performs
    unsigned division, which is faster than signed division on NVIDIA GPUs.

    For correctness, both arguments should be non-negative integers.

    Args:
        a: The dividend (treated as unsigned).
        b: The divisor (treated as unsigned).

    Returns:
        The quotient of unsigned division.
    """
    return Int(UInt(a) // UInt(b))


@always_inline
def umod(a: Int, b: Int) -> Int:
    """Perform unsigned modulo (`%`) on Int arguments.

    This function treats both arguments as unsigned values and performs
    unsigned modulo, which is faster than signed modulo on NVIDIA GPUs.

    For correctness, both arguments should be non-negative integers.

    Args:
        a: The dividend (treated as unsigned).
        b: The divisor (treated as unsigned).

    Returns:
        The remainder of unsigned division.
    """
    return Int(UInt(a) % UInt(b))


@always_inline
def udivmod(a: Int, b: Int) -> Tuple[Int, Int]:
    """Perform unsigned divmod on Int arguments.

    Computes the quotient and remainder in a single unsigned division,
    which is faster than signed divmod on NVIDIA GPUs.

    For correctness, both arguments should be non-negative integers.

    Args:
        a: The dividend (treated as unsigned).
        b: The divisor (treated as unsigned).

    Returns:
        A `Tuple` of `(quotient, remainder)` from unsigned division.
    """
    var ua = UInt(a)
    var ub = UInt(b)
    var q, r = divmod(ua, ub)
    return Int(q), Int(r)


@always_inline
def ualign_up(value: Int, alignment: Int) -> Int:
    """Returns the closest multiple of alignment that is greater than or equal
    to value.

    Args:
        value: The value to align.
        alignment: Value to align to.

    Returns:
        Closest multiple of the alignment that is greater than or equal to the
        input value. In other words, ceiling(value / alignment) * alignment.
    """
    return Int(align_up(UInt(value), UInt(alignment)))


@always_inline
def ualign_down(value: Int, alignment: Int) -> Int:
    """Returns the closest multiple of alignment that is less than or equal to
    value.

    Args:
        value: The value to align.
        alignment: Value to align to.

    Returns:
        Closest multiple of the alignment that is less than or equal to the
        input value. In other words, floor(value / alignment) * alignment.
    """
    return Int(align_down(UInt(value), UInt(alignment)))


@always_inline
def uceildiv(numerator: Int, denominator: Int) -> Int:
    """Return the rounded-up result of dividing numerator by denominator.

    Args:
        numerator: The numerator.
        denominator: The denominator.

    Returns:
        The ceiling of dividing numerator by denominator.
    """
    return Int(UInt(numerator).__ceildiv__(UInt(denominator)))
