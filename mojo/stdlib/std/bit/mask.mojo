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
"""Provides functions for bit masks.

You can import these APIs from the `bit` package. For example:

```mojo
from std.bit.mask import is_negative
```
"""

from std.sys.info import bit_width_of


@always_inline
def is_negative(value: Int) -> Int:
    """Get a bitmask of whether the value is negative.

    Args:
        value: The value to check.

    Returns:
        A bitmask filled with `1` if the value is negative, filled with `0`
        otherwise.
    """
    return Int(is_negative(Scalar[DType.int](value)))


@always_inline
def is_negative[dtype: DType, //](value: SIMD[dtype, _]) -> type_of(value):
    """Get a bitmask of whether the value is negative.

    Parameters:
        dtype: The DType.

    Args:
        value: The value to check.

    Returns:
        A bitmask filled with `1` if the value is negative, filled with `0`
        otherwise.
    """
    comptime assert (
        dtype.is_integral() and dtype.is_signed()
    ), "This function is for signed integral types."
    return value >> SIMD[dtype, value.size](bit_width_of[dtype]() - 1)


@always_inline
def splat[
    size: SIMDSize, //, dtype: DType
](value: SIMD[DType.bool, size]) -> SIMD[dtype, size]:
    """Elementwise splat the boolean value of each element in the SIMD vector
    into all bits of the corresponding element in a new SIMD vector.

    Parameters:
        size: The size of the SIMD vector.
        dtype: The DType of the output.

    Args:
        value: The value to check.

    Returns:
        A SIMD vector where each element is filled with `1` bits if the
        corresponding element in `value` is `True`, or filled with `0` bits
        otherwise.
    """
    return (-(value.cast[DType.int8]())).cast[dtype]()


@always_inline
def splat(value: Bool) -> Int:
    """Get a bitmask of whether the value is `True`.

    Args:
        value: The value to check.

    Returns:
        A bitmask filled with `1` if the value is `True`, filled with `0`
        otherwise.
    """
    return Int(splat[DType.int](Scalar[DType.bool](value)))


@always_inline
def ones[dtype: DType](start: Int, end: Int) -> Scalar[dtype]:
    """Returns a scalar with bits in the half-open range `[start, end)` set to 1
    and all other bits set to 0.

    Parameters:
        dtype: The integer `DType` of the returned scalar. Must be integral.

    Args:
        start: Index of the lowest bit to set (inclusive). Must be non-negative.
        end: Index one past the highest bit to set (exclusive). Must satisfy
             `start < end <= bit_width_of[dtype]()`.

    Returns:
        A `Scalar[dtype]` mask with bits `[start, end)` set.

    Examples:

    ```mojo
    from std.bit.mask import ones
    print(ones[DType.uint8](2, 5))   # 0b00011100 = 28
    print(ones[DType.uint16](0, 8))  # 0x00FF = 255
    ```
    """
    comptime assert dtype.is_integral(), "dtype must be an integral type"
    comptime bitwidth = bit_width_of[dtype]()
    assert start >= 0, "start must be non-negative"
    assert start < end, "start must be strictly less than end"
    assert end <= bitwidth, "end must not exceed the bit width of dtype"
    # When end == bitwidth, shifting 1 left by `end` bits overflows.
    # Use the all-ones complement approach instead.
    if end == bitwidth:
        return ~Scalar[dtype](0) << Scalar[dtype](start)
    return (Scalar[dtype](1) << Scalar[dtype](end)) - (
        Scalar[dtype](1) << Scalar[dtype](start)
    )
