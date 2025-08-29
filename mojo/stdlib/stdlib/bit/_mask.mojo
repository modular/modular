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
"""Provides functions for bit masks.

You can import these APIs from the `bit` package. For example:

```mojo
from bit.mask import is_negative
```
"""

from sys.info import bit_width_of


@always_inline
fn is_negative(value: Int) -> Int:
    """Get a bitmask of whether the value is negative.

    Args:
        value: The value to check.

    Returns:
        A bitmask filled with `1` if the value is negative, filled with `0`
        otherwise.
    """
    return Int(is_negative(Scalar[DType.index](value)))


@always_inline
fn is_negative[dtype: DType, //](value: SIMD[dtype, _]) -> __type_of(value):
    """Get a bitmask of whether the value is negative.

    Parameters:
        dtype: The DType.

    Args:
        value: The value to check.

    Returns:
        A bitmask filled with `1` if the value is negative, filled with `0`
        otherwise.
    """
    constrained[
        dtype.is_integral() and dtype.is_signed(),
        "This function is for signed integral types.",
    ]()

    # HACK(#5003): remove this workaround
    alias d = dtype if dtype is not DType.index else (
        DType.int32 if dtype.size_of() == 4 else DType.int64
    )
    return (value.cast[d]() >> (bit_width_of[d]() - 1)).cast[dtype]()


@always_inline
fn splat[
    size: Int, //, dtype: DType
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
fn splat(value: Bool) -> Int:
    """Get a bitmask of whether the value is `True`.

    Args:
        value: The value to check.

    Returns:
        A bitmask filled with `1` if the value is `True`, filled with `0`
        otherwise.
    """
    return Int(splat[DType.index](Scalar[DType.bool](value)))
