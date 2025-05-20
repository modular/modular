# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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

from sys.info import bitwidthof


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
    return value >> (bitwidthof[dtype]() - 1)


@always_inline
fn is_true[
    dtype: DType, size: Int = 1
](value: SIMD[DType.bool, size]) -> SIMD[dtype, size]:
    """Get a bitmask of whether the value is `True`.

    Parameters:
        dtype: The DType.
        size: The size of the SIMD vector.

    Args:
        value: The value to check.

    Returns:
        A bitmask filled with `1` if the value is `True`, filled with `0`
        otherwise.
    """
    return (-(value.cast[DType.int8]())).cast[dtype]()


@always_inline
fn is_false[
    dtype: DType, size: Int = 1
](value: SIMD[DType.bool, size]) -> SIMD[dtype, size]:
    """Get a bitmask of whether the value is `False`.

    Parameters:
        dtype: The DType.
        size: The size of the SIMD vector.

    Args:
        value: The value to check.

    Returns:
        A bitmask filled with `1` if the value is `False`, filled with `0`
        otherwise.
    """
    return is_true[dtype](~value)
