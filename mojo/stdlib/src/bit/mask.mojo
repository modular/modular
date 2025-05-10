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
from bit.mask import BitMask
```
"""

from os import abort
from sys.info import bitwidthof


struct BitMask:
    """Utils for building bitmasks."""

    @always_inline
    @staticmethod
    fn is_negative(value: Int) -> Int:
        """Get a bitmask of whether the value is negative.

        Args:
            value: The value to check.

        Returns:
            A bitmask filled with `1` if the value is negative, filled with `0`
            otherwise.
        """
        return Int(Self.is_negative(Scalar[DType.index](value)))

    @always_inline
    @staticmethod
    fn is_negative[D: DType](value: SIMD[D, _]) -> __type_of(value):
        """Get a bitmask of whether the value is negative.

        Parameters:
            D: The DType.

        Args:
            value: The value to check.

        Returns:
            A bitmask filled with `1` if the value is negative, filled with `0`
            otherwise.
        """
        constrained[
            D.is_integral() and D.is_signed(),
            "This function is for signed integral types.",
        ]()
        return value >> (bitwidthof[D]() - 1)

    @always_inline
    @staticmethod
    fn is_true[
        D: DType, size: Int = 1
    ](value: SIMD[DType.bool, size]) -> SIMD[D, size]:
        """Get a bitmask of whether the value is `True`.

        Parameters:
            D: The DType.
            size: The size of the SIMD vector.

        Args:
            value: The value to check.

        Returns:
            A bitmask filled with `1` if the value is `True`, filled with `0`
            otherwise.
        """
        return (-(value.cast[DType.int8]())).cast[D]()

    @always_inline
    @staticmethod
    fn is_false[
        D: DType, size: Int = 1
    ](value: SIMD[DType.bool, size]) -> SIMD[D, size]:
        """Get a bitmask of whether the value is `False`.

        Parameters:
            D: The DType.
            size: The size of the SIMD vector.

        Args:
            value: The value to check.

        Returns:
            A bitmask filled with `1` if the value is `False`, filled with `0`
            otherwise.
        """
        return Self.is_true[D](~value)
