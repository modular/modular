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
"""Implements partial SIMD load and store utilities."""

from std.math import iota
from std.sys.intrinsics import (
    masked_load,
    masked_store,
)


@always_inline
def partial_simd_load[
    dtype: DType, //, width: Int
](
    storage: UnsafePointer[mut=False, Scalar[dtype], ...],
    lbound: Int,
    rbound: Int,
    pad_value: Scalar[dtype],
) -> SIMD[dtype, width]:
    """Loads a vector with dynamic bound.

    Out of bound data will be filled with pad value. Data is valid if
    lbound <= idx < rbound for idx from 0 to (simd_width-1). For example:

        addr 0  1  2  3
        data x 42 43  x

        partial_simd_load[4](addr0, 1, 3) #gives [0 42 43 0]

    Parameters:
        dtype: The DType of storage.
        width: The system simd vector size.

    Args:
        storage: Pointer to the address to perform load.
        lbound: Lower bound of valid index within simd (inclusive).
        rbound: Upper bound of valid index within simd (non-inclusive).
        pad_value: Value to fill for out of bound indices.

    Returns:
        The SIMD vector loaded and zero-filled.
    """
    # Create a mask based on input bounds.
    var effective_lbound = SIMD[DType.int32, width](max(lbound, 0))
    var effective_rbound = SIMD[DType.int32, width](min(width, rbound))
    var incr = iota[DType.int32, width]()
    var mask = incr.ge(effective_lbound) & incr.lt(effective_rbound)

    return masked_load[width](storage, mask, pad_value)


@always_inline
def partial_simd_store[
    dtype: DType, //, width: Int
](
    storage: UnsafePointer[mut=True, Scalar[dtype], ...],
    lbound: Int,
    rbound: Int,
    data: SIMD[dtype, width],
):
    """Stores a vector with dynamic bound.

    Out of bound data will ignored. Data is valid if lbound <= idx < rbound for
    idx from 0 to (simd_width-1).

    e.g.
        addr 0 1 2  3
        data 0 0 0  0

        partial_simd_load[4](addr0, 1, 3, [-1, 42, 43, -1]) #gives [0 42 43 0]

    Parameters:
        dtype: The DType of storage.
        width: The system simd vector size.

    Args:
        storage: Pointer to the address to perform load.
        lbound: Lower bound of valid index within simd (inclusive).
        rbound: Upper bound of valid index within simd (non-inclusive).
        data: The vector value to store.
    """
    # Create a mask based on input bounds.
    var effective_lbound = SIMD[DType.int32, width](max(lbound, 0))
    var effective_rbound = SIMD[DType.int32, width](min(width, rbound))
    var incr = iota[DType.int32, width]()
    var mask = incr.ge(effective_lbound) & incr.lt(effective_rbound)

    return masked_store(data, storage, mask)
