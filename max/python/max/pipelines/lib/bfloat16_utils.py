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
"""Utility functions for vision language models."""

import numpy as np
import numpy.typing as npt


def float32_to_bfloat16_as_uint16(
    arr: npt.NDArray[np.float32],
) -> npt.NDArray[np.uint16]:
    """Convert float32 array to bfloat16 representation stored as uint16.

    BFloat16 is the upper 16 bits of float32 with proper rounding.
    This allows us to halve memory usage while maintaining the exponent range.

    Args:
        arr: Float32 numpy array

    Returns:
        Uint16 array containing bfloat16 bit representation with same shape
    """
    assert arr.dtype == np.float32, f"Expected float32, got {arr.dtype}"

    # Flatten and view as uint32 for bit manipulation.
    uint32_view = arr.ravel().view(np.uint32)

    # Round to nearest even: add (0x7FFF + bit16) then shift right 16.
    # bit16 is the LSB of the upper 16 bits, used for round-to-even on ties.
    # This is equivalent to: round_up when lower > 0x8000, or when
    # lower == 0x8000 and bit16 == 1 (round ties to even).
    rounded = uint32_view + (0x7FFF + ((uint32_view >> 16) & 1))

    # Extract upper 16 bits by viewing as uint16 and taking every other element.
    # On little-endian systems, upper 16 bits are at odd indices in uint16 view.
    # Copy to return a contiguous array that owns its data (slicing creates a view).
    return rounded.view(np.uint16)[1::2].reshape(arr.shape).copy()
