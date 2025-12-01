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

    # Flatten for processing.
    original_shape = arr.shape
    flat = arr.ravel()

    # View as uint32 for bit manipulation.
    uint32_view = flat.view(np.uint32)

    # Round to nearest even.
    round_bit = (uint32_view >> 16) & 1
    lower_half = uint32_view & 0xFFFF
    round_up = (lower_half > 0x8000) | (
        (lower_half == 0x8000) & (round_bit == 1)
    )
    uint32_rounded = uint32_view + (round_up.astype(np.uint32) * 0x8000)

    # Extract upper 16 bits as bfloat16.
    bfloat16_bits = (uint32_rounded >> 16).astype(np.uint16)

    # Restore original shape.
    return bfloat16_bits.reshape(original_shape)
