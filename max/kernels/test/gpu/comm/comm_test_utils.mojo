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
"""Shared utilities for multi-GPU communication tests."""

from math import floor


fn pretty_print_float(val: Float64) -> String:
    """Converts float to string, omitting fractional part if not needed.

    Examples:
        pretty_print_float(2.0) returns "2"
        pretty_print_float(2.5) returns "2.5"
    """
    if Float64(floor(val)) == val:
        return String(Int(val))
    return String(val)


fn human_readable_size(size: Int) -> String:
    """Formats a byte size into human-readable form (KB, MB, GB).

    Args:
        size: Size in bytes.

    Returns:
        Human-readable string (e.g., "4KB", "256MB", "2GB").
    """
    comptime KB = 1024
    comptime MB = KB * KB
    comptime GB = MB * KB

    if size >= GB:
        return pretty_print_float(Float64(size) / GB) + "GB"
    if size >= MB:
        return pretty_print_float(Float64(size) / MB) + "MB"
    if size >= KB:
        return pretty_print_float(Float64(size) / KB) + "KB"
    return String(size) + "B"


@always_inline
@parameter
fn test_value_for_gpu_element[
    dtype: DType,
](gpu_rank: Int, element_idx: Int) -> Scalar[dtype]:
    """Generates unique deterministic test values per GPU and element index.

    Creates predictable values for testing multi-GPU operations where each
    GPU's contribution needs to be distinguishable. Uses prime modulus to
    avoid power-of-two aliasing patterns.

    Args:
        gpu_rank: The rank/ID of the GPU (0-indexed).
        element_idx: The element index within the buffer.

    Returns:
        A unique scalar value for this GPU and element combination.

    Examples:
        test_value_for_gpu_element[DType.float32](0, 0) !=
        test_value_for_gpu_element[DType.float32](1, 0)
    """
    # 251 is the largest prime < 256; using a prime avoids power-of-two aliasing.
    return Scalar[dtype](
        Scalar[dtype](gpu_rank + 1) + Scalar[dtype](element_idx % 251)
    )
