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
"""Utility functions for graph operations."""

from collections.abc import Callable


def _axis_out_of_range_error(
    axis: int, lower_bound: int, upper_bound: int, axis_name: str = ""
) -> str:
    if axis_name:
        axis_name = f"{axis_name} "
    return f"Axis {axis_name}out of range (expected to be in range of [{lower_bound}, {upper_bound}], but got {axis})"


def check_axis_in_bounds(
    axis: int,
    rank: int,
    bounds_func: Callable[[int], tuple[int, int]],
    axis_name: str = "",
) -> None:
    """Check if an axis is within bounds for a given rank.

    Args:
        axis: The axis value to check.
        rank: The rank of the tensor.
        bounds_func: A function that takes a rank and returns a tuple of (lower_bound, upper_bound).
        axis_name: Optional name of the axis for more descriptive error messages.
            Defaults to empty string.

    Raises:
        IndexError: If the axis is out of bounds.
    """
    lower_bound, upper_bound = bounds_func(rank)
    if axis < lower_bound or axis > upper_bound:
        raise IndexError(
            _axis_out_of_range_error(axis, lower_bound, upper_bound, axis_name)
        )
