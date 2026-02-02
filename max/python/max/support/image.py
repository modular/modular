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

"""Image-related utilities."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from max._core import xxhash


def find_contiguous_ranges(
    arr: npt.NDArray[np.integer[Any]], targets: Sequence[int]
) -> list[tuple[int, int]]:
    """Find the contiguous ranges of the given token in the array

    eg:
        find_contiguous_ranges([1, 2, 99, 99, 3, 99, 98, 99], [98, 99]) -> [(2, 4), (5, 8)]
    """

    if arr.ndim != 1:
        raise ValueError(f"Array must be 1D, found {arr.shape}")

    # Boolean mask where arr == x
    mask = np.isin(arr, targets)
    # Find where mask changes value (True <-> False)
    diff = np.diff(mask.astype(int))
    # Start indices are where it changes from 0 -> 1
    starts = np.where(diff == 1)[0] + 1
    # End indices are where it changes from 1 -> 0
    ends = np.where(diff == -1)[0] + 1

    # Handle if the sequence starts or ends with True
    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        ends = np.concatenate([ends, [len(mask)]])

    # Cast values from int64 -> int
    starts = starts.tolist()
    ends = ends.tolist()

    return list(zip(starts, ends, strict=True))


def hash_image(pixel_values: npt.NDArray[Any]) -> int:
    """Compute the hash of an image.

    Supports any numpy array dtype (float32, uint16 for bfloat16 bits, etc.)
    since vision models may use different storage formats on CPU.

    Uses xxhash for fast hashing. Ensures C-contiguous memory layout for
    correct hashing (np.ascontiguousarray is a no-op if already contiguous).
    """
    hash_val = xxhash.xxh3_64_intdigest(np.ascontiguousarray(pixel_values).data)  # type: ignore[arg-type]
    # xxh3_64_intdigest returns unsigned 64-bit int; convert to signed for numpy compatibility
    return int(np.uint64(hash_val).astype(np.int64))
