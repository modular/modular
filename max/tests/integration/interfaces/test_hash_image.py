#!/usr/bin/env python3
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
"""Unit tests for hash_image function."""

import numpy as np
from max.support.image import hash_image


def test_hash_image_contiguous() -> None:
    """Test hash_image with C-contiguous arrays."""
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    assert arr.flags["C_CONTIGUOUS"]

    h1 = hash_image(arr)
    h2 = hash_image(arr)
    assert h1 == h2, "Same array should produce same hash"

    # Different array should produce different hash
    arr2 = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)
    h3 = hash_image(arr2)
    assert h1 != h3, "Different arrays should produce different hashes"


def test_hash_image_non_contiguous() -> None:
    """Test hash_image with non-contiguous arrays (transposed)."""
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    arr_t = arr.T
    assert not arr_t.flags["C_CONTIGUOUS"]

    h1 = hash_image(arr_t)
    h2 = hash_image(arr_t)
    assert h1 == h2, "Same non-contiguous array should produce same hash"


def test_hash_image_contiguous_vs_non_contiguous_same_data() -> None:
    """Test that contiguous copy produces same hash as non-contiguous original."""
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    arr_t = arr.T  # Non-contiguous
    arr_t_copy = np.ascontiguousarray(arr_t)  # Contiguous copy of transposed

    assert not arr_t.flags["C_CONTIGUOUS"]
    assert arr_t_copy.flags["C_CONTIGUOUS"]
    assert np.array_equal(arr_t, arr_t_copy)

    # Both should produce the same hash since they represent the same data
    h1 = hash_image(arr_t)
    h2 = hash_image(arr_t_copy)
    assert h1 == h2, "Contiguous and non-contiguous with same data should match"
