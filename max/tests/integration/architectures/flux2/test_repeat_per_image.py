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
"""Tests repeat_per_image, which sizes FLUX.2 per-prompt tensors per image."""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Buffer
from max.pipelines.architectures.flux2.flux2_executor import repeat_per_image


def test_one_image_returns_the_buffer() -> None:
    buffer = Buffer.from_dlpack(np.ones((1, 3, 2), dtype=np.float32))
    assert repeat_per_image(buffer, 1) is buffer


def test_repeats_the_row_per_image() -> None:
    row = np.arange(12, dtype=np.float32).reshape(1, 4, 3)
    repeated = repeat_per_image(Buffer.from_dlpack(row), 3)
    np.testing.assert_array_equal(
        np.from_dlpack(repeated), np.repeat(row, 3, axis=0)
    )


def test_repeats_a_rank_one_buffer() -> None:
    repeated = repeat_per_image(
        Buffer.from_dlpack(np.array([0.5], dtype=np.float32)), 2
    )
    np.testing.assert_array_equal(np.from_dlpack(repeated), [0.5, 0.5])


def test_repeats_an_empty_buffer() -> None:
    empty = Buffer.from_dlpack(np.zeros((1, 0, 4), dtype=np.int64))
    repeated = repeat_per_image(empty, 2)
    assert repeated.shape == (2, 0, 4)
    assert repeated.dtype == empty.dtype


def test_rejects_more_than_one_row() -> None:
    buffer = Buffer.from_dlpack(np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="one row"):
        repeat_per_image(buffer, 2)
