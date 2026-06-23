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
"""Tests for Gemma4 batched-vision buffer merging."""

from __future__ import annotations

import numpy as np
from max.driver import CPU, Buffer
from max.pipelines.architectures.gemma4.batch_vision_inputs import (
    merge_per_device_buffers,
)

_HIDDEN = 4


def _buf(rows: int) -> Buffer:
    """A rank-2 [rows, _HIDDEN] CPU buffer with distinct, recoverable values."""
    data = np.arange(rows * _HIDDEN, dtype=np.float32).reshape(rows, _HIDDEN)
    return Buffer.from_numpy(data).to(CPU())


def test_merge_concatenates_rank2_buffers() -> None:
    # Regression test: the on-device concat indexes ``combined`` with a leading
    # slice (``combined[:a_rows, :]``). Indexing a rank-2 buffer with a single
    # index raised "the provided number of indices (1) is not equal to the
    # tensor rank (2)" and crashed the model worker on any batch with 2+ images.
    a, b = _buf(2), _buf(3)
    a_np, b_np = a.to_numpy().copy(), b.to_numpy().copy()

    [merged] = merge_per_device_buffers([a], [b])

    out = merged.to_numpy()
    assert out.shape == (5, _HIDDEN)
    np.testing.assert_array_equal(out[:2], a_np)
    np.testing.assert_array_equal(out[2:], b_np)


def _buf1d(rows: int) -> Buffer:
    """A rank-1 [rows] CPU buffer (the scatter-index shape)."""
    data = np.arange(rows, dtype=np.int32)
    return Buffer.from_numpy(data).to(CPU())


def test_merge_concatenates_rank1_buffers() -> None:
    # Scatter indices are rank-1; regression for the rank-2-only slice that
    # crashed this caller with "indices (2) != tensor rank (1)".
    a, b = _buf1d(2), _buf1d(3)
    a_np, b_np = a.to_numpy().copy(), b.to_numpy().copy()

    [merged] = merge_per_device_buffers([a], [b])

    out = merged.to_numpy()
    assert out.shape == (5,)
    np.testing.assert_array_equal(out[:2], a_np)
    np.testing.assert_array_equal(out[2:], b_np)


def test_merge_is_elementwise_across_devices() -> None:
    # Two per-device replicas must be concatenated pairwise.
    a0, a1 = _buf(1), _buf(2)
    b0, b1 = _buf(2), _buf(1)

    merged = merge_per_device_buffers([a0, a1], [b0, b1])

    assert len(merged) == 2
    assert merged[0].to_numpy().shape == (3, _HIDDEN)
    assert merged[1].to_numpy().shape == (3, _HIDDEN)


def test_merge_returns_other_side_when_one_is_empty() -> None:
    a, empty = _buf(2), _buf(0)

    # Empty right side -> left returned untouched.
    assert merge_per_device_buffers([a], [empty]) == [a]
    # Empty left side -> right returned untouched.
    assert merge_per_device_buffers([empty], [a]) == [a]
    # Both empty -> first returned.
    assert merge_per_device_buffers([empty], [empty]) == [empty]
