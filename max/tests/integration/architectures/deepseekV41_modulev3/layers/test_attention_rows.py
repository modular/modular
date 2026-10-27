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
"""The ragged row bookkeeping against host arithmetic.

``row_ids`` derives each row's request index and its index within it with
the histogram-cumsum trick; every consumer of the forward's
``RaggedRows`` -- the store slots, the window lists, the positions, the
compressor's ring -- reads its exact output. The offsets below cover the
shapes a flat row-to-request mapping gets wrong: a zero-length request
(its end coincides with the next request's start) and a captured-decode
padded tail (rows past ``row_offsets[-1]`` pin to the last request).
``position_rope_table`` is checked against the same absolute positions the
bookkeeping produces.
"""

from __future__ import annotations

import numpy as np
from max.experimental.tensor import Tensor
from max.pipelines.architectures.deepseekV41_modulev3.layers.attention import (
    absolute_positions,
    position_rope_table,
    row_ids,
)


def _t32(x: np.ndarray) -> Tensor:
    return Tensor.from_dlpack(np.ascontiguousarray(x, dtype=np.int32))


def _oracle(offsets: list[int], total: int) -> tuple[np.ndarray, np.ndarray]:
    """Host bid/local: one request index per row, then the index within it."""
    bid = np.zeros(total, dtype=np.int32)
    for r in range(len(offsets) - 1):
        bid[offsets[r] : offsets[r + 1]] = r
    # Rows at or past the last end (zero-length requests' rows and the
    # padded tail) pin to the last request.
    bid[offsets[-1] :] = len(offsets) - 2
    local = (
        np.arange(total, dtype=np.int32)
        - np.asarray(offsets, dtype=np.int32)[bid]
    )
    return bid, local


def test_row_ids_names_each_row_request_and_local() -> None:
    """A normal request, a zero-length middle request, and a padded tail."""
    offsets = [0, 3, 3, 7]
    total = 10  # rows 7..9 are capture padding
    bid, local = row_ids(_t32(np.array(offsets, dtype=np.uint32)), total)
    want_bid, want_local = _oracle(offsets, total)
    np.testing.assert_array_equal(bid.to_numpy(), want_bid)
    np.testing.assert_array_equal(local.to_numpy(), want_local)


def test_row_ids_keeps_rows_of_zero_length_end_requests() -> None:
    """Ends at exactly the row count land in the histogram's sink slot."""
    offsets = [0, 2, 2, 2]
    bid, local = row_ids(_t32(np.array(offsets, dtype=np.uint32)), 2)
    want_bid, want_local = _oracle(offsets, 2)
    np.testing.assert_array_equal(bid.to_numpy(), want_bid)
    np.testing.assert_array_equal(local.to_numpy(), want_local)


def test_padded_rows_of_a_single_request_stay_in_it() -> None:
    """A lone request's padded tail must not read one past the batch."""
    offsets = [0, 1]
    bid, local = row_ids(_t32(np.array(offsets, dtype=np.uint32)), 4)
    want_bid, want_local = _oracle(offsets, 4)
    np.testing.assert_array_equal(bid.to_numpy(), want_bid)
    np.testing.assert_array_equal(local.to_numpy(), want_local)


def test_absolute_positions_start_at_each_requests_cache_length() -> None:
    """Positions are the request's cache length plus the local index."""
    offsets = [0, 3, 3, 7]
    total = 10
    lengths = np.array([5, 100, 11], dtype=np.int32)
    got = absolute_positions(
        _t32(np.array(offsets, dtype=np.uint32)), _t32(lengths), total
    ).to_numpy()
    want_bid, want_local = _oracle(offsets, total)
    np.testing.assert_array_equal(got, lengths[want_bid] + want_local)


def test_position_rope_table_covers_positions_then_padding() -> None:
    """One row per position up to the context bound plus the row tail."""
    inv = 10000.0 ** (-np.arange(32, dtype=np.float32) / 32)
    max_position, rows = 8, 6
    got = position_rope_table(
        inv, max_position, rows, _t32(np.zeros(1, dtype=np.uint32))
    ).to_numpy()
    angles = np.outer(
        np.arange(max_position + rows, dtype=np.float64), inv.astype(np.float64)
    )
    want = np.concatenate([np.cos(angles), np.sin(angles)], axis=1)
    # The table builds its angles in fp32, so compare at fp32 trig accuracy.
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
