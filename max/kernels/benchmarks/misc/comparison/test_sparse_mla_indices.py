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

import numpy as np
import pytest
from sparse_mla_indices import (
    copies_to_exceed,
    physical_rows,
    topk_positions,
    valid_counts,
)


@pytest.mark.parametrize("cache_len", [0, 5, 13, 40])
def test_topk_positions_are_causal_distinct_and_padded(cache_len: int) -> None:
    batch, q_len, top_k = 3, 4, 16
    positions = topk_positions(batch, q_len, cache_len, top_k, seed=1)

    assert positions.shape == (batch * q_len, top_k)
    assert positions.dtype == np.int32
    for row in range(batch * q_len):
        visible = cache_len + row % q_len + 1
        n = min(visible, top_k)
        valid = positions[row, :n]
        assert len(set(valid.tolist())) == n
        assert ((valid >= 0) & (valid < visible)).all()
        assert (positions[row, n:] == -1).all()
    assert valid_counts(positions).tolist() == [
        min(cache_len + j + 1, top_k)
        for _ in range(batch)
        for j in range(q_len)
    ]


def test_topk_positions_are_seeded() -> None:
    a = topk_positions(2, 3, 100, 8, seed=5)
    assert (a == topk_positions(2, 3, 100, 8, seed=5)).all()
    assert (a != topk_positions(2, 3, 100, 8, seed=6)).any()


def test_copies_to_exceed_counts_rows_per_request() -> None:
    # Both requests read positions 0..3: 8 distinct (request, position) rows,
    # and a request's repeated position counts once.
    positions = np.array([[0, 1, 2, 3], [3, 2, 1, 0], [0, 1, 2, 3]])
    rows = 8 * 10

    assert copies_to_exceed(positions, 2, 10, 5 * rows, min_copies=1) == 5
    assert copies_to_exceed(positions, 2, 10, 5 * rows + 1, min_copies=1) == 6
    assert copies_to_exceed(positions, 2, 10, 1, min_copies=3) == 3
    assert copies_to_exceed(positions, 2, 10, 10**9, max_copies=7) == 7


def test_copies_to_exceed_ignores_pads() -> None:
    positions = np.array([[0, -1, -1], [1, 0, -1]])
    assert copies_to_exceed(positions, 2, 1, 4, min_copies=1) == 2


def test_physical_rows_follow_the_page_table() -> None:
    page_size = 4
    page_table = np.array([[2, 0], [1, 3]])
    positions = np.array([[0, 5, -1], [3, 4, 7]])

    rows = physical_rows(positions, page_table, 1, page_size)

    assert rows.dtype == np.int32
    assert rows.tolist() == [[8, 1, -1], [7, 12, 15]]
