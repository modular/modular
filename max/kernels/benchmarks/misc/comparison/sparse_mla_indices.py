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
"""Top-k key positions for sparse attention benchmarks.

A sparse attention kernel reads the keys a lightning indexer selected. The
positions here stand in for that selection so every engine under test attends
the same keys: causal, distinct, in score order rather than position order,
and padded with -1 when a query has fewer candidates than `top_k`.
"""

from __future__ import annotations

import numpy as np


def topk_positions(
    batch: int, q_len: int, cache_len: int, top_k: int, *, seed: int = 0
) -> np.ndarray:
    """Returns logical key positions of shape `[batch * q_len, top_k]`, int32.

    Query `j` of a request sits at position `cache_len + j` and may attend
    positions `[0, cache_len + j]`.
    """
    rng = np.random.default_rng(seed)
    out = np.full((batch * q_len, top_k), -1, dtype=np.int32)
    for b in range(batch):
        for j in range(q_len):
            row = b * q_len + j
            candidates = cache_len + j + 1
            if candidates <= top_k:
                out[row, :candidates] = rng.permutation(candidates)
            else:
                out[row] = rng.choice(candidates, size=top_k, replace=False)
    return out


def valid_counts(positions: np.ndarray) -> np.ndarray:
    """Returns the number of non-pad positions per query row, int32."""
    return (positions >= 0).sum(axis=1).astype(np.int32)


def copies_to_exceed(
    positions: np.ndarray,
    q_len: int,
    row_bytes: int,
    target_bytes: int,
    *,
    min_copies: int = 4,
    max_copies: int = 32,
) -> int:
    """Returns how many cache copies one graph needs to read `target_bytes`.

    A call reads each distinct (request, position) row once. With enough
    copies that the rows of all of them exceed the L2 cache, no call finds
    the previous call's keys in L2, as in a model whose layers each read
    their own cache.
    """
    request = np.arange(positions.shape[0]) // q_len
    valid = positions >= 0
    distinct = np.unique(
        request[:, None].repeat(positions.shape[1], 1)[valid] * (1 << 32)
        + positions[valid]
    ).size
    needed = -(-target_bytes // max(distinct * row_bytes, 1))
    return int(min(max(needed, min_copies), max_copies))


def physical_rows(
    positions: np.ndarray, page_table: np.ndarray, q_len: int, page_size: int
) -> np.ndarray:
    """Maps logical positions to rows of a paged cache flattened by page.

    `page_table[b, i]` is the physical page holding request `b`'s logical page
    `i`. Pads stay -1.
    """
    request = np.arange(positions.shape[0]) // q_len
    page, offset = np.divmod(np.maximum(positions, 0), page_size)
    rows = page_table[request[:, None], page] * page_size + offset
    return np.where(positions >= 0, rows, -1).astype(np.int32)
