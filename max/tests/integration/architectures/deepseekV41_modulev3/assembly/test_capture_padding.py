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
"""Capture-padding regressions for the attention/compressor store.

A captured decode pads its row axis past ``row_offsets[-1]``; the padded
rows synthesize positions past the last request's start. Three
regressions pin that the padded rows can neither write nor gather:

1. The ratio-2 ring save keeps a padded row out of the state ring: with
   one live row and eight padded ones on a two-row ring, only the live
   row's slot saves (the wrap bound alone would let the padded rows
   wrap the ring and overwrite the valid carry of an odd-start pool).
2. The window store's slots mask padded rows to ``-1``: the store op
   launches over the physical row count and skips only negative slots,
   so an unmasked padded row could write through its synthesized
   position onto a page its request does not own.
3. The owner index-key store's lookup-table gather stays in bounds.
   (The attention index list's own padded-row masking runs main's kernel
   and is pinned in ``test_index_list_gpu.py``.)
"""

from __future__ import annotations

import numpy as np
from max import functional as F
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.pipelines.architectures.deepseekV41_modulev3.layers.attention import (
    RaggedRows,
    live_rows,
    window_store_slots,
)
from max.pipelines.architectures.deepseekV41_modulev3.layers.compressor import (
    _ring_save_slots,
)
from max.pipelines.architectures.deepseekV41_modulev3.layers.indexer import (
    index_key_store_slots,
)
from max.tensor import Tensor

_SLOTS = 128  # the interface's window-leaf page size


def _t32(x: np.ndarray) -> Tensor:
    return Tensor.from_dlpack(np.ascontiguousarray(x, dtype=np.int32))


def _leaf(
    blocks: int,
    layers: int,
    slots: int,
    pages: int,
    requests: int,
    width: int = 528,
) -> PagedCacheValues:
    """A window-shaped leaf with an identity lookup table.

    Request ``r``'s page ``p`` is physical block ``r * pages + p``.
    """
    lut = np.arange(requests * pages, dtype=np.int64).reshape(requests, pages)
    return PagedCacheValues(
        kv_blocks=Tensor.from_dlpack(
            np.zeros((blocks, 1, layers, slots, 1, width), dtype=np.uint8)
        ),
        cache_lengths=Tensor.from_dlpack(np.zeros(requests, dtype=np.uint32)),
        lookup_table=Tensor.from_dlpack(lut),
        max_prompt_length=Tensor.from_dlpack(
            np.array([slots], dtype=np.uint32)
        ),
        max_cache_length=Tensor.from_dlpack(np.array([slots], dtype=np.uint32)),
        page_stride=Tensor.from_dlpack(np.array([-1], dtype=np.int64)),
    )


def _rows(
    offsets: np.ndarray, positions: np.ndarray, lengths: np.ndarray
) -> RaggedRows:
    """RaggedRows with bid/local as the trunk derives them.

    The padded tail pins to the last request, exactly as
    :func:`~...layers.attention.row_ids` does, and its ``local`` indices
    run past that request's length.
    """
    total = int(positions.shape[0])
    b = int(offsets.shape[0]) - 1
    bid = np.zeros(total, dtype=np.int32)
    pos = 0
    for r in range(b):
        bid[pos : pos + int(lengths[r])] = r
        pos += int(lengths[r])
        bid[pos:] = r  # padded tail pins to the last request
    local = np.arange(total, dtype=np.int32) - offsets[bid].astype(np.int32)
    return RaggedRows(
        offsets=_t32(offsets.astype(np.int32)),
        bid=_t32(bid),
        local=_t32(local),
        positions=_t32(positions),
        lengths=_t32(lengths),
    )


def test_live_rows_flags_the_padded_tail() -> None:
    """``live_rows`` is True below ``row_offsets[-1]`` only."""
    offsets = np.array([0, 1], dtype=np.uint32)
    got = live_rows(_t32(offsets), 9).to_numpy()
    np.testing.assert_array_equal(got, np.array([True] + [False] * 8))


def test_ring_save_masks_padded_rows() -> None:
    """Live 1, padded 8, ring of 2: only the live row saves.

    An odd-start request's carry row must survive the padded tail: the
    wrap bound (``local >= length - ring_rows``) alone keeps every
    padded row too, and each would wrap the two-row ring and overwrite
    the carry with padding garbage.
    """
    total = 9
    lengths = np.array([1], dtype=np.int32)
    bid = np.zeros(total, dtype=np.int32)
    local = np.arange(total, dtype=np.int32)
    slots = np.arange(total, dtype=np.int32) + 100
    # The slot mask is a raw graph function; the eager test drives it
    # through the same ``F.functional`` bridge the compressor uses.
    got = F.functional(_ring_save_slots)(
        2,
        _t32(bid),
        _t32(local),
        _t32(lengths),
        _t32(slots),
    ).to_numpy()
    expected = np.array([100] + [-1] * 8, dtype=np.int32)
    np.testing.assert_array_equal(got, expected)


def test_window_store_slots_mask_padded_rows() -> None:
    """A padded row's store slot is ``-1``, never a foreign page.

    The single live row sits at position 127 (the last slot of its one
    allocated page); the eight padded rows synthesize positions 128
    onward, whose entries map to page 1 -- past the one page the
    request owns. Masked, every padded row takes ``-1`` and the store
    op skips it; the live row's slot is its own page's last record.
    """
    total = 9
    offsets = np.array([0, 1], dtype=np.uint32)
    positions = 127 + np.arange(total, dtype=np.int32)
    lengths = np.array([1], dtype=np.int32)
    rows = _rows(offsets, positions, lengths)
    leaf = _leaf(blocks=1, layers=4, slots=_SLOTS, pages=1, requests=1)
    layer = 2
    got = window_store_slots(leaf, rows, _t32(np.array([layer]))).to_numpy()
    live_slot = (0 * 4 + layer) * _SLOTS + 127 % _SLOTS
    expected = np.array([live_slot] + [-1] * 8, dtype=np.int32)
    np.testing.assert_array_equal(got, expected)


def test_index_key_store_slots_mask_padded_rows() -> None:
    """The owner index-key store's lookup-table gather stays in bounds.

    One live row at position 127 on a one-page leaf; the eight padded
    rows synthesize positions 128 onward, whose keys cross onto page 1
    -- past the request's single allocated page. Without clamping the
    dropped rows' keys to a safe page, the lookup-table gather runs
    out of bounds before any ``-1`` skip path.
    """
    total = 9
    offsets = np.array([0, 1], dtype=np.uint32)
    lengths = np.array([1], dtype=np.int32)
    positions = 127 + np.arange(total, dtype=np.int32)
    rows = _rows(offsets, positions, lengths)
    for ratio, keys_per_page, live_key in ((1, 128, 127), (2, 64, 63)):
        # At ratio 2 the live row at odd position 127 closes its group
        # (key 63); the padded rows 128..135 open keys 64..67, which
        # cross onto the ratio-2 leaf's second page.
        leaf = _leaf(
            blocks=1,
            layers=2,
            slots=keys_per_page,
            pages=1,
            requests=1,
            width=68,
        )
        got = index_key_store_slots(
            leaf,
            rows,
            ratio=ratio,
            cap=1 << 14,
            floor=None,
        ).to_numpy()
        expected = np.array([live_key] + [-1] * 8, dtype=np.int32)
        np.testing.assert_array_equal(got, expected)
