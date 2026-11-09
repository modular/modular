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
"""The host max lengths of a paged KV cache leaf, with its split record.

Some attention ops pick one route for the whole batch from its longest row,
so in a step that batches decode rows with prefill rows every decode row takes
the prefill route. The split record cuts each replica batch into a head and a
tail view, so the decode rows take the decode routes and the prefill rows the
prefill route, in one op call. The KV cache managers fill it for every batch;
the ops validate it and keep the unsplit route for any record they cannot use.

Each replica's max lengths are one ``uint32[9]`` host array. Slots 0 and 1 are
the leaf's ``max_prompt_length`` and ``max_cache_length``; slots 2-8 are its
``mixed_step_split`` record. The head view is rows ``[0, r)``, the tail view
the rest:

====  ==============================================
Slot  Value
====  ==============================================
0     Batch max query length
1     Batch max cache length
2     Head rows ``r`` (0 means no split)
3     Head query tokens
4     Head max query length
5     Head max cache length
6     Tail max query length
7     Tail max cache length
8     Total context length of the prefill view
====  ==============================================
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
from max.driver import Buffer
from max.nn.kv_cache.input_types import MIXED_STEP_SPLIT_SIZE

MIXED_STEP_SPLIT_DECODE_CAP = 8
"""Widest query the split-route ops run on their decode routes
(``MIXED_STEP_DECODE_WIDTH`` in ``kv_cache.mixed_step_split``). A mismatch only
turns the split off: the ops re-check it."""

MIXED_STEP_SPLIT_SWITCH = "MODULAR_MIXED_STEP_SPLIT"
"""Set to ``0`` to fill zeros on the same graph, for A/B runs. Mach reads the
same switch."""


def mixed_step_decode_width(num_draft_tokens: int) -> int:
    """Returns the widest row the split puts in the decode view.

    A decode row is ``1 + K`` tokens wide (``K = 0`` without spec). A wider
    bound would also take short prefill rows, which the persistent decode
    scratch, sized for ``max_batch * (1 + K)``, does not cover.
    """
    return min(1 + num_draft_tokens, MIXED_STEP_SPLIT_DECODE_CAP)


def mixed_step_split_record(
    widths: Sequence[int],
    valid_lengths: Sequence[int],
    decode_width: int,
) -> np.ndarray:
    """Builds one replica batch's split record, in the batch's row order.

    The decode view is the longest run of rows at most ``decode_width`` wide
    at either end of the batch (the longer of the two). A decode row past the
    first prefill row stays in the prefill view: correct, only slower.

    Args:
        widths: Each row's query tokens this step, drafts included (the
            widths behind ``input_row_offsets``).
        valid_lengths: Each row's cache bound for this step, the values the
            caches' ``max_cache_length`` and ``total_context_length`` are
            built from.
        decode_width: The widest decode row, from
            :func:`mixed_step_decode_width`.

    Returns:
        The record; zeros when the batch has no prefill row or no decode row
        at either end, or when :data:`MIXED_STEP_SPLIT_SWITCH` is ``0``.
    """
    if len(widths) != len(valid_lengths):
        raise ValueError(
            f"{len(widths)} widths but {len(valid_lengths)} valid lengths"
        )
    record = np.zeros(MIXED_STEP_SPLIT_SIZE, dtype=np.uint32)
    if os.environ.get(MIXED_STEP_SPLIT_SWITCH, "").strip() == "0":
        return record
    if not widths or max(widths) <= MIXED_STEP_SPLIT_DECODE_CAP:
        return record
    leading = _run_length(widths, decode_width)
    trailing = _run_length(widths[::-1], decode_width)
    if leading == 0 and trailing == 0:
        return record
    decode_leads = leading >= trailing
    head_rows = leading if decode_leads else len(widths) - trailing
    head = slice(0, head_rows)
    tail = slice(head_rows, len(widths))
    prefill = tail if decode_leads else head
    record[:] = [
        head_rows,
        sum(widths[head]),
        max(widths[head]),
        max(valid_lengths[head]),
        max(widths[tail]),
        max(valid_lengths[tail]),
        sum(valid_lengths[prefill]),
    ]
    return record


def max_lengths_buffers(
    max_prompt_length: int, max_cache_length: int, record: np.ndarray
) -> tuple[Buffer, Buffer, Buffer]:
    """Builds one replica's ``uint32[9]`` max lengths and returns its views.

    Returns:
        The leaf's ``max_prompt_length`` and ``max_cache_length`` (``[1]``
        each) and ``mixed_step_split`` (``[7]``), all views of one host
        buffer.
    """
    lengths = Buffer.from_numpy(
        np.array(
            [max_prompt_length, max_cache_length, *record], dtype=np.uint32
        )
    )
    return lengths[0:1], lengths[1:2], lengths[2:]


def lengths_and_record(
    widths: Sequence[int],
    valid_lengths: Sequence[int],
    num_draft_tokens: int,
    max_prompt_length: int,
    max_cache_length: int,
) -> tuple[Buffer, Buffer, Buffer]:
    """Builds one replica's max lengths from its rows and returns their views.

    The record comes from the real rows, not a replay's dispatch bounds: a
    padded batch shape is a decode-only replay, whose record is zeros.
    """
    record = mixed_step_split_record(
        widths, valid_lengths, mixed_step_decode_width(num_draft_tokens)
    )
    return max_lengths_buffers(max_prompt_length, max_cache_length, record)


def _run_length(widths: Sequence[int], decode_width: int) -> int:
    count = 0
    for width in widths:
        if width > decode_width:
            break
        count += 1
    return count
