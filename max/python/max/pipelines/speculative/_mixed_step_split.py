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
"""Host split record of a mixed speculative step.

Some attention ops pick one route for the whole batch from its longest row,
so in a step that batches decode rows with prompt rows every decode row takes
the prefill route. The split record cuts each replica batch into a head and a
tail view, so the decode rows take the decode routes and the prompt rows the
prefill route, in one op call. A spec graph built with
:attr:`~max.pipelines.speculative.config.SpeculativeConfig.mixed_step_split_inputs`
declares one record per device after every other input; the ops validate it
and keep the unsplit route for any record they cannot use.

The record is ``uint32[7]``; the head view is rows ``[0, r)``, the tail view
the rest:

====  ==============================================
Slot  Value
====  ==============================================
0     Head rows ``r`` (0 means no split)
1     Head query tokens
2     Head max query length
3     Head max cache length
4     Tail max query length
5     Tail max cache length
6     Total context length of the prefill view
====  ==============================================
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.nn.legacy.kernels import _MSA_SPLIT_ARGS_SIZE
from max.pipelines.context import TextContext
from max.pipelines.kv_cache.paged_kv_cache._common import (
    cache_valid_length_for_context,
    prompt_tokens_for_context,
)

__all__ = [
    "MIXED_STEP_SPLIT",
    "MIXED_STEP_SPLIT_DECODE_CAP",
    "fill_mixed_step_split",
    "mixed_step_decode_width",
    "mixed_step_split_buffers",
    "mixed_step_split_record",
    "mixed_step_split_record_for_contexts",
]

logger = logging.getLogger("max.pipelines")

MIXED_STEP_SPLIT = "mixed_step_split"
"""The records' name: the graph input group, the sidecar key, and the key the
records ride under in a verify batch's ``extra``."""

MIXED_STEP_SPLIT_DECODE_CAP = 8
"""Widest query the split-route ops run on their decode routes
(``MAX_SPEC_DRAFT`` in the MSA ops). A mismatch only turns the split off: the
ops re-check it."""


def mixed_step_decode_width(num_draft_tokens: int) -> int:
    """Returns the widest row the split puts in the decode view.

    A verify row is ``1 + K`` tokens wide. A wider bound would also take
    short prompt rows, which the persistent decode scratch, sized for
    ``max_batch * (1 + K)``, does not cover.
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
    first prompt row stays in the prefill view: correct, only slower.

    Args:
        widths: Each row's query tokens this step, drafts included (the
            widths behind ``input_row_offsets``).
        valid_lengths: Each row's cache bound for this step, the values the
            caches' ``max_cache_length`` and ``total_context_length`` are
            built from.
        decode_width: The widest decode row, from
            :func:`mixed_step_decode_width`.

    Returns:
        The record; zeros when the batch has no prompt row or no
        decode row at either end.
    """
    if len(widths) != len(valid_lengths):
        raise ValueError(
            f"{len(widths)} widths but {len(valid_lengths)} valid lengths"
        )
    record = np.zeros(_MSA_SPLIT_ARGS_SIZE, dtype=np.uint32)
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


def mixed_step_split_record_for_contexts(
    batch: Sequence[TextContext], num_draft_tokens: int
) -> np.ndarray:
    """Builds one replica batch's record.

    Builds it from the per-row values the KV manager builds the row offsets
    and ``max_cache_length`` from.
    """
    return mixed_step_split_record(
        [prompt_tokens_for_context(ctx) for ctx in batch],
        [
            cache_valid_length_for_context(ctx, num_draft_tokens)
            for ctx in batch
        ],
        mixed_step_decode_width(num_draft_tokens),
    )


def mixed_step_split_buffers(num_devices: int) -> list[Buffer]:
    """Allocates one CPU record buffer per device."""
    return [
        Buffer.zeros(shape=[_MSA_SPLIT_ARGS_SIZE], dtype=DType.uint32)
        for _ in range(num_devices)
    ]


def fill_mixed_step_split(
    records: Sequence[Buffer],
    replica_batches: Sequence[Sequence[TextContext]],
    num_draft_tokens: int,
) -> None:
    """Writes each replica's record into its devices' CPU buffers.

    A replica whose step splits logs its record at debug level, the
    smoke-visible sign a split step ran.

    Args:
        records: One CPU record buffer per device, replica-major.
        replica_batches: Each replica's rows, in the order the KV manager
            builds the row offsets from.
        num_draft_tokens: The drafts each row verifies this step.
    """
    devices_per_replica = len(records) // len(replica_batches)
    for replica_idx, batch in enumerate(replica_batches):
        record = mixed_step_split_record_for_contexts(batch, num_draft_tokens)
        if record.any():
            logger.debug(
                "mixed-step split record of replica %d: %s",
                replica_idx,
                record.tolist(),
            )
        start = replica_idx * devices_per_replica
        for buffer in records[start : start + devices_per_replica]:
            buffer.to_numpy()[:] = record


def _run_length(widths: Sequence[int], decode_width: int) -> int:
    count = 0
    for width in widths:
        if width > decode_width:
            break
        count += 1
    return count
