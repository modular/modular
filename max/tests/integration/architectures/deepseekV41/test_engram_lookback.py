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
"""Tests the Engram lookback window built from request contexts.

Each step's window must hold the tokens just before the step's first active
token, through chunked prefill, decode, and requests batched at different
offsets.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from max.pipelines.architectures.deepseekV41.model import (
    engram_lookback_ids_for_contexts,
)
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.modeling.types import RequestID

RAW_VOCAB = 300
DEPTH = 2


def _context(prompt: np.ndarray) -> TextContext:
    return TextContext(
        request_id=RequestID(),
        max_length=1000,
        tokens=TokenBuffer(prompt.astype(np.int64)),
    )


def _sequence(seed: int, length: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, RAW_VOCAB, size=length)


def _expected_window(sequence: np.ndarray, start: int) -> list[int]:
    """Column ``j`` is the token at ``start - 1 - j``, or -1 before 0."""
    return [
        int(sequence[start - 1 - j]) if start - 1 - j >= 0 else -1
        for j in range(DEPTH)
    ]


def _run_schedule(
    sequences: Sequence[np.ndarray],
    prompt_lens: Sequence[int],
    chunk_schedules: Sequence[Sequence[int]],
) -> int:
    """Drives each request through its prompt chunks, then decode steps.

    ``chunk_schedules[i]`` is request ``i``'s prompt chunk sizes (summing to
    ``prompt_lens[i]``); after its prompt, each request decodes the rest of
    ``sequences[i]`` one token per step. Every step batches every request
    that still has work and checks each request's lookback window.

    Returns:
        The number of steps run.
    """
    contexts = [
        _context(seq[:plen])
        for seq, plen in zip(sequences, prompt_lens, strict=True)
    ]
    steps = [
        [*sched, *([1] * (seq.shape[0] - plen))]
        for seq, plen, sched in zip(
            sequences, prompt_lens, chunk_schedules, strict=True
        )
    ]
    for sched, plen in zip(chunk_schedules, prompt_lens, strict=True):
        assert sum(sched) == plen
    num_steps = 0
    while batch := [i for i, todo in enumerate(steps) if todo]:
        for i in batch:
            size = steps[i].pop(0)
            tokens = contexts[i].tokens
            if size < tokens.active_length:
                tokens.chunk(size)
            assert tokens.active_length == size
        window = engram_lookback_ids_for_contexts(
            [contexts[i] for i in batch], DEPTH
        )
        assert window.shape == (len(batch), DEPTH)
        assert window.dtype == np.int64
        for row, i in enumerate(batch):
            tokens = contexts[i].tokens
            start = tokens.processed_length
            # The window is keyed to the first token this step processes.
            np.testing.assert_array_equal(
                np.asarray(tokens.active),
                sequences[i][start : start + tokens.active_length],
            )
            assert window[row].tolist() == _expected_window(
                sequences[i], start
            ), f"request {i} at position {start}"
        for i in batch:
            tokens = contexts[i].tokens
            if tokens.actively_chunked:
                tokens.advance_chunk()
            elif tokens.current_position < sequences[i].shape[0]:
                tokens.advance_with_token(
                    int(sequences[i][tokens.current_position])
                )
        num_steps += 1
    return num_steps


def test_full_prefill() -> None:
    seq = _sequence(0, 17)
    assert _run_schedule([seq], [17], [[17]]) == 1


@pytest.mark.parametrize("chunks", [[9, 8], [5, 1, 11], [1, 1, 15]])
def test_chunked_prefill(chunks: list[int]) -> None:
    seq = _sequence(1, 17)
    assert _run_schedule([seq], [17], [chunks]) == len(chunks)


def test_prefill_then_decode() -> None:
    seq = _sequence(2, 14)
    assert _run_schedule([seq], [6], [[6]]) == 1 + 14 - 6


def test_chunked_prefill_then_decode() -> None:
    seq = _sequence(3, 15)
    assert _run_schedule([seq], [8], [[3, 1, 4]]) == 3 + 15 - 8


def test_batch_of_two_at_different_offsets() -> None:
    """Checks one request decoding while another is mid chunked prefill."""
    a = _sequence(4, 12)
    b = _sequence(5, 16)
    _run_schedule([a, b], [4, 13], [[4], [2, 1, 5, 5]])
