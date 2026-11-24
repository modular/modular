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

"""Tests for which steps of a request the batch label scorer scores."""

from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import create_autospec

import numpy as np
import pytest
from max.driver import Buffer
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.lib import ModelOutputs
from max.pipelines.lib.pipeline_variants._label_scoring import (
    BatchLabelScorer,
)
from max.pipelines.modeling.types import RequestID
from max.pipelines.sampling import LabelScorer


def _context(*, label_token_ids: list[int] | None) -> TextContext:
    context = TextContext(
        request_id=RequestID(),
        max_length=32,
        tokens=TokenBuffer(np.ones(8, dtype=np.int64)),
    )
    context.label_token_ids = label_token_ids
    return context


def _batch_scorer() -> BatchLabelScorer:
    batch_scorer = object.__new__(BatchLabelScorer)

    def score(
        logits: Buffer, label_token_ids: Sequence[Sequence[int] | None]
    ) -> list[list[float] | None]:
        return [[-1.0] * len(ids) if ids else None for ids in label_token_ids]

    scorer = create_autospec(LabelScorer, instance=True)
    scorer.score.side_effect = score
    batch_scorer._scorer = scorer
    return batch_scorer


def _outputs(rows: int) -> ModelOutputs:
    return ModelOutputs(
        logits=Buffer.from_numpy(np.zeros((rows, 4), dtype=np.float32))
    )


def test_a_step_that_completes_the_prompt_is_scored() -> None:
    context = _context(label_token_ids=[1, 2])
    scores = _batch_scorer().score(_outputs(1), [context])
    assert scores == {context.request_id: [-1.0, -1.0]}


def test_a_chunked_prefill_step_is_not_scored() -> None:
    context = _context(label_token_ids=[1, 2])
    context.tokens.chunk(4)
    assert _batch_scorer().score(_outputs(1), [context]) == {}


def test_only_the_scoring_rows_of_a_mixed_batch_are_scored() -> None:
    scoring = _context(label_token_ids=[3])
    plain = _context(label_token_ids=None)
    scores = _batch_scorer().score(_outputs(2), [plain, scoring])
    assert scores == {scoring.request_id: [-1.0]}


def test_variable_logits_are_refused() -> None:
    outputs = _outputs(1)
    outputs.logit_offsets = Buffer.from_numpy(np.array([0, 1], dtype=np.uint32))
    with pytest.raises(ValueError, match="variable logits"):
        _batch_scorer().score(outputs, [_context(label_token_ids=[1])])


def test_a_batch_without_scoring_requests_does_not_compile_the_scorer() -> None:
    batch_scorer = object.__new__(BatchLabelScorer)
    batch_scorer._scorer = None
    plain = _context(label_token_ids=None)
    assert batch_scorer.score(_outputs(1), [plain]) == {}
    assert batch_scorer._scorer is None


def test_scoring_without_a_vocabulary_size_is_refused() -> None:
    batch_scorer = object.__new__(BatchLabelScorer)
    batch_scorer._scorer = None
    batch_scorer._vocab_size = None
    with pytest.raises(ValueError, match="vocabulary size"):
        batch_scorer.score(_outputs(1), [_context(label_token_ids=[1])])
