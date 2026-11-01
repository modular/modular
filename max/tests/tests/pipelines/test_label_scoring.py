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

"""Tests for the on-device label-gather used by ``/v1/decisions``."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.pipelines.sampling import LabelScorer

_VOCAB = 24
_PADDED_VOCAB = 32


def _log_softmax(logits: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


@pytest.fixture(scope="module")
def scorer() -> LabelScorer:
    return LabelScorer(
        InferenceSession(devices=[CPU()]), CPU(), DType.float32, _VOCAB
    )


def _logits(seed: int, rows: int) -> npt.NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(rows, _PADDED_VOCAB)).astype(np.float32) * 3
    # A padded lm_head column must never take probability mass.
    logits[:, _VOCAB:] = 50.0
    return logits


def test_scores_are_full_vocabulary_log_probabilities(
    scorer: LabelScorer,
) -> None:
    logits = _logits(0, rows=3)
    labels = [[1, 5, 7], None, [0, _VOCAB - 1]]

    scores = scorer.score(Buffer.from_numpy(logits), labels)

    expected = _log_softmax(logits[:, :_VOCAB].astype(np.float64))
    assert scores[1] is None
    assert scores[0] == pytest.approx(expected[0, [1, 5, 7]].tolist(), abs=1e-5)
    assert scores[2] == pytest.approx(
        expected[2, [0, _VOCAB - 1]].tolist(), abs=1e-5
    )


def test_label_probabilities_stay_below_one_and_sum_over_the_vocabulary(
    scorer: LabelScorer,
) -> None:
    logits = _logits(1, rows=1)
    every_token = list(range(_VOCAB))

    (scores,) = scorer.score(Buffer.from_numpy(logits), [every_token])

    assert scores is not None
    assert float(np.exp(scores).sum()) == pytest.approx(1.0, abs=1e-5)
    (partial,) = scorer.score(Buffer.from_numpy(logits), [[2, 3]])
    assert partial is not None
    assert 0.0 < float(np.exp(partial).sum()) < 1.0


def test_a_batch_with_no_scoring_rows_does_no_work(scorer: LabelScorer) -> None:
    logits = _logits(2, rows=2)
    assert scorer.score(Buffer.from_numpy(logits), [None, []]) == [None, None]


def test_label_ids_outside_the_vocabulary_are_rejected(
    scorer: LabelScorer,
) -> None:
    logits = _logits(3, rows=1)
    with pytest.raises(ValueError, match="out of vocabulary range"):
        scorer.score(Buffer.from_numpy(logits), [[_VOCAB]])
