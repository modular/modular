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
"""Tests draft scores across additional shard layouts."""

from __future__ import annotations

import numpy as np
import pytest
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.speculative.draft_scores import (
    top_token_logit,
    top_token_pieces,
)
from speculative_test_helpers import (
    check_batched_combine,
    reference_scores,
    run_sharded_scores,
    shard_maxima_and_indices,
)
from speculative_test_helpers import cpu_session as cpu_session
from top_token_helpers import run_model


@pytest.mark.parametrize("batch", [2, 6])
@pytest.mark.parametrize("widths", [[17], [5, 12], [3, 4, 5, 5]])
def test_sharded_combine_matches_the_joined_score(
    session: InferenceSession, widths: list[int], batch: int
) -> None:
    logits = (
        np.random.default_rng(len(widths))
        .normal(size=(batch, sum(widths)))
        .astype(np.float32)
    )
    scores = run_sharded_scores(session, logits, widths)
    np.testing.assert_allclose(
        scores, reference_scores(logits), rtol=1e-5, atol=1e-5
    )


def test_sharded_combine_keeps_hyper_peaked_resolution(
    session: InferenceSession,
) -> None:
    logits = np.full((2, 7), -60.0, dtype=np.float32)
    logits[:, 0] = 0.0
    logits[:, -1] = [-17.0, -25.0]
    scores = run_sharded_scores(session, logits, [2, 3, 2])
    np.testing.assert_allclose(scores, [17.0, 25.0], atol=1e-3)


def _score_model(session: InferenceSession, shape: tuple[int, ...]) -> Model:
    input_type = TensorType(DType.float32, shape, DeviceRef.CPU())
    with Graph("top_token_logit", input_types=[input_type]) as graph:
        graph.output(top_token_logit(graph.inputs[0].tensor))
    return session.load(graph)


@pytest.mark.parametrize("seed, shape", [(0, (3, 5, 17)), (1, (17,))])
def test_score_matches_float64(
    session: InferenceSession, seed: int, shape: tuple[int, ...]
) -> None:
    logits = np.random.default_rng(seed).normal(size=shape).astype(np.float32)
    scores = run_model(_score_model(session, shape), logits)
    assert scores.shape == shape[:-1]
    np.testing.assert_allclose(
        scores, reference_scores(logits), rtol=1e-5, atol=1e-5
    )


def test_peaked_and_flat_positions(session: InferenceSession) -> None:
    peaked = np.full((1, 2, 8), -15.0, dtype=np.float32)
    peaked[:, :, 0] = 0.0
    flat = np.zeros_like(peaked)
    model = _score_model(session, peaked.shape)
    assert (run_model(model, peaked)[0] > 10.0).all()
    assert np.allclose(run_model(model, flat)[0], np.log(1.0 / 7.0), atol=1e-4)


@pytest.mark.parametrize("seed, tied", [(7, (2, 5)), (11, (1, 8))])
def test_sharded_combine_keeps_tied_maxima(
    session: InferenceSession, seed: int, tied: tuple[int, int]
) -> None:
    logits = np.random.default_rng(seed).normal(size=(4, 12)).astype(np.float32)
    logits[:, tied] = 5.0
    scores = run_sharded_scores(session, logits, [6, 6])
    np.testing.assert_allclose(
        scores, reference_scores(logits), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("gap", [17.0, 25.0])
def test_sharded_combine_keeps_a_distant_runner_up(
    session: InferenceSession, gap: float
) -> None:
    rng = np.random.default_rng(int(gap))
    logits = (rng.normal(size=(4, 16)) * 0.01 - 60.0).astype(np.float32)
    logits[:, 0] = 0.0
    logits[:, 9] = -gap
    scores = run_sharded_scores(session, logits, [4, 5, 7])
    np.testing.assert_allclose(scores, [gap] * 4, atol=1e-3)


def test_tied_pieces_match_per_position_scores(
    session: InferenceSession,
) -> None:
    logits = (
        np.random.default_rng(20261118)
        .normal(size=(3, 2, 17))
        .astype(np.float32)
    )
    logits[:, :, [4, 9]] = 6.0
    start = 5
    input_type = TensorType(DType.float32, logits.shape, DeviceRef.CPU())
    with Graph("top_token_pieces", input_types=[input_type]) as graph:
        graph.output(top_token_pieces(graph.inputs[0].tensor, start))
    pieces = run_model(session.load(graph), logits)
    assert pieces.shape == (3, 2, 3)
    score_model = _score_model(session, (3, 17))
    masses = np.stack(
        [
            np.exp(
                -run_model(score_model, np.ascontiguousarray(row)).astype(
                    np.float64
                )
            )
            for row in logits.transpose(1, 0, 2)
        ],
        axis=1,
    )
    np.testing.assert_allclose(
        pieces[..., 0], logits.max(axis=-1), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_array_equal(
        pieces[..., 1], logits.argmax(axis=-1) + start
    )
    np.testing.assert_allclose(pieces[..., 2], masses, rtol=1e-5, atol=1e-5)


def test_batched_combine_matches_individual_positions(
    session: InferenceSession,
) -> None:
    rng = np.random.default_rng(20261118)
    positions, batch = 3, 5
    widths = [3, 4, 5, 5]
    logits = rng.normal(size=(positions, batch, sum(widths))).astype(np.float32)
    score_models = {
        width: _score_model(session, (batch, width))
        for width in dict.fromkeys(widths)
    }
    pieces = np.empty((len(widths), batch, positions, 3), dtype=np.float32)
    boundaries = np.cumsum(widths)[:-1]
    for position, rows in enumerate(logits):
        shards = np.split(rows, boundaries, axis=-1)
        maxima, indices = shard_maxima_and_indices(shards)
        pieces[:, :, position, 0] = maxima
        pieces[:, :, position, 1] = indices
        for rank, shard in enumerate(shards):
            scores = run_model(score_models[shard.shape[-1]], shard)
            pieces[rank, :, position, 2] = np.exp(-scores.astype(np.float64))
    top_indices = np.ascontiguousarray(logits.argmax(axis=-1).T, dtype=np.int32)
    top_max = np.ascontiguousarray(logits.max(axis=-1).T)
    check_batched_combine(session, pieces, top_indices, top_max)
