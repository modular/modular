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
"""Tests verify budgets across additional replica layouts."""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.speculative.config import MAGIC_DRAFT_TOKEN_ID
from max.pipelines.speculative.verify_budget import (
    draft_survival,
    select_replica_verify_lengths,
    unverified_draft_placeholders,
)
from speculative_test_helpers import (
    RANKED_SURVIVAL,
    build_floored_lengths_model,
    build_lengths_model,
    build_replica_lengths_model,
    check_replica_budgets,
    reference_lengths,
    reference_survival,
    run_replica_lengths,
    sorted_survival,
)
from speculative_test_helpers import cpu_session as cpu_session


@pytest.mark.parametrize(
    "batch, positions, widths", [(2, 3, (0, 2, 3, 4)), (6, 5, (2,))]
)
def test_one_replica_matches_the_whole_batch_selection(
    session: InferenceSession,
    batch: int,
    positions: int,
    widths: tuple[int, ...],
) -> None:
    survival = sorted_survival(np.random.default_rng(4), batch, positions)
    device = DeviceRef.CPU()
    with Graph(
        "one_replica",
        input_types=[
            TensorType(DType.float32, list(survival.shape), device),
            TensorType(DType.int32, [], device),
            TensorType(DType.int64, [2], device),
        ],
    ) as graph:
        scores, step_width, splits = (x.tensor for x in graph.inputs)
        graph.output(
            select_replica_verify_lengths(scores, step_width, None),
            select_replica_verify_lengths(scores, step_width, splits),
        )
    model = session.load(graph)
    for width in widths:
        implicit, explicit = model.execute(
            Buffer.from_numpy(survival),
            Buffer.from_numpy(np.array(width, dtype=np.int32)),
            Buffer.from_numpy(np.array([0, len(survival)], dtype=np.int64)),
        )
        _, expected = reference_lengths(survival, len(survival) * width)
        for output in (implicit, explicit):
            lengths = output.to_numpy()
            np.testing.assert_array_equal(lengths, expected)
            assert lengths.sum() == min(batch * width, batch * positions)


@pytest.mark.parametrize(
    "cases",
    [
        [(RANKED_SURVIVAL, [0, 2, 2, 8], 2)],
        [
            (RANKED_SURVIVAL, [0, 2, 4, 6, 8], 1),
            (RANKED_SURVIVAL, [0, 2, 4, 6, 8], 3),
        ],
    ],
    ids=["empty-middle-replica", "four-replicas"],
)
def test_replica_budgets_match_the_sorted_reference(
    session: InferenceSession, cases: list[tuple[np.ndarray, list[int], int]]
) -> None:
    check_replica_budgets(session, cases)


@pytest.fixture(scope="module")
def survival_model(
    session: InferenceSession, survival_batch: str | int
) -> Model:
    device = DeviceRef.CPU()
    with Graph(
        "draft_survival",
        input_types=[
            TensorType(DType.float32, [survival_batch, 5], device),
            TensorType(DType.float32, [6], device),
        ],
    ) as graph:
        graph.output(
            draft_survival(graph.inputs[0].tensor, graph.inputs[1].tensor)
        )
    return session.load(graph)


@pytest.mark.parametrize("survival_batch", ["batch_size", 4], scope="module")
def test_survival_matches_a_float64_reference(survival_model: Model) -> None:
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(4, 5)).astype(np.float32)
    coeffs = rng.normal(size=(6,)).astype(np.float32)
    (out,) = survival_model.execute(
        Buffer.from_numpy(scores), Buffer.from_numpy(coeffs)
    )
    np.testing.assert_allclose(
        out.to_numpy(), reference_survival(scores, coeffs), atol=1e-6
    )


@pytest.mark.parametrize("survival_batch", ["batch_size", 2], scope="module")
def test_non_finite_scores_match_their_bounds(survival_model: Model) -> None:
    coeffs = np.array([0.5, -0.1, 0.2, -0.3, 0.1, -0.2], dtype=np.float32)
    scores = np.array(
        [
            [np.nan, 0.0, 1.0, -1.0, 0.5],
            [np.inf, -np.inf, 0.0, 2.0, -2.0],
        ],
        dtype=np.float32,
    )
    (out,) = survival_model.execute(
        Buffer.from_numpy(scores), Buffer.from_numpy(coeffs)
    )
    np.testing.assert_allclose(
        out.to_numpy(), reference_survival(scores, coeffs), atol=1e-6
    )


def _select_lengths(
    model: Model, survival: np.ndarray, total: int
) -> np.ndarray:
    (out,) = model.execute(
        Buffer.from_numpy(survival),
        Buffer.from_numpy(np.array(total, dtype=np.int32)),
    )
    return out.to_numpy()


def test_selection_executes_with_a_symbolic_batch_dim(
    session: InferenceSession,
) -> None:
    model = build_lengths_model(session, ["batch_size", 5])
    rng = np.random.default_rng(3)
    for batch in (3, 8):
        survival = sorted_survival(rng, batch, 5)
        for total in (0, 4, batch * 5 // 2, batch * 5 + 3):
            actual = _select_lengths(model, survival, total)
            _, expected = reference_lengths(survival, total)
            np.testing.assert_array_equal(actual, expected)
            assert actual.sum() == min(total, batch * 5)


def test_selection_matches_the_sorted_reference(
    session: InferenceSession,
) -> None:
    rng = np.random.default_rng(1)
    for batch in (1, 3, 8):
        positions = 5
        survival = sorted_survival(rng, batch, positions)
        model = build_lengths_model(session, survival.shape)
        for total in (
            0,
            1,
            batch,
            batch * positions // 2,
            batch * positions,
            batch * positions + 3,
        ):
            actual = _select_lengths(model, survival, total)
            _, expected = reference_lengths(survival, total)
            np.testing.assert_array_equal(actual, expected)
            assert actual.sum() == min(total, batch * positions)


def test_ties_split_by_position_then_row(session: InferenceSession) -> None:
    positions = 4
    uniform = np.full((3, positions), 0.5, dtype=np.float32)
    model = build_lengths_model(session, uniform.shape)
    for width in (1, 2):
        actual = _select_lengths(model, uniform, 3 * width)
        np.testing.assert_array_equal(actual, np.full(3, width))
    tied = np.array(
        [
            [0.9, 0.5, 0.5, 0.1],
            [0.8, 0.5, 0.5, 0.1],
        ],
        dtype=np.float32,
    )
    model = build_lengths_model(session, tied.shape)
    actual = _select_lengths(model, tied, 3)
    np.testing.assert_array_equal(actual, [2, 1])
    position_tie = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.0],
        ],
        dtype=np.float32,
    )
    model = build_lengths_model(session, position_tie.shape)
    actual = _select_lengths(model, position_tie, 3)
    np.testing.assert_array_equal(actual, [2, 1])


def test_kept_positions_are_a_prefix(session: InferenceSession) -> None:
    rng = np.random.default_rng(2)
    survival = sorted_survival(rng, 4, 5)
    kept, lengths = reference_lengths(survival, 10)
    for row in range(4):
        assert kept[row, : lengths[row]].all()
        assert not kept[row, lengths[row] :].any()
    model = build_lengths_model(session, survival.shape)
    actual = _select_lengths(model, survival, 10)
    np.testing.assert_array_equal(actual, lengths)


def test_placeholders_mask_positions_past_the_verify_length(
    session: InferenceSession,
) -> None:
    device = DeviceRef.CPU()
    draft_type = TensorType(DType.int64, [3, 4], device)
    lengths_type = TensorType(DType.int32, [3], device)
    with Graph(
        "unverified_draft_placeholders",
        input_types=[draft_type, lengths_type],
    ) as graph:
        graph.output(
            unverified_draft_placeholders(
                graph.inputs[0].tensor, graph.inputs[1].tensor
            )
        )
    model = session.load(graph)
    rng = np.random.default_rng(9)
    drafts = rng.integers(0, 1000, size=(3, 4)).astype(np.int64)
    lengths = np.array([0, 2, 4], dtype=np.int32)
    (out,) = model.execute(
        Buffer.from_numpy(drafts), Buffer.from_numpy(lengths)
    )
    masked = out.to_numpy()
    expected = np.where(np.arange(4)[None, :] >= lengths[:, None], -1, drafts)
    np.testing.assert_array_equal(masked, expected)
    assert (masked[0] == -1).all()
    assert (masked[1, 2:] == -1).all()
    np.testing.assert_array_equal(masked[2], drafts[2])


def test_two_replicas_each_spend_their_own_budget(
    session: InferenceSession,
) -> None:
    splits = np.array([0, 2, 6], dtype=np.int64)
    survival = sorted_survival(np.random.default_rng(5), 6, 5)
    model = build_replica_lengths_model(session, survival.shape, len(splits))
    lengths = run_replica_lengths(model, survival, splits, 3)
    _, first = reference_lengths(survival[:2], 6)
    _, second = reference_lengths(survival[2:], 12)
    np.testing.assert_array_equal(lengths, np.concatenate([first, second]))
    assert lengths[:2].sum() == 6
    assert lengths[2:].sum() == 12

    high = sorted_survival(np.random.default_rng(6), 2, 5) * 0.5 + 0.5
    low = sorted_survival(np.random.default_rng(7), 4, 5) * 0.5
    isolated = np.concatenate([high, low]).astype(np.float32)
    assert run_replica_lengths(model, isolated, splits, 3)[2:].sum() == 12

    full = sorted_survival(np.random.default_rng(13), 6, 5)
    full_lengths = run_replica_lengths(
        model, full, np.array([0, 3, 6], dtype=np.int64), 5
    )
    np.testing.assert_array_equal(full_lengths, np.full(6, 5, dtype=np.int32))


def test_empty_first_replica_keeps_batch_order(
    session: InferenceSession,
) -> None:
    survival = sorted_survival(np.random.default_rng(8), 4, 5)
    splits = np.array([0, 0, 4], dtype=np.int64)
    model = build_replica_lengths_model(session, survival.shape, len(splits))
    lengths = run_replica_lengths(model, survival, splits, 2)
    _, expected = reference_lengths(survival, 8)
    assert lengths.shape == (4,)
    np.testing.assert_array_equal(lengths, expected)
    assert lengths.sum() == 8


def test_non_uniform_splits_match_reference(session: InferenceSession) -> None:
    survival = sorted_survival(np.random.default_rng(10), 6, 5)
    check_replica_budgets(
        session,
        [
            (survival, [0, boundary, 6], width)
            for boundary in (1, 4, 0)
            for width in (1, 2, 5)
        ],
    )


def test_empty_middle_replica_keeps_other_budgets(
    session: InferenceSession,
) -> None:
    survival = sorted_survival(np.random.default_rng(11), 6, 5)
    splits = np.array([0, 2, 2, 6], dtype=np.int64)
    model = build_replica_lengths_model(session, survival.shape, len(splits))
    lengths = run_replica_lengths(model, survival, splits, 2)
    _, first = reference_lengths(survival[:2], 4)
    _, last = reference_lengths(survival[2:], 8)
    np.testing.assert_array_equal(lengths, np.concatenate([first, last]))
    assert lengths[:2].sum() == 4
    assert lengths[2:].sum() == 8


def test_tied_survival_stays_within_replicas(session: InferenceSession) -> None:
    tied = np.full((4, 4), 0.5, dtype=np.float32)
    check_replica_budgets(
        session,
        [
            (tied, [0, boundary, 4], width)
            for boundary in (2, 3)
            for width in (1, 2)
        ],
    )
    survival = sorted_survival(np.random.default_rng(12), 8, 3)
    survival[4:] = survival[:4]
    check_replica_budgets(
        session, [(survival, [0, 2, 4, 6, 8], width) for width in (1, 3)]
    )


def test_finite_scores_floor_the_pad_row(session: InferenceSession) -> None:
    k = 4
    splits = np.array([0, 2, 3], dtype=np.int64)
    drafts = np.array(
        [[7, 8, 9, 10], [MAGIC_DRAFT_TOKEN_ID] * k, [11, 12, 13, 14]],
        dtype=np.int64,
    )
    scores = np.array(
        [[-2.0] * k, [0.0] * k, [1.0] * k],
        dtype=np.float32,
    )
    coeffs = np.array([1.0] + [0.0] * k, dtype=np.float32)
    model = build_floored_lengths_model(session)
    for width in (1, 2):
        _, out, _ = model.execute(
            Buffer.from_numpy(drafts),
            Buffer.from_numpy(scores),
            Buffer.from_numpy(coeffs),
            Buffer.from_numpy(np.array(width, dtype=np.int32)),
            Buffer.from_numpy(splits),
        )
        lengths = out.to_numpy()
        survival = reference_survival(scores, coeffs)
        _, real = reference_lengths(survival[:1], 2 * width)
        _, other = reference_lengths(survival[2:], width)
        np.testing.assert_array_equal(lengths, [real[0], 0, other[0]])
        assert lengths[0] == min(2 * width, k)
