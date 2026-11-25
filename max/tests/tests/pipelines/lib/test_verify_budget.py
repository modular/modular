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
"""Tests the per-row verify budget helpers."""

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
    select_verify_lengths,
    unverified_draft_placeholders,
)
from speculative_test_helpers import (
    RANKED_SURVIVAL,
    build_floored_lengths_model,
    build_lengths_model,
    check_replica_budgets,
    reference_lengths,
    reference_survival,
    sorted_survival,
)
from speculative_test_helpers import cpu_session as cpu_session


@pytest.fixture(scope="module")
def lengths_model(session: InferenceSession) -> Model:
    return build_lengths_model(session, ["batch_size", 3])


def test_selection_matches_the_sorted_reference(lengths_model: Model) -> None:
    rng = np.random.default_rng(1)
    for batch, totals in ((1, (0, 1)), (3, (3, 4, 9, 10))):
        survival = sorted_survival(rng, batch, 3)
        for total in totals:
            (out,) = lengths_model.execute(
                Buffer.from_numpy(survival),
                Buffer.from_numpy(np.array(total, dtype=np.int32)),
            )
            lengths = out.to_numpy()
            kept, expected = reference_lengths(survival, total)
            np.testing.assert_array_equal(lengths, expected)
            np.testing.assert_array_equal(np.arange(3) < lengths[:, None], kept)
            assert lengths.sum() == min(total, survival.size)


def test_ties_split_by_position_then_row(lengths_model: Model) -> None:
    for rows, total, expected in (
        ([[0.5] * 3] * 3, 3, [1, 1, 1]),
        ([[0.9, 0.5, 0.5], [0.8, 0.5, 0.5]], 3, [2, 1]),
        ([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], 3, [2, 1]),
    ):
        (out,) = lengths_model.execute(
            Buffer.from_numpy(np.array(rows, dtype=np.float32)),
            Buffer.from_numpy(np.array(total, dtype=np.int32)),
        )
        np.testing.assert_array_equal(out.to_numpy(), expected)


def test_graphs_build_on_a_gpu_device() -> None:
    # The selection's tie keys must live on the input's device, so the
    # graph must build with non-CPU input types.
    device = DeviceRef.GPU(0)
    with Graph(
        "draft_survival_gpu",
        input_types=[
            TensorType(DType.float32, [4, 5], device),
            TensorType(DType.float32, [6], device),
        ],
    ) as graph:
        graph.output(
            draft_survival(graph.inputs[0].tensor, graph.inputs[1].tensor)
        )
    with Graph(
        "select_verify_lengths_gpu",
        input_types=[
            TensorType(DType.float32, [4, 5], device),
            TensorType(DType.int32, [], device),
        ],
    ) as graph:
        graph.output(
            select_verify_lengths(
                graph.inputs[0].tensor, graph.inputs[1].tensor
            )
        )


def test_graphs_build_with_a_symbolic_batch_dim() -> None:
    # The spec graphs name their batch dim symbolically, so the
    # selection must build with a non-static batch input type.
    device = DeviceRef.GPU(0)
    with Graph(
        "select_verify_lengths_symbolic_batch",
        input_types=[
            TensorType(DType.float32, ["batch_size", 5], device),
            TensorType(DType.int32, [], device),
        ],
    ) as graph:
        graph.output(
            select_verify_lengths(
                graph.inputs[0].tensor, graph.inputs[1].tensor
            )
        )


def test_new_graphs_build_with_a_symbolic_batch_dim() -> None:
    # The spec graphs name their batch dim symbolically, so the new
    # entries must build with a non-static batch input type.
    device = DeviceRef.GPU(0)
    with Graph(
        "select_replica_verify_lengths_symbolic_batch",
        input_types=[
            TensorType(DType.float32, ["batch_size", 5], device),
            TensorType(DType.int32, [], device),
            TensorType(DType.int64, [3], DeviceRef.CPU()),
        ],
    ) as graph:
        graph.output(
            select_replica_verify_lengths(
                graph.inputs[0].tensor,
                graph.inputs[1].tensor,
                graph.inputs[2].tensor,
            )
        )
    with Graph(
        "unverified_draft_placeholders_symbolic_batch",
        input_types=[
            TensorType(DType.int64, ["batch_size", 4], device),
            TensorType(DType.int32, ["batch_size"], device),
        ],
    ) as graph:
        graph.output(
            unverified_draft_placeholders(
                graph.inputs[0].tensor, graph.inputs[1].tensor
            )
        )


_TIED = np.full_like(RANKED_SURVIVAL, 0.5)


def test_replica_budgets_match_the_sorted_reference(
    session: InferenceSession,
) -> None:
    check_replica_budgets(
        session,
        [
            (RANKED_SURVIVAL, [0, 1, 8], 1),
            (RANKED_SURVIVAL, [0, 6, 8], 2),
            (RANKED_SURVIVAL, [0, 0, 8], 1),
            (RANKED_SURVIVAL, [0, 4, 8], 1),
            (RANKED_SURVIVAL, [0, 4, 8], 3),
            (_TIED, [0, 4, 8], 1),
            (_TIED, [0, 6, 8], 2),
        ],
    )


@pytest.fixture(scope="module")
def floored_lengths_model(session: InferenceSession) -> Model:
    return build_floored_lengths_model(session)


@pytest.mark.parametrize("width", [0, 1, 2])
def test_a_magic_pad_row_yields_its_replica_budget_to_the_real_row(
    width: int,
    floored_lengths_model: Model,
) -> None:
    k = 4
    # Without the floor the pad row would outrank its replica's real drafts.
    splits = np.array([0, 2, 3], dtype=np.int64)
    drafts = np.array(
        [[7, 8, 9, 10], [MAGIC_DRAFT_TOKEN_ID] * k, [11, 12, 13, 14]],
        dtype=np.int64,
    )
    scores = np.array(
        [[-np.inf, -2.0, -1.0, -0.5], [0.0] * k, [np.inf, 0.0, 1.0, np.nan]],
        dtype=np.float32,
    )
    coeffs = np.array([0.025, -0.1, 0.2, -0.3, 0.1], dtype=np.float32)
    survival_out, lengths_out, masked = floored_lengths_model.execute(
        Buffer.from_numpy(drafts),
        Buffer.from_numpy(scores),
        Buffer.from_numpy(coeffs),
        Buffer.from_numpy(np.array(width, dtype=np.int32)),
        Buffer.from_numpy(splits),
    )
    lengths = lengths_out.to_numpy()
    survival = reference_survival(scores, coeffs)
    survival[1] = -1.0
    np.testing.assert_allclose(survival_out.to_numpy(), survival, atol=1e-6)
    # The real row ranks as if the pad row were absent from the rank, over
    # the replica's whole padded budget.
    _, real = reference_lengths(survival[:1], 2 * width)
    _, other = reference_lengths(survival[2:], width)
    np.testing.assert_array_equal(lengths, [real[0], 0, other[0]])
    assert lengths[0] == min(2 * width, k)
    expected_drafts = np.where(np.arange(k) >= lengths[:, None], -1, drafts)
    np.testing.assert_array_equal(masked.to_numpy(), expected_drafts)


def test_a_no_proposal_batch_verifies_nothing(
    floored_lengths_model: Model,
) -> None:
    """A batch of all-MAGIC rows takes no verify slots.

    At the top width the budget covers every entry, so the survival floor
    alone leaves every no-proposal row its full prefix.
    """
    k = 4
    drafts = np.full((2, k), MAGIC_DRAFT_TOKEN_ID, dtype=np.int64)
    # A high score cannot buy a row a slot it has no proposal for.
    scores = np.zeros((2, k), dtype=np.float32)
    coeffs = np.array([1.0] + [0.0] * k, dtype=np.float32)
    splits = np.array([0, 2, 2], dtype=np.int64)
    _, lengths_out, _ = floored_lengths_model.execute(
        Buffer.from_numpy(drafts),
        Buffer.from_numpy(scores),
        Buffer.from_numpy(coeffs),
        Buffer.from_numpy(np.array(k, dtype=np.int32)),
        Buffer.from_numpy(splits),
    )
    np.testing.assert_array_equal(lengths_out.to_numpy(), [0, 0])


def test_a_no_proposal_row_gets_no_slots_beside_real_rows(
    floored_lengths_model: Model,
) -> None:
    """A no-proposal row verifies nothing while real rows keep their prefixes."""
    k = 4
    splits = np.array([0, 3, 3], dtype=np.int64)
    drafts = np.array(
        [[7, 8, 9, 10], [MAGIC_DRAFT_TOKEN_ID] * k, [11, 12, 13, 14]],
        dtype=np.int64,
    )
    # The pad row scores highest, so the floor alone would still hand it
    # slots at this width.
    scores = np.array([[-5.0] * k, [0.0] * k, [-5.0] * k], dtype=np.float32)
    coeffs = np.array([1.0] + [0.0] * k, dtype=np.float32)
    _, lengths_out, _ = floored_lengths_model.execute(
        Buffer.from_numpy(drafts),
        Buffer.from_numpy(scores),
        Buffer.from_numpy(coeffs),
        Buffer.from_numpy(np.array(k, dtype=np.int32)),
        Buffer.from_numpy(splits),
    )
    np.testing.assert_array_equal(lengths_out.to_numpy(), [4, 0, 4])
