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
"""The sequential driver's per-row verify and head-free scores, on CPU."""

from __future__ import annotations

import re

import numpy as np
import pytest
from cpu_step_session import cpu_session as cpu_session
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import (
    Graph,
    TensorValue,
    ops,
)
from max.nn.legacy.identity import Identity
from max.pipelines.speculative.config import (
    MAGIC_DRAFT_TOKEN_ID,
    SpeculativeConfig,
    VerifyWidthRange,
)
from max.pipelines.speculative.driver import (
    Proposed,
    SequentialBatch,
    SequentialDriver,
)
from max.pipelines.speculative.spec_input_types import (
    SpecDecodeInputTypeSpec,
)
from max.pipelines.speculative.spec_target import Verified
from max.pipelines.speculative.verify_budget import SCORE_LIMIT
from sequential_step_helpers import (
    CPU_DEVICE,
    HIDDEN,
    PROMPT_LENS,
    B,
    K,
    RecordingProposer,
    W,
    execute,
    per_row_inputs,
    position_logits,
    run_per_row,
    sequential_step_graph,
)


@pytest.fixture(scope="module")
def graph_and_ir() -> tuple[Graph, str]:
    graph = sequential_step_graph(
        RecordingProposer(), per_row=True, observe_prefill=True
    )
    return graph, str(graph)


@pytest.fixture(scope="module")
def per_row_step(
    session: InferenceSession, graph_and_ir: tuple[Graph, str]
) -> Model:
    return session.load(graph_and_ir[0])


def test_each_row_verifies_its_own_prefix(per_row_step: Model) -> None:
    """The verify stub sees each row's prompt plus its kept drafts, and the
    batch's slot count is the static K + 1."""
    outputs = run_per_row(per_row_step)
    np.testing.assert_array_equal(outputs[3], [1, 3])
    tokens = np.arange(sum(PROMPT_LENS), dtype=np.int64) + 100
    drafts = np.arange(B * K, dtype=np.int64).reshape(B, K) + 200
    want = np.concatenate(
        [
            np.concatenate([tokens[0:1], drafts[0, :1]]),
            np.concatenate([tokens[1:3], drafts[1, :3]]),
        ]
    )
    np.testing.assert_array_equal(outputs[4], want)
    # The batch's slot count the adapter reads is the static K + 1.
    np.testing.assert_array_equal(outputs[6], [K + 1])
    np.testing.assert_array_equal(
        outputs[5], [100, 101, 102, 110, 111, 112, 113]
    )


@pytest.mark.parametrize("per_row", [False, True], ids=["uniform", "per_row"])
def test_the_host_mirror_strides_by_the_step_width(
    session: InferenceSession, per_row: bool
) -> None:
    """Checks that the uniform step width preserves the host mirror's
    replica boundaries."""
    model = session.load(
        sequential_step_graph(
            RecordingProposer(), per_row=per_row, host_mirrors=True
        )
    )
    width = W if per_row else K
    scores, coeffs = per_row_inputs((1, 3)) if per_row else (None, None)
    outputs = execute(
        model,
        per_row=per_row,
        drafts=np.arange(B * K, dtype=np.int64).reshape(B, K) + 200,
        logits=position_logits(),
        return_n_logits=np.array([width + 1], dtype=np.int64),
        scores=scores,
        coeffs=coeffs,
        host_offsets=np.array([0, 1, 3], dtype=np.uint32),
        splits=np.array([0, 2], dtype=np.int64),
    )
    if per_row:
        np.testing.assert_array_equal(outputs[3], [1, 3])
    expected_offsets = [0, 3, 7] if per_row else [0, 4, 9]
    np.testing.assert_array_equal(outputs[-1], expected_offsets)


def test_a_per_row_context_step_matches_the_flag_off_context_step(
    session: InferenceSession, per_row_step: Model
) -> None:
    """At ``return_n_logits = 1`` the per-row graph packs no draft and
    commits what the flag-off context step commits.

    The per-row target still reads K + 1 slots, each repeating the row's
    slot 0, because every slot reads the row's prefix slot min(t, 0).
    """
    slot0 = position_logits()[:: K + 1]
    plain_proposer = RecordingProposer()
    plain_proposer.prefill_slots = 1
    plain = execute(
        session.load(
            sequential_step_graph(
                plain_proposer, per_row=False, observe_prefill=True
            )
        ),
        per_row=False,
        drafts=np.zeros((B, 0), dtype=np.int64),
        logits=slot0,
        return_n_logits=np.array([1], dtype=np.int64),
    )
    drafts = np.arange(B * K, dtype=np.int64).reshape(B, K) + 200
    scores, coeffs = per_row_inputs([K, K])
    per_row = execute(
        per_row_step,
        per_row=True,
        drafts=drafts,
        logits=np.repeat(slot0, K + 1, axis=0),
        return_n_logits=np.array([1], dtype=np.int64),
        scores=scores,
        coeffs=coeffs,
    )
    np.testing.assert_array_equal(per_row[3], [0, 0])
    # Accepted, next tokens, next drafts, then the merged and shifted rows.
    for plain_out, row_out in zip(
        [*plain[:3], *plain[3:5]], [*per_row[:3], *per_row[4:6]], strict=True
    ):
        np.testing.assert_array_equal(row_out, plain_out)


def test_an_all_magic_row_discards_its_proposals(per_row_step: Model) -> None:
    """Checks that a row with no proposal yields its slots even when the target agrees."""
    drafts = np.arange(B * K, dtype=np.int64).reshape(B, K) + 200
    drafts[0] = MAGIC_DRAFT_TOKEN_ID
    scores, coeffs = per_row_inputs((1, 1))
    logits = position_logits()
    # Row 0's every slot peaks at the magic id, so the sampler would
    # accept its whole all-magic draft row; only the discard and the
    # selection floor keep it out of the commit and the budget.
    logits[0 : K + 1, MAGIC_DRAFT_TOKEN_ID] = 20.0
    num_accepted, next_tokens, _, lengths, _, _, _ = execute(
        per_row_step,
        per_row=True,
        drafts=drafts,
        logits=logits,
        return_n_logits=np.array([2], dtype=np.int64),
        scores=scores,
        coeffs=coeffs,
    )
    # The budget is B * 1: the live row takes both slots and the magic
    # row's floored entries win none, so it verifies nothing.
    np.testing.assert_array_equal(lengths, [0, 2])
    np.testing.assert_array_equal(num_accepted, [0, 0])
    # The committed token is re-picked at index 0: the magic row's own
    # slot-0 argmax (the magic id), not the sampler's first-reject pick.
    np.testing.assert_array_equal(next_tokens, [MAGIC_DRAFT_TOKEN_ID, 110])


def test_a_no_proposal_row_discards_at_full_width(per_row_step: Model) -> None:
    """At w = K a no-proposal row keeps no slot and commits nothing.

    Its drafts reach the sampler as unverified placeholders, so no slot
    is accepted and the row's commit token is its slot-0 argmax.
    """
    drafts = np.arange(B * K, dtype=np.int64).reshape(B, K) + 200
    drafts[0] = MAGIC_DRAFT_TOKEN_ID
    scores, coeffs = per_row_inputs((K, K))
    logits = position_logits()
    logits[:K, MAGIC_DRAFT_TOKEN_ID] = 20.0
    num_accepted, next_tokens, _, lengths, _, _, _ = execute(
        per_row_step,
        per_row=True,
        drafts=drafts,
        logits=logits,
        return_n_logits=np.array([K + 1], dtype=np.int64),
        scores=scores,
        coeffs=coeffs,
    )
    # The selection drops the no-proposal row, so only the live row keeps
    # its K entries.
    np.testing.assert_array_equal(lengths, [0, K])
    np.testing.assert_array_equal(num_accepted, [0, 0])
    np.testing.assert_array_equal(next_tokens, [MAGIC_DRAFT_TOKEN_ID, 110])


def test_a_no_proposal_row_spends_no_slots_at_a_wider_budget(
    per_row_step: Model,
) -> None:
    """The floor ranks a no-proposal row's entries last, on either row.

    Every real survival sits at the clamp floor, so only the -1 floor
    separates the magic row; the live row takes its whole K prefix
    whatever the row order, and the magic row keeps none. The 2 * W
    budget exceeds K, so one slot stays unspent; the host's spendable
    width cap keeps that shape off a real boot.
    """
    for magic_row in (0, 1):
        drafts = np.arange(B * K, dtype=np.int64).reshape(B, K) + 200
        drafts[magic_row] = MAGIC_DRAFT_TOKEN_ID
        scores = np.full((B, K), -SCORE_LIMIT, dtype=np.float32)
        coeffs = np.array([1.0] + [0.0] * K, dtype=np.float32)
        num_accepted, next_tokens, _, lengths, _, _, _ = execute(
            per_row_step,
            per_row=True,
            drafts=drafts,
            logits=position_logits(),
            return_n_logits=np.array([W + 1], dtype=np.int64),
            scores=scores,
            coeffs=coeffs,
        )
        expected = [0, K] if magic_row == 0 else [K, 0]
        np.testing.assert_array_equal(lengths, expected)
        np.testing.assert_array_equal(num_accepted, [0, 0])
        np.testing.assert_array_equal(next_tokens, [100, 110])


def test_the_discard_fires_with_adaptive_verify_off(
    session: InferenceSession,
) -> None:
    """The plain path has no selection, so the discard recounts itself."""
    proposer = RecordingProposer()
    model = session.load(sequential_step_graph(proposer, per_row=False))
    drafts = np.arange(B * K, dtype=np.int64).reshape(B, K) + 200
    drafts[0] = MAGIC_DRAFT_TOKEN_ID
    logits = position_logits()
    # The target echoes the magic id at row 0's slot 0, so only the
    # discard keeps the sampler's acceptance out of the commit.
    logits[0, MAGIC_DRAFT_TOKEN_ID] = 20.0
    num_accepted, next_tokens, *_ = execute(
        model,
        per_row=False,
        drafts=drafts,
        logits=logits,
        return_n_logits=np.array([K + 1], dtype=np.int64),
    )
    np.testing.assert_array_equal(num_accepted, [0, 0])
    np.testing.assert_array_equal(next_tokens, [MAGIC_DRAFT_TOKEN_ID, 110])


def test_a_magic_draft_the_sampler_accepts_is_still_discarded(
    per_row_step: Model,
) -> None:
    """A no-proposal row commits nothing even when the target echoes it.

    The target's slot-0 argmax is the magic id itself, so without the
    discard the sampler would report the row accepted and commit past
    tokens the iteration never produced.
    """
    drafts = np.arange(B * K, dtype=np.int64).reshape(B, K) + 200
    drafts[0] = MAGIC_DRAFT_TOKEN_ID
    scores, coeffs = per_row_inputs((1, 1))
    logits = position_logits()
    # Row 0's slot 0 peaks at the magic id, so the sampler accepts it.
    logits[0, MAGIC_DRAFT_TOKEN_ID] = 20.0
    num_accepted, next_tokens, *_ = execute(
        per_row_step,
        per_row=True,
        drafts=drafts,
        logits=logits,
        return_n_logits=np.array([W + 1], dtype=np.int64),
        scores=scores,
        coeffs=coeffs,
    )
    np.testing.assert_array_equal(num_accepted, [0, 0])
    # Row 0 commits the target's slot-0 token, the magic id itself.
    np.testing.assert_array_equal(next_tokens, [MAGIC_DRAFT_TOKEN_ID, 110])


def test_the_discard_shares_the_selections_magic_count(
    graph_and_ir: tuple[Graph, str],
) -> None:
    """One magic compare serves the selection floor and the discard."""
    _, ops_text = graph_and_ir
    # A static row lets vectorized loads assume alignment the draft buffer lacks.
    assert not re.search(
        r"mo\.equal[^\n]*\[batch_size, 3\][^\n]*si64", ops_text
    ), "the magic compare must read a symbolic row, not a static K row"
    # The magic compare runs once, over the padded drafts row; the
    # discard reads its result rather than recounting.
    magic_compares = re.findall(
        r"mo\.equal[^\n]*\[batch_size, add\(num_steps, 1\)\][^\n]*si64[^\n]*\n",
        ops_text,
    )
    assert len(magic_compares) == 1, "the discard must not recount"


def _unused_verify(batch: SequentialBatch) -> Verified[object]:
    raise NotImplementedError


def test_sequential_driver_rechecks_per_row_acceptance() -> None:
    config = SpeculativeConfig(
        speculative_method="mtp",
        num_speculative_tokens=K,
        synthetic_acceptance_rate=1.0,
    )
    config = config.model_copy(update={"per_row_verify": True})
    with pytest.raises(
        ValueError, match=r"per_row_verify.*synthetic_acceptance_rate"
    ):
        SequentialDriver(
            _unused_verify,
            RecordingProposer(),
            target_model=Identity(),
            draft_model=Identity(),
            input_spec=SpecDecodeInputTypeSpec(
                devices=[CPU_DEVICE], distributed=False
            ),
            speculative_config=config,
        )


def test_sequential_driver_rechecks_the_score_source_rule() -> None:
    """A copied config that drops the per-row verify keeps its score
    source, whose scores nothing reads; the driver's recheck must
    refuse it."""
    config = SpeculativeConfig(
        speculative_method="mtp",
        num_speculative_tokens=K,
        per_row_verify=True,
        draft_score_source="head_free",
    )
    config = config.model_copy(update={"per_row_verify": False})
    with pytest.raises(
        ValueError, match="draft_score_source needs per_row_verify"
    ):
        SequentialDriver(
            _unused_verify,
            RecordingProposer(),
            target_model=Identity(),
            draft_model=Identity(),
            input_spec=SpecDecodeInputTypeSpec(
                devices=[CPU_DEVICE], distributed=False
            ),
            speculative_config=config,
        )


def test_input_spec_cannot_enable_per_row_verify() -> None:
    with pytest.raises(
        ValueError, match="include_per_row_verify requires per_row_verify"
    ):
        SequentialDriver(
            _unused_verify,
            RecordingProposer(),
            target_model=Identity(),
            draft_model=Identity(),
            input_spec=SpecDecodeInputTypeSpec(
                devices=[CPU_DEVICE],
                distributed=False,
                include_per_row_verify=True,
                num_draft_positions=K,
            ),
            speculative_config=SpeculativeConfig(
                speculative_method="mtp",
                num_speculative_tokens=K,
                synthetic_acceptance_rate=1.0,
            ),
        )


def test_per_row_refuses_a_skippable_draft_graph() -> None:
    """A schedule that reaches width 0 builds the skippable loop, whose
    runtime row count the per-row merge's static rebind cannot follow."""

    driver = SequentialDriver.__new__(SequentialDriver)
    with pytest.raises(ValueError, match="skippable-draft"):
        SequentialDriver.__init__(
            driver,
            _unused_verify,
            RecordingProposer(),
            target_model=Identity(),
            draft_model=Identity(),
            input_spec=SpecDecodeInputTypeSpec(
                devices=[CPU_DEVICE], distributed=False
            ),
            speculative_config=SpeculativeConfig(
                speculative_method="mtp",
                num_speculative_tokens=K,
                num_speculative_tokens_per_batch_size=[
                    VerifyWidthRange(batch_start=1, batch_end=4, num_tokens=0)
                ],
                per_row_verify=True,
            ),
        )


def test_per_row_needs_a_proposer_that_reads_the_lengths() -> None:
    """A proposer without the declaration is refused at construction."""

    class _BlindProposer(RecordingProposer):
        reads_verify_lengths = False

    driver = SequentialDriver.__new__(SequentialDriver)
    with pytest.raises(ValueError, match="reads_verify_lengths"):
        SequentialDriver.__init__(
            driver,
            _unused_verify,
            _BlindProposer(),
            target_model=Identity(),
            draft_model=Identity(),
            input_spec=SpecDecodeInputTypeSpec(
                devices=[CPU_DEVICE], distributed=False
            ),
            speculative_config=SpeculativeConfig(
                speculative_method="mtp",
                num_speculative_tokens=K,
                per_row_verify=True,
            ),
        )


def test_the_sequential_driver_refuses_head_scores() -> None:
    """A head score source is refused at construction: the sequential
    driver mounts no head, and a silent head-free fallback would mislabel
    the export's scores."""

    driver = SequentialDriver.__new__(SequentialDriver)
    with pytest.raises(ValueError, match="draft_score_source='head'"):
        SequentialDriver.__init__(
            driver,
            _unused_verify,
            RecordingProposer(),
            target_model=Identity(),
            draft_model=Identity(),
            input_spec=SpecDecodeInputTypeSpec(
                devices=[CPU_DEVICE], distributed=False
            ),
            speculative_config=SpeculativeConfig(
                speculative_method="mtp",
                num_speculative_tokens=K,
                per_row_verify=True,
                draft_score_source="head",
            ),
        )


def test_per_row_refuses_a_proposer_that_gathers_its_own_token() -> None:
    """A prefill that returns its step-0 token hides the logits the
    per-row verify scores that step from, so the build is refused."""

    class _GatheringProposer(RecordingProposer):
        def prefill(
            self,
            batch: SequentialBatch,
            tokens: TensorValue,
            target_hidden: object,
        ) -> Proposed:
            carry = ops.constant(
                np.zeros((B, HIDDEN), dtype=np.float32),
                DType.float32,
                CPU_DEVICE,
            )
            return Proposed(
                logits=None,
                hidden=[carry],
                carry=[carry],
                token=ops.constant(
                    np.zeros(B, np.int64), DType.int64, CPU_DEVICE
                ),
            )

    with pytest.raises(ValueError, match="the per-row verify needs"):
        sequential_step_graph(
            _GatheringProposer(), per_row=True, head_free=True
        )
