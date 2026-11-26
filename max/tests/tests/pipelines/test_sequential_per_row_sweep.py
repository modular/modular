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
"""Additional sequential per row CPU cases."""

from __future__ import annotations

import numpy as np
from cpu_step_session import cpu_session as cpu_session
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    TensorValue,
    ops,
)
from max.pipelines.speculative.config import MAGIC_DRAFT_TOKEN_ID
from max.pipelines.speculative.driver import (
    DraftStepInput,
    Proposed,
    SequentialBatch,
)
from sequential_step_helpers import (
    CPU_DEVICE,
    HIDDEN,
    PROMPT_LENS,
    VOCAB,
    B,
    K,
    RecordingProposer,
    execute,
    per_row_inputs,
    position_logits,
    run_per_row,
    sequential_step_graph,
)


def test_the_draft_carry_reads_each_rows_accepted_position(
    session: InferenceSession,
) -> None:
    """The carry hidden equals the merged position end_r - 1 - k_r +
    accepted_r.

    The stub's prefill hidden is position-valued: each merged row holds
    its own position, so the gathered carry names the position the index
    picked. The carry feeds step 1, whose logits peak at it, so the
    step-1 draft token encodes the position.
    """
    lengths = [1, 3]

    class _PositionProposer(RecordingProposer):
        def prefill(
            self,
            batch: SequentialBatch,
            tokens: TensorValue,
            target_hidden: object,
        ) -> Proposed:
            call = super().prefill(batch, tokens, target_hidden)
            # One position-valued hidden row per merged row; the window
            # holds the prompts plus the packed drafts.
            total = sum(PROMPT_LENS) + sum(lengths)
            hidden = np.zeros((total, HIDDEN), dtype=np.float32)
            for pos in range(total):
                hidden[pos, 0] = pos
            return Proposed(
                logits=call.logits,
                hidden=[ops.constant(hidden, DType.float32, CPU_DEVICE)],
            )

        def step(
            self,
            batch: SequentialBatch,
            draft_input: DraftStepInput,
            index: int,
        ) -> Proposed:
            # The carried hidden names the accepted position; peak the
            # step's logits there so the argmax emits it as the token.
            carry = draft_input.hidden[0][:, 0].cast(DType.int64)
            vocab = ops.range(
                0, VOCAB, 1, out_dim=VOCAB, device=CPU_DEVICE, dtype=DType.int64
            )
            one_hot = vocab == ops.unsqueeze(carry, -1)
            logits = one_hot.cast(DType.float32) * 10.0
            return Proposed(
                logits=logits,
                hidden=[
                    ops.constant(
                        np.zeros((B, HIDDEN), dtype=np.float32),
                        DType.float32,
                        CPU_DEVICE,
                    )
                ],
            )

    model = session.load(
        sequential_step_graph(_PositionProposer(), per_row=True)
    )
    outputs = run_per_row(model, lengths)
    np.testing.assert_array_equal(outputs[0], [0, 0])
    np.testing.assert_array_equal(outputs[2][:, 1], [0, 3])


def test_head_free_scores_are_each_pass_top_token_logit(
    session: InferenceSession,
) -> None:
    """outputs[3] equals the top-token log-odds of each pass's logits.

    Row 0's slot-0 draft matches the target argmax, so it accepts one
    position and step 0 scores its slot-1 row; row 1 rejects everywhere
    and scores slot 0. Each slot and step peaks at its own height, so a
    score read off the wrong slot or pass differs.
    """
    proposer = RecordingProposer()
    model = session.load(
        sequential_step_graph(proposer, per_row=False, head_free=True)
    )
    drafts = np.arange(B * K, dtype=np.int64).reshape(B, K) + 200
    # Row 0's slot-0 draft matches the target's slot-0 argmax (100).
    drafts[0, 0] = 100
    outputs = execute(
        model,
        per_row=False,
        drafts=drafts,
        logits=position_logits(),
        return_n_logits=np.array([K + 1], dtype=np.int64),
    )
    scores = outputs[3]
    assert scores.shape == (B, K)

    def want(height: float) -> float:
        return height - float(np.log(VOCAB - 1))

    # Distinct peak heights expose a score read from the wrong slot or pass.
    expected = np.array(
        [
            [want(12.0), want(12.0), want(14.0)],
            [want(10.0), want(12.0), want(14.0)],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(scores, expected, rtol=1e-5)


def test_per_row_outputs_come_in_signature_order(
    session: InferenceSession,
) -> None:
    """Head-free + per-row gives its outputs in the named order.

    The sampled proposal's distributions are GPU-only, so the CPU run
    covers the argmax arm: scores, then verify_lengths, then (on a GPU
    build) the distributions.
    """
    proposer = RecordingProposer()
    model = session.load(
        sequential_step_graph(proposer, per_row=True, head_free=True)
    )
    outputs = run_per_row(model)
    assert len(outputs) == 5
    num_accepted, next_tokens, next_drafts, out_scores, lengths = outputs
    assert num_accepted.shape == (B,)
    assert next_tokens.shape == (B,)
    assert next_drafts.shape == (B, K)
    assert out_scores.shape == (B, K)
    assert out_scores.dtype == np.float32
    assert lengths.shape == (B,)
    assert lengths.dtype == np.int32


def test_full_width_rejection_recovers_each_rows_first_token(
    session: InferenceSession,
) -> None:
    model = session.load(
        sequential_step_graph(RecordingProposer(), per_row=True)
    )
    drafts = np.arange(B * K, dtype=np.int64).reshape(B, K) + 200
    drafts[0] = MAGIC_DRAFT_TOKEN_ID
    scores, coeffs = per_row_inputs((K, K))
    num_accepted, next_tokens, _, lengths = execute(
        model,
        per_row=True,
        drafts=drafts,
        logits=position_logits(),
        return_n_logits=np.array([K + 1], dtype=np.int64),
        scores=scores,
        coeffs=coeffs,
    )
    # The no-proposal row keeps no verify slot, so only the live row
    # keeps its K entries.
    np.testing.assert_array_equal(lengths, [0, K])
    np.testing.assert_array_equal(num_accepted, [0, 0])
    np.testing.assert_array_equal(next_tokens, [100, 110])
