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
"""A sampled per-row driver step on the GPU, at the inputs the engine binds.

The export declares the drafts and their distributions with the symbolic
``num_steps``, and the per-row graph binds all ``K`` proposals on every
step: a verify step at any average width, and a context step, whose drafts
and scores are zero and whose ``return_n_logits`` is 1.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import (
    BufferType,
    DeviceRef,
    DimLike,
    Graph,
    TensorType,
    TensorValue,
    ops,
)
from max.nn._graph_ops import (
    gumbel_argmax_from_probs,
    topk_topp_masked_probs,
)
from max.nn._graph_ops.sampling import _SEED_DOMAIN_RECOVERY
from max.nn.outputs import ReturnHiddenStates
from max.pipelines.speculative.driver import (
    CarryDimNames,
    DecodeKVSwap,
    DraftCache,
    DraftStepInput,
    Proposed,
    ReuseSpec,
    SequentialBatch,
)
from spec_step_stubs import (
    outputs_to_numpy,
    stub_kv_leaf,
    stub_sequential_driver,
    unrestricted_sampling_buffers,
)

pytestmark = pytest.mark.skipif(
    accelerator_count() == 0, reason="the sampled verdict is GPU-only"
)

B = 2
K = 5
VOCAB = 16
HIDDEN = 4
PROMPT_LENS = [1, 2]
PEAK_LOGIT = 20.0
# Slot t of every row peaks at token FIRST_PEAK + t.
FIRST_PEAK = 8

_GPU = DeviceRef.GPU(0)
_CPU = DeviceRef.CPU()


def _slot_logits(rows: int) -> np.ndarray:
    """Logits over ``rows`` rows of ``K + 1`` slots, slot t peaked at its own
    token, so the next token names the slot it was drawn from."""
    logits = np.zeros((rows * (K + 1), VOCAB), dtype=np.float32)
    for r in range(rows * (K + 1)):
        logits[r, FIRST_PEAK + r % (K + 1)] = PEAK_LOGIT
    return logits


class _StubProposer:
    """A proposer whose passes return fixed logits on the GPU."""

    reads_verify_lengths: bool = True
    hidden_dim: DimLike = HIDDEN
    reuse: ReuseSpec | None = None
    decode_swaps: tuple[DecodeKVSwap, ...] = ()
    passthrough_decode_swaps: tuple[str, ...] = ()
    draft_cache: DraftCache = DraftCache.OWN
    split_prefix: str = "stub"
    carry_dim_names: CarryDimNames = CarryDimNames()
    step_hidden_mode: ReturnHiddenStates = ReturnHiddenStates.ALL
    uses_thinking_phase: bool = False
    supports_zero_draft_rows: bool = False

    def prefill(
        self,
        batch: SequentialBatch,
        tokens: TensorValue,
        target_hidden: object,
    ) -> Proposed:
        rows = B * (K + 1)
        return Proposed(
            logits=ops.constant(_slot_logits(B), DType.float32, _GPU),
            hidden=[
                ops.constant(
                    np.zeros((rows, HIDDEN), dtype=np.float32),
                    DType.float32,
                    _GPU,
                )
            ],
        )

    def step(
        self,
        batch: SequentialBatch,
        draft_input: DraftStepInput,
        index: int,
    ) -> Proposed:
        logits = np.zeros((B, VOCAB), dtype=np.float32)
        logits[:, index % VOCAB] = PEAK_LOGIT
        return Proposed(
            logits=ops.constant(logits, DType.float32, _GPU),
            hidden=[
                ops.constant(
                    np.zeros((B, HIDDEN), dtype=np.float32),
                    DType.float32,
                    _GPU,
                )
            ],
        )


def _input_types() -> list[TensorType | BufferType]:
    """The step's inputs, with the export's declared shapes for the drafts,
    their distributions and the score pair."""
    return [
        TensorType(DType.int64, ["total_seq_len"], _GPU),
        TensorType(DType.uint32, ["input_row_offsets_len"], _GPU),
        TensorType(DType.int64, ["batch_size", "num_steps"], _GPU),
        TensorType(DType.float32, ["logits_rows", VOCAB], _GPU),
        TensorType(DType.int64, [1], _CPU),
        BufferType(DType.int64, [1], _GPU),
        TensorType(DType.uint64, ["batch_size"], _GPU),
        TensorType(DType.float32, ["batch_size"], _GPU),
        TensorType(DType.int64, ["batch_size"], _GPU),
        TensorType(DType.int64, [], _CPU),
        TensorType(DType.float32, ["batch_size"], _GPU),
        TensorType(DType.float32, [], _CPU),
        TensorType(DType.float32, ["batch_size", "num_steps", VOCAB], _GPU),
        TensorType(DType.float32, ["batch_size", K], _GPU),
        TensorType(DType.float32, [K + 1], _GPU),
    ]


@pytest.fixture(scope="module")
def model() -> Model:
    """One per-row driver step, built once for every case."""
    with Graph("sampled_per_row_step", input_types=_input_types()) as graph:
        (
            tokens,
            offsets,
            drafts,
            logits,
            return_n_logits,
            kv_blocks,
            seed,
            temperature,
            top_k,
            max_k,
            top_p,
            min_top_p,
            dists,
            scores,
            coeffs,
        ) = graph.inputs
        leaf = stub_kv_leaf(
            ops.constant(np.zeros(B, np.int64), DType.int64, _GPU),
            kv_blocks.buffer,
            _CPU,
        )
        driver = stub_sequential_driver(
            _StubProposer(),
            device=_GPU,
            num_draft_steps=K,
            per_row=True,
            head_free=True,
            draft_proposal="sampled",
            vocab_size=VOCAB,
        )
        outputs = driver._run(
            tokens.tensor,
            offsets.tensor,
            drafts.tensor,
            kv_collections=[leaf],
            draft_kv_collections=[leaf],
            passthrough_kv={},
            return_n_logits=return_n_logits.tensor,
            seed=seed.tensor,
            temperature=temperature.tensor,
            top_k=top_k.tensor,
            max_k=max_k.tensor,
            top_p=top_p.tensor,
            min_top_p=min_top_p.tensor,
            extra={"logits": logits.tensor},
            batch_context_lengths=[
                ops.constant(np.zeros(1, dtype=np.int64), DType.int64, _CPU)
            ],
            draft_probs_full=dists.tensor,
            draft_scores=scores.tensor,
            draft_score_coeffs=coeffs.tensor,
        )
        graph.output(*outputs)
    return InferenceSession(devices=[CPU(), Accelerator()]).load(graph)


def _execute(
    model: Model,
    *,
    drafts: np.ndarray,
    dists: np.ndarray,
    return_n_logits: int,
    scores: np.ndarray,
    coeffs: np.ndarray,
    logits: np.ndarray,
) -> list[np.ndarray]:
    """Runs one step: num_accepted, next_tokens, next drafts, next scores,
    verify_lengths and next distributions, in signature order."""
    gpu = Accelerator()
    buffers = [
        Buffer.from_numpy(np.arange(sum(PROMPT_LENS), dtype=np.int64) + 100).to(
            gpu
        ),
        Buffer.from_numpy(np.cumsum([0, *PROMPT_LENS]).astype(np.uint32)).to(
            gpu
        ),
        Buffer.from_numpy(drafts).to(gpu),
        Buffer.from_numpy(logits).to(gpu),
        Buffer.from_numpy(np.array([return_n_logits], dtype=np.int64)),
        Buffer.from_numpy(np.zeros(1, dtype=np.int64)).to(gpu),
        *unrestricted_sampling_buffers(B, VOCAB, gpu),
        Buffer.from_numpy(dists).to(gpu),
        Buffer.from_numpy(scores).to(gpu),
        Buffer.from_numpy(coeffs).to(gpu),
    ]
    return outputs_to_numpy(model.execute(*buffers))


def _verify_inputs(
    lengths: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Drafts that match every slot's peak, their one-hot distributions,
    and a score pair that keeps each row's first ``lengths[r]`` drafts."""
    drafts = np.tile(np.arange(K, dtype=np.int64) + FIRST_PEAK, (B, 1))
    dists = np.zeros((B, K, VOCAB), dtype=np.float32)
    for t in range(K):
        dists[:, t, FIRST_PEAK + t] = 1.0
    scores = np.full((B, K), -5.0, dtype=np.float32)
    for row, length in enumerate(lengths):
        scores[row, :length] = 5.0
    coeffs = np.array([1.0] + [0.0] * K, dtype=np.float32)
    return drafts, dists, scores, coeffs


@pytest.mark.parametrize(
    "width, lengths", [(3, [1, 5]), (K, [K, K])], ids=["w3", "w5"]
)
def test_a_verify_step_keeps_each_rows_prefix(
    model: Model, width: int, lengths: list[int]
) -> None:
    """A verify step at average width w binds all K drafts and asks for
    w + 1 logits; each row accepts its kept prefix and draws the next
    token from the slot after it."""
    drafts, dists, scores, coeffs = _verify_inputs(lengths)
    outputs = _execute(
        model,
        drafts=drafts,
        dists=dists,
        return_n_logits=width + 1,
        scores=scores,
        coeffs=coeffs,
        logits=_slot_logits(B),
    )
    assert len(outputs) == 6
    num_accepted, next_tokens, next_drafts, _scores, verify_lengths, _ = outputs
    np.testing.assert_array_equal(verify_lengths, lengths)
    np.testing.assert_array_equal(num_accepted, lengths)
    np.testing.assert_array_equal(
        next_tokens.reshape(-1), [FIRST_PEAK + n for n in lengths]
    )
    assert next_drafts.shape == (B, K)


def test_a_context_step_binds_zeroed_drafts_at_the_export_width(
    model: Model,
) -> None:
    """A context step's K zero drafts, zero distributions and zero score
    pair verify nothing: the next token comes from slot 0."""
    outputs = _execute(
        model,
        drafts=np.zeros((B, K), dtype=np.int64),
        dists=np.zeros((B, K, VOCAB), dtype=np.float32),
        return_n_logits=1,
        scores=np.zeros((B, K), dtype=np.float32),
        coeffs=np.zeros(K + 1, dtype=np.float32),
        logits=_slot_logits(B),
    )
    num_accepted, next_tokens, _drafts, _scores, verify_lengths, _ = outputs
    np.testing.assert_array_equal(verify_lengths, [0, 0])
    np.testing.assert_array_equal(num_accepted, [0, 0])
    np.testing.assert_array_equal(next_tokens.reshape(-1), [FIRST_PEAK] * B)


def test_a_context_step_draws_the_placeholder_id_from_the_full_target(
    model: Model,
) -> None:
    """A context step whose target puts mass on the placeholder draft id
    (token 0) still draws the next token from the full target, not from
    the target minus a one-hot at the placeholder."""
    logits = _placeholder_peak_logits()
    # The same-seed draw from the full target, which can name the
    # placeholder id itself; built first so precompile-mefs records it.
    expected = _full_target_draw(logits)
    outputs = _execute(
        model,
        drafts=np.zeros((B, K), dtype=np.int64),
        dists=np.zeros((B, K, VOCAB), dtype=np.float32),
        return_n_logits=1,
        scores=np.zeros((B, K), dtype=np.float32),
        coeffs=np.zeros(K + 1, dtype=np.float32),
        logits=logits,
    )
    num_accepted, next_tokens, _drafts, _scores, verify_lengths, _ = outputs
    np.testing.assert_array_equal(verify_lengths, [0, 0])
    np.testing.assert_array_equal(num_accepted, [0, 0])
    np.testing.assert_array_equal(next_tokens.reshape(-1), expected)


def test_a_context_step_draws_each_row_from_its_own_slot_zero(
    model: Model,
) -> None:
    """A context step whose rows peak at distinct tokens draws each row
    from its own slot-0 target, not from the first B rows of the logits."""
    logits = _distinct_row_peak_logits()
    # The same-seed draw from each row's own slot-0 target; built first
    # so precompile-mefs records it.
    expected = _full_target_draw(logits)
    outputs = _execute(
        model,
        drafts=np.zeros((B, K), dtype=np.int64),
        dists=np.zeros((B, K, VOCAB), dtype=np.float32),
        return_n_logits=1,
        scores=np.zeros((B, K), dtype=np.float32),
        coeffs=np.zeros(K + 1, dtype=np.float32),
        logits=logits,
    )
    num_accepted, next_tokens, _drafts, _scores, verify_lengths, _ = outputs
    np.testing.assert_array_equal(verify_lengths, [0, 0])
    np.testing.assert_array_equal(num_accepted, [0, 0])
    np.testing.assert_array_equal(next_tokens.reshape(-1), expected)


def _distinct_row_peak_logits() -> np.ndarray:
    """Slot-0 logits whose rows peak at distinct tokens, row 0 at the
    placeholder id, so a wrong row slice changes the draw."""
    logits = np.zeros((B * (K + 1), VOCAB), dtype=np.float32)
    non_slot_zero = [r for r in range(B * (K + 1)) if r % (K + 1) != 0]
    logits[non_slot_zero, B] = PEAK_LOGIT
    for row in range(B):
        logits[row * (K + 1), row] = PEAK_LOGIT
    return logits


def _placeholder_peak_logits() -> np.ndarray:
    """Slot-0 logits with real mass on the placeholder id (token 0), so
    removing it changes the draw."""
    logits = np.zeros((B * (K + 1), VOCAB), dtype=np.float32)
    logits[:, 0] = PEAK_LOGIT
    return logits


def _full_target_draw(logits: np.ndarray) -> np.ndarray:
    """The next token each row draws from the target's own distribution
    at slot 0, replayed with the same seed the step binds."""
    gpu = Accelerator()
    with Graph(
        "full_target_draw",
        input_types=[
            TensorType(DType.float32, [B, VOCAB], _GPU),
            TensorType(DType.uint64, [B], _GPU),
        ],
    ) as graph:
        logits_in, seed = (v.tensor for v in graph.inputs)
        probs = topk_topp_masked_probs(
            logits_in,
            top_k=ops.constant(
                np.full(B, VOCAB, dtype=np.int64), DType.int64, _GPU
            ),
            temperature=ops.constant(
                np.ones(B, dtype=np.float32), DType.float32, _GPU
            ),
            top_p=ops.constant(
                np.ones(B, dtype=np.float32), DType.float32, _GPU
            ),
        )
        graph.output(
            gumbel_argmax_from_probs(
                probs,
                seed=seed + _SEED_DOMAIN_RECOVERY,
            )
        )
    session = InferenceSession(devices=[CPU(), Accelerator()])
    full = session.load(graph)
    buffers = [
        Buffer.from_numpy(logits[:: K + 1]).to(gpu),
        Buffer.from_numpy(np.arange(B, dtype=np.uint64)).to(gpu),
    ]
    return outputs_to_numpy(full.execute(*buffers))[0].reshape(-1)


def test_a_context_step_without_drafts_is_refused(model: Model) -> None:
    """``[batch, 0]`` drafts, as a plain export's context step binds them,
    fail the per-row graph's rebind to K."""
    with pytest.raises(Exception, match="rebind shape check failed"):
        _execute(
            model,
            drafts=np.zeros((B, 0), dtype=np.int64),
            dists=np.zeros((B, 0, VOCAB), dtype=np.float32),
            return_n_logits=1,
            scores=np.zeros((B, K), dtype=np.float32),
            coeffs=np.zeros(K + 1, dtype=np.float32),
            logits=_slot_logits(B),
        )
