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
"""The block spec-decode driver's phases that every block draft shares."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from block_step_helpers import (
    BLOCK,
    PER_ROW_B,
    PER_ROW_K,
    PER_ROW_W,
    VOCAB,
    build_block_step,
    logits_with_argmax,
    run_per_row,
    stub_kv_blocks,
)
from cpu_step_session import cpu_session as cpu_session
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import (
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    ops,
)
from max.nn.kv_cache import PagedCacheValues
from max.nn.legacy.identity import Identity
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    extra_output_count,
)
from max.pipelines.speculative import block_driver
from max.pipelines.speculative.block_driver import (
    Accepted,
    BlockBatch,
    BlockDriver,
    DraftSampler,
    _block_kv_dispatch_metadata,
    block_dispatch_metadata,
    block_mla_dispatch_metadata,
)
from max.pipelines.speculative.config import (
    MAGIC_DRAFT_TOKEN_ID,
    SpeculativeConfig,
)
from max.pipelines.speculative.ragged_token_merger import RaggedTokenMerger
from max.pipelines.speculative.spec_input_types import SpecDecodeInputTypeSpec
from max.pipelines.speculative.spec_target import Verified
from max.pipelines.speculative.verify_budget import SCORE_LIMIT
from spec_step_stubs import stub_kv_leaf

NUM_DRAFTS = BLOCK - 1


def test_block_dispatch_metadata_is_rebuilt_at_block_width() -> None:
    """The rebuilt 4-int buffer keeps the batch size and cache bound, sets
    ``q_max_seq_len`` to the block width, and zeroes ``num_partitions`` so
    the kernel recomputes the split-K count for the draft's own heads.

    The leaf's own buffer carries the verify width, which on a prefill batch
    is the prompt length; an oversized query bound drove the draft's first
    attention layer to NaN.
    """
    with Graph(
        "block_dispatch_metadata",
        input_types=(TensorType(DType.int64, [4], DeviceRef.CPU()),),
    ) as graph:
        (meta,) = graph.inputs
        graph.output(block_dispatch_metadata(meta.tensor, 8))

    compiled = InferenceSession(devices=[CPU()]).load(graph)
    (out,) = compiled.execute(
        Buffer.from_numpy(np.array([3, 129, 5, 4321], dtype=np.int64))
    )
    assert isinstance(out, Buffer)
    np.testing.assert_array_equal(out.to_numpy(), [3, 8, 0, 4321])


def test_block_mla_dispatch_metadata_zeroes_the_decode_view() -> None:
    """A 6-int MLA record keeps its batch size and split-K count, takes the
    block width, and zeroes the decode view, since a draft block never
    splits; a 3-int record keeps its width.
    """
    for width, values, expected in (
        (3, [3, 129, 5], [3, 8, 5]),
        (6, [3, 129, 5, 2, 4, 7], [3, 8, 5, 0, 0, 0]),
    ):
        with Graph(
            f"block_mla_dispatch_metadata_{width}",
            input_types=(TensorType(DType.int64, [width], DeviceRef.CPU()),),
        ) as graph:
            (meta,) = graph.inputs
            graph.output(block_mla_dispatch_metadata(meta.tensor, 8))

        compiled = InferenceSession(devices=[CPU()]).load(graph)
        (out,) = compiled.execute(
            Buffer.from_numpy(np.array(values, dtype=np.int64))
        )
        assert isinstance(out, Buffer)
        np.testing.assert_array_equal(out.to_numpy(), expected)


def test_block_kv_dispatch_metadata_routes_records_by_width() -> None:
    """The router picks the builder from the record's width."""
    for width in (3, 6, 4):
        with Graph(
            f"block_kv_dispatch_metadata_{width}",
            input_types=(TensorType(DType.int64, [width], DeviceRef.CPU()),),
        ) as graph:
            (meta,) = graph.inputs
            out = _block_kv_dispatch_metadata(meta.tensor, 8)
            assert out.shape == [width]
            assert out.dtype == DType.int64
            graph.output(out)


class _AcceptOnlyProposer:
    """A block draft the accept phase never calls into."""

    block_size = BLOCK
    mask_token_id = VOCAB - 1
    samples_from_anchor = False
    supports_zero_draft_rows = False

    def materialize(
        self,
        batch: BlockBatch,
        target_hidden: None,
        ctx_kv: list[PagedCacheValues],
    ) -> None:
        raise NotImplementedError

    def embed_block(
        self, batch: BlockBatch, block_ids: TensorValue
    ) -> list[TensorValue]:
        raise NotImplementedError

    def forward_block(
        self,
        batch: BlockBatch,
        embeds: list[TensorValue],
        offsets: list[TensorValue],
        block_kv: list[PagedCacheValues],
    ) -> None:
        raise NotImplementedError

    def head(
        self,
        batch: BlockBatch,
        block_hs: None,
        accepted: Accepted,
        sampler: DraftSampler,
    ) -> TensorValue:
        raise NotImplementedError


def _unused_verify(batch: BlockBatch) -> Verified[None]:
    raise NotImplementedError


@pytest.fixture(scope="module", params=[False, True], ids=["default", "amd"])
def accept(request: pytest.FixtureRequest) -> Model:
    """Compiles the driver's greedy accept phase on its own, once per branch.

    Returns ``(num_accepted, next_tokens, commit_lengths)`` for a batch of
    tokens, row offsets, drafts ``[batch, num_steps]`` and target logits
    ``[batch * (num_steps + 1), vocab]``. ``num_steps`` stays symbolic, as
    in the fused graphs, so prefill (zero drafts) and decode run the same
    compiled graph.

    The ``amd`` case builds the AMD gather that drops the prefill select,
    which CI's CPU workers never reach on their own, so both branches have
    to produce the same results here.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(block_driver, "_is_amd_gpu", lambda: request.param)
        return _compile_accept()


def _compile_accept() -> Model:
    cpu = DeviceRef.CPU()
    driver: BlockDriver[None, None] = BlockDriver(
        _unused_verify,
        _AcceptOnlyProposer(),
        target_model=Identity(),
        draft_model=Identity(),
        input_spec=SpecDecodeInputTypeSpec(devices=[cpu], distributed=False),
        speculative_config=SpeculativeConfig(
            speculative_method="dflash", num_speculative_tokens=NUM_DRAFTS
        ),
        use_greedy_acceptance=True,
    )
    input_types = [
        TensorType(DType.int64, ["total_seq_len"], cpu),
        TensorType(DType.uint32, ["input_row_offsets_len"], cpu),
        TensorType(DType.int64, ["batch_size", "num_steps"], cpu),
        TensorType(DType.float32, ["total_output_len", VOCAB], cpu),
    ]
    with Graph("block_accept", input_types=input_types) as graph:
        tokens, offsets, drafts, logits = (v.tensor for v in graph.inputs)
        batch = BlockBatch(
            tokens=tokens,
            input_row_offsets=offsets,
            draft_tokens=drafts,
            kv_collections=[],
            draft_kv_collections=[],
            passthrough_kv={},
            return_n_logits=ops.constant(
                NUM_DRAFTS + 1, DType.int64, device=cpu
            ),
            devices=[cpu],
            signal_buffers=[],
            ep_inputs=None,
            merged_tokens=tokens,
            merged_offsets=offsets,
            merged_offsets_per_dev=[offsets],
            host_merged_offsets=None,
            data_parallel_splits=None,
            data_parallel_degree=1,
            batch_context_lengths=[],
            vision_embeddings=[],
            vision_scatter_indices=[],
            extra={},
            num_accepted=None,
            block_size=BLOCK,
        )
        unused = ops.constant(0, DType.float32, device=cpu)
        accepted = driver._accept(
            batch,
            logits,
            seed=ops.constant(0, DType.uint64, device=cpu),
            temperature=unused,
            top_k=ops.constant(1, DType.int64, device=cpu),
            max_k=ops.constant(1, DType.int64, device=cpu),
            top_p=unused,
            min_top_p=unused,
            in_thinking_phase=None,
            token_bitmasks=None,
            draft_probs_full=None,
        )
        graph.output(
            accepted.num_accepted,
            accepted.next_tokens,
            accepted.commit_lengths,
        )
    return InferenceSession(devices=[CPU()]).load(graph)


def _run(
    accept: Model,
    prompt_lens: list[int],
    drafts: npt.NDArray[np.int64],
    target_argmax: list[int],
) -> list[np.ndarray]:
    offsets = np.concatenate([[0], np.cumsum(prompt_lens)]).astype(np.uint32)
    outputs = accept.execute(
        Buffer.from_numpy(np.zeros(int(offsets[-1]), dtype=np.int64)),
        Buffer.from_numpy(offsets),
        Buffer.from_numpy(drafts),
        Buffer.from_numpy(logits_with_argmax(target_argmax)),
    )
    results = []
    for out in outputs:
        assert isinstance(out, Buffer)
        results.append(out.to_numpy().reshape(-1))
    return results


# Lengths whose prefill committed token 0 in serving, two of them through a
# device assert in the draft that consumed it.
_PROMPT_LENS = [121, 200, 137, 282]


def test_prefill_commits_the_target_token_and_accepts_nothing(
    accept: Model,
) -> None:
    """A prefill row commits the target's own next token and reports zero
    accepted, with its whole prompt as the commit length.

    The draft axis is zero-wide at prefill, but the accepted count arriving
    at the committed-token gather could still read the compiled width, which
    indexed past the one-column token table and committed id 0 for every
    prompt. Mach advances each sequence by the reported count, so a nonzero
    count at prefill would also skip tokens.
    """
    target_argmax = [5, 17, 29, 61]
    num_accepted, next_tokens, commit_lengths = _run(
        accept,
        _PROMPT_LENS,
        np.zeros((len(_PROMPT_LENS), 0), dtype=np.int64),
        target_argmax,
    )
    np.testing.assert_array_equal(num_accepted, [0, 0, 0, 0])
    np.testing.assert_array_equal(next_tokens, target_argmax)
    np.testing.assert_array_equal(commit_lengths, _PROMPT_LENS)


def test_decode_counts_the_accepted_prefix(accept: Model) -> None:
    """A decode row commits the accepted drafts plus the target's token at
    the first rejection; an all-placeholder row accepts nothing.

    This keeps a fix for prefill honest: a count pinned to zero everywhere
    would pass the prefill test but fail here.
    """
    # Row 0 drafts only placeholders. Row 1 matches the target at its first
    # two slots and misses at the third.
    row0 = [3, 4, 6, 7]
    row1 = [9, 11, 13, 15]
    drafts = np.array(
        [[MAGIC_DRAFT_TOKEN_ID] * NUM_DRAFTS, [row1[0], row1[1], 2]],
        dtype=np.int64,
    )
    num_accepted, next_tokens, commit_lengths = _run(
        accept, [1, 1], drafts, row0 + row1
    )
    np.testing.assert_array_equal(num_accepted, [0, 2])
    np.testing.assert_array_equal(next_tokens, [row0[0], row1[2]])
    np.testing.assert_array_equal(commit_lengths, [1, 3])


@pytest.mark.parametrize("method", ["dspark", "dflash"])
def test_per_row_verify_needs_the_argmax_proposal(method: str) -> None:
    """A block draft's sampled arm draws its own proposals and keeps their
    distributions; no consumer reads scores there, so the config refuses
    the combination rather than letting the graph build crash on the
    tuple the declaring proposer's head returns."""
    with pytest.raises(ValueError, match="argmax"):
        SpeculativeConfig.model_validate(
            {
                "speculative_method": method,
                "num_speculative_tokens": NUM_DRAFTS,
                "draft_proposal": "sampled",
                "per_row_verify": True,
            }
        )
    argmax = SpeculativeConfig.model_validate(
        {
            "speculative_method": method,
            "num_speculative_tokens": NUM_DRAFTS,
            "per_row_verify": True,
        }
    )
    assert argmax.per_row_verify
    assert argmax.draft_proposal == "argmax"


def test_a_sequential_draft_scores_sampled_proposals() -> None:
    """A sequential drafter keeps each pass's logits, so it can score
    sampled proposals; only a block draft's sampled arm cannot."""
    config = SpeculativeConfig.model_validate(
        {
            "speculative_method": "mtp",
            "num_speculative_tokens": 5,
            "draft_proposal": "sampled",
            "per_row_verify": True,
        }
    )
    assert config.per_row_verify
    assert config.draft_proposal == "sampled"


class _ScoredProposer(_AcceptOnlyProposer):
    confidence_outputs = True


def test_block_driver_rechecks_per_row_acceptance() -> None:
    config = SpeculativeConfig(
        speculative_method="dspark",
        num_speculative_tokens=NUM_DRAFTS,
        synthetic_acceptance_rate=1.0,
    )
    config = config.model_copy(update={"per_row_verify": True})
    with pytest.raises(
        ValueError, match=r"per_row_verify.*synthetic_acceptance_rate"
    ):
        BlockDriver(
            _unused_verify,
            _ScoredProposer(),
            target_model=Identity(),
            draft_model=Identity(),
            input_spec=SpecDecodeInputTypeSpec(
                devices=[DeviceRef.CPU()], distributed=False
            ),
            speculative_config=config,
        )


def test_block_driver_rechecks_the_block_draft_argmax_rule() -> None:
    """A copied config skips the construction validators, so the driver's
    recheck must refuse a block draft whose sampled arm cannot feed the
    per-row verify, rather than building a graph that emits neither
    scores nor verify lengths."""
    config = SpeculativeConfig(
        speculative_method="dspark",
        num_speculative_tokens=NUM_DRAFTS,
        draft_proposal="sampled",
    )
    config = config.model_copy(update={"per_row_verify": True})
    with pytest.raises(ValueError, match="argmax"):
        BlockDriver(
            _unused_verify,
            _ScoredProposer(),
            target_model=Identity(),
            draft_model=Identity(),
            input_spec=SpecDecodeInputTypeSpec(
                devices=[DeviceRef.CPU()], distributed=False
            ),
            speculative_config=config,
            vocab_size=VOCAB,
        )


def test_the_block_driver_declares_K_when_it_takes_the_default() -> None:
    """A driver built without the draft count resolves the default before
    it builds the spec, so the per-row verify inputs carry the block's
    own width."""

    driver: BlockDriver[None, None] = BlockDriver(
        _unused_verify,
        _ScoredProposer(),
        target_model=Identity(),
        draft_model=Identity(),
        input_spec=SpecDecodeInputTypeSpec(
            devices=[DeviceRef.CPU()], distributed=False
        ),
        speculative_config=SpeculativeConfig(
            speculative_method="dflash",
            per_row_verify=True,
        ),
        use_greedy_acceptance=True,
    )
    assert driver.input_spec.num_draft_positions == NUM_DRAFTS
    assert driver.input_spec.include_per_row_verify


def test_per_row_verify_needs_a_proposer_that_declares_scores() -> None:
    """A per-row graph emits scores beside the drafts, so a proposer that
    declares no ``confidence_outputs`` would mislabel the outputs; the
    constructor refuses it before the graph builds."""
    with pytest.raises(ValueError, match="confidence_outputs"):
        BlockDriver(
            _unused_verify,
            _AcceptOnlyProposer(),
            target_model=Identity(),
            draft_model=Identity(),
            input_spec=SpecDecodeInputTypeSpec(
                devices=[DeviceRef.CPU()], distributed=False
            ),
            speculative_config=SpeculativeConfig(
                speculative_method="dflash",
                per_row_verify=True,
            ),
            use_greedy_acceptance=True,
        )


class _HeadStubProposer:
    """A block draft whose head returns what the test hands it.

    The driver reads the ``confidence_outputs`` declaration with
    ``getattr`` (the protocol's documented option), so the default variant
    leaves the attribute absent, exactly like every proposer that emits
    drafts alone.
    """

    block_size = BLOCK
    mask_token_id = VOCAB - 1
    samples_from_anchor = False
    supports_zero_draft_rows = False

    def __init__(self, head_return: object, *, declared: bool = False) -> None:
        self._head_return = head_return
        if declared:
            self.confidence_outputs = True

    def materialize(self, *_args: object) -> None:
        return None

    def embed_block(self, *_args: object) -> list[TensorValue]:
        return []

    def forward_block(self, *_args: object) -> TensorValue:
        # The head stub ignores its inputs, so any graph value serves as
        # the block's hidden states.
        return ops.constant(
            np.zeros((1, BLOCK), dtype=np.float32),
            DType.float32,
            DeviceRef.CPU(),
        )

    def head(self, *_args: object) -> object:
        return self._head_return


def _scored_driver(
    proposer: _HeadStubProposer, lengths: TensorValue
) -> BlockDriver[None, None]:
    """A driver whose phases before the greedy arm run for real.

    The accept's values must carry the batch's symbolic row count: a
    ``[1]``-row constant would break a concat the real graph never
    sees (its accept returns one row per request).
    """
    cpu = DeviceRef.CPU()
    zeros = lengths * 0
    driver: BlockDriver[None, None] = BlockDriver.__new__(BlockDriver)
    vars(driver).update(
        _proposer=proposer,
        _verify=lambda batch: Verified(
            logits=zeros.cast(DType.float32),
            hidden=None,
        ),
        _accept=lambda batch, logits, **_: Accepted(
            num_accepted=ops.cast(zeros, DType.int32),
            next_tokens=zeros,
            commit_lengths=ops.cast(zeros, DType.uint32),
            is_prefill=ops.cast(zeros, DType.bool),
        ),
        _bitmask_side_stream=False,
        _skips_draft_rows=False,
        _sampled=False,
        _replica_probs=False,
        _ctx_at_draft_cache_length=False,
        block_size=BLOCK,
        merger=RaggedTokenMerger(cpu),
        num_speculative_tokens=NUM_DRAFTS,
        _vocab_size=None,
        data_parallel_degree=1,
        devices=[cpu],
        _input_spec=SpecDecodeInputTypeSpec(devices=[cpu], distributed=False),
    )
    return driver


def _run_head(
    head_return: object, *, declared: bool
) -> tuple[TensorValue, ...]:
    """Runs the driver's greedy arm over a stubbed step and returns its
    outputs, with the head's values built inside the same graph."""
    cpu = DeviceRef.CPU()
    with Graph(
        "block_head",
        input_types=[TensorType(DType.int64, ["batch_size"], cpu)],
    ) as graph:
        lengths = graph.inputs[0].tensor
        leaf = stub_kv_leaf(lengths, stub_kv_blocks(cpu), cpu)
        tokens = ops.constant(np.zeros((1,), dtype=np.int64), DType.int64, cpu)
        offsets = ops.constant(
            np.array([0, 1], dtype=np.uint32), DType.uint32, cpu
        )
        draft_tokens = ops.constant(
            np.zeros((1, 1), dtype=np.int64), DType.int64, cpu
        )
        if isinstance(head_return, tuple):
            values = tuple(
                ops.constant(
                    np.zeros((1, NUM_DRAFTS), dtype=np.float32),
                    DType.float32,
                    cpu,
                )
                for _ in head_return
            )
            proposer = _HeadStubProposer(values, declared=declared)
        else:
            drafts = ops.constant(
                np.zeros((1, NUM_DRAFTS), dtype=np.int64), DType.int64, cpu
            )
            proposer = _HeadStubProposer(drafts, declared=declared)
        one = ops.constant(1, DType.int64, cpu)
        one_float = ops.constant(1.0, DType.float32, cpu)
        return _scored_driver(proposer, lengths)._run(
            tokens,
            offsets,
            draft_tokens,
            kv_collections=[leaf],
            draft_kv_collections=[leaf],
            passthrough_kv={},
            return_n_logits=one,
            seed=ops.constant(0, DType.uint64, cpu),
            temperature=ops.constant(0.0, DType.float32, cpu),
            top_k=one,
            max_k=one,
            top_p=one_float,
            min_top_p=one_float,
        )


def test_the_greedy_arm_returns_three_outputs_when_no_proposer_declares_confidence_outputs() -> (
    None
):
    """A proposer that declares nothing returns three outputs: the head's
    drafts are the third output, and nothing else is appended."""
    outputs = _run_head("drafts", declared=False)
    assert len(outputs) == 3
    assert outputs[2].type == TensorType(
        DType.int64, [1, NUM_DRAFTS], device=DeviceRef.CPU()
    )


def test_the_greedy_arm_unpacks_the_declared_score_outputs() -> None:
    """A declaring proposer's head returns its drafts beside the scores;
    the driver unpacks each beside the drafts, so the graph's outputs
    line up with the score contract."""
    scored = _run_head(("drafts", "scores"), declared=True)
    assert len(scored) == 4
    assert scored[2].type == TensorType(
        DType.float32, [1, NUM_DRAFTS], device=DeviceRef.CPU()
    )
    assert scored[3].type == TensorType(
        DType.float32, [1, NUM_DRAFTS], device=DeviceRef.CPU()
    )


def test_pad_skipped_rows_fills_float_scores_with_zero(
    session: InferenceSession,
) -> None:
    """A row with no real proposal carries score 0, not the magic draft id
    that pads proposals: the fill is the score contract's pad value, and
    a float output must never see the int fill."""
    cpu = DeviceRef.CPU()
    input_types = [
        TensorType(DType.float32, ["draft_seqs", NUM_DRAFTS], cpu),
        TensorType(DType.int32, ["num_draft_slots"], cpu),
        TensorType(DType.float32, ["batch_size"], cpu),
    ]
    with Graph("pad_scores", input_types=input_types) as graph:
        drafted, slot_ids, _batch_bind = (v.tensor for v in graph.inputs)
        batch = BlockBatch(
            tokens=ops.constant(
                np.zeros((1,), dtype=np.int64), DType.int64, cpu
            ),
            input_row_offsets=ops.constant(
                np.array([0, 1], dtype=np.uint32), DType.uint32, cpu
            ),
            draft_tokens=ops.constant(
                np.zeros((1, 1), dtype=np.int64), DType.int64, cpu
            ),
            kv_collections=[],
            draft_kv_collections=[],
            passthrough_kv={},
            return_n_logits=ops.constant(1, DType.int64, cpu),
            devices=[cpu],
            signal_buffers=[],
            ep_inputs=None,
            merged_tokens=ops.constant(
                np.zeros((1,), dtype=np.int64), DType.int64, cpu
            ),
            merged_offsets=ops.constant(
                np.array([0, 1], dtype=np.uint32), DType.uint32, cpu
            ),
            merged_offsets_per_dev=[],
            host_merged_offsets=None,
            data_parallel_splits=None,
            data_parallel_degree=1,
            batch_context_lengths=[],
            vision_embeddings=[],
            vision_scatter_indices=[],
            extra={},
            num_accepted=None,
            draft_slot_ids=slot_ids,
            draft_block_offsets=None,
            block_size=BLOCK,
        )
        driver: BlockDriver[None, None] = BlockDriver.__new__(BlockDriver)
        graph.output(driver._pad_skipped_rows(batch, drafted, fill=0))

    compiled = session.load(graph)
    (out,) = compiled.execute(
        Buffer.from_numpy(np.array([[7.0, 8.0, 9.0]], dtype=np.float32)),
        Buffer.from_numpy(np.arange(BLOCK, dtype=np.int32)),
        Buffer.from_numpy(np.zeros(2, dtype=np.float32)),
    )
    assert isinstance(out, Buffer)
    np.testing.assert_array_equal(
        out.to_numpy(), [[7.0, 8.0, 9.0], [0.0, 0.0, 0.0]]
    )


@pytest.fixture(scope="module")
def per_row_step(session: InferenceSession) -> Model:
    return build_block_step(session, per_row=True)


def _reference_lengths(
    scores: npt.NDArray[np.float32],
    coeffs: npt.NDArray[np.float32],
    width: int,
) -> npt.NDArray[np.int32]:
    """The selection's reference: survival, then the top-T kept mask."""
    wide = scores.astype(np.float64)
    wide = np.clip(wide, -SCORE_LIMIT, SCORE_LIMIT)
    logits = wide * coeffs[0] + coeffs[1:].astype(np.float64)
    survival = np.cumprod(1.0 / (1.0 + np.exp(-logits)), axis=1)
    batch, positions = survival.shape
    flat = survival.reshape(-1)
    order = np.lexsort(
        (
            np.arange(batch * positions) // positions,
            np.arange(batch * positions) % positions,
            -flat,
        )
    )
    kept = np.zeros(batch * positions, dtype=bool)
    kept[order[: min(batch * width, batch * positions)]] = True
    return kept.reshape(batch, positions).sum(axis=1).astype(np.int32)


def test_per_row_selection_packs_each_rows_prefix(per_row_step: Model) -> None:
    """The output verify lengths match the reference selection and sum to
    the fixed total; the merged tokens hold exactly each row's prefix."""
    rng = np.random.default_rng(11)
    prompt_lens = [1, 1, 1]
    drafts = rng.integers(200, 300, size=(PER_ROW_B, PER_ROW_K)).astype(
        np.int64
    )
    # Row 0's scores outrank every other entry, so it keeps the first
    # slots of the budget.
    scores = np.full((PER_ROW_B, PER_ROW_K), -5.0, dtype=np.float32)
    scores[0] = 5.0
    coeffs = np.array([1.0] + [0.0] * PER_ROW_K, dtype=np.float32)
    merged_len = sum(prompt_lens) + PER_ROW_B * PER_ROW_W
    _accepted, _next, merged, lengths, _, _ = run_per_row(
        per_row_step,
        prompt_lens,
        drafts,
        scores,
        coeffs,
        np.zeros(PER_ROW_B, dtype=np.int64),
        merged_len,
    )
    expected = _reference_lengths(scores, coeffs, PER_ROW_W)
    np.testing.assert_array_equal(lengths, expected)
    assert lengths.sum() == PER_ROW_B * PER_ROW_W
    # The merged window holds each prompt followed by exactly its row's
    # kept prefix.
    tokens = np.arange(sum(prompt_lens), dtype=np.int64) + 100
    want = np.concatenate(
        [
            np.concatenate([tokens[i : i + 1], drafts[i, : expected[i]]])
            for i in range(PER_ROW_B)
        ]
    )
    np.testing.assert_array_equal(merged, want)


def test_per_row_selection_budgets_each_replica_by_its_splits(
    session: InferenceSession,
) -> None:
    """Checks that each replica spends its own verify budget."""
    model = build_block_step(session, per_row=True, splits=True)
    prompt_lens = [1, 1, 1]
    drafts = np.tile(np.array([[7, 11, 13, 15]], dtype=np.int64), (3, 1))
    # A whole-batch budget would give row 0 a slot owned by replica 1.
    scores = np.array(
        [
            [5.0, 5.0, 5.0, 5.0],
            [-5.0, -5.0, -5.0, -5.0],
            [0.0, -5.0, -5.0, -5.0],
        ],
        dtype=np.float32,
    )
    coeffs = np.array([1.0] + [0.0] * PER_ROW_K, dtype=np.float32)
    merged_len = sum(prompt_lens) + 3 * 1
    _accepted, _next, merged, lengths, _, _ = run_per_row(
        model,
        prompt_lens,
        drafts,
        scores,
        coeffs,
        np.zeros(3, dtype=np.int64),
        merged_len,
        splits=np.array([0, 2, 3], dtype=np.int64),
    )
    np.testing.assert_array_equal(lengths, [2, 0, 1])
    tokens = np.arange(sum(prompt_lens), dtype=np.int64) + 100
    want = np.concatenate(
        [
            np.concatenate([tokens[i : i + 1], drafts[i, :length]])
            for i, length in enumerate((2, 0, 1))
        ]
    )
    np.testing.assert_array_equal(merged, want)


def test_per_row_accept_stops_at_the_verified_prefix(
    per_row_step: Model,
) -> None:
    """A row whose drafts all match accepts exactly its k_r, not K; a k_r = 0
    row accepts nothing and still commits its bonus token."""
    prompt_lens = [1, 1]
    drafts = np.tile(np.array([[7, 11, 13, 15]], dtype=np.int64), (2, 1))
    # The budget is two slots (w = 1 over two rows): row 0's top two
    # survivals take both, row 1 keeps none.
    scores = np.array(
        [[5.0, 5.0, -5.0, -5.0], [-5.0, -5.0, -5.0, -5.0]],
        dtype=np.float32,
    )
    coeffs = np.array([1.0] + [0.0] * PER_ROW_K, dtype=np.float32)
    # Every draft position's target argmax matches the draft, so the
    # verdict alone would accept all four.
    argmax = np.tile(drafts[0], PER_ROW_K + 1).astype(np.int64)
    merged_len = sum(prompt_lens) + 2 * 1
    num_accepted, next_tokens, _drafts, lengths, commit_lengths, _ = (
        run_per_row(
            per_row_step,
            prompt_lens,
            drafts,
            scores,
            coeffs,
            argmax,
            merged_len,
        )
    )
    np.testing.assert_array_equal(lengths, [2, 0])
    np.testing.assert_array_equal(num_accepted, [2, 0])
    np.testing.assert_array_equal(commit_lengths, [3, 1])
    # Row 0 commits the recovery at its first unverified position; row 1
    # commits the target's own next token at position 0.
    np.testing.assert_array_equal(next_tokens, [13, 11])


def test_a_zero_width_per_row_step_verifies_nothing(
    per_row_step: Model,
) -> None:
    """A w = 0 step gives every row k_r = 0: the merged window is the
    prompt tokens alone and every row commits its bonus token."""
    prompt_lens = [1, 2]
    drafts = np.tile(np.array([[7, 11, 13, 15]], dtype=np.int64), (2, 1))
    scores = np.zeros((2, PER_ROW_K), dtype=np.float32)
    coeffs = np.array([1.0] + [0.0] * PER_ROW_K, dtype=np.float32)
    argmax = np.array([3, 5], dtype=np.int64)
    num_accepted, next_tokens, merged, lengths, commit_lengths, is_prefill = (
        run_per_row(
            per_row_step,
            prompt_lens,
            drafts,
            scores,
            coeffs,
            argmax,
            sum(prompt_lens),
        )
    )
    np.testing.assert_array_equal(lengths, [0, 0])
    np.testing.assert_array_equal(num_accepted, [0, 0])
    np.testing.assert_array_equal(next_tokens, [3, 5])
    np.testing.assert_array_equal(is_prefill, [True, True])
    np.testing.assert_array_equal(commit_lengths, prompt_lens)
    np.testing.assert_array_equal(merged, [100, 101, 102])


def test_a_row_with_no_proposal_takes_no_verify_slot(
    per_row_step: Model,
) -> None:
    """An all-MAGIC row's padded scores must not take real rows' slots.

    The engine pads a row that carried no proposal with zero scores, whose
    survival would outrank real rows' low-survival drafts and steal their
    verify slots; the selection must rank it last instead.
    """
    prompt_lens = [1, 1, 1]
    drafts = np.tile(np.array([[9, 11, 13, 15]], dtype=np.int64), (3, 1))
    drafts[0] = MAGIC_DRAFT_TOKEN_ID
    # Row 0's zero padding is what the engine feeds a proposal-less row;
    # the real rows' drafts score low enough that the padding would win.
    scores = np.full((3, PER_ROW_K), -5.0, dtype=np.float32)
    scores[0] = 0.0
    coeffs = np.array([1.0] + [0.0] * PER_ROW_K, dtype=np.float32)
    num_accepted, _next, _commit, lengths, _, _ = run_per_row(
        per_row_step,
        prompt_lens,
        drafts,
        scores,
        coeffs,
        np.zeros(3, dtype=np.int64),
        sum(prompt_lens) + 3 * 2,
    )
    # The MAGIC row verifies nothing; the real rows split the whole budget.
    np.testing.assert_array_equal(lengths, [0, 3, 3])
    np.testing.assert_array_equal(num_accepted, [0, 0, 0])


def test_an_all_magic_row_never_ties_a_real_row(per_row_step: Model) -> None:
    """An all-MAGIC row must not take a real row's slot even when the
    real row's survival underflows to exactly zero.

    The floor sits strictly below every real survival, so the earlier
    MAGIC row loses the (position, row) tie break to the later real row.
    """
    prompt_lens = [1, 1]
    drafts = np.tile(np.array([[9, 11, 13, 15]], dtype=np.int64), (2, 1))
    drafts[0] = MAGIC_DRAFT_TOKEN_ID
    # Slope 1 with intercepts far below the sigmoid's float32 range and
    # scores at the lower bound: every real survival underflows to 0.0.
    scores = np.full((2, PER_ROW_K), -SCORE_LIMIT, dtype=np.float32)
    coeffs = np.array([1.0] + [-200.0] * PER_ROW_K, dtype=np.float32)
    num_accepted, _next, _commit, lengths, _, _ = run_per_row(
        per_row_step,
        prompt_lens,
        drafts,
        scores,
        coeffs,
        np.zeros(2, dtype=np.int64),
        sum(prompt_lens) + 2 * 1,
    )
    # The budget is two slots; the real row's zero survival still beats
    # the MAGIC row's floor, so it takes both.
    np.testing.assert_array_equal(lengths, [0, 2])
    np.testing.assert_array_equal(num_accepted, [0, 0])


def test_the_extra_output_count_covers_every_flag() -> None:
    """The count is two under a per-row verify, which adds the scores and
    the lengths, and zero otherwise."""

    def config(speculative: SpeculativeConfig) -> PipelineConfig:
        return PipelineConfig.model_construct(speculative=speculative)

    plain = SpeculativeConfig(
        speculative_method="dspark",
        num_speculative_tokens=NUM_DRAFTS,
    )
    assert extra_output_count(config(plain)) == 0
    per_row = SpeculativeConfig(
        speculative_method="dspark",
        num_speculative_tokens=NUM_DRAFTS,
        per_row_verify=True,
    )
    assert extra_output_count(config(per_row)) == 2


def test_a_draft_score_source_needs_per_row_verify() -> None:
    """Scores reach the outputs only under a per-row verify, so a source
    without it is refused."""
    with pytest.raises(ValueError, match="draft_score_source needs"):
        SpeculativeConfig(
            speculative_method="dspark",
            num_speculative_tokens=NUM_DRAFTS,
            draft_score_source="head",
        )
