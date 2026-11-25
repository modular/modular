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
"""Shared block driver graph helpers."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    ops,
)
from max.pipelines.speculative import block_driver
from max.pipelines.speculative.block_driver import (
    Accepted,
    BlockBatch,
    BlockDriver,
    DraftSampler,
)
from max.pipelines.speculative.ragged_token_merger import RaggedTokenMerger
from max.pipelines.speculative.spec_input_types import SpecDecodeInputTypeSpec
from max.pipelines.speculative.spec_target import Verified
from spec_step_stubs import outputs_to_numpy, stub_kv_leaf

VOCAB = 64
BLOCK = 4


def stub_kv_blocks(device: DeviceRef) -> BufferValue:
    """Allocates a KV buffer for graph stubs that ignore its contents."""
    return ops.buffer_create(BufferType(DType.int64, [1], device))


def logits_with_argmax(argmax: list[int]) -> npt.NDArray[np.float32]:
    """Returns one logits row peaked at each supplied token."""
    logits = np.zeros((len(argmax), VOCAB), dtype=np.float32)
    logits[np.arange(len(argmax)), argmax] = 10.0
    return logits


PER_ROW_K = 4
PER_ROW_W = 2
PER_ROW_B = 3


class _PerRowProposer:
    """Exposes the merged tokens and commit metadata from a real driver step."""

    block_size = BLOCK
    mask_token_id = VOCAB - 1
    samples_from_anchor = False
    supports_zero_draft_rows = False

    accepted: Accepted
    host_offsets: TensorValue | None

    def materialize(self, *_args: object) -> None:
        return None

    def embed_block(self, *_args: object) -> list[TensorValue]:
        return []

    def forward_block(self, *_args: object) -> TensorValue:
        return ops.constant(
            np.zeros((1, BLOCK), dtype=np.float32),
            DType.float32,
            DeviceRef.CPU(),
        )

    def head(
        self,
        batch: BlockBatch,
        hidden: TensorValue,
        accepted: Accepted,
        sampler: DraftSampler,
    ) -> TensorValue:
        self.accepted = accepted
        self.host_offsets = batch.host_merged_offsets
        return batch.merged_tokens


def _per_row_driver(
    proposer: object,
    *,
    per_row: bool,
    stochastic: bool = False,
) -> BlockDriver[None, None]:
    """Builds a block driver with real merge, selection and acceptance.

    The verify is a stub that returns the batch's own logits input, so the
    accept judges the drafts the test binds; the proposer stubs the block
    phases after it.
    """
    cpu = DeviceRef.CPU()
    driver: BlockDriver[None, None] = BlockDriver.__new__(BlockDriver)
    vars(driver).update(
        _proposer=proposer,
        _verify=lambda batch: Verified(
            logits=batch.extra["logits"],
            hidden=None,
        ),
        _bitmask_side_stream=False,
        _skips_draft_rows=False,
        _sampled=False,
        _replica_probs=False,
        _ctx_at_draft_cache_length=False,
        block_size=PER_ROW_K + 1,
        merger=RaggedTokenMerger(cpu),
        num_speculative_tokens=PER_ROW_K,
        _vocab_size=None,
        data_parallel_degree=1,
        devices=[cpu],
        _input_spec=SpecDecodeInputTypeSpec(
            devices=[cpu],
            distributed=False,
            include_per_row_verify=per_row,
            num_draft_positions=PER_ROW_K if per_row else None,
        ),
        acceptance_sampler=block_driver.AcceptanceSampler(
            num_draft_steps=PER_ROW_K,
            use_stochastic=stochastic,
        ),
    )
    return driver


def _per_row_graph_inputs() -> list[TensorType]:
    cpu = DeviceRef.CPU()
    return [
        TensorType(DType.int64, ["total_seq_len"], cpu),
        TensorType(DType.uint32, ["input_row_offsets_len"], cpu),
        TensorType(DType.int64, ["batch_size", "num_steps"], cpu),
        TensorType(DType.float32, ["batch_size", PER_ROW_K], cpu),
        TensorType(DType.float32, [PER_ROW_K + 1], cpu),
        TensorType(DType.uint8, ["merged_seq_len"], cpu),
        TensorType(DType.float32, ["logits_rows", VOCAB], cpu),
    ]


def build_block_step(
    session: InferenceSession,
    *,
    per_row: bool,
    host_mirror: bool = False,
    stochastic: bool = False,
    splits: bool = False,
) -> Model:
    """Compiles a block step that returns the stub's merged tokens.

    Args:
        session: The CPU inference session.
        per_row: Whether to select a separate verify length per row.
        host_mirror: Whether to return host offsets.
        stochastic: Whether to use sampled acceptance.
        splits: Whether to bind the per-replica row boundaries the
            selector budgets by.

    Returns:
        The compiled block step.
    """
    cpu = DeviceRef.CPU()
    input_types = _per_row_graph_inputs()
    if host_mirror:
        input_types.append(
            TensorType(DType.uint32, ["input_row_offsets_len"], cpu)
        )
    if splits:
        input_types.append(TensorType(DType.int64, [3], cpu))
    with Graph("per_row_step", input_types=input_types) as graph:
        (
            _tokens_in,
            _offsets_in,
            drafts_in,
            scores_in,
            coeffs_in,
            _mask_in,
            logits_in,
            *extra_inputs,
        ) = (v.tensor for v in graph.inputs)
        host_offsets = extra_inputs[0] if host_mirror else None
        splits_in = extra_inputs[-1] if splits else None
        proposer = _PerRowProposer()
        driver = _per_row_driver(
            proposer, per_row=per_row, stochastic=stochastic
        )
        leaf = stub_kv_leaf(
            ops.constant(0, DType.int64, cpu).broadcast_to(["batch_size"]),
            stub_kv_blocks(cpu),
            cpu,
        )
        # The sampled path needs the production shapes: per-row tensors
        # for the per-request knobs, scalars for the batch-wide bound.
        if stochastic:
            temperature = ops.constant(
                1.0, DType.float32, device=cpu
            ).broadcast_to(["batch_size"])
            top_k = ops.constant(VOCAB, DType.int64, device=cpu).broadcast_to(
                ["batch_size"]
            )
            top_p = ops.constant(0.95, DType.float32, device=cpu).broadcast_to(
                ["batch_size"]
            )
            max_k = ops.constant(VOCAB, DType.int64, device=cpu)
            min_top_p = ops.constant(0.95, DType.float32, device=cpu)
        else:
            temperature = ops.constant(0.0, DType.float32, device=cpu)
            top_k = ops.constant(1, DType.int64, device=cpu)
            top_p = ops.constant(1.0, DType.float32, device=cpu)
            max_k = ops.constant(1, DType.int64, device=cpu)
            min_top_p = ops.constant(1.0, DType.float32, device=cpu)
        outputs = driver._run(
            _tokens_in,
            _offsets_in,
            drafts_in,
            kv_collections=[leaf],
            draft_kv_collections=[leaf],
            passthrough_kv={},
            return_n_logits=ops.constant(
                PER_ROW_K + 1, DType.int64, device=cpu
            ),
            seed=ops.constant(0, DType.uint64, device=cpu),
            temperature=temperature,
            top_k=top_k,
            max_k=max_k,
            top_p=top_p,
            min_top_p=min_top_p,
            draft_scores=scores_in,
            draft_score_coeffs=coeffs_in,
            host_input_row_offsets=host_offsets,
            data_parallel_splits=splits_in,
            extra={"logits": logits_in},
        )
        graph.output(
            *outputs,
            proposer.accepted.commit_lengths,
            proposer.accepted.is_prefill,
            *(
                (proposer.host_offsets,)
                if proposer.host_offsets is not None
                else ()
            ),
        )
    return session.load(graph)


def run_per_row(
    model: Model,
    prompt_lens: list[int],
    drafts: npt.NDArray[np.int64],
    scores: npt.NDArray[np.float32],
    coeffs: npt.NDArray[np.float32],
    target_argmax: npt.NDArray[np.int64],
    merged_len: int,
    *,
    host_offsets: npt.NDArray[np.uint32] | None = None,
    splits: npt.NDArray[np.int64] | None = None,
) -> tuple[np.ndarray, ...]:
    """Runs the step with prompt tokens and synthetic verification logits.

    Args:
        model: The compiled block step.
        prompt_lens: The prompt length of each row.
        drafts: The draft tokens by row.
        scores: The proposal scores by row and position.
        coeffs: The survival coefficients.
        target_argmax: The peak token per row or per logit slot.
        merged_len: The packed token extent.
        host_offsets: The host row boundaries for a graph with a host mirror.
        splits: The per-replica row boundaries for a graph with splits.

    Returns:
        Flattened accepted counts, next tokens and merged tokens, followed
        by verify lengths when enabled, commit lengths, prefill flags and
        optional host offsets.
    """
    tokens = np.arange(sum(prompt_lens), dtype=np.int64) + 100
    offsets = np.cumsum([0, *prompt_lens]).astype(np.uint32)
    # A per-row argmax repeats to every logit position the run's draft
    # width reads, so the narrow and per-row runs judge the same drafts.
    positions = len(prompt_lens) * (drafts.shape[1] + 1)
    argmax = (
        np.repeat(target_argmax, drafts.shape[1] + 1)
        if target_argmax.size == len(prompt_lens)
        else target_argmax[:positions]
    )
    logits = logits_with_argmax(list(argmax))
    buffers = model.execute(
        Buffer.from_numpy(tokens),
        Buffer.from_numpy(offsets),
        Buffer.from_numpy(drafts),
        Buffer.from_numpy(scores),
        Buffer.from_numpy(coeffs),
        Buffer.from_numpy(np.zeros(merged_len, dtype=np.uint8)),
        Buffer.from_numpy(logits),
        *(
            [Buffer.from_numpy(host_offsets)]
            if host_offsets is not None
            else []
        ),
        *([Buffer.from_numpy(splits)] if splits is not None else []),
    )
    return tuple(out.reshape(-1) for out in outputs_to_numpy(buffers))
