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
"""Shared sequential per row graph helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.engine import Model
from max.graph import (
    BufferType,
    DeviceRef,
    DimLike,
    Graph,
    TensorType,
    TensorValue,
    ops,
)
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
)

B = 2
K = 3
VOCAB = 128
HIDDEN = 4
PROMPT_LENS = [1, 2]
W = 2

CPU_DEVICE = DeviceRef.CPU()


class RecordingProposer:
    """Exposes prefill inputs and returns fixed logits."""

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
    # The prefill's logit slots per row: a per-row graph's static K + 1.
    prefill_slots: int = K + 1

    prefill_outputs: tuple[TensorValue, ...] = ()

    def prefill(
        self,
        batch: SequentialBatch,
        tokens: TensorValue,
        target_hidden: object,
    ) -> Proposed:
        self.prefill_outputs = (
            batch.merged_tokens,
            tokens,
            batch.return_n_logits,
        )
        # Logits over the batch's slots, peaked at the slot index and at
        # the slot's own height, so a wrong-slot score differs.
        slots = self.prefill_slots
        rows = B * slots
        logits = np.zeros((rows, VOCAB), dtype=np.float32)
        for r in range(rows):
            logits[r, r % slots] = 10.0 + 2.0 * (r % slots)
        return Proposed(
            logits=ops.constant(logits, DType.float32, CPU_DEVICE),
            hidden=[
                ops.constant(
                    np.zeros((rows, HIDDEN), dtype=np.float32),
                    DType.float32,
                    CPU_DEVICE,
                )
            ],
        )

    def step(
        self,
        batch: SequentialBatch,
        draft_input: DraftStepInput,
        index: int,
    ) -> Proposed:
        rows = B
        logits = np.zeros((rows, VOCAB), dtype=np.float32)
        for r in range(rows):
            logits[r, (r + index) % VOCAB] = 10.0 + 2.0 * index
        return Proposed(
            logits=ops.constant(logits, DType.float32, CPU_DEVICE),
            hidden=[
                ops.constant(
                    np.zeros((rows, HIDDEN), dtype=np.float32),
                    DType.float32,
                    CPU_DEVICE,
                )
            ],
        )


def _graph_inputs(
    per_row: bool, host_mirrors: bool = False
) -> list[TensorType | BufferType]:
    inputs: list[TensorType | BufferType] = [
        TensorType(DType.int64, ["total_seq_len"], CPU_DEVICE),
        TensorType(DType.uint32, ["input_row_offsets_len"], CPU_DEVICE),
        TensorType(DType.int64, ["batch_size", "num_steps"], CPU_DEVICE),
        TensorType(DType.float32, ["logits_rows", VOCAB], CPU_DEVICE),
        TensorType(DType.int64, [1], CPU_DEVICE),
        # The KV blocks the stub leaf carries; never read.
        BufferType(DType.int64, [1], CPU_DEVICE),
    ]
    if per_row:
        inputs.extend(
            [
                TensorType(DType.float32, ["batch_size", K], CPU_DEVICE),
                TensorType(DType.float32, [K + 1], CPU_DEVICE),
            ]
        )
    if host_mirrors:
        inputs += [
            TensorType(DType.uint32, ["input_row_offsets_len"], CPU_DEVICE),
            TensorType(DType.int64, [2], CPU_DEVICE),
        ]
    return inputs


def sequential_step_graph(
    proposer: RecordingProposer,
    *,
    per_row: bool,
    head_free: bool = False,
    observe_prefill: bool = False,
    host_mirrors: bool = False,
) -> Graph:
    """Builds a sequential driver graph around the recording proposer.

    Args:
        proposer: The draft stub.
        per_row: Whether to consume scores and survival coefficients.
        head_free: Whether to return per-pass top-token scores.
        observe_prefill: Whether to append merged tokens, shifted tokens
            and the prefill slot count.
        host_mirrors: Whether to build the step as a sharded target's,
            with the host offset mirror and splits, and append the
            verify's host merged offsets to the outputs.

    Returns:
        The graph with driver outputs, then any prefill observations, then
        the optional host offset mirror.
    """
    host_merged: list[TensorValue] = []

    def observe_host_offsets(batch: SequentialBatch) -> None:
        host_merged.append(batch.dist.host_merged_offsets)

    with Graph(
        "sequential_per_row",
        input_types=_graph_inputs(per_row, host_mirrors),
    ) as graph:
        (
            tokens_in,
            offsets_in,
            drafts_in,
            logits_in,
            return_n_logits_in,
        ) = (v.tensor for v in graph.inputs[:5])
        kv_blocks_in = graph.inputs[5].buffer
        scores_in = graph.inputs[6].tensor if per_row else None
        coeffs_in = graph.inputs[7].tensor if per_row else None
        host_offsets_in, splits_in = (
            (graph.inputs[-2].tensor, graph.inputs[-1].tensor)
            if host_mirrors
            else (None, None)
        )
        driver = stub_sequential_driver(
            proposer,
            device=CPU_DEVICE,
            num_draft_steps=K,
            per_row=per_row,
            head_free=head_free,
            distributed=host_mirrors,
            observe_batch=observe_host_offsets if host_mirrors else None,
        )
        leaf = stub_kv_leaf(
            ops.constant(np.zeros(B, dtype=np.int64), DType.int64, CPU_DEVICE),
            kv_blocks_in,
            CPU_DEVICE,
        )
        outputs = driver._run(
            tokens_in,
            offsets_in,
            drafts_in,
            kv_collections=[leaf],
            draft_kv_collections=[leaf],
            passthrough_kv={},
            return_n_logits=return_n_logits_in,
            seed=ops.constant(0, DType.uint64, CPU_DEVICE),
            temperature=ops.constant(0.0, DType.float32, CPU_DEVICE),
            top_k=ops.constant(1, DType.int64, CPU_DEVICE),
            max_k=ops.constant(1, DType.int64, CPU_DEVICE),
            top_p=ops.constant(1.0, DType.float32, CPU_DEVICE),
            min_top_p=ops.constant(1.0, DType.float32, CPU_DEVICE),
            extra={"logits": logits_in},
            batch_context_lengths=[
                ops.constant(
                    np.zeros(1, dtype=np.int64), DType.int64, CPU_DEVICE
                )
            ],
            draft_scores=scores_in,
            draft_score_coeffs=coeffs_in,
            host_input_row_offsets=host_offsets_in,
            data_parallel_splits=splits_in,
        )
        graph.output(
            *outputs,
            *(proposer.prefill_outputs if observe_prefill else ()),
            *host_merged,
        )
    return graph


def execute(
    model: Model,
    *,
    per_row: bool,
    drafts: np.ndarray,
    logits: np.ndarray,
    return_n_logits: np.ndarray,
    scores: np.ndarray | None = None,
    coeffs: np.ndarray | None = None,
    host_offsets: np.ndarray | None = None,
    splits: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Runs the built step with the given buffers."""
    buffers = [
        Buffer.from_numpy(np.arange(sum(PROMPT_LENS), dtype=np.int64) + 100),
        Buffer.from_numpy(np.cumsum([0, *PROMPT_LENS]).astype(np.uint32)),
        Buffer.from_numpy(drafts),
        Buffer.from_numpy(logits),
        Buffer.from_numpy(return_n_logits),
        Buffer.from_numpy(np.zeros(1, dtype=np.int64)),
    ]
    if per_row:
        assert scores is not None and coeffs is not None
        buffers.append(Buffer.from_numpy(scores))
        buffers.append(Buffer.from_numpy(coeffs))
    if host_offsets is not None:
        assert splits is not None
        buffers.append(Buffer.from_numpy(host_offsets))
        buffers.append(Buffer.from_numpy(splits))
    return outputs_to_numpy(model.execute(*buffers))


def per_row_inputs(lengths: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Returns scores and coefficients that select the given prefix lengths."""
    scores = np.full((B, K), -5.0, dtype=np.float32)
    for row, length in enumerate(lengths):
        scores[row, :length] = 5.0
    coeffs = np.array([1.0] + [0.0] * K, dtype=np.float32)
    return scores, coeffs


def position_logits() -> np.ndarray:
    """Returns target logits whose argmax identifies the row and slot.

    Slot t of row r peaks at token ``100 + r * 10 + t``, which no draft
    equals, so every position rejects and the recovered token names its
    own slot.
    """
    rows = B * (K + 1)
    logits = np.zeros((rows, VOCAB), dtype=np.float32)
    for r in range(rows):
        row, slot = divmod(r, K + 1)
        logits[r, 100 + row * 10 + slot] = 10.0
    return logits


def run_per_row(
    model: Model, lengths: Sequence[int] = (1, 3)
) -> list[np.ndarray]:
    """Runs the per-row graph with rejecting target logits.

    Args:
        model: The compiled sequential step.
        lengths: The selected draft prefixes.

    Returns:
        Accepted counts, next tokens, new drafts, optional scores and
        verify lengths, followed by any requested prefill observations.
    """
    drafts = np.arange(B * K, dtype=np.int64).reshape(B, K) + 200
    scores, coeffs = per_row_inputs(lengths)
    return execute(
        model,
        per_row=True,
        drafts=drafts,
        logits=position_logits(),
        return_n_logits=np.array([W + 1], dtype=np.int64),
        scores=scores,
        coeffs=coeffs,
    )
