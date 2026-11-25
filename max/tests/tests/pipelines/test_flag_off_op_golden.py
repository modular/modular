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
"""The flag-off spec graph's op sequence, pinned against main's.

Adaptive width off (or a plain export) must build exactly main's graph:
the same ops in the same order, so the compiler fuses the same way. Each
checked-in sequence comes from main; its header says how to regenerate it.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from max.dtype import DType
from max.graph import (
    BufferType,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    ops,
)
from max.nn._graph_ops.sampling import AcceptanceSampler
from max.nn.outputs import ReturnHiddenStates
from max.pipelines.speculative.driver import (
    CarryDimNames,
    DecodeKVSwap,
    DraftCache,
    Proposed,
    ReuseSpec,
    SequentialBatch,
    SequentialDriver,
)
from max.pipelines.speculative.ragged_token_merger import RaggedTokenMerger
from max.pipelines.speculative.spec_input_types import (
    SpecDecodeInputTypeSpec,
)
from max.pipelines.speculative.spec_target import Verified
from spec_step_stubs import stub_kv_leaf

B = 2
K = 3
VOCAB = 128
HIDDEN = 4
_CPU = DeviceRef.CPU()


class _RecordingProposer:
    """A stub proposer that returns fixed logits, as the per-row tests'."""

    reads_verify_lengths: bool = True
    hidden_dim: int = HIDDEN
    reuse: ReuseSpec | None = None
    decode_swaps: tuple[DecodeKVSwap, ...] = ()
    passthrough_decode_swaps: tuple[str, ...] = ()
    draft_cache: DraftCache = DraftCache.OWN
    split_prefix: str = "stub"
    carry_dim_names: CarryDimNames = CarryDimNames()
    step_hidden_mode: ReturnHiddenStates = ReturnHiddenStates.ALL
    uses_thinking_phase: bool = False
    supports_zero_draft_rows: bool = False
    prefill_slots: int = K + 1

    def prefill(
        self, batch: SequentialBatch, tokens: object, target_hidden: object
    ) -> Proposed:
        rows = B * self.prefill_slots
        logits = np.zeros((rows, VOCAB), dtype=np.float32)
        for r in range(rows):
            logits[r, r % self.prefill_slots] = 10.0
        return Proposed(
            logits=ops.constant(logits, DType.float32, _CPU),
            hidden=[
                ops.constant(
                    np.zeros((rows, HIDDEN), dtype=np.float32),
                    DType.float32,
                    _CPU,
                )
            ],
        )

    def step(
        self, batch: SequentialBatch, draft_input: object, index: int
    ) -> Proposed:
        rows = B
        logits = np.zeros((rows, VOCAB), dtype=np.float32)
        for r in range(rows):
            logits[r, (r + index) % VOCAB] = 10.0
        return Proposed(
            logits=ops.constant(logits, DType.float32, _CPU),
            hidden=[
                ops.constant(
                    np.zeros((rows, HIDDEN), dtype=np.float32),
                    DType.float32,
                    _CPU,
                )
            ],
        )


def _driver(
    verify: Callable[[SequentialBatch], Verified[list[TensorValue]]],
    *,
    distributed: bool,
) -> SequentialDriver[None]:
    driver = SequentialDriver.__new__(SequentialDriver)
    vars(driver).update(
        _proposer=_RecordingProposer(),
        _verify=verify,
        _draft_proposal="argmax",
        _vocab_size=None,
        _skips_draft_rows=False,
        _head_free_scores=False,
        num_draft_steps=K,
        data_parallel_degree=1,
        devices=[_CPU],
        merger=RaggedTokenMerger(_CPU),
        # Defaults only, so the same builder runs on main's sources.
        _input_spec=SpecDecodeInputTypeSpec(
            devices=[_CPU], distributed=distributed
        ),
        acceptance_sampler=AcceptanceSampler(
            num_draft_steps=K, use_stochastic=True
        ),
    )
    return driver


def _graph_inputs(host_mirrors: bool) -> list[TensorType | BufferType]:
    inputs: list[TensorType | BufferType] = [
        TensorType(DType.int64, ["total_seq_len"], _CPU),
        TensorType(DType.uint32, ["input_row_offsets_len"], _CPU),
        TensorType(DType.int64, ["batch_size", "num_steps"], _CPU),
        TensorType(DType.float32, ["logits_rows", VOCAB], _CPU),
        TensorType(DType.int64, [1], _CPU),
        BufferType(DType.int64, [1], _CPU),
    ]
    if host_mirrors:
        inputs += [
            TensorType(DType.uint32, ["input_row_offsets_len"], _CPU),
            TensorType(DType.int64, [2], _CPU),
        ]
    return inputs


def build_flag_off_graph(*, host_mirrors: bool = False) -> Graph:
    """Builds the flag-off sequential step graph.

    With ``host_mirrors`` the step is built as an MLA target's: distributed,
    with the host offset mirror and splits, and the verify reads the host
    merged offsets as a sharded target does.
    """
    with Graph(
        "flag_off_golden",
        input_types=_graph_inputs(host_mirrors),
    ) as graph:
        (
            tokens_in,
            offsets_in,
            drafts_in,
            logits_in,
            return_n_logits_in,
        ) = (v.tensor for v in graph.inputs[:5])
        kv_blocks_in = graph.inputs[5].buffer
        host_offsets_in, splits_in = (
            (graph.inputs[6].tensor, graph.inputs[7].tensor)
            if host_mirrors
            else (None, None)
        )
        host_merged: list[TensorValue] = []

        def verify(batch: SequentialBatch) -> Verified[list[TensorValue]]:
            if host_mirrors:
                host_merged.append(batch.dist.host_merged_offsets)
            return Verified(
                logits=batch.extra["logits"],
                hidden=[batch.extra["logits"]],
            )

        driver = _driver(verify, distributed=host_mirrors)
        leaf = stub_kv_leaf(
            ops.constant(np.zeros(B, dtype=np.int64), DType.int64, _CPU),
            kv_blocks_in,
            _CPU,
        )
        outputs = driver._run(
            tokens_in,
            offsets_in,
            drafts_in,
            kv_collections=[leaf],
            draft_kv_collections=[leaf],
            passthrough_kv={},
            return_n_logits=return_n_logits_in,
            seed=ops.constant(0, DType.uint64, _CPU),
            temperature=ops.constant(0.0, DType.float32, _CPU),
            top_k=ops.constant(1, DType.int64, _CPU),
            max_k=ops.constant(1, DType.int64, _CPU),
            top_p=ops.constant(1.0, DType.float32, _CPU),
            min_top_p=ops.constant(1.0, DType.float32, _CPU),
            host_input_row_offsets=host_offsets_in,
            data_parallel_splits=splits_in,
            extra={"logits": logits_in},
            batch_context_lengths=[
                ops.constant(np.zeros(1, dtype=np.int64), DType.int64, _CPU)
            ],
        )
        graph.output(*outputs, *host_merged)
    return graph


_TESTDATA = Path(__file__).parent / "testdata"


@pytest.mark.parametrize(
    "host_mirrors, golden",
    [
        (False, "flag_off_op_sequence.txt"),
        (True, "flag_off_host_mirror_op_sequence.txt"),
    ],
)
def test_the_flag_off_op_sequence_matches_main(
    host_mirrors: bool, golden: str
) -> None:
    """The plain graph emits main's ops in main's order, with and without
    the host offset mirrors an MLA target passes."""
    with build_flag_off_graph(host_mirrors=host_mirrors) as graph:
        got = re.findall(r"mo\.[a-z_.]+", str(graph))
    expected = [
        line
        for line in (_TESTDATA / golden).read_text().splitlines()
        if line and not line.startswith("#")
    ]
    assert got == expected, _first_diff(got, expected)


def _first_diff(got: list[str], expected: list[str]) -> str:
    for i, (g, e) in enumerate(zip(got, expected, strict=False)):
        if g != e:
            return f"first differing index {i}: branch {g!r} vs main {e!r}"
    if len(got) != len(expected):
        return f"lengths differ: branch {len(got)} vs main {len(expected)}"
    return "sequences match"
