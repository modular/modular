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
"""The block driver's flag-off op sequences, pinned against main's."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    ops,
)
from max.nn.legacy.identity import Identity
from max.pipelines.speculative.block_driver import (
    Accepted,
    BlockBatch,
    BlockDriver,
    DraftSampler,
)
from max.pipelines.speculative.config import SpeculativeConfig
from max.pipelines.speculative.spec_input_types import SpecDecodeInputTypeSpec
from max.pipelines.speculative.spec_target import Verified
from spec_step_stubs import stub_kv_leaf

VOCAB = 64
BLOCK = 4
NUM_DRAFTS = BLOCK - 1
_CPU = DeviceRef.CPU()


class _AcceptOnlyProposer:
    """A block draft the accept phase never calls into."""

    block_size = BLOCK
    mask_token_id = VOCAB - 1
    samples_from_anchor = False
    supports_zero_draft_rows = False

    def materialize(self, *args: object) -> None:
        raise NotImplementedError

    def embed_block(self, *args: object) -> list[TensorValue]:
        raise NotImplementedError

    def forward_block(self, *args: object) -> None:
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


class _StubBlockProposer(_AcceptOnlyProposer):
    """A block draft whose phases return placeholders, so the whole step
    builds around the driver's own ops."""

    def materialize(self, *args: object) -> None:
        return None

    def embed_block(self, *args: object) -> list[TensorValue]:
        return []

    def forward_block(self, *args: object) -> None:
        return None

    def head(
        self,
        batch: BlockBatch,
        block_hs: None,
        accepted: Accepted,
        sampler: DraftSampler,
    ) -> TensorValue:
        return batch.draft_tokens


def _driver(
    verify: Callable[[BlockBatch], Verified[None]] = _unused_verify,
    proposer: _AcceptOnlyProposer | None = None,
) -> BlockDriver[None, None]:
    return BlockDriver(
        verify,
        proposer or _AcceptOnlyProposer(),
        target_model=Identity(),
        draft_model=Identity(),
        input_spec=SpecDecodeInputTypeSpec(devices=[_CPU], distributed=False),
        speculative_config=SpeculativeConfig(
            speculative_method="dflash",
            num_speculative_tokens=NUM_DRAFTS,
        ),
        use_greedy_acceptance=True,
    )


def build_flag_off_accept_graph() -> Graph:
    """Builds the flag-off block accept graph, as the accept fixture's."""
    with Graph(
        "block_flag_off_accept_golden",
        input_types=[
            TensorType(DType.int64, ["total_seq_len"], _CPU),
            TensorType(DType.uint32, ["input_row_offsets_len"], _CPU),
            TensorType(DType.int64, ["batch_size", "num_steps"], _CPU),
            TensorType(DType.float32, ["total_output_len", VOCAB], _CPU),
        ],
    ) as graph:
        tokens, offsets, drafts, logits = (v.tensor for v in graph.inputs)
        batch = BlockBatch(
            tokens=tokens,
            input_row_offsets=offsets,
            draft_tokens=drafts,
            kv_collections=[],
            draft_kv_collections=[],
            passthrough_kv={},
            return_n_logits=ops.constant(
                NUM_DRAFTS + 1, DType.int64, device=_CPU
            ),
            devices=[_CPU],
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
        unused = ops.constant(0, DType.float32, device=_CPU)
        accepted = _driver()._accept(
            batch,
            logits,
            seed=ops.constant(0, DType.uint64, device=_CPU),
            temperature=unused,
            top_k=ops.constant(1, DType.int64, device=_CPU),
            max_k=ops.constant(1, DType.int64, device=_CPU),
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
    return graph


def build_flag_off_step_graph() -> Graph:
    """Builds the flag-off block step with the host offset mirror, whose
    merged offsets the verify reads."""
    with Graph(
        "block_flag_off_step_golden",
        input_types=[
            TensorType(DType.int64, ["total_seq_len"], _CPU),
            TensorType(DType.uint32, ["input_row_offsets_len"], _CPU),
            TensorType(DType.int64, ["batch_size", "num_steps"], _CPU),
            TensorType(DType.float32, ["total_output_len", VOCAB], _CPU),
            TensorType(DType.uint32, ["input_row_offsets_len"], _CPU),
        ],
    ) as graph:
        tokens, offsets, drafts, logits, host_offsets = (
            v.tensor for v in graph.inputs
        )
        host_merged: list[TensorValue] = []

        def verify(batch: BlockBatch) -> Verified[None]:
            assert batch.host_merged_offsets is not None
            host_merged.append(batch.host_merged_offsets)
            return Verified(logits=batch.extra["logits"], hidden=None)

        leaf = stub_kv_leaf(
            ops.constant(np.zeros(2, dtype=np.int64), DType.int64, _CPU),
            cast("BufferValue", None),
            _CPU,
        )
        unused = ops.constant(0, DType.float32, device=_CPU)
        outputs = _driver(verify, _StubBlockProposer())._run(
            tokens,
            offsets,
            drafts,
            kv_collections=[leaf],
            draft_kv_collections=[leaf],
            passthrough_kv={},
            return_n_logits=ops.constant(
                NUM_DRAFTS + 1, DType.int64, device=_CPU
            ),
            seed=ops.constant(0, DType.uint64, device=_CPU),
            temperature=unused,
            top_k=ops.constant(1, DType.int64, device=_CPU),
            max_k=ops.constant(1, DType.int64, device=_CPU),
            top_p=unused,
            min_top_p=unused,
            host_input_row_offsets=host_offsets,
            extra={"logits": logits},
        )
        graph.output(*outputs, *host_merged)
    return graph


_TESTDATA = Path(__file__).parent / "testdata"


@pytest.mark.parametrize(
    "build, golden",
    [
        (build_flag_off_accept_graph, "block_flag_off_accept_op_sequence.txt"),
        (build_flag_off_step_graph, "block_flag_off_step_op_sequence.txt"),
    ],
)
def test_the_block_flag_off_op_sequence_matches_main(
    build: Callable[[], Graph], golden: str
) -> None:
    """The flag-off block accept and step emit main's ops in main's order."""
    with build() as graph:
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
