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
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.nn.identity import Identity
from max.nn.kv_cache import PagedCacheValues
from max.pipelines.speculative.block_driver import (
    Accepted,
    BlockBatch,
    BlockDriver,
    DraftSampler,
    block_dispatch_metadata,
)
from max.pipelines.speculative.config import (
    MAGIC_DRAFT_TOKEN_ID,
    SpeculativeConfig,
)
from max.pipelines.speculative.spec_input_types import SpecDecodeInputTypeSpec
from max.pipelines.speculative.spec_target import Verified

VOCAB = 64
BLOCK = 4
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


@pytest.fixture(scope="module")
def accept() -> Model:
    """Compiles the driver's greedy accept phase on its own, once.

    Returns ``(num_accepted, next_tokens, commit_lengths)`` for a batch of
    tokens, row offsets, drafts ``[batch, num_steps]`` and target logits
    ``[batch * (num_steps + 1), vocab]``. ``num_steps`` stays symbolic, as
    in the fused graphs, so prefill (zero drafts) and decode run the same
    compiled graph.
    """
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


def _logits_with_argmax(argmax: list[int]) -> npt.NDArray[np.float32]:
    """One logits row per entry, each peaked at that entry's token."""
    logits = np.zeros((len(argmax), VOCAB), dtype=np.float32)
    logits[np.arange(len(argmax)), argmax] = 10.0
    return logits


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
        Buffer.from_numpy(_logits_with_argmax(target_argmax)),
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
