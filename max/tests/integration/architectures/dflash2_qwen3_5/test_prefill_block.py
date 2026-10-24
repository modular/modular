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
"""Prefill on the fused DFlash2 graph, at the prompt lengths that crashed.

A prefill step hands the graph a zero-width draft axis, and the graph does not
agree with itself about that dim: the tensor shapes say zero -- ``recovered``
really is ``[batch, 0]`` -- while ``RaggedTokenMerger``'s ``_shape_to_scalar``
of the same dim can yield the compiled K. The accepted count therefore reaches
the committed-token gather as anything in ``[0, K]``, and ``ops.gather_nd`` is
not bounds-checked: against an unpadded width-1 table an index of K read off
the end of the row and returned 0. That 0 was every prompt's first token, and
it then became the drafter's block anchor, which is broadcast into a gather of
the selector's ``[248320, 256]`` predecessor codebook -- where an out-of-range
id is a device assert rather than a wrong answer.

Lengths 121 and 200 are the ones that turned that read into an assert in
serving; 137 and 282 only produced the wrong token. Both are covered here,
since a fix validated only on the quiet lengths proves nothing.

The inputs mirror Mach's per-phase contract: draft width 0 on prefill,
``return_n_logits = draft_width + 1``, and KV metadata shaped the way
``make_spec_kv_metadata`` shapes it.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from dflash2_pairing_harness import Geometry, fused_graph, state_regions
from max.driver import CPU, Accelerator, Buffer, CompletionFlag
from max.dtype import DType
from max.engine import InferenceSession
from max.nn.comm.allreduce import Signals
from max.nn.kv_cache import PACKED_PAGE_STRIDE
from max.pipelines.speculative.config import MAGIC_DRAFT_TOKEN_ID

CRASHING_LENGTHS = (121, 200)
QUIET_LENGTHS = (137, 282)

GEOMETRY = Geometry()
VOCAB, PAGE, BLOCK = GEOMETRY.vocab, GEOMETRY.page_size, GEOMETRY.block
NUM_LINEAR, NUM_FULL = GEOMETRY.num_linear, GEOMETRY.num_full
DRAFT_LAYERS = GEOMETRY.draft_layers
DRAFT_KVH, DRAFT_HD = GEOMETRY.draft_kv_heads, GEOMETRY.draft_head_dim
CONV_KERNEL, LV, LKD, LVD = 4, 2, 128, 128
CONV_DIM = 2 * 1 * LKD + LV * LVD
PACKED_VOCAB = (VOCAB + 31) // 32

MAGIC = MAGIC_DRAFT_TOKEN_ID
Step = Callable[..., list[np.ndarray]]


def _build() -> Step:
    fused = fused_graph(GEOMETRY)
    ring_row_shape = state_regions(fused.target)[2].row_shape
    dev = Accelerator()
    session = InferenceSession(devices=[dev])
    model = session.load(fused.graph, weights_registry=fused.weights_registry)
    sig = Signals.allocate([dev])

    def b(x: np.ndarray) -> Buffer:
        return Buffer.from_numpy(np.ascontiguousarray(x)).to(dev)

    def h(x: np.ndarray) -> Buffer:
        # ascontiguousarray promotes 0-d to (1,), which breaks the scalar slots.
        a = np.asarray(x)
        return Buffer.from_numpy(a if a.ndim == 0 else np.ascontiguousarray(a))

    # Every token is allowed: the acceptance sampler reads the mask, and a
    # flag already at the expected value lets its transfer through.
    flag = CompletionFlag(dev)
    flag.signal(1)
    wait_payload = Buffer(shape=[2], dtype=DType.int64)
    wait_payload[0] = flag._unsafe_ptr
    wait_payload[1] = 1
    pinned_bitmask = h(np.full((1, BLOCK, PACKED_VOCAB), -1, np.int32))
    bitmask_scratch = b(np.zeros((1, BLOCK, PACKED_VOCAB), np.int32))

    MAXPAGES, SLOTS = 512, 4
    tblocks = Buffer.from_numpy(
        np.zeros((MAXPAGES + 1, 2, NUM_FULL, PAGE, 1, 16), np.float32)
    ).to(dev)
    dblocks = Buffer.from_numpy(
        np.zeros(
            (MAXPAGES + 1, 2, DRAFT_LAYERS, PAGE, DRAFT_KVH, DRAFT_HD),
            np.float32,
        )
    ).to(dev)

    def step(
        P: int,
        num_steps: int,
        cached: int,
        drafts: list[int] | None = None,
    ) -> list[np.ndarray]:
        """One forward with Mach's contract. Returns (num_accepted, next, drafts)."""
        # The payload holds the flag's address, so the flag must outlive it.
        assert flag.load() == 1
        batch = 1
        active = P if num_steps == 0 else 1
        target_width = active + num_steps
        abs_max_cache = cached + active + num_steps + BLOCK
        lut_w = 24
        lut = np.arange(lut_w, dtype=np.uint32)[None, :]
        toks = np.random.default_rng(P).integers(
            0, VOCAB - 1, active, dtype=np.int64
        )
        dtv = (
            np.full((batch, num_steps), MAGIC, dtype=np.int64)
            if drafts is None
            else np.array(drafts, dtype=np.int64).reshape(batch, num_steps)
        )
        args = [
            b(toks),
            b(np.array([0, active], np.uint32)),
            h(np.array([0, active], np.uint32)),
            h(np.array([num_steps + 1], np.int64)),
            h(np.array([0, 1], np.int64)),
            *sig,
            tblocks,
            b(np.array([cached] * batch, np.uint32)),
            b(lut),
            h(np.array([target_width], np.uint32)),
            h(np.array([abs_max_cache], np.uint32)),
            h(np.array([PACKED_PAGE_STRIDE], np.int64)),
            h(np.array([batch, target_width, 1, abs_max_cache], np.int64)),
            dblocks,
            b(np.array([cached] * batch, np.uint32)),
            b(lut),
            h(np.array([target_width], np.uint32)),
            h(np.array([abs_max_cache], np.uint32)),
            h(np.array([PACKED_PAGE_STRIDE], np.int64)),
            h(np.array([batch, target_width, 1, abs_max_cache], np.int64)),
            h(np.array([0], np.int32)),
            b(dtv),
            b(np.array([7], np.uint64)),
            b(np.zeros(batch, np.float32)),
            b(np.ones(batch, np.int64)),
            h(np.array(1, np.int64).reshape(())),
            b(np.ones(batch, np.float32)),
            h(np.array(1.0, np.float32).reshape(())),
            b(np.zeros(batch, np.bool_)),
            pinned_bitmask,
            wait_payload,
            bitmask_scratch,
        ]

        def conv_pool() -> Buffer:
            # One pool per leaf now: a block's layers are consecutive rows,
            # so the row count is pages times layers.
            return Buffer.from_numpy(
                np.zeros(
                    (SLOTS * NUM_LINEAR, CONV_DIM, CONV_KERNEL - 1), np.float32
                )
            ).to(dev)

        def rec_pool() -> Buffer:
            return Buffer.from_numpy(
                np.zeros((SLOTS * NUM_LINEAR, LV, LKD, LVD), np.float32)
            ).to(dev)

        def ring_pool() -> Buffer:
            return Buffer.from_numpy(
                np.zeros((SLOTS * NUM_LINEAR, *ring_row_shape), np.float32)
            ).to(dev)

        # Request i holds block i, whose layers are rows i*L..i*L+L-1.
        rows = b(
            np.ascontiguousarray(
                np.arange(batch * NUM_LINEAR, dtype=np.uint32)
                .reshape(batch, NUM_LINEAR)
                .T
            )
        )
        args += [conv_pool(), rec_pool(), ring_pool(), rows, rows, rows]
        o = model.execute(*args)
        return [np.array(x.to(CPU()).to_numpy()) for x in o]

    return step


@pytest.fixture(scope="module")
def step() -> Step:
    """Compiles the fused graph once; returns a one-forward driver."""
    return _build()


@pytest.mark.parametrize("prompt_len", CRASHING_LENGTHS + QUIET_LENGTHS)
def test_prefill_commits_a_real_token(step: Step, prompt_len: int) -> None:
    """Every id the graph emits at prefill has to be a real vocabulary id.

    The committed token is the drafter's block anchor, and the selector
    gathers a codebook row with it, so an out-of-range value here is a device
    assert. Before the committed-token table was padded this was 0 for every
    prompt, and at 121 and 200 the drafts that followed asserted.
    """
    _, next_tokens, drafts = step(prompt_len, 0, 0)
    assert 0 <= int(next_tokens[0]) < VOCAB
    assert ((drafts >= 0) & (drafts < VOCAB)).all()


def test_prefill_does_not_commit_one_stuck_token(step: Step) -> None:
    """Different prompts must commit different tokens.

    The bug's signature was a *universal* id 0: an off-the-end read returns
    the same value whatever the prompt, so a per-prompt range check alone
    would still pass. Comparing across prompts is what catches it.
    """
    committed = {
        p: int(step(p, 0, 0)[1][0]) for p in CRASHING_LENGTHS + QUIET_LENGTHS
    }
    assert set(committed.values()) != {0}, f"universal id-0: {committed}"
    assert len(set(committed.values())) > 1, (
        f"every prompt committed the same token: {committed}"
    )


def test_decode_is_unaffected(step: Step) -> None:
    """Decode was healthy before the fix and has to stay that way."""
    num_accepted, next_tokens, drafts = step(1, BLOCK - 1, 200)
    assert int(num_accepted[0]) == 0, "all-magic drafts accept nothing"
    assert 0 <= int(next_tokens[0]) < VOCAB
    assert drafts.shape == (1, BLOCK - 1)


def test_prefill_reports_no_accepted_tokens(step: Step) -> None:
    """A prefill step accepted nothing, and must report nothing.

    Mach advances the sequence by this count, so a prefill row claiming the
    compiled K would skip K tokens. It read K until the empty-axis reduction
    was fixed in the kernels (a reduce over a zero-extent axis returned
    without writing, so the guard downstream of it never ran).
    """
    for prompt_len in CRASHING_LENGTHS + QUIET_LENGTHS:
        num_accepted, _, _ = step(prompt_len, 0, 0)
        assert int(num_accepted[0]) == 0, f"prompt_len={prompt_len}"


def test_the_count_still_rises_when_a_draft_is_accepted(step: Step) -> None:
    """Guard against a fix that pins the count to zero instead of correcting it.

    Every other case here drafts all-magic tokens, which are always rejected,
    so a count wrongly forced to zero would look identical. Feeding back the
    target's own token at the first draft slot makes that slot acceptable, and
    the count has to follow.
    """
    _, next_tokens, _ = step(1, BLOCK - 1, 200)
    targets_own = int(next_tokens[0])
    drafts = [targets_own] + [MAGIC] * (BLOCK - 2)
    num_accepted, _, _ = step(1, BLOCK - 1, 200, drafts=drafts)
    assert int(num_accepted[0]) >= 1, (
        "the target's own token at slot 0 must be accepted; a count stuck at"
        " zero would pass every other case in this file"
    )
